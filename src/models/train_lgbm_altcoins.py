from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.backtest.core import CostModel, compute_strategy_returns, metrics_from_returns
from src.features.sanity import sanity_check_features
from src.features.tech import add_volatility_features, make_features, make_target
from src.validation.walk_forward import rolling_walk_forward

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "models" / "lgbm"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_price_df(path: Path) -> pd.DataFrame:
    df = pl.read_parquet(path).sort("datetime")
    pdf = df.to_pandas()
    pdf["datetime"] = pd.to_datetime(pdf["datetime"], utc=True)
    pdf = pdf.set_index("datetime")
    return pdf[["open", "high", "low", "close", "volume"]]


def train_symbol(path: Path, cfg: dict, timeframe: str) -> dict:
    pdf = load_price_df(path)
    pdf = add_volatility_features(pdf)
    fcfg = cfg.get("features", {})
    X = make_features(
        pdf,
        ema_spans=fcfg.get("ema_spans", [6, 12, 24, 48, 72]),
        vol_windows=fcfg.get("vol_windows", [12, 24, 72]),
        mom_windows=fcfg.get("mom_windows", [6, 12, 24]),
        rsi_window=fcfg.get("rsi_window", 14),
        bb_window=fcfg.get("bb_window", 20),
        bb_k=fcfg.get("bb_k", 2.0),
        vol_z_windows=fcfg.get("vol_z_windows", [24, 72]),
    )
    y = make_target(pdf, horizon_bars=int(cfg["target"]["horizon_bars"]))
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    X = data.drop(columns=["y"])
    y = data["y"].astype(int)

    sanity = sanity_check_features(X, allow_nan_head=0, check_leakage=True)
    if not sanity["ok"]:
        msg = f"Sanity failed for {path.name}: {sanity['issues']}"
        if cfg.get("ignore_sanity", False):
            print("[Sanity][WARN]", msg)
        else:
            raise RuntimeError(msg)

    wf = cfg["walk_forward"]
    folds = list(
        rolling_walk_forward(
            n_samples=len(data),
            n_splits=int(wf["n_splits"]),
            train_size=int(wf["train_size_bars"]),
            test_size=int(wf["test_size_bars"]),
            embargo=int(wf["embargo_bars"]),
            expanding=True,
        )
    )

    lgb_params = cfg["lightgbm"].copy()
    # Set global seeds for reproducibility (numpy, random)
    try:
        seed = int(lgb_params.get("seed", 42))
    except Exception:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # Map scikit-style params to lgb.train arguments
    num_rounds = int(lgb_params.pop("n_estimators", 100))
    early_stopping_rounds = int(lgb_params.pop("early_stopping_rounds", 0))
    valid_fraction = float(lgb_params.pop("valid_fraction", 0.0) or 0.0)
    ensemble_n = int(lgb_params.pop("ensemble_n_models", 1))
    # Avoid dataset-level params that cannot be changed once Dataset is constructed
    lgb_params.pop("data_random_seed", None)
    # Optional: strategy for combining ensemble members (default: simple mean)
    ensemble_weighting = lgb_params.pop("ensemble_weighting", "mean")
    metrics_rows = []
    feat_imp = pd.Series(0.0, index=X.columns, dtype=float)

    costs = CostModel(
        taker_bps=float(cfg.get("costs", {}).get("taker_bps", 7.5)),
        slippage_bps=float(cfg.get("costs", {}).get("slippage_bps", 2.0)),
    )
    close = pdf.loc[X.index, "close"]

    for i, (tr_idx, te_idx) in enumerate(folds, 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # Time-aware split of training fold into train/valid (use last part as validation)
        if early_stopping_rounds > 0 and valid_fraction > 0.0:
            n_tr = len(X_tr)
            n_val = max(1, int(round(n_tr * valid_fraction)))
            tr_core = X_tr.iloc[: n_tr - n_val]
            y_tr_core = y_tr.iloc[: n_tr - n_val]
            X_val = X_tr.iloc[n_tr - n_val :]
            y_val = y_tr.iloc[n_tr - n_val :]
            dtrain_full = lgb.Dataset(tr_core, label=y_tr_core)
            dvalid_full = lgb.Dataset(X_val, label=y_val, reference=dtrain_full)
        else:
            dtrain_full = lgb.Dataset(X_tr, label=y_tr)
            dvalid_full = None

        # Train ensemble of models with different seeds and average probabilities
        proba_models = []
        val_losses = []  # keep per-member validation loss (for future weighting)
        for j in range(ensemble_n):
            params_j = lgb_params.copy()
            # Bump seeds per member for diversity
            params_j["seed"] = int(params_j.get("seed", seed)) + j
            if "bagging_seed" in params_j:
                params_j["bagging_seed"] = int(params_j["bagging_seed"]) + j
            if "feature_fraction_seed" in params_j:
                params_j["feature_fraction_seed"] = int(params_j["feature_fraction_seed"]) + j
            # Ensure dataset-level params are not passed after Dataset is built
            params_j.pop("data_random_seed", None)

            if dvalid_full is not None:
                callbacks = []
                # Use callbacks for broader compatibility across lightgbm versions
                if early_stopping_rounds > 0:
                    callbacks.append(lgb.early_stopping(early_stopping_rounds))
                # Silence logging in training loop
                callbacks.append(lgb.log_evaluation(period=0))

                model = lgb.train(
                    params_j,
                    dtrain_full,
                    num_boost_round=num_rounds,
                    valid_sets=[dvalid_full],
                    valid_names=["valid"],
                    callbacks=callbacks,
                )
            else:
                model = lgb.train(params_j, dtrain_full, num_boost_round=num_rounds)

            best_iter = getattr(model, "best_iteration", None)
            proba_j = model.predict(X_te, num_iteration=best_iter)
            proba_models.append(proba_j)

            # Capture validation loss if available (for potential weighted averaging)
            if dvalid_full is not None:
                # Prefer LightGBM reported best_score; fallback to manual logloss
                vloss = None
                try:
                    vloss = model.best_score.get("valid", {}).get("binary_logloss")
                except Exception:
                    vloss = None
                if vloss is None:
                    try:
                        # Compute on the explicit validation split
                        y_val_pred = model.predict(X_val, num_iteration=best_iter)
                        vloss = float(log_loss(y_val, y_val_pred, labels=[0, 1]))
                    except Exception:
                        vloss = None
                val_losses.append(vloss)

            # Accumulate feature importance per member
            try:
                imp = pd.Series(model.feature_importance(importance_type="gain"), index=X.columns)
                feat_imp = feat_imp.add(imp, fill_value=0.0)
            except Exception:
                pass

        # Stack ensemble predictions and compute mean and std across members
        probas = np.vstack(proba_models) if len(proba_models) > 1 else np.vstack([proba_models[0]])

        # Optional weighted averaging scaffold (default: simple mean)
        weights = None
        if ensemble_weighting == "linear" and probas.shape[0] > 1:
            # Linearly decreasing weights as a simple example
            weights = np.linspace(1.0, 0.0, num=probas.shape[0])
        elif (
            ensemble_weighting == "val_loss"
            and probas.shape[0] > 1
            and all(v is not None for v in val_losses)
        ):
            inv = 1.0 / (np.asarray(val_losses) + 1e-12)
            weights = inv / inv.sum()

        if weights is not None:
            proba_mean = np.average(probas, axis=0, weights=weights)
        else:
            proba_mean = np.mean(probas, axis=0)
        proba_std = np.std(probas, axis=0)

        # Predictions from the ensemble mean probability
        y_pred = (proba_mean >= 0.5).astype(int)

        row = {
            "fold": i,
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "f1": float(f1_score(y_te, y_pred)),
            "roc_auc": (
                float(roc_auc_score(y_te, proba_mean))
                if len(np.unique(y_te)) == 2
                else float("nan")
            ),
            "brier": float(brier_score_loss(y_te, proba_mean)),
            "logloss": float(log_loss(y_te, proba_mean, labels=[0, 1])),
            "precision": float(precision_score(y_te, y_pred)),
            "recall": float(recall_score(y_te, y_pred)),
        }

        # Basic trading rule using ensemble confidence filter (only trade when consistent)
        std_threshold = 0.05
        position = pd.Series(
            np.where((proba_mean >= 0.5) & (proba_std < std_threshold), 1.0, 0.0),
            index=X_te.index,
        )
        ret = compute_strategy_returns(close.loc[X_te.index], position, costs)
        for k, v in metrics_from_returns(ret).items():
            row[f"tr_{k}"] = float(v)

        metrics_rows.append(row)

        # Log ensemble probability stats for the fold
        try:
            print(
                f"[Fold {i}] ensemble proba_mean(avg)={proba_mean.mean():.4f}, "
                f"proba_std(avg)={proba_std.mean():.4f}"
                + (f", weighting={ensemble_weighting}" if ensemble_n > 1 else "")
            )
        except Exception:
            pass

        # (feature importances already accumulated per model above)

        sym = path.stem.rsplit("_", 1)[0]
        fold_root = REPORTS / sym if timeframe == "1h" else REPORTS / f"{sym}_{timeframe}"
        fold_dir = fold_root / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Save OOF with both mean probability and ensemble dispersion
        pd.DataFrame(
            {
                "y_true": y_te.values,
                "proba": proba_mean,
                "proba_std": proba_std,
                "pred": y_pred.astype(int),
            },
            index=X_te.index,
        ).to_csv(fold_dir / "oof_predictions.csv")
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    mdf = pd.DataFrame(metrics_rows)
    agg = mdf.mean(numeric_only=True).to_dict()
    sym = path.stem.rsplit("_", 1)[0]
    out_dir = REPORTS / sym if timeframe == "1h" else REPORTS / f"{sym}_{timeframe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(out_dir / "metrics_folds.csv", index=False)
    feat_imp.sort_values(ascending=False).to_csv(out_dir / "feature_importance.csv")
    return {"symbol": sym, "timeframe": timeframe, **agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="ALL", help="np. SOL_USDT,AVAX_USDT lub ALL")
    ap.add_argument("--timeframe", type=str, default="1h", help="np. 1h, 5m, 15m, 4h, 1d")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "model.yaml"))
    files = sorted(RAW.glob(f"*_{args.timeframe}.parquet"))
    if not files:
        print(f"Brak danych w data/raw/ dla {args.timeframe}. Uruchom downloader.")
        return

    if args.symbols != "ALL":
        wanted = set(s.strip().upper() for s in args.symbols.split(","))
        files = [f for f in files if f.stem.upper().replace("_1H", "") in wanted]

    rows = []
    for f in files:
        print("Training:", f.name)
        res = train_symbol(f, cfg, args.timeframe)
        rows.append(res)

    summary = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS / "summary.csv", index=False)
    print(summary)
    print("Zapisano:", REPORTS / "summary.csv")


if __name__ == "__main__":
    main()
