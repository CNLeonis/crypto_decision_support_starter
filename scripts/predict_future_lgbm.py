from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

if __package__ is None or __package__ == "":
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.features.tech import (  # noqa: E402
    add_volatility_features,
    make_features,
    make_target,
)
from src.models.train_lgbm_altcoins import load_price_df  # noqa: E402

RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "models" / "lgbm"


def prepare_dataset(symbol: str, cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    path = RAW / f"{symbol}_1h.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku z danymi: {path}")
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
    close = pdf.loc[X.index, "close"]
    return X, y, close


def train_ensemble(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    lgb_params: dict,
) -> tuple[list[lgb.Booster], dict]:
    params = lgb_params.copy()
    num_rounds = int(params.pop("n_estimators", 100))
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 0))
    valid_fraction = float(params.pop("valid_fraction", 0.0) or 0.0)
    ensemble_n = int(params.pop("ensemble_n_models", 1))
    params.pop("data_random_seed", None)

    seed = int(params.get("seed", 42))
    models = []
    history = {}
    for j in range(ensemble_n):
        params_j = params.copy()
        params_j["seed"] = int(params_j.get("seed", seed)) + j
        if "bagging_seed" in params_j:
            params_j["bagging_seed"] = int(params_j["bagging_seed"]) + j
        if "feature_fraction_seed" in params_j:
            params_j["feature_fraction_seed"] = int(params_j["feature_fraction_seed"]) + j

        if early_stopping_rounds > 0 and valid_fraction > 0.0:
            n_tr = len(X_tr)
            n_val = max(1, int(round(n_tr * valid_fraction)))
            core_end = n_tr - n_val
            X_core = X_tr.iloc[:core_end]
            y_core = y_tr.iloc[:core_end]
            X_val = X_tr.iloc[core_end:]
            y_val = y_tr.iloc[core_end:]
            dtrain = lgb.Dataset(X_core, label=y_core)
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = None

        callbacks = [lgb.log_evaluation(period=0)]
        if early_stopping_rounds > 0 and dvalid is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        if dvalid is not None:
            booster = lgb.train(
                params_j,
                dtrain,
                num_boost_round=num_rounds,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=callbacks,
            )
        else:
            booster = lgb.train(params_j, dtrain, num_boost_round=num_rounds)

        models.append(booster)
        history[f"model_{j}"] = {
            "best_iteration": getattr(booster, "best_iteration", num_rounds),
            "params": params_j,
        }
    history["num_rounds"] = num_rounds
    return models, history


def aggregate_predictions(
    models: list[lgb.Booster], X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    for booster in models:
        best_iter = getattr(booster, "best_iteration", None)
        preds.append(booster.predict(X, num_iteration=best_iter))
    probas = np.vstack(preds)
    return probas.mean(axis=0), probas.std(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trenuje model na pełnym zbiorze (poza najnowszym oknem) i tworzy prognozy out-of-sample."
    )
    parser.add_argument("--symbol", required=True, help="np. ADA_USDT")
    parser.add_argument(
        "--test-size", type=int, default=500, help="Liczba ostatnich barów użytych jako OOS."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Katalog wyjściowy na predykcje. Domyślnie reports/models/lgbm/<symbol>/",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    symbol = args.symbol.upper()
    outdir = args.outdir or (REPORTS / symbol)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, close = prepare_dataset(symbol, cfg)
    if len(X) <= args.test_size + 100:
        raise SystemExit("Zbyt mało danych względem wybranego test_size.")

    X_tr = X.iloc[: -args.test_size]
    y_tr = y.iloc[: -args.test_size]
    X_te = X.iloc[-args.test_size :]
    y_te = y.iloc[-args.test_size :]
    close_te = close.loc[X_te.index]

    models, history = train_ensemble(X_tr, y_tr, cfg["lightgbm"])
    proba_mean, proba_std = aggregate_predictions(models, X_te)
    y_pred = (proba_mean >= 0.5).astype(int)

    out_df = pd.DataFrame(
        {
            "datetime": X_te.index,
            "close": close_te.values,
            "proba_up": proba_mean,
            "proba_std": proba_std,
            "y_true": y_te.values,
            "pred": y_pred,
            "fold_id": "inference",
        }
    )
    out_df.to_csv(outdir / "inference_predictions.csv", index=False)

    metrics = {
        "accuracy": float((y_pred == y_te.values).mean()),
        "samples": int(len(out_df)),
        "start": str(out_df["datetime"].iloc[0]),
        "end": str(out_df["datetime"].iloc[-1]),
    }
    with open(outdir / "inference_meta.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "metrics": metrics}, f, indent=2)

    print(f"Zapisano prognozy do {outdir / 'inference_predictions.csv'}")
    print(f"Accuracy OOS: {metrics['accuracy']:.4f} na {metrics['samples']} próbkach")


if __name__ == "__main__":
    main()
