from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.backtest.core import CostModel, compute_strategy_returns, metrics_from_returns

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "models" / "lgbm"


def _read_costs() -> CostModel:
    cfg_p = ROOT / "configs" / "backtest.yaml"
    if cfg_p.exists():
        cfg = yaml.safe_load(open(cfg_p))
        c = cfg.get("costs", {})
        return CostModel(
            taker_bps=float(c.get("taker_bps", 7.5)),
            slippage_bps=float(c.get("slippage_bps", 2.0)),
        )
    return CostModel()


def _load_oof_for_symbol(sym: str) -> pd.DataFrame:
    rows = []
    sym_dir = REPORTS / sym
    for fold_dir in sorted(sym_dir.glob("fold_*")):
        f = fold_dir / "oof_predictions.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True).rename_axis("datetime")
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No OOF files for {sym}")
    return pd.concat(rows).sort_index().dropna()


def _load_close(sym: str) -> pd.Series:
    pq = RAW / f"{sym}_1h.parquet"
    df = pl.read_parquet(pq).sort("datetime").to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime")["close"]


def _train_isotonic(x, y):
    m = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    return m


def _train_platt(x, y):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    logit = np.log(x / (1 - x)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000).fit(logit, y)
    return lr


def _predict_platt(lr, x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    logit = np.log(x / (1 - x)).reshape(-1, 1)
    return lr.predict_proba(logit)[:, 1]


def _build_position(proba: pd.Series, p_enter: float, p_exit: float) -> pd.Series:
    state = 0.0
    out = []
    for p in proba.values:
        if state <= 0 and p >= p_enter:
            state = 1.0
        elif state >= 1 and p <= p_exit:
            state = 0.0
        out.append(state)
    return pd.Series(out, index=proba.index, dtype=float)


def _eval_trading(close: pd.Series, proba: pd.Series, costs: CostModel, thr_grid) -> pd.DataFrame:
    rows = []
    for thr in thr_grid:
        p_enter = float(thr)
        p_exit = float(max(0.5, thr - 0.03))
        pos = _build_position(proba, p_enter, p_exit)
        ret = compute_strategy_returns(close.loc[proba.index], pos, costs)
        met = metrics_from_returns(ret)
        rows.append({"p_enter": p_enter, "p_exit": p_exit, **met})
    return pd.DataFrame(rows)


def _time_split(df: pd.DataFrame, eval_frac: float):
    cut = int(round(len(df) * (1 - eval_frac)))
    return df.iloc[:cut], df.iloc[cut:]


def calibrate_symbol(sym: str, method: str, eval_frac: float) -> dict:
    oof = _load_oof_for_symbol(sym)
    close = _load_close(sym)
    costs = _read_costs()
    oof = oof[oof.index.isin(close.index)]

    fit_df, eval_df = _time_split(oof, eval_frac=eval_frac)
    x_fit, y_fit = fit_df["proba"].values, fit_df["y_true"].astype(int).values
    x_eval, y_eval = eval_df["proba"].values, eval_df["y_true"].astype(int).values
    idx_eval = eval_df.index

    cand = {}
    if method in ("auto", "isotonic"):
        ir = _train_isotonic(x_fit, y_fit)
        p_ir = ir.predict(x_eval)
        cand["isotonic"] = (p_ir, brier_score_loss(y_eval, p_ir))
    if method in ("auto", "platt"):
        lr = _train_platt(x_fit, y_fit)
        p_pl = _predict_platt(lr, x_eval)
        cand["platt"] = (p_pl, brier_score_loss(y_eval, p_pl))

    best_name, (p_best, brier_best) = sorted(cand.items(), key=lambda kv: kv[1][1])[0]
    proba_eval = pd.Series(p_best, index=idx_eval)

    thr_grid = np.round(np.arange(0.51, 0.79, 0.01), 2)
    tdf = _eval_trading(close, proba_eval, costs, thr_grid).sort_values(
        ["sharpe", "cagr"], ascending=[False, False]
    )
    p_enter = float(tdf.iloc[0]["p_enter"])
    p_exit = float(tdf.iloc[0]["p_exit"])
    tr_metrics = tdf.iloc[0].to_dict()

    try:
        auc = roc_auc_score(y_eval, p_best) if len(np.unique(y_eval)) == 2 else float("nan")
        # ll = log_loss(y_eval, p_best, labels=[0, 1])
    except Exception:
        auc = float("nan")
        # ll = float("nan")

    out_dir = REPORTS / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"proba_cal": proba_eval, "p_enter": p_enter, "p_exit": p_exit}).to_csv(
        out_dir / "calibrated_eval_timeseries.csv"
    )
    pd.DataFrame(
        [
            {
                "symbol": sym,
                "method": best_name,
                "brier_eval": float(brier_best),
                "roc_auc_eval": float(auc),
                "p_enter": p_enter,
                "p_exit": p_exit,
                **{f"tr_{k}": float(v) for k, v in tr_metrics.items()},
            }
        ]
    ).to_csv(out_dir / "calibration_report.csv", index=False)
    return {
        "symbol": sym,
        "method": best_name,
        "brier_eval": float(brier_best),
        "roc_auc_eval": float(auc),
        "p_enter": p_enter,
        "p_exit": p_exit,
        **{f"tr_{k}": float(v) for k, v in tr_metrics.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="ALL")
    ap.add_argument("--method", type=str, default="auto", choices=["auto", "isotonic", "platt"])
    ap.add_argument("--eval_frac", type=float, default=0.30)
    args = ap.parse_args()

    sym_dirs = [p for p in (REPORTS).glob("*") if p.is_dir()]
    symbols = [p.name for p in sym_dirs]
    if args.symbols != "ALL":
        want = set(s.strip().upper() for s in args.symbols.split(","))
        symbols = [s for s in symbols if s.upper() in want]

    rows = []
    for sym in symbols:
        try:
            print("Calibrating:", sym)
            rows.append(calibrate_symbol(sym, args.method, args.eval_frac))
        except Exception as e:
            rows.append({"symbol": sym, "error": str(e)})
    if rows:
        df = pd.DataFrame(rows)
        REPORTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(REPORTS / "summary_calibrated.csv", index=False)
        print(df)
        print("Zapisano:", REPORTS / "summary_calibrated.csv")


if __name__ == "__main__":
    main()
