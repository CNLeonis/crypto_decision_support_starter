from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import yaml

from src.backtest.core import CostModel, compute_strategy_returns, metrics_from_returns
from src.strategies.proba_calibrated import position_hysteresis, position_sized

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
LGBM_DIR = ROOT / "reports" / "models" / "lgbm"
OUT_DIR = ROOT / "reports" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_costs() -> CostModel:
    cfg = yaml.safe_load(open(ROOT / "configs" / "backtest.yaml"))
    c = cfg.get("costs", {})
    return CostModel(
        taker_bps=float(c.get("taker_bps", 7.5)),
        slippage_bps=float(c.get("slippage_bps", 2.0)),
    )


def load_close(symbol: str) -> pd.Series:
    pq = RAW / f"{symbol}_1h.parquet"
    df = pl.read_parquet(pq).sort("datetime").to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime")["close"]


def min_hold(pos, bars=6):
    last = 0.0
    cnt = 0
    out = []
    for p in pos.values:
        if p != last:
            if cnt < bars:
                p = last
            else:
                cnt = 0
        out.append(p)
        last = p
        cnt += 1
    return pd.Series(out, index=pos.index)


def run_one(symbol: str, costs: CostModel) -> list[dict]:
    rep_path = LGBM_DIR / symbol / "calibration_report.csv"
    ts_path = LGBM_DIR / symbol / "calibrated_eval_timeseries.csv"
    if not rep_path.exists() or not ts_path.exists():
        raise FileNotFoundError(f"Brak plików kalibracji dla {symbol} ({rep_path} / {ts_path})")
    rep = pd.read_csv(rep_path).iloc[0]
    p_enter = float(rep["p_enter"])
    p_exit = float(rep["p_exit"])

    ts = pd.read_csv(ts_path, index_col=0, parse_dates=True)
    proba_cal = ts["proba_cal"].astype(float)
    close = load_close(symbol).loc[proba_cal.index]

    rows = []
    pos_h = position_hysteresis(proba_cal, p_enter, p_exit)
    pos_h = min_hold(pos_h, bars=6)
    ret_h = compute_strategy_returns(close, pos_h, costs)
    met_h = metrics_from_returns(ret_h)
    rows.append(
        {"symbol": symbol, "variant": "cal_hysteresis", **{k: float(v) for k, v in met_h.items()}}
    )

    pos_s = position_sized(proba_cal, step=0.10)
    pos_s = min_hold(pos_s, bars=6)
    ret_s = compute_strategy_returns(close, pos_s, costs)
    met_s = metrics_from_returns(ret_s)
    rows.append(
        {"symbol": symbol, "variant": "cal_sized_0.25", **{k: float(v) for k, v in met_s.items()}}
    )

    return rows


def main(symbols=None):
    costs = read_costs()
    sym_dirs = sorted([p.name for p in LGBM_DIR.glob("*") if p.is_dir()])
    if symbols is None:
        symbols = sym_dirs
    else:
        symbols = [s for s in sym_dirs if s in symbols]

    all_rows = []
    for s in symbols:
        try:
            all_rows.extend(run_one(s, costs))
        except Exception as e:
            all_rows.append({"symbol": s, "variant": "ERROR", "error": str(e)})
    df = pd.DataFrame(all_rows)
    out = OUT_DIR / "metrics_calibrated.csv"
    df.to_csv(out, index=False)
    print(df)
    print(f"Zapisano: {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="ALL")
    args = ap.parse_args()
    syms = None if args.symbols == "ALL" else [s.strip() for s in args.symbols.split(",")]
    main(syms)
