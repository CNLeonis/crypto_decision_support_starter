from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.backtest.run_confidence_long_short import (
    load_config,
    load_inference,
    run_for_symbol,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "backtest"
OUT.mkdir(parents=True, exist_ok=True)


def tune(symbols: list[str], p_vals: list[float], max_std_vals: list[float]) -> pd.DataFrame:
    defaults, per_symbol_cfg, _ = load_config(ROOT / "configs" / "strategy_confidence_ls.yaml")
    rows: list[dict] = []
    for sym in symbols:
        df = load_inference(sym)
        if df is None:
            continue
        base = defaults.copy()
        base.update(per_symbol_cfg.get(sym, {}))
        for p_long, p_short, max_std in itertools.product(p_vals, p_vals, max_std_vals):
            params = base.copy()
            params.update(
                {
                    "p_long_enter": p_long,
                    "p_short_enter": p_short,
                    "p_long_exit": base.get("p_long_exit", 0.5),
                    "p_short_exit": base.get("p_short_exit", 0.5),
                    "max_std": max_std,
                }
            )
            metrics = run_for_symbol(sym, params)
            if metrics is None:
                continue
            metrics.update({"p_long_enter": p_long, "p_short_enter": p_short, "max_std": max_std})
            rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="ALL")
    ap.add_argument("--p", type=str, default="0.52,0.54,0.55,0.56")
    ap.add_argument("--max-std", type=str, default="0.08,0.10,0.12")
    args = ap.parse_args()

    if args.symbols == "ALL":
        symbol_dirs = sorted(
            [p.name for p in (ROOT / "reports" / "models" / "lgbm").glob("*_USDT") if p.is_dir()]
        )
    else:
        symbol_dirs = [s.strip() for s in args.symbols.split(",") if s.strip()]

    p_vals = [float(x) for x in args.p.split(",")]
    max_std_vals = [float(x) for x in args.max_std.split(",")]

    df = tune(symbol_dirs, p_vals, max_std_vals)
    grid_path = OUT / "metrics_confidence_long_short_grid.csv"
    df.to_csv(grid_path, index=False)
    print(df)
    print(f"Saved grid results to {grid_path}")

    if not df.empty:
        best = df.sort_values("sharpe", ascending=False).groupby("symbol").first().reset_index()
        best_path = OUT / "metrics_confidence_long_short_best.csv"
        best.to_csv(best_path, index=False)
        print("Best per symbol:")
        print(best[["symbol", "sharpe", "p_long_enter", "p_short_enter", "max_std"]])
        print(f"Saved best results to {best_path}")


if __name__ == "__main__":
    main()
