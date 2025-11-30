from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.backtest.run_confidence_long import load_config, load_inference, run_for_symbol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "backtest"
OUT.mkdir(parents=True, exist_ok=True)


def tune(
    symbols: list[str],
    p_enter_vals: list[float],
    max_std_vals: list[float],
    p_exit: float,
    stop_loss_pct: float,
) -> pd.DataFrame:
    defaults, per_symbol, _ = load_config(ROOT / "configs" / "strategy_confidence.yaml")

    rows: list[dict] = []
    for symbol in symbols:
        df = load_inference(symbol)
        if df is None:
            continue
        base = defaults.copy()
        base.update(per_symbol.get(symbol, {}))
        for p_enter, max_std in itertools.product(p_enter_vals, max_std_vals):
            params = {
                "p_enter": p_enter,
                "p_exit": base.get("p_exit", p_exit),
                "max_std": max_std,
                "stop_loss_pct": base.get("stop_loss_pct", stop_loss_pct),
                "min_size": base.get("min_size", 0.1),
                "max_size": base.get("max_size", 1.0),
            }
            metrics = run_for_symbol(symbol, **params)
            if metrics is None:
                continue
            metrics.update({"p_enter": p_enter, "max_std": max_std})
            rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="ALL")
    ap.add_argument("--p-enter", type=str, default="0.52,0.54,0.55,0.57")
    ap.add_argument("--max-std", type=str, default="0.05,0.08,0.1,0.12")
    args = ap.parse_args()

    if args.symbols == "ALL":
        symbol_dirs = sorted(
            [p.name for p in (ROOT / "reports" / "models" / "lgbm").glob("*_USDT") if p.is_dir()]
        )
    else:
        symbol_dirs = [s.strip() for s in args.symbols.split(",") if s.strip()]

    p_enter_vals = [float(x) for x in args.p_enter.split(",")]
    max_std_vals = [float(x) for x in args.max_std.split(",")]

    df = tune(symbol_dirs, p_enter_vals, max_std_vals, p_exit=0.5, stop_loss_pct=0.04)
    out_path = OUT / "metrics_confidence_long_grid.csv"
    df.to_csv(out_path, index=False)
    print(df)
    print(f"Saved grid results to {out_path}")

    if not df.empty:
        best = df.sort_values("sharpe", ascending=False).groupby("symbol").first().reset_index()
        best_path = OUT / "metrics_confidence_long_best.csv"
        best.to_csv(best_path, index=False)
        print("Best per symbol:")
        print(best[["symbol", "sharpe", "p_enter", "max_std"]])
        print(f"Saved best results to {best_path}")


if __name__ == "__main__":
    main()
