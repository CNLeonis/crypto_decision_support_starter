from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BT = ROOT / "reports" / "backtest" / "metrics_altcoins.csv"
CAL = ROOT / "reports" / "backtest" / "metrics_calibrated.csv"
OUT = ROOT / "reports" / "backtest" / "compare_baselines_vs_calibrated.csv"


def main():
    bt = pd.read_csv(BT)
    cal = pd.read_csv(CAL)

    # Best baseline Sharpe per symbol (between BH and MA)
    base = bt.pivot_table(
        index="symbol", columns="strategy", values="sharpe", aggfunc="mean"
    ).reset_index()
    base["baseline_best_sharpe"] = base[["buy_and_hold", "ma_crossover"]].max(axis=1)

    # Best calibrated Sharpe per symbol (across variants)
    cal_best = (
        cal.groupby("symbol")["sharpe"]
        .max()
        .reset_index()
        .rename(columns={"sharpe": "cal_sharpe_best"})
    )

    out = base.merge(cal_best, on="symbol", how="left")
    out["delta_sharpe"] = out["cal_sharpe_best"] - out["baseline_best_sharpe"]
    out.to_csv(OUT, index=False)
    print(out)
    print("Zapisano:", OUT)


if __name__ == "__main__":
    main()
