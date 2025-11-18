from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .core import CostModel, compute_strategy_returns, metrics_from_returns, read_price
from .strategies import position_buy_and_hold, position_ma_crossover

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "backtest"
REPORTS.mkdir(parents=True, exist_ok=True)  # create folder if it doesn’t exist


# Run backtest for a single file and return list of metrics dicts
def run_for_file(path: Path, cfg: dict) -> list[dict]:
    # Load OHLCV price data for this market.
    pdf = read_price(str(path))
    close = pdf["close"]
    # Initialize cost model (taker + slippage) from config file.
    costs = CostModel(
        taker_bps=float(cfg["costs"].get("taker_bps", 7.5)),
        slippage_bps=float(cfg["costs"].get("slippage_bps", 2.0)),
    )
    rows = []

    # ---- Strategy 1: Buy & Hold ----
    # Always stay long one unit of the asset.
    ret_bh = compute_strategy_returns(close, position_buy_and_hold(close), costs)
    # Compute performance metrics and store results.
    rows.append({"strategy": "buy_and_hold", **metrics_from_returns(ret_bh)})

    # ---- Strategy 2: Moving Average Crossover (MAC) ----
    # Load strategy-specific parameters from YAML config (with defaults).
    s_cfg = cfg.get("strategies", {}).get("ma_crossover", {})
    ret_ma = compute_strategy_returns(
        close,
        position_ma_crossover(
            close, int(s_cfg.get("fast_window", 20)), int(s_cfg.get("slow_window", 50))
        ),
        costs,
    )
    # Add metrics for MA crossover strategy.
    rows.append({"strategy": "ma_crossover", **metrics_from_returns(ret_ma)})
    return rows


# Main function to run backtests for all altcoin files in data/raw/
# iterate through all markets and save combined results to a single CSV file
def main() -> None:
    # Load backtest configuration from YAML file.
    cfg = yaml.safe_load(open(ROOT / "configs" / "backtest.yaml"))
    # Find all hourly Parquet data files.
    files = sorted(RAW.glob("*_1h.parquet"))
    if not files:
        print("No data in data/raw/. Please run the downloader first.")
        return
    out_rows = []
    for f in files:
        # Extract symbol from filename  (e.g., BTCUSDT_1h.parquet -> BTCUSDT).
        sym = f.stem.replace("_1h", "")
        # Run all strategies for this symbol.
        for row in run_for_file(f, cfg):
            row["symbol"] = sym
            out_rows.append(row)
    # Convert collected results
    df = pd.DataFrame(out_rows)
    # Save metrics to CSV report.
    out_csv = REPORTS / "metrics_altcoins.csv"
    df.to_csv(out_csv, index=False)

    print(df)
    print("Zapisano:", out_csv)


if __name__ == "__main__":
    main()
