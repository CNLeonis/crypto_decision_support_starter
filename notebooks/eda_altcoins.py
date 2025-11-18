from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

# EDA for altcoins (1h) — performs sanity checks and basic plots.
# Run this script from the project root:
# python notebooks/eda_altcoins.py
# Outputs:
#     - reports/eda/*.png  (plots of close prices)
#     - reports/eda/eda_summary.csv (summary of data quality)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist


def eda_one(path: Path) -> dict:
    df = pl.read_parquet(path).sort("datetime")
    assert "datetime" in df.columns
    sym = path.stem.replace("_1h", "")

    # Check for duplicate timestamps
    dupe = df.select("datetime").to_series().is_duplicated().sum()

    # ---- Check for missing hourly data ----
    # Convert datetime to a timezone-aware Pandas Series (UTC).
    dt = pd.to_datetime(df["datetime"].to_pandas(), utc=True)

    # Generate a complete hourly date range between min and max timestamps.
    full = pd.date_range(dt.min(), dt.max(), freq="1H", tz="UTC")

    # Find any missing timestamps
    missing = full.difference(pd.Index(dt))

    # ---- Plot closing price over time ----
    fig, ax = plt.subplots()
    ax.plot(dt, df["close"].to_pandas())
    ax.set_title(f"{sym} close (1h)")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("price")

    # Save the figure to the EDA reports folder.
    fig.savefig(OUT / f"{sym}_close.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "file": path.name,
        "rows": int(df.height),
        "duplicates": int(dupe),
        "missing_hours": int(len(missing)),
    }


def main() -> None:
    # Find all hourly parquet files in the raw data folder.
    files = sorted(RAW.glob("*_1h.parquet"))
    if not files:
        print("No files found in data/raw/. Please run the downloader.")
        return
    # Run EDA for each file and collect the results.
    rows = [eda_one(p) for p in files]
    # Convert the list of dicts into a Polars DataFrame and save as CSV.
    pl.from_dicts(rows).write_csv(OUT / "eda_summary.csv")
    print("EDA summary saved:", OUT / "eda_summary.csv")


if __name__ == "__main__":
    main()
