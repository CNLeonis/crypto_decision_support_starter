from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

# Sets the page title
st.set_page_config(page_title="Altcoin Signals (MVP)", layout="wide")
ROOT = Path(__file__).resolve().parents[1]  # root of the project
RAW = ROOT / "data" / "raw"  # raw market data folder

st.title("Altcoin Signals — MVP")

# Find all hourly Parquet files in the raw data directory.
# Example filename: "BTCUSDT_1h.parquet"
files = sorted(RAW.glob("*_1h.parquet"))
if not files:
    st.warning("No data found in data/raw/. Please run the downloader first.")
else:
    # Extract base market names from file names (remove "_1h" suffix).
    # Example: ["BTCUSDT", "ETHUSDT"]
    names = [f.stem.replace("_1h", "") for f in files]
    # Create a dropdown menu for selecting the market.
    choice = st.selectbox("Select market", options=names)
    # Match the selected name to its corresponding file.
    f = files[names.index(choice)]
    # Read the selected Parquet file into a Polars DataFrame and sort by datetime.
    df = pl.read_parquet(f).sort("datetime")
    # Convert to Pandas DataFrame for easier plotting with Streamlit.
    pdf = df.to_pandas()
    # Convert datetime column to timezone-aware datetime and set as index.
    pdf["datetime"] = pd.to_datetime(pdf["datetime"], utc=True)
    pdf.set_index("datetime", inplace=True)  # Set "datetime" as the index
    # Plot the "close" price over time as a line chart.
    st.line_chart(pdf["close"])
    st.caption(f"File: {f.name}")
