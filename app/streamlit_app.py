from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

st.set_page_config(page_title="Altcoin Signals (MVP)", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LGBM = ROOT / "reports" / "models" / "lgbm"
BACKTEST = ROOT / "reports" / "backtest"


@st.cache_data(show_spinner=False)
def load_price(symbol: str) -> pd.DataFrame:
    """Read raw 1h candles for symbol."""
    path = RAW / f"{symbol}_1h.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Brak danych: {path}")
    df = pl.read_parquet(path).sort("datetime").to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime")


@st.cache_data(show_spinner=False)
def load_predictions(symbol: str, kind: str) -> pd.DataFrame | None:
    """Return inference or OOF predictions for overlay."""
    filename = "inference_predictions.csv" if kind == "inference" else "predictions_vs_price.csv"
    csv = LGBM / symbol / filename
    if not csv.exists():
        return None
    df = pd.read_csv(csv, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_confidence_metrics() -> pd.DataFrame | None:
    csv = BACKTEST / "metrics_confidence_long.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


@st.cache_data(show_spinner=False)
def load_confidence_ls_metrics() -> pd.DataFrame | None:
    csv = BACKTEST / "metrics_confidence_long_short.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def list_symbols() -> list[str]:
    candidates = sorted(
        p.name for p in LGBM.glob("*_USDT") if (p / "inference_predictions.csv").exists()
    )
    if candidates:
        return candidates
    return sorted(f.stem.replace("_1h", "") for f in RAW.glob("*_1h.parquet"))


st.title("Altcoin Signals — MVP")

symbols = list_symbols()
if not symbols:
    st.warning("Brak danych. Uruchom pipeline (scripts/run-all.ps1) aby wygenerować pliki.")
    st.stop()

col_left, col_right = st.columns([2, 1], gap="medium")
with col_left:
    choice = st.selectbox("Select market", options=symbols)
with col_right:
    view_mode = st.radio("View", options=["Line", "Candlestick"], horizontal=True)
range_options = ["All", "Last 3 months", "Last 1 month"]
view_range = st.selectbox("Range", options=range_options, index=1)
signal_source = st.radio("Signals", options=["inference", "OOF"], horizontal=True)
col_filter1, col_filter2 = st.columns(2, gap="small")
with col_filter1:
    proba_threshold = st.slider("Min P(up)", min_value=0.5, max_value=0.7, value=0.53, step=0.01)
with col_filter2:
    sample_step = st.slider("Co ile sygnałów", min_value=1, max_value=10, value=1, step=1)

try:
    price_df = load_price(choice)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

price_df = price_df.reset_index()
pred_df = load_predictions(choice, signal_source)
if pred_df is not None:
    pred_df = pred_df.loc[pred_df.index >= price_df["datetime"].min()]
    pred_df = pred_df.reset_index()
    pred_df["is_correct"] = pred_df["pred"].astype(int) == pred_df["y_true"].astype(int)

if view_range != "All":
    months = 3 if view_range == "Last 3 months" else 1
    cutoff = price_df["datetime"].max() - pd.DateOffset(months=months)
    price_df = price_df.loc[price_df["datetime"] >= cutoff]
    if pred_df is not None:
        pred_df = pred_df.loc[pred_df["datetime"] >= cutoff]

base = alt.Chart(price_df).encode(
    x=alt.X("datetime:T", title="UTC time"),
)

if view_mode == "Line":
    price_layer = base.mark_line(color="#62A1FF").encode(
        y=alt.Y("close:Q", title="Close [USDT]"),
        tooltip=[alt.Tooltip("datetime:T"), alt.Tooltip("close:Q", format=".4f")],
    )
else:
    price_layer = base.mark_rule().encode(
        y=alt.Y("high:Q", title="Price [USDT]"),
        y2="low:Q",
        tooltip=[
            alt.Tooltip("datetime:T"),
            alt.Tooltip("open:Q", format=".4f"),
            alt.Tooltip("close:Q", format=".4f"),
            alt.Tooltip("high:Q", format=".4f"),
            alt.Tooltip("low:Q", format=".4f"),
        ],
    )
    price_layer += base.mark_bar(opacity=0.6).encode(
        y="open:Q",
        y2="close:Q",
        color=alt.condition(
            "datum.close >= datum.open",
            alt.value("#2ecc71"),
            alt.value("#e74c3c"),
        ),
    )

if pred_df is not None and not pred_df.empty:
    pred_df = pred_df[pred_df["proba_up"] >= proba_threshold]
    pred_df = pred_df.iloc[::sample_step]
    pred_chart = (
        alt.Chart(pred_df)
        .mark_point(size=60, filled=True)
        .encode(
            x="datetime:T",
            y="close:Q",
            color=alt.Color(
                "is_correct:N",
                scale=alt.Scale(domain=[True, False], range=["#2ecc71", "#e74c3c"]),
                legend=alt.Legend(title="Signal outcome"),
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Time"),
                alt.Tooltip("close:Q", title="Close", format=".4f"),
                alt.Tooltip("proba_up:Q", title="P(up)", format=".3f"),
                alt.Tooltip("pred:Q", title="Pred"),
                alt.Tooltip("y_true:Q", title="y_true"),
                alt.Tooltip("fold_id:N", title="Fold/Mode"),
            ],
        )
    )
    chart = (price_layer + pred_chart).properties(title=f"{choice} | price & model signals")
    accuracy = pred_df["is_correct"].mean() * 100.0
    label = "Inference" if signal_source == "inference" else "OOF"
    st.caption(
        f"{label} samples: {len(pred_df)} | Accuracy: {accuracy:.2f}% "
        f"| Range: {pred_df['datetime'].min().date()} - {pred_df['datetime'].max().date()}"
    )
else:
    chart = price_layer.properties(title=f"{choice} | price history")
    file_name = (
        "inference_predictions.csv" if signal_source == "inference" else "predictions_vs_price.csv"
    )
    st.caption(f"Brak pliku {file_name} dla wybranego symbolu (wyświetlamy tylko cenę).")

st.altair_chart(chart.interactive(), use_container_width=True)
signals_file = (
    "inference_predictions.csv" if signal_source == "inference" else "predictions_vs_price.csv"
)
st.caption(f"Dane: {choice}_1h.parquet | sygnały: {signals_file}")

metrics_table = load_confidence_metrics()
if metrics_table is not None:
    row = metrics_table.loc[metrics_table["symbol"] == choice]
    st.subheader("Confidence-long strategy metrics")
    if not row.empty:
        view_cols = [
            "symbol",
            "sharpe",
            "max_drawdown",
            "cagr",
            "hit_ratio",
            "turnover",
            "p_enter",
            "p_exit",
            "max_std",
            "stop_loss_pct",
        ]
        st.dataframe(row[view_cols].set_index("symbol"))
    else:
        st.info("Brak wyników strategii dla wybranego symbolu (uruchom run_confidence_long).")
else:
    st.info(
        "Nie znaleziono raportu metrics_confidence_long.csv – wykonaj pipeline, aby wygenerować metryki strategii."
    )

ls_table = load_confidence_ls_metrics()
if ls_table is not None:
    row = ls_table.loc[ls_table["symbol"] == choice]
    st.subheader("Confidence long/short (V2) metrics")
    if not row.empty:
        view_cols = [
            "symbol",
            "sharpe",
            "max_drawdown",
            "cagr",
            "hit_ratio",
            "turnover",
            "p_long_enter",
            "p_short_enter",
            "max_std",
            "stop_loss_pct",
        ]
        st.dataframe(row[view_cols].set_index("symbol"))
    else:
        st.info(
            "Brak wyników strategii V2 dla wybranego symbolu (uruchom run_confidence_long_short)."
        )
else:
    st.info(
        "Nie znaleziono raportu metrics_confidence_long_short.csv – wykonaj pipeline, aby wygenerować metryki strategii V2."
    )
