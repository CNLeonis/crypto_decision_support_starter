from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _bollinger(close: pd.Series, window: int, k: float):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / (ma + 1e-12)
    return upper, lower, width


def make_features(
    df: pd.DataFrame,
    ema_spans=(6, 12, 24, 48, 72),
    vol_windows=(12, 24, 72),
    mom_windows=(6, 12, 24),
    rsi_window=14,
    bb_window=20,
    bb_k=2.0,
    vol_z_windows=(24, 72),
) -> pd.DataFrame:
    """
    Wejście: df z kolumnami [open, high, low, close, volume] indeks czasowy (UTC).
    Zwraca: DataFrame z cechami ZALAGOWANYMI (shift(1)), bez targetu.
    """
    out = pd.DataFrame(index=df.index.copy())
    close = df["close"]
    volume = df["volume"].replace(0, np.nan)

    # Stopy zwrotu (log) + zmienność
    ret1 = np.log(close).diff()
    out["ret_1"] = ret1.shift(1)
    for w in vol_windows:
        out[f"rv_{w}"] = ret1.rolling(w).std(ddof=0).shift(1)

    # EMA i relacje
    for span in ema_spans:
        ema = _ema(close, span)
        out[f"ema_{span}"] = ema.shift(1)
        out[f"close_over_ema_{span}"] = (close / (ema + 1e-12)).shift(1) - 1.0

    # Momentum
    for w in mom_windows:
        out[f"mom_{w}"] = (close / close.shift(w) - 1.0).shift(1)

    # RSI
    out[f"rsi_{rsi_window}"] = _rsi(close, rsi_window).shift(1)

    # Bollinger width
    _, _, bb_w = _bollinger(close, bb_window, bb_k)
    out[f"bb_width_{bb_window}"] = bb_w.shift(1)

    # Wolumen Z-score
    for w in vol_z_windows:
        mu = volume.rolling(w).mean()
        sd = volume.rolling(w).std(ddof=0)
        out[f"vol_z_{w}"] = ((volume - mu) / (sd + 1e-12)).shift(1)

    # ATR light
    tr = (df["high"] - df["low"]).abs()
    out["atr_14"] = tr.rolling(14).mean().shift(1)

    # Clean
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def make_target(df: pd.DataFrame, horizon_bars: int = 1) -> pd.Series:
    """Etykieta binarna: 1 jeśli zwrot w kolejnej świecy > 0, inaczej 0."""
    fut_ret = df["close"].pct_change(horizon_bars).shift(-horizon_bars)
    y = (fut_ret > 0).astype(int)
    return y


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodatkowe wskaźniki zmienności i momentum do użytku przy retrainingu bottom-5 altów.
    Nie wymaga biblioteki pandas-ta, korzysta tylko z numpy/pandas.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- RSI (14)
    df["rsi14"] = _rsi(close, window=14)

    # --- ATR (14)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # <- Series
    df["atr14"] = tr.rolling(14).mean()

    # --- Bollinger Band %B (pozycja względem kanału)
    ma = close.rolling(20).mean()
    sd = close.rolling(20).std(ddof=0)
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    df["bbands_b"] = (close - lower) / ((upper - lower) + 1e-12)

    # --- MACD histogram (12, 26, 9)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    df["macd_hist"] = macd - signal

    return df
