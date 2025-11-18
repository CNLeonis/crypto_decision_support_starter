from __future__ import annotations

import pandas as pd


def position_buy_and_hold(close: pd.Series) -> pd.Series:
    # Create a Series of ones (long exposure) indexed by the same datetime as close prices.
    # This means the strategy is always fully invested (no timing decisions).
    # Always returns a constant long position (1.0).
    # Represents a benchmark for comparison (no timing, no trades).
    return pd.Series(1.0, index=close.index)


def position_ma_crossover(
    close: pd.Series, fast_window: int = 20, slow_window: int = 50
) -> pd.Series:
    # Create a Series indicating long (1) or no position (0) based on moving average crossover.
    # When the fast moving average is above the slow moving average, we take a long position
    fast = close.ewm(span=fast_window, adjust=False).mean()
    slow = close.ewm(span=slow_window, adjust=False).mean()
    # Uses two exponential moving averages:
    # fast EMA (short-term trend)
    # slow EMA (long-term trend)
    # When the fast EMA crosses above the slow EMA → go long (1.0).
    # Otherwise → exit / stay flat (0.0).
    # - Long (1.0) when fast EMA > slow EMA
    # - Flat (0.0) when fast EMA <= slow EMA
    return (fast > slow).astype(float)
