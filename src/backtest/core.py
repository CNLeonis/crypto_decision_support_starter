from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

ANNUAL_BARS_1H = 365 * 24


@dataclass
class CostModel:
    taker_bps: float = 7.5
    slippage_bps: float = 2.0

    # Total cost (bps) per leg (single execution, e.g. entry or exit).
    # For a full round-trip you effectively pay ~2 × total_bps.
    @property
    def total_bps(self) -> float:
        return self.taker_bps + self.slippage_bps


# read price data from parquet file into a pandas DataFrame
def read_price(path: str) -> pd.DataFrame:
    df = pl.read_parquet(path).sort("datetime")
    pdf = df.to_pandas()
    pdf["datetime"] = pd.to_datetime(pdf["datetime"], utc=True)
    return pdf.set_index("datetime")[["open", "high", "low", "close", "volume"]]


# compute strategy returns given close prices and position sizes
def compute_strategy_returns(close: pd.Series, position: pd.Series, costs: CostModel) -> pd.Series:
    # Compute strategy returns given close prices and position sizes.
    r = close.pct_change().fillna(0.0)
    # Shift position to align with returns (assumes we take position at the close price).
    pos = position.shift(1).fillna(0.0)
    # Gross returns before costs = position × price change
    gross = pos * r
    # Detect trade legs: absolute change in position (entry/exit/resize).
    trade_legs = position.diff().abs().fillna(position.abs())
    # Cost per leg (decimal) times number of legs.
    cost = trade_legs * (costs.total_bps / 10_000.0)
    return gross - cost


def position_from_proba_consistency(
    proba: pd.Series,
    proba_std: pd.Series,
    p_enter: float = 0.5,
    std_threshold: float = 0.05,
) -> pd.Series:
    # Enter only when probability is high and ensemble disagreement is low.
    # Produces a long-only position: 1.0 if (proba >= p_enter) and (proba_std < std_threshold), else 0.0.
    pos = (proba >= p_enter) & (proba_std < std_threshold)
    return pd.Series(pos.astype(float).values, index=proba.index)


# compute the annualized Sharpe ratio of returns
def sharpe(returns: pd.Series, periods_per_year: int = ANNUAL_BARS_1H) -> float:
    # Mean and standard deviation of returns.
    mu = returns.mean()
    sd = returns.std(ddof=0)
    # Handle edge cases where sd is zero or NaN.
    if sd == 0 or np.isnan(sd):
        return 0.0
    # Annualize and return Sharpe ratio
    return float((mu / sd) * np.sqrt(periods_per_year))


#  compute the maximum peak-to-trough equity decline
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    # Compute drawdown as percentage below the peak.
    # Drawdown = (equity - peak) / peak
    dd = equity / peak - 1.0
    # Return the minimum (most negative) drawdown value.
    return float(dd.min())


# compute the compound annual growth rate (CAGR) of equity
def cagr(equity: pd.Series, periods_per_year: int = ANNUAL_BARS_1H) -> float:
    if len(equity) == 0:
        return 0.0
    # Total cumulative return.
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    # Convert total bars to years.
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    # Annualized growth rate formula.
    # CAGR = (Ending Value / Starting Value)^(1/Years) - 1
    return float((1.0 + total) ** (1.0 / years) - 1.0)


# compute key performance metrics from returns
def metrics_from_returns(returns: pd.Series) -> dict:
    # Build cumulative equity curve.
    eq = (1.0 + returns).cumprod()
    # Compute and return key metrics.
    return {
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(eq),
        "cagr": cagr(eq),
        "hit_ratio": float((returns > 0).mean()),  # fraction of profitable bars
        "turnover": float(np.abs(returns).sum()),  # total absolute return magnitude
    }
