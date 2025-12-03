from __future__ import annotations

import numpy as np
import pandas as pd


def sanity_check_features(
    df: pd.DataFrame, *, allow_nan_head: int = 5, check_leakage: bool = False
) -> dict:
    """Basic sanity checks for engineered features.

    Checks:
    - no NaN/inf beyond an optional head allowance
    - monotonic datetime index (if present)
    - RSI in [0, 100] if present
    - volatility (rv_*, atr_*, bb_width_*) >= 0
    - optional leakage check: ensure no forward shifts (requires datetime index and sorted index)
    """
    issues: list[str] = []

    # Monotonic index (datetime)
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            issues.append("index_not_monotonic")

    # NaN/inf checks
    nan_mask = df.isna()
    if nan_mask.to_numpy().any():
        nan_rows = nan_mask.any(axis=1)
        if nan_rows.sum() > allow_nan_head:
            issues.append("nan_exceeds_allowance")
    if np.isinf(df.to_numpy()).any():
        issues.append("inf_detected")

    # Range checks
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
        if ((rsi < 0) | (rsi > 100)).any():
            issues.append("rsi_out_of_range")
    # Volatility-like columns >= 0
    vol_cols = [c for c in df.columns if c.startswith(("rv_", "atr_", "bb_width_"))]
    for c in vol_cols:
        if (df[c] < 0).any():
            issues.append(f"negative_vol_{c}")
    # Optional leakage check: crude detection of forward shift (look for future-close usage)
    if (
        check_leakage
        and isinstance(df.index, pd.DatetimeIndex)
        and df.index.is_monotonic_increasing
    ):
        # If close is present alongside features like ret_*, close_over_ema_*, ensure they lag by 1
        if "close" in df.columns:
            close = df["close"]
            for col in [c for c in df.columns if c.startswith(("ret_", "close_over_ema_"))]:
                series = df[col]
                # If correlation with future close is unusually high, flag (very rough heuristic)
                shifted = close.shift(-1)
                corr = series.corr(shifted)
                if corr is not None and corr > 0.9:
                    issues.append(f"potential_leakage_{col}")

    return {"ok": not issues, "issues": issues}
