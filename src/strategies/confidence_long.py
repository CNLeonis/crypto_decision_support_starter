from __future__ import annotations

import pandas as pd


def position_confidence_long(
    df: pd.DataFrame,
    *,
    p_enter: float = 0.6,
    p_exit: float = 0.55,
    max_std: float = 0.05,
    stop_loss_pct: float = 0.04,
    min_size: float = 0.1,
    max_size: float = 1.0,
) -> pd.Series:
    """Long-only position sizing based on probability confidence."""

    pos = []
    current = 0.0
    entry_price: float | None = None

    closes = df["close"].astype(float)
    probas = df["proba_up"].astype(float)
    stds = df.get("proba_std")
    if stds is None:
        stds = pd.Series(0.0, index=df.index)

    for close, proba, std in zip(closes, probas, stds, strict=False):
        desired = current

        if current > 0 and entry_price is not None:
            if close <= entry_price * (1 - stop_loss_pct):
                desired = 0.0
                entry_price = None

        if std > max_std:
            desired = 0.0
        elif proba >= p_enter:
            confidence = (proba - p_enter) / max(1e-8, 1 - p_enter)
            size = min_size + confidence * (max_size - min_size)
            desired = float(max(min(size, max_size), min_size))
            if current == 0.0:
                entry_price = close
        elif proba <= p_exit:
            desired = 0.0

        if desired == 0.0:
            entry_price = None

        current = desired
        pos.append(current)

    return pd.Series(pos, index=df.index, dtype=float)


__all__ = ["position_confidence_long"]
