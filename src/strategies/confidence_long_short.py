from __future__ import annotations

import pandas as pd


def position_confidence_long_short(
    df: pd.DataFrame,
    *,
    p_long_enter: float = 0.55,
    p_long_exit: float = 0.50,
    p_short_enter: float = 0.55,
    p_short_exit: float = 0.50,
    max_std: float = 0.10,
    stop_loss_pct: float = 0.04,
    min_size: float = 0.1,
    max_size: float = 1.0,
) -> pd.Series:
    """Long/short sizing based on probability confidence."""

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

        # stop-loss for open position
        if current > 0 and entry_price is not None:
            if close <= entry_price * (1 - stop_loss_pct):
                desired = 0.0
                entry_price = None
        elif current < 0 and entry_price is not None:
            if close >= entry_price * (1 + stop_loss_pct):
                desired = 0.0
                entry_price = None

        proba_down = 1.0 - proba

        if std > max_std:
            desired = 0.0
        else:
            # candidate long
            long_size = None
            if proba >= p_long_enter:
                conf = (proba - p_long_enter) / max(1e-8, 1 - p_long_enter)
                long_size = min_size + conf * (max_size - min_size)
            # candidate short
            short_size = None
            if proba_down >= p_short_enter:
                conf = (proba_down - p_short_enter) / max(1e-8, 1 - p_short_enter)
                short_size = min_size + conf * (max_size - min_size)

            if long_size is not None and (short_size is None or proba >= proba_down):
                desired = float(max(min(long_size, max_size), min_size))
                if current <= 0:
                    entry_price = close
            elif short_size is not None:
                desired = -float(max(min(short_size, max_size), min_size))
                if current >= 0:
                    entry_price = close
            else:
                # exit if below exit thresholds
                if current > 0 and proba <= p_long_exit:
                    desired = 0.0
                    entry_price = None
                elif current < 0 and proba_down <= p_short_exit:
                    desired = 0.0
                    entry_price = None

        if desired == 0.0:
            entry_price = None

        current = desired
        pos.append(current)

    return pd.Series(pos, index=df.index, dtype=float)


__all__ = ["position_confidence_long_short"]
