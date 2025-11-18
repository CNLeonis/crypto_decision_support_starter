from __future__ import annotations

import numpy as np
import pandas as pd


def position_hysteresis(proba: pd.Series, p_enter: float, p_exit: float) -> pd.Series:
    state = 0.0
    out = []
    for p in proba.values:
        if state <= 0 and p >= p_enter:
            state = 1.0
        elif state >= 1 and p <= p_exit:
            state = 0.0
        out.append(state)
    return pd.Series(out, index=proba.index, dtype=float)


def position_sized(proba_cal: pd.Series, step: float = 0.10) -> pd.Series:
    w = np.clip(2.0 * proba_cal - 1.0, 0.0, 1.0)
    if step and step > 0:
        w = (w / step).round() * step
    return pd.Series(w, index=proba_cal.index, dtype=float)
