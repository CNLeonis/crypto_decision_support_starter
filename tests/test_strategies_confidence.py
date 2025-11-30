from __future__ import annotations

import pandas as pd

from src.strategies.confidence_long import position_confidence_long
from src.strategies.confidence_long_short import position_confidence_long_short


def test_confidence_long_stop_loss_and_std() -> None:
    df = pd.DataFrame(
        {
            "close": [100, 102, 95, 96],
            "proba_up": [0.7, 0.7, 0.7, 0.4],
            "proba_std": [0.02, 0.2, 0.02, 0.02],
        }
    )
    pos = position_confidence_long(
        df,
        p_enter=0.6,
        p_exit=0.55,
        max_std=0.1,  # bar 1 has high std -> should flatten
        stop_loss_pct=0.04,  # bar 2 drops below stop-loss from entry 100 -> flat
        min_size=0.5,
        max_size=1.0,
    )
    # Entry with sizing, flatten on high std, re-enter, then exit on low proba
    assert pos.tolist() == [0.625, 0.0, 0.625, 0.0]


def test_confidence_long_short_switch_and_stop() -> None:
    df = pd.DataFrame(
        {
            "close": [100, 101, 99, 98, 105],
            "proba_up": [0.7, 0.4, 0.3, 0.7, 0.8],
            "proba_std": [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )
    pos = position_confidence_long_short(
        df,
        p_long_enter=0.6,
        p_long_exit=0.5,
        p_short_enter=0.6,
        p_short_exit=0.5,
        max_std=0.1,
        stop_loss_pct=0.02,
        min_size=0.5,
        max_size=1.0,
    )
    # Flow: long sized, flip to short, resize short, flip long again, resize long
    assert pos.tolist() == [0.625, -0.5, -0.625, 0.625, 0.75]
