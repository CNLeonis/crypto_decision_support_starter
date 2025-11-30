from __future__ import annotations

import pandas as pd

from src.features.tech import make_target


def test_make_target_shift_and_sign() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 2.0]})
    y = make_target(df, horizon_bars=1)
    # Future returns: [100%, 50%, -33%, nan] -> labels [1,1,0,0]
    assert y.tolist() == [1, 1, 0, 0]
