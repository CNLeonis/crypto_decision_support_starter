from __future__ import annotations

import pandas as pd

from src.features.sanity import sanity_check_features
from src.features.tech import make_target


def test_make_target_shift_and_sign() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 2.0]})
    y = make_target(df, horizon_bars=1)
    # Future returns: [100%, 50%, -33%, nan] -> labels [1,1,0,0]
    assert y.tolist() == [1, 1, 0, 0]


def test_sanity_check_features_no_nan_or_inf() -> None:
    df = pd.DataFrame(
        {
            "rv_12": [0.1, 0.2, 0.0],
            "rsi_14": [30.0, 50.0, 70.0],
            "atr_14": [0.05, 0.04, 0.03],
        },
        index=pd.date_range("2021-01-01", periods=3, freq="h", tz="UTC"),
    )
    res = sanity_check_features(df, allow_nan_head=0)
    assert res["ok"] is True
    assert res["issues"] == []


def test_sanity_check_features_flags_outliers() -> None:
    df = pd.DataFrame(
        {
            "rv_12": [-0.1, 0.2, 0.0],  # negative volatility
            "rsi_14": [30.0, 150.0, 70.0],  # out of range
        },
        index=pd.date_range("2021-01-01", periods=3, freq="h", tz="UTC"),
    )
    res = sanity_check_features(df, allow_nan_head=0)
    assert res["ok"] is False
    assert "negative_vol_rv_12" in res["issues"]
    assert "rsi_out_of_range" in res["issues"]


def test_sanity_check_features_leakage_heuristic() -> None:
    dt = pd.date_range("2021-01-01", periods=4, freq="h", tz="UTC")
    # Construct a leaking feature (future close)
    close = pd.Series([100, 101, 102, 103], index=dt)
    df = pd.DataFrame({"close": close, "ret_1": close.shift(-1)}, index=dt)
    res = sanity_check_features(df, allow_nan_head=0, check_leakage=True)
    assert res["ok"] is False
    assert any("potential_leakage_ret_1" in x for x in res["issues"])
