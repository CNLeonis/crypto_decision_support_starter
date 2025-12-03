from __future__ import annotations

import pandas as pd

from src.models.train_lgbm_altcoins import sanity_check_features


def test_sanity_check_features_in_train() -> None:
    # Minimal DF to ensure sanity_check_features callable is exposed
    df = pd.DataFrame({"rv_12": [0.1, 0.0], "rsi_14": [50, 60]})
    res = sanity_check_features(df, allow_nan_head=0, check_leakage=False)
    assert res["ok"] is True
