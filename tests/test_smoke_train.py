from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from scripts.predict_future_lgbm import aggregate_predictions, train_ensemble


@pytest.mark.skipif(not HAS_LGBM, reason="lightgbm not available")
def test_train_ensemble_smoke() -> None:
    # Tiny synthetic dataset to catch regressions in training pipeline
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, size=len(X)))

    params = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 8,
        "valid_fraction": 0.2,
        "early_stopping_rounds": 0,
        "ensemble_n_models": 1,
        "seed": 123,
        "deterministic": True,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
    }

    models, history = train_ensemble(X, y, params.copy())
    assert len(models) == 1
    assert "num_rounds" in history

    mean, std = aggregate_predictions(models, X.head(5))
    assert mean.shape[0] == 5
    assert std.shape[0] == 5
