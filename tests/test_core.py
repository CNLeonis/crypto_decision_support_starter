from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.core import CostModel, compute_strategy_returns
from src.validation.walk_forward import rolling_walk_forward


def test_compute_strategy_returns_no_costs() -> None:
    close = pd.Series([100.0, 110.0, 100.0])
    position = pd.Series([0.0, 1.0, 1.0], index=close.index)
    costs = CostModel(taker_bps=0.0, slippage_bps=0.0)

    r = compute_strategy_returns(close, position, costs)

    expected = pd.Series([0.0, 0.0, -0.09090909], index=close.index)
    assert np.allclose(r.values, expected.values, atol=1e-6)


def test_compute_strategy_returns_with_costs() -> None:
    close = pd.Series([100.0, 110.0, 100.0])
    position = pd.Series([0.0, 1.0, 1.0], index=close.index)
    costs = CostModel(taker_bps=10.0, slippage_bps=0.0)  # 10 bps per leg

    r = compute_strategy_returns(close, position, costs)

    # Enter at step 1 -> pay 0.001 cost (10 bps); position is 0 during entry bar
    assert np.isclose(r.iloc[1], -0.001, atol=1e-6)
    # Third bar should include price move without extra cost
    assert np.isclose(r.iloc[2], -0.09090909, atol=1e-6)


def test_rolling_walk_forward_shapes() -> None:
    splits = list(
        rolling_walk_forward(n_samples=20, n_splits=3, train_size=5, test_size=3, embargo=1)
    )
    # Expect 3 splits within the 20-sample window
    assert len(splits) == 3
    # First split: train 0-4, test 6-8
    tr0, te0 = splits[0]
    assert tr0.tolist() == list(range(5))
    assert te0.tolist() == [6, 7, 8]
    # Subsequent splits grow (expanding=True)
    tr1, te1 = splits[1]
    assert tr1[0] == 0 and tr1[-1] == 8
    assert len(te1) == 3
    tr2, te2 = splits[2]
    assert tr2[0] == 0 and len(te2) == 3
