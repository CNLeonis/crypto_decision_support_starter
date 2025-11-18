from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def rolling_walk_forward(
    n_samples: int,
    n_splits: int,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    expanding: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    train_end = train_size
    for _ in range(n_splits):
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        if expanding:
            train_idx = np.arange(0, train_end)
        else:
            train_idx = np.arange(test_start - embargo - train_size, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
        train_end = test_end
