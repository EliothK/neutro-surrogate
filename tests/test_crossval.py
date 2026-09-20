"""Cross-validation fold invariants. numpy only"""

import numpy as np
import pytest

from src.crossval import make_folds


def test_folds_partition_the_pool():
    pool = np.arange(100, 1100)
    folds = make_folds(pool, 5)
    assert len(folds) == 5
    all_val = np.concatenate([va for _, va in folds])
    assert sorted(all_val) == sorted(pool), "every sample validated exactly once"
    for tr, va in folds:
        assert not set(tr) & set(va), "train and val must not overlap"
        assert sorted(np.concatenate([tr, va])) == sorted(pool)


def test_folds_are_balanced_and_deterministic():
    pool = np.arange(1003)
    sizes = [len(va) for _, va in make_folds(pool, 5)]
    assert max(sizes) - min(sizes) <= 1
    a, b = make_folds(pool, 5, seed=3), make_folds(pool, 5, seed=3)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_needs_two_folds():
    with pytest.raises(ValueError):
        make_folds(np.arange(10), 1)
