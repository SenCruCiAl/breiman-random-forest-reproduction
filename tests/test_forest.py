"""Fast checks on synthetic data; no network access needed."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_tree import DecisionTree, bootstrap_indices  # noqa: E402
from proximity import compute_proximity_matrix  # noqa: E402
from random_forest import forest_predict, oob_error, train_forest  # noqa: E402


def _blobs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n // 2, 4)), rng.normal(2.5, 1, (n // 2, 4))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_bootstrap_coverage_matches_theory():
    rng = np.random.default_rng(0)
    cov = [1 - len(bootstrap_indices(500, rng)[1]) / 500 for _ in range(400)]
    assert abs(np.mean(cov) - (1 - 1 / math.e)) < 0.005


def test_fully_grown_tree_fits_training_data():
    X, y = _blobs()
    tree = DecisionTree(rng=np.random.default_rng(0))
    tree.fit(X, y)
    assert np.all(tree.predict(X) == y)


def test_forest_beats_chance_and_oob_is_sane():
    X, y = _blobs(seed=1)
    Xt, yt = _blobs(seed=2)
    trees = train_forest(X, y, n_trees=25, seed=0)
    assert np.mean(forest_predict(trees, Xt) == yt) > 0.85
    err, total = oob_error(trees, X, y)
    assert total == len(y) and err < 0.2


def test_proximity_structure_both_modes():
    X, y = _blobs(n=60)
    trees = train_forest(X, y, n_trees=30, seed=0)
    for oob_only in (False, True):
        P = compute_proximity_matrix(trees, X, oob_only=oob_only)
        assert np.allclose(np.diag(P), 1.0)
        assert np.allclose(P, P.T)
        assert P.min() >= 0 and P.max() <= 1
        same = (y[:, None] == y[None, :]) & ~np.eye(len(y), dtype=bool)
        assert P[same].mean() > P[y[:, None] != y[None, :]].mean()
