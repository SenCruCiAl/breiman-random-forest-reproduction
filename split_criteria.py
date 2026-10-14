from decision_tree import DecisionTree, bootstrap_indices
import numpy as np
import math
from random_forest import SEED, N_TREES

# Follow-up study: selectable split criterion, kept separate from the original
# reproduction code. With criterion="gini" the trees and forests are identical to
# decision_tree.DecisionTree and random_forest.train_forest (same arithmetic, same RNG calls).


class CriterionTree(DecisionTree):
    """DecisionTree with a selectable split criterion.

    criterion: "gini" (default, 1 - sum p_k^2) or "entropy" (-sum p_k log2 p_k, i.e. information gain).
    """

    def __init__(self, min_samples_split=2, max_depth=None, n_features=None, rng=None, criterion="gini"):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
        super().__init__(min_samples_split=min_samples_split, max_depth=max_depth,
                         n_features=n_features, rng=rng)
        self.criterion = criterion
        self._impurity = self._gini if criterion == "gini" else self._entropy

    def _gini_gain(self, y, X_column, threshold):
        # Overrides the parent's split score so _best_split uses the chosen impurity
        parent_impurity = self._impurity(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        i_l, i_r = self._impurity(y[left_idxs]), self._impurity(y[right_idxs])
        child_impurity = (n_l / n) * i_l + (n_r / n) * i_r
        return parent_impurity - child_impurity

    def _entropy(self, y):
        ps = np.bincount(y) / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))


def train_forest(X_train, y_train, n_trees=N_TREES, seed=SEED, n_features=None, criterion="gini"):
    """Same as random_forest.train_forest, with a split criterion; return list of (tree, oob_idx)."""
    rng = np.random.default_rng(seed)
    m = n_features if n_features is not None else max(1, int(math.sqrt(X_train.shape[1])))
    trees = []
    for _ in range(n_trees):
        sample_idx, oob_idx = bootstrap_indices(len(y_train), rng)
        tree = CriterionTree(max_depth=None, min_samples_split=2, n_features=m, rng=rng, criterion=criterion)
        tree.fit(X_train[sample_idx], y_train[sample_idx])
        trees.append((tree, oob_idx))
    return trees
