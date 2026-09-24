import numpy as np
from collections import Counter

def bootstrap_indices(n, rng):
    sample_idx = rng.integers(0, n, size=n)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[sample_idx] = False
    oob_idx = np.where(oob_mask)[0]
    return sample_idx, oob_idx


class Node:
    _next_id = 0  # class-level counter, shared across all Node instances

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        if self.is_leaf_node():
            self.leaf_id = Node._next_id
            Node._next_id += 1
        else:
            self.leaf_id = None

    def is_leaf_node(self):
        return self.value is not None




class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, rng=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = self.rng.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)[:-1]
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def get_leaf(self, x):
        return self._traverse_tree_leaf(x, self.root)

    def _traverse_tree_leaf(self, x, node):
        if node.is_leaf_node():
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse_tree_leaf(x, node.left)
        return self._traverse_tree_leaf(x, node.right)
    

  


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rng = np.random.default_rng(42)

    coverage = []
    for _ in range(1000):
        s_idx, o_idx = bootstrap_indices(len(y), rng)
        coverage.append(1 - len(o_idx) / len(y))
    print(np.mean(coverage))

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = np.sum(preds == y_test) / len(y_test)
    print(f"Accuracy: {acc:.4f}")