import numpy as np
import math
from collections import defaultdict
from decision_tree import DecisionTree, bootstrap_indices
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split




def compute_proximity_matrix(trees, X):
    n = X.shape[0]
    proximity = np.zeros((n, n))
    for tree, _ in trees:
        leaf_groups = defaultdict(list)
        for i in range(n):
            leaf = tree.get_leaf(X[i])
            leaf_groups[leaf.leaf_id].append(i)
        for group in leaf_groups.values():
            for i in group:
                for j in group:
                    proximity[i][j] += 1
    proximity /= len(trees)
    return proximity

def compute_raw_outlier_scores(prox, y):
    n = len(y)
    outlier_raw = np.zeros(n)
    for i in range(n):
        same_class = np.where(y == y[i])[0]
        sum_sq_prox = np.sum(prox[i, same_class] ** 2)
        outlier_raw[i] = len(same_class) / sum_sq_prox
    return outlier_raw


if __name__ == "__main__":
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets

    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    y = y.replace({2: 0, 4: 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy().ravel()

    rng = np.random.default_rng(42)
    n_trees = 100
    m = max(1, int(math.sqrt(X_train.shape[1])))

    trees = []
    for i in range(n_trees):
        sample_idx, oob_idx = bootstrap_indices(len(y_train), rng)
        tree = DecisionTree(max_depth=10, n_features=m, rng=rng)
        tree.fit(X_train[sample_idx], y_train[sample_idx])
        trees.append((tree, oob_idx))

    prox = compute_proximity_matrix(trees, X_train)
    print(prox)
    print("Diagonal:", np.diag(prox))
    print("Symmetric?", np.allclose(prox, prox.T))
