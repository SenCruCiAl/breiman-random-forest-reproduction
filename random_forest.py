from decision_tree import DecisionTree, bootstrap_indices
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from collections import Counter

SEED = 42
N_TREES = 100

# Mirror of the same UCI file (ID column removed), used only if the UCI API is
# unreachable. Verified identical to UCI id=15: 699 rows, 458/241 class split,
# 16 '?' entries, all in Bare Nuclei.
MIRROR_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv"


def load_wbc(verbose=False):
    """Load Breast Cancer Wisconsin (Original), drop rows with missing values, labels 0=benign 1=malignant."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=15)
        X, y = ds.data.features, ds.data.targets.iloc[:, 0]
        source = "UCI ML Repository (ucimlrepo, id=15)"
    except Exception:
        raw = pd.read_csv(MIRROR_URL, header=None, na_values="?")
        X, y = raw.iloc[:, :9], raw.iloc[:, 9]
        source = f"mirror ({MIRROR_URL})"

    n_before = len(X)
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    y = y.replace({2: 0, 4: 1})
    if verbose:
        print(f"Data source: {source}")
        print(f"Rows before dropping missing: {n_before}, after: {len(X)}")
    return X.to_numpy(dtype=float), y.to_numpy(dtype=int)


def load_split(verbose=False):
    """Return the fixed 80/20 train/test split used throughout the project."""
    X, y = load_wbc(verbose)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def train_forest(X_train, y_train, n_trees=N_TREES, seed=SEED, n_features=None):
    """Train n_trees fully grown Gini trees on bootstrap samples; return list of (tree, oob_idx)."""
    rng = np.random.default_rng(seed)
    m = n_features if n_features is not None else max(1, int(math.sqrt(X_train.shape[1])))
    trees = []
    for _ in range(n_trees):
        sample_idx, oob_idx = bootstrap_indices(len(y_train), rng)
        tree = DecisionTree(max_depth=None, min_samples_split=2, n_features=m, rng=rng)
        tree.fit(X_train[sample_idx], y_train[sample_idx])
        trees.append((tree, oob_idx))
    return trees


def forest_predict(trees, X):
    """Majority vote of all trees for each row of X."""
    all_preds = np.array([tree.predict(X) for tree, _ in trees])
    final_preds = np.array([
        Counter(all_preds[:, i]).most_common(1)[0][0]
        for i in range(all_preds.shape[1])
    ])
    return final_preds


def oob_error(trees, X_train, y_train):
    """OOB error: each row is voted on only by trees that did not see it; return (error, rows evaluated)."""
    oob_votes = [[] for _ in range(len(X_train))]
    for tree, oob_idx in trees:
        oob_preds = tree.predict(X_train[oob_idx])
        for row_idx, pred in zip(oob_idx, oob_preds):
            oob_votes[row_idx].append(pred)

    correct = 0
    total = 0
    for i in range(len(X_train)):
        if len(oob_votes[i]) == 0:
            continue
        final_vote = Counter(oob_votes[i]).most_common(1)[0][0]
        if final_vote == y_train[i]:
            correct += 1
        total += 1
    return 1 - (correct / total), total


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split(verbose=True)
    print(f"Train/test sizes: {len(y_train)}/{len(y_test)}")

    # Bootstrap coverage check: expected 1 - 1/e ~= 0.6321
    rng = np.random.default_rng(SEED)
    coverage = [1 - len(bootstrap_indices(len(y_train), rng)[1]) / len(y_train) for _ in range(1000)]
    print(f"Bootstrap coverage (1000 trials): {np.mean(coverage):.4f} (theoretical {1 - 1 / math.e:.4f})")

    trees = train_forest(X_train, y_train)
    m = trees[0][0].n_features
    print(f"Forest: {N_TREES} trees, Gini, max_depth=None, min_samples_split=2, m={m}")

    root_feats = Counter(int(tree.root.feature) for tree, _ in trees)
    print(f"Root-split features (feature: count over {N_TREES} trees): {dict(sorted(root_feats.items()))}")
    control = train_forest(X_train, y_train, n_features=X_train.shape[1])
    control_feats = Counter(int(tree.root.feature) for tree, _ in control)
    print(f"Control, m=p (no subsetting), root-split features: {dict(sorted(control_feats.items()))}")

    err, total = oob_error(trees, X_train, y_train)
    print(f"OOB error: {err:.4f}  (evaluated on {total}/{len(X_train)} rows)")

    train_preds = forest_predict(trees, X_train)
    print(f"Train misclassified (in-bag, not a generalization metric): "
          f"{np.sum(train_preds != y_train)}/{len(y_train)}")

    forest_preds = forest_predict(trees, X_test)
    n_wrong = np.sum(forest_preds != y_test)
    print(f"Forest test accuracy: {1 - n_wrong / len(y_test):.4f}  ({n_wrong}/{len(y_test)} misclassified)")

    single = DecisionTree(max_depth=None, rng=np.random.default_rng(SEED))
    single.fit(X_train, y_train)
    single_acc = np.mean(single.predict(X_test) == y_test)
    print(f"Single fully grown tree (all features) test accuracy: {single_acc:.4f}")
