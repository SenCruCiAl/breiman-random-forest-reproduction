import numpy as np
from collections import defaultdict
from random_forest import load_split, train_forest, forest_predict

OUTLIER_THRESHOLD = 3.0
OUTLIER_PERCENTILE = 95


def compute_proximity_matrix(trees, X):
    """Fraction of trees in which each pair of rows of X shares a leaf (leaf-grouped, not naive n^2 per tree)."""
    n = X.shape[0]
    proximity = np.zeros((n, n))
    for tree, _ in trees:
        leaf_groups = defaultdict(list)
        for i in range(n):
            leaf = tree.get_leaf(X[i])
            leaf_groups[leaf.leaf_id].append(i)
        for group in leaf_groups.values():
            idx = np.array(group)
            proximity[np.ix_(idx, idx)] += 1
    proximity /= len(trees)
    return proximity


def compute_raw_outlier_scores(prox, y):
    """Raw outlier score: class size / sum of squared proximities to same-class samples."""
    n = len(y)
    outlier_raw = np.zeros(n)
    for i in range(n):
        same_class = np.where(y == y[i])[0]
        sum_sq_prox = np.sum(prox[i, same_class] ** 2)
        outlier_raw[i] = len(same_class) / sum_sq_prox
    return outlier_raw


def normalize_outlier_scores(raw, y):
    """Per-class (raw - median) / MAD normalization of raw outlier scores."""
    normalized = np.zeros_like(raw)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        med = np.median(raw[idx])
        mad = np.median(np.abs(raw[idx] - med))
        normalized[idx] = (raw[idx] - med) / mad if mad > 0 else raw[idx] - med
    return normalized


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    trees = train_forest(X_train, y_train)

    prox = compute_proximity_matrix(trees, X_train)
    n = len(y_train)
    print(f"Proximity matrix shape: {prox.shape}")
    print(f"Diagonal all exactly 1.0? {np.all(np.diag(prox) == 1.0)}")
    print(f"Symmetric? {np.allclose(prox, prox.T)}")
    off = prox[~np.eye(n, dtype=bool)]
    print(f"Off-diagonal proximity: mean {off.mean():.4f}, fraction of pairs with prox > 0: {np.mean(off > 0):.4f}")
    same = (y_train[:, None] == y_train[None, :]) & ~np.eye(n, dtype=bool)
    diff = y_train[:, None] != y_train[None, :]
    print(f"Mean proximity, same class: {prox[same].mean():.4f}; different class: {prox[diff].mean():.4f}")

    raw = compute_raw_outlier_scores(prox, y_train)
    print(f"Raw outlier score: median {np.median(raw):.2f}, mean {raw.mean():.2f}, max {raw.max():.2f}")
    norm = normalize_outlier_scores(raw, y_train)
    print(f"Normalized outlier score: median {np.median(norm):.2f}, mean {norm.mean():.2f}, max {norm.max():.2f}")

    flag_fixed = norm > OUTLIER_THRESHOLD
    cutoff = np.percentile(norm, OUTLIER_PERCENTILE)
    flag_pct = norm > cutoff
    print(f"Flagged (normalized > {OUTLIER_THRESHOLD}): {flag_fixed.sum()}/{n} ({flag_fixed.mean():.1%})")
    print(f"Flagged (top {100 - OUTLIER_PERCENTILE}%, normalized > {cutoff:.2f}): {flag_pct.sum()}/{n}")
    for c, name in [(0, "benign"), (1, "malignant")]:
        print(f"  top-{100 - OUTLIER_PERCENTILE}% flags in class {name}: {np.sum(flag_pct & (y_train == c))}"
              f"/{np.sum(y_train == c)}")

    for c, name in [(0, "benign"), (1, "malignant")]:
        idx = y_train == c
        med = np.median(raw[idx])
        mad = np.median(np.abs(raw[idx] - med))
        n_c = np.sum(norm[idx] > np.percentile(norm[idx], OUTLIER_PERCENTILE))
        print(f"  class {name}: raw median {med:.3f}, MAD {mad:.4f}; "
              f"per-class top-{100 - OUTLIER_PERCENTILE}% flags: {n_c}/{idx.sum()}")

    top = np.argmax(norm)
    same_idx = np.where(y_train == y_train[top])[0]
    n_nonzero = np.sum(prox[top, same_idx] > 0) - 1  # exclude self
    print(f"Most outlying train row {top}: normalized {norm[top]:.2f}, "
          f"prox > 0 to {n_nonzero}/{len(same_idx) - 1} same-class samples")

    train_wrong = forest_predict(trees, X_train) != y_train
    print(f"Train misclassified: {train_wrong.sum()} -> overlap with outlier flags is not a meaningful test")
