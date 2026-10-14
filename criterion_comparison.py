import math
import os
from multiprocessing import Pool
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from decision_tree import bootstrap_indices
from random_forest import load_wbc, forest_predict, oob_error
from split_criteria import CriterionTree, train_forest
from paper_comparison import N_REPEATS, TEST_SIZE, N_TREES, mean_tree_oob_error

# Gini vs. entropy (information gain) under Breiman (2001) Section 4 protocol.
# Paired design: for each repetition r both criteria see the same train/test split
# (random_state=r) and the same forest seed (seed=r). The Gini rows therefore
# reproduce results/paper_comparison.txt exactly.
CRITERIA = ("gini", "entropy")
OUT_DIR_RES = "results"


def one_repeat(r, X, y):
    f_high = int(math.log2(X.shape[1]) + 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=r)
    out = {}
    for crit in CRITERIA:
        runs = {}
        for f in (1, f_high):
            trees = train_forest(X_tr, y_tr, n_trees=N_TREES, seed=r, n_features=f, criterion=crit)
            runs[f] = (np.mean(forest_predict(trees, X_te) != y_te), oob_error(trees, X_tr, y_tr)[0], trees)
        best = 1 if runs[1][1] <= runs[f_high][1] else f_high
        out[(crit, "Selection")] = runs[best][0]
        out[(crit, "Single input (F=1)")] = runs[1][0]
        out[(crit, f"F={f_high}")] = runs[f_high][0]
        out[(crit, "One tree")] = mean_tree_oob_error(runs[best][2], X_tr, y_tr)
    return out


class SplitAuditTree(CriterionTree):
    """Gini tree that, at every internal node, also records whether entropy would pick the same split
    from the same candidate features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, criterion="gini", **kwargs)
        self.agree, self.node_sizes = [], []

    def _best_split(self, X, y, feat_idxs):
        g = super()._best_split(X, y, feat_idxs)
        self._impurity = self._entropy
        e = super()._best_split(X, y, feat_idxs)
        self._impurity = self._gini
        if g[0] is not None:
            self.agree.append(g == e)
            self.node_sizes.append(len(y))
        return g


def split_agreement(X, y, n_trees=N_TREES, seed=0):
    """Per F: (agree flags, node sizes) over all internal nodes; agree = same (feature, threshold) chosen."""
    f_high = int(math.log2(X.shape[1]) + 1)
    res = {}
    for f in (1, f_high, X.shape[1]):
        rng = np.random.default_rng(seed)
        agree, sizes = [], []
        for _ in range(n_trees):
            idx, _ = bootstrap_indices(len(y), rng)
            t = SplitAuditTree(n_features=f, rng=rng)
            t.fit(X[idx], y[idx])
            agree += t.agree
            sizes += t.node_sizes
        res[f] = (np.array(agree), np.array(sizes))
    return res


if __name__ == "__main__":
    os.makedirs(OUT_DIR_RES, exist_ok=True)
    X, y = load_wbc(verbose=True)
    with Pool() as pool:
        reps = pool.starmap(one_repeat, [(r, X, y) for r in range(N_REPEATS)])
    settings = [k[1] for k in reps[0] if k[0] == "gini"]

    lines = [f"Gini vs. entropy, Breiman (2001) Section 4 protocol: {N_REPEATS} paired repetitions, "
             f"{int(TEST_SIZE * 100)}% test, {N_TREES} trees, same split and seed per repetition",
             f"{'setting':<20}{'gini %':>9}{'entropy %':>11}{'diff (e-g) %':>14}{'SE diff':>9}"
             f"{'95% CI diff':>18}{'Wilcoxon p':>12}{'g<e/tie/g>e':>14}"]
    cols, names = [], []
    for s in settings:
        g = np.array([rep[("gini", s)] for rep in reps]) * 100
        e = np.array([rep[("entropy", s)] for rep in reps]) * 100
        d = e - g
        se = d.std(ddof=1) / math.sqrt(len(d))
        p = wilcoxon(e, g).pvalue if np.any(d != 0) else float("nan")
        lines.append(f"{s:<20}{g.mean():>9.2f}{e.mean():>11.2f}{d.mean():>+14.2f}{se:>9.2f}"
                     f"{f'[{d.mean() - 1.96 * se:+.2f}, {d.mean() + 1.96 * se:+.2f}]':>18}{p:>12.3f}"
                     f"{f'{np.sum(d > 0)}/{np.sum(d == 0)}/{np.sum(d < 0)}':>14}")
        cols += [g / 100, e / 100]
        names += [f"gini {s}", f"entropy {s}"]
    lines.append("diff = entropy - gini, in percentage points; Wilcoxon signed-rank on paired repetitions.")
    lines.append("g<e/tie/g>e = repetitions where Gini was better / equal / worse than entropy.")

    lines.append("")
    lines.append(f"Split agreement (all internal nodes of {N_TREES} Gini trees on bootstrap samples of the full "
                 "683 rows; same candidate features offered to both criteria):")
    for f, (agree, sizes) in split_agreement(X, y).items():
        by_size = ", ".join(f"n {lo}-{hi - 1}: {100 * (1 - agree[(sizes >= lo) & (sizes < hi)].mean()):.1f}%"
                            for lo, hi in ((2, 10), (10, 50), (50, len(y) + 1)))
        lines.append(f"  F={f}: disagree at {100 * (1 - agree.mean()):.1f}% of {len(agree)} nodes; "
                     f"by node size {by_size}; median node size agree {np.median(sizes[agree]):.0f}, "
                     f"disagree {np.median(sizes[~agree]):.0f}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR_RES, "criterion_comparison.txt"), "w") as fh:
        fh.write(report + "\n")
    np.savetxt(os.path.join(OUT_DIR_RES, "criterion_comparison_per_repeat.csv"), np.column_stack(cols),
               delimiter=",", fmt="%.6f", header=",".join(names), comments="")
    print(f"Saved {OUT_DIR_RES}/criterion_comparison.txt and {OUT_DIR_RES}/criterion_comparison_per_repeat.csv")
