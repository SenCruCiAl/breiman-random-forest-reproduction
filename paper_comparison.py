import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random_forest import load_wbc, train_forest, forest_predict, oob_error

# Breiman (2001), Section 4 protocol: 100 repetitions; each time a random 10% is
# set aside, forests of 100 trees are grown with F = 1 and F = int(log2 M + 1),
# and the test error of the run with the lower OOB error is kept ("selection").
N_REPEATS = 100
TEST_SIZE = 0.1
N_TREES = 100

# Breiman (2001), Table 2, breast cancer row (test set error, %).
PAPER = {"Adaboost": 3.2, "Selection": 2.9, "Single input (F=1)": 2.7, "One tree": 6.3}

OUT_DIR_FIG = "figures"
OUT_DIR_RES = "results"
C_OURS, C_PAPER = "#2a78d6", "#eb6834"
TEXT, TEXT_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def mean_tree_oob_error(trees, X_train, y_train):
    """Average over trees of each single tree's error on its own out-of-bag rows."""
    errs = [np.mean(tree.predict(X_train[oob]) != y_train[oob]) for tree, oob in trees if len(oob)]
    return float(np.mean(errs))


def run_protocol(n_repeats=N_REPEATS):
    """Run Breiman's Section 4 protocol on WBC; return per-repetition errors as fractions."""
    X, y = load_wbc()
    f_high = int(math.log2(X.shape[1]) + 1)
    rec = {"Selection": [], "Single input (F=1)": [], f"F={f_high}": [], "F=3 (sqrt p)": [], "One tree": []}
    for r in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=r)
        runs = {}
        for f in (1, f_high, 3):
            trees = train_forest(X_tr, y_tr, n_trees=N_TREES, seed=r, n_features=f)
            test_err = np.mean(forest_predict(trees, X_te) != y_te)
            runs[f] = (test_err, oob_error(trees, X_tr, y_tr)[0], trees)
        best = 1 if runs[1][1] <= runs[f_high][1] else f_high
        rec["Selection"].append(runs[best][0])
        rec["Single input (F=1)"].append(runs[1][0])
        rec[f"F={f_high}"].append(runs[f_high][0])
        rec["F=3 (sqrt p)"].append(runs[3][0])
        rec["One tree"].append(mean_tree_oob_error(runs[best][2], X_tr, y_tr))
    return {k: np.array(v) for k, v in rec.items()}, f_high


def plot_comparison(rec):
    rows = ["Selection", "Single input (F=1)", "One tree"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ypos = np.arange(len(rows))[::-1]
    for yp, name in zip(ypos, rows):
        v = rec[name] * 100
        se = v.std(ddof=1) / math.sqrt(len(v))
        ax.errorbar(v.mean(), yp + 0.13, xerr=1.96 * se, fmt="o", ms=8, color=C_OURS, capsize=3, lw=2,
                    label="this reproduction (mean, 95% CI)" if yp == ypos[0] else None)
        ax.plot(PAPER[name], yp - 0.13, "D", ms=8, color=C_PAPER, linestyle="none",
                label="Breiman (2001), Table 2" if yp == ypos[0] else None)
        ax.text(9.9, yp, f"{v.mean():.2f}% vs {PAPER[name]:.1f}%", va="center", ha="right",
                color=TEXT, fontsize=9)
    ax.axvline(PAPER["Adaboost"], color=TEXT_MUTED, lw=1, ls="--", label="Adaboost, paper (3.2%)")
    ax.set_yticks(ypos, rows)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("test set error (%)")
    ax.set_xlim(0, 10)
    ax.set_title("Breast cancer: reproduction vs. Breiman (2001) Table 2", loc="left", color=TEXT, pad=28)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5,
              handletextpad=0.4, columnspacing=1.2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR_FIG, "paper_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR_FIG, exist_ok=True)
    os.makedirs(OUT_DIR_RES, exist_ok=True)
    rec, f_high = run_protocol()

    lines = [f"Breiman (2001) Section 4 protocol: {N_REPEATS} repetitions, {int(TEST_SIZE * 100)}% test, "
             f"{N_TREES} trees, F in {{1, {f_high}}}, selection by lower OOB error",
             f"{'setting':<22}{'ours mean %':>12}{'sd %':>8}{'SE %':>8}{'paper %':>9}"]
    for name, v in rec.items():
        v = v * 100
        paper = f"{PAPER[name]:>9.1f}" if name in PAPER else f"{'-':>9}"
        lines.append(f"{name:<22}{v.mean():>12.2f}{v.std(ddof=1):>8.2f}"
                     f"{v.std(ddof=1) / math.sqrt(len(v)):>8.2f}{paper}")
    lines.append(f"{'Adaboost':<22}{'-':>12}{'-':>8}{'-':>8}{PAPER['Adaboost']:>9.1f}")
    lines.append("'One tree' = mean OOB error of individual trees in the selected forest (paper's 4th column).")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR_RES, "paper_comparison.txt"), "w") as fh:
        fh.write(report + "\n")

    names = list(rec)
    np.savetxt(os.path.join(OUT_DIR_RES, "paper_comparison_per_repeat.csv"),
               np.column_stack([rec[k] for k in names]), delimiter=",", fmt="%.6f",
               header=",".join(names), comments="")

    plot_comparison(rec)
    print(f"Saved {OUT_DIR_RES}/paper_comparison.txt, {OUT_DIR_RES}/paper_comparison_per_repeat.csv "
          f"and {OUT_DIR_FIG}/paper_comparison.png")
