import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from random_forest import load_split, train_forest
from proximity import (compute_proximity_matrix, compute_raw_outlier_scores,
                       normalize_outlier_scores, OUTLIER_THRESHOLD, OUTLIER_PERCENTILE)

OUT_DIR = "figures"
CLASS_NAMES = {0: "benign", 1: "malignant"}
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "axes.titlecolor": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight",
})


def classical_mds(dissimilarity, k=2):
    """Classical (Torgerson) MDS: embed a dissimilarity matrix into k dimensions."""
    n = dissimilarity.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dissimilarity ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:k]
    return eigvecs[:, order] * np.sqrt(np.maximum(eigvals[order], 0))


def plot_heatmap(prox, y):
    order = np.lexsort((np.arange(len(y)), y))
    P = prox[np.ix_(order, order)]
    n_benign = np.sum(y == 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(P, cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
    for pos in (n_benign - 0.5,):
        ax.axhline(pos, color=TEXT, lw=0.8)
        ax.axvline(pos, color=TEXT, lw=0.8)
    ticks = [n_benign / 2, n_benign + (len(y) - n_benign) / 2]
    labels = [f"benign (n={n_benign})", f"malignant (n={len(y) - n_benign})"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels, rotation=90, va="center")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("proximity (fraction of trees sharing a leaf)")
    cb.outline.set_visible(False)
    ax.set_title("Random forest proximity matrix, training set (sorted by class)", loc="left")
    fig.savefig(os.path.join(OUT_DIR, "proximity_heatmap.png"))
    plt.close(fig)


def plot_outlier_hist(raw, norm, y):
    cutoff = np.percentile(norm, OUTLIER_PERCENTILE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bins = np.logspace(np.log10(raw.min()), np.log10(raw.max()), 40)
    for c in (0, 1):
        ax.hist(raw[y == c], bins=bins, histtype="step", color=CLASS_COLORS[c],
                linewidth=2, label=CLASS_NAMES[c])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("raw outlier score (class size / Σ prox², log scale)")
    ax.set_ylabel("samples (log scale)")
    ax.set_title("Raw outlier scores: heavy right tail", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.6)

    ax = axes[1]
    shifted = norm - norm.min() + 1  # normalized scores go negative; shift for a log axis
    bins = np.logspace(0, np.log10(shifted.max()), 40)
    for c in (0, 1):
        ax.hist(shifted[y == c], bins=bins, histtype="step", color=CLASS_COLORS[c],
                linewidth=2, label=CLASS_NAMES[c])
    for val, name, ls in [(OUTLIER_THRESHOLD, f"fixed threshold = {OUTLIER_THRESHOLD:g}", "--"),
                          (cutoff, f"top {100 - OUTLIER_PERCENTILE}% cutoff = {cutoff:.1f}", "-")]:
        ax.axvline(val - norm.min() + 1, color=TEXT, lw=1, ls=ls)
        ax.text(val - norm.min() + 1, ax.get_ylim()[1] * 0.97, " " + name, color=TEXT,
                fontsize=8.5, va="top", rotation=90, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"normalized score − min + 1  (min = {norm.min():.2f}; log scale)")
    ax.set_ylabel("samples (log scale)")
    ax.set_title("Per-class median/MAD-normalized scores", loc="left")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.8))
    ax.grid(axis="y", color=GRID, lw=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "outlier_score_distribution.png"))
    plt.close(fig)


def plot_mds(prox, y, norm):
    emb = classical_mds(1 - prox)
    flagged = norm > np.percentile(norm, OUTLIER_PERCENTILE)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for c in (0, 1):
        idx = y == c
        ax.scatter(emb[idx, 0], emb[idx, 1], s=22, color=CLASS_COLORS[c],
                   edgecolor="white", linewidth=0.8, label=CLASS_NAMES[c], zorder=2)
    ax.scatter(emb[flagged, 0], emb[flagged, 1], s=70, facecolor="none", edgecolor=TEXT,
               linewidth=1, label=f"flagged outlier (top {100 - OUTLIER_PERCENTILE}%)", zorder=3)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title("Classical MDS of 1 − proximity, training set", loc="left")
    ax.legend(frameon=False)
    ax.grid(color=GRID, lw=0.6)
    fig.savefig(os.path.join(OUT_DIR, "proximity_mds.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    trees = train_forest(X_train, y_train)
    prox = compute_proximity_matrix(trees, X_train)
    raw = compute_raw_outlier_scores(prox, y_train)
    norm = normalize_outlier_scores(raw, y_train)

    plot_heatmap(prox, y_train)
    plot_outlier_hist(raw, norm, y_train)
    plot_mds(prox, y_train, norm)
    print(f"Saved figures to {OUT_DIR}/")
