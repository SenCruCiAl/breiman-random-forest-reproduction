# Breiman Random Forest Reproduction

From-scratch implementation of Random Forests reproducing the core algorithmic
mechanisms described in Breiman, L. (2001), *Random Forests*, Machine Learning,
45(1), 5–32, extended with a proximity-based outlier detection method also
proposed in the original paper (Section 4) but rarely implemented in typical
from-scratch reproductions.

This is a reproduction study and educational project, not a peer-reviewed
publication. Results are archived here and, once complete, on Zenodo with a DOI.

## Overview

This project implements, from first principles (NumPy only — no
`sklearn.ensemble`), the three core mechanisms of Breiman's Random Forest
algorithm:

1. **Bootstrap aggregation (bagging)** — each tree trained on a bootstrap
   resample of the training data
2. **Random feature subsetting** — at every split, only a random subset of
   `m = √p` features considered as candidates
3. **Out-of-bag (OOB) error estimation** — internal validation using each
   tree's held-out bootstrap samples, without touching the test set

and extends it with:

4. **Proximity-based outlier detection** (Breiman, 2001, Section 4) — a
   forest-derived similarity matrix used to flag atypical samples within each
   class

## Repository structure
├── decision_tree.py # Node, DecisionTree, bootstrap_indices
├── random_forest.py # Data loading, forest training, OOB validation
├── proximity.py # Proximity matrix + outlier detection extension
├── README.md
└── .gitignore


## Dataset

**Breast Cancer Wisconsin (Original)**, UCI ML Repository, id=15
(Wolberg, 1990; DOI: 10.24432/C5HP4Z) — 699 instances, 9 integer-valued
features, 2 classes (benign/malignant). 16 rows with missing values dropped,
matching the paper's reported 683-instance evaluation set.

Note: this is the *original* 9-feature WBC dataset used in Breiman (2001),
**not** the 30-feature Wisconsin Diagnostic (WDBC) dataset bundled with
`sklearn.datasets.load_breast_cancer()`. The two are frequently confused;
this reproduction uses the dataset the paper actually reports on.

## Method summary

- **Bagging**: each of `n_trees` trees is trained on `n` samples drawn with
  replacement from the training set. On average ≈63.2% of training rows are
  included per tree (`1 - 1/e`), the remaining ≈36.8% form that tree's
  out-of-bag set.
- **Feature subsetting**: at each split, `m = √p ≈ 3` (for this 9-feature
  dataset) randomly chosen features are evaluated as split candidates, rather
  than all `p` features — this decorrelates trees beyond what bagging alone
  achieves (Breiman, 1996).
- **OOB error**: for each training row, only trees that did not see that row
  during training vote on it; majority vote across those trees is compared
  against the true label.
- **Proximity matrix**: for every pair of samples, the fraction of trees in
  which they land in the same leaf node, computed efficiently via leaf-based
  grouping (avoids the naive $O(n^2 T)$ pairwise comparison).

## Results

| Metric                          | Value                  |
|----------------------------------|-------------------------|
| Bootstrap coverage (1000 trials) | 0.6322 (theoretical: 0.6321) |
| Forest size                      | 100 trees, max_depth=10 |
| OOB error                        | 0.0311 (546/546 rows evaluated) |
| Forest accuracy (held-out test)  | 0.9649–0.9737 (run-dependent) |
| Single-tree baseline accuracy    | 0.9386–0.9474 |
| Breiman (2001) reported error range (this dataset) | ≈0.029–0.037 |

The OOB error falls within Breiman's reported range for this dataset,
supporting a valid reproduction of the algorithm's core statistical behavior.

Feature-subsetting validation: root-split feature distribution flattened
from being dominated by a single feature (6/10 trees) under unrestricted
splitting to a spread across 5–6 distinct features once random subsetting
was correctly wired in — direct evidence the mechanism is functioning as
intended, not merely present in the code.

Proximity matrix validated on a 546-sample slice: diagonal exactly 1.0
(self-proximity), matrix symmetric, full computation in ~4 seconds using
leaf-grouped aggregation.

## Limitations

- Single train/test split; Breiman's reported numbers are typically averaged
  across multiple trials
- Hyperparameters (`max_depth`, `n_trees`) set by inspection, not tuned to
  match Breiman's exact experimental configuration
- Proximity computation is `O(n²)` in the worst case (unbalanced leaves);
  tractable at this dataset's scale but not demonstrated at larger `n`

## Usage

```bash
pip install numpy scikit-learn ucimlrepo

python random_forest.py    # trains forest, reports OOB error and accuracy
python proximity.py        # computes proximity matrix and outlier scores
```

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140.
- Wolberg, W. (1990). Breast Cancer Wisconsin (Original) [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5HP4Z

## License

MIT
