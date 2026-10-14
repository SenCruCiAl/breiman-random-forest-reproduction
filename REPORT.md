# Reproducing Breiman's Random Forests on the Wisconsin Breast Cancer Data, with a Proximity-Based Outlier Extension

*Reproduction study / student portfolio project. It is not peer-reviewed and
not a novel research contribution.*

## Abstract

I implement the Random Forest algorithm of Breiman (2001) from scratch in
NumPy: CART-style Gini trees grown to full depth, bootstrap aggregation,
per-split random feature subsetting, and out-of-bag (OOB) error estimation. I
evaluate it on the original 9-feature Breast Cancer Wisconsin dataset (UCI
id=15). On a fixed 80/20 split, a 100-tree forest reaches an OOB error of
0.0293 and a test accuracy of 0.9635 (5/137 errors). A single fully grown
tree reaches 0.9197. Following the paper's own Section 4 protocol (100
repetitions, 10% test, F ∈ {1, 4}, selection by OOB error), the mean test
error is 2.58% (SE 0.19) against 2.9% in Breiman's Table 2. The single-input
error is 2.57% against 2.7%, and the individual-tree error is 6.08% against
6.3%. The extension computes the forest proximity matrix and the per-class
outlier measure from Breiman and Cutler's later Random Forests documentation;
the 2001 paper itself doesn't discuss proximities. The proximity matrix passes structural checks:
unit diagonal, symmetry, and within-class mean proximity 0.681 against
between-class 0.008. However, per-class median/MAD normalization behaves
poorly on this dataset. The benign class's MAD is about 78× smaller than the
malignant class's, so normalized scores are not comparable across classes and
a global cutoff flags only benign samples. I report this as a finding and a
limitation, not a success.

## 1. Introduction

Random Forests (Breiman, 2001) combine bagging (Breiman, 1996) with random
feature selection at each split. The result is an ensemble of decorrelated,
low-bias trees whose majority vote generalises well. The same forest also
yields a free internal error estimate (OOB error) and a sample-to-sample
similarity measure (proximity).

This project's goals are to:

1. implement the algorithm without `sklearn.ensemble`,
2. check each mechanism in isolation (bootstrap coverage, feature
   subsetting, OOB estimation),
3. compare the error rate with the published figure for the same dataset,
   and
4. implement and critically examine the proximity-based outlier measure.

## 2. Background and method summary

**Trees.** Binary CART-style classification trees. At each node, candidate
thresholds are the distinct values of each candidate feature, and the split
that maximises the decrease in Gini impurity `1 − Σ p_k²` is chosen. Trees
are grown until nodes are pure or hold fewer than 2 samples. There is no
depth cap and no pruning, following Breiman's specification.

**Bagging.** Each tree is trained on `n` rows drawn with replacement. The
expected fraction of distinct rows is `1 − (1 − 1/n)^n → 1 − 1/e ≈ 0.632`.
The remaining ≈36.8% of rows form that tree's out-of-bag set.

**Random feature subsetting.** At every node (not once per tree), `m`
features are drawn without replacement, and only they are considered for the
split. Breiman (2001, §4) calls this Forest-RI and uses two group sizes,
`F = 1` and "the first integer less than log₂ M + 1" (= 4 for M = 9). The
test error reported in the paper's "Selection" column comes from whichever
of the two has the lower OOB error. The main run here uses `m = ⌊√p⌋ = 3`,
the later convention of the `randomForest` software. The paper's two values
are what §5.3 uses. The paper doesn't mention `√p`.

**OOB error.** Each training row is classified by majority vote over only
the trees for which it was out-of-bag, then compared with its label.

**Proximity.** `prox(i, j)` is the fraction of trees in which rows `i` and `j`
land in the same terminal node.

**Outlier measure.** Proximities and the outlier measure are **not** part of
Breiman (2001); the word "proximity" doesn't occur in the paper. The measure
below follows Breiman and Cutler's later Random Forests documentation
(Breiman, 2002):

- raw score: `out(i) = n_c / Σ_{k ∈ class(i)} prox(i, k)²`
- normalized score, per class: `(out(i) − median_c) / MAD_c`

A large value means sample `i` is far, in forest terms, from the rest of its
class.

## 3. Implementation

| Component | Source |
|---|---|
| `DecisionTree`, `Node`, Gini split search, prediction, leaf lookup | from scratch (`decision_tree.py`) |
| `bootstrap_indices`, forest training, majority vote, OOB error | from scratch (`decision_tree.py`, `random_forest.py`) |
| Proximity matrix, raw and normalized outlier scores | from scratch (`proximity.py`) |
| Classical (Torgerson) MDS for the embedding figure | from scratch, NumPy `eigh` (`generate_plots.py`) |
| Train/test split | `sklearn.model_selection.train_test_split` |
| Dataset download | `ucimlrepo.fetch_ucirepo` (with a verified mirror fallback) |
| Plotting | `matplotlib` |

The proximity matrix is built per tree by grouping rows by leaf id and adding
1 to every pair within each leaf group. This costs Σ over leaves of |leaf|²
per tree, rather than a comparison of all n² pairs per tree.

## 4. Experimental setup

- **Data**: Breast Cancer Wisconsin (Original), 699 rows, 9 integer features
  (1–10), 458 benign / 241 malignant. The 16 rows with a missing *Bare
  Nuclei* value are dropped (683 remain). Labels: 2 → 0 (benign),
  4 → 1 (malignant).
  - UCI's API was unreachable from the environment that produced the recorded
    results, so the data came from a GitHub mirror of the same file.
  - I checked the mirror against UCI's published statistics (699 rows,
    458/241 split, 16 `?` entries, all in Bare Nuclei) and against the
    independent CRAN `mlbench::BreastCancer` copy. Features and labels were
    identical row by row.
- **Main run**: `train_test_split(test_size=0.2, random_state=42)` gives 546
  train and 137 test rows. 100 trees, NumPy `default_rng(42)`, `m = 3`.
- **Paper protocol** (Breiman, 2001, §4): 100 repetitions (`random_state =
  0…99`), each with a random 10% set aside as the test set. On each
  repetition:
  - Forests of 100 trees are grown with `F = 1` and `F = 4`.
  - "Selection" keeps the test error of whichever forest has the lower OOB
    error.
  - "One tree" is the mean OOB error of the individual trees in the selected
    forest (the paper's fourth column).
  - `F = 3` is also run, for comparison with the main configuration.
- **Dataset difference from the paper**: the paper's Table 1 lists breast
  cancer with 699 instances and doesn't say how missing values were handled.
  Here the 16 incomplete rows are dropped (683 rows).
- **Outlier flagging**: two rules on the normalized score, computed on the
  training set.
  - Fixed threshold: `> 3`.
  - Top 5%: 95th percentile. I report it both globally and per class.

## 5. Results

### 5.1 Core mechanism checks

| Check | Result | Expected |
|---|---|---|
| Bootstrap coverage, mean of 1000 draws (n = 546) | 0.6321 | 1 − 1/e = 0.6321 |
| Root-split features, `m = 3` (0-based index: count) | `{1: 31, 2: 24, 4: 9, 5: 13, 6: 19, 7: 4}` (6 features) | spread across features |
| Root-split features, control `m = p = 9` | `{1: 81, 2: 18, 6: 1}` (3 features) | dominated by the strongest feature |
| Train-set misclassification (all trees vote) | 0/546 | ≈0 for unpruned trees that saw most rows |

The train-set figure is **not** a performance result. Each row is in-bag for
about 63% of the trees, and fully grown trees fit their in-bag rows exactly.

### 5.2 Classification error (main split)

| Model | Metric | Value |
|---|---|---|
| Random forest, 100 trees, m = 3 | OOB error (546/546 rows) | **0.0293** |
| Random forest, 100 trees, m = 3 | Test accuracy | **0.9635** (5/137 wrong) |
| Single fully grown tree, all features | Test accuracy | 0.9197 |

The OOB error (2.93%) and the test error (3.65%) differ by 0.7 points, which
is one test sample. With n = 137, the binomial standard error of a 3.6% error
rate is √(0.036 · 0.964 / 137) ≈ 1.6 points. The two estimates are therefore
consistent, and the single split can't distinguish them.

### 5.3 Comparison with Breiman (2001), Table 2 (paper protocol)

The paper protocol is described in §4. Test set error is in %, as the mean
over 100 repetitions.

| Setting | This reproduction | SD | SE of mean | Breiman (2001) Table 2 |
|---|---|---|---|---|
| Forest-RI, selection (F ∈ {1, 4} by OOB) | 2.58 | 1.86 | 0.19 | 2.9 |
| Forest-RI, single input (F = 1) | 2.57 | 1.79 | 0.18 | 2.7 |
| One tree (mean individual-tree OOB error) | 6.08 | 0.42 | 0.04 | 6.3 |
| F = 4 only | 2.88 | 1.95 | 0.19 | – |
| F = 3 (`⌊√p⌋`) | 2.70 | 2.03 | 0.20 | – |
| Adaboost, 50 trees (not reimplemented) | – | – | – | 3.2 |

![Comparison with Breiman (2001) Table 2](figures/paper_comparison.png)

**Reading the comparison.**

- **The reproduction matches the paper's pattern.** Single-input forests do
  about as well as selection. Both beat the paper's Adaboost figure (3.2%),
  and individual trees are more than twice as error-prone as the forest.
- **The absolute gaps to Table 2 are small.** They are 0.32 points for
  selection (1.7 SE) and 0.13 for single input (0.7 SE), both within
  sampling noise. The one-tree gap is 0.22 points, which is 5.5 SE because
  that estimate is very stable, so it is a real but small difference (3.5%
  relative). The reproduction is slightly *lower* in every row.
  One plausible contributor is that the 16 rows with missing values are
  excluded here, but this was not tested. The paper also doesn't state its
  random seeds or its exact tree-growing details, so exact equality isn't
  expected.
- **Sensitivity to F is small.** F = 1, 3 and 4 differ by at most 0.31
  points. This agrees with the paper: "the procedure is not overly sensitive
  to the value of F" (§4). The paper's Figure 2 also finds its minimum
  breast-cancer error at F = 1, for its linear-combination variant.
- **An independent check from the paper's own text.** §7 reports an average
  random-forest error of 2.94% on the Wisconsin Breast Cancer data, from a
  separate experiment.

### 5.4 Proximity matrix

| Check | Value |
|---|---|
| Shape | 546 × 546 |
| Diagonal | exactly 1.0 for all rows |
| Symmetric | yes (`np.allclose(P, Pᵀ)`) |
| Mean off-diagonal proximity | 0.382 |
| Fraction of off-diagonal pairs with prox > 0 | 0.602 |
| Mean proximity, same class | 0.681 |
| Mean proximity, different class | 0.008 |

![Proximity heatmap](figures/proximity_heatmap.png)

The matrix is almost block-diagonal. Cross-class proximity is rare (0.8% on
average): benign and malignant rows hardly ever share a leaf. The benign
block is much denser than the malignant one. Many benign rows have very
similar, low feature values, so they reach the same leaves.

![MDS embedding of 1 − proximity](figures/proximity_mds.png)

Classical MDS of `1 − prox` separates the two classes along the first
dimension. Benign samples collapse onto a tight arm, and malignant samples
are spread out. The flagged outliers are covered in §7.

## 6. Discussion

**What is solid.**

- **Bagging**: bootstrap coverage matches `1 − 1/e` to four decimals.
- **Feature subsetting**: it measurably diversifies trees. Root splits spread
  over 6 features with subsetting, against 3 features with one taking 81/100
  roots without it.
- **OOB estimate**: consistent with held-out error on the main split.
- **Agreement with the paper**: under the paper's own protocol, the errors
  land within a few tenths of a point of Table 2 (§5.3).
- **Ensemble effect**: the forest beats a single fully grown tree by about
  4.4 points on the main split.
- **Proximity mechanics**: structurally correct (unit diagonal, symmetry,
  strong within-class vs between-class contrast).

**Limitations, stated plainly.**

1. **The single split is noisy.** The 80/20 split gives only 137 test rows,
   which is a 0.73-point resolution and ≈1.6-point standard error. Cite §5.3,
   not §5.2.
2. **Hyperparameters not tuned.** 100 trees and `m = ⌊√p⌋` for the main run
   were set by convention. §5.3 uses the paper's settings, but no wider
   search was done.
3. **Missing values dropped.** The 16 incomplete rows are removed (683 rows).
   The paper lists 699 instances (Table 1) and doesn't describe how it
   handled the missing values, so the evaluation sets differ.
4. **Adaboost not reimplemented.** The 3.2% Adaboost figure is quoted from
   Table 2, not reproduced.
5. **Outlier detection is train-set only.** Scores are computed on the 546
   training rows. No test-set proximities to the training forest are
   computed, so there is no out-of-sample check of the outlier measure.
6. **No outlier-vs-error correlation claim.** Train-set misclassification is
   trivially 0/546, so comparing flags against misclassified rows on the same
   set tests nothing. This is a scoped limitation, not evidence either way.
7. **Scale.** The dense n × n proximity matrix is O(n²) in memory. That is
   fine at n = 546, but larger n wasn't tested.

## 7. Custom extension: proximity-based outlier detection

### 7.1 Observed score distributions

| Statistic | Raw score | Normalized score |
|---|---|---|
| Median | 1.60 | 0.00 |
| Mean | 12.65 | 138.47 |
| Max | 349.48 | 8209.25 |

![Outlier score distributions](figures/outlier_score_distribution.png)

Both distributions are strongly right-skewed. For the raw score, the
mechanism is that fully grown trees isolate a few rows in tiny leaves across
most trees. The most extreme row (training index 211, benign) has nonzero
proximity to only 14 of its 364 same-class neighbours. Its `Σ prox²` is
therefore close to its self-proximity of 1, so its raw score approaches the
class size (365).

### 7.2 The normalization problem

The per-class normalization divides by each class's MAD:

| Class | n | Raw median | Raw MAD |
|---|---|---|---|
| Benign | 365 | 1.295 | **0.0424** |
| Malignant | 181 | 6.473 | **3.3209** |

The benign core is extremely tight: most benign rows share leaves with most
other benign rows. That makes the benign MAD about 78× smaller than the
malignant MAD. The same raw deviation is therefore amplified about 78× more
for benign rows. Consequences:

- **The fixed threshold (> 3) flags 153/546 (28.0%).** That is far too many
  to be a useful outlier rate, mainly because tiny benign deviations
  become large normalized scores.
- **A global top-5% cutoff (normalized > 65.04) flags 28/546, and all 28 are
  benign (0/181 malignant).** The earlier version of this project adopted the
  global top 5% as "more interpretable". The class breakdown shows it is
  really a benign-only ranking, not a class-neutral outlier rule.
- **A per-class top-5% cutoff flags 19/365 benign and 9/181 malignant.** It is
  class-balanced by construction. I recommend it for this dataset, with the
  caveat that "5%" is a chosen rate, not a detected one.

### 7.3 What the flagged points are

In the MDS embedding, the flagged benign rows sit away from the tight benign
arm, in the direction of the malignant cloud. Several lie inside it. This
fits their definition: rows that the forest rarely groups with typical benign
cases. Plausibly they are benign cases with malignant-like feature values.
Checking that interpretation would require the out-of-sample and
misclassification analyses listed as limitations 4 and 5. It is a hypothesis,
not a result.

## 8. Conclusion

The from-scratch implementation reproduces the core Random Forest mechanisms,
and each one is checked directly:

- bootstrap coverage matches theory,
- feature subsetting measurably diversifies trees,
- OOB error agrees with held-out error, and
- the forest clearly beats a single tree.

Under the paper's own protocol, the error rates reproduce Breiman (2001),
Table 2, to within a few tenths of a percentage point: 2.58% vs. 2.9%
(selection) and 2.57% vs. 2.7% (single input) (§5.3). The proximity matrix is correct and cleanly separates the classes.
The outlier extension's main finding, however, is a caution: with per-class
median/MAD normalization, scores are not comparable across classes when class
compactness differs this much. Global thresholds should be replaced by
per-class ones, and the outlier measure needs an out-of-sample check before
its flags mean anything.

## References

- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L. (2002). *Manual on Setting Up, Using, and Understanding Random
  Forests V3.1*. Statistics Department, University of California, Berkeley.
  (Source of the proximity-based outlier measure; the 2001 paper doesn't
  cover proximities.)
- Wolberg, W. (1990). Breast Cancer Wisconsin (Original) [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5HP4Z
