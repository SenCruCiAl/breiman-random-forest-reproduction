# Part 2: Gini vs. Entropy, a Follow-Up Study

This is the second stage of the Breiman Random Forest reproduction. Part 1
([`README.md`](README.md), [`REPORT.md`](REPORT.md)) rebuilt Random Forests
from scratch and matched Breiman (2001) Table 2 on the Wisconsin Breast
Cancer data. That work fixed one design choice without testing it: every
tree split on **Gini impurity**, the CART and Breiman default.

Part 2 tests that choice. The question is:

> Does replacing Gini impurity with entropy (information gain) change the
> forest's error, or the trees it builds, under the paper's own protocol?

The Part 1 code and results are left unchanged. Everything in this study
lives in its own files and reuses Part 1 only through imports.

## What carries over from Part 1

- **Data:** Wisconsin Breast Cancer (Original), 683 complete rows, 9
  features, 2 classes.
- **Trees and forests:** the from-scratch `DecisionTree`, bootstrap sampling,
  majority vote and OOB error from Part 1, all unchanged.
- **Protocol:** Breiman (2001) §4. There are 100 repetitions, each with a 10%
  test set and 100 trees. F is selected from {1, ⌊log₂ 9 + 1⌋ = 4} by OOB
  error.
- **Baseline:** the Gini rows below reproduce the Part 1 paper-protocol
  results exactly (2.58% selection, 2.57% single input, 6.08% one tree).

## What is new

| File | Purpose |
|---|---|
| [`split_criteria.py`](split_criteria.py) | `CriterionTree`, a subclass of the Part 1 tree that adds `criterion="entropy"`, plus a matching `train_forest`. With `criterion="gini"` it gives the same trees as Part 1. |
| [`criterion_comparison.py`](criterion_comparison.py) | Paired 100-repetition comparison and a per-node split-agreement audit. |
| [`results/criterion_comparison.txt`](results/criterion_comparison.txt) | Summary output. |
| [`results/criterion_comparison_per_repeat.csv`](results/criterion_comparison_per_repeat.csv) | Per-repetition test errors for every setting and criterion. |

Entropy is `−Σ p_k log₂ p_k`, and a split is scored by its information gain.
Gini is `1 − Σ p_k²`.

## Design

**Paired comparison.** In repetition r, both criteria get the same
train/test split (`random_state=r`) and the same forest seed (`seed=r`). The
only thing that changes is the split criterion, so per-repetition
differences measure the criterion effect directly. They're tested with a
Wilcoxon signed-rank test.

**Split-agreement audit.** Equal error doesn't mean equal trees. 100 Gini
trees are grown on bootstrap samples of all 683 rows. At every internal node,
the audit records whether entropy would choose the same feature and threshold
from the same candidate features. This is repeated for F = 1, 4 and 9.

## Results

### Test error (100 paired repetitions)

| Setting | Gini, % | Entropy, % | Entropy − Gini, points (95% CI) | Wilcoxon p | Gini better / tie / entropy better |
|---|---|---|---|---|---|
| Selection | 2.58 | 2.54 | −0.04 (−0.23, +0.15) | 0.61 | 13 / 70 / 17 |
| Single input (F = 1) | 2.57 | 2.51 | −0.06 (−0.23, +0.11) | 0.54 | 9 / 76 / 15 |
| F = 4 | 2.88 | 2.80 | −0.09 (−0.32, +0.15) | 0.81 | 24 / 47 / 29 |
| One tree | 6.08 | 6.02 | −0.06 (−0.14, +0.03) | 0.36 | 45 / 0 / 55 |

### Split agreement

| F | Internal nodes | Disagree, all | n 2–9 | n 10–49 | n ≥ 50 | Median node size, agree / disagree |
|---|---|---|---|---|---|---|
| 1 | 3183 | 8.8% | 0.4% | 5.5% | 18.8% | 21 / 170 |
| 4 | 2608 | 15.3% | 0.8% | 9.3% | 35.1% | 16 / 239 |
| 9 | 2139 | 20.3% | 0.9% | 12.2% | 42.3% | 17 / 266 |

## Findings

1. **No detectable difference in error.** Every 95% confidence interval
   contains 0, and no Wilcoxon p is below 0.3. Entropy is nominally lower by
   0.04–0.09 points, which is well inside the noise. In 47–76 of the 100
   repetitions, the two forests make exactly the same test error.
2. **The trees do differ.** Entropy and Gini choose a different split at
   8.8% of nodes for F = 1, 15.3% for F = 4 and 20.3% for F = 9. Disagreement
   grows with F because more candidate features give the two criteria more
   room to rank splits differently. This is well above the ≈2% disagreement
   reported by Raileanu and Stoffel (2004), so their figure shouldn't be
   treated as a general rule.
3. **The disagreements are near the root.** They are rare at small nodes
   (under 1% below 10 rows) but common at large ones (19–42% at 50 rows or
   more). The median node size is 16–21 where the criteria agree and 170–266
   where they disagree. The two criteria build visibly different upper trees,
   yet the fully grown forests end up with the same error. A plausible reason,
   not tested here, is that this dataset is close to separable, so several
   different top splits lead to equally good partitions once the trees are
   grown out.
4. **Conclusion.** The Part 1 choice of Gini is confirmed, not just assumed.
   Switching to entropy wouldn't change any Part 1 conclusion. This agrees
   with Raileanu and Stoffel's (2004) theoretical result that the two
   criteria rarely lead to different accuracy.

## Limitations

- The study uses a single dataset, and a near-separable one. Criteria may
  diverge more on noisier or multi-class data.
- Only Gini and entropy were compared. Gain ratio and other criteria were
  not tested.
- Agreement is measured on Gini-grown trees. Below the first disagreement,
  an entropy-grown tree would see different subsets.

## Reproduce

```bash
pip install -r requirements.txt scipy
python criterion_comparison.py  # about 2–3 min on 4 cores; writes results/criterion_comparison.*
```

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
  *Classification and Regression Trees*. Wadsworth.
- Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*,
  1(1), 81–106.
- Raileanu, L. E., & Stoffel, K. (2004). Theoretical comparison between the
  Gini Index and Information Gain criteria. *Annals of Mathematics and
  Artificial Intelligence*, 41(1), 77–93.
  https://doi.org/10.1023/B:AMAI.0000018580.96245.c6
- Wolberg, W. (1990). Breast Cancer Wisconsin (Original) [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5HP4Z
