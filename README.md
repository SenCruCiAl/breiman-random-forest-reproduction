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
