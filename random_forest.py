from dec import DecisionTree, bootstrap_indices
import numpy as np
import math
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from collections import Counter


def forest_predict(trees, X):
    all_preds = np.array([tree.predict(X) for tree, _ in trees])
    final_preds = np.array([
        Counter(all_preds[:, i]).most_common(1)[0][0]
        for i in range(all_preds.shape[1])
    ])
    return final_preds

breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

print("Missing values per column:")
print(X.isna().sum())
print("Total rows before dropping:", len(X))

# Drop rows with missing values
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]
print("Total rows after dropping:", len(X))

# Remap labels: 2=benign -> 0, 4=malignant -> 1
y = y.replace({2: 0, 4: 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

rng = np.random.default_rng(42)
n_trees = 100
m = max(1, int(math.sqrt(X_train.shape[1])))

trees = []

for i in range(n_trees):
    sample_idx, oob_idx = bootstrap_indices(len(y_train), rng)
    tree = DecisionTree(max_depth=10, n_features=m, rng=rng)
    tree.fit(X_train[sample_idx], y_train[sample_idx])
    trees.append((tree, oob_idx))

test_tree, _ = trees[0]
leaf1 = test_tree.get_leaf(X_train[0])
leaf2 = test_tree.get_leaf(X_train[1])
print(f"Sample 0 -> leaf_id {leaf1.leaf_id}")
print(f"Sample 1 -> leaf_id {leaf2.leaf_id}")
print(f"Same leaf? {leaf1.leaf_id == leaf2.leaf_id}")

for tree, _ in trees[:10]:
    root = tree.root
    print(f"feature={root.feature}, threshold={root.threshold}")

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

oob_error = 1 - (correct / total)
print(f"OOB error: {oob_error:.4f}  (evaluated on {total}/{len(X_train)} rows)")

forest_preds = forest_predict(trees, X_test)
forest_acc = np.sum(forest_preds == y_test) / len(y_test)
print(f"Forest accuracy: {forest_acc:.4f}")
