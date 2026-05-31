# Decision Tree / XGBoost Implementation: Current Status and Remaining Work

This README documents the current state of the `ANonSeriousDecisionTree` implementation, especially the XGBoost-style regression tree path with exact split search, approximate split search, weighted quantile candidates, and optional column-block preprocessing.

The current code can work under a narrow, clean setup. The core training path is usable, but several surrounding API paths are still unfinished or unsafe.

---

## Safe Usage Profile

For now, the most reliable setup is:

```python
tree = ANonSeriousDecisionTree(
    tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
    xgboost=True,
    xgboost_optimized=True,
    xgboost_split=ANonSeriousDecisionTree.XGBoostSplit.EXACT,  # safest baseline
    preprocess=True,  # or False for comparison
    purpose_missing=False,
    vectorized=False,
)
```

Before fitting, make sure the feature matrix has no missing values:

```python
assert not np.isnan(X).any()
```

If using approximate split search:

```python
tree = ANonSeriousDecisionTree(
    tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
    xgboost=True,
    xgboost_optimized=True,
    xgboost_split=ANonSeriousDecisionTree.XGBoostSplit.APPROXIMATE,
    xgboost_proposal=ANonSeriousDecisionTree.XGBoostProposal.LOCAL,  # or GLOBAL
    xgboost_candidate_proposal=ANonSeriousDecisionTree.XGBoostCandidate.WEIGHTED_QUANTILE,
    preprocess=True,
    purpose_missing=False,
    vectorized=False,
)
```

For approximate mode, tune:

```python
config.max_xgboost_bins
```

Approximate split search checks fewer thresholds than exact split search, so worse validation loss is not automatically a bug. If approximate mode performs much worse than exact mode, increase the number of bins or use exact mode as the baseline.

---

## What Currently Works

The following pieces are now structurally usable under the safe setup above:

- Constructor-level XGBoost settings are now self-contained.
- `preprocess`, `xgboost_optimized`, `purpose_missing`, `l2`, and `gamma` are now constructor parameters.
- Column-block preprocessing is implemented for XGBoost-style trees.
- Exact split search can use preprocessed sorted column blocks.
- Approximate split search can use global or local candidate thresholds.
- Weighted quantile candidate generation is structurally aligned with the row-index system.
- The normal `_build_tree()` path is the recommended training path.
- Clean, no-missing XGBoost regression trees should work with `vectorized=False`.

---

## Remaining Undone or Risky Areas

### 1. `node=True` / `fit_node()` is still broken for XGBoost

The `fit_node()` path is not safe yet.

The function now accepts `row_indices`, which is the right direction, but recursion calls:

```python
self.fit_node(node.left, row_indices=left_rows)
self.fit_node(node.right, row_indices=right_rows)
```

without passing `gradient` or `hessian`.

Inside `fit_node()`, the XGBoost leaf value still uses:

```python
node.value = -np.sum(gradient) / (np.sum(hessian) + self.l2)
```

After the first recursive call, `gradient` and `hessian` can be `None`, which can cause crashes or invalid values.

There is also a row-count bug:

```python
np.sum(left_rows) < self.minimum_split_size
np.sum(right_rows) < self.minimum_split_size
```

This sums row IDs, not the number of rows. It should be:

```python
len(left_rows) < self.minimum_split_size
len(right_rows) < self.minimum_split_size
```

Until fixed, avoid:

```python
fit(..., node=True)
```

Use the normal `_build_tree()` path instead.

---

### 2. Non-optimized XGBoost path is effectively dead

Inside `_regression_split()`, once `self.xgboost` is true, the function returns from the XGBoost block:

```python
if self.xgboost:
    ...
    if self.xgboost_optimized:
        ...
    return best_feature, best_threshold
```

So if:

```python
xgboost=True
xgboost_optimized=False
```

then the method can return `None, None` without reaching the older unoptimized XGBoost loop.

Practical rule:

```text
xgboost=True and xgboost_optimized=False => unsafe / likely no useful splits
```

Either always use:

```python
xgboost_optimized=True
```

or rewrite `_regression_split()` so the old unoptimized XGBoost path is reachable.

---

### 3. Missing values are unsupported under the current reduced scope

Missing-value-aware split search was intentionally skipped.

That means the model should only be trained on data with no missing values.

If `X` contains `np.nan`, this logic is unsafe:

```python
left_mask = values > threshold
right_mask = values <= threshold
```

For `np.nan`, both comparisons are false. The row goes neither left nor right and can silently disappear during training.

Always check:

```python
assert not np.isnan(X).any()
```

Also keep:

```python
purpose_missing=False
```

The older `purpose_missing=True` path still uses separate missing-value logic that is not integrated with the optimized column-block split search.

---

### 4. Vectorized prediction is not safe for missing/default routing

The current `vectorized_predict_search()` uses bounds and leaf indicators.

That is only safe for clean, no-missing, axis-aligned tree logic. It does not correctly handle learned default missing directions.

Since missing-value support is intentionally skipped, keep:

```python
vectorized=False
```

for the XGBoost work.

---

### 5. Global approximate candidate generation ignores candidate type

In `fit()`, global approximate mode currently always uses weighted quantile candidates:

```python
self.global_candidates[feature] = self.weighted_quantile_candidates(...)
```

This means the following settings are not fully respected in global mode:

```python
xgboost_candidate_proposal=UNWEIGHTED_QUANTILE
xgboost_candidate_proposal=RANDOM
```

Current status:

```text
GLOBAL + WEIGHTED_QUANTILE: okay
GLOBAL + UNWEIGHTED_QUANTILE: not respected
GLOBAL + RANDOM: not respected
```

If only weighted quantile is used, this is fine. Otherwise, global candidate generation should call `propose_candidates(...)`.

---

### 6. Approximate mode may perform worse than exact mode

Approximate split search uses fewer candidate thresholds. Worse performance is expected if too few bins are used.

Recommended comparison:

```text
EXACT + preprocess=False
EXACT + preprocess=True
APPROXIMATE + preprocess=False
APPROXIMATE + preprocess=True
```

If exact works well but approximate performs worse, increase:

```python
config.max_xgboost_bins
```

or use exact mode as the baseline.

---

### 7. `weighted_quantile_candidates()` needs stronger safety guards

The function currently has an empty-array guard, but it should also guard against too few values and invalid total Hessian weight.

Recommended extra guards:

```python
if len(sorted_values) <= 1:
    return np.array([])

total_weight = cumulative_weight[-1]

if total_weight <= 0:
    return np.array([])
```

This prevents invalid candidate generation when a node has too few observed values or broken Hessian weights.

---

### 8. `verbose=True` can break for regression/XGBoost

In `fit()`, verbose mode does:

```python
_, accuracy = self.evaluate_dataset(self.X_train_, self.y_train_)
```

But for regression, `evaluate_dataset()` returns a single metric, not `(predictions, accuracy)`.

So for XGBoost regression trees, `verbose=True` can crash or behave incorrectly.

This should be separated into classification and regression verbose output.

---

### 9. XGBoost feature importance is not implemented

Classification split search updates:

```python
self.feature_importance[best_feature] += ...
```

The optimized XGBoost split methods do not currently update `feature_importance`.

So feature importance output is incomplete for XGBoost.

Also, verbose feature importance can divide by zero if:

```python
total = sum(self.feature_importance.values())
```

is zero.

---

### 10. `export_text()` is classification-biased

For regression/XGBoost leaves, `export_text()` still prints classification-style output:

```python
class: {node.majority_class}
value: {node.number_of_classes}
```

For XGBoost regression trees, leaves should show something like:

```text
leaf value: node.value
samples: node.number_of_samples
```

So tree export/printing is unfinished for regression/XGBoost.

---

### 11. Pruning methods are not integrated with XGBoost

The pruning methods are mostly classification-tree-era code. They rely on majority class, classification error, and routing assumptions that do not fully match the XGBoost regression tree path.

Treat these as not integrated with XGBoost:

```text
prune_reduced_error
prune_post_complexity
prune_pessimistic
prune_error_based
prune_minimum_error
```

Do not use pruning with the current XGBoost implementation.

---

### 12. `predict_probability()` is classification-only

For regression/XGBoost, this method is not meaningful:

```python
leaf.number_of_classes / leaf.number_of_samples
```

Do not call `predict_probability()` on regression or XGBoost trees.

---

### 13. `fit()` should reset state before refitting

The tree uses:

```python
self.is_root_set
```

but `fit()` does not currently reset it.

If the same tree object is fitted twice, the second root may not be marked correctly.

At the start of `fit()`, add:

```python
self.is_root_set = False
self.root = None
self.feature_importance = {}
```

This prevents stale state from leaking across multiple fits.

---

### 14. `ANonSeriousNode` must initialize `default_missing_value_direction`

The tree accesses:

```python
node.default_missing_value_direction
```

Even if missing values are not being used, the attribute is still checked.

Make sure `ANonSeriousNode` initializes:

```python
self.default_missing_value_direction = None
```

Otherwise, you may get an `AttributeError`.

---

## Performance Debugging Checklist

If performance gets worse unexpectedly, check these first:

```python
print(tree.xgboost_optimized)
print(tree.xgboost_split)
print(tree.preprocess)
print(tree.purpose_missing)
print(tree.vectorized)
print(np.isnan(X).any())
```

Expected safe output:

```text
True
EXACT or APPROXIMATE
True or False
False
False
False
```

Most likely causes of unexpected worse performance:

```text
1. xgboost_optimized is accidentally False.
2. X contains NaNs and rows are being silently dropped.
3. The model switched from EXACT to APPROXIMATE with too few bins.
4. node=True is being used.
5. purpose_missing=True is accidentally still on.
6. vectorized=True is being used with unsupported routing assumptions.
```

---

## Bottom Line

The core training path can work:

```text
clean data
no missing values
node=False
vectorized=False
xgboost=True
xgboost_optimized=True
purpose_missing=False
```

The surrounding API is still not fully finished.

Current status:

```text
Main XGBoost training path:
    usable

Column blocks:
    usable

Exact split search:
    usable

Approximate weighted-quantile split search:
    usable but may need bin tuning

Missing-value-aware training:
    intentionally skipped / unsupported

Vectorized prediction with missing/default routing:
    intentionally skipped / unsupported

node=True path:
    unfinished

Pruning / export / probability / feature importance:
    not fully integrated with XGBoost
```

Do not treat the whole class as finished yet. Treat the clean XGBoost `_build_tree()` training path as the working chapter, and treat the rest as future cleanup.
