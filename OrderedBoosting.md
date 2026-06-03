# CatBoost Ordered Boosting: Implementation Notes

This README is a design document for implementing **actual CatBoost-style Ordered Boosting** after already building:

1. ordinary decision trees,
2. ordinary gradient boosting,
3. CatBoost-style ordered target statistics,
4. an ordered-boosting-lite approximation.

The goal here is not to dump final production code. The goal is to explain the real algorithm clearly enough that the implementation can be planned deliberately instead of guessed.

---

## 1. Why Ordered Boosting Exists

Standard gradient boosting has a subtle training bias.

At boosting iteration `t`, the model has a current prediction for each training row:

```text
F_{t-1}(x_i)
```

Then it computes a gradient or residual:

```text
gradient_i = dLoss(y_i, F_{t-1}(x_i))
```

The problem is that `F_{t-1}` was trained using the same training row `i` in earlier iterations.

So the prediction for a training object is not truly out-of-sample. It has already been influenced by that object's own target through previous trees.

This can create **prediction shift**:

```text
training prediction distribution != test prediction distribution
```

In plain words:

> The model may behave too optimistically on training rows because the training predictions are produced by models that have already seen those rows.

Ordered Boosting tries to reduce this.

---

## 2. Core Principle

The principle is the same as ordered target statistics.

### Ordered Target Statistics

For categorical encoding:

```text
encoding for row i uses only rows before i in a permutation
```

So row `i` does not use its own target while creating its categorical feature value.

### Ordered Boosting

For gradient computation:

```text
gradient for row i is computed using a model trained only on rows before i in a permutation
```

So row `i` does not use a prediction from a model that already learned from row `i`.

The shared rule is:

```text
When estimating something for row i,
use only previous rows in the permutation.
```

---

## 3. Plain Boosting vs Ordered Boosting

### Plain Boosting

```text
initialize F with base prediction

for each boosting round:
    residual = -loss_gradient(y, F)
    train tree on all rows using residual
    update F on all rows
```

Problem:

```text
F for row i was produced by earlier trees trained using row i.
```

### Ordered Boosting

```text
choose permutation sigma

for each row i:
    position = position of i in sigma
    prediction_i = model trained only on rows before position
    residual_i = -loss_gradient(y_i, prediction_i)

train/update model using these ordered residuals
```

This makes each training row behave more like an unseen future row.

---

## 4. The Naive Actual Algorithm

Imagine a permutation:

```text
sigma = [row_7, row_2, row_5, row_1, row_4, ...]
```

The theoretically clean ordered boosting approach keeps prefix models:

```text
M_0 = model trained on first 0 rows
M_1 = model trained on first 1 row
M_2 = model trained on first 2 rows
...
M_n = model trained on all n rows
```

For a row at permutation position `j`, compute its gradient using:

```text
M_{j-1}
```

not the full model.

Pseudocode:

```text
choose random permutation sigma

initialize M_0, M_1, ..., M_n with base prediction

for each boosting iteration:

    for each row i:
        j = position of i in sigma
        prediction_i = M_{j-1}(x_i)
        residual_i = -loss_gradient(y_i, prediction_i)

    for each prefix length j:
        train/update M_j using rows with position <= j
```

This is conceptually clean but computationally brutal.

It requires too many models.

---

## 5. Why the Naive Version Is Too Expensive

If there are `n` rows, the naive method maintains approximately:

```text
n prefix models
```

If there are multiple permutations, it becomes even worse:

```text
number_of_permutations * n prefix models
```

This is why actual CatBoost uses supporting approximations and shared tree structures instead of literally training a separate full model for every prefix.

---

## 6. Supporting Models / Supporting Predictions

Actual CatBoost keeps training-time prediction states of the form:

```text
M_{r, j}(i)
```

Where:

```text
r = permutation index
j = prefix length
i = object being predicted
```

Interpretation:

```text
prediction for object i
using a model trained only on the first j objects
from permutation r
```

For row `i`, CatBoost uses:

```text
M_{r, position_r(i) - 1}(i)
```

Meaning:

```text
Use permutation r.
Find where row i appears.
Use the prediction state trained only on rows before i.
```

Then compute:

```text
ordered_gradient_i = loss_gradient(y_i, M_{r, position_r(i)-1}(i))
```

This is the real ordered gradient.

---

## 7. Multiple Permutations

Actual CatBoost uses multiple permutations, not just one.

Why?

If a row appears very early in one permutation, it has very little previous history:

```text
position 1 -> no previous rows
position 2 -> one previous row
position 3 -> two previous rows
```

This can make ordered estimates noisy.

With several permutations, the same object may appear early in one permutation and later in another. This reduces variance.

Conceptually:

```text
permutations = [sigma_0, sigma_1, ..., sigma_s]
```

Some permutations are used for split selection. One permutation may be used for final leaf estimation.

---

## 8. Coupling Ordered Target Statistics With Ordered Boosting

Ordered target statistics and ordered boosting must not fight each other.

For row `i`, the system must ensure:

```text
1. categorical encoding for row i does not use y_i
2. gradient prediction for row i does not use y_i
3. tree split scoring for row i does not indirectly use y_i
```

This means the permutation used for ordered categorical statistics should be aligned with the permutation used for ordered boosting.

Otherwise, even if the gradient side is ordered, the feature side may leak target information.

Practical rule:

```text
For a given permutation r:
    use sigma_r for both:
        ordered target statistics
        ordered boosting prediction/gradient logic
```

---

## 9. Oblivious Trees

CatBoost commonly uses **oblivious trees**, also called **symmetric trees**.

In a normal decision tree, different branches can use different splits:

```text
root: feature A
    left: feature B
    right: feature C
```

In an oblivious tree, every level uses the same split across all nodes at that depth:

```text
level 0: split A
level 1: split B
level 2: split C
```

Every row follows the same sequence of questions:

```text
A? B? C?
```

The final leaf is represented by a binary path code.

Example:

```text
A = false
B = true
C = false

leaf = 010
```

This structure is important because CatBoost can share one tree structure across many supporting models while using different leaf values for different prefixes.

---

## 10. One Shared Tree Structure, Many Leaf-Value Tables

This is the major difference from the simplified ordered-boosting-lite version.

### Ordered-Boosting-Lite

The simplified version may train helper trees:

```text
prefix model for block 1
prefix model for block 2
prefix model for block 3
...
```

That is easy to understand but not how real CatBoost is organized.

### Actual CatBoost

At each boosting iteration, CatBoost builds:

```text
one tree structure
```

Then it updates many supporting prediction states using:

```text
same tree structure
different prefix-specific leaf values
```

So the tree structure is shared, but leaf values depend on which prefix model is being updated.

Conceptually:

```text
tree structure:
    split_1, split_2, split_3

leaf values for M_{r, 10}:
    values learned from first 10 rows of permutation r

leaf values for M_{r, 50}:
    values learned from first 50 rows of permutation r

leaf values for M_{r, 100}:
    values learned from first 100 rows of permutation r
```

Same structure. Different leaf values.

---

## 11. Ordered Split Scoring

In ordinary regression-tree split scoring, you might do:

```text
for each candidate split:
    compute left residual mean
    compute right residual mean
    calculate MSE improvement
```

Actual ordered boosting is stricter.

For a candidate split, each row's candidate leaf estimate should use only previous rows in the same leaf.

Suppose permutation order is:

```text
A, B, C, D, E
```

Candidate split creates leaves:

```text
left leaf:  A, C, E
right leaf: B, D
```

For row `E`:

```text
same leaf = left
previous rows in same leaf = A, C
candidate_delta_E = average gradients from A and C
```

For row `D`:

```text
same leaf = right
previous rows in same leaf = B
candidate_delta_D = average gradient from B
```

For row `A`:

```text
same leaf = left
previous rows in same leaf = none
candidate_delta_A = fallback / prior / ignored depending on implementation detail
```

Then CatBoost compares:

```text
candidate_delta_vector
```

against:

```text
ordered_gradient_vector
```

and chooses the split whose candidate deltas best match the ordered gradients.

Conceptual scoring:

```text
for each candidate split:

    for each object i:
        leaf_i = leaf of i under candidate tree

        candidate_delta_i =
            average ordered gradients of previous objects p such that:
                p appears before i in permutation
                p is in same leaf as i

    score = similarity(candidate_delta_vector, ordered_gradient_vector)

choose best candidate
```

The paper describes this as a cosine-similarity-style criterion.

---

## 12. Updating Supporting Predictions

After the tree structure is selected:

```text
tree = selected oblivious tree structure
```

CatBoost updates each supporting state.

For each permutation `r` and prefix `j`:

```text
allowed rows = first j rows of permutation r
```

For each leaf:

```text
leaf_value_{r,j,leaf} =
    average / Newton correction / gradient-based value
    computed only from allowed rows inside that leaf
```

Then update predictions:

```text
M_{r,j}(i) += learning_rate * leaf_value_{r,j,leaf(i)}
```

Again:

```text
same tree structure
different prefix-specific leaf values
```

---

## 13. Final Model vs Supporting Models

Supporting models are training machinery.

They are used to compute less biased gradients and evaluate splits.

They are not the final model used at prediction time.

The final model is the normal ensemble of trees:

```text
F(x) = base_prediction + sum(learning_rate * tree_t(x))
```

At prediction time for new data:

```text
1. compute categorical statistics using full training statistics
2. pass encoded features through final trees
3. sum tree outputs
```

No ordered prefix logic is needed during prediction.

---

## 14. Power-of-Two / Prefix Compression

The fully naive ordered setup would store too many states:

```text
M_{r,j}(i)
for every permutation r
for every prefix j
for every object i
```

That can become roughly:

```text
O(number_of_permutations * n^2)
```

CatBoost reduces this with compressed prefix storage.

Instead of storing every prefix:

```text
1, 2, 3, 4, 5, 6, 7, 8, ...
```

it stores selected prefix sizes, often conceptually like powers of two:

```text
1, 2, 4, 8, 16, 32, ...
```

This reduces memory and computation while keeping the ordered approximation useful.

---

## 15. Bayesian Bootstrap Weights

CatBoost also uses random weights as part of training regularization/sampling.

Conceptually:

```text
weighted_gradient_i = bootstrap_weight_i * gradient_i
```

These weights can affect:

```text
split scoring
leaf value calculation
```

You do not need this for the first implementation, but it is part of the actual CatBoost machinery.

---

## 16. Categorical Feature Combinations

CatBoost does not only encode raw categorical columns.

It can also create combinations of categorical features, such as:

```text
country
device
country + device
country + device + browser
```

Then it computes target statistics for those combinations.

This is one reason CatBoost is strong on tabular datasets with complex categorical interactions.

For an educational implementation, this should come much later.

---

## 17. Actual Ordered Boosting Pseudocode

This is high-level pseudocode for the real idea.

It intentionally avoids low-level optimization.

```text
fit_ordered_catboost(X, y):

    generate permutations:
        sigma_0, sigma_1, ..., sigma_s

    initialize final model F_final with base prediction

    initialize supporting predictions:
        M[r][prefix][i] = base prediction
        for each permutation r
        for selected prefix states
        for each object i

    for boosting_iteration in 1..T:

        choose permutation r for this iteration

        ordered_gradient = empty array

        for each object i:

            position = position of i in sigma_r
            prefix = position - 1

            prediction_i = supporting_prediction(M, r, prefix, i)

            ordered_gradient[i] =
                loss_gradient(y[i], prediction_i)

        tree_structure = empty oblivious tree

        for depth_level in 1..max_depth:

            best_split = None
            best_score = -infinity

            for candidate_split in candidate_splits:

                temporary_tree =
                    tree_structure plus candidate_split

                candidate_delta = empty array

                for each object i:

                    leaf_i = leaf_index(temporary_tree, X[i])

                    previous_same_leaf =
                        objects p such that:
                            position(p) < position(i)
                            leaf_index(temporary_tree, X[p]) == leaf_i

                    candidate_delta[i] =
                        average ordered_gradient[p]
                        over previous_same_leaf

                score =
                    similarity(candidate_delta, ordered_gradient)

                if score > best_score:
                    best_score = score
                    best_split = candidate_split

            add best_split to tree_structure

        for each permutation r_prime:
            for each stored prefix state j:

                allowed_rows =
                    first j rows of sigma_r_prime

                leaf_values =
                    calculate_leaf_values(
                        tree_structure,
                        allowed_rows,
                        ordered_gradient
                    )

                update M[r_prime][j] using:
                    tree_structure + leaf_values

        final_leaf_values =
            calculate_final_leaf_values(
                tree_structure,
                all training rows,
                gradients or Newton corrections
            )

        add tree_structure + final_leaf_values to F_final
```

---

## 18. What Makes This Different From Ordered-Boosting-Lite

### Ordered-Boosting-Lite

```text
- split data into blocks
- train helper/prefix trees
- compute ordered-ish gradients
- train normal final tree
```

### Actual CatBoost Ordered Boosting

```text
- uses multiple permutations
- uses object-level prefix logic
- maintains supporting prediction states
- builds one shared oblivious tree structure per iteration
- scores splits using previous-in-leaf ordered estimates
- updates many prefix states with different leaf values
- stores final model separately
```

The lite version is useful for learning.

The actual version is a much deeper algorithm.

---

## 19. Required Components To Implement Actual Ordered Boosting

To implement the actual version, the project needs these components:

```text
1. Oblivious tree builder
2. Multiple random permutations
3. Mapping from object index to permutation position
4. Ordered target statistics tied to the same permutation
5. Supporting prediction states M[r][prefix][i]
6. Prefix compression strategy
7. Ordered gradient computation
8. Ordered split scoring
9. Prefix-specific leaf value calculation
10. Final leaf value calculation
11. Final prediction path
12. Careful raw/encoded feature handling
13. Tests on tiny datasets
```

Do not implement all of these at once.

---

## 20. Recommended Implementation Ladder

### Stage 1: Oblivious Tree Only

Before ordered boosting, implement a basic oblivious regression tree.

Rules:

```text
at each depth:
    choose one best split globally
    apply it to all current leaves
```

Goal:

```text
train and predict with symmetric trees
```

---

### Stage 2: Ordered Split Scoring Toy Version

Use one permutation.

Ignore multiple prefix states.

Implement candidate scoring:

```text
candidate_delta_i =
    mean gradient of previous rows in same candidate leaf
```

Goal:

```text
understand previous-in-leaf scoring
```

---

### Stage 3: One Permutation, Full Prefix States

Maintain:

```text
M[j][i]
```

for selected prefixes.

Goal:

```text
compute gradients using prefix predictions
```

---

### Stage 4: Shared Tree Structure, Prefix Leaf Values

Build one tree structure.

For each prefix state, compute its own leaf values.

Goal:

```text
same tree structure
different prefix-specific leaf tables
```

---

### Stage 5: Multiple Permutations

Extend from one permutation to multiple permutations.

Goal:

```text
reduce variance and follow actual CatBoost design more closely
```

---

### Stage 6: Prefix Compression

Replace all prefixes with selected prefix states.

Example:

```text
1, 2, 4, 8, 16, 32, ...
```

Goal:

```text
make memory/computation less ridiculous
```

---

### Stage 7: Coupled Ordered Target Statistics

Make sure the ordered categorical encoding uses the same permutation as ordered boosting.

Goal:

```text
prevent leakage through categorical features
```

---

### Stage 8: Final Cleanup

Add:

```text
bootstrap weights
categorical feature combinations
Newton leaf values
validation support
early stopping
restore best
```

Only after the core algorithm is correct.

---

## 21. Debugging Strategy

Use tiny datasets.

Example:

```text
X:
    category, numeric_noise

y:
    category A mostly high
    category B mostly low
```

Start with:

```text
n_samples = 8
max_depth = 1
boosting_rounds = 1
one permutation
SSE loss only
```

Print everything:

```text
permutation
position of each row
prefix used for each row
ordered prediction for each row
ordered gradient for each row
candidate leaves
previous rows in same leaf
candidate delta
chosen split
leaf values
```

If this does not make sense for 8 rows, it will not magically work for 8,000 rows.

---

## 22. SSE Loss First

Use SSE first.

For SSE:

```text
loss = 1/2 * (y - F)^2

gradient = F - y
negative_gradient = y - F
leaf_value = mean(y - F) inside leaf
```

This is easy to inspect.

Binary cross entropy can come later:

```text
p = sigmoid(F)

gradient = p - y
negative_gradient = y - p

Newton leaf value =
    sum(y - p) / sum(p * (1 - p))
```

---

## 23. Important Terminology

### Ordered Target Statistics

Leakage-safe categorical encoding.

```text
category encoding for row i uses previous rows only
```

### Ordered Boosting

Prediction-shift-safe gradient estimation.

```text
gradient for row i uses a model trained on previous rows only
```

### Supporting Model

Training-only prefix model/state.

```text
M_{r,j}
```

### Final Model

The ensemble used at prediction time.

```text
F_final
```

### Oblivious Tree

Symmetric tree where the same split is used at each depth level.

### Prefix

The part of a permutation before a given row.

### Prefix-Specific Leaf Values

Different leaf values for the same tree structure depending on which prefix model is being updated.

---

## 24. Final Mental Model

The actual CatBoost Ordered mode is best understood as:

```text
Build each boosting tree while pretending that every training row is being predicted as a future unseen row relative to a random permutation.
```

That requires:

```text
- ordered categorical statistics
- ordered gradients
- ordered split scoring
- supporting prediction states
- shared tree structure
- prefix-specific leaf values
```

It is not a small patch to ordinary gradient boosting.

It is a different training protocol.

---

## 25. Practical Verdict

Your ordered-boosting-lite implementation is a good educational approximation.

The actual algorithm is much heavier because the tree builder itself must participate in the ordering logic.

If implementing this from scratch, do not try to jump directly from current gradient boosting to full CatBoost Ordered mode.

The safest path is:

```text
1. implement oblivious trees
2. implement previous-in-leaf split scoring
3. implement one permutation with prefix states
4. add shared tree structure with prefix leaf values
5. add multiple permutations
6. add compression and optimizations
```

Do not let the algorithm bully you into writing a giant untestable blob.

Small verified stages beat heroic spaghetti.
