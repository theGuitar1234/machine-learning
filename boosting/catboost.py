import copy
from dataclasses import dataclass
from enum import Enum

from sklearn import datasets
from sklearn.model_selection import train_test_split
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import os

from trees import ANonSeriousDecisionTree
from trees import ANonSeriousSymmetricalTree

import math

from trees.a_non_serious_path_element import ANonSeriousPathElement


class CatBoost:

    class LossType(Enum):
        BINARY_CROSS_ENTROPY = 1
        SSE = 2

    class BoostingType:
        PLAIN = 1
        ORDERED = 2

    @dataclass
    class TrainDefaults:
        epsilon = 1e-12
        smoothing_strength: int = 10
        ordered_boosting_blocks: int = 10
        max_ctr_complexity: int = 2
        min_combo_count: int = 5
        max_number_of_combos: int = 20
        one_hot_max_size: int = 2

    def __init__(
        self,
        seed=42,
        boosting_rounds=100,
        learning_rate=0.03,
        max_depth=10,
        minimum_population_size=2,
        minimum_gain=0.001,
        adjacent=False,
        random_criterion=None,
        loss_type=None,
        sub_sample=0.8,
        column_sub_sample=1,
        early_stopping=False,
        restore_best=False,
        validation=False,
        symmetrical=False,
        tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
        information_gain=ANonSeriousDecisionTree.InformationGain.GINI,
        boosting_type=BoostingType.PLAIN,
        encoding_target=None,
        categorical_features=None,
        categorical_combinations=None,
        config=TrainDefaults(),
    ):
        if not (0 < sub_sample <= 1) or not (0 < column_sub_sample <= 1):
            raise ValueError(
                "sub_sample and column_sub_sample must be a positive fraction less than 1"
            )
        if restore_best and not validation:
            raise ValueError("restore_best=True requires validation=True")
        self.sub_sample = sub_sample
        self.column_sub_sample = column_sub_sample
        self.boosting_rounds = boosting_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.minimum_population_size = minimum_population_size
        self.minimum_gain = minimum_gain
        self.adjacent = adjacent
        self.information_gain = information_gain
        self.random_criterion = random_criterion
        self.tree_type = tree_type
        self.loss_type = loss_type
        self.early_stopping = early_stopping
        self.restore_best = restore_best
        self.validation = validation
        self.categorical_features = categorical_features or []
        self.categorical_combinations = categorical_combinations or []
        self.one_hot_features = []
        self.ctr_features = []
        self.one_hot_categories = {}
        self.ctr_statistics = {}
        self.combo_ctr_statistics = {}
        self.coalition_values = {}
        self.encoding_target = encoding_target
        self.boosting_type = boosting_type
        self.symmetrical = symmetrical
        self.rng = np.random.default_rng(seed=seed)
        self.config = config
        self.F_x = None
        self.F_0x = None

    def fit(self, X, y, X_val, y_val):
        self.X_train_ = X
        self.y_train_ = y

        self.trees = []
        self.feature_indices = []
        self.encoding_statistics = []

        X_model = self._feature_transformer(X, y)
        self.X_train_ = X_model
        X = X_model

        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        sample_size = max(1, int(self.sub_sample * number_of_samples))
        number_of_selected_features = max(
            1, int(self.column_sub_sample * number_of_features)
        )

        epsilon = self.config.epsilon
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p0 = np.clip(np.mean(y), epsilon, 1 - epsilon)
                self.F_0x = np.log(p0 / (1 - p0))
            case self.LossType.SSE:
                self.F_0x = np.mean(y)
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )
        self.F_x = np.repeat(self.F_0x, y.shape[0])

        best_val_loss = float("inf")
        patience = 100
        best_round = -1
        best_number_of_trees = 0
        patience_counter = 0
        for round in range(self.boosting_rounds):
            pseudo_residual = -self.dloss(y, self.F_x)

            if self.symmetrical:
                tree = ANonSeriousSymmetricalTree(
                    max_depth=self.max_depth,
                    minimum_split_size=self.minimum_population_size,
                    minimum_gain=self.minimum_gain,
                )
            else:
                tree = ANonSeriousDecisionTree(
                    max_depth=self.max_depth,
                    minimum_population_size=self.minimum_population_size,
                    minimum_gain=self.minimum_gain,
                    categorical=False,
                    adjacent=self.adjacent,
                    information_gain=self.information_gain,
                    random_criterion=self.random_criterion,
                    tree_type=self.tree_type,
                    catboost=True,
                )
            tree.smoothing_strength = self.config.smoothing_strength

            indices = np.random.choice(
                number_of_samples, size=sample_size, replace=False
            )
            feature_indices = np.random.choice(
                number_of_features, size=number_of_selected_features, replace=False
            )

            X_sub = X[indices][:, feature_indices]
            pseudo_residual_sub = pseudo_residual[indices]
            y_sub_original = y[indices]

            # Deprecated - obselete after feature transformer
            local_categorical_features = self._local_categorical_features(
                feature_indices
            )
            tree.categorical_features = local_categorical_features

            if self.symmetrical:
                tree.fit(X_sub, pseudo_residual_sub)
                tree.leaf_correction(
                    y=y_sub_original,
                    F=self.F_x[indices],
                    correction_function=self._correct,
                )
            else:
                tree.fit(X_sub, pseudo_residual_sub, encoding_target=y_sub_original)
                self.leaf_correction(tree, tree.X_train_, y[indices], self.F_x[indices])

            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
            self.F_x += self.learning_rate * tree.predict(X[:, feature_indices])

            if self.validation:
                _, val_loss = self.evaluate_dataset(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_round = round
                    patience_counter = 0
                    best_number_of_trees = len(self.trees)
                else:
                    patience_counter += 1
                if self.early_stopping and patience_counter >= patience:
                    print(
                        "Overfitting detected. Early stopping at round: ",
                        round,
                        "Best Round : ",
                        best_round,
                    )
                    break
        if self.restore_best and self.validation:
            self.trees = self.trees[:best_number_of_trees]
            self.feature_indices = self.feature_indices[:best_number_of_trees]
            self.F_x = np.repeat(self.F_0x, X.shape[0])
            for tree, feature_indices in zip(self.trees, self.feature_indices):
                self.F_x += self.learning_rate * tree.predict(X[:, feature_indices])
        return self

    def fit_ordered(self, X, y, X_val=None, y_val=None):
        self.X_train_ = X
        self.y_train_ = y

        self.trees = []
        self.feature_indices = []

        X_model = self._feature_transformer(X, y)
        self.X_train_ = X_model
        X = X_model

        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        sample_size = max(1, int(self.sub_sample * number_of_samples))
        number_of_selected_features = max(
            1, int(self.column_sub_sample * number_of_features)
        )

        epsilon = self.config.epsilon
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p0 = np.clip(np.mean(y), epsilon, 1 - epsilon)
                self.F_0x = np.log(p0 / (1 - p0))
            case self.LossType.SSE:
                self.F_0x = np.mean(y)
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )
        self.F_x = np.repeat(self.F_0x, y.shape[0])

        block_size = self.config.ordered_boosting_blocks
        permutation = self.rng.permutation(number_of_samples)
        ordered_blocks = [
            permutation[i : i + block_size]
            for i in range(0, len(permutation), block_size)
        ]
        number_of_blocks = len(ordered_blocks)
        self.F_prefix = [
            np.repeat(self.F_0x, number_of_samples).astype(float)
            for _ in range(number_of_blocks)
        ]

        best_val_loss = float("inf")
        patience = 100
        best_round = -1
        best_number_of_trees = 0
        patience_counter = 0
        for round in range(self.boosting_rounds):
            ordered_prediction = np.empty(number_of_samples, dtype=float)
            for b, rows_b in enumerate(ordered_blocks):
                ordered_prediction[rows_b] = self.F_prefix[b][rows_b]
            ordered_residual = -self.dloss(y, ordered_prediction)

            if self.symmetrical:
                tree = ANonSeriousSymmetricalTree(
                    max_depth=self.max_depth,
                    minimum_split_size=self.minimum_population_size,
                    minimum_gain=self.minimum_gain,
                )
            else:
                tree = ANonSeriousDecisionTree(
                    max_depth=self.max_depth,
                    minimum_population_size=self.minimum_population_size,
                    minimum_gain=self.minimum_gain,
                    categorical=False,
                    adjacent=self.adjacent,
                    information_gain=self.information_gain,
                    random_criterion=self.random_criterion,
                    tree_type=self.tree_type,
                    catboost=True,
                )
            tree.smoothing_strength = self.config.smoothing_strength

            indices = np.random.choice(
                number_of_samples, size=sample_size, replace=False
            )
            feature_indices = np.random.choice(
                number_of_features, size=number_of_selected_features, replace=False
            )

            X_sub = X[indices][:, feature_indices]
            ordered_residual_sub = ordered_residual[indices]
            y_sub_original = y[indices]

            # Deprecated - obselete after feature transformer
            local_categorical_features = self._local_categorical_features(
                feature_indices
            )
            tree.categorical_features = local_categorical_features

            if self.symmetrical:
                tree.fit(X_sub, ordered_residual_sub)
                tree.leaf_correction(
                    y=y_sub_original,
                    F=ordered_prediction[indices],
                    correction_function=self._correct,
                )
            else:
                tree.fit(X_sub, ordered_residual_sub, encoding_target=y_sub_original)
                self.leaf_correction(
                    tree, tree.X_train_, y[indices], ordered_prediction[indices]
                )
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
            self.F_x += self.learning_rate * tree.predict(X[:, feature_indices])

            for b in range(number_of_blocks):
                prefix_rows = np.concatenate(ordered_blocks[: b + 1])
                auxiliary_tree = ANonSeriousDecisionTree(
                    max_depth=self.max_depth,
                    minimum_population_size=self.minimum_population_size,
                    minimum_gain=self.minimum_gain,
                    categorical=False,
                    adjacent=self.adjacent,
                    information_gain=self.information_gain,
                    random_criterion=self.random_criterion,
                    tree_type=self.tree_type,
                    catboost=True,
                )
                auxiliary_tree.smoothing_strength = self.config.smoothing_strength
                auxiliary_tree.categorical_features = local_categorical_features

                X_prefix = X[prefix_rows][:, feature_indices]
                y_prefix_target = ordered_residual[prefix_rows]
                y_prefix_original = y[prefix_rows]

                auxiliary_tree.fit(
                    X_prefix,
                    y_prefix_target,
                    encoding_target=y_prefix_original,
                )

                self.leaf_correction(
                    auxiliary_tree,
                    auxiliary_tree.X_train_,
                    y_prefix_original,
                    ordered_prediction[prefix_rows],
                )
                self.F_prefix[b] += self.learning_rate * auxiliary_tree.predict(
                    X[:, feature_indices]
                )

            if self.validation:
                _, val_loss = self.evaluate_dataset(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_round = round
                    patience_counter = 0
                    best_number_of_trees = len(self.trees)
                else:
                    patience_counter += 1
                if self.early_stopping and patience_counter >= patience:
                    print(
                        "Overfitting detected. Early stopping at round: ",
                        round,
                        "Best Round : ",
                        best_round,
                    )
                    break
        if self.restore_best and self.validation:
            self.trees = self.trees[:best_number_of_trees]
            self.feature_indices = self.feature_indices[:best_number_of_trees]
            self.F_x = np.repeat(self.F_0x, X.shape[0])
            for tree, feature_indices in zip(self.trees, self.feature_indices):
                self.F_x += self.learning_rate * tree.predict(X[:, feature_indices])
        return self

    def _local_categorical_features(self, feature_indices):
        local_categorical_features = []
        for local_position, original_feature in enumerate(feature_indices):
            if original_feature in self.categorical_features:
                local_categorical_features.append(local_position)
        return local_categorical_features

    def _feature_transformer(self, X, y):
        self.numeric_features = []

        for feature in range(X.shape[1]):
            if feature not in self.categorical_features:
                self.numeric_features.append(feature)
        self.preprocess_categorical_features(X)
        self._one_hot_encode(X)
        self.preprocess_categorical_combinations()
        return self._transform_features(X, y, training_mode=True)

    def _transform_features(self, X, y=None, training_mode=False):
        columns = []

        for numeric_feature in self.numeric_features:
            columns.append(X[:, numeric_feature].astype(float))
        for one_hot_column in self._transform_one_hot(X):
            columns.append(one_hot_column)
        if training_mode:
            self.ctr_statistics = {}
            self.combo_ctr_statistics = {}
            for ctr_feature in self.ctr_features:
                keys = X[:, ctr_feature]
                encoded_col, ctr_stats = self._fit_ordered_ctr(keys, y)
                self.ctr_statistics[ctr_feature] = ctr_stats
                columns.append(encoded_col)
            for combo in self.categorical_combinations:
                keys = self._generate_combo_keys(X, combo)
                encoded_col, combo_stats = self._fit_ordered_ctr(keys, y)
                self.combo_ctr_statistics[combo] = combo_stats
                columns.append(encoded_col)
        else:
            for ctr_feature in self.ctr_features:
                keys = X[:, ctr_feature]
                stats = self.ctr_statistics[ctr_feature]
                encoded_col = self._transform_ctr(keys, stats)
                columns.append(encoded_col)
            for combo in self.categorical_combinations:
                keys = self._generate_combo_keys(X, combo)
                stats = self.combo_ctr_statistics[combo]
                encoded_col = self._transform_ctr(keys, stats)
                columns.append(encoded_col)
        return np.column_stack(columns).astype(float)

    def _fit_ordered_ctr(self, keys, y):
        keys = np.asarray(keys, dtype=object)
        encoded = np.empty(len(keys), dtype=float)

        counts = {}
        sums = {}
        prior = float(np.mean(y))
        prior_weight = self.config.smoothing_strength

        permutation = self.rng.permutation(len(keys))

        for row in permutation:
            key = keys[row]
            previous_count = counts.get(key, 0)
            previous_sum = sums.get(key, 0.0)

            encoded[row] = (previous_sum + prior_weight * prior) / (
                previous_count + prior_weight
            )
            counts[key] = previous_count + 1
            sums[key] = previous_sum + y[row]
        stats = {
            "counts": counts,
            "sums": sums,
            "prior": prior,
            "prior_weight": prior_weight,
        }
        return encoded, stats

    def _transform_ctr(self, keys, stats):
        keys = np.asarray(keys, dtype=object)
        output = np.empty(len(keys), dtype=float)

        counts = stats["counts"]
        sums = stats["sums"]
        prior = stats["prior"]
        prior_weight = stats["prior_weight"]

        for row in range(len(keys)):
            key = keys[row]

            previous_count = counts.get(key, 0)
            previous_sum = sums.get(key, 0.0)

            output[row] = (previous_sum + prior_weight * prior) / (
                previous_count + prior_weight
            )
        return output

    def _generate_combo_keys(self, X, combo):
        return np.array(
            [tuple(row) for row in X[:, combo]],
            dtype=object,
        )

    def preprocess_categorical_combinations(self):
        self.categorical_combinations = []
        all_combos = []

        max_ctr_complexity = self.config.max_ctr_complexity
        for size in range(2, max_ctr_complexity + 1):
            all_combos.extend(combinations(self.ctr_features, size))
        if len(all_combos) > self.config.max_number_of_combos:
            chosen_indices = self.rng.choice(
                len(all_combos),
                size=self.config.max_number_of_combos,
                replace=False,
            )
            all_combos = [all_combos[i] for i in chosen_indices]
        self.categorical_combinations = all_combos

    def preprocess_categorical_features(self, X):
        self.one_hot_features = []
        self.ctr_features = []
        for feature in self.categorical_features:
            unique_count = len(np.unique(X[:, feature]))
            if unique_count <= self.config.one_hot_max_size:
                self.one_hot_features.append(feature)
            else:
                self.ctr_features.append(feature)

    def _one_hot_encode(self, X):
        self.one_hot_categories = {}
        for feature in self.one_hot_features:
            categories = np.unique(X[:, feature])
            self.one_hot_categories[feature] = categories

    def _transform_one_hot(self, X):
        columns = []
        for feature in self.one_hot_features:
            categories = self.one_hot_categories[feature]
            for category in categories:
                column = X[:, feature] == category
                columns.append(column.astype(float))
        return columns

    def dloss(self, y, F_x):
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p_x = self.sigmoid(F_x)
                return p_x - y
            case self.LossType.SSE:
                return F_x - y
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )

    def sse(self, y, F_x):
        return 1 / 2 * (y - F_x) ** 2

    def sse_mean(self, y, F_x):
        return np.mean(self.sse(y, F_x))

    def sigmoid(self, z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)

        pos = z >= 0
        neg = ~pos

        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)

        return out

    def binary_cross_entropy_loss(self, y, F_x, epsilon=1e-12):
        p_x = self.sigmoid(F_x)
        m = y.shape[0]
        p_x = np.clip(p_x, epsilon, 1.0 - epsilon)
        return -np.sum(y * np.log(p_x) + (1 - y) * np.log(1 - p_x)) / m

    def leaf_correction(self, tree, X, y, F):
        leaf_to_indices = {}
        for i, x in enumerate(X):
            leaf = tree.predict_one(x, leaf_node=True)
            leaf_id = id(leaf)
            if leaf_id not in leaf_to_indices:
                leaf_to_indices[leaf_id] = {"leaf": leaf, "indices": []}
            leaf_to_indices[leaf_id]["indices"].append(i)
        for item in leaf_to_indices.values():
            leaf = item["leaf"]
            index = np.array(item["indices"])
            y_leaf = y[index]
            F_leaf = F[index]
            leaf.value = self._correct(y_leaf, F_leaf)

    def _correct(self, y, F, epsilon=1e-12):
        match self.loss_type:
            case self.LossType.SSE:
                return np.mean(y - F)
            case self.LossType.BINARY_CROSS_ENTROPY:
                p = self.sigmoid(F)
                return np.sum(y - p) / (np.sum(p * (1 - p)) + epsilon)
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )

    def evaluate_dataset(self, X, y):
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        F_x = self.predict(X)
        predictions = np.asarray(F_x)
        loss = None
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                loss = self.binary_cross_entropy_loss(y, F_x)
            case self.LossType.SSE:
                loss = self.sse_mean(y, F_x)
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )
        return predictions, loss

    def predict(self, X):
        if self.F_0x is None:
            raise RuntimeError("Model is not fit")
        X_model = self._transform_features(X, training_mode=False)
        prediction = np.repeat(self.F_0x, X.shape[0])
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            prediction += self.learning_rate * tree.predict(X_model[:, feature_indices])
        return prediction

    def agnostic_shapley(
        self,
        X,
        background_X,
        verbose=False,
        # optimized=False
    ):
        if self.F_0x is None:
            raise RuntimeError("Model is not fit.")
        M = X.shape[1]
        # features = np.arange(M)

        self.compute_coalition_values(X, background_X, M)
        v_empty = self.coalition_values[0]
        # v_empty = self._condition_value(background_X)
        phi_0 = v_empty

        phi = np.zeros(M, dtype=float)
        # for j in features:
        for j in range(M):
            shapley_value = self._shapley_value(M, j)
            # shapley_value = self._shapley_value(X, S, features, j)
            phi[j] = shapley_value
        prediction = self.predict(X)
        reconstructed = phi_0 + np.sum(phi)
        additivity_error = abs(prediction - reconstructed)

        if verbose:
            print("\nShapley Additive Explanations: ")
            print(f"""
    phi_0:            {phi_0},
    phi:              {phi},
    prediction:       {prediction},
    reconstructed:    {reconstructed},
    additivity_error: {additivity_error}
    """)

        return {
            "phi_0": phi_0,
            "phi": phi,
            "prediction": prediction,
            "reconstructed": reconstructed,
            "additivity_error": additivity_error,
        }

    def sampling_shapley(
        self,
        X,
        background_X,
        number_of_permutations=100,
        verbose=False,
    ):
        if self.F_0x is None:
            raise RuntimeError("Model is not fit.")
        M = X.shape[1]
        all_mask = (1 << M) - 1
        cache = {}

        def v(mask):
            if mask not in cache:
                cache[mask] = self._condition_value(X, background_X, mask)
            return cache[mask]

        samples = np.zeros((number_of_permutations, M), dtype=float)
        phi_0 = v(0)

        for i in range(number_of_permutations):
            order = self.rng.permutation(M)
            current_mask = 0
            previous_value = v(current_mask)
            for j in order:
                new_mask = current_mask | (1 << j)
                new_value = v(new_mask)
                marginal = new_value - previous_value
                samples[i, j] = marginal
                current_mask = new_mask
                previous_value = new_value
        phi = np.mean(samples, axis=0)
        phi_std = np.std(samples, axis=0)

        prediction = v(all_mask)
        reconstructed = phi_0 + np.sum(phi)
        additivity_error = abs(prediction - reconstructed)

        if verbose:
            print("\nSampling SHAP Explanations:")
            print(f"""
    phi_0:                  {phi_0}
    phi:                    {phi}
    phi_std:                {phi_std}
    prediction:             {prediction}
    reconstructed:          {reconstructed}
    additivity_error:       {additivity_error}
    number_of_permutations: {number_of_permutations}
    cached_coalitions:      {len(cache)}
    background_size:        {background_X.shape[0]}
    """)
        return {
            "phi_0": phi_0,
            "phi": phi,
            "phi_std": phi_std,
            "prediction": prediction,
            "reconstructed": reconstructed,
            "additivity_error": additivity_error,
            "number_of_permutations": number_of_permutations,
            "background_size": background_X.shape[0],
            "cached_coalitions": len(cache),
        }

    def kernel_shapley(
        self,
        X,
        background_X,
        number_of_samples=100,
        large_weight=1e6,
        additivity_correction=False,
        verbose=False,
    ):
        if self.F_0x is None:
            raise RuntimeError("Model is not fit.")
        M = X.shape[1]
        all_mask = (1 << M) - 1
        cache = {}

        def v(mask):
            if mask not in cache:
                cache[mask] = self._condition_value(X, background_X, mask)
            return cache[mask]

        baseline = v(0)
        prediction = v(all_mask)

        masks = []
        targets = []
        weights = []

        masks.append(0)
        targets.append(0.0)
        weights.append(large_weight)

        masks.append(all_mask)
        targets.append(prediction - baseline)
        weights.append(large_weight)

        for _ in range(number_of_samples):
            mask = self._sample_kernel_mask(M)
            k = mask.bit_count()

            value = v(mask)
            adjusted_target = value - baseline
            weight = self._shap_kernel_weight(M, k)

            masks.append(mask)
            targets.append(adjusted_target)
            weights.append(weight)
        Z = np.vstack([self._mask_to_binary_vector(mask, M) for mask in masks])

        y = np.asarray(targets, dtype=float)
        w = np.asarray(weights, dtype=float)
        phi = self._weighted_linear_regression(Z, y, w)

        reconstructed = baseline + np.sum(phi)
        additivity_error = abs(prediction - reconstructed)

        if additivity_correction:
            phi = self._apply_additivity_correction(
                phi=phi,
                baseline=baseline,
                prediction=prediction,
            )
            reconstructed = baseline + np.sum(phi)
            additivity_error = abs(prediction - reconstructed)

        if verbose:
            print("\nKernel SHAP Explanations:")
            print(f"""
    phi_0:              {baseline}
    phi:                {phi}
    prediction:         {prediction}
    reconstructed:      {reconstructed}
    additivity_error:   {additivity_error}
    number_of_samples:  {number_of_samples}
    cached_coalitions:  {len(cache)}
    background_size:    {background_X.shape[0]}
        """)

        return {
            "phi_0": baseline,
            "phi": phi,
            "prediction": prediction,
            "reconstructed": reconstructed,
            "additivity_error": additivity_error,
            "number_of_samples": number_of_samples,
            "background_size": background_X.shape[0],
            "cached_coalitions": len(cache),
            "method": "kernel_shap",
        }

    def tree_shap_single_tree(
        self,
        tree,
        x_local,
        number_of_local_features,
    ):
        phi = np.zeros(number_of_local_features)
        path = []
        self._tree_shapley(
            node=tree.root,
            x_local=x_local,
            path=path,
            parent_zero_fraction=1,
            parent_one_fraction=1,
            parent_feature_index=-1,
            phi=phi,
        )
        expected_value = self._expected_tree_value(tree.root)
        tree_prediction = tree.predict_one(x_local)
        reconstructed = expected_value + np.sum(phi)
        additivity_error = abs(tree_prediction - reconstructed)

        return {
            "tree_expected": expected_value,
            "tree_phi": phi,
            "tree_prediction": tree_prediction,
            "reconstructed": reconstructed,
            "additivity_error": additivity_error,
        }

    def tree_shap_ensemble(self, raw_row):
        transformed_row = self._transform_features(raw_row, training_mode=False)
        number_of_transformed_features = transformed_row.shape[1]
        total_phi = np.zeros(number_of_transformed_features)
        base_value = self.F_0x

        for tree_index in range(self.boosting_rounds):
            tree = self.trees[tree_index]
            feature_indices = self.feature_indices[tree_index]
            x_local = transformed_row[0, feature_indices]

            tree_result = self.tree_shap_single_tree(
                tree=tree,
                x_local=x_local,
                number_of_local_features=len(feature_indices),
            )

            base_value += self.learning_rate * tree_result["tree_expected"]
            for local_feature_position in range(len(feature_indices)):
                global_transformed_feature = feature_indices[local_feature_position]
                total_phi[global_transformed_feature] += (
                    self.learning_rate * tree_result["tree_phi"][local_feature_position]
                )
        prediction = self.predict(raw_row)
        reconstructed = base_value + np.sum(total_phi)
        additivity_error = abs(prediction - reconstructed)

        return {
            "base_value": base_value,
            "total_phi": total_phi,
            "prediction": prediction,
            "reconstructed": reconstructed,
            "additivity_error": additivity_error,
        }

    def _tree_shapley(
        self,
        node,
        x_local,
        path,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
        phi,
    ):
        path = copy.deepcopy(path)
        self._extend_path(
            path,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        )
        path_length = len(path)
        if node.is_leaf:
            for path_index in range(1, path_length):
                path_element = path[path_index]
                feature = path_element.feature_index
                weight = self._unwound_path_sum(path, path_index)
                contribution = (
                    weight
                    * (path_element.one_fraction - path_element.zero_fraction)
                    * node.value
                )
                phi[feature] += contribution
            return
        split_feature = node.feature
        hot_child, cold_child = self._get_hot_and_cold_children(node, x_local)
        node_count = node.number_of_samples

        hot_zero_fraction = hot_child.number_of_samples / node_count
        cold_zero_fraction = cold_child.number_of_samples / node_count

        incoming_zero_fraction = 1
        incoming_one_fraction = 1

        previous_path_index = None
        for index in range(path_length):
            if path[index].feature_index == split_feature:
                previous_path_index = index
                break
        if previous_path_index is not None:
            incoming_zero_fraction = path[previous_path_index].zero_fraction
            incoming_one_fraction = path[previous_path_index].one_fraction
            path = self._unwind_path(path, previous_path_index)
        self._tree_shapley(
            node=hot_child,
            x_local=x_local,
            path=path,
            parent_zero_fraction=hot_zero_fraction * incoming_zero_fraction,
            parent_one_fraction=incoming_one_fraction,
            parent_feature_index=split_feature,
            phi=phi,
        )
        self._tree_shapley(
            node=cold_child,
            x_local=x_local,
            path=path,
            parent_zero_fraction=cold_zero_fraction * incoming_zero_fraction,
            parent_one_fraction=0,
            parent_feature_index=split_feature,
            phi=phi,
        )

    def _shapley_value(self, M, j):
        phi_j = 0.0
        for mask in range(1 << M):  # math.exp(2, M)
            if mask & (1 << j):  # j in mask
                continue
            S_size = mask.bit_count()  # len(mask_with_j == 1)
            mask_with_j = mask | (1 << j)  # S.append(j)
            weight = (
                math.factorial(S_size) * math.factorial(M - S_size - 1)
            ) / math.factorial(M)
            marginal = self.coalition_values[mask_with_j] - self.coalition_values[mask]
            phi_j += weight * marginal
        return phi_j

    def _condition_value(self, X_masked, background_X, mask):
        # if S is not None:
        #     for feature in S:
        #         X_masked[:, feature] = X[feature]
        X_masked = copy.deepcopy(background_X)
        M = X_masked.shape[1]
        selected_features = [feature for feature in range(M) if mask & (1 << feature)]
        if selected_features:
            X_masked[:, selected_features] = X[0, selected_features]
        predictions = self.predict(X_masked)
        return np.mean(predictions)

    def compute_coalition_values(self, X, background_X, M):
        self.coalition_values = {}
        # for mask in range(0, math.exp(2, M)):
        #     self.coalition_values[mask] = self._condition_value(X, background_X, mask)
        for mask in range(1 << M):
            self.coalition_values[mask] = self._condition_value(X, background_X, mask)

    def _sample_kernel_mask(self, M):
        all_mask = (1 << M) - 1
        while True:
            mask = int(self.rng.integers(1, all_mask))
            if mask != 0 and mask != all_mask:
                return mask

    def _shap_kernel_weight(self, M, k):
        if k == 0 or k == M:
            return 1e6
        return (M - 1) / (math.comb(M, k) * k * (M - k))

    def _mask_to_binary_vector(self, mask, M):
        return np.array(
            [1.0 if mask & (1 << feature) else 0.0 for feature in range(M)],
            dtype=float,
        )

    def _weighted_linear_regression(self, Z, y, weights):
        sqrt_w = np.sqrt(weights)

        Z_weighted = Z * sqrt_w[:, None]
        y_weighted = y * sqrt_w

        phi, *_ = np.linalg.lstsq(
            Z_weighted,
            y_weighted,
            rcond=None,
        )
        return phi

    def _apply_additivity_correction(self, phi, baseline, prediction):
        reconstructed = baseline + np.sum(phi)
        difference = prediction - reconstructed
        total_abs = np.sum(np.abs(phi))
        if total_abs > 0:
            return phi + difference * np.abs(phi) / total_abs
        return phi + difference / len(phi)

    def _expected_tree_value(self, node):
        if node.is_leaf:
            return node.value
        left_count = node.left.number_of_samples
        right_count = node.right.number_of_samples
        total_count = left_count + right_count

        if total_count == 0:
            return 0
        left_weight = left_count / total_count
        right_weight = right_count / total_count

        return left_weight * self._expected_tree_value(
            node.left
        ) + right_weight * self._expected_tree_value(node.right)

    def _extend_path(self, path, zero_fraction, one_fraction, feature_index):
        depth = len(path)
        pathElement = ANonSeriousPathElement(
            feature_index=feature_index,
            zero_fraction=zero_fraction,
            one_fraction=one_fraction,
            path_weight=0,
        )
        path.append(pathElement)
        if depth == 0:
            path[0].path_weight = 1
            return path
        for i in range(depth - 1, -1, -1):
            path[i + 1].path_weight += (
                one_fraction * path[i].path_weight * (i + 1) / (depth + 1)
            )
            path[i].path_weight = (
                zero_fraction * path[i].path_weight * (depth - i) / (depth + 1)
            )
        return path

    def _unwind_path(self, path, path_index):
        depth = len(path) - 1
        removed_one = path[path_index].one_fraction
        removed_zero = path[path_index].zero_fraction
        next_one_portion = path[depth].path_weight
        for i in range(depth - 1, -1, -1):
            if removed_one != 0:
                old_weight = path[i].path_weight
                path[i].path_weight = (
                    next_one_portion * (depth + 1) / ((i + 1) * removed_one)
                )
                next_one_portion = old_weight - path[i].path_weight * removed_zero * (
                    depth - i
                ) / (depth + 1)
            else:
                path[i].path_weight = (
                    path[i].path_weight * (depth + 1) / (removed_zero * (depth - i))
                )
        for i in range(path_index, depth):
            path[i] = path[i + 1]
        return path[:depth]

    def _unwound_path_sum(self, path, path_index):
        depth = len(path) - 1
        one = path[path_index].one_fraction
        zero = path[path_index].zero_fraction
        next_one_portion = path[depth].path_weight
        total = 0.0

        if one != 0:
            for i in range(depth - 1, -1, -1):
                temp = next_one_portion * (depth + 1) / ((i + 1) * one)
                total += temp
                next_one_portion = path[i].path_weight - temp * zero * (depth - i) / (
                    depth + 1
                )
        else:
            for i in range(depth - 1, -1, -1):
                total += path[i].path_weight * (depth + 1) / (zero * (depth - i))
        return total

    def _get_hot_and_cold_children(self, node, x_local):
        feature = node.feature
        value = x_local[feature]

        if value > node.threshold:
            hot_child = node.left
            cold_child = node.right
        else:
            hot_child = node.right
            cold_child = node.left
        return hot_child, cold_child

    def tree_contributions(self, row, top_k=10, verbose=False):
        if self.F_0x is None:
            raise RuntimeError("Model is not fit.")
        row = np.asarray(row)
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if row.ndim != 2:
            raise ValueError("row must be a 1D row or a 2D array with one row.")
        if row.shape[0] != 1:
            raise ValueError("tree_contributions explains one row at a time.")
        transformed_row = self._transform_features(row, training_mode=False)
        baseline = self.F_0x
        running_prediction = baseline
        tree_explanations = []
        number_of_trees = self.boosting_rounds
        for tree_index in range(number_of_trees):
            tree = self.trees[tree_index]
            feature_indices = self.feature_indices[tree_index]
            tree_rows = transformed_row[:, feature_indices]
            raw_tree_output = tree.predict(tree_rows)
            contribution = self.learning_rate * raw_tree_output
            running_prediction = running_prediction + contribution
            record = {
                "tree_index": tree_index,
                "raw_tree_output": raw_tree_output,
                "contribution": contribution,
                "absolute_contribution": abs(contribution),
                "feature_indices": feature_indices,
                "prediction": running_prediction,
            }
            tree_explanations.append(record)
        sorted_tree_explanations = sorted(
            tree_explanations,
            key=lambda record: record["absolute_contribution"],
            reverse=True,
        )
        top_tree_explanations = sorted_tree_explanations[:top_k]

        if verbose:
            print(f"""
    Baseline: {baseline},
    Final Raw Prediction: {running_prediction},
    """)
            for record in top_tree_explanations:
                print(f"""
    tree: {record["tree_index"]},
        raw_tree_output: {record["raw_tree_output"]},
        contribution: {record["contribution"]},
        Prediction After Tree: {record["prediction"]}
    """)
        return {
            "baseline": baseline,
            "final_raw_prediction": running_prediction,
            "all_tree_explanations": tree_explanations,
            "top_tree_explanations": top_tree_explanations,
        }

    def permutation_importance(
        self,
        X,
        y,
        feature_names=None,
        n_repeats=50,
        verbose=False,
    ):
        _, baseline_loss = self.evaluate_dataset(X, y)
        rng = self.rng

        permutation_importance_averages = {}
        permutation_importance_standart_deviations = {}

        for feature in range(X.shape[1]):
            drop_losses = []

            for _ in range(n_repeats):
                X_shuffled = X.copy()
                rng.shuffle(X_shuffled[:, feature])
                _, drop_loss = self.evaluate_dataset(X_shuffled, y)
                drop_losses.append(baseline_loss - drop_loss)
            feature_name = (
                feature_names[feature]
                if feature_names is not None
                else f"feature{feature}"
            )
            permutation_importance_averages[f"{feature_name}_avg"] = np.average(
                drop_losses
            )
            permutation_importance_standart_deviations[f"{feature_name}_std"] = np.std(
                drop_losses
            )
        if verbose:
            features_avg = list(permutation_importance_averages.keys())
            features_std = [feature.replace("_avg", "_std") for feature in features_avg]
            feature_labels = [feature.replace("_avg", "") for feature in features_avg]
            avg_values = np.array(
                [
                    float(
                        np.asarray(
                            permutation_importance_averages[feature], dtype=float
                        ).mean()
                    )
                    for feature in features_avg
                ],
                dtype=float,
            )
            std_values = np.array(
                [
                    float(
                        np.asarray(
                            permutation_importance_standart_deviations[feature],
                            dtype=float,
                        ).mean()
                    )
                    for feature in features_std
                ],
                dtype=float,
            )
            total_avg = np.sum(avg_values)
            total_std = np.sum(std_values)
            values_avg = (
                (avg_values / total_avg) * 100
                if total_avg != 0
                else np.zeros_like(avg_values)
            )
            values_std = (
                (std_values / total_std) * 100
                if total_std != 0
                else np.zeros_like(std_values)
            )
            self.verbose_permutation_importance(
                permutation_importance_averages,
                permutation_importance_standart_deviations,
                std_values,
                values_std,
                avg_values,
                values_avg,
            )
            self.visualize_permutation_importance(
                feature_labels,
                values_avg,
                values_std,
            )
        return (
            permutation_importance_averages,
            permutation_importance_standart_deviations,
        )

    def ablation_importances(self, X_train, y_train, X_val, y_val):
        baseline_model = copy.deepcopy(self)
        baseline_model.fit(X_train, y_train, X_val, y_val)
        _, baseline_loss = baseline_model.evaluate_dataset(X_val, y_val)

        results = []

        for feature in range(X.shape[1]):
            X_train_removed = np.delete(X_train, feature, axis=1)
            X_val_removed = np.delete(X_val, feature, axis=1)

            adjusted_categorical_features = self._adjust_categorical_features(feature)

            ablated_model = copy.deepcopy(baseline_model)
            ablated_model.categorical_features = adjusted_categorical_features
            ablated_model.fit(X_train_removed, y_train, X_val_removed, y_val)
            _, ablated_loss = ablated_model.evaluate_dataset(X_val_removed, y_val)

            ablation_importance = ablated_loss - baseline_loss

            results.append(
                {
                    "feature": feature,
                    "baseline_loss": baseline_loss,
                    "ablated_loss": ablated_loss,
                    "ablation_importance": ablation_importance,
                }
            )
        return results

    def ablate_feature_groups(self, X_train, X_val, y_train, y_val, features_to_remove):
        baseline_model = copy.deepcopy(self)
        baseline_model.fit(X_train, y_train, X_val, y_val)
        _, baseline_loss = baseline_model.evaluate_dataset(X_val, y_val)

        results = []
        for feature_to_remove in features_to_remove:
            X_train_removed = np.delete(X_train, feature_to_remove, axis=1)
            X_val_removed = np.delete(X_val, feature_to_remove, axis=1)
            adjusted_categorical_features = self._adjust_categorical_features(
                feature_to_remove
            )
        ablated_model = copy.deepcopy(baseline_model)
        ablated_model.categorical_features = adjusted_categorical_features
        ablated_model.fit(X_train_removed, y_train, X_val_removed, y_val)
        _, ablated_loss = ablated_model.evaluate_dataset(X_val_removed, y_val)

        ablation_importance = ablated_loss - baseline_loss

        results.append(
            {
                "features": features_to_remove,
                "baseline_loss": baseline_loss,
                "ablated_loss": ablated_loss,
                "ablation_importance": ablation_importance,
            }
        )
        return results

    def _adjust_categorical_features(self, removed_feature):
        adjusted_categorical_features = []
        for feature in self.categorical_features:
            if feature == removed_feature:
                continue
            elif feature > removed_feature:
                adjusted_categorical_features.append(feature - 1)
            else:
                adjusted_categorical_features.append(feature)
        return adjusted_categorical_features

    def partial_dependence(self, X, feature, grid_values, verbose=False):
        results = []
        for value in grid_values:
            X_temp = copy.deepcopy(X)
            X_temp[:, feature] = value
            predictions = self.predict(X_temp)
            average_prediction = np.mean(predictions)
            results.append(
                {"feature_value": value, "average_prediction": average_prediction}
            )
        if verbose:
            os.makedirs("boosting/img", exist_ok=True)
            print("\nPartial Dependence : \n")
            average_predictions = []
            for result in results:
                print(
                    f"Average prediction is {result["average_prediction"]} when all features are set to {result["feature_value"]}"
                )
                average_predictions.append(result["average_prediction"])
            plt.title("Partial Dependence Plot")
            plt.ylabel("Given Values")
            plt.xlabel("Average Prediction")
            plt.plot(average_predictions, grid_values)
            plt.savefig("boosting/img/PDP.png")
            plt.show()
        return results
    
    def individual_conditional_expectation(self, X, feature, grid_values, verbose=False):
        results = []
        for value in grid_values:
            X_temp = copy.deepcopy(X)
            X_temp[:, feature] = value
            predictions = self.predict(X_temp)
            results.append({
                "feature_value": value,
                "predictions": predictions,
            })
        if verbose:
            os.makedirs("boosting/img", exist_ok=True)
            print("\nICE Curve : \n")
            predictions = []
            for result in results:
                print(f"Predictions are {result["predictions"]} when all features are set to {result["feature_value"]}")
                predictions.append(result["predictions"])
            plt.title("ICE Plot")
            plt.ylabel("Given Values")
            plt.xlabel("Predictions")
            plt.plot(predictions, grid_values)
            plt.savefig("boosting/img/ICE.png")
            plt.show()
        return results

    def _choose_grid_values(self, values, grid_size):
        percentiles = np.linspace(5, 95, grid_size)
        grid = np.percentile(values, percentiles)
        return np.unique(grid)
    
    def _choose_categorical_grid_values(self, values, max_categories):
        print(self.categorical_features)
        category_counts = [self.categorical_features.count(value) for value in values]
        print(category_counts)
        return np.sort(category_counts)[:max_categories]

    def verbose_permutation_importance(
        self,
        permutation_importance_averages,
        permutation_importance_standart_deviations,
        std_values,
        values_std,
        avg_values,
        values_avg,
    ):
        print("\nPermutation Importances : ")
        for key_avg, value_avg in permutation_importance_averages.items():
            print(f"feature : {key_avg}, importance is {value_avg}")
        print()
        for key_std, value_std in permutation_importance_standart_deviations.items():
            print(f"feature: {key_std}, standart deviation is {value_std}")
        print("\nstd_values:", std_values)
        print("values_std:", values_std)
        print("\navg_values:", avg_values)
        print("values_avg", values_avg)

    def visualize_permutation_importance(
        self,
        feature_labels,
        values_avg,
        values_std,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle("Permutation Importances")

        axes[0].barh(feature_labels, values_avg)
        axes[0].set_xlabel("Percentage")
        axes[0].set_ylabel("Features")
        axes[0].set_title("Average Importance")

        axes[1].barh(feature_labels, values_std)
        axes[1].set_xlabel("Percentage")
        axes[1].set_ylabel("Features")
        axes[1].set_title("Std Importance")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs("boosting/img", exist_ok=True)

        plt.savefig("boosting/img/prmttn_mprtncs.png")
        plt.show()

    def visualize(self):
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Model is not fit")

        os.makedirs("boosting/img", exist_ok=True)

        X = np.asarray(self.X_train_)
        y = np.asarray(self.y_train_).ravel()

        y_pred = self.predict(X).ravel()

        plt.figure(figsize=(7, 5))
        plt.scatter(y, y_pred, alpha=0.7)

        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())

        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        plt.xlabel("Actual y")
        plt.ylabel("Predicted y")
        plt.title("Actual vs Predicted")
        plt.grid(True)
        plt.savefig("boosting/img/actual_vs_predicted.png")
        plt.show()

        residuals = y - y_pred

        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(0, linestyle="--")

        plt.xlabel("Predicted y")
        plt.ylabel("Residual: y - prediction")
        plt.title("Residual Plot")
        plt.grid(True)
        plt.savefig("boosting/img/residual_plot.png")
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.hist(y_pred, bins=30, alpha=0.7)

        plt.xlabel("Prediction")
        plt.ylabel("Frequency")
        plt.title("Prediction Distribution")
        plt.grid(True)
        plt.savefig("boosting/img/prediction_distribution.png")
        plt.show()

        losses = []

        current_pred = np.repeat(self.F_0x, X.shape[0])
        initial_mse = np.mean((y - current_pred) ** 2)
        losses.append(initial_mse)

        for tree, feature_indices in zip(self.trees, self.feature_indices):
            current_pred += self.learning_rate * tree.predict(X[:, feature_indices])
            mse = np.mean((y - current_pred) ** 2)
            losses.append(mse)

        plt.figure(figsize=(7, 5))
        plt.plot(losses)

        plt.xlabel("Boosting Round")
        plt.ylabel("Training MSE")
        plt.title("Training Loss Over Boosting Rounds")
        plt.grid(True)
        plt.savefig("boosting/img/training_loss.png")
        plt.show()

        sample_count = min(10, X.shape[0])

        prediction_history = []

        current_pred = np.repeat(self.F_0x, X.shape[0])
        prediction_history.append(current_pred[:sample_count].copy())

        for tree, feature_indices in zip(self.trees, self.feature_indices):
            current_pred += self.learning_rate * tree.predict(X[:, feature_indices])
            prediction_history.append(current_pred[:sample_count].copy())

        prediction_history = np.asarray(prediction_history)

        plt.figure(figsize=(8, 5))

        for sample_index in range(sample_count):
            plt.plot(prediction_history[:, sample_index], alpha=0.7)

        plt.xlabel("Boosting Round")
        plt.ylabel("Prediction")
        plt.title("Prediction Movement During Boosting")
        plt.grid(True)
        plt.savefig("boosting/img/prediction_movement.png")
        plt.show()

        feature_usage = np.zeros(X.shape[1])

        for feature_indices in self.feature_indices:
            for feature_index in feature_indices:
                feature_usage[feature_index] += 1

        plt.figure(figsize=(8, 5))
        plt.bar(np.arange(X.shape[1]), feature_usage)

        plt.xlabel("Feature")
        plt.ylabel("Times Used")
        plt.title("Feature Usage Across Trees")
        plt.grid(True)
        plt.savefig("boosting/img/feature_usage.png")
        plt.show()

        for feature in range(X.shape[1]):
            values = X[:, feature]
            unique_values = np.unique(values)

            plt.figure(figsize=(8, 5))

            if len(unique_values) <= 25:
                mean_actual = []
                mean_prediction = []

                for value in unique_values:
                    mask = values == value
                    mean_actual.append(np.mean(y[mask]))
                    mean_prediction.append(np.mean(y_pred[mask]))

                positions = np.arange(len(unique_values))

                plt.plot(positions, mean_actual, marker="o", label="Mean actual")
                plt.plot(
                    positions, mean_prediction, marker="o", label="Mean prediction"
                )

                plt.xticks(positions, unique_values, rotation=45)

                plt.xlabel(f"Feature {feature}")
                plt.ylabel("Mean value")
                plt.title(f"Feature {feature} Response")
                plt.legend()
                plt.grid(True)

            else:
                plt.scatter(values, y, alpha=0.4, label="Actual")
                plt.scatter(values, y_pred, alpha=0.4, label="Predicted")

                plt.xlabel(f"Feature {feature}")
                plt.ylabel("Target / Prediction")
                plt.title(f"Feature {feature} Response")
                plt.legend()
                plt.grid(True)

            plt.savefig(f"boosting/img/feature_{feature}_response.png")
            plt.show()

        if X.shape[1] < 2:
            return

        x1_values = np.unique(X[:, 0])
        x2_values = np.unique(X[:, 1])

        use_discrete_grid = len(x1_values) <= 30 and len(x2_values) <= 30

        if use_discrete_grid:
            xx1, xx2 = np.meshgrid(x1_values, x2_values)
        else:
            x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
            x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

            x1_grid = np.linspace(x1_min, x1_max, 100)
            x2_grid = np.linspace(x2_min, x2_max, 100)

            xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

        if X.shape[1] == 2:
            grid_points = np.c_[xx1.ravel(), xx2.ravel()]
        else:
            average_point = np.mean(X, axis=0)
            grid_points = np.tile(average_point, (xx1.ravel().shape[0], 1))

            grid_points[:, 0] = xx1.ravel()
            grid_points[:, 1] = xx2.ravel()

        grid_predictions = self.predict(grid_points)
        zz = grid_predictions.reshape(xx1.shape)

        if use_discrete_grid:
            plt.figure(figsize=(8, 6))
            plt.imshow(zz, origin="lower", aspect="auto")
            plt.colorbar(label="Predicted y")

            plt.xticks(np.arange(len(x1_values)), x1_values)
            plt.yticks(np.arange(len(x2_values)), x2_values)

            x_positions = np.searchsorted(x1_values, X[:, 0])
            y_positions = np.searchsorted(x2_values, X[:, 1])

            plt.scatter(x_positions, y_positions, alpha=0.7)

            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Discrete Prediction Heatmap")
            plt.grid(False)
            plt.savefig("boosting/img/discrete_prediction_heatmap.png")
            plt.show()

        else:
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(xx1, xx2, zz, levels=30)
            plt.colorbar(contour, label="Predicted y")

            plt.scatter(X[:, 0], X[:, 1], alpha=0.7)

            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Prediction Surface")
            plt.grid(True)
            plt.savefig("boosting/img/prediction_surface.png")
            plt.show()
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(xx1, xx2, zz, alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], y, alpha=0.7)

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target / Prediction")
        ax.set_title("Learned Regression Surface")

        plt.savefig("boosting/img/3d_ctbst_srfc.png")
        plt.show()

    # def _shapley_value(self, X, S, features, M, j):
    #     phi_j = 0
    #     features_except_j = features[features != j]
    #     for subset_size in range(len(features_except_j) + 1):
    #         for S in combinations(features_except_j, subset_size):
    #             X_S = X[:, S]
    #             S_with_j = S.append(j)
    #             X_S_with_j = X[:, S_with_j]
    #             weight = (
    #                 math.factorial(len(S)) *
    #                 math.factorial(M - len(S) - 1)
    #                 / math.factorial(M)
    #             )
    #             contribution = self._condition_value(X_S_with_j, X, S_with_j) - self._condition_value(X_S, X, S)
    #             phi_j += weight * contribution
    #     return phi_j


if __name__ == "__main__":

    from .generate_categories import create_categorical_dataset

    seed = 42
    X, y = create_categorical_dataset(n_samples=300, seed=seed)
    X = np.asarray(X)
    y = np.asarray(y)

    catboost = CatBoost(
        loss_type=CatBoost.LossType.SSE,
        restore_best=False,
        validation=False,
        early_stopping=False,
        sub_sample=1,
        column_sub_sample=1,
        symmetrical=False,
        boosting_type=CatBoost.BoostingType.PLAIN,
    )
    catboost.fit(X, y, None, None)

    # catboost.visualize()
    # catboost.permutation_importance(X, y, verbose=True)

    row = X[[0]]
    rng = np.random.default_rng(seed)
    background_index = rng.choice(
        X.shape[0],
        size=min(20, X.shape[0]),
        replace=False,
    )
    background_X = X[background_index]
    # result = catboost.agnostic_shapley(
    #     X=row,
    #     background_X=background_X,
    #     verbose=True,
    # )
    # sampled_result = catboost.sampling_shapley(
    #     X=row,
    #     background_X=background_X,
    #     number_of_permutations=100,
    #     verbose=True,
    # )
    # kernel_result = catboost.kernel_shapley(
    #     X=row,
    #     background_X=background_X,
    #     number_of_samples=500,
    #     verbose=True,
    # )
    # catboost.tree_contributions(row, verbose=True)
    # tree_result = catboost.tree_shap_ensemble(row)
    # tree_result = catboost.ablation_importances(X, y, X, y)
    # print(tree_result)
    
    values = np.arange(X.shape[1] - 1)
    grid_values = catboost._choose_grid_values(values, 10)
    # grid_values = catboost._choose_categorical_grid_values(values, 10)
    # catboost.partial_dependence(X, 1, grid_values, verbose=True)
    catboost.individual_conditional_expectation(X, 1, grid_values, verbose=True)
    