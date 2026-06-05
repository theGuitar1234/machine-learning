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


class CatBoost:
    
    class LossType(Enum):
        BINARY_CROSS_ENTROPY = 1
        SSE = 2
    
    class BoostingType:
        PLAIN = 1
        ORDERED = 2
    
    @dataclass
    class TrainDefaults:
        epsilon=1e-12
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
            raise ValueError("sub_sample and column_sub_sample must be a positive fraction less than 1")
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
        number_of_selected_features = max(1, int(self.column_sub_sample * number_of_features))

        epsilon = self.config.epsilon
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p0 = np.clip(np.mean(y), epsilon, 1 - epsilon)
                self.F_0x = np.log(p0 / (1 - p0))
            case self.LossType.SSE:
                self.F_0x = np.mean(y)
            case _:
                raise ValueError(f"Unsupported {self.LossType}, supported values are {list(self.LossType)}")
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
                    minimum_gain=self.minimum_gain
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
            
            indices = np.random.choice(number_of_samples, size=sample_size, replace=False)
            feature_indices = np.random.choice(number_of_features, size=number_of_selected_features, replace=False)
            
            X_sub = X[indices][:, feature_indices]
            pseudo_residual_sub = pseudo_residual[indices]
            y_sub_original = y[indices]
            
            # Deprecated - obselete after feature transformer
            local_categorical_features = self._local_categorical_features(feature_indices)
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
                    print("Overfitting detected. Early stopping at round: ", round, "Best Round : ", best_round)
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
        number_of_selected_features = max(1, int(self.column_sub_sample * number_of_features))

        epsilon = self.config.epsilon
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p0 = np.clip(np.mean(y), epsilon, 1 - epsilon)
                self.F_0x = np.log(p0 / (1 - p0))
            case self.LossType.SSE:
                self.F_0x = np.mean(y)
            case _:
                raise ValueError(f"Unsupported {self.LossType}, supported values are {list(self.LossType)}")
        self.F_x = np.repeat(self.F_0x, y.shape[0])
        
        block_size = self.config.ordered_boosting_blocks
        permutation = self.rng.permutation(number_of_samples)
        ordered_blocks = [permutation[i:i + block_size] for i in range(0, len(permutation), block_size)]
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
                    minimum_gain=self.minimum_gain
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
            
            indices = np.random.choice(number_of_samples, size=sample_size, replace=False)
            feature_indices = np.random.choice(number_of_features, size=number_of_selected_features, replace=False)
            
            X_sub = X[indices][:, feature_indices]
            ordered_residual_sub = ordered_residual[indices]
            y_sub_original = y[indices]
            
            # Deprecated - obselete after feature transformer
            local_categorical_features = self._local_categorical_features(feature_indices)
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
                self.leaf_correction(tree, tree.X_train_, y[indices], ordered_prediction[indices])
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
            self.F_x += self.learning_rate * tree.predict(X[:, feature_indices])
            
            for b in range(number_of_blocks):
                prefix_rows = np.concatenate(ordered_blocks[:b + 1])
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
                self.F_prefix[b] += self.learning_rate * auxiliary_tree.predict(X[:, feature_indices])
                
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
                    print("Overfitting detected. Early stopping at round: ", round, "Best Round : ", best_round)
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
            
            encoded[row] = (
                previous_sum + prior_weight * prior
            ) / (
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
            
            output[row] = (
                previous_sum + prior_weight * prior
            ) / (
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
                raise ValueError(f"Unsupported {self.LossType}, supported values are {list(self.LossType)}")

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
                leaf_to_indices[leaf_id] = {
                    "leaf": leaf,
                    "indices": []
                }
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
                raise ValueError(f"Unsupported {self.LossType}, supported values are {list(self.LossType)}")

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
                raise ValueError(f"Unsupported {self.LossType}, supported values are {list(self.LossType)}")
        return predictions, loss

    def predict(self, X):
        if self.F_0x is None:
            raise RuntimeError("Model is not fit")
        X_model = self._transform_features(X, training_mode=False)
        prediction = np.repeat(self.F_0x, X.shape[0])
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            prediction += self.learning_rate * tree.predict(X_model[:, feature_indices])
        return prediction
    
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
                plt.plot(positions, mean_prediction, marker="o", label="Mean prediction")

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


if __name__ == "__main__":
    
    from .generate_categories import create_categorical_dataset

    X, y = create_categorical_dataset(n_samples=300, seed=42)
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
        boosting_type=CatBoost.BoostingType.ORDERED,
    )
    catboost.fit_ordered(X, y, None, None)
    
    catboost.visualize()
