from enum import Enum

from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_regression
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

import numpy as np
import sys
import os
import numpy as np
import copy
import math


class Node:
    def __init__(self, feature=None, threshold=None):
        self.feature = feature
        self.threshold = threshold
        self.is_leaf = False
        self.left = None
        self.right = None
        self.value = None
        self.majority_class = None
        self.number_of_samples = None
        self.number_of_classes = None
        self.number_of_leaves = None
        self.leaf_error = None
        self.subtree_error = None
        self.is_root = False
        self.depth = None
        self.max_depth = float("inf")
        self.max_depth_reached = "MAX_DEPTH_REACHED"

    def left_add_prefix(self, text):
        if self.depth > self.max_depth:
            return self.max_depth_reached
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_add_prefix(self, text):
        if self.depth > self.max_depth:
            return self.max_depth_reached

        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        if self.depth > self.max_depth:
            return self.max_depth_reached

        if self.is_root:
            text = f"root [feature={self.feature}, " f"threshold={self.threshold}]"
        else:
            text = f"-> node [feature={self.feature}, " f"threshold={self.threshold}]"

        if self.left is not None:
            text += "\n" + self.left_add_prefix(str(self.left).rstrip("\n"))

        if self.right is not None:
            if self.left is None:
                text += "\n"
            text += self.right_add_prefix(str(self.right).rstrip("\n"))

        return text

    def max_depth_below(self):
        max_depth = self.depth

        if self.left is not None:
            max_depth = max(max_depth, self.left.max_depth_below())
        if self.right is not None:
            max_depth = max(max_depth, self.right.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        if only_leaves:
            if self.left is None and self.right is None:
                return 1

            count = 0
        else:
            count = 1

        if self.left is not None:
            count += self.left.count_nodes_below(only_leaves=only_leaves)

        if self.right is not None:
            count += self.right.count_nodes_below(only_leaves=only_leaves)

        return count


class ANonSeriousDecisionTree:

    class InformationGain(Enum):
        GINI = 1
        ENTROPY = 2

    class MinimumError(Enum):
        M_PROBABILITY = 1
        MINIMUM_ERROR_ESTIMATE = 2

    class RandomCriterion(Enum):
        RANDOM_FEATURE = 1
        RANDOM_SPLIT = 2

    class MaxFeatures(Enum):
        SQRT = 1
        LOG2 = 2

    class TreeType(Enum):
        CLASSIFICATION = 1
        REGRESSION = 2

    class Error(Enum):
        MSE = 1
        SSE = 2
        RMSE = 3
        R2 = 4

    def __init__(
        self,
        minimum_population_size=2,
        minimum_split_size=1,
        minimum_gain=0.001,
        max_depth=3,
        str_max_depth=float("inf"),
        categorical=False,
        adjacent=False,
        log=False,
        seed=0,
        information_gain=None,
        random_criterion=None,
        max_features_ratio=0.2,
        max_number_of_features=5,
        max_features=MaxFeatures.SQRT,
        tree_type=TreeType.CLASSIFICATION,
        error=Error.MSE,
    ):
        self.rng = np.random.default_rng(seed)
        self.root = None
        self.max_depth = max_depth
        self.categorical = categorical
        self.adjacent = adjacent
        self.minimum_population_size = minimum_population_size
        self.minimum_split_size = minimum_split_size
        self.minimum_gain = minimum_gain
        self.information_gain = information_gain
        self.log = log
        self.is_root_set = False
        self.str_max_depth = str_max_depth
        self.feature_importance = {}
        self.max_depth_reached = "MAX_DEPTH_REACHED"
        self.random_criterion = random_criterion
        self.max_features_ratio = max_features_ratio
        self.max_number_of_features = max_number_of_features
        self.max_features = max_features
        self.tree_type = tree_type
        self.error = error

    def __str__(self):
        return str(self.root)

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def fit(self, X, y, verbose=False):
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must match in shapes : {X.shape[0]} != {y.shape[0]}"
            )
        self.X_train_ = X
        self.y_train_ = y
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.class_priors_ = class_counts / len(y)

        self.root = self._build_tree(X, y, depth=0)

        if verbose:
            number_of_leaves = self.count_nodes(only_leaves=True)
            _, accuracy = self.evaluate_dataset(self.X_train_, self.y_train_)
            print(f"""\nTraining finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {number_of_leaves}
    - Accuracy on training data : {accuracy}""")

            print("\nFeature Importances :")
            total = sum(self.feature_importance.values())
            for key, value in self.feature_importance.items():
                print(f"For feature{key}, importance is {(value/total) * 100}%")

            features = self.feature_importance.keys()
            values = [(v / total) * 100 for v in self.feature_importance.values()]
            plt.figure("Feature Importances")
            plt.xlabel("Features")
            plt.ylabel("Values")
            plt.barh(features, values)

            os.makedirs("img", exist_ok=True)

            plt.savefig("img/ftr_mprtncs.png")
            plt.show()

        return self

    def _build_tree(self, X, y, depth):
        node = Node()
        if not self.is_root_set:
            node.is_root = True
            self.is_root_set = True
        node.depth = depth
        node.max_depth = self.str_max_depth

        if self.tree_type is self.TreeType.CLASSIFICATION:
            node.majority_class = self._majority_class(y)
        node.number_of_samples = len(y)
        _, node.leaf_error = self._leaf_error(y)
        node.number_of_classes = self._class_counts(y)

        if self.tree_type is self.TreeType.REGRESSION:
            node.value = np.mean(y)

        if (
            depth >= self.max_depth
            or len(set(y)) == 1
            or len(y) <= self.minimum_population_size
        ):
            node.is_leaf = True
            match self.tree_type:
                case self.TreeType.CLASSIFICATION:
                    node.value = self._majority_class(y)
                case self.TreeType.REGRESSION:
                    node.value = np.mean(y)
                case _:
                    raise ValueError(
                        f"Unsupported {self.tree_type}, supported values are {list(self.TreeType)}"
                    )
            return node

        match self.tree_type:
            case self.TreeType.CLASSIFICATION:
                feature, threshold = self._classification_split(X, y)
            case self.TreeType.REGRESSION:
                feature, threshold = self._regression_split(X, y)
            case _:
                raise ValueError(
                    f"Unsupported {self.tree_type}, supported values are {list(self.TreeType)}"
                )

        if self.log:
            print(
                f"Fitting the tree... [Best Feature : {feature}] [Best Split : {threshold}]"
            )

        if feature is None:
            node.is_leaf = True
            match self.tree_type:
                case self.TreeType.CLASSIFICATION:
                    node.value = self._majority_class(y)
                case self.TreeType.REGRESSION:
                    node.value = np.mean(y)
                case _:
                    raise ValueError(
                        f"Unsupported {self.TreeType}, supported values are {list(self.TreeType)}"
                    )
            return node
        node.feature = feature
        node.threshold = threshold
        node.value = None

        if self.categorical:
            left_mask = X[:, feature] == threshold
            right_mask = X[:, feature] != threshold
        else:
            left_mask = X[:, feature] > threshold
            right_mask = X[:, feature] <= threshold

        if (
            len(X[left_mask]) < self.minimum_split_size
            or len(X[right_mask]) < self.minimum_split_size
        ):
            node.is_leaf = True
            match self.tree_type:
                case self.TreeType.CLASSIFICATION:
                    node.value = self._majority_class(y)
                case self.TreeType.REGRESSION:
                    node.value = np.mean(y)
                case _:
                    raise ValueError(
                        f"Unsupported {self.TreeType}, supported values are {list(self.TreeType)}"
                    )
            return node

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _classification_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = float("inf")

        number_of_features = X.shape[1]
        features = np.arange(number_of_features)

        if self.random_criterion is not None:
            match self.random_criterion:
                case self.RandomCriterion.RANDOM_FEATURE:
                    max_feature_method = None
                    match self.max_features:
                        case self.MaxFeatures.SQRT:
                            max_feature_method = np.sqrt
                        case self.MaxFeatures.LOG2:
                            max_feature_method = np.log2
                        case _:
                            raise ValueError(
                                f"Unsupported {self.max_features}, supported values are {list(self.MaxFeatures)}"
                            )
                    ratio_size = self.max_features_ratio * number_of_features
                    number_of_features = (
                        ratio_size
                        if ratio_size <= number_of_features
                        else number_of_features
                    )
                    number_of_features = (
                        number_of_features
                        if number_of_features <= self.max_number_of_features
                        else self.max_number_of_features
                    )

                    candidate_count = max(
                        1, int(max_feature_method(number_of_features))
                    )
                    features = self.rng.choice(
                        features, size=candidate_count, replace=False
                    )
                case self.RandomCriterion.RANDOM_SPLIT:
                    while True:
                        random_feature = self.rng.choice(
                            features, size=1, replace=False
                        )[0]

                        values = X[:, random_feature]

                        min_value = min(values)
                        max_value = max(values)

                        if min_value == max_value:
                            continue

                        threshold = self.rng.uniform(min_value, max_value)

                        left_mask = values > threshold
                        right_mask = values <= threshold

                        if len(values[left_mask]) == 0 or len(values[right_mask]) == 0:
                            continue

                        return random_feature, threshold
                case _:
                    raise ValueError(
                        f"Unsupported {self.random_criterion}, supported values are {list(self.RandomCriterion)}"
                    )

        for feature in features:
            values = X[:, feature]
            order = np.argsort(values)
            sorted_values = values[order]
            categories = np.unique(sorted_values)

            if not self.categorical:
                if self.adjacent:
                    sorted_y = y[order]
                    mask = sorted_y[1:] != sorted_y[:-1]
                    mask &= sorted_values[1:] != sorted_values[:-1]
                    categories = (
                        sorted_values[1:][mask] + sorted_values[:-1][mask]
                    ) / 2
                else:
                    categories = (categories[1:] + categories[:-1]) / 2
            for category in categories:
                if self.categorical:
                    left_mask = values == category
                    right_mask = values != category
                else:
                    left_mask = values > category
                    right_mask = values <= category
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_group = y[left_mask]
                right_group = y[right_mask]

                match self.information_gain:
                    case self.InformationGain.GINI:
                        __left = self.gini(left_group)
                        __right = self.gini(right_group)
                    case self.InformationGain.ENTROPY:
                        __left = self.entropy(left_group)
                        __right = self.entropy(right_group)
                    case _:
                        raise ValueError(
                            f"Unsupported Information Gain, supported values are {list(self.InformationGain)}"
                        )

                _weighted_impurity = self.weighted_impurity(
                    __left, __right, len(left_group), len(right_group)
                )

                if _weighted_impurity < best_gain:
                    best_gain = _weighted_impurity
                    best_feature = feature
                    best_threshold = category

        if best_feature is None:
            return None, None

        match self.information_gain:
            case self.InformationGain.GINI:
                __parent_impurity = self.gini(y)
            case self.InformationGain.ENTROPY:
                __parent_impurity = self.entropy(y)
            case _:
                raise ValueError(
                    f"Unsupported Information Gain, supported values are {list(self.InfomrationGain)}"
                )

        if __parent_impurity - best_gain < self.minimum_gain:
            return None, None

        impurity_decrease = __parent_impurity - best_gain

        self.feature_importance[best_feature] = (
            self.feature_importance.get(best_feature, 0) + len(y) * impurity_decrease
        )

        return best_feature, best_threshold

    def _regression_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_score = float("inf")

        parent_score = self.mse_mean(y)
        
        number_of_features = X.shape[1]
        features = np.arange(number_of_features)

        if self.random_criterion is not None:
            match self.random_criterion:
                case self.RandomCriterion.RANDOM_FEATURE:
                    max_feature_method = None
                    match self.max_features:
                        case self.MaxFeatures.SQRT:
                            max_feature_method = np.sqrt
                        case self.MaxFeatures.LOG2:
                            max_feature_method = np.log2
                        case _:
                            raise ValueError(
                                f"Unsupported {self.max_features}, supported values are {list(self.MaxFeatures)}"
                            )
                    ratio_size = self.max_features_ratio * number_of_features
                    number_of_features = (
                        ratio_size
                        if ratio_size <= number_of_features
                        else number_of_features
                    )
                    number_of_features = (
                        number_of_features
                        if number_of_features <= self.max_number_of_features
                        else self.max_number_of_features
                    )
                    candidate_count = max(1, int(max_feature_method(number_of_features)))
                    features = self.rng.choice(
                        features, size=candidate_count, replace=False
                    )
                case self.RandomCriterion.RANDOM_SPLIT:
                    while True:
                        random_feature = self.rng.choice(features, size=1, replace=False)[0]

                        values = X[:, random_feature]

                        min_value = min(values)
                        max_value = max(values)

                        if min_value == max_value:
                            continue

                        threshold = self.rng.uniform(min_value, max_value)

                        left_mask = values > threshold
                        right_mask = values <= threshold

                        if len(values[left_mask]) == 0 or len(values[right_mask]) == 0:
                            continue

                        return random_feature, threshold
                case _:
                    raise ValueError(
                        f"Unsupported {self.random_criterion}, supported values are {list(self.RandomCriterion)}"
                    )

        for feature in features:
            values = X[:, feature]
            unique_values = np.unique(values)

            if len(unique_values) <= 1:
                continue

            thresholds = (unique_values[1:] + unique_values[:-1]) / 2

            for threshold in thresholds:
                left_mask = values > threshold
                right_mask = values <= threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]

                score = self.weighted_mse(left_y, right_y)

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None:
            return None, None

        improvement = parent_score - best_score

        if improvement < self.minimum_gain:
            return None, None
        return best_feature, best_threshold

    def mse(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def mse_mean(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def sse(self, y, y_hat):
        return np.sum((y - y_hat) ** 2)

    def rmse(self, y, y_hat):
        return np.sqrt(self.mse(y, y_hat))

    def r2(self, y, y_hat):
        ss_res = np.sum((y - y_hat) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        if ss_total == 0:
            return 0
        return 1 - ss_res / ss_total

    def weighted_mse(self, left_y, right_y):
        total = len(left_y) + len(right_y)
        return len(left_y) / total * self.mse_mean(left_y) + len(
            right_y
        ) / total * self.mse_mean(right_y)

    def gini(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions**2)

    def entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return -np.sum(proportions * np.log2(proportions))

    def weighted_impurity(self, left, right, left_size, right_size):
        total_size = left_size + right_size
        return (left_size / total_size) * left + (right_size / total_size) * right

    def calculate_error(self, node):
        if node is None:
            return

        if node.is_leaf or node.value is not None:
            node.number_of_leaves = 1
            node.subtree_error = node.leaf_error
            return

        self.calculate_error(node.left)
        self.calculate_error(node.right)

        node.number_of_leaves = node.left.number_of_leaves + node.right.number_of_leaves
        node.subtree_error = node.left.subtree_error + node.right.subtree_error

    def weakest_node(self, node):
        if (
            node is None
            or node.is_leaf
            or node.value is not None
            or node.number_of_leaves <= 1
        ):
            return None, float("inf")

        alpha = (node.leaf_error - node.subtree_error) / (node.number_of_leaves - 1)

        left_node, left_alpha = self.weakest_node(node.left)
        right_node, right_alpha = self.weakest_node(node.right)

        weakest = node
        weakest_alpha = alpha

        if left_alpha < weakest_alpha:
            weakest = left_node
            weakest_alpha = left_alpha
        if right_alpha < weakest_alpha:
            weakest = right_node
            weakest_alpha = right_alpha

        return weakest, weakest_alpha

    def prune_reduced_error(self, node, X_val, y_val):
        if node is None:
            return
        if node.value is not None:
            return

        if self.categorical:
            left_mask = X_val[:, node.feature] == node.threshold
            right_mask = X_val[:, node.feature] != node.threshold
        else:
            left_mask = X_val[:, node.feature] > node.threshold
            right_mask = X_val[:, node.feature] <= node.threshold

        self._prune_node(node.left, X_val[left_mask], y_val[left_mask])
        self._prune_node(node.right, X_val[right_mask], y_val[right_mask])

        if len(y_val) == 0:
            return

        subtree_predictions = np.array([self.predict_one(x, node) for x in X_val])
        subtree_accuracy = np.mean(subtree_predictions == y_val)

        leaf_predictions = np.full_like(y_val, node.majority_class)
        leaf_accuracy = np.mean(leaf_predictions == y_val)

        if leaf_accuracy >= subtree_accuracy:
            self._prune_node(node)

    def prune_post_complexity(self, X_val, y_val):
        candidates = []
        current_tree = copy.deepcopy(self)

        while True:
            current_tree.calculate_error(current_tree.root)
            candidates.append(copy.deepcopy(current_tree))
            weakest, alpha = current_tree.weakest_node(current_tree.root)

            if weakest is None or alpha == float("inf"):
                break
            self._prune_node(weakest)

        best_tree = None
        best_accuracy = -1

        for candidate in candidates:
            _, accuracy = candidate.evaluate_dataset(X_val, y_val)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_tree = candidate
        self.root = best_tree.root

        return self

    def prune_pessimistic(self):
        self._prune_node_pessimistic(self.root)
        return self

    def _prune_node_pessimistic(self, node):
        if node is None:
            return
        if node.value is not None:
            self._prune_node(node)
            return

        self._prune_node_pessimistic(node.left)
        self._prune_node_pessimistic(node.right)

        node.number_of_leaves = node.left.number_of_leaves + node.right.number_of_leaves
        node.subtree_error = node.left.subtree_error + node.right.subtree_error

        leaf_pessimistic_error = node.leaf_error + 0.5
        subtree_pessimistic_error = node.subtree_error + 0.5 + node.number_of_leaves

        standart_error = math.sqrt(
            (
                subtree_pessimistic_error
                * (node.number_of_samples - subtree_pessimistic_error)
            )
            / node.number_of_samples
        )

        if leaf_pessimistic_error <= subtree_pessimistic_error + standart_error:
            self._prune_node(node)

    def prune_error_based(self, confidence_factor=0.25, subtree_raising=True):
        self._prune_error_based_node(
            self.root,
            self.X_train_,
            self.y_train_,
            confidence_factor,
            subtree_raising,
        )
        return self

    def _prune_error_based_node(
        self, node, X_node, y_node, confidence_factor, subtree_raising
    ):
        if node is None or node.is_leaf or node.value is not None:
            return
        X_left, y_left, X_right, y_right = self._route_node_data(node, X_node, y_node)

        self._prune_error_based_node(
            node.left, X_left, y_left, confidence_factor, subtree_raising
        )
        self._prune_error_based_node(
            node.right, X_right, y_right, confidence_factor, subtree_raising
        )

        N = len(y_node)
        tree_errors = self._count_errors(node, X_node, y_node)
        tree_estimate = N * self.ucf(tree_errors, N, confidence_factor)
        majority, leaf_errors = self._leaf_error(y_node)
        leaf_estimate = N * self.ucf(leaf_errors, N, confidence_factor)

        raised_estimate = float("inf")
        raised_subtree = None

        if subtree_raising and node.left is not None and node.right is not None:
            if len(y_left) >= len(y_right):
                raised_subtree = node.left
            else:
                raised_subtree = node.right
            raised_errors = self._count_errors(raised_subtree, X_node, y_node)
            raised_estimate = N * self.ucf(raised_errors, N, confidence_factor)

        if leaf_estimate <= tree_estimate and leaf_estimate <= raised_estimate:
            self._prune_node(node)
        elif raised_estimate < tree_estimate and raised_estimate < leaf_estimate:
            node.feature = raised_subtree.feature
            node.threshold = raised_subtree.threshold
            node.left = raised_subtree.left
            node.right = raised_subtree.right
            node.value = raised_subtree.value
            node.is_leaf = raised_subtree.is_leaf

            node.majority_class, node.leaf_error = self._leaf_error(y_node)
            node.number_of_samples = len(y_node)
        else:
            pass

    def _route_node_data(self, node, X, y):
        if self.categorical:
            left_mask = X[:, node.feature] == node.threshold
            right_mask = X[:, node.feature] != node.threshold
        else:
            left_mask = X[:, node.feature] > node.threshold
            right_mask = X[:, node.feature] <= node.threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _count_errors(self, node, X, y):
        predictions = np.array([self.predict_one(x, node) for x in X])
        return np.sum(predictions != y)

    def _leaf_error(self, y):
        majority = self._majority_class(y)
        errors = np.sum(y != majority)
        return majority, errors

    def ucf(self, errors, total, confidence_factor=0.25):
        if total == 0:
            return 0.0
        if errors == 0:
            return 1.0 - confidence_factor ** (1.0 / total)
        if errors == total:
            return 1.0

        low = errors / total
        high = 1.0

        while high - low > 1e-12:
            mid = (low + high) / 2
            probability = self.binomial_cdf(errors, total, mid)
            if probability > confidence_factor:
                low = mid
            else:
                high = mid
        return high

    def binomial_cdf(self, errors, total, p):
        return sum(
            math.comb(total, i) * (p**i) * ((1 - p) ** (total - i))
            for i in range(errors + 1)
        )

    def prune_minimum_error(self):
        self._prune_minimum_error(self.root)
        return self

    def _prune_minimum_error(self, node):
        if node is None:
            return 0

        static_error = self._static_error(node)

        if node.is_leaf:
            node.expected_error = static_error
            return node.expected_error
        left_error = self._prune_minimum_error(node.left)
        right_error = self._prune_minimum_error(node.right)

        dynamic_error = (
            node.left.number_of_samples / node.number_of_samples
        ) * left_error + (
            node.right.number_of_samples / node.number_of_samples
        ) * right_error

        if static_error <= dynamic_error:
            self._prune_node(node)
            node.expected_error = static_error
        else:
            node.expected_error = dynamic_error

        return node.expected_error

    def _static_error(
        self, node, m=3, minimum_error=MinimumError.MINIMUM_ERROR_ESTIMATE
    ):
        error_estimate = 0.0
        N = node.number_of_samples

        match (minimum_error):
            case self.MinimumError.M_PROBABILITY:
                priors = self.class_priors_
                number_of_classes = node.number_of_classes
                smoothed_probabilities = (number_of_classes + m * priors) / (N + m)
                error_estimate = 1 - max(smoothed_probabilities)
            case self.MinimumError.MINIMUM_ERROR_ESTIMATE:
                number_of_classes = len(self.y_train_)
                number_of_majority_class = np.sum(self.y_train_ == node.majority_class)
                error_estimate = (
                    N - number_of_majority_class + number_of_classes - 1
                ) / (N + number_of_classes)
            case _:
                raise ValueError(
                    f"Unsupported {minimum_error}, supported values are {list(self.MinimumError)}"
                )
        return error_estimate

    def _class_counts(self, y):
        counts = np.zeros(len(self.classes_))
        for i, cls in enumerate(self.classes_):
            counts[i] = np.sum(y == cls)
        return counts

    def _prune_node(self, node):
        node.is_leaf = True
        node.number_of_leaves = 1
        node.subtree_error = node.leaf_error
        node.left = None
        node.right = None
        node.feature = None
        node.threshold = None
        node.value = node.majority_class

    def predict_one(self, x, node=None, verbose=False, leaf_node=False):
        if node is None:
            node = self.root

        if node.value is not None:
            if leaf_node:
                return node
            return node.value

        if self.categorical:
            if x[node.feature] == node.threshold:
                return self.predict_one(x, node.left, leaf_node=leaf_node)
            else:
                return self.predict_one(x, node.right, leaf_node=leaf_node)
        else:
            if x[node.feature] > node.threshold:
                return self.predict_one(x, node.left, leaf_node=leaf_node)
            else:
                return self.predict_one(x, node.right, leaf_node=leaf_node)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def predict_probability(self, X):
        return np.array([self.predict_probability_one(x) for x in X])

    def predict_probability_one(self, X):
        leaf = self.predict_one(X, leaf_node=True)
        return leaf.number_of_classes / leaf.number_of_samples

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def evaluate_dataset(self, X, y):
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        predictions = np.asarray(self.predict(X))
        accuracy = np.mean(predictions == y) * 100.0

        if self.tree_type is self.TreeType.REGRESSION:
            match self.error:
                case self.Error.MSE:
                    return self.mse(y, predictions)
                case self.Error.RMSE:
                    return self.rmse(y, predictions)
                case self.Error.R2:
                    return self.r2(y, predictions)
                case _:
                    raise ValueError(
                        f"Unsupported {self.error}, supported values are {list(self.Error)}"
                    )
        return predictions, accuracy

    def permutation_importance(
        self, X, y, feature_names=None, n_repeats=50, seed=42, verbose=False
    ):
        _, baseline_accuracy = self.evaluate_dataset(X, y)
        rng = self.rng

        permutation_importance_averages = {}
        permutation_importance_standart_deviations = {}

        for feature in range(X.shape[1]):
            drop_accuracies = []

            for _ in range(n_repeats):
                X_shuffled = X.copy()
                rng.shuffle(X_shuffled[:, feature])
                _, shuffled_accuracy = self.evaluate_dataset(X_shuffled, y)
                drop_accuracies.append(baseline_accuracy - shuffled_accuracy)
            feature_name = (
                feature_names[feature]
                if feature_names is not None
                else f"feature{feature}"
            )
            permutation_importance_averages[f"{feature_name}_avg"] = np.average(
                drop_accuracies
            )
            permutation_importance_standart_deviations[f"{feature_name}_std"] = (
                np.std(drop_accuracies),
            )
        if verbose:
            print(f"\nPermutation Importances : ")
            for key_avg, value_avg in permutation_importance_averages.items():
                print(f"feature : {key_avg}, importance is {value_avg}")
            print()
            for (
                key_std,
                value_std,
            ) in permutation_importance_standart_deviations.items():
                print(f"feature : {key_std}, standart deviation is {value_std}")
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

            print("\nstd_values:", std_values)
            print("\nvalues_std:", values_std)

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

            os.makedirs("img", exist_ok=True)

            plt.savefig("img/prmttn_mprtncs.png")
            plt.show()
        return (
            permutation_importance_averages,
            permutation_importance_standart_deviations,
        )

    def visualize_tree(self, feature1, feature2, cmap=plt.cm.Set1):
        os.makedirs("img", exist_ok=True)

        X_train = self.X_train_[:, [feature1, feature2]]
        y_train = self.y_train_

        x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
        y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()

        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(X, Y)

        baseline = self.X_train_.mean(axis=0)
        grid_points = np.tile(baseline, (XX.ravel().shape[0], 1))

        # grid_points = np.c_[XX.ravel(), YY.ravel()]
        grid_points[:, feature1] = XX.ravel()
        grid_points[:, feature2] = YY.ravel()

        Z = self.predict(grid_points).reshape(XX.shape)

        plt.title("Scatter Plot of features and true labels")
        plt.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors="black"
        )
        plt.savefig("img/scatter.png")
        plt.show()

        plt.title("Contour plot of the splits")
        levels = np.arange(len(self.classes_) + 1) - 0.5
        plt.contourf(XX, YY, Z, levels=levels, alpha=0.3, cmap=cmap)
        plt.savefig("img/contour.png")
        plt.show()

        plt.title("Spliting of the instance Space")
        plt.pcolormesh(XX, YY, Z, cmap=cmap, shading="auto")
        plt.savefig("img/bassins.png")
        plt.show()

    def export_text(self, node, feature_names=None, class_names=None, text=""):
        if node.depth > node.max_depth:
            return node.max_depth_reached
        if node.is_root:
            text += "root"
        padding = "    " + "|   " * node.depth

        if node.is_leaf or node.value is not None:
            root_str = "root" if node.is_root else ""
            if class_names is not None:
                return f"{root_str}\n{padding}+---> class: {class_names[node.majority_class]} samples: {node.number_of_samples} value: {node.number_of_classes}"
            else:
                return f"{root_str}\n{padding}+---> class: {node.majority_class} samples: {node.number_of_samples} value: {node.number_of_classes}"
        else:
            feature_name = (
                feature_names[node.feature]
                if feature_names is not None
                else node.feature
            )
            if self.categorical:
                text += f"\n{padding}+---> {feature_name} == {node.threshold}"
                text += self.export_text(node.left, feature_names, class_names)
                text += f"\n{padding}+---> {feature_name} != {node.threshold}"
                text += self.export_text(node.right, feature_names, class_names)
            else:
                text += f"\n{padding}+---> {feature_name} > {node.threshold}"
                text += self.export_text(node.left, feature_names, class_names)
                text += f"\n{padding}+---> {feature_name} <= {node.threshold}"
                text += self.export_text(node.right, feature_names, class_names)
        return text

    @classmethod
    def generate_config(cls, max_depth, max_gain=None):
        configs = []
        for max_depth in [i for i in range(max_depth + 1)]:
            for minimum_gain in [0.0, 0.001, 0.01, 0.05]:
                for information_gain in [
                    cls.InformationGain.GINI,
                    cls.InformationGain.ENTROPY,
                ]:
                    configs.append(
                        {
                            "max_depth": max_depth,
                            "minimum_gain": minimum_gain,
                            "information_gain": information_gain,
                        }
                    )
        return configs

    @staticmethod
    def validate_tree(configs):
        best_tree = None

        best_config = None
        best_val_accuracy = -1

        for config in configs:
            tree = ANonSeriousDecisionTree(
                minimum_population_size=2,
                minimum_split_size=1,
                minimum_gain=config["minimum_gain"],
                max_depth=config["max_depth"],
                categorical=False,
                adjacent=True,
                information_gain=config["information_gain"],
            )

            tree.fit(X_train, y_train)

            _, train_accuracy = tree.evaluate_dataset(X_train, y_train)
            _, val_accuracy = tree.evaluate_dataset(X_val, y_val)

            print(config, f"Train: {train_accuracy:.2f}%", f"Val: {val_accuracy:.2f}%")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_tree = tree
                best_config = config
        return best_val_accuracy, best_tree, best_config

    @staticmethod
    def cross_validate_tree(X, y, config, splits=5):
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

        scores = []

        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            tree = ANonSeriousDecisionTree(
                minimum_population_size=2,
                minimum_split_size=1,
                minimum_gain=config["minimum_gain"],
                max_depth=config["max_depth"],
                categorical=False,
                adjacent=True,
                information_gain=config["information_gain"],
            )

            tree.fit(X_train, y_train)
            scores.append(tree.evaluate_dataset(X_val, y_val)[1])
        return np.mean(scores)

    @classmethod
    def choose_best_cross_validation(
        cls,
        X_train_val,
        y_train_val,
        configs,
        splits=5,
        categorical=False,
        adjacent=True,
        log=False,
        verbose=False,
    ):
        best_config = None
        best_cv_score = -1
        loading = "."
        if log:
            i = 0
            space = " "
            loading_bar_length = 10
            padding = 10
            print()
        for config in configs:
            cv_score = cls.cross_validate_tree(
                X_train_val, y_train_val, config, splits=splits
            )
            if log:
                progression = i % loading_bar_length
                print(
                    f"Choosing the best Configuration{loading*progression + space*(loading_bar_length - progression)} {config} - [ CV Score: {cv_score:.2f}% ]{space*padding}",
                    end="\r",
                )
                i += 1
            if cv_score > best_cv_score:
                best_cv_score = cv_score
                best_config = config
        if verbose:
            print(f"\n\nBest config: {best_config} Best CV score: {best_cv_score}\n")

        final_tree = ANonSeriousDecisionTree(
            minimum_population_size=2,
            minimum_split_size=1,
            minimum_gain=best_config["minimum_gain"],
            max_depth=best_config["max_depth"],
            str_max_depth=float("inf"),
            categorical=categorical,
            adjacent=adjacent,
            log=log,
            information_gain=best_config["information_gain"],
        )
        final_tree.fit(X_train_val, y_train_val, verbose=verbose)
        return final_tree

    @staticmethod
    def circle_of_clouds(
        n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0
    ):
        rng = np.random.default_rng(seed)
        if not sigma:
            sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

        def rotate(x, k):
            theta = 2 * k * np.pi / n_clouds + angle
            m = np.matrix(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            return np.matmul(x, m)

        def cloud():
            return (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(
                n_objects_by_cloud, 2
            ) + np.array([radius, 0])

        def target():
            return np.array(
                ([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32"
            ).ravel()

        return (
            np.concatenate(
                [np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0
            ),
            target(),
        )


class ANonSeriousBaggingTrees:

    def __init__(
        self,
        number_of_trees=100,
        max_depth=10,
        minimum_population_size=1,
        seed=0,
        minimum_gain=0.001,
        categorical=False,
        adjacent=False,
        information_gain=None,
    ):
        self.X_train_ = None
        self.y_train_ = None
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.minimum_population_size = minimum_population_size
        self.seed = seed
        self.minimum_gain = minimum_gain
        self.categorical = categorical
        self.adjacent = adjacent
        self.information_gain = information_gain

    def fit(self, X, y, verbose=False):
        self.X_train_ = X
        self.y_train_ = y
        self.trees = []
        self.bootstrap_indices_ = []
        self.oob_indices_ = []
        self.classes_, class_counts = np.unique(y, return_counts=True)

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(self.number_of_trees):
            tree = ANonSeriousDecisionTree(
                max_depth=self.max_depth,
                minimum_population_size=self.minimum_population_size,
                minimum_gain=self.minimum_gain,
                categorical=self.categorical,
                adjacent=self.adjacent,
                information_gain=self.information_gain,
                seed=self.seed + i,
            )
            rng = np.random.default_rng(self.seed + i)
            X_bootstrap, y_bootstrap, indices, oob_indices = self.bootstrap(X, y, rng)
            self.bootstrap_indices_.append(indices)
            self.oob_indices_.append(oob_indices)
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            _, accuracy = tree.evaluate_dataset(tree.X_train_, tree.y_train_)
            accuracies.append(accuracy)
        if verbose:
            _, bagging_accuracy = self.evaluate_bagging_trees(X, y)
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the Bagging on td   : {bagging_accuracy}""")

        return self

    def bootstrap(self, X, y, rng):
        number_of_samples = len(X)
        indices = rng.choice(len(X), len(X), replace=True)
        in_bag_mask = np.zeros(number_of_samples, dtype=np.bool_)
        in_bag_mask[indices] = True
        out_of_bag_mask = ~in_bag_mask
        oob_indices = np.where(out_of_bag_mask)[0]
        return X[indices], y[indices], indices, oob_indices

    def oob_evaluation(self, verbose=False):
        votes = [[] for _ in self.X_train_]
        for tree, oob_indices in zip(self.trees, self.oob_indices_):
            predictions = tree.predict(self.X_train_[oob_indices])
            for sample_index, prediction in zip(oob_indices, predictions):
                votes[sample_index].append(prediction)
        oob_predictions = []
        oob_true_labels = []
        for sample_index, sample_votes in enumerate(votes):
            if len(sample_votes) == 0:
                continue
            values, counts = np.unique(sample_votes, return_counts=True)
            majority_prediction = values[np.argmax(counts)]

            oob_predictions.append(majority_prediction)
            oob_true_labels.append(self.y_train_[sample_index])
        evaluated_samples = len(oob_predictions)
        oob_accuracy = (
            np.mean(np.array(oob_predictions) == np.array(oob_true_labels)) * 100
            if evaluated_samples > 0
            else None
        )

        if verbose:
            total_samples = len(self.X_train_)
            print(f"\nOOB accuracy: {oob_accuracy}")
            print(f"\nSamples evaluated: {evaluated_samples}/{total_samples}")
        return oob_accuracy

    def evaluate_bagging_trees(self, X, y):
        predictions = self.predict(X)
        accuracies = np.mean(predictions == y) * 100.0
        return predictions, accuracies

    def predict(self, X):
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []

        for sample_predictions in all_predictions.T:
            values, counts = np.unique(sample_predictions, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])

        return np.array(final_predictions)

    def visualize_tree(self, feature1, feature2, cmap=plt.cm.Set1):
        os.makedirs("img", exist_ok=True)

        X_train = self.X_train_[:, [feature1, feature2]]
        y_train = self.y_train_

        x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
        y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()

        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(X, Y)

        baseline = self.X_train_.mean(axis=0)
        grid_points = np.tile(baseline, (XX.ravel().shape[0], 1))

        # grid_points = np.c_[XX.ravel(), YY.ravel()]
        grid_points[:, feature1] = XX.ravel()
        grid_points[:, feature2] = YY.ravel()

        Z = self.predict(grid_points).reshape(XX.shape)

        plt.title("Forest Scatter Plot of features and true labels")
        plt.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors="black"
        )
        plt.savefig("img/frst_scatter.png")
        plt.show()

        plt.title("Bagging Contour plot of the splits")
        levels = np.arange(len(self.classes_) + 1) - 0.5
        plt.contourf(XX, YY, Z, levels=levels, alpha=0.3, cmap=cmap)
        plt.savefig("img/bggng_contour.png")
        plt.show()

        plt.title("Bagging Spliting of the instance Space")
        plt.pcolormesh(XX, YY, Z, cmap=cmap, shading="auto")
        plt.savefig("img/bassins_bggng_trs.png")
        plt.show()

    @staticmethod
    def circle_of_clouds(
        n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0
    ):
        rng = np.random.default_rng(seed)
        if not sigma:
            sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

        def rotate(x, k):
            theta = 2 * k * np.pi / n_clouds + angle
            m = np.matrix(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            return np.matmul(x, m)

        def cloud():
            return (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(
                n_objects_by_cloud, 2
            ) + np.array([radius, 0])

        def target():
            return np.array(
                ([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32"
            ).ravel()

        return (
            np.concatenate(
                [np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0
            ),
            target(),
        )


class ANonSeriousRandomForest:

    class ForestType(Enum):
        CLASSIFICATION = 1
        REGRESSION = 2

    class Voting(Enum):
        HARD = 1
        SOFT = 2

    class Error(Enum):
        MSE = 1
        SSE = 2
        RMSE = 3
        R2 = 4

    def __init__(
        self,
        number_of_trees=100,
        max_depth=5,
        minimum_population_size=2,
        seed=0,
        minimum_gain=0.05,
        categorical=False,
        adjacent=False,
        information_gain=None,
        _bootstrap=True,
        random_criterion=None,
        forest_type=None,
        voting=Voting.HARD,
        error=Error.MSE,
    ):
        self.X_train_ = None
        self.y_train_ = None
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.minimum_population_size = minimum_population_size
        self.seed = seed
        self.minimum_gain = minimum_gain
        self.categorical = categorical
        self.adjacent = adjacent
        self.information_gain = information_gain
        self.random_criterion = random_criterion
        self._bootstrap = _bootstrap
        self.forest_type = forest_type
        self.voting = voting
        self.error = error

    def fit(self, X, y, verbose=False):
        self.X_train_ = X
        self.y_train_ = y
        self.trees = []
        self.bootstrap_indices_ = []
        self.oob_indices_ = []
        self.classes_, class_counts = np.unique(y, return_counts=True)

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        tree_type = (
            ANonSeriousDecisionTree.TreeType.REGRESSION
            if self.forest_type is self.ForestType.REGRESSION
            else ANonSeriousDecisionTree.TreeType.CLASSIFICATION
        )

        for i in range(self.number_of_trees):
            tree = ANonSeriousDecisionTree(
                max_depth=self.max_depth,
                minimum_population_size=self.minimum_population_size,
                minimum_gain=self.minimum_gain,
                categorical=self.categorical,
                adjacent=self.adjacent,
                information_gain=self.information_gain,
                random_criterion=self.random_criterion,
                seed=self.seed + i,
                tree_type=tree_type,
            )

            if self._bootstrap:
                rng = np.random.default_rng(self.seed + i)
                X_bootstrap, y_bootstrap, indices, oob_indices = self.bootstrap(
                    X, y, rng
                )
                self.bootstrap_indices_.append(indices)
                self.oob_indices_.append(oob_indices)

                tree.fit(X_bootstrap, y_bootstrap)
            else:
                tree.fit(X, y)

            self.trees.append(tree)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            match self.forest_type:
                case self.ForestType.CLASSIFICATION:
                    _, accuracy = tree.evaluate_dataset(tree.X_train_, tree.y_train_)
                    accuracies.append(accuracy)
                case self.ForestType.REGRESSION:
                    metrics = tree.evaluate_dataset(tree.X_train_, tree.y_train_)
                case _:
                    raise ValueError(
                        f"Unsupported {self.forest_type}, supported values are {list(self.ForestType)}"
                    )
        if verbose:
            match self.forest_type:
                case self.ForestType.CLASSIFICATION:
                    _, forest_accuracy = self.evaluate_random_forest(X, y)
                case self.ForestType.REGRESSION:
                    evaluation = self.evaluate_random_forest(X, y)
                case _:
                    raise ValueError(
                        f"Unsupported {self.forest_type}, supported values are {list(self.ForestType)}"
                    )
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")
            match self.forest_type:
                case self.ForestType.CLASSIFICATION:
                    print(f"Mean Accuracy : {accuracies}")
                    print(f"Accuracy of the forest on td   : {forest_accuracy}")
                case self.ForestType.REGRESSION:
                    print(f"Regression Metrics {metrics}")
                    print(f"Evaluation {evaluation}")
                case _:
                    raise ValueError(
                        f"Unsupported {self.forest_type}, supported values are {list(self.ForestType)}"
                    )
        return self

    def bootstrap(self, X, y, rng):
        number_of_samples = len(X)
        indices = rng.choice(len(X), len(X), replace=True)
        in_bag_mask = np.zeros(number_of_samples, dtype=np.bool_)
        in_bag_mask[indices] = True
        out_of_bag_mask = ~in_bag_mask
        oob_indices = np.where(out_of_bag_mask)[0]
        return X[indices], y[indices], indices, oob_indices

    def oob_evaluation(self, verbose=False):
        if not self._bootstrap:
            raise RuntimeError("OOB evaluation requires _bootstrap=True")

        votes = [[] for _ in self.X_train_]
        for tree, oob_indices in zip(self.trees, self.oob_indices_):
            predictions = tree.predict(self.X_train_[oob_indices])
            for sample_index, prediction in zip(oob_indices, predictions):
                votes[sample_index].append(prediction)
        oob_predictions = []
        oob_true_labels = []

        for sample_index, sample_votes in enumerate(votes):
            if len(sample_votes) == 0:
                continue
            match self.forest_type:
                case self.ForestType.CLASSIFICATION:
                    values, counts = np.unique(sample_votes, return_counts=True)
                    majority_prediction = values[np.argmax(counts)]
                case self.ForestType.REGRESSION:
                    majority_prediction = np.mean(sample_votes)
                case _:
                    raise ValueError(
                        f"Unsupported {self.forest_type}, supported values are {list(self.ForestType)}"
                    )
            oob_predictions.append(majority_prediction)
            oob_true_labels.append(self.y_train_[sample_index])
        oob_true_labels = np.array(oob_true_labels)
        oob_predictions = np.array(oob_predictions)
        match self.forest_type:
            case self.ForestType.CLASSIFICATION:
                evaluated_samples = len(oob_predictions)
                oob_accuracy = (
                    np.mean(oob_predictions == oob_true_labels)
                    * 100
                    if evaluated_samples > 0
                    else None
                )
                if verbose:
                    total_samples = len(self.X_train_)
                    print(
                        f"\nOOB accuracy: {oob_accuracy} Samples evaluated: {evaluated_samples}/{total_samples}\n"
                    )
                return oob_accuracy
            case self.ForestType.REGRESSION:
                oob_mse = self.mse(oob_true_labels, oob_predictions)
                oob_rmse = self.rmse(oob_true_labels, oob_predictions)
                oob_r2 = self.r2(oob_true_labels, oob_predictions)
                if verbose:
                    print(f"OOB MSE : {oob_mse}, RMSE : {oob_rmse}, R2 : {oob_r2}")
                return oob_mse, oob_rmse, oob_r2
            case _:
                raise ValueError(
                    f"Unsupported {self.forest_type}, supported values are {list(self.ForestType)}"
                )

    def evaluate_random_forest(self, X, y):
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        predictions = self.predict(X)
        accuracies = np.mean(predictions == y) * 100.0

        if self.forest_type is self.ForestType.REGRESSION:
            match self.error:
                case self.Error.MSE:
                    return self.mse(y, predictions)
                case self.Error.RMSE:
                    return self.rmse(y, predictions)
                case self.Error.R2:
                    return self.r2(y, predictions)
                case _:
                    raise ValueError(
                        f"Unsupported {self.error}, supported values are {list(self.Error)}"
                    )
        return predictions, accuracies

    def predict(self, X):
        match self.forest_type:
            case self.ForestType.CLASSIFICATION:
                probabilities = self.predict_probabilities(X)
                class_indices = np.argmax(probabilities, axis=1)
                return self.classes_[class_indices]
            case self.ForestType.REGRESSION:
                all_predictions = np.array([tree.predict(X) for tree in self.trees])
                return np.mean(all_predictions, axis=0)
            case _:
                raise ValueError(
                    f"Unsupported {self.ForestType}, supported values are {list(self.ForestType)}"
                )
        return np.array(final_predictions)

    def predict_probabilities(self, X):
        match self.voting:
            case self.Voting.HARD:
                all_predictions = np.array([tree.predict(X) for tree in self.trees])
                probability_matrix = np.zeros((len(X), len(self.classes_)))
                class_indexes = {clazz: i for i, clazz in enumerate(self.classes_)}

                for i, sample_predictions in enumerate(all_predictions.T):
                    votes = np.zeros(len(self.classes_))

                    for sample_prediction in sample_predictions:
                        class_index = class_indexes[sample_prediction]
                        votes[class_index] += 1
                    probability_matrix[i] = votes / len(self.trees)
                return probability_matrix
            case self.Voting.SOFT:
                all_probabilities = np.array(
                    [tree.predict_probability(X) for tree in self.trees]
                )
                return np.mean(all_probabilities, axis=0)
            case _:
                raise ValueError(
                    f"Unsupported {self.voting}, supported values are {list(self.Voting)}"
                )

    def sse(self, y, y_hat):
        return np.sum((y - y_hat) ** 2)

    def mse(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def rmse(self, y, y_hat):
        return np.sqrt(self.mse(y, y_hat))

    def r2(self, y, y_hat):
        ss_res = np.sum((y - y_hat) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        if ss_total == 0:
            return 0
        return 1 - ss_res / ss_total

    def visualize_tree(self, feature1, feature2, cmap=plt.cm.Set1):
        os.makedirs("img", exist_ok=True)

        X_train = self.X_train_[:, [feature1, feature2]]
        y_train = self.y_train_

        x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
        y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()

        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(X, Y)

        baseline = self.X_train_.mean(axis=0)
        grid_points = np.tile(baseline, (XX.ravel().shape[0], 1))

        # grid_points = np.c_[XX.ravel(), YY.ravel()]
        grid_points[:, feature1] = XX.ravel()
        grid_points[:, feature2] = YY.ravel()

        Z = self.predict(grid_points).reshape(XX.shape)

        plt.title("Forest Scatter Plot of features and true labels")
        plt.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors="black"
        )
        plt.savefig("img/frst_scatter.png")
        plt.show()

        plt.title("Forest Contour plot of the splits")
        levels = np.arange(len(self.classes_) + 1) - 0.5
        plt.contourf(XX, YY, Z, levels=levels, alpha=0.3, cmap=cmap)
        plt.savefig("img/frst_contour.png")
        plt.show()

        plt.title("Forest Spliting of the instance Space")
        plt.pcolormesh(XX, YY, Z, cmap=cmap, shading="auto")
        plt.savefig("img/bassins_rndm_frst.png")
        plt.show()

    @staticmethod
    def circle_of_clouds(
        n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0
    ):
        rng = np.random.default_rng(seed)
        if not sigma:
            sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

        def rotate(x, k):
            theta = 2 * k * np.pi / n_clouds + angle
            m = np.matrix(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            return np.matmul(x, m)

        def cloud():
            return (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(
                n_objects_by_cloud, 2
            ) + np.array([radius, 0])

        def target():
            return np.array(
                ([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32"
            ).ravel()

        return (
            np.concatenate(
                [np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0
            ),
            target(),
        )


if __name__ == "__main__":

    # dataset = datasets.load_iris()
    seed = np.random.randint(0, 100)
    X, y = ANonSeriousDecisionTree.circle_of_clouds(10, 30, seed=seed)
    feature1 = 0
    feature2 = 1

    # X = dataset.data[:, [feature_x, feature_y]]
    # X = dataset.data
    # y = dataset.target
    
    # X, y = make_regression(
    #     n_samples=300,
    #     n_features=2,
    #     noise=15,
    #     random_state=42,
    # )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # 60% training
    # 20% validation
    # 20% test

    # configs = ANonSeriousDecisionTree.generate_config(max_depth=10)

    # final_tree = ANonSeriousDecisionTree.choose_best_cross_validation(
    #     X_train_val, y_train_val, configs, log=True, verbose=False
    # )
    # _, final_test_accuracy = final_tree.evaluate_dataset(X_test, y_test)

    # print(f"\nFinal Test Accuracy: {final_test_accuracy}%")

    # # print("\nPruning the tree...")
    # # final_tree.prune_minimum_error()

    # print(
    #     "After pruning Validation Accuracy:",
    #     final_tree.evaluate_dataset(X_val, y_val)[1],
    # )
    # print(
    #     "Final test accuracy:",
    #     final_tree.evaluate_dataset(X_test, y_test)[1],
    # )

    # tree_str = str(final_tree)
    # print(f"\n{tree_str}")

    # # tree_vis = final_tree.export_text(
    # #     final_tree.root,
    # #     feature_names=dataset.feature_names,
    # #     class_names=dict(zip(np.unique(dataset.target), dataset.target_names)),
    # # )
    # tree_vis = final_tree.export_text(final_tree.root)
    # print(f"{tree_vis}")

    # # final_tree.permutation_importance(X_test, y_test, dataset.feature_names)
    # final_tree.permutation_importance(X_test, y_test, verbose=True)

    # final_tree.visualize_tree(feature1, feature2)

    # print("\nInitializing Bagging Trees...")
    # bagging_trees = ANonSeriousBaggingTrees(
    #     information_gain=ANonSeriousDecisionTree.InformationGain.GINI,
    # )
    # print("\nFitting the Bagging Trees :")
    # bagging_trees.fit(X_train, y_train, verbose=True)
    # _, bagging_accuracy = bagging_trees.evaluate_bagging_trees(X_test, y_test)
    # print(f"Test Accuracy : {bagging_accuracy}")

    # bagging_trees.visualize_tree(feature1, feature2)

    # tree = ANonSeriousDecisionTree(
    #     max_depth=5,
    #     minimum_population_size=5,
    #     minimum_gain=0.0,
    #     seed=42,
    #     tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
    # )

    # tree.fit(X_train, y_train)

    # predictions = tree.predict(X_test)
    # print(f"Predictions : {predictions}")
    # print("\nRegression Tree evaluation : ")
    # print(tree.evaluate_dataset(X_test, y_test))

    print("\nInitializing a Random Forest...")
    random_forest = ANonSeriousRandomForest(
        information_gain=ANonSeriousDecisionTree.InformationGain.ENTROPY,
        random_criterion=ANonSeriousDecisionTree.RandomCriterion.RANDOM_SPLIT,
        _bootstrap=True,
        forest_type=ANonSeriousRandomForest.ForestType.REGRESSION,
        voting=ANonSeriousRandomForest.Voting.HARD,
    )

    print("\nFitting the Random Forest...")
    limit = 5
    random_forest.fit(X_train, y_train, verbose=True)

    if random_forest.forest_type is ANonSeriousRandomForest.ForestType.CLASSIFICATION:
        _, forest_accuracy = random_forest.evaluate_random_forest(X_test, y_test)
        probabilities = random_forest.predict_probabilities(X_test[:limit])
        print(f"\nTest Accuracy : {forest_accuracy}")
        print(f"{probabilities.sum(axis=1)}\nProbabilities {probabilities}")
    else:
        evaluation = random_forest.evaluate_random_forest(X_test, y_test)
        predictions = random_forest.predict(X_test)
        print(f"\nEvaluation {evaluation}")
        print(f"Predictions {predictions}")

    print("\nOOB Evaluation : ")
    random_forest.oob_evaluation(verbose=True)

    random_forest.visualize_tree(feature1, feature2)
