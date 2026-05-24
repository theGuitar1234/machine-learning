from dataclasses import dataclass
from enum import Enum
from typing import override

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

from a_non_serious_node import ANonSeriousNode


class ANonSeriousIsolationRandomTree:

    def __init__(
        self,
        max_depth=10,
        seed=0,
        minimum_population_size=1,
        vectorized=False,
        categorical=False,
        categorical_features=None,
    ):
        if vectorized and (categorical or categorical_features is not None):
            raise RuntimeError("Categorical data can't be used for vectorized search")
        self.rng = np.random.default_rng(seed)
        self.root = None
        self.max_depth = max_depth
        self.minimum_population_size = minimum_population_size

        self.vectorized = vectorized
        self.categorical = categorical
        self.categorical_features = categorical_features

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_node(self, node, sub_population):
        _node = ANonSeriousNode()
        _node.depth = node.depth + 1
        _node.sub_population = sub_population
        return _node

    def get_leaves(self):
        return self.root.get_leaves_below()

    @override
    def fit(self, X, verbose=False):
        self.X_train_ = X
        self.root = ANonSeriousNode()
        self.root.is_root = True
        self.root.depth = 0
        self.root.sub_population = np.ones(self.X_train_.shape[0], dtype=np.bool_)

        if self.categorical_features is not None:
            self.categorical_features_ = set(self.categorical_features)
        elif self.categorical:
            self.categorical_features_ = set(range(self.X_train_.shape[1]))
        else:
            self.categorical_features_ = set()

        self.fit_node(self.root)
        if verbose:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")

        return self

    @override
    def fit_node(self, node):
        population_size = int(np.sum(node.sub_population))

        if (
            population_size <= self.minimum_population_size
            or node.depth >= self.max_depth
        ):
            return self._prune_node(node)

        feature, threshold, categorical_split = self.random_split_criterion(node)

        if feature is None:
            return self._prune_node(node)
        node.feature, node.threshold, node.categorical_split = (
            feature,
            threshold,
            categorical_split,
        )

        feat_col = self.X_train_[:, node.feature]
        if node.categorical_split:
            go_left = feat_col == node.threshold
        else:
            go_left = feat_col.astype(float) > node.threshold
        left_population = node.sub_population & go_left
        right_population = node.sub_population & (~go_left)
        child_depth = node.depth + 1

        is_left_leaf = (
            child_depth >= self.max_depth
            or np.sum(left_population) <= self.minimum_population_size
        )
        is_right_leaf = (
            child_depth >= self.max_depth
            or np.sum(right_population) <= self.minimum_population_size
        )

        if is_left_leaf:
            node.left = self.get_leaf(node, left_population)
        else:
            node.left = self.get_node(node, left_population)
            self.fit_node(node.left)
        if is_right_leaf:
            node.right = self.get_leaf(node, right_population)
        else:
            node.right = self.get_node(node, right_population)
            self.fit_node(node.right)

    def random_split_criterion(self, node):
        X_node = self.X_train_[node.sub_population]

        valid_features = []

        for feature in range(X_node.shape[1]):
            values = X_node[:, feature]

            if feature in self.categorical_features_:
                if np.unique(values).size > 1:
                    valid_features.append(feature)
            else:
                numeric_values = values.astype(float)
                if np.ptp(numeric_values) > 0:
                    valid_features.append(feature)

        if len(valid_features) == 0:
            return None, None, None

        feature = self.rng.choice(valid_features)
        values = X_node[:, feature]

        if feature in self.categorical_features_:
            categories = np.unique(values)
            threshold = self.rng.choice(categories)
            categorical_split = True
        else:
            numeric_values = values.astype(float)
            threshold = self.rng.uniform(np.min(numeric_values), np.max(numeric_values))
            categorical_split = False

        return feature, threshold, categorical_split

    def _prune_node(self, node):
        node.is_leaf = True
        node.value = node.depth
        node.number_of_samples = int(np.sum(node.sub_population))
        node.left = None
        node.right = None
        node.feature = None
        node.threshold = None
        return node

    def predict_one(self, x, node=None, verbose=False, leaf_node=False):
        if node is None:
            node = self.root

        if node.value is not None:
            if leaf_node:
                return node
            return node.value

        if node.categorical_split:
            if x[node.feature] == node.threshold:
                return self.predict_one(x, node.left, leaf_node=leaf_node)
            return self.predict_one(x, node.right, leaf_node=leaf_node)

        if float(x[node.feature]) > node.threshold:
            return self.predict_one(x, node.left, leaf_node=leaf_node)

        return self.predict_one(x, node.right, leaf_node=leaf_node)

    def predict(self, X):
        if self.vectorized:
            return self.vectorized_predict_search(X)
        return np.array([self.predict_one(x) for x in X])

    def vectorized_predict_search(self, X):
        self._update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        values = np.array([leaf.value for leaf in leaves], dtype=float)
        indicators = np.array([leaf.indicator(X) for leaf in leaves], dtype=float)
        if not np.all(np.sum(indicators, axis=0) == 1):
            raise RuntimeError("All column sums in the matrix must be equal to 1")
        return values @ indicators

    def _update_bounds(self):
        self.root.update_bounds_below()

    def get_leaf(self, node, sub_population):
        leaf = ANonSeriousNode()
        leaf.is_leaf = True
        leaf_depth = node.depth + 1
        leaf.depth = leaf_depth
        leaf.value = leaf_depth
        leaf.sub_population = sub_population
        leaf.number_of_samples = sum(sub_population)
        return leaf

    def path_length(self, X):
        return np.array([self.path_length_one(x) for x in X])

    def path_length_one(self, x, node=None):
        if node is None:
            node = self.root

        if node.is_leaf or node.value is not None:
            return node.depth + self.c(node.number_of_samples)

        if node.categorical_split:
            if x[node.feature] == node.threshold:
                return self.path_length_one(x, node.left)
            return self.path_length_one(x, node.right)

        if float(x[node.feature]) > node.threshold:
            return self.path_length_one(x, node.left)

        return self.path_length_one(x, node.right)

    def c(self, n):
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        euler_gamma = 0.5772156649
        return 2.0 * (np.log(n - 1) + euler_gamma) - (2.0 * (n - 1) / n)

    @staticmethod
    def visualize_tree(X, min=1, max=9, max_depth=10, cmap=plt.cm.Set1):
        os.makedirs("img", exist_ok=True)

        def visualize_bassins(ax, model, x_min, x_max, y_min, y_max, cmap):
            assert model.X_train_.shape[1] == 2, "Not a 2D example"
            X = np.linspace(x_min, x_max, 100)
            Y = np.linspace(y_min, y_max, 100)
            XX, YY = np.meshgrid(X, Y)
            XX_flat = XX.ravel()
            YY_flat = YY.ravel()
            Z = model.predict(np.vstack([XX_flat, YY_flat]).T)
            ax.pcolormesh(XX, YY, Z.reshape([100, 100]), cmap=cmap, shading="auto")

        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        axes[0, 0].scatter(X[:, 0], X[:, 1])
        axes[0, 0].set_title("a cloud and an outlier")
        for i in range(min, max):
            T = ANonSeriousIsolationRandomTree(max_depth=max_depth, seed=i)
            T.fit(X)
            visualize_bassins(
                axes[i % 3, i // 3], T, -1.2, 1.5, -0.5, 0.5, cmap=plt.cm.RdBu
            )
            axes[i % 3, i // 3].set_title(
                f"Bassins of the isolation tree for seed={i}", fontsize=6
            )
        plt.savefig("img/sltn_bassins.png")
        plt.show()
