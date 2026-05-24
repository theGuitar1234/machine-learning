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

from a_non_serious_isolation_random_tree import ANonSeriousIsolationRandomTree


class ANonSeriousIsolationRandomForest:

    def __init__(
        self,
        number_of_trees=100,
        max_depth=10,
        minimum_population_size=1,
        seed=0,
        vectorized=False,
        sample_size=256,
        categorical=False,
        categorical_features=None,
    ):
        if vectorized and (categorical or categorical_features is not None):
            raise RuntimeError("Categorical data can't be used for vectorized search")
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.seed = seed
        self.vectorized = vectorized
        self.rng = np.random.default_rng(seed=seed)
        self.sample_size = sample_size
        self.minimum_population_size = minimum_population_size
        self.categorical = categorical
        self.categorical_features = categorical_features

    def fit(self, X, verbose=False):
        self.X_train_ = X
        self.trees = []

        number_of_samples = self.X_train_.shape[0]
        if self.categorical_features is not None:
            self.categorical_features_ = set(self.categorical_features)
        elif self.categorical:
            self.categorical_features_ = set(range(self.X_train_.shape[1]))
        else:
            self.categorical_features_ = set()

        depths = []
        nodes = []
        leaves = []

        for i in range(self.number_of_trees):
            tree = ANonSeriousIsolationRandomTree(
                max_depth=self.max_depth,
                seed=self.seed + i,
                vectorized=self.vectorized,
                minimum_population_size=self.minimum_population_size,
                categorical=self.categorical,
                categorical_features=self.categorical_features_,
            )
            index = self.rng.choice(
                number_of_samples,
                size=min(self.sample_size, number_of_samples),
                replace=False,
            )
            tree.fit(self.X_train_[index])
            self.trees.append(tree)

            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
        if verbose:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

        return self

    def predict(self, X):
        return self.path_length(X)

    def path_length(self, X):
        paths = np.array([tree.path_length(X) for tree in self.trees])
        return paths.mean(axis=0)

    def anomaly_score(self, X):
        average_path = self.path_length(X)
        normalizer = self.c(min(self.sample_size, self.X_train_.shape[0]))
        if normalizer == 0:
            return np.ones(X.shape[0])
        return 2.0 ** (-average_path / normalizer)

    def suspects(self, X, number_of_suspects, use_scores=True):
        if use_scores:
            scores = self.anomaly_score(X)
            index = np.argsort(scores)[-number_of_suspects:][::-1]
            return X[index], scores[index]

        depths = self.path_length(X)
        index = np.argsort(depths)[:number_of_suspects]
        return X[index], depths[index]

    def predict_labels(self, X, contamination=0.01):
        if not (0 < contamination < 1):
            raise ValueError("contamination must be between 0 and 1")

        scores = self.anomaly_score(X)
        threshold = np.quantile(scores, 1.0 - contamination)

        labels = np.ones(X.shape[0], dtype=int)
        labels[scores >= threshold] = -1

        return labels

    def visualize_tree(
        self,
        feature1=0,
        feature2=1,
        resolution=100,
        cmap=plt.cm.RdBu,
    ):
        os.makedirs("img", exist_ok=True)

        X_plot = self.X_train_[:, [feature1, feature2]].astype(float)

        x_min, x_max = np.min(X_plot[:, 0]), np.max(X_plot[:, 0])
        y_min, y_max = np.min(X_plot[:, 1]), np.max(X_plot[:, 1])

        x_margin = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
        y_margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        XX, YY = np.meshgrid(xs, ys)

        baseline = self._baseline_row()
        grid_points = np.tile(baseline, (XX.size, 1))

        grid_points[:, feature1] = XX.ravel()
        grid_points[:, feature2] = YY.ravel()

        Z = self.predict(grid_points).reshape(XX.shape)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(X_plot[:, 0], X_plot[:, 1], edgecolors="black")
        axes[0].set_title("Training points")
        axes[0].set_xlabel(f"feature {feature1}")
        axes[0].set_ylabel(f"feature {feature2}")

        mesh = axes[1].pcolormesh(
            XX,
            YY,
            Z,
            cmap=cmap,
            shading="auto",
        )
        axes[1].scatter(X_plot[:, 0], X_plot[:, 1], edgecolors="black")
        axes[1].set_title("Average isolation depth")
        axes[1].set_xlabel(f"feature {feature1}")
        axes[1].set_ylabel(f"feature {feature2}")

        fig.colorbar(mesh, ax=axes[1], label="average path length")

        plt.tight_layout()
        plt.savefig("img/sltn_random_forest.png")
        plt.show()

    def _baseline_row(self):
        baseline = np.empty(self.X_train_.shape[1], dtype=object)

        for feature in range(self.X_train_.shape[1]):
            values = self.X_train_[:, feature]

            if feature in self.categorical_features_:
                unique_values, counts = np.unique(values, return_counts=True)
                baseline[feature] = unique_values[np.argmax(counts)]
            else:
                baseline[feature] = np.mean(values.astype(float))

        return baseline

    def c(self, n):
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        euler_gamma = 0.5772156649
        return 2.0 * (np.log(n - 1) + euler_gamma) - (2.0 * (n - 1) / n)
