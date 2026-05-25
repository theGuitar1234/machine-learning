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

from .a_non_serious_decision_tree import ANonSeriousDecisionTree


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
