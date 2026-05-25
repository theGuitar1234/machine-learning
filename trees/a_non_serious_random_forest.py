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
                    np.mean(oob_predictions == oob_true_labels) * 100
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
