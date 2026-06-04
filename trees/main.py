from .a_non_serious_node import ANonSeriousNode
from .a_non_serious_decision_tree import ANonSeriousDecisionTree
from .a_non_serious_random_forest import ANonSeriousRandomForest
from .a_non_serious_isolation_random_tree import ANonSeriousIsolationRandomTree
from .a_non_serious_isolation_random_forest import ANonSeriousIsolationRandomForest

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


if __name__ == "__main__":

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

    def logarithmic_spirals(
        n_objects_by_class,
        a=0.08,
        b=0.22,
        theta_max=4 * np.pi,
        sigma=0.03,
        seed=0,
        angle=0,
    ):
        rng = np.random.default_rng(seed)

        def spiral_arm(class_id):
            theta = np.linspace(0, theta_max, n_objects_by_class)

            theta = theta + class_id * np.pi + angle

            r = a * np.exp(b * theta)

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            points = np.column_stack([x, y])

            # points += rng.normal(scale=sigma, size=points.shape)

            return points

        X0 = spiral_arm(0)
        X1 = spiral_arm(1)

        X = np.concatenate([X0, X1], axis=0)

        target = np.array(
            [0] * n_objects_by_class + [1] * n_objects_by_class,
            dtype="int32"
        )

        return X, target
    
    def spiral_of_clouds(
        n_objects_by_class,
        radius=5,
        n_turns=3,
        sigma=0.12,
        seed=0,
        angle=0,
        b=0.18,
    ):
        rng = np.random.default_rng(seed)
        theta = np.linspace(0, 2 * np.pi * n_turns, n_objects_by_class)

        r = np.exp(b * theta) - 1
        r = radius * r / r.max()

        def arm(k):
            t = theta + k * np.pi + angle

            x = r * np.cos(t)
            y = r * np.sin(t)

            points = np.column_stack([x, y])

            # points += rng.normal(scale=sigma, size=points.shape)

            return points

        X0 = arm(0)
        X1 = arm(1)

        X = np.vstack([X0, X1])
        y = np.array(
            [0] * n_objects_by_class + [1] * n_objects_by_class,
            dtype="int32"
        )

        return X, y
        
    # dataset = datasets.load_iris()
    seed = np.random.randint(0, 100)

    # X, y = circle_of_clouds(10, 30, seed=seed)
    # X, y = logarithmic_spirals(
    #     n_objects_by_class=500,
    #     a=0.08,
    #     b=0.22,
    #     theta_max=4 * np.pi,
    #     sigma=0.04,
    #     seed=42
    # )
    X, y = spiral_of_clouds(
        n_objects_by_class=150,
        radius=5,
        n_turns=3,
        sigma=0.08,
        seed=42,
        b=0.12
    )
    feature1 = 0
    feature2 = 1

    # X = dataset.data[:, [feature1, feature2]]
    # X = dataset.data
    # y = dataset.target

    # X, y = make_regression(
    #     n_samples=300,
    #     n_features=2,
    #     noise=15,
    #     random_state=seed,
    # )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        # stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed,
        # stratify=y_train_val
    )

    # 60% training
    # 20% validation
    # 20% test

    configs = ANonSeriousDecisionTree.generate_config(max_depth=10)

    # final_tree = ANonSeriousDecisionTree.choose_best_cross_validation(
    #     X_train_val, y_train_val, configs, log=True, verbose=False
    # )
    # _, final_test_accuracy = final_tree.evaluate_dataset(X_test, y_test)

    # print(f"\nFinal Test Accuracy: {final_test_accuracy}%")

    # # print("\nPruning the tree...")
    # # final_tree.prune_minimum_error()

    # # print(
    # #     "After pruning Validation Accuracy:",
    # #     final_tree.evaluate_dataset(X_val, y_val)[1],
    # # )
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
    # # final_tree.permutation_importance(X_test, y_test, verbose=True)

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
    #     seed=seed,
    #     tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
    #     vectorized=True,
    # )

    # tree.fit(X_train, y_train, node=True)

    # tree.visualize_tree(feature1, feature2)

    # predictions = tree.predict(X_test)
    # print(f"Predictions : {predictions}")
    # print("\nRegression Tree evaluation : ")
    # print(tree.evaluate_dataset(X_test, y_test))

    print("\nInitializing a Random Forest...")
    random_forest = ANonSeriousRandomForest(
        information_gain=ANonSeriousDecisionTree.InformationGain.ENTROPY,
        random_criterion=ANonSeriousDecisionTree.RandomCriterion.RANDOM_SPLIT,
        _bootstrap=False,
        forest_type=ANonSeriousRandomForest.ForestType.CLASSIFICATION,
        voting=ANonSeriousRandomForest.Voting.SOFT,
        max_depth=15,
        number_of_trees=500,
    )

    # print("\nFitting the Random Forest...")
    # limit = 5
    random_forest.fit(X_train, y_train, verbose=True)

    # if random_forest.forest_type is ANonSeriousRandomForest.ForestType.CLASSIFICATION:
    #     _, forest_accuracy = random_forest.evaluate_random_forest(X_test, y_test)
    #     probabilities = random_forest.predict_probabilities(X_test[:limit])
    #     print(f"\nTest Accuracy : {forest_accuracy}")
    #     print(f"{probabilities.sum(axis=1)}\nProbabilities {probabilities}")
    # else:
    #     evaluation = random_forest.evaluate_random_forest(X_test, y_test)
    #     predictions = random_forest.predict(X_test)
    #     print(f"\nEvaluation {evaluation}")
    #     print(f"Predictions {predictions}")

    # # print("\nOOB Evaluation : ")
    # # random_forest.oob_evaluation(verbose=True)

    random_forest.visualize_tree(feature1, feature2)

    # X_train_, _ = circle_of_clouds(1, 100, sigma=0.2)  # a cloud
    # X_train_[0] = [-1, 0]  # an outlier
    # X_train_[1] = [-2, 1]  # an outlier
    # X_train_[2] = [-3, 2]  # an outlier

    # ANonSeriousIsolationRandomTree.visualize_tree(X_train_)

    # X_train_, _ = circle_of_clouds(3, 100, sigma=0.2)
    # IRF = ANonSeriousIsolationRandomForest(max_depth=15)
    # IRF.fit(X_train_, verbose=1)
    # suspects, depths = IRF.suspects(X_train_, number_of_suspects=3)

    # print("suspects :", suspects)
    # print("depths of suspects :", depths)

    # IRF.visualize_tree()
