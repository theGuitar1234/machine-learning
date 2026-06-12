from dataclasses import dataclass

from enum import Enum
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from boosting.catboost import CatBoost
from boosting.gradient_boosting import GradientBoosting
from neural_network.NeuralNetwork import NeuralNetwork
from trees.a_non_serious_decision_tree import ANonSeriousDecisionTree

from .data.load_streambox_churn_npz_example import load_stream_churn_npz


class ABTesting:

    class ClassType(Enum):
        MULTI_CLASS = 1
        BINARY = 2

    @dataclass
    class TestDefaults:
        epsilon: float = 1e-12

    def __init__(
        self,
        class_type=None,
        config=None,
    ):
        self.config = config or self.TestDefaults()
        self.class_type = class_type or self.ClassType.BINARY

    def analyze_thresholds(
        self,
        model,
        X=None,
        y=None,
        thresholds=None,
        number_of_thresholds=99,
        log=False,
        clazz=1,
    ):
        print(f"\nDataset size: {X.shape[0]} samples\n")
        os.makedirs("calibration/img", exist_ok=True)
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, number_of_thresholds)

        print("Getting the model probabilities...")
        A = model.predict_proba(X)
        y_true = np.asarray(y).ravel().astype(int)
        match self.class_type:
            case self.ClassType.BINARY:
                if A.ndim == 2 and A.shape[1] == 2:
                    y_score = A[:, 1]
                else:
                    y_score = A.ravel()
            case self.ClassType.MULTI_CLASS:
                y_true = (y_true == clazz).astype(int)
                y_score_class = A[:, clazz]
        precisions = []
        recalls = []
        tprs = []
        fprs = []

        print("\nAnalyzing Thresholds...\n")
        print(f"Total Predictions: {A.shape[0]} Total True Label: {y.shape[0]}")
        for threshold in thresholds:
            match self.class_type:
                case self.ClassType.BINARY:
                    y_pred = (y_score >= threshold).astype(int)
                case self.ClassType.MULTI_CLASS:
                    y_pred = (y_score_class >= threshold).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            precision = TP / (TP + FP + self.config.epsilon)
            recall = TP / (TP + FN + self.config.epsilon)
            tpr = precision
            fpr = FP / (FP + TN + self.config.epsilon)

            if log:
                print(
                    f"[ Threshold: {threshold:.2f} ] [ True Positives: {TP} ] [ True Negatives: {TN} ] [ False Positives: {FP} ] [ False Negatives: {FN} ] [ Precision: {precision} ] [ Recall: {recall} ]"
                )
            precisions.append(precision)
            recalls.append(recall)
            tprs.append(tpr)
            fprs.append(fpr)
        self.precision_recall_tradeoff(precisions, recalls, thresholds)
        self.roc_curve(tprs, fprs)

    def roc_auc(self, true_positive_rate, false_positive_rate):
        true_positive_rate = np.asarray(true_positive_rate)
        false_positive_rate = np.asarray(false_positive_rate)
        order = np.argsort(true_positive_rate)
        order = np.argsort(false_positive_rate)
        fpr = false_positive_rate[order]
        tpr = true_positive_rate[order]
        return np.trapezoid(tpr, fpr)

    def pr_auc(self, precision, recall):
        precision = np.asarray(precision)
        recall = np.asarray(recall)
        order = np.argsort(precision)
        precision = precision[order]
        recall = recall[order]
        return np.trapezoid(precision, recall)

    def precision_recall_tradeoff(self, precisions, recalls, thresholds):
        plt.title(
            f"Precision/Recall Tradeoff: PR-AUC: {self.pr_auc(precisions, recalls)}"
        )
        plt.ylabel("Precision/Recall")
        plt.xlabel("Thresholds")
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.legend()
        plt.savefig("calibration/img/prcsn_rcll_trdff.png")
        plt.show()

    def roc_curve(self, true_positive_rate, false_positive_rate):
        plt.title(
            f"ROC-Curve ROC-AUC: {self.roc_auc(true_positive_rate, false_positive_rate)}"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(false_positive_rate, true_positive_rate)
        plt.savefig("calibration/img/rc_crv.png")
        plt.show()


if __name__ == "__main__":

    number_of_classes = 2
    seed = 42

    (
        X,
        Y,
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        X_calib,
        Y_calib,
        X_test,
        Y_test,
        feature_names,
        categorical_indices,
        metadata,
    ) = load_stream_churn_npz(verbose=True)

    sys.exit(0)

    ab_testing = ABTesting(
        class_type=ABTesting.ClassType.MULTI_CLASS,
    )
    
    ab_testing.analyze_thresholds(
        model=loaded_model,
        X=X_valid,
        y=Y_valid,
        log=True,
        clazz=3,
    )
