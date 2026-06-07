from sklearn.datasets import make_regression

from trees.a_non_serious_decision_tree import ANonSeriousDecisionTree
import numpy as np


class ANonSeriousTreeExplainability:

    @staticmethod
    def explain_worst_regressions(model, X, y, top_k=10, verbose=False):
        predictions = model.predict(X)
        errors = y - predictions
        absolute_errors = abs(errors)

        worst_indices = np.argsort(absolute_errors)[-top_k:]

        reports = []
        for index in worst_indices:
            report = {
                "row_index": index,
                "y": y[index],
                "y_hat": predictions[index],
                "error": errors[index],
                "absolute_error": absolute_errors[index],
            }
            reports.append(report)
        if verbose:
            print(f"\nTop {top_k} worst Regression predictions : ")
            for report in reports:
                print(f"""
    Worst Regression Case: row {report["row_index"]}
    True Value: {report["y"]}
    Predicted Value: {report["y_hat"]}
    Error: {report["error"]} {"The Model overpredicted" if report["error"] > 0 else "The Model underpredicted"}
    """)
        return reports

    @staticmethod
    def explain_worst_classifications(
        model, X, y, threshold=0.5, top_k=10, verbose=False
    ):
        scores = model.predict(X)

        def sigmoid(z):
            z = np.asarray(z, dtype=float)
            out = np.empty_like(z)

            pos = z >= 0
            neg = ~pos

            out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
            ez = np.exp(z[neg])
            out[neg] = ez / (1.0 + ez)

            return out

        probabilities = sigmoid(scores)
        predicted_labels = probabilities >= threshold
        false_positive_indices = np.where((y == 0) & (predicted_labels == 1))[0]
        false_negative_indices = np.where((y == 1) & (predicted_labels == 0))[0]

        false_positive_indices = np.sort(false_positive_indices)[:-top_k]
        false_negative_indices = np.sort(false_negative_indices)[:-top_k]

        false_positive_reports = []
        false_negative_reports = []
        for false_positive_index, false_negative_index in zip(
            false_positive_indices, false_negative_indices
        ):
            false_positive_report = {
                "row_index": false_positive_index,
                "y": y[false_positive_index],
                "y_hat": probabilities[false_positive_index],
            }
            false_negative_report = {
                "row_index": false_negative_index,
                "y": y[false_negative_index],
                "y_hat": probabilities[false_negative_index],
            }
            false_positive_reports.append(false_positive_report)
            false_negative_reports.append(false_negative_report)
        if verbose:
            for false_positive_report, false_negative_reports in zip(
                false_positive_reports, false_negative_reports
            ):
                print(f"""
    False positive: row {false_positive_report["row_index"]}
    True label: {false_positive_report["y"]}
    Predicted probability: {false_positive_report["y_hat"]}
    Model Decision: "predicted" {f"1 because probability >= {threshold}" if false_positive_report["y_hat"] >= threshold else f"0 because probability < {threshold}"}

    False negative: row {false_negative_report["row_index"]}
    True label: {false_negative_report["y"]}
    Predicted probability: {false_negative_report["y_hat"]}
    Model Decision: "predicted" {f"1 because probability >= {threshold}" if false_negative_report["y_hat"] >= threshold else f"0 because probability < {threshold}"}
    """)
        return false_positive_reports, false_negative_reports
    
    # def summarize_error_patterns(error_reports):
    #     feature_error_counts = np.empty()
    #     feature_error_magnitude = np.empty()
        
    #     for report in error_reports:
    #         for feature, 


if __name__ == "__main__":
    seed = 42
    X, y = make_regression(
        n_samples=300,
        n_features=2,
        noise=15,
        random_state=seed,
    )

    tree = ANonSeriousDecisionTree(
        max_depth=10,
        minimum_population_size=2,
        minimum_gain=0.001,
        categorical=False,
        adjacent=False,
        information_gain=ANonSeriousDecisionTree.InformationGain.GINI,
        tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
    )

    tree.fit(X, y, verbose=False)
    ANonSeriousTreeExplainability.explain_worst_classifications(tree, X, y, verbose=True)
