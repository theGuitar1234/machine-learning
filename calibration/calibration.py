from dataclasses import dataclass
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import os

from neural_network.NeuralNetwork import NeuralNetwork


class Calibration:

    @dataclass
    class TrainDefaults:
        learning_rate: float = 0.03
        epochs: int = 5000
        epsilon: float = 1e-12
        h: float = 1e-5

    def __init__(self, config=None):
        self.config = config or self.TrainDefaults()

    def reliability_diagram(self, y, y_hat, number_of_bins, verbose=False):
        bins = np.linspace(0, 1, number_of_bins + 1)
        average_predicted_probabilities = []
        actual_positive_rates = []
        bin_sizes = []
        for bin in range(number_of_bins):
            lower = bins[bin]
            upper = bins[bin + 1]

            if bin == number_of_bins - 1:
                indices = np.where((lower <= y_hat) & (y_hat <= upper))[0]
            else:
                indices = np.where((lower <= y_hat) & (y_hat < upper))[0]

            if len(indices) == 0:
                continue
            average_predicted_probability = np.mean(y_hat[indices])
            actual_positive_rate = np.mean(y[indices])
            bin_size = len(indices)
            average_predicted_probabilities.append(average_predicted_probability)
            actual_positive_rates.append(actual_positive_rate)
            bin_sizes.append(bin_size)
        if verbose:
            os.makedirs("calibration/img", exist_ok=True)
            plt.title("Reliability Diagram")
            plt.xlabel("Average Predicted Probability")
            plt.ylabel("Actual Positive Rate")
            plt.plot(average_predicted_probabilities, actual_positive_rates, marker="o")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.savefig("calibration/img/rlblty_dgrm.png")
            plt.show()
        return {
            "average_predicted_probabilities": average_predicted_probabilities,
            "actual_positive_rates": actual_positive_rates,
            "bin_sizes": bin_sizes,
        }

    def expected_calibration_error(self, y, y_hat, number_of_bins):
        ECE = 0.0
        bins = np.linspace(0, 1, number_of_bins + 1)
        for bin in range(number_of_bins):
            lower = bins[bin]
            upper = bins[bin + 1]

            if bin == number_of_bins - 1:
                indices = np.where((lower <= y_hat) & (y_hat <= upper))[0]
            else:
                indices = np.where((lower <= y_hat) & (y_hat < upper))[0]

            if len(indices) == 0:
                continue
            accuracy_bin = np.mean(y[indices])
            confidence_bin = np.mean(y_hat[indices])
            weight_bin = len(indices) / len(y)

            ECE += weight_bin * abs(accuracy_bin - confidence_bin)
        return ECE

    def platt_scaling(self, model, X_cal, Y_cal, number_of_classes):
        epsilon = self.config.epsilon
        probabilities = model.predict_proba(X_cal)
        true_classes = np.argmax(Y_cal, axis=1)
        self.calibrators = []
        for class_id in range(number_of_classes):
            p_class = probabilities[:, class_id]
            p_class = np.clip(p_class, epsilon, 1 - epsilon)
            score = np.log(p_class / (1 - p_class))
            binary_target = (true_classes == class_id).astype(np.float32)
            
            A, B = 1, 0
            self.calibrators.append(self._fit_platt_scaling(A, B, score, binary_target))            
        return self.calibrators

    def _fit_platt_scaling(self, A, B, score, binary_target):
        for _ in range(self.config.epochs):
            z = A * score + B
            p = self.sigmoid(z)
            error = p - binary_target

            dA = np.mean(error * score)
            dB = np.mean(error)

            A -= self.config.learning_rate * dA
            B -= self.config.learning_rate * dB
        return A, B

    def predict_calibrated(self, model, X_new, number_of_classes):
        epsilon = self.config.epsilon
        probabilities = model.predict_proba(X_new)
        calibrated_scores = np.empty(probabilities.shape)
        for class_id in range(number_of_classes):
            p_class = probabilities[:, class_id]
            p_class = np.clip(p_class, epsilon, 1 - epsilon)
            score = np.log(p_class / (1 - p_class))
            A, B = self.calibrators[class_id]
            calibrated_scores[:, class_id] = self.sigmoid(A * score + B)
        return self._normalize_calibrated_scores(calibrated_scores)
    
    def _normalize_calibrated_scores(self, calibrated_scores):
        row_sums = np.sum(calibrated_scores, axis=1, keepdims=True)
        row_sums = np.clip(row_sums, self.config.epsilon, None)
        return calibrated_scores / row_sums
    
    def isotonic_calibration(self, model, X_cal, Y_cal, number_of_classes):
        probabilities = model.predict_proba(X_cal)
        true_classes = np.argmax(Y_cal, axis=1)
        self.calibrators = []
        for class_id in range(number_of_classes):
            scores = probabilities[:, class_id]
            binary_target = (true_classes == class_id).astype(np.float32)
            self.calibrators.append(self._fit_isotonic(scores, binary_target))
        return self.calibrators

    def _fit_isotonic(self, scores, targets):
        ordered = np.argsort(scores)
        scores = scores[ordered]
        targets = targets[ordered]
        
        blocks = []
        
        for score, target in zip(scores, targets):
            block = {
                "min_score": score,
                "max_score": score,
                "value": target,
                "weight": 1.0,
            }
            
            blocks.append(block)
            
            while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
                right = blocks.pop()
                left = blocks.pop()
                
                total_weight = left["weight"] + right["weight"]
                
                merged = {
                    "min_score": left["min_score"],
                    "max_score": right["max_score"],
                    "value": (
                        left["value"] * left["weight"] +
                        right["value"] * right["weight"]
                    ) / total_weight,
                    "weight": total_weight,
                }
                
                blocks.append(merged)
        thresholds = np.array([block["max_score"] for block in blocks])
        values = np.array([block["value"] for block in blocks])
        
        return thresholds, values
    
    def predict_isotonic(self, model, X_new, number_of_classes):
        probabilities = model.predict_proba(X_new)
        calibrated_scores = np.empty(probabilities.shape)
        for class_id in range(number_of_classes):
            scores = probabilities[:, class_id]
            calibrator = self.calibrators[class_id]
            calibrated_scores[:, class_id] = self._predict_isotonic_binary(
                scores,
                calibrator,
            )
        return self._normalize_calibrated_scores(calibrated_scores)
    
    def _predict_isotonic_binary(self, scores, calibrator):
        thresholds, values = calibrator
        indices = np.searchsorted(thresholds, scores, side="left")
        indices = np.clip(indices, 0, len(values) - 1)
        return values[indices]
    
    def temperature_scaling(self, neural_net, X_cal, Y_cal):
        logits = neural_net.predict_logits(X_cal)
        self._fit_temperature_scaling(Y_cal=Y_cal, logits=logits)
        return self.T
    
    def _fit_temperature_scaling(self, Y_cal, logits):
        s = 0.0
        for _ in range(self.config.epochs):
            self.T = np.exp(s)
            gradient = self._temperature_gradient(logits, Y_cal, s)
            s -= self.config.learning_rate * gradient
        self.T = np.exp(s)
        return self.T
    
    def _temperature_gradient(self, logits, Y, s):
        h = self.config.h
        loss_plus = self._temperature_loss(logits, Y, s + h)
        loss_minus = self._temperature_loss(logits, Y, s - h)
        return (loss_plus - loss_minus) / (2 * h)
    
    def _temperature_loss(self, logits, Y, s):
        T = np.exp(s)
        probabilities = self.softmax(logits / T)
        return self.log_loss(Y, probabilities)
    
    def predict_temperature(self, neural_net, X_new):
        logits = neural_net.predict_logits(X_new)
        return self.softmax(logits / self.T)

    def softmax(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def brier_score(self, y, y_hat):
        return np.mean(np.sum((y_hat - y) ** 2, axis=1))

    def log_loss(self, y, y_hat):
        y_hat = np.clip(y_hat, self.config.epsilon, 1.0 - self.config.epsilon)
        return -np.mean(np.sum(y * np.log(y_hat), axis=1))

    def sigmoid(self, z):
        z = np.asarray(z, dtype=np.float32)
        out = np.empty_like(z)

        pos = z >= 0
        neg = ~pos

        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)

        return out


if __name__ == "__main__":

    number_of_classes = 10

    dataset = NeuralNetwork.load_from_npz("neural_network/data/npz/MNIST.npz")
    prepared_dataset = NeuralNetwork.prepare_datasets(dataset, number_of_classes)

    X_test = prepared_dataset["X_test"]
    Y_test = prepared_dataset["Y_test"]

    X_test = X_test.astype(np.float32) / 255.0

    model_name = "trained_model_digit_recognizer"
    loaded_model, _ = NeuralNetwork.load_model(
        f"neural_network/models/{model_name}.pkl", device=NeuralNetwork.Device.CPU
    )

    predictions = loaded_model.predict_proba(X_test)

    y = np.argmax(Y_test, axis=1)
    y_hat = np.argmax(predictions, axis=1)

    confidence = np.max(predictions, axis=1)
    correct = (y_hat == y).astype(int)

    calibration = Calibration()
    calibration.reliability_diagram(correct, confidence, 10, verbose=True)
    ECE = calibration.expected_calibration_error(correct, confidence, 10)
    brier_score = calibration.brier_score(Y_test, predictions)
    log_loss = calibration.log_loss(Y_test, predictions)

    print(f"ECE Score: {ECE}")
    print(f"Brier Score: {brier_score}")
    print(f"Log Loss : {log_loss}")
    
    X_cal = prepared_dataset["X_valid"]
    Y_cal = prepared_dataset["Y_valid"]
    
    X_cal = X_cal.astype(np.float32) / 255.0
    
    # calibration.platt_scaling(loaded_model, X_cal, Y_cal, number_of_classes)
    # predictions = calibrated_prediction = calibration.predict_calibrated(
    #     loaded_model,
    #     X_test,
    #     number_of_classes
    # )
    
    # confidence = np.max(predictions, axis=1)
    # calibration.reliability_diagram(correct, confidence, number_of_classes, verbose=True)
    
    # calibration.isotonic_calibration(loaded_model, X_cal, Y_cal, number_of_classes)

    # predictions = calibration.predict_isotonic(
    #     loaded_model,
    #     X_test,
    #     number_of_classes,
    # )
    
    # confidence = np.max(predictions, axis=1)
    # calibration.reliability_diagram(correct, confidence, number_of_classes, verbose=True)

    calibration.temperature_scaling(loaded_model, X_cal, Y_cal)
    
    predictions = calibration.predict_temperature(loaded_model, X_test)

    y_hat = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    correct = (y_hat == y).astype(int)

    calibration.reliability_diagram(correct, confidence, 10, verbose=True)
