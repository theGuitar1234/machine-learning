from dataclasses import dataclass
from enum import Enum

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from trees import ANonSeriousDecisionTree
import matplotlib.pyplot as plt
import os


class XGBoost:

    class LossType(Enum):
        BINARY_CROSS_ENTROPY = 1
        SSE = 2

    @dataclass
    class TrainDefaults:
        epsilon: float = 1e-12
        l2: float = 0.5
        gamma: float = 0.2
        threshold: float = 0.5
        learning_rate: float = 0.03

    def __init__(
        self,
        boosting_rounds=100,
        max_depth=10,
        minimum_population_size=2,
        minimum_gain=0.001,
        categorical=False,
        adjacent=False,
        information_gain=0.01,
        random_criterion=None,
        tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
        loss_type=None,
        sub_sample=0.8,
        column_sub_sample=1,
        early_stopping=False,
        restore_best=False,
        log=False,
        config=None,
    ):
        if not (0 < sub_sample <= 1) or not (0 < column_sub_sample <= 1):
            raise ValueError(
                "sub_sample and column_sub_sample must be a positive fraction less than 1"
            )
        if config is None:
            config = self.TrainDefaults()
        self.config = config
        self.sub_sample = sub_sample
        self.column_sub_sample = column_sub_sample
        self.boosting_rounds = boosting_rounds
        self.max_depth = max_depth
        self.minimum_population_size = minimum_population_size
        self.minimum_gain = minimum_gain
        self.categorical = categorical
        self.adjacent = adjacent
        self.information_gain = information_gain
        self.random_criterion = random_criterion
        self.tree_type = tree_type
        self.loss_type = loss_type
        self.early_stopping = early_stopping
        self.restore_best = restore_best
        self.log = log

    def fit(self, X, y, X_val, y_val, optimized=False):
        self.X_train_ = X
        self.y_train_ = y

        self.trees = []
        self.feature_indices = []

        epsilon = self.config.epsilon
        learning_rate = self.config.learning_rate

        if self.log:
            space = " "
            loading_bar_length = 10
            padding = 10
            loading = "."
            print()

        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p0 = np.clip(np.mean(y), epsilon, 1 - epsilon)
                self.F_0x = np.log(p0 / (1 - p0))
            case self.LossType.SSE:
                self.F_0x = np.mean(y)
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )

        self.F_x = np.repeat(self.F_0x, y.shape[0])

        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        sample_size = max(1, int(self.sub_sample * number_of_samples))
        number_of_selected_features = max(
            1, int(self.column_sub_sample * number_of_features)
        )

        total_loss = 0
        best_val_loss = float("inf")
        patience = 100
        best_round = -1
        best_number_of_trees = 0
        patience_counter = 0
        for round in range(self.boosting_rounds):
            tree = ANonSeriousDecisionTree(
                max_depth=self.max_depth,
                minimum_population_size=self.minimum_population_size,
                minimum_gain=self.minimum_gain,
                categorical=self.categorical,
                adjacent=self.adjacent,
                information_gain=self.information_gain,
                random_criterion=self.random_criterion,
                tree_type=self.tree_type,
                xgboost=True,
                vectorized=True,
            )
            tree.xgboost_optimized = optimized
            tree.l2 = self.config.l2
            tree.gamma = self.config.gamma

            pseudo_residual = -self.dloss(y, self.F_x)
            match self.loss_type:
                case self.LossType.SSE:
                    total_loss += self.sse_mean(y, self.F_x)
                case self.LossType.BINARY_CROSS_ENTROPY:
                    total_loss += self.binary_cross_entropy_loss(y, self.F_x)
                case _:
                    raise ValueError(
                        f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                    )

            indices = np.random.choice(
                number_of_samples, size=sample_size, replace=False
            )
            feature_indices = np.random.choice(
                number_of_features, size=number_of_selected_features, replace=False
            )

            X_sub = X[indices][:, feature_indices]
            pseudo_residual_sub = pseudo_residual[indices]

            g = self.gradient(y, self.F_x)
            h = self.hessian(y, self.F_x)

            g_sub = g[indices]
            h_sub = h[indices]

            tree.fit(
                X_sub,
                pseudo_residual_sub,
                gradient=g_sub,
                hessian=h_sub,
            )

            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
            self.F_x += learning_rate * tree.predict(X[:, feature_indices])

            _, val_loss = self.evaluate_dataset(X_val, y_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_round = round
                patience_counter = 0
                best_number_of_trees = len(self.trees)
            else:
                patience_counter += 1
            if self.early_stopping and patience_counter >= patience:
                print(
                    "Overfitting detected. Early stopping at round: ",
                    round,
                    "Best Round : ",
                    best_round,
                )
                break
            if self.log:
                penalty = self.tree_complexity_penalty(total_loss)
                tree_score = self.tree_structure_score(tree)
                progression = round % loading_bar_length
                print(
                    f"Fitting the tree{loading*progression + space*(loading_bar_length - progression)} [ XGBoost Penalty : {penalty} ] [ Tree Structure Score : {tree_score} ] [ Validation Loss : {val_loss} ]{space*padding}",
                    end="\r"
                )
        if self.restore_best:
            self.trees = self.trees[:best_number_of_trees]
            self.feature_indices = self.feature_indices[:best_number_of_trees]
            self.F_x = np.repeat(self.F_0x, X.shape[0])
            for tree, feature_indices in zip(self.trees, self.feature_indices):
                self.F_x += learning_rate * tree.predict(X[:, feature_indices])
        return self

    def dloss(self, y, F_x):
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p_x = self.sigmoid(F_x)
                return p_x - y
            case self.LossType.SSE:
                return F_x - y
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )

    def gradient(self, y, F_x):
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p_x = self.sigmoid(F_x)
                return p_x - y
            case self.LossType.SSE:
                return F_x - y
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )

    def hessian(self, y, F_x):
        match self.loss_type:
            case self.LossType.BINARY_CROSS_ENTROPY:
                p_x = self.sigmoid(F_x)
                return p_x * (1 - p_x)
            case self.LossType.SSE:
                return np.repeat(1, F_x.shape[0])
            case _:
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )

    def tree_complexity_penalty(self, loss):
        gamma = self.config.gamma

        l2 = self.config.l2

        complexity = 0
        for tree in self.trees:
            number_of_leaves = len(tree.get_leaves())
            leaf_scores = self.get_leaf_scores(tree)
            complexity += gamma * number_of_leaves + 1 / 2 * l2 * np.sum(leaf_scores**2)
        return loss + complexity

    def tree_structure_score(self, tree):
        l2 = self.config.l2
        gamma = self.config.gamma
        tree_score = 0
        leaves = tree.get_leaves()
        number_of_leaves = len(leaves)
        for leaf in leaves:
            gradients = leaf.gradient
            hessians = leaf.hessian
            tree_score += -0.5 * np.sum(gradients) ** 2 / (np.sum(hessians) + l2)
        tree_score += gamma * number_of_leaves
        return tree_score

    def get_leaf_scores(self, tree):
        return np.asarray([leaf.value for leaf in tree.get_leaves()])

    def sse(self, y, F_x):
        return 1 / 2 * (y - F_x) ** 2

    def sse_mean(self, y, F_x):
        return np.mean(self.sse(y, F_x))

    def sse_sum(self, y, F_x):
        return np.sum(self.sse(y, F_x))

    def sigmoid(self, z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)

        pos = z >= 0
        neg = ~pos

        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)

        return out

    def binary_cross_entropy_loss(self, y, F_x):
        epsilon = self.config.epsilon

        p_x = self.sigmoid(F_x)
        m = y.shape[0]
        p_x = np.clip(p_x, epsilon, 1.0 - epsilon)
        return -np.sum(y * np.log(p_x) + (1 - y) * np.log(1 - p_x)) / m

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
                raise ValueError(
                    f"Unsupported {self.LossType}, supported values are {list(self.LossType)}"
                )
        return predictions, loss

    def predict(self, X):
        prediction = np.repeat(self.F_0x, X.shape[0])
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            prediction += self.config.learning_rate * tree.predict(X[:, feature_indices])
        return prediction

    def predict_proba(self, X):
        return self.sigmoid(self.predict(X))

    def predict_class(self, X):
        threshold = self.config.threshold
        return (self.predict_proba(X) >= threshold).astype(int)

    def visualize(self):
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Model is not fit")
        os.makedirs("boosting/img", exist_ok=True)
        X = self.X_train_
        y = self.y_train_
        y_pred = self.predict(X)

        plt.figure(figsize=(7, 5))
        plt.scatter(y, y_pred, alpha=0.7)

        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())

        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        plt.xlabel("Actual y")
        plt.ylabel("Predicted y")
        plt.title("Actual vs Predicted")
        plt.grid(True)
        plt.savefig("boosting/img/prdctns.png")
        plt.show()

        residuals = y - y_pred

        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(0, linestyle="--")

        plt.xlabel("Predicted y")
        plt.ylabel("Residual: y - prediction")
        plt.title("Residual Plot")
        plt.grid(True)
        plt.savefig("boosting/img/rsdl.png")
        plt.show()

        losses = []

        current_pred = np.repeat(self.F_0x, X.shape[0])

        initial_mse = np.mean((y - current_pred) ** 2)
        losses.append(initial_mse)

        for tree, feature_indices in zip(self.trees, self.feature_indices):
            current_pred += self.config.learning_rate * tree.predict(X[:, feature_indices])
            mse = np.mean((y - current_pred) ** 2)
            losses.append(mse)

        plt.figure(figsize=(7, 5))
        plt.plot(losses)

        plt.xlabel("Boosting Round")
        plt.ylabel("Training MSE")
        plt.title("Training Loss Over Boosting Rounds")
        plt.grid(True)
        plt.savefig("boosting/img/lsss.png")
        plt.show()

        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

        x1_grid = np.linspace(x1_min, x1_max, 100)
        x2_grid = np.linspace(x2_min, x2_max, 100)

        xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

        grid_points = np.c_[xx1.ravel(), xx2.ravel()]

        grid_predictions = self.predict(grid_points)
        zz = grid_predictions.reshape(xx1.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(xx1, xx2, zz, levels=30)
        plt.colorbar(contour, label="Predicted y")

        plt.scatter(X[:, 0], X[:, 1], alpha=0.7)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Gradient Boosting Prediction Surface")
        plt.grid(True)
        plt.savefig("boosting/img/prdctn_srfc.png")
        plt.show()

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(xx1, xx2, zz, alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], y, alpha=0.7)

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target / Prediction")
        ax.set_title("Learned Regression Surface")

        plt.savefig("boosting/img/3d_srfc.png")
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=300,
        n_features=2,
        noise=15,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        # stratify=y_train_val
    )

    xgboost = XGBoost(
        loss_type=XGBoost.LossType.SSE,
        restore_best=True,
        early_stopping=True,
        log=True,
    )
    xgboost.fit(X_train, y_train, X_val, y_val, optimized=False)

    xgboost.visualize()
