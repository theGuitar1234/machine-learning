from sklearn import datasets
import numpy as np
from trees import ANonSeriousDecisionTree
import matplotlib.pyplot as plt
import os


class GradientBoosting:

    def __init__(
        self,
        boosting_rounds=100,
        learning_rate=0.03,
        max_depth=10,
        minimum_population_size=2,
        minimum_gain=0.001,
        categorical=False,
        adjacent=False,
        information_gain=0.01,
        random_criterion=None,
        tree_type=ANonSeriousDecisionTree.TreeType.REGRESSION,
    ):
        self.boosting_rounds = boosting_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.minimum_population_size = minimum_population_size
        self.minimum_gain = minimum_gain
        self.categorical = categorical
        self.adjacent = adjacent
        self.information_gain = information_gain
        self.random_criterion = random_criterion
        self.tree_type = tree_type

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        
        self.trees = []

        self.F_0x = np.mean(y)
        self.F_x = np.repeat(self.F_0x, y.shape[0])

        for _ in range(self.boosting_rounds):
            pseudo_residual = -self.dloss(y, self.F_x)
            tree = ANonSeriousDecisionTree(
                max_depth=self.max_depth,
                minimum_population_size=self.minimum_population_size,
                minimum_gain=self.minimum_gain,
                categorical=self.categorical,
                adjacent=self.adjacent,
                information_gain=self.information_gain,
                random_criterion=self.random_criterion,
                tree_type=self.tree_type,
            )
            tree.fit(X, pseudo_residual)
            self.trees.append(tree)

            self.F_x += self.learning_rate * tree.predict(X)
        return self

    def loss(self, y, F_x):
        return 1 / 2 * (y - F_x) ** 2

    def dloss(self, y, F_x):
        return F_x - y

    def predict(self, X):
        if self.F_0x is None:
            raise RuntimeError("Model is not fit")
        prediction = np.repeat(self.F_0x, X.shape[0])
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return prediction
    
    def visualize(self):
        os.makedirs("img", exist_ok=True)
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
        plt.savefig("img/prdctns.png")
        plt.show()
        
        residuals = y - y_pred

        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(0, linestyle="--")

        plt.xlabel("Predicted y")
        plt.ylabel("Residual: y - prediction")
        plt.title("Residual Plot")
        plt.grid(True)
        plt.savefig("img/rsdl.png")
        plt.show()
        
        losses = []

        current_pred = np.repeat(self.F_0x, X.shape[0])

        initial_mse = np.mean((y - current_pred) ** 2)
        losses.append(initial_mse)

        for tree in self.trees:
            current_pred += self.learning_rate * tree.predict(X)
            mse = np.mean((y - current_pred) ** 2)
            losses.append(mse)

        plt.figure(figsize=(7, 5))
        plt.plot(losses)

        plt.xlabel("Boosting Round")
        plt.ylabel("Training MSE")
        plt.title("Training Loss Over Boosting Rounds")
        plt.grid(True)
        plt.savefig("img/lsss.png")
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
        plt.savefig("img/prdctn_srfc.png")
        plt.show()
        
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(xx1, xx2, zz, alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], y, alpha=0.7)

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target / Prediction")
        ax.set_title("Learned Regression Surface")
        
        plt.savefig("img/3d_srfc.png")
        plt.show()

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=300,
        n_features=2,
        noise=15,
        random_state=42,
    )

    gradient_boost = GradientBoosting()
    gradient_boost.fit(X, y)
    
    gradient_boost.visualize()
