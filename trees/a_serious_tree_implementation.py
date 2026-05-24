#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
    ):

        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):

        max_depth = self.depth

        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        return max_depth

    def count_nodes_below(self, only_leaves=False):

        count = 0 if only_leaves else 1

        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves=only_leaves)

        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves=only_leaves)

        return count

    def left_child_add_prefix(self, text):

        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):

        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        if self.is_root:
            text = f"root [feature={self.feature}, " f"threshold={self.threshold}]"
        else:
            text = f"-> node [feature={self.feature}, " f"threshold={self.threshold}]"

        if self.left_child is not None:
            text += "\n" + self.left_child_add_prefix(str(self.left_child).rstrip("\n"))

        if self.right_child is not None:
            if self.left_child is None:
                text += "\n"
            text += self.right_child_add_prefix(str(self.right_child).rstrip("\n"))

        return text

    def get_leaves_below(self):
        leaves = []

        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())

        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves

    def update_bounds_below(self):

        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            child.lower = dict(self.lower)
            child.upper = dict(self.upper)

            if child is self.left_child:
                prev = child.lower.get(self.feature, -np.inf)
                child.lower[self.feature] = max(prev, self.threshold)
            else:
                prev = child.upper.get(self.feature, np.inf)
                child.upper[self.feature] = min(prev, self.threshold)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):

        def is_large_enough(x):

            if not hasattr(self, "lower") or self.lower is None or len(self.lower) == 0:
                return np.ones(x.shape[0], dtype=bool)
            checks = np.array(
                [np.greater(x[:, key], self.lower[key]) for key in self.lower.keys()]
            )
            return np.all(checks, axis=0)

        def is_small_enough(x):

            if not hasattr(self, "upper") or self.upper is None or len(self.upper) == 0:
                return np.ones(x.shape[0], dtype=bool)
            checks = np.array(
                [np.less_equal(x[:, key], self.upper[key]) for key in self.upper.keys()]
            )
            return np.all(checks, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0,
        )

    def pred(self, x):

        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):

    def __init__(self, value, depth=None):

        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):

        return self.depth

    def count_nodes_below(self, only_leaves=False):

        return 1

    def __str__(self):

        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def pred(self, x):
        return self.value


class Decision_Tree:

    def __init__(
        self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None
    ):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        return self.root.__str__()

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):

        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

    def update_predict(self):

        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        values = np.array([leaf.value for leaf in leaves], dtype=int)

        def predict_func(A):

            indicators = np.array([leaf.indicator(A) for leaf in leaves], dtype=int)
            return values @ indicators

        self.predict = predict_func

    def pred(self, x):
        return self.root.pred(x)

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):

        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype="bool")
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            str = self.count_nodes(only_leaves=True)
            str2 = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {str}
    - Accuracy on training data : {str2}""")

    def fit_node(self, node):

        node.feature, node.threshold = self.split_criterion(node)
        feat_col = self.explanatory[:, node.feature]
        go_left = feat_col > node.threshold
        left_population = node.sub_population & go_left
        right_population = node.sub_population & (~go_left)
        child_depth = node.depth + 1

        def is_leaf_population(sub_pop):

            n = np.sum(sub_pop)
            if n < self.min_pop:
                return True
            if child_depth >= self.max_depth:
                return True
            y = self.target[sub_pop]
            if y.size > 0 and np.min(y) == np.max(y):
                return True
            return False

        if is_leaf_population(left_population):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)
        if is_leaf_population(right_population):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):

        y = self.target[sub_population]
        value = int(np.argmax(np.bincount(y)))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):

        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):

        preds = self.predict(test_explanatory)
        return np.sum(np.equal(preds, test_target)) / test_target.size

    def possible_thresholds(self, node, feature):

        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):

        x = (self.explanatory[:, feature])[node.sub_population]
        y = self.target[node.sub_population]

        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return (0.0, np.inf)

        classes = np.unique(y)

        c = classes.size
        n = y.size

        y_idx = np.searchsorted(classes, y)
        Y = np.eye(c, dtype=int)[y_idx]

        left_mask = x[:, None] > thresholds[None, :]
        left_counts = left_mask.T.astype(int) @ Y
        total_counts = np.sum(Y, axis=0, keepdims=True)
        right_counts = total_counts - left_counts

        left_tot = np.sum(left_counts, axis=1)
        right_tot = n - left_tot

        left_p = left_counts / left_tot[:, None]
        right_p = right_counts / right_tot[:, None]

        gini_left = 1.0 - np.sum(left_p**2, axis=1)
        gini_right = 1.0 - np.sum(right_p**2, axis=1)
        gini_avg = (left_tot / n) * gini_left + (right_tot / n) * gini_right

        j = int(np.argmin(gini_avg))

        return (float(thresholds[j]), float(gini_avg[j]))

    def Gini_split_criterion(self, node):

        X = np.array(
            [
                self.Gini_split_criterion_one_feature(node, i)
                for i in range(self.explanatory.shape[1])
            ]
        )
        i = int(np.argmin(X[:, 1]))
        return i, X[i, 0]


class Random_Forest:

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.explanatory = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        preds = np.array([pred(explanatory) for pred in self.numpy_preds])

        n_classes = int(preds.max()) + 1
        counts = np.eye(n_classes, dtype=int)[preds]  # (t, n, c)
        votes = counts.sum(axis=0)  # (n, c)

        return votes.argmax(axis=1)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i,
            )
            tree.fit(explanatory, target)
            self.numpy_preds.append(tree.predict)

            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            accuracies.append(tree.accuracy(tree.explanatory, tree.target))

        if verbose == 1:
            forest_acc = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {forest_acc}""")

    def accuracy(self, test_explanatory, test_target):
        preds = self.predict(test_explanatory)
        return np.sum(np.equal(preds, test_target)) / test_target.size


class Isolation_Random_Tree:

    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        return self.root.__str__()

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        self.root.update_bounds_below()

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        values = np.array([leaf.value for leaf in leaves], dtype=float)

        def predict_func(a):
            lst = [leaf.indicator(a) for leaf in leaves]
            indicators = np.array(lst, dtype=int)
            return values @ indicators

        self.predict = predict_func

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        leaf_depth = node.depth + 1
        leaf_child = Leaf(leaf_depth, depth=leaf_depth)
        leaf_child.depth = leaf_depth
        leaf_child.subpopulation = sub_population
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        pop_size = int(np.sum(node.sub_population))
        if pop_size <= self.min_pop or node.depth >= self.max_depth:
            node.left_child = self.get_leaf_child(
                node, np.zeros_like(node.sub_population, dtype=bool)
            )
            a = self.get_leaf_child(node, node.sub_population.copy())
            node.right_child = a
            return
        sub_x = self.explanatory[node.sub_population, :]
        if np.all(np.ptp(sub_x, axis=0) == 0):
            node.feature = 0
            node.threshold = float(sub_x[0, 0])
            node.left_child = self.get_leaf_child(
                node, np.zeros_like(node.sub_population, dtype=bool)
            )
            b = self.get_leaf_child(node, node.sub_population.copy())
            node.right_child = b
            return
        node.feature, node.threshold = self.random_split_criterion(node)
        feat_col = self.explanatory[:, node.feature]
        go_left = feat_col > node.threshold
        left_population = node.sub_population & go_left
        right_population = node.sub_population & (~go_left)
        child_depth = node.depth + 1
        is_left_leaf = (
            child_depth >= self.max_depth or np.sum(left_population) <= self.min_pop
        )
        is_right_leaf = (
            child_depth >= self.max_depth or np.sum(right_population) <= self.min_pop
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):

        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")


class Isolation_Random_Forest:

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.explanatory = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            tree = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i,
            )
            tree.fit(explanatory)
            self.numpy_preds.append(tree.predict)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        depths = self.predict(explanatory)
        idx = np.argsort(depths)[:n_suspects]
        return explanatory[idx], depths[idx]


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

    def np_extrema(arr):
        return np.min(arr), np.max(arr)

    def visualize_bassins(ax, model, x_min, x_max, y_min, y_max, cmap):
        assert T.explanatory.shape[1] == 2, "Not a 2D example"
        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(X, Y)
        XX_flat = XX.ravel()
        YY_flat = YY.ravel()
        Z = model.predict(np.vstack([XX_flat, YY_flat]).T)
        ax.pcolormesh(XX, YY, Z.reshape([100, 100]), cmap=cmap, shading="auto")

    def visualize_training_dataset_2D(ax, model, cmap):
        ax.scatter(
            model.explanatory[:, 0], model.explanatory[:, 1], c=model.target, cmap=cmap
        )

    def visualize_model_2D(model, cmap=plt.cm.Set1):
        assert model.explanatory.shape[1] == 2, "Not a 2D example"

        x_min, x_max = np_extrema(model.explanatory[:, 0])
        y_min, y_max = np_extrema(model.explanatory[:, 1])
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        visualize_training_dataset_2D(axes[0], model, cmap)
        visualize_bassins(axes[1], model, x_min, x_max, y_min, y_max, cmap)
        plt.savefig("bassins1.png")
        plt.show()

    explanatory, target = circle_of_clouds(10, 30)
    T = Decision_Tree(split_criterion="Gini")
    T.fit(explanatory, target, verbose=0)

    visualize_model_2D(T)

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

    def iris():
        iris = datasets.load_iris()
        return iris.data, iris.target

    def wine():
        wine = datasets.load_wine()
        return wine.data, wine.target

    def split(explanatory, target, seed=0, proportion=0.1):
        rng = np.random.default_rng(seed)
        test_indices = rng.choice(
            target.size, int(target.size * proportion), replace=False
        )
        test_filter = np.zeros_like(target, dtype="bool")
        test_filter[test_indices] = True

        return {
            "train_explanatory": explanatory[np.logical_not(test_filter), :],
            "train_target": target[np.logical_not(test_filter)],
            "test_explanatory": explanatory[test_filter, :],
            "test_target": target[test_filter],
        }

    for d, name in zip(
        [split(*circle_of_clouds(10, 30)), split(*iris()), split(*wine())],
        ["circle of clouds", "iris dataset", "wine dataset"],
    ):
        print("-" * 52 + "\n" + name + " :")
        F = Random_Forest(max_depth=6)
        F.fit(d["train_explanatory"], d["train_target"], verbose=1)
        print(
            f"    - Accuracy on test          : {F.accuracy(d['test_explanatory'],d['test_target'])}"
        )
    print("-" * 52)

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

    def np_extrema(arr):
        return np.min(arr), np.max(arr)

    def visualize_bassins(ax, model, x_min, x_max, y_min, y_max, cmap):
        assert model.explanatory.shape[1] == 2, "Not a 2D example"
        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(X, Y)
        XX_flat = XX.ravel()
        YY_flat = YY.ravel()
        Z = model.predict(np.vstack([XX_flat, YY_flat]).T)
        ax.pcolormesh(XX, YY, Z.reshape([100, 100]), cmap=cmap, shading="auto")

    def visualize_training_dataset_2D(ax, model, cmap):
        ax.scatter(
            model.explanatory[:, 0], model.explanatory[:, 1], c=model.target, cmap=cmap
        )

    def visualize_model_2D(model, cmap=plt.cm.Set1):
        assert model.explanatory.shape[1] == 2, "Not a 2D example"

        x_min, x_max = np_extrema(model.explanatory[:, 0])
        y_min, y_max = np_extrema(model.explanatory[:, 1])
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        visualize_training_dataset_2D(axes[0], model, cmap)
        visualize_bassins(axes[1], model, x_min, x_max, y_min, y_max, cmap)
        plt.savefig("bassins2.png")
        plt.show()

    explanatory, target = circle_of_clouds(10, 30)
    F = Random_Forest()
    F.fit(explanatory, target, verbose=0)

    visualize_model_2D(F)

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

    def np_extrema(arr):
        return np.min(arr), np.max(arr)

    def visualize_bassins(ax, model, x_min, x_max, y_min, y_max, cmap):
        assert model.explanatory.shape[1] == 2, "Not a 2D example"
        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(X, Y)
        XX_flat = XX.ravel()
        YY_flat = YY.ravel()
        Z = model.predict(np.vstack([XX_flat, YY_flat]).T)
        ax.pcolormesh(XX, YY, Z.reshape([100, 100]), cmap=cmap, shading="auto")

    def visualize_training_dataset_2D(ax, model, cmap):
        ax.scatter(
            model.explanatory[:, 0], model.explanatory[:, 1], c=model.target, cmap=cmap
        )

    def visualize_model_2D(model, cmap=plt.cm.Set1):
        assert model.explanatory.shape[1] == 2, "Not a 2D example"

        x_min, x_max = np_extrema(model.explanatory[:, 0])
        y_min, y_max = np_extrema(model.explanatory[:, 1])
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        visualize_training_dataset_2D(axes[0], model, cmap)
        visualize_bassins(axes[1], model, x_min, x_max, y_min, y_max, cmap)
        plt.savefig("bassins3.png")
        plt.show()

    explanatory, _ = circle_of_clouds(1, 100, sigma=0.2)  # a cloud
    explanatory[0] = [-1, 0]  # an outlier

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    axes[0, 0].scatter(explanatory[:, 0], explanatory[:, 1])
    axes[0, 0].set_title("a cloud and an outlier")
    for i in range(1, 9):
        T = Isolation_Random_Tree(max_depth=8, seed=i, root=None)
        T.fit(explanatory)
        visualize_bassins(
            axes[i % 3, i // 3], T, -1.2, 1.5, -0.5, 0.5, cmap=plt.cm.RdBu
        )
        axes[i % 3, i // 3].set_title(f"bassins of isolation tree for seed={i}")
    plt.savefig("bassins3.png")
    plt.show()

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

    def np_extrema(arr):
        return np.min(arr), np.max(arr)

    def visualize_bassins(ax, model, x_min, x_max, y_min, y_max, cmap):
        assert model.explanatory.shape[1] == 2, "Not a 2D example"
        X = np.linspace(x_min, x_max, 100)
        Y = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(X, Y)
        XX_flat = XX.ravel()
        YY_flat = YY.ravel()
        Z = model.predict(np.vstack([XX_flat, YY_flat]).T)
        ax.pcolormesh(XX, YY, Z.reshape([100, 100]), cmap=cmap, shading="auto")

    def visualize_training_dataset_2D(ax, model, cmap):
        ax.scatter(
            model.explanatory[:, 0], model.explanatory[:, 1], c=model.target, cmap=cmap
        )

    def visualize_model_2D(model, cmap=plt.cm.Set1):
        assert model.explanatory.shape[1] == 2, "Not a 2D example"

        x_min, x_max = np_extrema(model.explanatory[:, 0])
        y_min, y_max = np_extrema(model.explanatory[:, 1])
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        visualize_training_dataset_2D(axes[0], model, cmap)
        visualize_bassins(axes[1], model, x_min, x_max, y_min, y_max, cmap)
        plt.savefig("bassins4.png")
        plt.show()

    explanatory, _ = circle_of_clouds(3, 100, sigma=0.2)
    IRF = Isolation_Random_Forest(max_depth=15)
    IRF.fit(explanatory, verbose=1)
    suspects, depths = IRF.suspects(explanatory, n_suspects=3)
    print("suspects :", suspects)
    print("depths of suspects :", depths)

    visualize_model_2D(IRF, cmap="RdBu")
