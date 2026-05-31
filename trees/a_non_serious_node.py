import numpy as np


class ANonSeriousNode:
    def __init__(self, feature=None, threshold=None):
        self.feature = feature
        self.threshold = threshold
        self.is_leaf = False
        self.left = None
        self.right = None
        self.value = None
        self.majority_class = None
        self.number_of_samples = None
        self.number_of_classes = None
        self.number_of_leaves = None
        self.leaf_error = None
        self.subtree_error = None
        self.is_root = False
        self.depth = None
        self.max_depth = float("inf")
        self.max_depth_reached = "MAX_DEPTH_REACHED"
        self.categorical_split = False
        self.gradient = None
        self.hessian = None
        self.default_missing_value_direction = None

    def left_add_prefix(self, text):
        if self.depth > self.max_depth:
            return self.max_depth_reached
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_add_prefix(self, text):
        if self.depth > self.max_depth:
            return self.max_depth_reached

        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        if self.depth > self.max_depth:
            return self.max_depth_reached

        if self.is_root:
            text = f"root [feature={self.feature}, " f"threshold={self.threshold}]"
        else:
            text = f"-> node [feature={self.feature}, " f"threshold={self.threshold}]"

        if self.left is not None:
            text += "\n" + self.left_add_prefix(str(self.left).rstrip("\n"))

        if self.right is not None:
            if self.left is None:
                text += "\n"
            text += self.right_add_prefix(str(self.right).rstrip("\n"))

        return text

    def max_depth_below(self):
        max_depth = self.depth

        if self.left is not None:
            max_depth = max(max_depth, self.left.max_depth_below())
        if self.right is not None:
            max_depth = max(max_depth, self.right.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        if only_leaves:
            if self.left is None and self.right is None:
                return 1

            count = 0
        else:
            count = 1

        if self.left is not None:
            count += self.left.count_nodes_below(only_leaves=only_leaves)

        if self.right is not None:
            count += self.right.count_nodes_below(only_leaves=only_leaves)

        return count

    def get_leaves_below(self):
        leaves = []
        if self.is_leaf or self.value is not None:
            return [self]

        if self.left is not None:
            leaves.extend(self.left.get_leaves_below())

        if self.right is not None:
            leaves.extend(self.right.get_leaves_below())

        return leaves

    def update_bounds_below(self):
        if self.is_root:
            # self.upper = {0: np.inf}
            # self.lower = {0: -np.inf}
            self.upper = {}
            self.lower = {}

        if self.left is not None:
            self.left.lower = dict(self.lower)
            self.left.upper = dict(self.upper)

            previous = self.left.lower.get(self.feature, -np.inf)
            self.left.lower[self.feature] = max(previous, self.threshold)
            self.left.update_bounds_below()
        if self.right is not None:
            self.right.lower = dict(self.lower)
            self.right.upper = dict(self.upper)

            previous = self.right.upper.get(self.feature, np.inf)
            self.right.upper[self.feature] = min(previous, self.threshold)
            self.right.update_bounds_below()

    def update_indicator(self):

        def is_large_enough(X):
            if not hasattr(self, "lower") or self.lower is None or len(self.lower) == 0:
                return np.ones(X.shape[0], dtype=np.bool_)
            checks = np.array(
                [np.greater(X[:, key], self.lower[key]) for key in self.lower.keys()]
            )
            return np.all(checks, axis=0)

        def is_small_enough(X):
            if not hasattr(self, "upper") or self.upper is None or len(self.upper) == 0:
                return np.ones(X.shape[0], dtype=np.bool_)
            checks = np.array(
                [np.less_equal(X[:, key], self.upper[key]) for key in self.upper.keys()]
            )
            return np.all(checks, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0,
        )
