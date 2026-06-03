import numpy as np


class ANonSeriousSymmetricalTree:
    def __init__(
        self,
        max_depth=3,
        minimum_split_size=1,
        minimum_gain=0.0,
    ):
        self.max_depth = max_depth
        self.minimum_split_size = minimum_split_size
        self.minimum_gain = minimum_gain
        self.splits = []
        self.leaf_values = None
        self.leaf_indices_train_ = None
    
    def _compute_leaf_indices(self, X):
        leaf_indices = np.zeros(X.shape[0], dtype=int)
        for depth, (feature, threshold) in enumerate(self.splits):
            decisions = X[:, feature] > threshold
            leaf_indices += decisions.astype(int) * (2 ** depth)
        return leaf_indices
    
    def predict(self, X):
        if self.leaf_values is None:
            raise RuntimeError("Tree is not fit")
        leaf_indices = self._compute_leaf_indices(X)
        return self.leaf_values[leaf_indices]
    
    def sse_mean(self, y):
        if len(y) == 0:
            return 0.0
        mean = np.mean(y)
        return np.sum((y - mean) ** 2)
    
    def _candidate_thresholds(self, X, feature):
        values = X[:, feature]
        unique_values = np.unique(values)
        
        if len(unique_values) <= 1:
            return np.array([])
        return (unique_values[1:] + unique_values[:-1]) / 2
    
    def _score_candidate(self, X, y, leaf_indices, feature, threshold, depth):
        total_score = 0.0
        number_of_current_leaves = 2 ** depth
        
        values = X[:, feature]
        decision = values > threshold
        
        for leaf in range(number_of_current_leaves):
            rows_in_leaf = leaf_indices == leaf
            
            if np.sum(rows_in_leaf) == 0:
                continue
            left_rows = rows_in_leaf & decision
            right_rows = rows_in_leaf & ~decision
            
            if (
                np.sum(left_rows) < self.minimum_split_size
                or np.sum(right_rows) < self.minimum_split_size
            ):
                return float("inf")
            total_score += self.sse_mean(y[left_rows])
            total_score += self.sse_mean(y[right_rows])
        return total_score
    
    def _current_score(self, y, leaf_indices, depth):
        total_score = 0.0
        number_of_current_leaves = 2 ** depth
        
        for leaf in range(number_of_current_leaves):
            rows = leaf_indices == leaf
            
            if np.sum(rows) == 0:
                continue
            total_score += self.sse_mean(y[rows])
        return total_score
    
    def _find_best_split(self, X, y, leaf_indices, depth):
        best_feature = None
        best_threshold = None
        best_score = float("inf")
        
        number_of_features = X.shape[1]
        
        for feature in range(number_of_features):
            thresholds = self._candidate_thresholds(X, feature)
            
            for threshold in thresholds:
                score = self._score_candidate(
                    X=X,
                    y=y,
                    leaf_indices=leaf_indices,
                    feature=feature,
                    threshold=threshold,
                    depth=depth,
                )
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None:
            return None, None, None
        return best_feature, best_threshold, best_score
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must match: {X.shape[0]} != {y.shape[0]}")
        self.splits = []
        leaf_indices = np.zeros(X.shape[0], dtype=int)
        
        for depth in range(self.max_depth):
            current_score = self._current_score(y, leaf_indices, depth)
            
            feature, threshold, best_score = self._find_best_split(
                X=X,
                y=y,
                leaf_indices=leaf_indices,
                depth=depth,
            )
            
            if feature is None:
                break
            improvement = current_score - best_score
            
            if improvement < self.minimum_gain:
                break
            self.splits.append((feature, threshold))
            
            decisions = X[:, feature] > threshold
            leaf_indices += decisions.astype(int) * (2 ** depth)
        self.leaf_indices_train_ = leaf_indices
        self._set_leaf_values(y, leaf_indices)
        
        return self 
    
    def _set_leaf_values(self, y, leaf_indices):
        number_of_leaves = 2 ** len(self.splits)
        self.leaf_values = np.zeros(number_of_leaves, dtype=float)
        
        global_values = np.mean(y)
        
        for leaf in range(number_of_leaves):
            rows = leaf_indices == leaf
            
            if np.sum(rows) == 0:
                self.leaf_values[leaf] = global_values
            else:
                self.leaf_values[leaf] = np.mean(y[rows])
    
    def leaf_correction(self, y, F, correction_function):
        if self.leaf_indices_train_ is None:
            raise RuntimeError("Symmetric tree is not fit")
        for leaf in range(len(self.leaf_values)):
            rows = self.leaf_indices_train_ == leaf
            if np.sum(rows) == 0:
                continue
            self.leaf_values[leaf] = correction_function(y[rows], F[rows])
        return self
    