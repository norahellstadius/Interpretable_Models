import random
import numpy as np
from typing import Callable, Tuple, Union, List


class SplitPoint:
    def __init__(
        self,
        feature: int,
        value: float,
        feature_name: str,
        improved_score: float = None,
    ) -> None:
        """
        A location where the tree splits.

        Attributes:
        - feature: Feature index.
        - split_value: Value of split.
        - feature_name: Name of the feature which is used for pretty printing.
        """

        self.feature = feature  # feature index
        self.split_value = value
        self.feature_name = feature_name
        self.improved_score = improved_score


class Node:
    def __init__(
        self,
        split_point: SplitPoint,
        left,
        right,
        val=None,
        is_leaf: bool = False,
        depth: int = 0,
    ) -> None:
        self.splitpoint = split_point
        self.right = left
        self.left = right
        self.val = val
        self.is_leaf = is_leaf
        self.depth = depth

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf Node - Value: {self.val}"
        else:
            return f"Internal Node - Depth: {self.depth}, Feature: {self.splitpoint.feature}, Splitvalue {self.splitpoint.split_value}"


class DecisionTree:
    def __init__(
        self,
        max_depth: int = None,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        max_split_candidates: int = None,
        quantiles: list = None,
        random_state: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = random_state
        self.max_split_candidates = max_split_candidates
        self.quantiles = quantiles
        self.max_leaf_nodes = max_leaf_nodes
        self.root_node = None  # to save the fitted tree

    @staticmethod
    def get_max_split_candidates(X: np.ndarray):
        """
        Calculate the maximum split candidates for a given input.

        Parameters:
        - X: int
            The input value (representing the number of features).

        Returns:
        - int
            The rounded square root of the input value.
        """
        _, p = X.shape
        return round(np.sqrt(p))

    @staticmethod
    def _split_y(
        data: np.ndarray,
        y: np.ndarray,
        comparison: Callable[[float, float], bool],
        cutpoint: float,
    ) -> np.ndarray:
        """
        Returns a `y` for which the `comparison` holds in `data`.

        Parameters:
        - y_new (array): Mutable array to store `y` for which the `comparison` holds in `data`.
        - data (iterable): Array for comparison.
        - y (iterable): Original array.
        - comparison (function): A function to determine if a comparison holds.
        - cutpoint: A value against which the comparison is made.

        Returns:
        - y_new (array): `y_new` contains the valid elements.
        """

        indices = np.where(comparison(data, cutpoint))
        y_new = y[indices]

        return y_new

    @staticmethod
    def _split_data(
        X: np.ndarray,
        y: np.ndarray,
        splitpoint: SplitPoint,
        comparison: Callable[[float, float], bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the feature matrix and target vector based on a given split value and comparison.

        Parameters:
        - X: numpy.ndarray
            The feature matrix.
        - y: numpy.ndarray
            The target vector.
        - splitpoint: SplitPoint
            The SplitPoint object specifying the feature and split value.
        - comparison: function
            A comparison function that takes a value and the split value,
            returning a boolean result.

        Returns:
        - Tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing the split feature matrix (X_split) and the split target vector (y_split).

        Raises:
        - AssertionError: If the length of the data and target vector does not match.
        """

        data = X[:, splitpoint.feature]
        assert len(data) == len(y), "Length mismatch between data and target vector."

        mask = comparison(data, splitpoint.split_value)
        X_split = X[mask, :]
        y_split = y[mask]

        return X_split, y_split

    def _get_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        colnms: np.ndarray,
        quantiles: np.ndarray,
        max_split_candidates: int,
        seed_add: int = 0,
    ) -> Union[SplitPoint, None]:
        """
        Find the best split point for a decision tree node based on maximising Gini index.

        Parameters:
        - X: numpy.ndarray
            The feature matrix.
        - y: numpy.ndarray
            The target vector.
        - classes: list
            List of unique classes in the target vector.
        - colnms: list
            List of column names corresponding to features in X.
        - quantiles: list of lists
            List of quantiles for each feature, defining potential cutpoints.
        - max_split_candidates: int, optional
            Maximum number of features to consider for splitting.
            If not provided, it defaults to the total number of features in X.

        Returns:
        - Union[None, SplitPoint]
            If a split point is found that improves the Gini index, returns a SplitPoint object.
            Otherwise, returns None.
        """

        score_improved_bool = False
        best_feature_index = 0
        best_split_value = 0.0
        best_score = self._start_score()  # gini the larger the better

        N, p = X.shape

        # get feature indicies to used in the split
        random.seed(self.seed + seed_add)
        possible_features = (
            list(range(0, p))
            if max_split_candidates == p
            else random.sample(list(range(0, p)), max_split_candidates)
        )

        parent_score = self._get_parent_score(
            y, classes
        )  # Data to be re-used in thpe loop on features and splitpoints

        for feature_idx in possible_features:
            # go through all q quantiles of feature (splitting options)
            split_values = (
                X[:, feature_idx] if quantiles is None else quantiles[feature_idx]
            )

            for split_value in split_values:
                # get the left and right y based on the cutpoint
                y_left = self._split_y(X[:, feature_idx], y, np.less, split_value)
                if len(y_left) == 0:
                    continue

                y_right = self._split_y(
                    X[:, feature_idx], y, np.greater_equal, split_value
                )
                if len(y_right) == 0:
                    continue

                assert len(y) == (
                    len(y_left) + len(y_right)
                ), "Splittig of y into y_left and y_right incorrect shape."

                # get weighted gini score for the split
                current_score = self._current_score(
                    y, y_left, y_right, classes, parent_score
                )

                if self._score_improved(best_score, current_score):
                    score_improved_bool = True
                    best_score = current_score
                    best_feature_index = feature_idx
                    best_split_value = split_value

        if score_improved_bool:
            feature_name = colnms[best_feature_index] if colnms is not None else None
            return SplitPoint(
                feature=best_feature_index,
                value=best_split_value,
                feature_name=feature_name,
                improved_score=current_score - best_score,
            )
        else:
            return None

    def tree_DepthFirstBuilder(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        max_split_candidates: int,
        colnms=None,
        depth: int = 0,
        max_depth: int = 2,
        quantiles=None,
        min_samples_leaf: int = 5,
        seed_add: int = 0,
    ) -> Node:
        """
        Build a decision tree recursively.

        Parameters:
        - X (array-like): Input features.
        - y (array-like): Target values.
        - classes (list): List of unique class labels.
        - colnms (list, optional): List of column names for the input features.
        - max_split_candidates (int, optional): Maximum number of split candidates to consider.
        - depth (int, optional): Current depth of the tree.
        - max_depth (int, optional): Maximum depth of the tree.
        - q (int, optional): Number of quantiles to use for cutpoints.
        - cps (array-like, optional): Precomputed cutpoints for splitting.
        - min_samples_leaf (int, optional): Minimum number of data points in a leaf node.

        Returns:
        - Node: The root node of the decision tree.
        """

        num_samples = X.shape[0]

        if num_samples >= min_samples_leaf and depth < max_depth:
            sp = self._get_best_split(
                X=X,
                y=y,
                classes=classes,
                colnms=colnms,
                quantiles=quantiles,
                max_split_candidates=max_split_candidates,
                seed_add=seed_add + depth,
            )

            if sp is not None:
                depth += 1

                right = self.tree_DepthFirstBuilder(
                    *self._split_data(X, y, sp, np.greater_equal),
                    classes=classes,
                    max_split_candidates=max_split_candidates,
                    colnms=colnms,
                    max_depth=max_depth,
                    quantiles=quantiles,
                    depth=depth,
                    seed_add=depth
                    + 90000,  # adding 90000 so it wont be the same seed as left
                )
                left = self.tree_DepthFirstBuilder(
                    *self._split_data(X, y, sp, np.less),
                    classes=classes,
                    max_split_candidates=max_split_candidates,
                    colnms=colnms,
                    max_depth=max_depth,
                    quantiles=quantiles,
                    depth=depth,
                    seed_add=depth + 2,
                )

                # TODO figure out why it needs to be reversed with righ and left
                return Node(sp, right, left, self._get_node_value(y, classes), False)

        pred_val = self._get_node_value(y, classes)
        return Node(None, None, None, pred_val, True)

    def tree_BestFirstBuilder(
        self,
        X,
        y,
        classes: np.ndarray,
        max_split_candidates: int,
        colnms=None,
        max_depth=None,
        quantiles=None,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = 5,
    ):
        # Initialize the frontier with the root node
        frontier = []

        # add root to frontier
        sp = self._get_best_split(
            X=X,
            y=y,
            classes=classes,
            colnms=colnms,
            quantiles=quantiles,
            max_split_candidates=max_split_candidates,
            seed_add=0,
        )
        if sp is not None:
            root_node = Node(split_point=sp, left=None, right=None, depth=0)
            frontier.append([root_node, (X, y)])

        max_split_nodes = max_leaf_nodes - 1

        # Build the tree in a best-first fashion
        while len(frontier) > 0:
            frontier.sort(key=lambda x: x[0].splitpoint.improved_score, reverse=False)
            curr_node, (X_curr, y_curr) = frontier.pop(0)

            num_samples = len(y_curr)
            # it must be a leaf node is it excides max depth
            if (
                (max_depth is not None and curr_node.depth >= max_depth)
                or num_samples <= min_samples_leaf
                or max_split_nodes <= 0
            ):
                curr_node.is_leaf = True
                curr_node.splitpoint = None
                curr_node.val = self._get_node_value(y_curr, classes)

            else:
                if curr_node.splitpoint is not None:
                    max_split_nodes -= 1

                    X_left, y_left = self._split_data(
                        X_curr, y_curr, curr_node.splitpoint, np.less
                    )
                    X_right, y_right = self._split_data(
                        X_curr, y_curr, curr_node.splitpoint, np.greater_equal
                    )

                    left_sp = self._get_best_split(
                        X_left,
                        y_left,
                        classes=classes,
                        max_split_candidates=max_split_candidates,
                        colnms=colnms,
                        quantiles=quantiles,
                        seed_add=curr_node.depth + 1,
                    )

                    if left_sp is not None:
                        left_node = Node(left_sp, None, None, depth=curr_node.depth + 1)
                        curr_node.left = left_node
                        frontier.append([left_node, (X_left, y_left)])

                    right_sp = self._get_best_split(
                        X_right,
                        y_right,
                        classes=classes,
                        max_split_candidates=max_split_candidates,
                        colnms=colnms,
                        quantiles=quantiles,
                        seed_add=curr_node.depth + 90000,
                    )

                    if right_sp is not None:
                        right_node = Node(
                            right_sp, None, None, depth=curr_node.depth + 1
                        )
                        curr_node.right = right_node
                        frontier.append([right_node, (X_right, y_right)])

                    if right_sp is None or left_sp is None:
                        curr_node.is_leaf = True
                        curr_node.left = None
                        curr_node.right = None
                        curr_node.val = self._get_node_value(y_curr, classes)

        return root_node

    def fit(self, X: np.ndarray, y: np.ndarray, colnms=None) -> Node:
        assert len(X) == len(y), "Length mismatch between data and target vector."
        if not self.check_valid_y(y):
            raise ValueError("Invalid target variable for the problem.")
        self.X = X
        self.y = y
        self.classes = self._get_classes(y)

        if self.max_split_candidates is None:
            self.max_split_candidates = self.get_max_split_candidates(X)

        if self.max_leaf_nodes < 0:
            self.root_node = self.tree_DepthFirstBuilder(
                X=self.X,
                y=self.y,
                classes=self.classes,
                colnms=colnms,
                max_split_candidates=self.max_split_candidates,
                depth=0,
                max_depth=self.max_depth,
                quantiles=self.quantiles,
                min_samples_leaf=self.min_samples_leaf,
            )
        else:
            self.root_node = self.tree_BestFirstBuilder(
                X=self.X,
                y=self.y,
                classes=self.classes,
                max_split_candidates=self.max_split_candidates,
                colnms=colnms,
                max_depth=self.max_depth,
                quantiles=self.quantiles,
                min_samples_leaf=self.min_samples_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
            )

        return self

    def _predict_one_sample(self, node: Node, x):
        """
        Predict the output for a single sample given the root node of the decision tree.

        Parameters:
        - node (Node): The root node of the decision tree.
        - x (np.ndarray): The input features for a single sample.

        Returns:
        - The predicted output for the given sample.
        """
        if node.is_leaf:
            return node.val
        else:
            if np.less(x[node.splitpoint.feature], node.splitpoint.split_value):
                return self._predict_one_sample(node.left, x)
            else:
                return self._predict_one_sample(node.right, x)

    def print_decision_tree(self, root, level=0, operation="<"):
        if root is not None:
            if root.is_leaf:
                text = "Left Leaf Node" if operation == "<" else "Right Leaf Node"
                print(" " * (level * 4), f"{text}: {root.val}")
            else:
                print(
                    " " * (level * 4),
                    f"X[{root.splitpoint.feature}] " + operation,
                    f"{root.splitpoint.split_value}",
                )
            if root.left:
                self.print_decision_tree(root.left, level + 1, "<")
            if root.right:
                self.print_decision_tree(root.right, level + 1, ">=")


# -------- Classification ---------------
def _count_equal(y: np.ndarray, label_type: np.ndarray) -> int:
    """
    Count occurrences of a specific class label in a list.

    Parameters:
    - y (List): List of class labels.
    - label_type (Any): Specific class label to count.

    Returns:
    - int: Count of occurrences of the specified class label in the list.
    """
    count = sum(1 for label in y if label == label_type)
    return count


def _gini(y: np.ndarray, classes: np.ndarray) -> float:
    """
    Calculate Gini index for multiclass classification.

    Parameters:
    - y: List of class labels.
    - classes: List of unique classes.

    Returns:
    - Gini index.
    """
    total_samples = len(y)

    # Calculate the Gini index
    gini_impurity = 1.0
    for c in classes:
        p_c = _count_equal(y, c) / total_samples
        gini_impurity -= p_c**2

    return gini_impurity


def _weighted_gini(
    y: np.ndarray, yl: np.ndarray, yr: np.ndarray, classes: np.ndarray
) -> float:
    """
    Calculate the weighted Gini index for a binary split.

    Parameters:
    - y: List of class labels.
    - yl: List of class labels for the left split.
    - yr: List of class labels for the right split.
    - classes: List of unique classes.

    Returns:
    - Weighted Gini index.
    """
    # Check if proportions add up to 1
    if abs(len(yl) / len(y) + len(yr) / len(y) - 1) > 1e-10:
        raise ValueError("Proportions of yl and yr must add up to 1.")

    p = len(yl) / len(y)
    weighted_gini = p * _gini(yl, classes) + (1 - p) * _gini(yr, classes)
    return weighted_gini


def _information_gain(
    y: np.ndarray,
    yl: np.ndarray,
    yr: np.ndarray,
    classes: np.ndarray,
    starting_impurity: float,
) -> float:
    """
    Calculate information gain for a binary split.

    Parameters:
    - y: List of class labels.
    - yl: List of class labels for the left split.
    - yr: List of class labels for the right split.
    - classes: List of unique classes.
    - starting_impurity: Initial impurity measure (e.g., Gini index).

    Returns:
    - Information gain.
    """
    impurity_change = _weighted_gini(y, yl, yr, classes)
    return starting_impurity - impurity_change


class DecisionTreeClassification(DecisionTree):
    def __init__(
        self,
        max_depth=None,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        max_split_candidates=None,
        quantiles=None,
        random_state=1,
    ) -> None:
        super().__init__(
            max_depth,
            min_samples_leaf,
            max_leaf_nodes,
            max_split_candidates,
            quantiles,
            random_state,
        )

    @staticmethod
    def check_valid_y(y: np.ndarray) -> bool:
        # Check if y is a numpy array
        if not isinstance(y, np.ndarray):
            return False

        # Check if y contains only integers or strings
        if not np.issubdtype(y.dtype, np.integer) and not np.issubdtype(y.dtype, np.unicode):
            return False

        # Check if y contains at least two unique values
        if len(np.unique(y)) < 2:
            return False

        return True

    @staticmethod
    def _score_improved(best_score: float, current_score: float) -> bool:
        """
        Check if the current score is an improvement over the best score.
        The bigger the better

        Parameters:
        - best_score: The best (lowest or highest, depending on the context) score achieved so far.
        - current_score: The current score to compare with the best score.

        Returns:
        - True if the current score is an improvement, False otherwise.
        """
        return current_score >= best_score

    "Data to be re-used in the loop on features and splitpoints. In julia library they call it _reused_data "

    @staticmethod
    def _get_parent_score(y: np.ndarray, classes: np.ndarray) -> float:
        return _gini(y, classes)

    @staticmethod
    def _current_score(
        y: np.ndarray,
        yl: np.ndarray,
        yr: np.ndarray,
        classes: np.ndarray,
        reused_data: float,
    ) -> float:
        return _information_gain(y, yl, yr, classes, reused_data)

    @staticmethod
    def _get_node_value(y: np.ndarray, classes: np.ndarray):
        num_points = len(y)
        probabilities = [_count_equal(y, c) / num_points for c in classes]
        return probabilities

    @staticmethod
    def _get_classes(y: np.ndarray):
        unique_sorted_y = sorted(set(y))
        return unique_sorted_y

    @staticmethod
    def _start_score():
        """Return the start score for the maximization problem.
        The larger the better for gini bounded by[0, 1]
        """
        return 0

    def predict_proba(self, X):
        """Returns the predicted probabilities for a given data set"""

        n_samples = X.shape[0]
        n_classes = len(
            self.classes
        )  # Assuming self.classes_ contains the unique classes in your model

        predictions = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            predictions[i, :] = self._predict_one_sample(self.root_node, X[i, :])

        return predictions

    def predict(self, X_set):
        """Returns the predicted probs for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        return preds


# ----------- Regression -----------
class DecisionTreeRegression(DecisionTree):
    def __init__(
        self,
        max_depth=None,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        max_split_candidates=None,
        quantiles=None,
        random_state=1,
    ) -> None:
        super().__init__(
            max_depth,
            min_samples_leaf,
            max_leaf_nodes,
            max_split_candidates,
            quantiles,
            random_state,
        )

    @staticmethod
    def check_valid_y(y: np.ndarray) -> bool:
        # Check if y is a numpy array
        if not isinstance(y, np.ndarray):
            return False

        # Check if y contains numerical data
        if not np.issubdtype(y.dtype, np.number):
            return False

        # Check if y contains at least two unique values
        if len(np.unique(y)) < 2:
            return False

        return True
    

    @staticmethod
    def _score_improved(best_score: float, current_score: float) -> bool:
        """
        Check if the current score is an improvement over the best score.
        The smaller the better

        Parameters:
        - best_score: The best (lowest or highest, depending on the context) score achieved so far.
        - current_score: The current score to compare with the best score.

        Returns:
        - True if the current score is an improvement, False otherwise.
        """
        return current_score >= best_score

    "Data to be re-used in the loop on features and splitpoints. In julia library they call it _reused_data "

    @staticmethod
    def _get_parent_score(y: list, classes: list = []) -> float:
        return np.var(y)

    @staticmethod
    def _current_score(
        y: list, yl: list, yr: list, classes: list = [], reused_data: float = 0
    ) -> float:
        if abs(len(yl) / len(y) + len(yr) / len(y) - 1) > 1e-10:
            raise ValueError("Proportions of yl and yr must add up to 1.")

        p = len(yl) / len(y)
        return reused_data - (p * np.var(yl) + (1 - p) * np.var(yr))

    @staticmethod
    def _get_node_value(y, classes=[]):
        pred_val = [np.mean(y)]
        return pred_val

    @staticmethod
    def _start_score():
        return 0

    @staticmethod
    def _get_classes(y: np.ndarray):
        """Return an empty array since there are no classes in regression"""
        return []

    def predict(self, X):
        """Returns the predicted avg for a given data set"""

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            predictions[i] = self._predict_one_sample(self.root_node, X[i, :])[0]
        return predictions
