import sys
import numpy as np
from scipy.stats import mode

sys.path.append("..")
from src.data import DataType
from src.tree import DecisionTreeClassification, DecisionTreeRegression


class RandomForest:
    def __init__(
        self,
        data_type: DataType = DataType.REGRESSION,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        num_trees: int = 10,
        partial_sampling: float = 0.75,
        max_split_candidates: int = None,
        quantiles: list = None,
        random_state: int = 1,
    ) -> None:
        if data_type.name not in [DataType.CLASSIFICATION.name, DataType.REGRESSION.name]:
            raise ValueError(
                "Invalid value for self.type. Expected 'Classification' or 'Regression', but got '{}'.".format(
                    data_type
                )
            )

        self.data_type = data_type
        self.max_depth = max_depth
        self.seed = random_state
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.num_trees = num_trees
        self.partial_sampling = partial_sampling
        self.max_split_candidates = max_split_candidates
        self.quantiles = quantiles

        self.X_train = None
        self.X_test = None
        self.estimators_ = None

    def set_attributes(self, **kwargs):
        """
        Method to set multiple attributes of the class instance.
        Usage: set_attributes(attr1=value1, attr2=value2, ...)
        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        colnms: list,
        max_split_candidates: int = None,
        partial_sampling: float = 0.75,
        num_trees: int = 100,
        max_depth: int = 2,
        quantiles: list = None,
        min_data_in_leaf: int = 5,
        max_leaf_nodes: int = -1,
    ):
        n_samples = int(partial_sampling * len(y))

        trees = [None] * num_trees
        seeds = list(range(num_trees))

        # TODO: implement with threads to make this parrallel
        for i in range(num_trees):
            seed = seeds[i]

            # TODO: check if this should be with or without replacement
            np.random.seed(seed)
            row_idxs = np.random.choice(range(len(y)), size=n_samples, replace=True)
            X_samp = X[row_idxs, :]
            y_samp = y[row_idxs]  # Y NEEDS TO BE ARRAY FOR THIS TO WORK

            if self.data_type.name == DataType.CLASSIFICATION.name:
                tree_model = DecisionTreeClassification(
                    max_depth=max_depth,
                    min_samples_leaf=min_data_in_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    max_split_candidates=max_split_candidates,
                    quantiles=quantiles,
                    random_state=self.seed + i,
                )
            elif self.data_type.name == DataType.REGRESSION.name:
                tree_model = DecisionTreeRegression(
                    max_depth=max_depth,
                    min_samples_leaf=min_data_in_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    max_split_candidates=max_split_candidates,
                    quantiles=quantiles,
                    random_state=self.seed + i,
                )
            else:
                raise ValueError(
                    "Invalid value for self.type. Expected 'Classification' or 'Regression', but got '{}'.".format(
                        self.data_type
                    )
                )

            tree_model.fit(X=X_samp, y=y_samp, colnms=colnms)

            trees[i] = tree_model

        self.estimators_ = trees

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        colnms: list = None,
    ):
        assert X.shape[0] == len(y), "X and y must have same number of samples"

        self.X_train = X
        self.y_train = y

        self.forest(
            X=self.X_train,
            y=self.y_train,
            colnms=colnms,
            max_split_candidates=self.max_split_candidates,
            partial_sampling=self.partial_sampling,
            num_trees=self.num_trees,
            max_depth=self.max_depth,
            quantiles=self.quantiles,
            min_data_in_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
        )

        return self

    def predict(self, X_test: np.ndarray):
        assert (
            self.X_train.shape[-1] == X_test.shape[-1]
        ), "X test must have same number features as X train"
        # get preds for all trees
        all_preds = np.array([tree.predict(X_test) for tree in self.estimators_])
        # aggregate pred over the trees
        if self.data_type.name == DataType.CLASSIFICATION.name:
            return mode(all_preds, axis=0, keepdims=True).mode.squeeze()
        else:
            return np.mean(all_preds, axis=0)
