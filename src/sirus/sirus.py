import sys
import numpy as np
from enum import Enum

sys.path.append("../..")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.rules import (
    get_rules_count,
    get_rule_feature_matrix,
    get_quantile_rules,
    FilterType,
    Rule,
)
from src.ruleEsemble import RuleEnsembleClassification, RuleEnsembleRegression
from src.forest import RandomForest
from src.linear import fit_ridge
from src.sirus.dependent import filter_linearly_dependent
from src.data import DataType


class Sirus:
    def __init__(
        self,
        threshold: int = 0.072,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        num_trees: int = 100,
        partial_sampling: int = 0.70,
        quantiles: list = None,
        max_split_candidates: int = None,
        random_state: int = 10,
        remove_ld: bool = True,
        filter_type: FilterType = FilterType(1),
    ) -> None:
        self.threshold = threshold
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.num_trees = num_trees
        self.partial_sampling = partial_sampling
        self.quantiles = quantiles
        self.max_split_candidates = max_split_candidates
        self.seed = random_state
        self.remove_ld = remove_ld  # bool to remove linearly dependent rules
        self.filter_type = filter_type

        if self.filter_type.name == FilterType(2).name and self.quantiles is None:
            raise ValueError(
                "Quantiles cannot be None when filter_type is INSIDE_SAME_QUANTILE"
            )

        self.rf_model = None  # Random forest instance
        self.rules = None
        self.intercept = None
        self.coefs = None
        self.y_train = None
        self.X_train = None
        self.scalar = None  # scale instance for feature matrix
        self.estimators_ = None #same as rules 

    def set_attributes(self, **kwargs):
        """
        Method to set multiple attributes of the Sirus class instance.
        Usage: set_attributes(attr1=value1, attr2=value2, ...)
        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)
    
    def fit_linear_model(self, X: np.ndarray, y: np.ndarray, data_type: DataType):
        self.linear_model = fit_ridge(X, y, data_type)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, colnms: list = None):
        assert X_train.shape[0] == len(
            y_train
        ), "X and y train must have same number of samples"

        self.X_train = X_train
        self.y_train = y_train

        # git random o
        self.fit_rf(
            X_train=self.X_train,
            y_train=self.y_train,
            colnms=colnms,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            num_trees=self.num_trees,
            partial_sampling=self.partial_sampling,
            quantiles=self.quantiles,
            max_split_candidates=self.max_split_candidates,
        )

        tree_list = self.rf_model.estimators_
        all_rules = self.extract_rules(tree_list)  # get all the rules from the trees
        self.rules = self.filter_rules(
            all_rules
        )  # only select rules that base the threshold frequency
        # remove linearly dependent rules
        if self.remove_ld:
            self.rules = filter_linearly_dependent(self.rules)
            self.estimators_ = self.rules #this is double but its for multistudy to get uniform naming

        if len(self.rules) > 0:
            X_feature_matrix = self.get_feature_matrix(self.rules, self.X_train)
            X_feature_matrix_scaled = self.scale_data(X_feature_matrix)
            self.fit_linear_model(
                X_feature_matrix_scaled, self.y_train, self.data_type
            )  # fit ridge and get weights
        else:
            print("0 rules were found")
        
        return self

    # scales data and saved the scalar as an attribut
    def scale_data(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            self.scalar = StandardScaler().fit(X)
        X_scaled = self.scalar.transform(X)
        return X_scaled

    def get_high_frequency_rules(self, rules: list[Rule], num_rules_threshold: float, filter_type: FilterType) -> list[Rule]:
        '''Return a list of rule which has a frequency which is higher or equal to num_rules_threshold'''
        if filter_type == FilterType(1):
            rules_count_dict = get_rules_count(rules)
            filtered_rules = [
                rule
                for rule, count in rules_count_dict.items()
                if count >= num_rules_threshold
            ]
        else: 
            rules_in_same_quantile = get_quantile_rules(rules, self.quantiles)
            # using set to ensure we only get unqiue rules
            filtered_rules = [
                list(set(rules))
                for rules in rules_in_same_quantile.values()
                if len(rules) >= num_rules_threshold
            ]
            filtered_rules = [item for sublist in filtered_rules for item in sublist]
        
        return filtered_rules

    def filter_rules(self, all_rules: list[Rule]) -> list[Rule]:
        num_trees = len(self.rf_model.estimators_)
        threshold_num = self.threshold * num_trees
        filtered_rules = self.get_high_frequency_rules(all_rules, threshold_num, self.filter_type)
        return filtered_rules

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert (
            X_test.shape[1] == self.X_train.shape[1]
        ), "train and test should have same number of features"
        if len(self.rules) > 0:
            X_test_rules = self.get_feature_matrix(self.rules, X_test)
            X_test_rules_scaled = self.scale_data(X_test_rules, train=False)
            y_pred = self.linear_model.predict(X_test_rules_scaled)
            return y_pred
        else:
            return []
    
    def get_feature_matrix(self, rules: list[Rule], X: np.ndarray) -> np.ndarray:
        return get_rule_feature_matrix(rules, X, binary = False)


class SirusClassification(Sirus):
    def __init__(
        self,
        threshold: int = 0.072,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        num_trees: int = 100,
        partial_sampling: int = 0.7,
        quantiles: list = None,
        max_split_candidates: int = None,
        random_state: int = 10,
        remove_ld: bool = True,
        filter_type: FilterType = FilterType(1),
    ) -> None:
        super().__init__(
            threshold,
            max_depth,
            min_samples_leaf,
            max_leaf_nodes,
            num_trees,
            partial_sampling,
            quantiles,
            max_split_candidates,
            random_state,
            remove_ld,
            filter_type,
        )

        self.data_type = DataType.CLASSIFICATION

    def fit_rf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        colnms: list = None,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        num_trees: int = 100,
        partial_sampling: int = 0.75,
        quantiles: list = None,
        max_split_candidates: list = None,
    ):
        if quantiles is None or self.filter_type.name == FilterType(2).name:
            self.rf_model = RandomForestClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_estimators=num_trees,
                max_features=max_split_candidates,
                max_samples=partial_sampling,
                random_state=self.seed,
            )
        else:
            self.rf_model = RandomForest(
                data_type=self.data_type,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                num_trees=num_trees,
                partial_sampling=partial_sampling,
                max_split_candidates=max_split_candidates,
                quantiles=quantiles,
                random_state=self.seed,
            )

        self.rf_model.fit(X=X_train, y=y_train)

    def extract_rules(self, tree_list: list) -> list[Rule]:
        stablerules = RuleEnsembleClassification(tree_list, self.X_train, self.y_train)
        return stablerules.rules


class SirusRegression(Sirus):
    def __init__(
        self,
        threshold: int = 0.072,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        max_leaf_nodes: int = -1,
        num_trees: int = 100,
        partial_sampling: int = 0.7,
        quantiles: list = None,
        max_split_candidates: int = None,
        random_state: int = 10,
        remove_ld: bool = True,
        filter_type: FilterType = FilterType(1),
    ) -> None:
        super().__init__(
            threshold,
            max_depth,
            min_samples_leaf,
            max_leaf_nodes,
            num_trees,
            partial_sampling,
            quantiles,
            max_split_candidates,
            random_state,
            remove_ld,
            filter_type,
        )

        self.data_type = DataType.REGRESSION

    def fit_rf(
        self,
        X_train,
        y_train,
        colnms=None,
        max_depth=2,
        min_samples_leaf=5,
        max_leaf_nodes=-1,
        num_trees=100,
        partial_sampling=0.75,
        quantiles=None,
        max_split_candidates=None,
    ):
        # Option 1) Dont provide quantiles
        # Option 2) Provide quantiles but perform RF without quantiles and only use when filtering the rules (if rules are in the same quantile they are classified as the same)
        if quantiles is None or self.filter_type.name == FilterType(2).name:
            self.rf_model = RandomForestRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_estimators=num_trees,
                max_features=max_split_candidates,
                max_samples=partial_sampling,
                random_state=self.seed,
            )

        else:
            self.rf_model = RandomForest(
                data_type=self.data_type,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                num_trees=num_trees,
                partial_sampling=partial_sampling,
                max_split_candidates=max_split_candidates,
                quantiles=quantiles,
                random_state=self.seed,
            )

        self.rf_model.fit(X=X_train, y=y_train)

    def extract_rules(self, tree_list: list) -> list[Rule]:
        stablerules = RuleEnsembleRegression(tree_list, self.X_train, self.y_train)
        return stablerules.rules

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        accuracy_score,
    )
    from data import get_BW_data, get_boston_housing
    from quantiles import cutpoints
    from forest import RandomForest

    NUM_TREES = 100
    MAX_DEPTH = 2
    P0 = 0.080
    MIN_DATA_LEAF = 5
    PARTIAL_SAMPLING = 0.7
    NUM_QUANTILES = 10
    SEED = 4
    M_TRY = 1  # ratio of used features

    def train_and_predict(X_train, X_test, y_train, model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

    def print_metrics(y_test, y_pred, model_type):
        if model_type == DataType.REGRESSION:
            print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
            print(
                "Unexplained variance (smaller better):",
                1 - explained_variance_score(y_test, y_pred),
            )
            print("\n\n")

        elif model_type == DataType.CLASSIFICATION:
            print("Accuracy Score:", accuracy_score(y_test, y_pred))
            print(
                "Unexplained variance (smaller better):",
                1 - explained_variance_score(y_test, y_pred),
            )
            print("\n\n")

    print("--------- REGRESSION -------------")
    X, y = get_boston_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    SPLITS = cutpoints(X_train, NUM_QUANTILES)
    NUM_FEATURES = int(M_TRY * X_train.shape[1])

    for splits, filter_type, description in [
        (SPLITS, FilterType(1), "Perform RF With Splits & FilterType 1"),
        (None, FilterType(1), "Perform RF Without Splits & FilterType 1"),
        (SPLITS, FilterType(2), "Perform RF Without Splits & FilterType 2"),
    ]:
        sr_model = SirusRegression(
            threshold=P0,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_DATA_LEAF,
            num_trees=NUM_TREES,
            partial_sampling=PARTIAL_SAMPLING,
            quantiles=splits,
            max_split_candidates=NUM_FEATURES,
            random_state=SEED,
            remove_ld=True,
            filter_type=filter_type,
        )
        print(f"\n\n{description}:")
        y_pred, trained_model = train_and_predict(X_train, X_test, y_train, sr_model)
        print(f"Number rules: {len(trained_model.rules)}")
        print_metrics(y_test, y_pred, DataType.REGRESSION)

    print("\n\n--------- CLASSIFICATION -------------")
    X, y = get_BW_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    SPLITS = cutpoints(X_train, NUM_QUANTILES)
    NUM_FEATURES = int(M_TRY * X_train.shape[1])

    for splits, filter_type, description in [
        (SPLITS, FilterType(1), "Perform RF With Splits & FilterType 1"),
        (None, FilterType(1), "Perform RF Without Splits & FilterType 1"),
        (SPLITS, FilterType(2), "Perform RF Without Splits & FilterType 2"),
    ]:
        sr_model = SirusClassification(
            threshold=P0,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_DATA_LEAF,
            num_trees=NUM_TREES,
            partial_sampling=PARTIAL_SAMPLING,
            quantiles=splits,
            max_split_candidates=NUM_FEATURES,
            random_state=SEED,
            remove_ld=True,
            filter_type=filter_type,
        )
        print(f"\n\n{description}:")
        y_pred, trained_model = train_and_predict(X_train, X_test, y_train, sr_model)
        print(f"NUM RULES AFTER REG: {len(trained_model.rules)}")
        print_metrics(y_test, y_pred, DataType.CLASSIFICATION)
