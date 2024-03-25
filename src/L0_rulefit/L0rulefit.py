import numpy as np
import sys
import copy

import l0learn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.append("../..")
from src.ruleEsemble import RuleEnsembleClassification, RuleEnsembleRegression
from src.rules import get_rule_feature_matrix, Rule
from src.data import DataType
from src.linear import fit_L0, fit_lm, RegType

class L0_Rulefit:
    def __init__(
        self,
        data_type: DataType = DataType(1),
        max_depth: int = 2,
        partial_sampling: int = 0.70,
        min_samples_leaf: int = 5,
        num_trees: int = 100,
        max_rules: int = 20,
        max_split_candidates: int = None,
        regularization: RegType = RegType.L0,
        random_state: int = 1,
    ):
        """
        L0_Rulefit class for trains a random forest to generate rules 
        and the filters/selects rules by using L0 regularisation

        Parameters:
        -----------
        data_type : DataType
            Type of the data, Regression or Classification.
        max_depth : int, optional
            Maximum depth of the trees (default is 2).
        partial_sampling : float, optional
            Percentage of samples for bootstrapping (default is 0.70).
        min_samples_leaf : int, optional
            Minimum number of samples required to be at a leaf node (default is 5).
        num_trees : int, optional
            Number of trees to fit for random forest(default is 100).
        max_rules : int, optional
            Maximum number of rules to generate (default is 20). Used as a paramter in L0 regularisation. 
            This is ignored is regularize is set to False
        max_split_candidates : int or None, optional
            Maximum number of feature splits to consider for each node (default is None).
        regularize : bool, optional
            Whether to apply L0 regularization (default is True). 
            If set to False then normal linear regression is applied on all the rules
        random_state : int, optional
            Random seed for reproducibility (default is 1).
        """
        self.max_depth = max_depth
        self.data_type = data_type
        self.min_samples_leaf = min_samples_leaf
        self.num_trees = num_trees
        self.max_split_candidates = max_split_candidates
        self.max_rules = max_rules
        self.partial_sampling = partial_sampling
        self.regularization = regularization
        self.random_state = random_state

        if data_type.name not in [DataType.REGRESSION.name, DataType.CLASSIFICATION.name]:
            raise ValueError(
                "Invalid value for data_type. Expected 'Classification' or 'Regression', but got '{}'.".format(
                    data_type
                )
            )
        if regularization.name not in [RegType(1).name, RegType(2).name, RegType(3).name]:
            raise ValueError(
                "Invalid value for regularization. Expected 'RegType.NONE', 'RegType.L0' or 'RegType.RIDGE', but got '{}'.".format(
                    regularization
                )
            )

        self.ml_model = None  # random forest model
        self.pre_regularized_rules = []
        self.estimators_ = []  # rules after regularisation

        self.linear_model = None  # L0 model
        self.optimal_gamma = None
        self.optimal_lambda = None
        self.coeffs = []

    def set_attributes(self, **kwargs):
        """
        Method to set multiple attributes of the Sirus class instance.
        Usage: set_attributes(attr1=value1, attr2=value2, ...)
        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def scale_data(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            self.scalar = StandardScaler().fit(X)
        X_scaled = self.scalar.transform(X)
        return X_scaled
    
    def get_max_features(self):
        return round(np.sqrt(self.X_train.shape[1]))

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == len(y), "X and y train must have same number of samples"

        self.X_train = X
        self.y_train = y

        if self.max_split_candidates is None:
            self.max_split_candidates = self.get_max_features() 

        if self.data_type.name == DataType.REGRESSION.name:
            self.ml_model = RandomForestRegressor(
                n_estimators=self.num_trees,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_split_candidates,
                max_samples=self.partial_sampling,
                random_state=self.random_state,
            )
        elif self.data_type.name == DataType.CLASSIFICATION.name:
            self.ml_model = RandomForestClassifier(
                n_estimators=self.num_trees,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_split_candidates,
                max_samples=self.partial_sampling,  # TODO: CHECK IF THIS IS CORRECT
                random_state=self.random_state,
            )

        # fit the random forest
        self.ml_model.fit(self.X_train, self.y_train)

        if self.data_type.name == DataType.REGRESSION.name:
            self.rule_ensemble = RuleEnsembleRegression(
                 self.ml_model.estimators_, self.X_train, self.y_train
            )
        else:
            self.rule_ensemble = RuleEnsembleClassification(
                 self.ml_model.estimators_, self.X_train, self.y_train
            )

        # single rules are turned left and duplicates are removed
        self.pre_regularized_rules = self.rule_ensemble.filter_rules()

        X_rules = self.get_feature_matrix(self.pre_regularized_rules, self.X_train)
        X_rules_scaled = self.scale_data(X_rules)
        #apply L0 regularisation if regularize is True
        self.fit_linear_model(X_rules_scaled, self.y_train, self.regularization)
        # if regularize is True: get rules which have non zero coefficents
        self.estimators_ = self.get_active_rules(self.pre_regularized_rules, self.coeffs) if self.regularization.name == RegType.L0.name else copy.deepcopy(self.pre_regularized_rules)
        return self
    
    def get_feature_matrix(self, rules: list[Rule], X: np.ndarray) -> np.ndarray:
        return get_rule_feature_matrix(rules, X)

    def get_active_rules(self, rules: list[Rule], coeffs: list) -> list[Rule]:
        assert len(rules) == len(coeffs), "number of rules and coeffs (not including intercept) must be the same"
        active_rules = []
        for r, c in zip(rules, coeffs):
            if c > 0:
                active_rules.append(r)
        return active_rules

    def fit_linear_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        regularize: RegType, 
        penalty: str = "L0L2", # check "L0"
        num_folds: int = 5,
        num_gamma: int = 5,
        gamma_min: float = 0.0001,
        gamma_max: float = 0.1,
        algorithm: str = "CDPSI",
    ):  
        if regularize.name == RegType.L0.name:
            result_dict = fit_L0(
                X_train = X_train,
                y_train = y_train,
                data_type = self.data_type,
                max_rules = self.max_rules,
                penalty = penalty,
                num_folds = num_folds,
                num_gamma = num_gamma,
                gamma_min = gamma_min,
                gamma_max = gamma_max,
                algorithm = algorithm,
                random_state = self.random_state, 
                )

            self.linear_model = result_dict["model"]
            self.optimal_gamma = result_dict["optimal_gamma"]
            self.optimal_lambda = result_dict["optimal_lambda"]
            self.coeffs = result_dict["coeffs"]
        else: 
            self.linear_model = fit_lm(X_train, y_train, self.data_type)

    def predict(self, X_test: np.ndarray):
        assert (
            X_test.shape[1] == self.X_train.shape[1]
        ), "X_train and X_test must have same number feature"

        X_rules = get_rule_feature_matrix(self.pre_regularized_rules, X_test)
        X_rules_scaled = self.scale_data(X_rules, train=False)
        y_pred = self.linear_model.predict(
            X_rules_scaled, lambda_0=self.optimal_lambda, gamma=self.optimal_gamma
        ) if self.regularization == RegType.L0 else self.linear_model.predict(X_rules_scaled)
        y_pred = (
            (y_pred >= 0.5).astype(int)
            if self.data_type.name == DataType.CLASSIFICATION.name
            else y_pred
        )
        return y_pred.flatten()


if __name__ == "__main__":
    from data import get_BW_data, get_boston_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_absolute_error,
        explained_variance_score,
        accuracy_score,
    )
    from quantiles import cutpoints

    MAX_DEPTH = 2
    NUM_QUANTILES = 10
    MIN_LEAF_DATA = 5
    SEED = 2
    SAMPLE_FRAC = 0.70
    NUM_TREES = 100
    MTRY = 1 / 3
    MAX_NUM_RULES = 10

    def get_scores(data_type, y_true, y_pred):
        score2 = 1 - explained_variance_score(y_true, y_pred)
        if data_type.name == DataType.CLASSIFICATION.name:
            score1 = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {score1}")
        elif data_type.name == DataType.REGRESSION.name:
            score1 = mean_absolute_error(y_true, y_pred)
            print(f"Mean Absolute Error: {score1}")

        print(f"Unexplained Variance: {score2}")
        return score1, score2

    data_list = [[(DataType.CLASSIFICATION, get_BW_data), ("No Split", False)], [( DataType.REGRESSION, get_boston_housing), ("No Split", False)]]

    for (data_type, data), (split_type, quantiles) in data_list:
        print(f"\nDATA TYPE: {data_type} --- SPLIT/QUANTILE TYPE: {split_type} \n{'-'*30}")
        X, y = get_BW_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        MAX_SPLIT_CANDIDATES = int(X.shape[1])
        SPLITS = cutpoints(X_train, 10) if quantiles else None

        sparse_rulefit = L0_Rulefit(
            data_type=data_type,
            max_depth=MAX_DEPTH,
            partial_sampling=SAMPLE_FRAC,
            min_samples_leaf=MIN_LEAF_DATA,
            num_trees=NUM_TREES,
            max_rules=MAX_NUM_RULES,
            max_split_candidates=MAX_SPLIT_CANDIDATES,
            random_state=SEED,
        )

        sparse_rulefit.fit(X_train, y_train)
        y_pred_train = sparse_rulefit.predict(X_train)
        y_pred_test = sparse_rulefit.predict(X_test)

        print("\nTrain Set")
        get_scores(data_type, y_train, y_pred_train)
        print("\nTest Set")
        get_scores(data_type, y_test, y_pred_test)

        print(f"\nNumber of active rules: {len(sparse_rulefit.estimators_)}")
