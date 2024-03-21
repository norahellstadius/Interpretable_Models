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

class SimplyRules:
    def __init__(
        self,
        data_type: DataType = DataType(1),
        max_depth: int = 2,
        partial_sampling: int = 0.70,
        min_samples_leaf: int = 5,
        num_trees: int = 100,
        max_split_candidates: int = None,
        random_state: int = 1,
    ):
        """
        SimplyRules class for trains a random forest to generate rules 
        and fits a simple linear model on the rules

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
        max_split_candidates : int or None, optional
            Maximum number of feature splits to consider for each node (default is None).
            If set to False then normal linear regression is applied on all the rules
        random_state : int, optional
            Random seed for reproducibility (default is 1).
        """
        self.max_depth = max_depth
        self.data_type = data_type
        self.min_samples_leaf = min_samples_leaf
        self.num_trees = num_trees
        self.max_split_candidates = max_split_candidates
        self.partial_sampling = partial_sampling
        self.random_state = random_state

        if data_type not in [DataType.REGRESSION, DataType.CLASSIFICATION]:
            raise ValueError(
                "Invalid value for data_type. Expected 'Classification' or 'Regression', but got '{}'.".format(
                    data_type
                )
            )

        self.ml_model = None  # random forest model
        self.estimators_ = []  # rules after regularisation
        self.linear_model = None  # L0 model

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

        if self.data_type == DataType.REGRESSION:
            self.ml_model = RandomForestRegressor(
                n_estimators=self.num_trees,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_split_candidates,
                max_samples=self.partial_sampling,
                random_state=self.random_state,
            )
        elif self.data_type == DataType.CLASSIFICATION:
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

        if self.data_type == DataType.REGRESSION:
            self.rule_ensemble = RuleEnsembleRegression(
                 self.ml_model.estimators_, self.X_train, self.y_train
            )
        else:
            self.rule_ensemble = RuleEnsembleClassification(
                 self.ml_model.estimators_, self.X_train, self.y_train
            )

        # single rules are turned left and duplicates are removed
        self.estimators_ = self.rule_ensemble.filter_rules()

        X_rules = self.get_feature_matrix(self.estimators_, self.X_train)
        X_rules_scaled = self.scale_data(X_rules)
        self.fit_linear_model(X_rules_scaled, self.y_train) #fit a simple linear regression
        return self
    
    def get_feature_matrix(self, rules: list[Rule], X: np.ndarray) -> np.ndarray:
        return get_rule_feature_matrix(rules, X)

    def fit_linear_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):  
        self.linear_model = fit_lm(X_train, y_train, self.data_type)

    def predict(self, X_test: np.ndarray):
        assert (
            X_test.shape[1] == self.X_train.shape[1]
        ), "X_train and X_test must have same number feature"

        X_rules = get_rule_feature_matrix(self.estimators_, X_test)
        X_rules_scaled = self.scale_data(X_rules, train=False)
        y_pred = self.linear_model.predict(
            X_rules_scaled
        )
        y_pred = (
            (y_pred >= 0.5).astype(int)
            if self.data_type == DataType.CLASSIFICATION
            else y_pred
        )
        return y_pred.flatten()
