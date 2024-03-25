import sys
import unittest
import numpy as np

from sklearn.ensemble import RandomForestRegressor

sys.path.append("..")
from src.quantiles import cutpoints
from src.linear import fit_ridge, fit_L0
from src.rules import get_rules_count
from src.data import get_boston_housing, get_BW_data, DataType
from src.tree import DecisionTreeRegression
from src.ruleEsemble import RuleEnsembleRegression
from src.sirus.dependent import filter_linearly_dependent

class Test_LinearModels(unittest.TestCase):
    def setUp(self):
        self.data = zip([get_boston_housing, get_BW_data], [DataType.REGRESSION, DataType.CLASSIFICATION])
        self.max_rules = 10

    def test_positive_coeff_ridge(self):
        for data_func, data_type in self.data:
            X, y = data_func()
            model = fit_ridge(X, y, data_type)  # data_type instead of data_types
            are_positive = np.all(model.coef_ >= 0)
            self.assertTrue(are_positive, "Not all coefficients are positive")
            # Add more test cases as needed
    
    def test_max_num_rules_l0(self):
        for data_func, data_type in self.data:
            X, y = data_func()
            result_dict = fit_L0(X_train = X, 
                           y_train = y, 
                           data_type = data_type, 
                           max_rules=self.max_rules)  # data_type instead of data_types
            is_max_rules_satisfied = sum(result_dict["coeffs"] > 0) <= self.max_rules
            self.assertTrue(is_max_rules_satisfied, "More positve coeffs can allowed by max rules")


class Test_Dependent(unittest.TestCase):
    def setUp(self):
        X, y = get_boston_housing()
        self.X = X
        self.y = y
        self.splits = cutpoints(self.X, 3)
        self.max_depth = 1
        self.tree = DecisionTreeRegression(max_depth=self.max_depth, min_samples_leaf=5, max_split_candidates=3, random_state=1, quantiles=self.splits).fit(self.X, self.y)

    def test_filter_dependent(self):
        rules_class = RuleEnsembleRegression([self.tree, self.tree], self.X, self.y)
        rules_pre_filter = rules_class.rules
        rules_post_filter = filter_linearly_dependent(rules_pre_filter)
        self.assertEqual(len(rules_pre_filter)/ 2, len(rules_post_filter), msg="post filter should return 1 rule when two rules are identical")

class Test_RuleEsemble(unittest.TestCase):
    def setUp(self):
        X, y = get_boston_housing()
        self.X = X
        self.y = y
        self.max_depth = 2
        self.num_trees = 2
        self.random_state = 1
        self.min_leaf_data = 5
        self.max_split_candidates = self.X.shape[1]
        self.model_sklearn = RandomForestRegressor(
            n_estimators=self.num_trees,
            max_depth=self.max_depth,
            random_state=self.random_state,
            min_samples_leaf=self.min_leaf_data,
        ).fit(self.X, self.y)
        self.sklearn_tree_list = [x for x in self.model_sklearn.estimators_]
        self.rule_ensemble = RuleEnsembleRegression(
            self.sklearn_tree_list, self.X, self.y
        )
        self.splits = cutpoints(self.X, 3)

    def test_is_sklearn(self):
        self.assertTrue(
            self.rule_ensemble.isSklearn,
            "RuleEsemble attribute isSklearn should be True",
        )

    def test_num_rules_sklearn(self):
        self.assertEqual(
            len(self.rule_ensemble.rules),
            5 * self.num_trees,
            "With tree depth 2 number of rules per tree is 5",
        )
    
    def test_filter_function(self):
        tree = DecisionTreeRegression(max_depth=2, min_samples_leaf=5, max_split_candidates=3, random_state=1, quantiles=self.splits).fit(self.X, self.y)
        rule_ensemble = RuleEnsembleRegression([tree, tree], self.X, self.y)
        num_pre_filter = len(rule_ensemble.rules)
        num_post_filter = len(rule_ensemble.filter_rules())
        self.assertEqual(num_pre_filter / 2, num_post_filter, msg="filter should remove half the rules, since rules are doubled")
    
    def test_rule_counter(self):
        tree = DecisionTreeRegression(max_depth=2, min_samples_leaf=5, max_split_candidates=3, random_state=1, quantiles=self.splits).fit(self.X, self.y)
        rule_ensemble = RuleEnsembleRegression([tree, tree], self.X, self.y)
        rules = rule_ensemble.rules
        rule_count_dict = get_rules_count(rules)
        num_keys = len(rule_count_dict.keys())
        count_sum = sum(list(rule_count_dict.values()))
        self.assertEqual(num_keys, len(rules)/ 2, "num keys should be half since we have duplicate rules")
        self.assertEqual(count_sum, len(rules), "count should be the number of total rules")


if __name__ == "__main__":
    unittest.main()
