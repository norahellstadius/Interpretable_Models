import sys
import unittest
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression

sys.path.append("..")
from src.quantiles import cutpoints
from src.forest import RandomForest
from src.linear import fit_ridge, fit_L0, RegType
from src.rules import get_rules_count, FilterType
from src.data import get_boston_housing, get_BW_data, DataType
from src.tree import DecisionTreeClassification, DecisionTreeRegression
from src.ruleEsemble import RuleEnsembleRegression, RuleEnsembleClassification
from src.sirus.dependent import filter_linearly_dependent
from src.sirus.sirus import SirusRegression, SirusClassification
from src.l0_rulefit.l0rulefit import L0_Rulefit
from src.rule_generator.simply_rules import SimplyRules
from tests.utils import valid_split_point_with_quantile, valid_split_value_in_rules

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

class Test_DecisionTree(unittest.TestCase):
    def setUp(self):
        # Initialize some sample data for testing
        np.random.seed(42)
        self.X = np.random.rand(50, 5)
        self.y = np.random.randint(0, 2, self.X.shape[0])
        self.max_depth = 2
        self.random_state = 1
        self.min_leaf_data = 5
        self.max_split_candidates = self.X.shape[1]
        self.delta = 0.09

    def test_predict_regression(self):
        dtr_own = DecisionTreeRegression(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            max_split_candidates=self.max_split_candidates,
            random_state=self.random_state,
        ).fit(self.X, self.y)
        dtr_sklearn = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            max_features=self.max_split_candidates,
            random_state=self.random_state,
        ).fit(self.X, self.y)

        score_own = explained_variance_score(self.y, dtr_own.predict(self.X))
        score_sklearn = explained_variance_score(self.y, dtr_sklearn.predict(self.X))

        self.assertAlmostEqual(
            score_own,
            score_sklearn,
            delta=self.delta,
            msg="Own implementation of DT Regression does not give the same score as sklearn",
        )

    def test_predict_classification(self):
        dtc_own = DecisionTreeClassification(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            max_split_candidates=self.max_split_candidates,
            random_state=self.random_state,
        ).fit(self.X, self.y)

        dtc_sklearn = DecisionTreeClassifier(
            criterion="gini",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            max_features=self.max_split_candidates,
            random_state=self.random_state,
        ).fit(self.X, self.y)

        score_own = explained_variance_score(self.y, dtc_own.predict(self.X))
        score_sklearn = explained_variance_score(self.y, dtc_sklearn.predict(self.X))

        self.assertAlmostEqual(
            score_own,
            score_sklearn,
            delta=self.delta,
            msg="Own implementation of DT Classification does not give the same score as sklearn",
        )
    
    def test_valid_y(self):
        invalid_y =  np.random.randn(self.X.shape[0])
        with self.assertRaises(ValueError):
            DecisionTreeClassification(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_leaf_data,
                max_split_candidates=self.max_split_candidates,
                random_state=self.random_state,
            ).fit(self.X, invalid_y)

class Test_RandomForest(unittest.TestCase):
    def setUp(self):
        # Initialize some sample data for testing
        X, y = get_BW_data()
        self.X_bw = X
        self.y_bw = y
        self.num_trees = 2
        self.max_depth = 2
        self.random_state = 1
        self.min_leaf_data = 5
        self.max_split_candidates = self.X_bw.shape[1]
        self.partial_sampling = 0.75
        self.delta = 0.09
        self.splits = cutpoints(self.X_bw, 10)

    def test_rf_classification(self):
        rf_own = RandomForest(
            data_type=DataType.CLASSIFICATION,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            num_trees=self.num_trees,
            random_state=self.random_state,
            partial_sampling=self.partial_sampling,
        ).fit(self.X_bw, self.y_bw)

        rf_sklearn = RandomForestClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            n_estimators=self.num_trees,
            random_state=self.random_state,
            max_features=self.max_split_candidates,
            max_samples=self.partial_sampling,
        ).fit(self.X_bw, self.y_bw)

        score_own = explained_variance_score(self.y_bw, rf_own.predict(self.X_bw))
        score_sklearn = explained_variance_score(self.y_bw, rf_sklearn.predict(self.X_bw))

        self.assertAlmostEqual(
            score_own,
            score_sklearn,
            delta=self.delta,
            msg="Own implementation of rf Classification does not give the same score as sklearn",
        )

    def test_fit_dimension_mismatch(self):
        y_wrong_dim = self.y_bw[:-3]
        with self.assertRaises(AssertionError):
            RandomForest(
                data_type=DataType.CLASSIFICATION,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_leaf_data,
                num_trees=self.num_trees,
                random_state=self.random_state,
                partial_sampling=self.partial_sampling,
            ).fit(self.X_bw, y_wrong_dim)

    def test_predict_dimension_mismatch(self):
        dt_model = RandomForest(
            data_type=DataType.CLASSIFICATION,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            num_trees=self.num_trees,
            random_state=self.random_state,
            partial_sampling=self.partial_sampling,
        ).fit(self.X_bw, self.y_bw)

        X_test_wrong_dim = np.random.rand(self.X_bw.shape[0] // 2, self.X_bw.shape[-1] - 1)
        # Check if an error is raised when dimensions do not match
        with self.assertRaises(AssertionError):
            dt_model.predict(X_test_wrong_dim)

    
    def test_split_value_in_quantile(self):
        rf_own = RandomForest(
            data_type=DataType.CLASSIFICATION,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_data,
            num_trees=self.num_trees,
            random_state=self.random_state,
            partial_sampling=self.partial_sampling,
            quantiles=self.splits
        ).fit(self.X_bw, self.y_bw)

        bool_list = [valid_split_point_with_quantile(tree, self.splits) for tree in rf_own.estimators_]
        self.assertTrue(all(bool_list))


class Test_SIRUS(unittest.TestCase):
    def setUp(self):
        X, y = get_boston_housing()
        self.X = X
        self.y = y
        self.threshold = 0.03
        self.num_trees = 100
        self.splits = cutpoints(self.X, 5)
        self.sirus_filter1 = SirusRegression(threshold=self.threshold,
                                        max_depth=2,
                                        num_trees=self.num_trees,
                                        max_split_candidates=4, 
                                        quantiles=self.splits, 
                                        filter_type=FilterType(1)).fit(X, y)
        
    def test_rule_filter(self):
        trees = self.sirus_filter1.rf_model.estimators_
        bool_list = [valid_split_point_with_quantile(tree, self.splits) for tree in trees]
        self.assertTrue(all(bool_list), msg="all split values should be in the quantile splits")
    
    def test_feature_matrix_dim(self):
        rules = self.sirus_filter1.rules 
        feature_matrix = self.sirus_filter1.get_feature_matrix(rules, self.X)
        fm_dim = feature_matrix.shape
        self.assertEqual(fm_dim, (self.X.shape[0], len(rules)), msg="feature matrxi should be of dim (number samples x number rules)")

    def test_valid_datatype(self):
        self.assertEqual(self.sirus_filter1.data_type, DataType.REGRESSION, "Datatype should be regression")
        sirus_c = SirusClassification(threshold=self.threshold,
                                        max_depth=2,
                                        num_trees=self.num_trees,
                                        max_split_candidates=4, 
                                        quantiles=self.splits, 
                                        filter_type=FilterType(1))
        self.assertEqual(sirus_c.data_type, DataType.CLASSIFICATION, "Datatype should be classification")

    def test_high_frequency_filter(self):
        #filtertype 1 (same split_value by quantizing the data)
        tree_1 = DecisionTreeRegression(max_depth=2, min_samples_leaf=5, max_split_candidates=3, random_state=1, quantiles=self.splits).fit(self.X, self.y)
        tree_2 = DecisionTreeRegression(max_depth=2, min_samples_leaf=5, max_split_candidates=1, random_state=10, quantiles=self.splits).fit(self.X, self.y)
        rules_pre_filter = RuleEnsembleRegression([tree_1, tree_1, tree_1, tree_2], self.X, self.y).rules
        rule_post_filter = self.sirus_filter1.get_high_frequency_rules(rules_pre_filter, self.threshold * self.num_trees, FilterType(1))
        self.assertEqual(len(rule_post_filter), 5, msg = "only 5 rules should pass the threshold of 3 rules. The tree_1 generated 5 rules for each of the 3 instances")

        #filtertype 2 (same quantile by not quantizing the data)
        tree_1 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, max_features=3, random_state=1).fit(self.X, self.y)
        tree_2 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, max_features=1, random_state=10).fit(self.X, self.y)
        rules_pre_filter = RuleEnsembleRegression([tree_1, tree_1, tree_1, tree_2], self.X, self.y).rules
        rule_post_filter = self.sirus_filter1.get_high_frequency_rules(rules_pre_filter, self.threshold * self.num_trees, FilterType(2))
        self.assertEqual(len(rule_post_filter), 5, msg = "only 5 rules should pass the threshold of 3 rules. The tree_1 generated 5 rules for each of the 3 instances")

class Test_L0Rulefit(unittest.TestCase):
    def setUp(self):
        X, y = get_boston_housing()
        self.X = X
        self.y = y
        self.max_rules = 20
        self.model_r = L0_Rulefit(data_type=DataType.REGRESSION,
                                  max_depth=2,
                                  partial_sampling=0.5,
                                  num_trees=10, 
                                  max_rules=self.max_rules, 
                                  max_split_candidates=self.X.shape[1], 
                                  random_state=1).fit(self.X, self.y)
    
    def test_feature_matrix_dim(self):
        fm = self.model_r.get_feature_matrix(self.model_r.pre_regularized_rules, self.X)
        num_rules = len(self.model_r.pre_regularized_rules)
        self.assertEqual(fm.shape, (self.X.shape[0], num_rules), msg = "shape of feature matrix should be (num samples x num rules")
    
    def test_active_rules_func(self):
        num_active_rules = len(self.model_r.get_active_rules(self.model_r.pre_regularized_rules, self.model_r.coeffs))
        num_positive_coeffs = sum(self.model_r.coeffs > 0)
        self.assertEqual(num_active_rules, num_positive_coeffs, "num active rules should be equal to num coeffs > 0")
    
    def test_num_pre_reg_rules(self):
        num_pre_reg_rules = len(self.model_r.pre_regularized_rules)
        self.assertLessEqual(num_pre_reg_rules, 50, msg = "Correct answer: >= 50 (since 5 rules per tree is generated). \n Note: some rules may be eliminated due to duplicates")

    def test_num_post_reg_rules(self):
        num_post_reg_rules = len(self.model_r.estimators_)
        self.assertLessEqual(num_post_reg_rules, self.max_rules, "number rules should be less than max rules")

    def test_non_regularize(self):
        non_reg_model = L0_Rulefit(data_type=DataType.REGRESSION,
                                  max_depth=2,
                                  partial_sampling=0.5,
                                  num_trees=10, 
                                  max_rules=10, 
                                  max_split_candidates=self.X.shape[1], 
                                  regularization=RegType.NONE,
                                  random_state=1).fit(self.X, self.y)
        self.assertEqual(len(non_reg_model.pre_regularized_rules), len(non_reg_model.estimators_), msg = "No rules should be removed")
        self.assertIsInstance(non_reg_model.linear_model, LinearRegression)

class Test_SimplyRules(unittest.TestCase):
    def setUp(self):
        X, y = get_boston_housing()
        self.X = X
        self.y = y
        self.num_trees = 10
        self.model_r = SimplyRules(data_type=DataType.REGRESSION,
                                  max_depth=2,
                                  partial_sampling=0.5,
                                  num_trees=self.num_trees, 
                                  max_split_candidates=self.X.shape[1], 
                                  random_state=1).fit(self.X, self.y)   

    def test_feature_matrix_dim(self):
        feature_matrix = self.model_r.get_feature_matrix(self.model_r.estimators_, self.X)
        self.assertEqual(feature_matrix.shape, (self.X.shape[0], len(self.model_r.estimators_)), msg = "Correct shape: number samples x number rules")
    
    def test_num_rules_generated(self):
        num_rules = len(self.model_r.estimators_)
        self.assertLessEqual(num_rules, 5 * self.num_trees, "if contains duplicates may be less than 50. With depth 2 you generate 5 rules per tree")

    def test_split_in_quantile(self):
        splits = cutpoints(self.X, 4)
        model_with_quantile = SimplyRules(data_type=DataType.REGRESSION,
                                  max_depth=2,
                                  partial_sampling=0.5,
                                  num_trees=self.num_trees, 
                                  max_split_candidates=self.X.shape[1],
                                  quantiles= splits,
                                  random_state=1).fit(self.X, self.y)   
        valid_rules = valid_split_value_in_rules(model_with_quantile.estimators_, splits)
        self.assertTrue(valid_rules)

if __name__ == "__main__":
    unittest.main()
