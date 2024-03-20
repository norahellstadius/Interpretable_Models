import sys
import numpy as np
from typing import Union
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.append("../")
from src.rules import Rule, SubClause, Clause, satisfies_rule
from src.tree import SplitPoint, Node


class RuleEnsemble:
    def __init__(
        self, tree_list: list, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        self.tree_list = tree_list
        self.X_train = X_train
        self.y_train = y_train

        # store rules from trees
        self.rules = []

        # Check if the trees are from sklearn or own implementation
        self.isSklearn = False
        self._is_sklearn_instance()
        # Extract rules
        self._extract_rules()

    def _is_sklearn_instance(self):
        if hasattr(self.tree_list[0], "root_node"):
            self.isSklearn = False
        else:
            self.isSklearn = True

    def filter_rules(self) -> list:
        """Turn single clauses to left and return only unique rules

        Example: clause x >= 0.3 is converted to x < 0.3

        Note: this is used in Rulefit and L0 rulefit but not SIRUS (since then you want to count frequency and not only get the unqiue rules)
        """
        unique_rules = set()

        for rule in self.rules:
            if len(rule.subclauses()) == 1:
                adjusted_rule = rule.left_rule()
            else:
                adjusted_rule = rule
            unique_rules.add(adjusted_rule)

        # Convert back to a list
        self.rules = list(unique_rules)
        return self.rules

    def _extract_rules(self):
        for tree in self.tree_list:
            # Extract rules based on whether it's a custom or sklearn tree
            if not self.isSklearn:
                rules_no_values = self.extract_rules_from_tree_own(tree.root_node)
            else:
                rules_no_values = self.extract_rules_from_tree_sklearn(tree.tree_)

            # Add rule if and then values and append to self.rules
            rules = self.get_rule_values(rules_no_values)
            self.rules.extend(rules)

    def extract_rules_from_tree_own(
        self, node: Node, subclauses: list = None, rules: set = None, root: Node = None
    ) -> list:
        """
        Return all rules for a tree.
        Note rule exists for each level of the tree (a rule does not have to end at a leaf node)
        """
        if rules is None:
            rules = set()
        if root is None:
            root = node
        if subclauses is None:
            subclauses = []

        if node.is_leaf:
            rule = Rule(Clause(subclauses))
            rule = (
                rule.left_rule() if len(rule.subclauses()) == 1 else rule
            )  # convert single rule to a left rule
            rules.add(rule)
        else:
            if subclauses:
                rule = Rule(Clause(subclauses))
                rule = (
                    rule.left_rule() if len(rule.subclauses()) == 1 else rule
                )  # convert single rule to a left rule
                rules.add(rule)

            subclause_L = SubClause(node.splitpoint, np.less)
            new_subclauses_L = [subclause_L] + subclauses
            self.extract_rules_from_tree_own(node.left, new_subclauses_L, rules, root)

            subclause_R = SubClause(node.splitpoint, np.greater_equal)
            new_subclauses_R = [subclause_R] + subclauses
            self.extract_rules_from_tree_own(node.right, new_subclauses_R, rules, root)

        return list(rules)

    def extract_rules_from_tree_sklearn(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ) -> list:
        rules = set()

        def traverse_nodes(
            node_id=0, operator=None, threshold=None, feature=None, conditions=[]
        ):
            if node_id != 0:
                split_info = SplitPoint(
                    feature=feature, value=threshold, feature_name=None
                )
                rule_condition = SubClause(sp=split_info, direction=operator)
                new_conditions = [rule_condition] + conditions
            else:
                new_conditions = []

            ## if not terminal node
            if tree.children_left[node_id] != tree.children_right[node_id]:
                if new_conditions:
                    rule = Rule(Clause(new_conditions))
                    rule = rule.left_rule() if len(rule.subclauses()) == 1 else rule
                    rules.add(rule)

                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]

                left_node_id = tree.children_left[node_id]
                traverse_nodes(
                    left_node_id, np.less, threshold, feature, new_conditions
                )

                right_node_id = tree.children_right[node_id]
                traverse_nodes(
                    right_node_id, np.greater_equal, threshold, feature, new_conditions
                )
            else:  # a leaf node
                if len(new_conditions) > 0:
                    new_rule = Rule(Clause(new_conditions))
                    new_rule = (
                        new_rule.left_rule()
                        if len(new_rule.subclauses()) == 1
                        else new_rule
                    )
                    rules.add(new_rule)
                else:
                    pass  # tree only has a root node!
                return None

        traverse_nodes()
        return list(rules)

    def get_rule_values(self, rules: list[Rule]) -> list:
        rules_with_values = []
        for r in rules:
            all_y_then = []
            all_y_else = []

            for i in range(self.X_train.shape[0]):
                row_data = self.X_train[i, :]
                if satisfies_rule(row_data, r):
                    all_y_then.append(self.y_train[i])
                else:
                    all_y_else.append(self.y_train[i])

            if all_y_else and all_y_then:
                r_with_values = self.get_value(r, all_y_then, all_y_else)
                rules_with_values.append(r_with_values)

        return rules_with_values


class RuleEnsembleRegression(RuleEnsemble):
    def __init__(
        self, tree_list: list, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        super().__init__(tree_list, X_train, y_train)

    def get_value(self, rule: Rule, all_y_then: list, all_y_else: list) -> Rule:
        y_then = np.mean(all_y_then)
        y_else = np.mean(all_y_else)
        return Rule(rule.clause, y_then, y_else)


class RuleEnsembleClassification(RuleEnsemble):
    def __init__(
        self, tree_list: list, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        super().__init__(tree_list, X_train, y_train)

    def get_value(self, rule: Rule, all_y_then: list, all_y_else: list) -> Rule:
        counts_then = Counter(all_y_then)
        y_then = max(counts_then, key=counts_then.get)  # predict the majority class

        counts_else = Counter(all_y_else)
        y_else = max(counts_else, key=counts_else.get)
        return Rule(rule.clause, y_then, y_else)
