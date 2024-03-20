import sys
import numpy as np
import copy
from enum import Enum
from typing import Callable
from collections import defaultdict

sys.path.append("../../")
from src.tree import SplitPoint
from src.quantiles import get_quantile_index


# NOTE: np.less == Left  np.greater_equal == Right
class FilterType(Enum):
    """
    Enumeration defining types of filters for data analysis.
    """

    SAME_SPLIT_VALUE = 1
    """
    Filter type for exact split value match.
    This filter considers rules as the same if they have exactly the same split value.
    """

    SAME_QUANTILE = 2
    """
    Filter type for inside same quantile match.
    This filter considers rules as the same if they belong to the same quantile without quantizing the data.
    """


class SubClause:
    """
    A SubClause is equivalent to a split in a decision tree.
    Each rule contains a clause with one or more SubClause.
    For example, the rule `if X[i, 1] > 3 & X[i, 2] < 4, then ...` contains two subclauses.

    Attributes:
        sp (SplitPoint): instance of the SplitPoint class
        direction (Callable[[float, float], bool]): A callable representing the conditional operation.
            This operation takes two float values and returns a boolean result.
    """

    def __init__(self, sp: SplitPoint, direction: Callable[[float, float], bool]):
        if direction not in [np.greater_equal, np.less]:
            raise AssertionError("Invalid direction: {}".format(self.direction))

        self.feature_idx = sp.feature
        self.feature_name = sp.feature_name
        self.split_value = round(sp.split_value, 4)
        self.direction = direction

        self.quantile_idx = None  # which quantile the split_value belongs to

    def print_condition(self):
        print(f"Feature {self.feature_idx} {self.direction} {self.split_value}")

    def reverse(self):
        new_direction = np.greater_equal if self.direction == np.less else np.less
        sp = SplitPoint(self.feature_idx, self.split_value, self.feature_name)
        sc_reverse = SubClause(sp, new_direction)
        if self.quantile_idx is not None:
            sc_reverse.quantile_idx = self.quantile_idx
        return sc_reverse

    # needed for set
    def __hash__(self):
        # Use a tuple of relevant attributes for hashing
        if self.quantile_idx is None:
            return hash((self.feature_idx, self.direction, self.split_value))
        else:
            return hash(
                (self.feature_idx, self.direction, self.split_value, self.quantile_idx)
            )

    def __eq__(self, other):
        if isinstance(other, SubClause):
            if self.quantile_idx is None:
                return (
                    self.feature_idx == other.feature_idx
                    and self.direction == other.direction
                    and self.split_value == other.split_value
                )
            else:
                return (
                    self.feature_idx == other.feature_idx
                    and self.direction == other.direction
                    and self.split_value == other.split_value
                    and self.quantile_idx == other.quantile_idx
                )
        return False

    def string(self):
        sign = "<" if self.direction == np.less else ">="
        if self.quantile_idx is None:
            return f"X[i, {self.feature_idx}] {sign} {self.split_value}"
        else:
            return f"X[i, {self.feature_idx}] {sign} {self.split_value} (quantile_idx = {self.quantile_idx})"

    def __str__(self):
        return self.string()


class Clause:
    """
    A path denotes a conditional on one or more features.
    Each rule contains a path with one or more conditions.

    A Path is equivalent to a path in a decision tree.
    For example, the path `X[i, 1] > 3 & X[i, 2] < 4` can be interpreted as a path
    going through two nodes.

    Note that a path can also be a path to a node; not necessarily a leaf.
    """

    def __init__(self, conditions: list):
        self.subclauses = conditions

    def __hash__(self):
        # Use a tuple of relevant attributes for hashing
        return hash(tuple(hash(sc) for sc in self.subclauses))

    def __eq__(self, other):
        if isinstance(other, Clause):
            for cond1, cond2 in zip(self.subclauses, other.subclauses):
                if not (cond1 == cond2):
                    return False
            return True
        return False

    def string(self):
        if not self.subclauses:
            return ""

        subclause_strings = [sc.string() for sc in self.subclauses[::-1]]
        return " & ".join(subclause_strings)

    def __str__(self):
        return self.string()


def get_clause_count(clauses):
    counter = {}
    for c in clauses:
        if c in counter:
            counter[c] += 1
        else:
            counter[c] = 1
    return counter


class Rule:
    """
    A rule is a Path with a then and otherwise predictions.
    For example, the rule
    `if X[i, 1] > 3 & X[i, 2] < 4, then 5 else 4` is a rule with two
    conditions. The name `otherwise` is used internally instead of `else` since
    `else` is a reserved keyword.
    """

    def __init__(self, path: Clause, then: float = None, otherwise: float = None):
        self.clause = path
        self.then = then  # in julia its LeafContent = Vector{Float64}
        self.otherwise = otherwise  # in julia its LeafContent = Vector{Float64}

    def subclauses(self):
        return self.clause.subclauses

    def reverse(self):
        """
        Return a reversed version of the `rule`.
        Assumes that the rule has only one split (conditions) since two conditions
        cannot be reversed.
        """
        conditions = self.subclauses()
        assert len(conditions) == 1, "Can only reverse a rule with one condition"
        condition = conditions[0]
        path = Clause([condition.reverse()])
        return Rule(path, self.otherwise, self.then)

    def left_rule(self):
        conditions = self.subclauses()
        assert len(conditions) == 1, "Can only make a rule left that has one condition"
        condition = conditions[0]
        return self if condition.direction == np.less else self.reverse()

    def __eq__(self, other):
        if isinstance(other, Rule):
            if (
                other.clause == self.clause
                and other.then == self.then
                and other.otherwise == self.otherwise
            ):
                return True
            return False

        return False

    def __str__(self):
        path_str = self.clause.string()
        return path_str + f" then {self.then} else {self.otherwise} \n"

    def __hash__(self):
        # Use hash based on subclauses, then, and otherwise
        return hash(
            (
                tuple(hash(sc) for sc in self.subclauses()),
                hash(self.then),
                hash(self.otherwise),
            )
        )

    def add_quantile_idxs(self, quantiles: np.ndarray):
        """
        Adds the quantile index (to which each split value belongs to) for every subclause.

        For example: X[0] > 0.1, we want to find which quantile of feature 0 that 0.1 belongs to
        and add it to the subclause attribute quantile_idx.
        """
        sub_clauses = []
        for subclause in self.clause.subclauses:
            feature_idx = subclause.feature_idx
            split_value = subclause.split_value
            q_idx = get_quantile_index(split_value, quantiles[feature_idx])
            subclause.quantile_idx = q_idx
            sub_clauses.append(subclause)
        self.clause = Clause(sub_clauses)


def get_rules_count(rules: list[Rule]) -> dict:
    """Return a dictionary containing the count of occurrences for each rule."""
    rules_count = defaultdict(int)
    for rule in rules:
        adjusted_rule = rule.left_rule() if len(rule.subclauses()) == 1 else rule
        rules_count[adjusted_rule] += 1
    return rules_count


def get_quantile_rules(rules: list[Rule], quantiles: list) -> dict:
    """Returns a dictionary where the keys are unique quantile Clauses
    and the values are the rules that belong to those Clauses.

     In this function, split values of the subclauses are not considered,
     and a Clause is considered the same if they have the same set of
     subclauses split on the same feature and have the same quantile index.

     Example:
         Clause 1: x[0] < quantile_idx 1 & x[1] < quantile_idx 3
         Clause 2: x[0] < quantile_idx 4 & x[1] < quantile_idx 3
         Clause 3: x[0] < quantile_idx 1 & x[1] < quantile_idx 3

     Above Clause 1 and Clause 3 are considered the same
     and placed as the same key even if they may have different split values.
    """

    for r in rules:
        r.add_quantile_idxs(quantiles)
    quantile_rules = defaultdict(list)
    for r in rules:
        subclauses = copy.deepcopy(
            r.subclauses()[::-1]
        )  # need to create a copy to not change the subclauses in the rules
        for c in subclauses:
            c.split_value = None
        key = Clause(subclauses)
        quantile_rules[key].append(r)
    return quantile_rules


def satisfies_sc(row_data: np.ndarray, subclause: SubClause) -> bool:
    comparison_func, feature_idx, value = (
        subclause.direction,
        subclause.feature_idx,
        subclause.split_value,
    )
    satisfies_constraint = comparison_func(row_data[feature_idx], value)
    return satisfies_constraint


def satisfies_rule(row_data: np.ndarray, rule: Rule) -> bool:
    """
    Return whether data row (a sample point) satisfies rule.
    """
    constraints = [satisfies_sc(row_data, subclause) for subclause in rule.subclauses()]
    return all(constraints)


def get_rule_feature_matrix(
    rules: list[Rule], data: np.ndarray, binary=False
) -> np.ndarray:
    """
    Convert the dataset such that each rule becomes a feature matrix.
    """
    n, p = data.shape
    X = np.empty((n, len(rules)))

    for col_idx, rule in enumerate(rules):
        for row_index in range(n):
            row = data[row_index, :]
            if not binary:
                X[row_index, col_idx] = (
                    rule.then if satisfies_rule(row, rule) else rule.otherwise
                )
            else:
                X[row_index, col_idx] = 1 if satisfies_rule(row, rule) else 0

    return X
