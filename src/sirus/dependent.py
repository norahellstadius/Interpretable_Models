import copy
import sys
import numpy as np

sys.path.append("../..")
from src.rules import SubClause, Rule


def implies(subclause1: SubClause, subclause2: SubClause):
    if subclause1.feature_idx == subclause2.feature_idx:
        if subclause1.direction == np.less:
            if subclause2.direction == np.less:
                return subclause1.split_value <= subclause2.split_value
            else:
                return False
        else:
            if subclause2.feature_idx == np.greater_equal:
                return subclause1.split_value >= subclause2.split_value
            else:
                return False
    else:
        return False


# check if ondition 1 & condition 2 --> Rule
def implies_condition(subclauses: tuple[SubClause, SubClause], rule: Rule) -> bool:
    cond1, cond2 = subclauses
    implied = [
        any(
            implies(cond1, subclause) or implies(cond2, subclause)
            for subclause in rule.subclauses()
        )
    ]
    return all(implied)


def feature_space(rules: list, sc1: SubClause, sc2: SubClause):
    num_rules = len(rules)
    data_matrix = np.empty(
        (4, num_rules + 1), dtype=bool
    )  # matrix of size 4 x (#rules + 1)
    data_matrix[:, 0] = np.ones((4))  # add ones to the first column

    reverse_sc1 = sc1.reverse()
    reverse_sc2 = sc2.reverse()

    for col_idx in range(1, (num_rules + 1)):
        cur_rule = rules[col_idx - 1]
        data_matrix[0, col_idx] = implies_condition((sc1, sc2), cur_rule)
        data_matrix[1, col_idx] = implies_condition((sc1, reverse_sc2), cur_rule)
        data_matrix[2, col_idx] = implies_condition((reverse_sc1, sc2), cur_rule)
        data_matrix[3, col_idx] = implies_condition(
            (reverse_sc1, reverse_sc2), cur_rule
        )

    return data_matrix


def canonicalize(sc: SubClause):
    "Canonicalize a SubClause by ensuring that the direction is left."
    if sc.direction != np.less:
        return sc.reverse()
    return sc


def unique_left_subclauses(rules: list):
    """
    Return a vector of unique left splits for `rules`.
    These splits will be used to form `(A, B)` pairs and generate the feature space.
    For example, the pair `x[i, 1] < 32000` (A) and `x[i, 3] < 64` (B) will be used to generate
    the f
    """
    unique_left_sc = set()
    for rule in rules:
        for sc in rule.subclauses():
            canonicalized_cs = canonicalize(sc)
            unique_left_sc.add(canonicalized_cs)
    return list(unique_left_sc)


def left_triangular_product(subclause_list: list):
    """
    Return all unique pairs of elements in `V`.
    More formally, return all pairs (v_i, v_j) where i < j.
    """
    n = len(subclause_list)
    product = []
    for i in range(n):
        left = subclause_list[i]
        for j in range(n):
            if i <= j:
                right = subclause_list[j]
                product.append((left, right))
    return product


def related_rule(rule: Rule, sc1: SubClause, sc2: SubClause):
    """
    Return whether some rule is either related to `A` or `B` or both.
    Here, it is very important to get rid of rules which are about the same feature but different thresholds.
    Otherwise, rules will be wrongly classified as linearly dependent in the next step.
    """
    assert (
        sc1.direction == np.less
    ), "Assertion failed: cond1 should be < (something is wrong with _unique_left_conditions)"
    assert (
        sc2.direction == np.less
    ), "Assertion failed: cond 2 should be < (something is wrong with _unique_left_conditions)"
    subclauses = rule.subclauses()  # get subclauses of the rule
    if len(subclauses) == 1:
        single_sc = subclauses[0]
        left_subclause = canonicalize(single_sc)
        return left_subclause == sc1 or left_subclause == sc2
    elif len(subclauses) == 2:
        single_sc_1, single_sc_2 = subclauses
        left_sc_1, left_sc_2 = canonicalize(single_sc_1), canonicalize(single_sc_2)
        return (left_sc_1 == sc1 and left_sc_2 == sc2) or (
            left_sc_1 == sc2 and left_sc_2 == sc1
        )
    else:
        raise ValueError(
            "Unexpected number of conditions in the path. Expected 1 or 2 conditions, but got {} conditions.".format(
                len(subclauses)
            )
        )


def linearly_dependent(rules: list, sc1: SubClause, sc2: SubClause):
    """
    Return a vector of booleans with a true for every rule in `rules` that is linearly dependent on a combination of the previous rules.
    To find rules for this method, collect all rules containing some feature for each pair of features.
    That should be a fairly quick way to find subsets that are easy to process.
    """
    data_matrix = feature_space(rules, sc1, sc2)
    num_rules = len(rules)
    dependent = np.empty(num_rules, dtype=bool)
    atol = 1e-6
    current_rank = np.linalg.matrix_rank(
        data_matrix[:, 0], tol=atol
    )  # should be 1 since data_matrix[:, 0] is an array of 1's

    for i in range(num_rules):
        # adding an additional column and checking if rank increases or decreases
        new_rank = np.linalg.matrix_rank(data_matrix[:, 0 : i + 2], tol=atol)
        if current_rank < new_rank:
            dependent[i] = False
            current_rank = new_rank
        else:
            dependent[i] = True

    return dependent


def gap_size(rule: Rule):
    """
    gap_size(rule::Rule)

    Return the gap size for a rule.
    The gap size is used by Bénard et al. in the appendix of their PMLR paper
    (<https://proceedings.mlr.press/v130/benard21a.html>). FOUND IN SECTION 4 LAST PARAGRAPH
    Via an example, they specify that the gap size is the difference between the
    then and otherwise (else) probabilities.

    A smaller gap size implies a smaller CART-splitting criterion, which implies a
    smaller occurrence frequency.
    """
    # assert len(rule.then) == len(rule.otherwise), "Then and otherwise clause must be the same size"  # TODO: check that then and otherwise is not just a number
    # gap_size_per_class = [abs(then_value - otherwise_value) for then_value,otherwise_value in zip(rule.then,rule.otherwise)]
    return abs(rule.then - rule.otherwise)


def sort_by_gap_size(rules: list) -> list:
    """
    Return the vector rule sorted by decreasing gap size.
    This allows the linearly dependent filter to remove the rules further down the list since
    they have a smaller gap.
    """
    return sorted(rules, key=gap_size, reverse=True)


"""
Return a vector of rules that are not linearly dependent on any other rule.

This is done by considering each pair of splits.
For example, considers the pair `x[i, 1] < 32000` (A) and `x[i, 3] < 64` (B).
Then, for each rule, it checks whether the rule is linearly dependent on the pair.
As soon as a dependent rule is found, it is removed from the set to avoid considering it again.
If we don't do this, we might remove some rule `r` that causes another rule to be linearly
dependent in one related set, but then is removed in another related set.
"""


def filter_linearly_dependent(rules: list):
    sorted = sort_by_gap_size(
        rules
    )  # sort by gapsize so if you add a dependent rule the right one is eliminated
    unique_subclauses = unique_left_subclauses(sorted)
    pairs = left_triangular_product(unique_subclauses)
    independent_rules = copy.deepcopy(sorted)

    for sc1, sc2 in pairs:
        independent_rules_idxs = [
            rule_idx
            for rule_idx, rule in enumerate(independent_rules)
            if related_rule(rule, sc1, sc2)
        ]
        independent_rules_subset = [
            independent_rules[i] for i in independent_rules_idxs
        ]
        dependent_subset = linearly_dependent(
            independent_rules_subset, sc1, sc2
        )  # a list indicating if rule is dependent or not

        assert len(independent_rules_idxs) == len(independent_rules_subset)
        assert len(dependent_subset) == len(independent_rules_subset)

        dependent_indexes = [
            independent_rules_idxs[i]
            for i, is_dependent in enumerate(dependent_subset)
            if is_dependent
        ]
        dependent_indexes.sort()  # is this needed?? #TODO: CHECK if this is needed
        for index in reversed(dependent_indexes):
            independent_rules.pop(index)

    return independent_rules
