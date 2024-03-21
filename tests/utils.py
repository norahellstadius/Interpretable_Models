import sys 
from typing import Union

sys.path.append("..")
from src.rules import Rule

sys.path.append("..")
from src.tree import DecisionTreeRegression, DecisionTreeClassification

def valid_split_point_with_quantile(tree: Union[DecisionTreeRegression, DecisionTreeClassification], splits: list) -> bool:
    '''Return true if split_value exists in quantiles'''
    stack = [tree.root_node]

    while stack: 
        node = stack.pop()
        if node.splitpoint.split_value not in splits[node.splitpoint.feature]:
                return False 
        if node.left and not node.left.is_leaf: stack.append(node.left)
        if node.right and not node.right.is_leaf: stack.append(node.right)

    return True

def valid_split_value_in_rules(rules: list[Rule], splits: list) -> bool:
    '''Return true if split_value exists in quantiles'''
    for r in rules: 
        for sc in r.subclauses():
            if sc.split_value not in [round(v, 4) for v in splits[sc.feature_idx]]:
                return False 
    
    return True
        