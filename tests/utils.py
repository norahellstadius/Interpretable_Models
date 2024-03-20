import sys 
from typing import Union

sys.path.append("..")
from src.tree import DecisionTreeRegression, DecisionTreeClassification

def valid_split_point_with_quantile(tree: Union[DecisionTreeRegression, DecisionTreeClassification], splits: list) -> bool:
    stack = [tree.root_node]

    while stack: 
        node = stack.pop()
        if node.splitpoint.split_value not in splits[node.splitpoint.feature]:
                return False 
        if node.left and not node.left.is_leaf: stack.append(node.left)
        if node.right and not node.right.is_leaf: stack.append(node.right)

    return True