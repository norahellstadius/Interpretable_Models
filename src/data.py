from enum import Enum
from typing import Tuple
import numpy as np
import pandas as pd

class DataType(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2

# classfication
def get_BW_data(
    data_path="/Users/norahallqvist/Code/Interpretable_Models/data/BreastWisconsin.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    full_data = pd.read_csv(data_path)
    y = full_data["diagnosis"].replace({"B": 0, "M": 1})
    X = full_data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
    return X.to_numpy(), y.to_numpy()


# regression
def get_boston_housing(
    data_path: str = "/Users/norahallqvist/Code/Interpretable_Models/data/boston_housing.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    df.columns = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
        "BIAS_COL",
    ]
    X = df.drop(["MEDV", "BIAS_COL"], axis=1)
    y = df["MEDV"]

    return X.to_numpy(), y.to_numpy()
