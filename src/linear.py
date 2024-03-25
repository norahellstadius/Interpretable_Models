import sys
import numpy as np
from enum import Enum
from sklearn.linear_model import (
    LogisticRegressionCV,
    RidgeCV,
    LinearRegression, 
    LogisticRegression
)
import l0learn
sys.path.append("..")
from src.data import DataType

class RegType(Enum):
    NONE = 1
    RIDGE = 2
    L0 = 3

def fit_lm(X: np.ndarray, y: np.ndarray, data_type: DataType): 
    if data_type.name == DataType.REGRESSION.name:
        model = LinearRegression(fit_intercept=True)
    elif data_type.name == DataType.CLASSIFICATION.name:
        model = LogisticRegression(fit_intercept=True)
    else: 
        raise ValueError(f"data_type must be of DataType.REGRESSION or DataType.CLASSIFICATION, but given {data_type}")
    model.fit(X, y)
    return model

def fit_ridge(X: np.ndarray, y: np.ndarray, data_type: DataType):
    if data_type.name == DataType.REGRESSION.name:
        alphas = np.logspace(np.log10(1e-4), np.log10(1e4), num=100, endpoint=True)
        model = RidgeCV(cv=5, alphas=alphas, fit_intercept=True)
        model.fit(X, y)
        model.coef_ = np.maximum(model.coef_, 0)
    elif data_type.name == DataType.CLASSIFICATION.name:
        model = LogisticRegressionCV(
            cv=5, Cs=100, fit_intercept=True, penalty="l2", max_iter=10000
        )
        model.fit(X, y)
        model.coef_[0] = np.maximum(model.coef_[0], 0)
    else: 
        raise ValueError(f"data_type must be of DataType.REGRESSION or DataType.CLASSIFICATION, but given {data_type}")

    return model
    
def fit_L0(
    X_train: np.ndarray,
    y_train: np.ndarray,
    data_type: DataType,
    max_rules: int,
    penalty: str = "L0L2",  # check "L0"
    num_folds: int = 5,
    num_gamma: int = 5,
    gamma_min: float = 0.0001,
    gamma_max: float = 0.1,
    algorithm: str = "CDPSI",
    random_state: int = 1, 
):

    assert X_train.shape[0] == len(y_train), "X_train and y_train must have same number of samples"
    if data_type.name == DataType.REGRESSION.name:
        loss_type = "SquaredError"
    elif data_type.name == DataType.CLASSIFICATION.name:
        loss_type = "Logistic"
    else: 
        raise ValueError(f"data_type must be of DataType.REGRESSION or DataType.CLASSIFICATION, but given {data_type}")

    cv_fit_result = l0learn.cvfit(
        X_train,
        y_train.astype(np.float64),
        num_folds=num_folds,
        seed=random_state,
        penalty=penalty,
        num_gamma=num_gamma,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        max_support_size=max_rules,
        algorithm=algorithm,
        loss=loss_type,
    )

    fit_model = l0learn.fit(
        X_train,
        y_train.astype(np.float64),
        penalty=penalty,
        num_gamma=num_gamma,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        max_support_size=max_rules,
        algorithm=algorithm,
        loss=loss_type,
    )

    gamma_mins = [
        (i, np.argmin(cv_mean), np.min(cv_mean))
        for i, cv_mean in enumerate(cv_fit_result.cv_means)
    ]
    optimal_gamma_index, optimal_lambda_index, min_error = min(
        gamma_mins, key=lambda t: t[2]
    )
    optimal_gamma, optimal_lambda = (
        fit_model.gamma[optimal_gamma_index],
        fit_model.lambda_0[optimal_gamma_index][optimal_lambda_index],
    )

    linear_model = fit_model
    optimal_gamma = optimal_gamma
    optimal_lambda = optimal_lambda
    coeffs = cv_fit_result.coeff(
        lambda_0=optimal_lambda, gamma=optimal_gamma, include_intercept=False
    ).toarray()
    
    
    return {
        "model": linear_model,
        "optimal_gamma": optimal_gamma,
        "optimal_lambda": optimal_lambda,
        "coeffs": coeffs
    }

if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from src.data import get_boston_housing, get_BW_data

    X, y = get_boston_housing()
    data_type = DataType.REGRESSION
    model = fit_lm(X, y, data_type)
    print(model.coef_)

    X, y = get_BW_data()
    data_type = DataType.CLASSIFICATION
    model = fit_lm(X, y, data_type)
    print(model.coef_)
