import numpy as np


def rough_cutpoint_index_estimate(n: int, quantile: float) -> int:
    """
    Return a rough estimate for the index of the cutpoint.

    Choose the highest suitable index if there is more than one suitable index.

    Parameters:
    - n (int): The length of the data array.
    - quantile (float): The desired quantile value between 0.0 and 1.0.

    Returns:
    int: The estimated index of the cutpoint.
    """
    return int(quantile * n)


def empirical_quantile(data_array: np.ndarray, quantile: float) -> float:
    """
    Return the empirical `quantile` for data `data_array`.

    Parameters:
    - data_array (array-like): The input data array.
    - quantile (float): The desired quantile value between 0.0 and 1.0.

    Returns:
    np.float32: The empirical quantile value.
    """
    assert 0.0 <= quantile <= 1.0
    n = len(data_array)
    index = rough_cutpoint_index_estimate(n, quantile)
    sorted_data = np.sort(data_array)
    return sorted_data[index]


def cutpoint_for_feature(data_array: np.ndarray, q: int) -> np.ndarray:
    """
    Return a vector of `q` cutpoints taken from the empirical distribution of data `data_array`.

    Parameters:
    - data_array (array-like): The input data array.
    - q (int): The number of cutpoints to generate.

    Returns:
    np.ndarray: A vector of `q` cutpoints.
    """
    assert q >= 2
    # Taking 2 extra to avoid getting minimum(data_array) and maximum(data_array) becoming cutpoints.
    # Tree on left and right have always respectively length 0 and 1 then anyway.
    length = q + 2
    quantiles = np.linspace(0.0, 1.0, length)[1:-1]
    return np.array(
        [empirical_quantile(data_array, quantile) for quantile in quantiles]
    )


def cutpoints(X: np.ndarray, q: int) -> np.ndarray:
    """
    Return a vector of vectors containing cutpoints for each feature in the dataset `X`.

    Parameters:
    - X (array-like): The dataset with features.
    - q (int): The number of cutpoints to generate for each feature. Default is 10.

    Returns:
    list: A list of vectors containing `q` unique cutpoints for each feature.
    """
    _, p = X.shape  # Assuming X is a 2D array-like structure
    cps = [np.unique(cutpoint_for_feature(X[:, feature], q)) for feature in range(p)]
    return cps


def get_quantile_index(split_value: float, quantile_values: np.ndarray):
    quantile_values.sort()  # Ensure quantile_values is sorted
    for i in range(len(quantile_values)):
        q_val = quantile_values[i]
        if split_value <= q_val:
            return i
    return len(quantile_values)


if __name__ == "__main__":
    from data import get_BW_data, get_boston_housing
    from sklearn.model_selection import train_test_split

    # X, y = get_BW_data("/Users/norahallqvist/Code/SIRUS/data/BreastWisconsin.csv")
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1
    # )
    # splits = cutpoints(X=X_train, q=10)

    X, y = get_boston_housing("/Users/norahallqvist/Code/SIRUS/data/boston_housing.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    splits = cutpoints(X=X_train, q=10)
    print(len(splits[0]))
