import numpy as np
import pandas
from numpy.typing import NDArray
from scipy.stats import yeojohnson, gaussian_kde
from matplotlib import pyplot as plt
from scipy.stats import rankdata


def zero_to_one_normalization(dataframe):
    """
    This function accepts a dataframe in the canonical format
    and min/max scales each feature to between 0 to 1
    """
    dfs = []
    features = dataframe.feature.unique()

    for f in features:
        feat = dataframe[dataframe["feature"] == f].copy()
        feat["value"] = process_zero_to_one(feat["value"])
        dfs.append(feat)
    return pandas.concat(dfs)


def process_zero_to_one(data):
    """
    This function takes in an array and performs 0 to 1 normalization on it.
    It is robust to NaN values and ignores them (leaves as NaN).
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Outlier Robust Normalizations

BoolMask = NDArray[np.bool_]  # type alias for boolean mask


def min_max_clip(X: np.ndarray, outliers: BoolMask | None = None) -> np.ndarray:
    """Scale data to [0-1] range using min and max from non-outlier data. Outliers are clamped to range."""
    masked_X = X[~outliers] if outliers is not None else X
    min_value, max_value = np.min(masked_X, axis=0), np.max(masked_X, axis=0)
    assert max_value > min_value, "max_value must be greater than min_value"
    min_bound, max_bound = 0.0, 1.0
    return np.clip((X - min_value) / (max_value - min_value), min_bound, max_bound)


def log_lin_log(X: np.ndarray, outliers: BoolMask | None = None) -> np.ndarray:
    """all non-outlier data is spaced linearly, while more extreme data is spaced logarithmically"""

    # allocate the output array
    output = np.zeros_like(X)

    # center the data around the mean of the non-outlier data
    masked_X = X[~outliers] if outliers is not None else X
    mean = np.mean(masked_X, axis=0)
    X -= mean
    masked_X -= mean

    # separate the positive outliers from the negative outliers
    min_value, max_value = np.min(masked_X, axis=0), np.max(masked_X, axis=0)
    neg_outliers = X < min_value
    pos_outliers = X > max_value

    # center the data around the mean, and handle the pos and negative outliers separately
    # negative is handled by taking the log of the absolute value, then negating the result
    output[~outliers] = X[~outliers]
    output[pos_outliers] = np.log(X[pos_outliers]) - np.log(max_value) + max_value
    output[neg_outliers] = -(np.log(-X[neg_outliers]) - np.log(-min_value) - min_value)

    # scale the data to [0-1] range
    output = (output - np.min(output, axis=0)) / (
        np.max(output, axis=0) - np.min(output, axis=0)
    )

    return output


def symmetric_log(X: np.ndarray, outliers: BoolMask | None = None) -> np.ndarray:

    # center the data around the mean of the non-outlier data
    masked_X = X[~outliers] if outliers is not None else X
    mean = np.mean(masked_X, axis=0)
    X -= mean
    masked_X -= mean

    # take the log of the absolute value of the data
    n_log_X = np.log(np.abs(X) + 1) * np.sign(X)
    n_log_X = min_max_clip(n_log_X)

    return n_log_X
