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


def min_max_clip(X: np.ndarray, outliers: BoolMask = None) -> np.ndarray:
    """Scale data to [0-1] range using min and max from non-outlier data. Outliers are clamped to range."""
    masked_X = X[~outliers] if outliers is not None else X
    min_value, max_value = np.min(masked_X, axis=0), np.max(masked_X, axis=0)
    min_bound, max_bound = 0.0, 1.0
    return np.clip((X - min_value) / (max_value - min_value), min_bound, max_bound)


def log_lin_log(X: np.ndarray, outliers: BoolMask = None) -> np.ndarray:
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


def symmetric_log(X: np.ndarray, outliers: BoolMask = None) -> np.ndarray:

    # center the data around the mean of the non-outlier data
    masked_X = X[~outliers] if outliers is not None else X
    mean = np.mean(masked_X, axis=0)
    X -= mean
    masked_X -= mean

    # take the log of the absolute value of the data
    n_log_X = np.log(np.abs(X) + 1) * np.sign(X)
    n_log_X = min_max_clip(n_log_X)

    return n_log_X


# Outlier Detection Methods


def IQR_outlier_detection(X: np.ndarray, threshold: float = 1.5) -> BoolMask:
    """
    Detect outliers in a dataset using the interquartile range method.
    Data need not be time-series.
    """
    q1, q3 = np.percentile(X, [1, 99], axis=0)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return np.logical_or(X < lower_bound, X > upper_bound)


# Outlier Robust Main Call


def robust_normalization(dataframe, method=None):
    """Runs outlier robust normalization on a pandas dataframe.

    Args:
        dataframe (pandas.Datafrme): _description_
        method (string, optional): The method to run for outlier
            normalization. Can be "min_max", "log_lin_log", or "symmetric_log". Defaults to None.

    Returns:
        pandas.Dataframe: Dataframe with normalized value.
    """
    dfs = []
    features = dataframe.feature.unique()

    for f in features:
        feat = dataframe[dataframe["feature"] == f].copy()
        outliers = IQR_outlier_detection(feat["value"])
        if method == "min_max" or not method:
            feat["value"] = min_max_clip(feat["value"], outliers)
        if method == "log_lin_log":
            feat["value"] = log_lin_log(feat["value"], outliers)
        if method == "symmetric_log":
            feat["value"] = symmetric_log(feat["value"], outliers)
        dfs.append(feat)
    return pandas.concat(dfs)
