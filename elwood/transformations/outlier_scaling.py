from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.stats import yeojohnson, gaussian_kde
from matplotlib import pyplot as plt
from scipy.stats import rankdata


# Collection of different normalization methods, not part of core functionality.
# Left in just in case any of these need to be used in the future.


"""
stuff that doesn't work:
- taking the percentile ranks of the data. turns out to literally just be linspace(0, 1, len(data))
- cdf of kernel density estimate of the data. Doesn't sample the data well near normal points
- naive approaches to taking the log of the data (e.g. np.log(data-min(data)+1))
"""


def get_random_dataset(
    num_points: int = 100,
    outlier_prob: float = 0.05,
    mean_range: tuple[int, int] = (-1, 1),
    std_range: tuple[int, int] = (0.5, 1.5),
    num_dimensions: int = 1,
    outlier_scale: float = 1.0,
    outlier_decay: float = 2.0,
) -> np.ndarray:
    """
    Generates a random dataset with outliers.

    Parameters:
    num_points (int): The number of data points to generate. Default is 100.
    outlier_prob (float): The probability of introducing an outlier. Default is 0.05.
    mean_range (tuple): A tuple specifying the range of possible mean values to generate. Default is 0.
    std_range (tuple): A tuple specifying the range of possible standard deviation values to generate. Default is 1.
    num_dimensions (int): The number of dimensions to generate for the data. Default is 1.
    outlier_scale (float): The scale factor used to control the distance of the outlier from the normal data. Default is 1.0.
    outlier_decay (float): The decay factor used to control how quickly the distance of the outlier grows as its index increases. Default is 2.0.

    Returns:
    np.ndarray: A NumPy array of the generated data.
    """
    mean_min, mean_max = mean_range
    std_min, std_max = std_range

    # Generate random mean and standard deviation
    mean = np.random.uniform(mean_min, mean_max, num_dimensions)
    std = np.random.uniform(std_min, std_max, num_dimensions)

    # Generate random data
    data = np.random.normal(mean, std, (num_points, num_dimensions))

    # Introduce outliers
    num_outliers = int(num_points * outlier_prob)
    outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
    outlier_scale = outlier_scale ** np.arange(num_outliers)
    outlier_scale = outlier_scale[:, np.newaxis]
    outlier_decay = outlier_decay ** np.arange(num_dimensions)
    outlier_decay = outlier_decay[np.newaxis, :]
    outlier_mean = np.random.uniform(
        mean_min - 5, mean_max + 5, (num_outliers, num_dimensions)
    )
    outlier_std = np.random.uniform(std_min, std_max, (num_outliers, num_dimensions))
    data[outlier_indices] = np.random.normal(
        outlier_mean, outlier_std * outlier_scale * outlier_decay
    )

    return data


BoolMask = NDArray[np.bool_]  # type alias for boolean mask


####################### Outlier Detection Functions #######################


def mahalanobis_outlier_detection(X: np.ndarray, threshold: float = 3.0) -> BoolMask:
    """
    Detect outliers in a dataset using the Mahalanobis distance method.
    Data need not be time-series. Nor do features need to be independent.
    """
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    inv_cov = np.linalg.inv(cov)
    dist = np.sqrt(np.sum((X - mean) @ inv_cov * (X - mean), axis=1))
    return dist > threshold


def z_score_outlier_detection(X: np.ndarray, threshold: float = 3.0) -> BoolMask:
    """
    Detect outliers in a dataset using the z-score method.
    Data need not be time-series. Assumes features are independent.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return np.abs(X - mean) > threshold * std


def IQR_outlier_detection(X: np.ndarray, threshold: float = 1.5) -> BoolMask:
    """
    Detect outliers in a dataset using the interquartile range method.
    Data need not be time-series.
    """
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return np.logical_or(X < lower_bound, X > upper_bound)


def MAD_outlier_detection(X: np.ndarray, threshold: float = 3.0) -> BoolMask:
    """
    Detect outliers in a dataset using the median absolute deviation method.
    Data need not be time-series.
    """
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    return np.abs(X - median) > threshold * mad


####################### Normalization Functions #######################


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def min_max_clip(X: np.ndarray, outliers: BoolMask | None = None) -> np.ndarray:
    """Scale data to [0-1] range using min and max from non-outlier data. Outliers are clamped to range."""
    masked_X = X[~outliers] if outliers is not None else X
    min_value, max_value = np.min(masked_X, axis=0), np.max(masked_X, axis=0)
    assert max_value > min_value, "max_value must be greater than min_value"
    min_bound, max_bound = 0.0, 1.0
    return np.clip((X - min_value) / (max_value - min_value), min_bound, max_bound)


def min_max_sigmoid(X: np.ndarray, outliers: BoolMask | None = None) -> np.ndarray:
    """Scale data to [-1,1] range using min and max from non-outlier data. Then squashes data including outliers to [0-1] range using sigmoid function."""
    masked_X = X[~outliers] if outliers is not None else X
    min_value, max_value = np.min(masked_X, axis=0), np.max(masked_X, axis=0)
    assert max_value > min_value, "max_value must be greater than min_value"
    X = (X - min_value) / (max_value - min_value) * 2 - 1
    return sigmoid(X)


def IQR_sigmoid(
    X: np.ndarray, outliers: BoolMask | None = None, threshold: float = 1.5
) -> np.ndarray:
    """Scale/center data using IQR+median from non-outlier data. Then squashes data including outliers to [0-1] range using sigmoid function."""
    masked_X = X[~outliers] if outliers is not None else X
    q1, q3 = np.percentile(masked_X, [25, 75], axis=0)
    iqr = q3 - q1
    return sigmoid((X - np.median(masked_X, axis=0)) / (threshold * iqr))


def z_score_sigmoid(
    X: np.ndarray, outliers: BoolMask | None = None, threshold: float = 3.0
) -> np.ndarray:
    """Scale data to [-1,1] range using z-score from non-outlier data. Then squashes data including outliers to [0-1] range using sigmoid function."""
    masked_X = X[~outliers] if outliers is not None else X
    mean = np.mean(masked_X, axis=0)
    std = np.std(masked_X, axis=0)
    return sigmoid((X - mean) / (threshold * std))


def ranker(X: np.ndarray) -> np.ndarray:
    """use rankdata to scale data to [0-1] range"""
    return rankdata(X, method="min") / len(X)


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


# def log_min_max_scaling(data, epsilon=1e-8):
#     """
#     Apply log transformation and min-max scaling to the input data.

#     :param data: Input data as a 1D NumPy array or list
#     :param epsilon: Small constant to add to the data before log transformation (default: 1e-8)
#     :return: Scaled data as a NumPy array
#     """

#     # Ensure the input data is a NumPy array
#     data = np.array(data)

#     # Subtract the minimum value from the data
#     data = data - np.min(data)

#     # Add a small epsilon value to the data
#     data = data + epsilon

#     # Pass the data through the log function
#     log_data = np.log(data)

#     # Subtract the new min from the data
#     log_data = log_data - np.min(log_data)

#     # Divide by the range of the data
#     normalized_data = log_data / (np.max(log_data) - np.min(log_data))

#     return normalized_data

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import pdb

    np.random.seed(42)
    for _ in range(1):
        # data = np.array([10, 25, 12, 50, 75, 984, 5, 100, 95, -4567, 22, 30, -15, 0]).reshape(-1, 1)
        data = get_random_dataset(
            100, mean_range=(-100, 100), std_range=(1, 10), outlier_scale=100
        )

        # generate data with an exponential distribution
        # data = np.random.exponential(1, 100)[:,None] - np.random.exponential(1, 100)[:,None]

        # sort the data
        data = np.sort(data, axis=0)

        outliers = IQR_outlier_detection(data)
        # outliers = z_score_outlier_detection(data)
        # outliers = MAD_outlier_detection(data)

        # make a copy of the data and set outliers to the mean of the non-outliers
        no_outliers_y = min_max_clip(data, outliers)[~outliers]
        no_outliers_x = np.arange(data.shape[0])[:, None][~outliers]

        # Apply the transformations, and normalize the data
        yj_data, _ = yeojohnson(data)
        # n_log_data = log_min_max_scaling(data)
        # nn_log_data = -log_min_max_scaling(-data) + 1
        n_log_data = symmetric_log(data, outliers)
        n_yj_data = min_max_clip(yj_data)
        n_data = min_max_clip(data)

        # outlier aware approaches
        mmc_data = min_max_clip(data, outliers)
        mms_data = min_max_sigmoid(data, outliers)
        iqr_data = IQR_sigmoid(data, outliers)
        z_data = z_score_sigmoid(data, outliers)
        ranks = ranker(data)
        lll = log_lin_log(data, outliers)

        to_plot = [
            (no_outliers_x, no_outliers_y, "Data without outliers"),
            # (n_data, 'min-max normalized data'),
            # (n_yj_data, 'Yeo-Johnson Data'),
            (mmc_data, "Min-Max Clip"),
            (mms_data, "Min-Max Sigmoid"),
            (lll, "Log-Linear-Log"),
            (n_log_data, "Symmetric Log"),
            # (iqr_data, 'IQR Sigmoid'),
            # (z_data, 'Z-Score Sigmoid'),
            # (ranks, 'Percentiles'),
        ]

        # plot each of the transformations as a curve
        fig, ax = plt.subplots(1, 2, sharey=True)
        for row in to_plot:
            data, label = row[:-1], row[-1]
            ax[1].plot(*data, label=label)

        # draw circles around outliers
        ax[1].scatter(
            np.where(outliers)[0],
            mmc_data[outliers],
            c="r",
            marker="o",
            label="outliers",
        )

        ax[1].legend()

        # plot the data as a vertical scatter plot for each transformation
        # plt.figure()
        for i, row in enumerate(to_plot):
            data, label = row[:-1], row[-1]
            if len(data) == 1:
                data = data[0]
            else:
                data = data[1]
            # plot points vertically and semi-transparently
            ax[0].scatter(np.ones_like(data) * i, data, label=label, alpha=0.25, s=100)
        # leg = ax[0].legend()
        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)
        ax[0].set_xticks([])

        # set to tight layout
        plt.tight_layout()

        plt.show()

        # second plot where


# def normalize_data(
#         X:np.ndarray,
#         outlier_fn:Callable[[np.ndarray], np.ndarray],
#         normalize_fn:Callable[[np.ndarray, NDArray[np.bool_]|None], np.ndarray],
#         max_iter:int=10,
#         include_outliers:bool=True
# ) -> np.ndarray:
#     """
#     Normalize a dataset by removing outliers and then normalizing the data.

#     The ordering of points in the dataset is maintained (including if outliers are included in the final result).

#     Parameters
#     ----------
#     `X: np.ndarray`
#         The dataset to normalize. Each row is a data point.
#     `outlier_fn: Callable[[np.ndarray, float], np.ndarray]`
#         The function to use to detect outliers in the dataset.
#     `normalize_fn: Callable[[np.ndarray, NDArray[np.bool_]], np.ndarray]`
#         The function to use to normalize the dataset. The first argument is the dataset, and the second argument is a boolean mask indicating which points are outliers.
#     `threshold: float`
#         The threshold to use for outlier detection.
#     `max_iter: int`, optional
#         The maximum number of iterations to perform, by default `10`
#     `include_outliers: bool`, optional
#         Whether to include outliers in the normalized dataset, by default True

#     Returns
#     -------
#     `np.ndarray`
#         The normalized dataset.
#     """
#     # initialize the outlier mask
#     outlier_mask = np.full(X.shape[0], False)
#     # iterate until no outliers are detected
#     for _ in range(max_iter):
#         # remove outliers from the dataset
#         masked_X = X[~outlier_mask]
#         # detect new outliers in the masked dataset
#         new_outliers = outlier_fn(masked_X)
#         # get the indices of the new outliers in the original dataset
#         new_outlier_indices = np.where(~outlier_mask)[0][new_outliers]
#         # update the outlier mask
#         outlier_mask[new_outlier_indices] = True
#         # if no new outliers were detected, break
#         if not np.any(new_outliers):
#             break

#     # normalize the dataset
#     X = normalize_fn(X, outlier_mask)

#     if not include_outliers:
#         # remove outliers from the dataset
#         X = X[~outlier_mask]

#     return X


# def fast_data_normalization(X:np.ndarray, threshold=1.5) -> np.ndarray:


# def correct_data_normalization(): ...


# def get_covariance_matrix(X:np.ndarray) -> np.ndarray:
#     """Compute the covariance matrix for a set of data points."""
#     return np.cov(X.T)


# def rolling_outlier_detection(y:np.ndarray, window:int=10, threshold:float=3.0):
#     """Detect outliers in time-series data by computing the rolling mean and standard deviation."""
#     #use convolution to generate the rolling mean and standard deviation

#     mean = np.convolve(y, np.ones(window), 'same') / window
#     stdev = np.sqrt(np.convolve(y**2, np.ones(window), 'same') / window - mean**2)
#     return np.abs(y - mean) > threshold * stdev


# def stdev_outlier_detection(y:np.ndarray, threshold:float=3.0):
#     """
#     Detect outliers in a dataset using the standard deviation method.
#     Data need not be time-series.
#     """
#     mean = np.mean(y)
#     stdev = np.std(y)
#     return np.abs(y - mean) > threshold * stdev


# def iqr_outlier_detection(y:np.ndarray, threshold:float=1.5):
#     """
#     Detect outliers in a dataset using the interquartile range method.
#     Data need not be time-series.
#     """
#     q1, q3 = np.percentile(y, [25, 75])
#     iqr = q3 - q1
#     return np.abs(y - np.median(y)) > threshold * iqr


# #create several random datasets all with outliers of varying prominence
# np.random.seed(0)
# n = 1000
# x = np.linspace(0, 10, n)
# y1 = np.sin(x) + np.random.normal(0, 0.2, n)
# y2 = np.sin(x) + np.random.normal(0, 0.5, n)
