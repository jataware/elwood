import math
import numpy
import pandas as pd
from typing import Dict, List
import xarray
from datetime import datetime

import sys


def regrid_dataframe(
    dataframe,  # Can now be a pandas DF or an xarray Dataset
    geo_columns: Dict[str, str],
    time_column: List[str],
    scale_multi: float,
    aggregation_functions: Dict[str, str],
    scale=None,  # Can either be a float or a dict with two keys, one for "lat_scale" and one for "lon_scale".
    xarray_return=False,
) -> pd.core.frame.DataFrame:
    """Uses nump digitize and pandas functions to regrid geography in a dataframe.

    Args:
        dataframe (pandas.Dataframe): Dataframe of a dataset that has detectable gridden geographical resolution ie. points that represent 1sqkm areas
        geo_columns (Dict[str]): A dictionary containing the geo_columns for the latitude and longitude pairs, with keys 'lat_column' and 'lon_column'.
        time_column (List[str]): A list containing the name of the datetime column(s) in the dataset.
        scale_multi (float): The number by which to divide to geographical scale to regrid larger.
        aggregation_functions (Dict[str, str]): Example: {"T": "sum", "EPV": "sum", "SLP": "mean", "H": "mean"}
        scale (float): Overwrites automatic scale detection with a user input scale. Defaults to None.

    Returns:
        pandas.Dataframe: Dataframe with geographical extend regridded to
    """

    if isinstance(dataframe, xarray.Dataset):
        print("Xarray Dataset input")
        if xarray_return:
            xarray_dataset_indices = list(dataframe.dims)
        dataframe = (
            dataframe.to_dataframe()
        )  # Loses indices, hence why we persist them above.

        dataframe = dataframe.reset_index()

        print(dataframe)

    # Create arrays for latitude and longitude
    df_lats = dataframe[geo_columns["lat_column"]]
    df_lons = dataframe[geo_columns["lon_column"]]

    # Automatic scale detection, or user provided scale overwrite.
    dataframe_scale = 0
    if scale:
        if isinstance(scale, dict):
            # Allows user to provide separate scales for lat and lon
            lat_scale = scale["lat_scale"]
            lon_scale = scale["lon_scale"]
        else:
            lat_scale = scale
            lon_scale = scale
    else:
        lat_scale = abs((df_lats.unique()[1] - df_lats.unique()[0]) * scale_multi)
        lon_scale = abs((df_lons.unique()[1] - df_lons.unique()[0]) * scale_multi)

    lat = numpy.arange(df_lats.min(), df_lats.max() + 1, lat_scale)
    lon = numpy.arange(df_lons.min(), df_lons.max() + 1, lon_scale)

    # calculate the indices of the bins to which each value in input array belongs
    dataframe["lon_bin"] = numpy.digitize(dataframe[geo_columns["lon_column"]], lon) - 1
    dataframe["lat_bin"] = numpy.digitize(dataframe[geo_columns["lat_column"]], lat) - 1

    # use bin indices to create bin labels (which are the actual bin values)
    dataframe["lon_bin_label"] = lon[dataframe["lon_bin"]]
    dataframe["lat_bin_label"] = lat[dataframe["lat_bin"]]

    print(dataframe)

    # mapper = {"T": "sum", "EPV": "sum", "SLP": "mean", "H": "mean"}
    mapper = aggregation_functions

    aggregation_indices = [time_column, "lon_bin_label", "lat_bin_label"]
    column_drop = {
        "lon_bin_label",
        "lon_bin",
        "lat_bin_label",
        "lat_bin",
    }

    # Instance for one aggregation function.
    if isinstance(mapper, list):
        agg_func = mapper[0]

        result_frame = (
            dataframe.groupby(aggregation_indices)
            .agg(lambda x: aggregation_by_type(x, agg_func))
            .reset_index()
        )

        print(f"FRAME AFTER GROUPING: {result_frame}")

        #  dataframe.groupby(aggregation_indices).sum().reset_index()
        result = result_frame.drop(columns=column_drop)
        try:
            result[time_column] = pd.to_datetime(result[time_column])
        except TypeError:
            result[time_column] = result[time_column].apply(
                lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
            )
        if xarray_return:
            print(f"RESULTING DATA: {result}")
            result.set_index(xarray_dataset_indices, inplace=True)
            result = result[~result.index.duplicated()]

            xr_dataframe = result.to_xarray()
            return xr_dataframe
        return result

    # Take agg functions and do aggregation on respective slices of the dataset.
    aggregation_processing = {}

    for key, value in mapper.items():
        if value in aggregation_processing:
            aggregation_processing[value].append(key)
        else:
            aggregation_processing[value] = [key]

    # Process all dataframe slices with correct aggregation function and store all the result slices for merging.
    all_result_frames = []
    for agg_func, columns_list in aggregation_processing.items():
        frame_selector = aggregation_indices + columns_list

        target_frame = dataframe[frame_selector]

        # group by bins (i.e., group by 1-degree grid cells) and calculate aggregation in each grid cell
        result_frame = (
            dataframe.groupby(aggregation_indices)
            .agg(lambda x: aggregation_by_type(x, agg_func))
            .reset_index()
        )

        all_result_frames.append(result_frame)

    # Combine all result frames
    result = None
    for frame in all_result_frames:
        if result == None:
            result = frame
        result = pd.merge(
            result,
            frame,
            how="left",
            left_on=aggregation_indices,
            right_on=aggregation_indices,
        )

    try:
        result[time_column] = pd.to_datetime(result[time_column])
    except TypeError:
        result[time_column] = result[time_column].apply(
            lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
        )
    try:
        result = result.drop(columns=column_drop)
    except KeyError:
        column_drop.remove("lon_bin")
        column_drop.remove("lat_bin")
        result = result.drop(columns=column_drop)

    if xarray_return:
        result.set_index(xarray_dataset_indices, inplace=True)
        result = result[~result.index.duplicated()]

        xr_dataframe = result.to_xarray()
        return xr_dataframe
    return result


def aggregation_by_type(series, aggregation_method):
    if pd.api.types.is_numeric_dtype(series):
        return getattr(series, aggregation_method)()
    else:
        return series.mode().iloc[0]
