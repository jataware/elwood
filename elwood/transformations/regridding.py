import math
import numpy
import pandas as pd
from typing import Dict, List

import sys


def regrid_dataframe(
    dataframe: pd.core.frame.DataFrame,
    geo_columns: Dict[str, str],
    time_column: List[str],
    scale_multi: float,
    aggregation_functions: Dict[str, str],
    scale: float = None,
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

    # Create arrays for latitude and longitude
    df_lats = dataframe[geo_columns["lat_column"]]
    df_lons = dataframe[geo_columns["lon_column"]]

    # Automatic scale detection, or user provided scale overwrite.
    dataframe_scale = 0
    if scale:
        lat_scale = scale
        lon_scale = scale
    else:
        # TODO check that this is how the scale multi is being used in the frontend.
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

        result_frame = getattr(
            dataframe.groupby(aggregation_indices), agg_func
        )().reset_index()

        result = result_frame.drop(columns=column_drop)
        result[time_column] = pd.to_datetime(result[time_column])

        return result

    # Take agg functions and do aggregation on respective slices of the dataset.
    aggregation_processing = {}

    for key, value in mapper.items():
        if value in aggregation_processing:
            aggregation_processing[value].append(key)
        else:
            aggregation_processing[value] = [key]

    print(f"AGG PROCESS: {aggregation_processing}")

    # Process all dataframe slices with correct aggregation function and store all the result slices for merging.
    all_result_frames = []
    for agg_func, columns_list in aggregation_processing.items():
        frame_selector = aggregation_indices + columns_list

        target_frame = dataframe[frame_selector]

        # group by bins (i.e., group by 1-degree grid cells) and calculate aggregation in each grid cell
        result_frame = getattr(
            target_frame.groupby(aggregation_indices), agg_func
        )().reset_index()

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

    result[time_column] = pd.to_datetime(result[time_column])
    try:
        result = result.drop(columns=column_drop)
    except KeyError:
        column_drop.remove("lon_bin")
        column_drop.remove("lat_bin")
        result = result.drop(columns=column_drop)

    return result
