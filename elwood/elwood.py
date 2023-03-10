"""Main module."""
import json
import logging
import os
import sys

import geofeather as gf
import numpy as np
import pandas as pd

from . import constants
from .file_processor import (
    process_file_by_filetype,
    raster2df_processor,
    netcdf2df_processor,
)
from .normalizer import normalizer
from .feature_scaling import scale_dataframe
from .transformations.clipping import construct_multipolygon, clip_dataframe, clip_time
from .transformations.scaling import scale_time
from .transformations.regridding import regrid_dataframe
from .transformations.geo_utils import calculate_boundary_box
from .transformations.temporal_utils import calculate_temporal_boundary

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)


# Standardization processor


def process(
    fp: str, mp: str, admin: str, output_file: str, write_output=True, gadm=None
):
    """
    Parameters
    ----------
    mp: str
        Filename for JSON mapper from spacetag.
        Schema: https://github.com/jataware/spacetag/blob/schema/schema.py
        Example: https://github.com/jataware/spacetag/blob/schema/example.json

    gadm: gpd.GeoDataFrame, default None
        optional specification of a GeoDataFrame of GADM shapes of the appropriate
        level (admin2/3) for geocoding
    """

    # Read JSON schema to be mapper.
    mapper = dict
    with open(mp) as f:
        mapper = json.loads(f.read())

    # Validate JSON mapper schema against SpaceTag schema.py model.
    # model = SpaceModel(geo=mapper['geo'], date=mapper['date'], feature=mapper['feature'], meta=mapper['meta'])

    # "meta" portion of schema specifies transformation type
    transform = mapper["meta"]

    # Check transform for meta.geocode_level. Update admin to this if present.
    if admin == None and "geocode_level" in transform:
        admin = transform["geocode_level"]

    ftype = transform["ftype"]
    df = process_file_by_filetype(
        filepath=fp, file_type=ftype, transformation_metadata=transform
    )

    ## Make mapper contain only keys for date, geo, and feature.
    mapper = {k: mapper[k] for k in mapper.keys() & {"date", "geo", "feature"}}

    ## To speed up normalize(), reduce the memory size of the dataframe by:
    # 1. Optimize the dataframe types.
    # 2. Reset the index so it is a RangeIndex instead of Int64Index.
    df = optimize_df_types(df)
    df.reset_index(inplace=True, drop=True)

    ## Run normalizer.
    norm, renamed_col_dict = normalizer(df, mapper, admin, gadm=gadm)

    # Normalizer will add NaN for missing values, e.g. when appending
    # dataframes with different columns. GADM will return None when geocoding
    # but not finding the entity (e.g. admin3 for United States).
    # Replace None with NaN for consistency.
    norm.fillna(value=np.nan, inplace=True)

    if write_output:
        # If any qualify columns were added, the feature_type must be enforced
        # here because pandas will have cast strings as ints etc.
        qualify_cols = set(norm.columns).difference(set(constants.COL_ORDER))
        for col in qualify_cols:
            for feature_dict in mapper["feature"]:
                if (
                    feature_dict["name"] == col
                    and feature_dict["feature_type"] == "string"
                ):
                    norm[col] = norm[col].astype(str)

        # Separate string from other dtypes in value column.
        # This is predicated on the assumption that qualifying feature columns
        # are of a single dtype.

        norm["type"] = norm[["value"]].applymap(type)
        norm_str = norm[norm["type"] == str]
        norm = norm[norm["type"] != str]
        del norm_str["type"]
        del norm["type"]

        # Write parquet files
        norm.to_parquet(f"{output_file}.parquet.gzip", compression="gzip")
        if len(norm_str) > 0:
            norm_str.to_parquet(f"{output_file}_str.parquet.gzip", compression="gzip")

        norm = norm.append(norm_str)

    # Reduce memory size of returned dataframe.
    norm = optimize_df_types(norm)
    norm.reset_index(inplace=True, drop=True)

    return norm, renamed_col_dict


# Transformation functions


def normalize_features(dataframe, output_file: str = None):
    """normalize_features takes a dataframe and scales all numerical features on a 0 to 1 scale.
    This normalizes the data for comparison and visualization.

    Args:
        dataframe (pandas.Dataframe): A pandas dataframe with a "feature" and "value" column.
        Will scale numerical values in the "value" column from 0 to 1.

    Returns:
        pandas.Dataframe: Returns a pandas Dataframe with numerical features scaled from 0 to 1.
    """
    df = scale_dataframe(dataframe)

    if output_file:
        df.to_parquet(f"{output_file}_normalized.parquet.gzip", compression="gzip")
    return df


def clip_geo(dataframe, geo_columns, polygons_list):
    """Clips data based on geographical shape(s) or shapefile (NOT IMPLEMENTED).

    Args:
        dataframe (pandas.Dataframe): A pandas dataframe containing geographical data.
        geo_columns (list): A list containing the two column names for the lat/lon columns in the dataframe.
        polygons_list (list[list[obj]]): A list containing lists of objects that represent polygon shapes to clip to.

    Returns:
        pandas.Dataframe: A pandas dataframe only containing the clipped data.
    """

    mask = construct_multipolygon(polygons_list=polygons_list)

    return clip_dataframe(dataframe=dataframe, geo_columns=geo_columns, mask=mask)


def clip_dataframe_time(dataframe, time_column, time_ranges):
    """Clips data in a dataframe based on a list of time ranges.

    Args:
        dataframe (pandas.Dataframe): Dataframe with some time column that is the target for the clip
        time_column (string): Name of target time column
        time_ranges (List[Dict]): List of dictionaries containing "start" and "end" datetime values

    Returns:
        _type_: _description_
    """

    return clip_time(
        dataframe=dataframe, time_column=time_column, time_ranges=time_ranges
    )


def rescale_dataframe_time(
    dataframe, time_column, time_bucket, aggregation_function_list
):
    """Rescales a dataframes time periodicity using aggregation functions.

    Args:
        dataframe (pandas.Dataframe): A dataframe containing a column of time values to be rescaled
        time_column (string): Name of target time column
        time_bucket (DateOffset, Timedelta or str): Some time bucketing rule to lump the time in to. ex. 'M', 'A', '2H'
        aggregation_function_list (List[strings]): List of aggregation functions to apply to the data. ex. ['sum'] or ['sum', 'min', 'max']

    Returns:
        _type_: _description_
    """
    return scale_time(
        dataframe=dataframe,
        time_column=time_column,
        time_bucket=time_bucket,
        aggregation_function_list=aggregation_function_list,
    )


def regrid_dataframe_geo(dataframe, geo_columns, time_column, scale_multi, scale=None):
    """Regrids a dataframe with detectable geo-resolution

    Args:
        dataframe (_type_): _description_
        geo_columns (_type_): _description_
        scale_multi (_type_): _description_
    """

    return regrid_dataframe(
        dataframe=dataframe,
        geo_columns=geo_columns,
        time_column=time_column,
        scale_multi=scale_multi,
        scale=scale,
    )


def get_boundary_box(dataframe, geo_columns):
    """Returns the minimum and maximum x,y coordinates for a geographical dataset with latitude and longitude.

    Args:
        dataframe (pandas.Dataframe): Pandas dataframe with latitude and longitude
        geo_columns (List[string]): A list of the two column names that represent latitude and longitude.

    Returns:
        Dict: An object containing key value pairs for xmin, xmax,  ymin, and ymax.
    """

    return calculate_boundary_box(dataframe=dataframe, geo_columns=geo_columns)


def get_temporal_boundary(dataframe, time_column):
    return calculate_temporal_boundary(dataframe=dataframe, time_column=time_column)


# File processing functions


def raster2df(
    InRaster: str,
    feature_name: str = "feature",
    band: int = 0,
    nodataval: int = -9999,
    date: str = None,
    band_name: str = "feature2",
    bands: dict = None,
    band_type: str = "category",
):
    return raster2df_processor(
        InRaster, feature_name, band, nodataval, date, band_name, bands, band_type
    )


def netcdf2df(netcdf):
    return netcdf2df_processor(netcdf)


# CLI Utilities


class mixdata:
    def load_gadm2(self):
        cdir = os.path.expanduser("~")
        download_data_folder = f"{cdir}/elwood_data"

        # Admin 0 - 2
        gadm_fn = f"gadm36_2.feather"
        gadmDir = f"{download_data_folder}/{gadm_fn}"
        gadm = gf.from_geofeather(gadmDir)
        gadm["country"] = gadm["NAME_0"]
        # gadm["state"] = gadm["NAME_1"]
        gadm["admin1"] = gadm["NAME_1"]
        gadm["admin2"] = gadm["NAME_2"]
        gadm0 = gadm[["geometry", "country"]]
        # gadm1 = gadm[["geometry", "country", "state", "admin1"]]
        # gadm2 = gadm[["geometry", "country", "state", "admin1", "admin2"]]
        gadm1 = gadm[["geometry", "country", "admin1"]]
        gadm2 = gadm[["geometry", "country", "admin1", "admin2"]]

        self.gadm0 = gadm0
        self.gadm1 = gadm1
        self.gadm2 = gadm2

    def load_gadm3(self):
        # Admin 3
        cdir = os.path.expanduser("~")
        download_data_folder = f"{cdir}/elwood_data"
        gadm_fn = f"gadm36_3.feather"
        gadmDir = f"{download_data_folder}/{gadm_fn}"
        gadm3 = gf.from_geofeather(gadmDir)
        gadm3["country"] = gadm3["NAME_0"]
        # gadm3["state"] = gadm3["NAME_1"]
        gadm3["admin1"] = gadm3["NAME_1"]
        gadm3["admin2"] = gadm3["NAME_2"]
        gadm3["admin3"] = gadm3["NAME_3"]
        # gadm3 = gadm3[["geometry", "country", "state", "admin1", "admin2", "admin3"]]
        gadm3 = gadm3[["geometry", "country", "admin1", "admin2", "admin3"]]
        self.gadm3 = gadm3


def optimize_df_types(df: pd.DataFrame):
    """
    Pandas will upcast essentially everything. This will use the built-in
    Pandas function to_numeeric to downcast dataframe series to types that use
    less memory e.g. float64 to float32.

    For very large dataframes the memory reduction should translate into
    increased efficieny.
    """
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")

    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")

    return df
