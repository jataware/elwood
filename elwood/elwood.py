"""Main module."""
import json
import click
import os
import sys

import geofeather as gf
import numpy as np
import pandas as pd

from . import constants
from .feature_normalization import robust_normalization, zero_to_one_normalization
from .file_processor import (
    netcdf2df_processor,
    process_file_by_filetype,
    raster2df_processor,
)
from .standardizer import standardizer
from .transformations.clipping import clip_dataframe, clip_time, construct_multipolygon
from .transformations.geo_utils import calculate_boundary_box
from .transformations.regridding import regrid_dataframe
from .transformations.scaling import scale_time
from .transformations.temporal_utils import calculate_temporal_boundary
from .utils import gadm_fuzzy_match

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


# Standardization processor


def dict_get(nested, path, fallback=None):
    """
    Receives a nested dictionary and a string describing a dictionary path,
    and returns the corresponding value.
    [Optional]`fallback` if path doesn't exists, defaults to None.
    `path`: str describing the nested dictionary deep path.
            Each key in the path is separated by a dot ('.').
            eg for {'a': {'b': {'c': 42}}}, `path` 'a.b.c' returns 42.
    """

    if not type(nested) == dict:
        return fallback

    keys = path.split('.')
    value = nested
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return fallback


def get_only_key(my_dict, path):
    """
    Given a dict and a path, returns the only key if my_dict:
    - is a dict
    - only contains one key

    else returns None

    """
    if not len(dict_get(my_dict, path, {})) == 1:
        return None

    return list(dict_get(my_dict, path).keys())[0]


def process(
    fp: str, mp: str, admin: str,
    output_file: str, write_output=True, gadm=None, overrides=None
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
    norm, renamed_col_dict = standardizer(df, mapper, admin, gadm=gadm)

    # Normalizer will add NaN for missing values, e.g. when appending
    # dataframes with different columns. GADM will return None when geocoding
    # but not finding the entity (e.g. admin3 for United States).
    # Replace None with NaN for consistency.
    norm.fillna(value=np.nan, inplace=True)

    # GADM Resolver - Apply manual user overrides to hardcoded countries for now.
    if dict_get(overrides, "gadm"):
        field_name = get_only_key(overrides, "gadm")
        field_overrides = dict_get(overrides, f"gadm.{field_name}")
        if field_overrides:
            click.echo(f"Applying GADM country overrides provided by user:\n{overrides}")
            norm["country"] = norm["country"].replace(field_overrides)

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
    df = zero_to_one_normalization(dataframe)

    if output_file:
        df.to_parquet(f"{output_file}_normalized.parquet.gzip", compression="gzip")
    return df


def normalize_features_robust(dataframe, output_file: str = None):
    df = robust_normalization(dataframe, method="min_max")

    if output_file:
        df.to_parquet(
            f"{output_file}_normalized_robust.parquet.gzip", compression="gzip"
        )
    return df


def clip_geo(dataframe, geo_columns, polygons_list):
    """Clips data based on geographical shape(s) or shapefile (NOT IMPLEMENTED).

    Args:
        dataframe (pandas.Dataframe): A pandas dataframe containing geographical data.
        geo_columns (Dict): A dict containing the two column names for the lat/lon columns in the dataframe.
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
    dataframe, time_column, time_bucket, aggregation_functions, geo_columns=None
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
        aggregation_functions=aggregation_functions,
        geo_columns=geo_columns,
    )


regrid_dataframe_geo = regrid_dataframe


def get_boundary_box(dataframe, geo_columns):
    """Returns the minimum and maximum x,y coordinates for a geographical dataset with latitude and longitude.

    Args:
        dataframe (pandas.Dataframe): Pandas dataframe with latitude and longitude
        geo_columns (Dict[str,str]): A dict with the two column names that represent latitude and longitude.

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
    Pandas function to_numeric to downcast dataframe series to types that use
    less memory e.g. float64 to float32.

    For very large dataframes the memory reduction should translate into
    increased efficieny.
    """
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")

    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")

    return df


def get_gadm_matches(dataframe, geo_column, admin_level):
    # Get geofeather object
    cdir = os.path.expanduser("~")
    download_data_folder = f"{cdir}/elwood_data"
    gadm_fn = f"gadm36_2.feather"
    gadmDir = f"{download_data_folder}/{gadm_fn}"
    gadm = gf.from_geofeather(gadmDir)

    gadm["country"] = gadm["NAME_0"]
    gadm["state"] = gadm["NAME_1"]
    gadm["admin1"] = gadm["NAME_1"]
    gadm["admin2"] = gadm["NAME_2"]

    # Run match
    matches_object = gadm_fuzzy_match(dataframe, geo_column, gadm, admin_level)

    # Add in dataframe column
    matches_object["field"] = geo_column

    print(matches_object)

    return matches_object


def gadm_list_all(admin_level):
    # Get geofeather object
    cdir = os.path.expanduser("~")
    download_data_folder = f"{cdir}/elwood_data"
    gadm_fn = f"gadm36_2.feather"
    gadmDir = f"{download_data_folder}/{gadm_fn}"
    gadm = gf.from_geofeather(gadmDir)

    gadm["country"] = gadm["NAME_0"]
    gadm["state"] = gadm["NAME_1"]
    gadm["admin1"] = gadm["NAME_1"]
    gadm["admin2"] = gadm["NAME_2"]

    return gadm[admin_level].unique()
