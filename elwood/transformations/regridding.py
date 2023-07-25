import math
import xarray
import numpy
import pandas as pd
from typing import Dict, List


def regrid_dataframe(dataframe: pd.core.frame.DataFrame,
                     geo_columns: Dict[str, str],
                     time_column: List[str],
                     scale_multi: int,
                     scale=None) -> pd.core.frame.DataFrame:
    """Uses xarray interpolation to regrid geography in a dataframe.

    Args:
        dataframe (pandas.Dataframe): Dataframe of a dataset that has detectable gridden geographical resolution ie. points that represent 1sqkm areas
        geo_columns (Dict[str]): A dictionary containing the geo_columns for the latitude and longitude pairs, with keys 'lat_column' and 'lon_column'.
        time_column (List[str]): A list containing the name of the datetime column(s) in the dataset.
        scale_multi (int): The number by which to divide to geographical scale to regrid larger.

    Returns:
        pandas.Dataframe: Dataframe with geographical extend regridded to
    """

    geo_columns_list = [geo_columns['lat_column'], geo_columns['lon_column']]
    geo_columns_list.extend(time_column)
    ds = None

    try:

        ds = xarray.Dataset.from_dataframe(dataframe.set_index(geo_columns_list))

    except KeyError as error:
        print(error)

    ds_scale = 0
    if scale:
        ds_scale = scale
    else:
        ds_scale = getScale(
            ds[geo_columns['lat_column']][0],
            ds[geo_columns['lon_column']][0],
            ds[geo_columns['lat_column']][1],
            ds[geo_columns['lon_column']][1],
        )

    multiplier = ds_scale / scale_multi

    new_lat = numpy.linspace(
        ds[geo_columns['lat_column']][0],
        ds[geo_columns['lat_column']][-1],
        round(ds.dims[geo_columns['lat_column']] * multiplier),
    )
    new_lon = numpy.linspace(
        ds[geo_columns['lon_column']][0],
        ds[geo_columns['lon_column']][-1],
        round(ds.dims[geo_columns['lon_column']] * multiplier),
    )

    interpolation = {geo_columns['lat_column']: new_lat, geo_columns['lon_column']: new_lon}

    ds2 = ds.interp(**interpolation)

    final_dataframe = ds2.to_dataframe()
    final_dataframe.reset_index(inplace=True)

    return final_dataframe


def getScale(lat0, lon0, lat1, lon1):
    """
    Description
    -----------
    Return an estimation of the scale in km of a netcdf dataset.
    The estimate is based on the first two data points, and returns
    the scale distance at that lat/lon.

    """
    r = 6371  # Radius of the earth in km
    dLat = numpy.radians(lat1 - lat0)
    dLon = numpy.radians(lon1 - lon0)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(
        numpy.radians(lat0)
    ) * math.cos(numpy.radians(lat1)) * math.sin(dLon / 2) * math.sin(dLon / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = r * c
    # Distance in km
    return d
