import math
import xarray
import numpy


def regrid_dataframe(dataframe, geo_columns, time_column, scale_multi, scale=None):
    """Uses xarray interpolation to regrid geography in a dataframe.

    Args:
        dataframe (pandas.Dataframe): Dataframe of a dataset that has detectable gridden geographical resolution ie. points that represent 1sqkm areas
        geo_columns (List[str]): A list containing the geo_columns for the latitude and longitude pairs.
        time_column (List[str]): A list containing the name of the datetime column(s) in the dataset.
        scale_multi (int): The number by which to divide to geographical scale to regrid larger.

    Returns:
        pandas.Dataframe: Dataframe with geographical extend regridded to
    """

    geo_columns.extend(time_column)
    ds = None

    try:

        ds = xarray.Dataset.from_dataframe(dataframe.set_index(geo_columns))

    except KeyError as error:
        print(error)

    ds_scale = 0
    if scale:
        ds_scale = scale
    else:
        ds_scale = getScale(
            ds[geo_columns[0]][0],
            ds[geo_columns[1]][0],
            ds[geo_columns[0]][1],
            ds[geo_columns[1]][1],
        )

    multiplier = ds_scale / scale_multi

    new_0 = numpy.linspace(
        ds[geo_columns[0]][0],
        ds[geo_columns[0]][-1],
        round(ds.dims[geo_columns[0]] * multiplier),
    )
    new_1 = numpy.linspace(
        ds[geo_columns[1]][0],
        ds[geo_columns[1]][-1],
        round(ds.dims[geo_columns[1]] * multiplier),
    )

    interpolation = {geo_columns[0]: new_0, geo_columns[1]: new_1}

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
