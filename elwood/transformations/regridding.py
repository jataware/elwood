import math
import xarray
import numpy


def regrid_dataframe(dataframe, geo_columns, scale_multi, scale=None):
    """Uses xarray interpolation to regrid geography in a dataframe.

    Args:
        dataframe (pandas.Dataframe): Dataframe of a dataset that has detectable gridden geographical resolution ie. points that represent 1sqkm areas
        geo_columns (List[str]): The geo_columns for the latitude and longitude pairs.
        scale_multi (int): The number by which to divide to geographical scale to regrid larger.

    Returns:
        pandas.Dataframe: Dataframe with geographical extend regridded to
    """

    ds = xarray.Dataset.from_dataframe(dataframe)

    geo_dim_0 = geo_columns[0] + "_dim"
    geo_dim_1 = geo_columns[1] + "_dim"

    coord_dict = {
        geo_columns[0]: (geo_dim_0, ds[geo_columns[0]].data),
        geo_columns[1]: (geo_dim_1, ds[geo_columns[1]].data),
    }

    ds = ds.assign_coords(**coord_dict)

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
        round(ds.dims[geo_dim_0] * multiplier),
    )
    new_1 = numpy.linspace(
        ds[geo_columns[1]][0],
        ds[geo_columns[1]][-1],
        round(ds.dims[geo_dim_1] * multiplier),
    )

    interpolation = {geo_dim_0: new_0, geo_dim_1: new_1}

    ds2 = ds.interp(**interpolation)
    print(ds2)

    p_dataframe = ds2.to_dataframe()

    return p_dataframe


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
