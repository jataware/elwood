import geopandas


def calculate_boundary_box(dataframe, geo_columns):
    """
    """
    x_geo = dataframe[geo_columns['lon_column']]
    y_geo = dataframe[geo_columns['lat_column']]
    geo_dataframe = geopandas.GeoDataFrame(
        dataframe, geometry=geopandas.points_from_xy(x_geo, y_geo)
    )

    xmin, ymin, xmax, ymax = geo_dataframe.total_bounds

    boundary_dict = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    return boundary_dict
