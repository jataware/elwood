from __future__ import annotations
import xarray as xr

from cdo import *
import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

cdo = Cdo()
cdo.debug = False

from enum import Enum


def regridding_interface(dataframe, geo_columns, scale_multi, aggregation_functions):
    # Construct renaming mapping for regridding:
    renaming_map = {geo_columns[0]: "x", geo_columns[1]: "y"}

    # Pandas dataframe to Xarray
    dataframe.rename(renaming_map)
    xr_dataframe = dataframe.to_xarray()

    # Check if only one agg function was passed.
    aggregator = aggregation_functions
    if isinstance(aggregation_functions, list):
        aggregator = aggregation_functions[0]

        regridded_data = regrid(
            data=xr_dataframe,
            resolution=scale_multi,
            method=aggregation_value_mapping(aggregator),
        )

        return regridded_data.to_dataframe()

    else:
        # Get methods and convert to RegridMethod
        for key, value in aggregator.items():
            aggregator[key] = aggregation_value_mapping(value)

        regridded_data = multi_feature_regrid(
            data=xr_dataframe, resolution=scale_multi, methods=aggregator
        )

        return regridded_data.to_dataframe()


class RegridMethod(Enum):
    SUM = (
        "remapsum",
        cdo.remapsum,
        "Sum remapping, suitable for fields where the total quantity should be conserved (e.g., mass, population, water fluxes)",
    )
    MINIMUM = (
        "remapmin",
        cdo.remapmin,
        "Minimum remapping, suitable for fields where you want to preserve the minimum value within an area (e.g., minimum temperature, lowest pressure)",
    )
    MAXIMUM = (
        "remapmax",
        cdo.remapmax,
        "Maximum remapping, suitable for fields where you want to preserve the maximum value within an area (e.g., peak wind speeds, maximum temperature)",
    )
    MEDIAN = (
        "remapmedian",
        cdo.remapmedian,
        "Median remapping, suitable for fields where you want to preserve the central tendency of the data, while being less sensitive to extreme values (e.g., median income, median precipitation)",
    )
    AVERAGE = (
        "remapavg",
        cdo.remapavg,
        "Average remapping, suitable for fields representing average quantities (e.g., temperature, humidity, wind speed)",
    )
    BILINEAR = (
        "remapbil",
        cdo.remapbil,
        "Bilinear interpolation, suitable for smooth fields (e.g., temperature, pressure, geopotential height)",
    )
    BICUBIC = (
        "remapbic",
        cdo.remapbic,
        "Bicubic interpolation, suitable for smooth fields with higher-order accuracy (e.g., temperature, pressure, geopotential height)",
    )
    CONSERVATIVE = (
        "remapcon",
        cdo.remapcon,
        "First-order conservative remapping. See: https://journals.ametsoc.org/view/journals/mwre/127/9/1520-0493_1999_127_2204_fasocr_2.0.co_2.xml",
    )
    CONSERVATIVE2 = (
        "remapcon2",
        cdo.remapcon2,
        "Second-order conservative remapping. See: https://journals.ametsoc.org/view/journals/mwre/127/9/1520-0493_1999_127_2204_fasocr_2.0.co_2.xml",
    )
    NEAREST_NEIGHBOR = (
        "remapnn",
        cdo.remapnn,
        "Nearest neighbor remapping, suitable for categorical data (e.g., land use types, biome type, election area winners)",
    )

    def __init__(self, method_name, cdo_function, description):
        self.method_name = method_name
        self.cdo_function = cdo_function
        self.description = description

    def __str__(self):
        return f"<RegridMethod.{self.name}>"

    def __repr__(self):
        return f"<RegridMethod.{self.name}>"


from dataclasses import dataclass


@dataclass
class Resolution:
    dx: float
    dy: float = None

    def __init__(self, dx: float, dy: float | None = None):
        """
        dx (float): The target resolution in the x-direction (longitude)
        dy (float, optional): The target resolution in the y-direction (latitude). If None, then sets dy=dx
        """
        self.dx = dx
        self.dy = dy if dy is not None else dx


def regrid(
    data: xr.Dataset, resolution: float | Resolution, method: RegridMethod
) -> xr.Dataset:
    """
    Regrids the data to the target resolution using the specified aggregation method.
    """
    data.to_netcdf("tmp_data.nc")
    create_target_grid(resolution)  # creates tmp_gridfile.txt

    regridded_data = method.cdo_function(
        "tmp_gridfile.txt", input="tmp_data.nc", options="-f nc", returnXDataset=True
    )

    # clip the regridded data to the maximum extent of the original data
    regridded_data = regridded_data.rio.write_crs(4326)
    regridded_data = regridded_data.rio.clip_box(*data.rio.bounds())

    # Clean up temporary files
    os.remove("tmp_data.nc")
    os.remove("tmp_gridfile.txt")

    return regridded_data


def create_target_grid(resolution: float | Resolution) -> None:
    """
    Creates a target grid with the specified resolution, and saves to tmp_gridfile.txt
    """

    if not isinstance(resolution, Resolution):
        resolution = Resolution(resolution)

    # create a grid file
    content = f"""
gridtype  = latlon
xsize     = {int(360/resolution.dx)}
ysize     = {int(180/resolution.dy)}
xfirst    = {-180 + resolution.dx / 2}
xinc      = {resolution.dx}
yfirst    = {-90 + resolution.dy / 2}
yinc      = {resolution.dy}
"""
    gridfile = "tmp_gridfile.txt"
    with open(gridfile, "w") as f:
        f.write(content)


def multi_feature_regrid(
    data: xr.Dataset, resolution: float | Resolution, methods: dict[str, RegridMethod]
) -> xr.Dataset:
    """
    Regrids data with multiple features using specified aggregation methods per each feature.
    """

    # collect all features that use the same aggregation method
    features_by_method = {}
    for feature, method in methods.items():
        if method not in features_by_method:
            features_by_method[method] = []
        features_by_method[method].append(feature)

    # regrid each group of features using the specified aggregation method
    results = [
        regrid(data[features], resolution, method)
        for method, features in features_by_method.items()
    ]

    # merge the results and return
    return xr.merge(results)


def aggregation_value_mapping(value):
    if value == "median":
        return RegridMethod.MEDIAN
    if value == "sum":
        return RegridMethod.SUM
    if value == "mean":
        return RegridMethod.AVERAGE
    if value == "bilinear":
        return RegridMethod.BILINEAR
    if value == "bicubic":
        return RegridMethod.BICUBIC
    if value == "conservative":
        return RegridMethod.CONSERVATIVE
    if value == "min":
        return RegridMethod.MINIMUM
    if value == "max":
        return RegridMethod.MAXIMUM
    if value == "nearest_neighbor":
        return RegridMethod.NEAREST_NEIGHBOR
    if value == "conservative2":
        return RegridMethod.CONSERVATIVE2


def test1():
    import geopandas as gpd
    from matplotlib import pyplot as plt

    # Load population data
    gpw = xr.open_dataset("gpw_v4_2pt5_min.nc", decode_coords="all")
    gpw = gpw.rio.write_crs(4326)

    # Load shapefile and select example countries
    shapefile = "gadm_0/gadm36_0.shp"
    sf = gpd.read_file(shapefile)
    countries = ["Ethiopia", "South Sudan", "Somalia", "Kenya"]
    countries_shp = sf[sf["NAME_0"].isin(countries)]

    # Clip population data to the example countries
    gpw_c = gpw.rio.clip(countries_shp.geometry)

    # Regrid the clipped population data using the remapsum method
    regridded_data = regrid(gpw_c, 1.0, RegridMethod.SUM)

    # save the variable that has the data for plotting
    var_name = list(gpw.data_vars.keys())[0]

    # plot the original and regridded data
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    gpw_c.isel(raster=0)[var_name].plot(ax=ax1, robust=True)
    regridded_data.isel(raster=0)[var_name].plot(ax=ax2, robust=True)
    plt.show()


def test4():
    from matplotlib import pyplot as plt

    # testing multiple column regridding with different aggregation methods

    # load climate data
    data = xr.open_dataset(
        "MERRA2_400.inst3_3d_asm_Np.20220101.nc4", decode_coords="all"
    )

    # need to set the spatial dimensions to x and y
    data = data.rename({"lon": "x", "lat": "y"})

    # regrid the data with different aggregation methods per feature
    r_data = multi_feature_regrid(
        data,
        Resolution(4.0, 4.8),
        {
            "SLP": RegridMethod.AVERAGE,
            "T": RegridMethod.AVERAGE,
            "U": RegridMethod.MAXIMUM,
            "V": RegridMethod.MINIMUM,
        },
    )

    # plot original and regridded data 4x1 vertically
    var_names = [
        "sea_level_pressure",
        "air_temperature",
        "eastward_wind",
        "northward_wind",
    ]
    vars = ["SLP", "T", "U", "V"]

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        data.isel(time=0, lev=0)[vars[i]].plot(ax=ax, robust=True)
        ax.set_title(var_names[i])

    plt.subplots_adjust(hspace=0.5)

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        r_data.isel(time=0, lev=0)[vars[i]].plot(ax=ax, robust=True)
        ax.set_title(var_names[i])

    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    test1()
    # test4()
