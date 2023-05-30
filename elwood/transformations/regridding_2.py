import xarray as xr
from matplotlib import pyplot as plt

from cdo import * 
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"

cdo = Cdo()
cdo.debug = False

from enum import Enum




class RegridMethod(Enum):
    CONSERVATIVE = ('remapcon', cdo.remapcon, 'Conservative remapping, suitable for preserving the total value of the field (e.g., mass, population, water fluxes)')
    SUM = ('remapsum', cdo.remapsum, 'Sum remapping, suitable for fields representing accumulated quantities (e.g., precipitation, snowfall, radiation fluxes)')
    MINIMUM = ('remapmin', cdo.remapmin, 'Minimum remapping, suitable for fields where you want to preserve the minimum value within an area (e.g., minimum temperature, lowest pressure)')
    MAXIMUM = ('remapmax', cdo.remapmax, 'Maximum remapping, suitable for fields where you want to preserve the maximum value within an area (e.g., peak wind speeds, maximum temperature)')
    MEDIAN = ('remapmedian', cdo.remapmedian, 'Median remapping, suitable for fields where you want to preserve the central tendency of the data, while being less sensitive to extreme values (e.g., median income, median precipitation)')
    AVERAGE = ('remapavg', cdo.remapavg, 'Average remapping, suitable for fields representing average quantities (e.g., temperature, humidity, wind speed)')
    BILINEAR = ('remapbil', cdo.remapbil, 'Bilinear interpolation, suitable for smooth fields (e.g., temperature, pressure, geopotential height)')
    BICUBIC = ('remapbic', cdo.remapbic, 'Bicubic interpolation, suitable for smooth fields with higher-order accuracy (e.g., temperature, pressure, geopotential height)')
    NEAREST_NEIGHBOR = ('remapnn', cdo.remapnn, 'Nearest neighbor remapping, suitable for categorical data (e.g., land use types, soil types, vegetation types)')

    def __init__(self, method_name, cdo_function, description):
        self.method_name = method_name
        self.cdo_function = cdo_function
        self.description = description


def regrid(data: xr.Dataset, resolution: float, method: RegridMethod) -> xr.Dataset:
    """
    Regrids the data to the target resolution using the specified aggregation method.
    """
    data.to_netcdf('tmp_data.nc')
    create_target_grid(resolution) # creates tmp_gridfile.txt

    regridded_data = method.cdo_function('tmp_gridfile.txt', input='tmp_data.nc', options='-f nc', returnXDataset=True)

    #clip the regridded data to the maximum extent of the original data
    regridded_data = regridded_data.rio.write_crs(4326)
    regridded_data = regridded_data.rio.clip_box(*data.rio.bounds())

    # Clean up temporary files
    os.remove('tmp_data.nc')
    os.remove('tmp_gridfile.txt')

    return regridded_data


def create_target_grid(resolution: float) -> None:
    """
    Creates a target grid with the specified resolution, and saves to tmp_gridfile.txt
    """

    # create a grid file
    content = f"""
gridtype  = latlon
xsize     = {int(360/resolution)}
ysize     = {int(180/resolution)}
xfirst    = {-180 + resolution / 2}
xinc      = {resolution}
yfirst    = {-90 + resolution / 2}
yinc      = {resolution}
"""
    gridfile = 'tmp_gridfile.txt'
    with open(gridfile, 'w') as f:
        f.write(content)



def test1():
    import geopandas as gpd

    # Load population data
    gpw = xr.open_dataset('gpw_v4_2pt5_min.nc', decode_coords='all')
    gpw = gpw.rio.write_crs(4326)

    # Load shapefile and select example countries
    shapefile = 'gadm_0/gadm36_0.shp'
    sf = gpd.read_file(shapefile)
    countries = ['Ethiopia', 'South Sudan', 'Somalia', 'Kenya']
    countries_shp = sf[sf['NAME_0'].isin(countries)]

    # Clip population data to the example countries
    gpw_c = gpw.rio.clip(countries_shp.geometry)


    # Regrid the clipped population data using the remapsum method
    regridded_data = regrid(gpw_c, 1.0, RegridMethod.CONSERVATIVE)
    
    
    # save the variable that has the data for plotting
    var_name = list(gpw.data_vars.keys())[0]

    # plot the original and regridded data
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    gpw_c.isel(raster=0)[var_name].plot(ax=ax1, robust=True)
    regridded_data.isel(raster=0)[var_name].plot(ax=ax2, robust=True)
    plt.show()






if __name__ == '__main__':
    test1()
