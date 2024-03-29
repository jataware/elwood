---
layout: default
title: Overview
nav_order: 4
has_toc: true
---

# Elwood Functions

- [Standardization](#standardization-processor)
- [Transformation](#transformation-functions)
- [File Processing](#file-processing-functions)

## Main Module

The main module of Elwood provides functionality for data transformation and standardization. It includes functions for reading and processing data, as well as performing various transformations.

## Standardization processor

### `process` Function

The `process` function in the Elwood library is responsible for data transformation and standardization based on a JSON schema. It takes various parameters and performs a series of steps to transform input data and produce standardized output. Below is an overview of the function's parameters and its workflow:

### Parameters

- `fp` (str): The file path to the input data file that needs to be processed.
- `mp` (str): The filename for the JSON mapper from spacetag. This mapper defines the schema for the data transformation.
- `admin` (str): The administrative level for geocoding. If not provided, it may be inferred from the JSON schema.
- `output_file` (str): The base filename for the output parquet files.
- `write_output` (bool, default True): Specifies whether to write output files.
- `gadm` (gpd.GeoDataFrame, default None): Optional specification of a GeoDataFrame of GADM shapes for geocoding.
- `overrides` (dict, default None): User-provided overrides for data transformation.

### Workflow

1. Read the User defined values from the specified `mp` (mapper) file.
5. Determine the file type from the transformation details.
6. Process the input data file based on the determined file type and transformation metadata.
8. Run the `standardizer` function to perform data standardization using the provided mapper, admin level, and GADM shapes.
10. Apply manual user overrides to hardcoded countries for GADM.
11. Optionally enforce feature column data types and separate string data from other types.
12. Write normalized data to parquet files with compression, if specified.
14. Return the normalized dataframe and a dictionary of renamed columns.

### Example Usage

```python
from elwood import elwood

norm_df, renamed_col_dict = elwood.process(
    fp="data.csv",
    mp="mapper.json",
    admin="admin2",
    output_file="output",
    gadm=gadm_data,
    overrides=overrides_data
)
```
## Transformation Functions

Elwood provides several transformation functions that operate on dataframes:

- `normalize_features`: Scales numerical features in a dataframe to a 0 to 1 scale. Takes in a dataframe and an optional output_file name. Outputs a normalized dataframe and optionally writes an output parquet file.
```python
output = elwood.normalize_features(
    dataframe=df, 
    output_file="data_parquet"
)
# `output` is the normalized dataframe result. Also outputs a parquet file called data_parquet_normalized.parquet.gzip
```
- `normalize_features_robust`: Performs robust normalization of numerical features. Takes in a dataframe and an optional output_file name. Outputs a normalized dataframe and optionally writes an output parquet file.
```python
output = elwood.normalize_features_robust(
    dataframe=df, 
    output_file="data_parquet"
)
# `output` is the normalized dataframe result. Also outputs a parquet file called data_parquet_normalized.parquet.gzip
```
- `clip_geo`: Clips data based on geographical shapes or shapefiles. The geo_columns input is a dict object containing the two column names for the lat/lon columns in the dataframe. The polygons_list is a list of lists containing lists of objects that represent polygon shapes to clip to. This polygons_list is ultimately a [geopandas clipping mask](https://geopandas.org/en/stable/docs/reference/api/geopandas.clip.html)
```python
clipped_frame = elwood.clip_geo(
    dataframe=df, 
    geo_columns={"lon_column": "longitude", "lat_column": "latitude"}, 
    polygons_list=[[10.0, 5.0], [20.0, 5.0], [20.0, 10.0], [10.0, 10.0]]
)
# Returns a dataframe with all columns removed outside of the specified shapes.
```
- `clip_dataframe_time`: Clips data based on time ranges. The time_column input is a string which is the name of target time column. The time_ranges input is a list of dictionary objects containing "start" and "end" datetime values
```python
clipped_frame = elwood.clip_dataframe_time(
    dataframe=df, time_column="Date", 
    time_ranges=[{"start":"01-01-2022", "end":"01-05-2022"}, {"start": "01-01-2023", "end": "12-01-2023"}]
)
# Returns a dataframe with all columns removed outside of the specified time ranges.
```
- `rescale_dataframe_time`: Rescales a dataframe's time periodicity using aggregation functions. The `dataframe` input is a `pandas.DataFrame` containing a column of time values to be rescaled. The `time_column` input is a string representing the name of the target time column. The `time_bucket` input is a `DateOffset`, `Timedelta`, or string representing a time bucketing rule (e.g., 'M', 'A', '2H') used to aggregate the time. The `aggregation_functions` input is a list of strings containing aggregation functions to apply to the data (e.g., ['sum']). The optional `geo_columns` input can be used for specifying geo columns.
```python
scaled_frame = elwood.rescale_dataframe_time(
    dataframe=df,
    time_column="Date",
    time_bucket='M',
    aggregation_functions=['sum'],
    geo_columns=None
)
# Returns a dataframe with rescaled time periodicity using specified aggregation functions.
```
`regrid_dataframe_geo`: Regrids a dataframe with detectable geo-resolution. 
- The `dataframe` input is of type `pandas.DataFrame` or of type `xarray.Dataset` and represents the dataframe to be regridded. 
- The `geo_columns` input is of type `Dict` and contains the columns representing geo information. 
- The `time_column` input is of type `String` and is the name of the target time column. 
- The `scale_multi` input is of type `Float` and represents the scaling factor for geo-resolution. The `aggregation_functions` input is of type `Dict[str, str]` and is a Dict of column names and aggregation functions to apply to that column. 
- The optional `scale` input, if provided, is of type `float` or `dict`. It represents a user override for initial scale of the data. If the desired latitude and longitude scales are different, pass a dict with `{"lat_scale":  <Desired Latitude Scale>, "lon_scale": <Desired Longitude Scale>}` The function will auto assess data scale if not provided. 
- The optional `xarray_return` input, if provided, expects a boolean value. This argument determines if the function should return the data in `xarray.Dataset` format. If `True`, the regridded data will be returned as an `xarray.Dataset` object.
```python
regridded_frame = regrid_dataframe_geo(
    dataframe=df,
    geo_columns=["latitude", "longitude"],
    time_column="Date",
    scale_multi=2,
    aggregation_functions={"Rainfall": "sum", "Temperature": "mean"}
)
# Returns a dataframe regridded with newly specified geo-resolution.
```
- `get_boundary_box`: Returns the minimum and maximum x, y coordinates for a geographical dataset with latitude and longitude.
```python
boundary_box = elwood.get_boundary_box(
    dataframe=df,
    geo_columns={"latitude_column": "Latitude", "longitude_column": "Longitude"}
)
# Returns an object containing xmin, xmax, ymin, and ymax coordinates.
```
- `get_temporal_boundary`: Returns the minimum and maximum time values in a dataframe. The return is a dictionary object with a 'min' key and a 'max' key.
```python
temporal_boundary = elwood.get_temporal_boundary(
    dataframe=df,
    time_column="Timestamp"
)
# Returns an object containing a 'min' key for the minimum time value, and a 'max' key for the maximum time value.
```


## File Processing Functions

Elwood includes functions to process raster and NetCDF files:

- `raster2df`: Converts a raster file to a dataframe. The `InRaster` input is a string representing the path to the input raster file. The optional `feature_name` input is a string that specifies the name of the feature column in the resulting dataframe. The optional `band` input is an integer representing the band number to extract from the raster. The optional `nodataval` input is an integer representing the nodata value in the raster. The optional `date` input is a string representing a date associated with the raster data. The optional `band_name` input is a string specifying the name of the band column in the resulting dataframe. The optional `bands` input is a dictionary specifying additional bands to extract with their corresponding names. The optional `band_type` input is a string indicating the type of data in the band column.
```python
df_result = elwood.raster2df(
    InRaster="path/to/raster/file.tif",
    feature_name="feature",
    band=0,
    nodataval=-9999,
    date="2023-08-25",
    band_name="feature2",
    bands={"band1": 1, "band2": 2},
    band_type="category"
)
# Returns a dataframe converted from the raster file.
```
- `netcdf2df`: Converts a NetCDF file to a dataframe. The `netcdf` input is a string representing the path to the NetCDF file.
```python
df_result = elwood.netcdf2df("path/to/netcdf/file.nc")
# Returns a dataframe converted from the NetCDF file.
```