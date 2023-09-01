[![codecov](https://codecov.io/github/jataware/elwood/graph/badge.svg?token=5F2QK26ZBA)](https://codecov.io/github/jataware/elwood)

# Elwood
https://jataware.github.io/elwood/

An open source dataset transformation, standardization, and normalization python library.

# Requirements

This package requires the GDAL dev binaries to be installed. The version requirements are below:

```
gdal = 3.3.2
```

# Usage

Refer to the docs for more complete usage instructions: 

- https://jataware.github.io/elwood/installation.html
- https://jataware.github.io/elwood/usage.html
- https://jataware.github.io/elwood/overview.html

To use start using Elwood, simply run:

`pip install elwood`

Now you are able to use any of the dataset transformation, standardization, or normalization functions exposed through this library. To start, simply include `from elwood import elwood` in your python file. 

## Standardization
`elwood.process(args)`

Given an arbitrary dataset containing geospatial data (with columns and rows) with arbitrary non-standard format, and given some annotations/dictionary about the dataset, Elwood can standardize. Standardization means creating an output dataset with stable and predictable columns. The data can be normalized, regridded, scaled, and resolved using GADM to standard country names, as well as resolve the latitude,longitude of the event/measurement. A usual standard output will contain the following columns: `timestamp`, `country`, `admin1`, `admin2`, `admin3`, `lat`, `lng`, alongside other measurements/events/features of interest (additional columns to the right of the standard ones) contained within the input dataset.

## Transformation

The transformation functions include geographical extent clipping (latitude/longitude), geographical regridding (gridded data such as NetCDF or GeoTIFF), temporal clipping, and temporal scaling. 

### Geospatial Clipping

`elwood.clip_geo(dataframe, geo_columns, polygons_list)`

This function takes a pandas dataframe, a geo_columns dict of the column names for latitude and longitude, ex:
`{'lat_column': 'latitude', 'lon_column': 'longitude'}`, and a list containing lists of objects representing the polygons to clip the data to. ex: 
```
[
     [
        {
            "lat": 11.0,
            "lng": 42.0
        },
        {
            "lat": 11.0,
            "lng": 43.0
        },
        {
            "lat": 12.0,
            "lng": 43.0
        },
        {
            "lat": 12.0,
            "lng": 42.0
        }
    ],
    ...
]
```
### Geospatial regridding

`elwood.regrid_dataframe_geo(dataframe, geo_columns, scale_multi)`

This function takes a dataframe and regrids it's geography by some scale multiplier that is provided. This multiplier will be used to divide the current geographical scale in order to make a more coarse grained resolution dataset. The dataframe must have a detectable geographical scale, meaning each lat/lon represents a point in the middle of a gridded cell for the data provided. Lat and lon and determined by the geo_columns passed in: a dict of the column names ex: `{'lat_column': 'my_latitude', 'lon_column': 'my_longitude'}`

### Temporal Clipping
`elwood.clip_dataframe_time(dataframe, time_column, time_ranges)`

This function will produce a dataframe that only includes rows with `time_column` values contained within `time_ranges`. The time_ranges argument is a list of objects containing a start and end time. ex: `[{"start": datetime, "end": datetime}, ...]`

### Temporal Scaling
`elwood.rescale_dataframe_time(dataframe, time_column, time_bucket, aggregation_function_list)`

This function will produce a dataframe who's rows are the aggregated data based on some time bucket and some aggregation function list provided. The `time_column` is the name of the column containing targeted time values for rescaling. The `time_bucket` is some DateOffset, Timedelta or str representing the desired time granularity, ex. `'M', 'A', '2H'`. The `aggregation_function_list` is a list of aggregation functions to apply to the data.  ex. `['sum']` or `['sum', 'min', 'max']`

## 0 to 1 Normalization

`elwood.normalize_features(dataframe, output_file)`

This function expects a dataframe with a "feature" column and a "value" column, or long data. Each entry for a feature has its own feature/value row.
This function returns a dataframe in which all numerical values under the "value" column for each "feature" have been 0 to 1 scaled.
Optionally you may specify an `output_file` name to generate a parquet file of the dataframe.


# Testing

The easiest way to test Elwood is to build the docker container and run the unit tests inside of it. This method makes sure that all the libraries and system dependencies are installed. To start the container use `docker compose up`. Next, run:
```
$ python3 -m pip install pytest; pytest -vs
``` 
This command installs the pytest library on the container and runs all the of the available unit tests.

There available categories of unit tests are:
* Standardization tests: these tests are unit tests for different variations of the standardization flow, accessed from the entrypoint `elwood.process`
* Transformation tests: these tests represent unit tests for each of the different data transformations available in elwood.
