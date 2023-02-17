# Elwood
An open source dataset transformation, standardization, and normalization python library.

# Usage

To use start using Elwood, simply run:

`pip install elwood`

Now you are able to use any of the dataset transformation, standardization, or normalization functions exposed through this library. To start, simply include `from elwood import elwood` in your python file. 

## Standardization
`elwood.process(args)`

#TODO STUB

## Transformation

The transformation functions include geographical extent clipping (latitude/longitude), geographical regridding (gridded data such as NetCDF or GeoTIFF), temporal clipping, and temporal scaling. 

### Geospatial Clipping

`elwood.clip_geo(dataframe, geo_columns, polygons_list)`

This function takes a pandas dataframe, a list of the column names for latitude and longitude, ex: `["lat", "lng"]`, and a list containing lists of objects representing the polygons to clip the data to. ex: 
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
