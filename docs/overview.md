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

- `normalize_features`: Scales numerical features in a dataframe to a 0 to 1 scale.
- `normalize_features_robust`: Performs robust normalization of numerical features.
- `clip_geo`: Clips data based on geographical shapes or shapefiles.
- `clip_dataframe_time`: Clips data based on time ranges.
- `rescale_dataframe_time`: Rescales a dataframe's time periodicity using aggregation functions.
- `regrid_dataframe_geo`: Regrids a dataframe with detectable geo-resolution.
- `get_boundary_box`: Returns the boundary box of a geographical dataset.
- `get_temporal_boundary`: Returns the temporal boundary of a dataset.


## File Processing Functions

Elwood includes functions to process raster and NetCDF files:

- `raster2df`: Converts a raster file to a dataframe.
- `netcdf2df`: Converts a NetCDF file to a dataframe.