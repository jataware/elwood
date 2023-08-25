---
layout: default
title: Usage
nav_order: 3
has_toc: true
---

# Usage

The Elwood Library provides a main entrypoint object for its functionality. This entrypoint exposes both standardization and transformation functions of the library.
```python
from elwood import elwood
```

For example, consider the following function which clips the extent of the data in a dataframe using the `polygons_list` argument to specify the extent for the provided `geo_columns` argument:

```python
from elwood import elwood

clipped_df = elwood.clip_geo(
    dataframe=original_dataframe,
    geo_columns=geo_columns,
    polygons_list=shape_list
)
```
This function, and all others, are exposed via the top-level elwood object.

Each function provided by Elwood, along with their respective arguments, is detailed in the documentation. Every Elwood function also includes a docstring that describes the expected arguments and functionality.