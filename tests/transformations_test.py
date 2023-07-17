"""Tests for Transformation features."""

import unittest
import warnings
import subprocess
import logging
import json
import gc
import os

from pathlib import Path
from os.path import join as path_join

# from elwood import elwood
from pandas.util.testing import assert_frame_equal, assert_dict_equal, assert_series_equal
import pandas as pd

from elwood.transformations.clipping import clip_dataframe, construct_multipolygon
from elwood.transformations.geo_utils import calculate_boundary_box
from elwood.transformations.regridding import regrid_dataframe
from elwood.transformations.scaling import scale_time

# Unused, but MultiPolygon is what our util fn returns when clipping
# from shapely import Polygon, MultiPolygon


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def input_path(filename):
    return path_join(get_project_root(), "tests", "inputs", filename)

def output_path(filename):
    return path_join(get_project_root(), "tests", "outputs", filename)


"""
Assuming within container or with virtualenv. Run with:
```
$ python -m pytest tests/transformations_test.py
```

Install pytest if necessary.
"""


class TestGeoUtils(unittest.TestCase):
    """Unit tests for `elwood` geo utils."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        gc.collect()

    def test_calculate_boundary_box__default(self):
        """
        """

        geo_columns = {'lat_column': 'latitude', 'lon_column': 'longitude'}

        input_df = pd.read_csv(input_path("calculate_boundary_input.csv"))

        actual = calculate_boundary_box(input_df, geo_columns)

        expected = {'xmin': 42.0864, 'xmax': 43.2906, 'ymin': 10.5619, 'ymax': 12.595}

        assert actual == expected


class TestClipping(unittest.TestCase):
    """Unit tests for `elwood` clipping transformation fn."""

    def setUp(self):
        """Set up test fixtures, if any."""

        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        gc.collect()
        pass

    # @pytest.mark.filterwarnings("ignore: deprecated")
    def test_clip_dataframe__default(self):
        """"""
        geo_columns = {'lat_column': 'latitude', 'lon_column': 'longitude'}

        polygons_list = [
            [
                {
                    "lat": -34.52466147177172,
                    "lng": -19.892688974499873
                },
                {
                    "lat": 40.78054143186033,
                    "lng": -19.892688974499873
                },
                {
                    "lat": 40.78054143186033,
                    "lng": 53.13263437520608
                },
                {
                    "lat": -34.52466147177172,
                    "lng": 53.13263437520608
                }
            ]
        ]

        mask = construct_multipolygon(polygons_list)

        # Input df with 100 rows
        input_df = pd.read_csv(input_path("clip_dataframe_input.csv"))

        # Clipping with mask above- should clip to 10 rows
        actual = clip_dataframe(input_df, geo_columns, mask)
        # round so that any values aren't affected by weird runtime/float issues
        # reset_index so that, instead of using original row number, we reset back to start to 0
        actual = actual.round(5).reset_index()

        # Loading the expected index uses numbers starting from 0,
        # hence why we reset the index for actual above.
        expected = pd.read_csv(output_path("clip_dataframe_output.csv"))
        expected = expected.round(5)

        # compare the cols we care for, since asserting for full df causes
        # problems with the auto-generated geometry/points column
        assert_series_equal(actual['latitude'], expected['latitude'])
        assert_series_equal(actual['longitude'], expected['longitude'])
        assert_series_equal(actual['value'], expected['value'])
        assert_series_equal(actual['color_hue'], expected['color_hue'])


class TestRegridding(unittest.TestCase):
    """Unit tests for `elwood` regridding transformation fn."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        gc.collect()

    def test_regrid_dataframe__default(self):
        """"""

        # TODO
        # input_path("filename")
        # regrid_dataframe()
        assert 3 == 3


class TestScaling(unittest.TestCase):
    """Unit tests for `elwood` scaling transformation fn."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        gc.collect()

    def test_scale_time__default(self):
        """"""

        time_column = "event_date"
        time_bucket = "Y"
        aggregation_functions = ["std"]
        geo_columns = {'lat_column': 'latitude', 'lon_column': 'longitude'}
        input_df = pd.read_csv(input_path("scale_time_input.csv"))

        actual = scale_time(input_df, time_column, time_bucket, aggregation_functions, geo_columns)

        expected = pd.read_csv(output_path("scale_time_output.csv"))

        actual = actual.reindex(columns=sorted(actual.columns))
        expected = expected.reindex(columns=sorted(expected.columns))

        expected['event_date'] = pd.to_datetime(expected['event_date'])

        print(f"actual:\n{actual}")
        print(f"expected:\n{expected}")

        for col in ["event_date", "fatalities", "latitude", "longitude"]:
            assert_series_equal(actual[col], expected[col])

