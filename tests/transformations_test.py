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

from shapely import Polygon, MultiPolygon


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

        # turn off warnings unless refactoring.
        # warnings.simplefilter("ignore")

    def tearDown(self):
        """Tear down test fixtures, if any."""
        gc.collect()
        # Delete any output parquet files.
        # try:
        #     os.remove(output_path("unittests.parquet.gzip"))
        # except FileNotFoundError as e:
        #     pass

        # try:
        #     os.remove(output_path("unittests_str.parquet.gzip"))
        # except FileNotFoundError as e:
        #     pass

    def test_calculate_boundary_box__default(self):
        """"""

        # TODO
        # input_path("filename")
        # calculate_boundary_box()
        assert 1 == 1


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

        # turn off warnings unless refactoring.
        # warnings.simplefilter("ignore")

    def tearDown(self):
        """Tear down test fixtures, if any."""
        gc.collect()
        # Delete any output parquet files.
        # try:
        #     os.remove(output_path("unittests.parquet.gzip"))
        # except FileNotFoundError as e:
        #     pass

        # try:
        #     os.remove(output_path("unittests_str.parquet.gzip"))
        # except FileNotFoundError as e:
        #     pass

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

        # turn off warnings unless refactoring.
        # warnings.simplefilter("ignore")

    def tearDown(self):
        """Tear down test fixtures, if any."""
        gc.collect()
        # Delete any output parquet files.
        # try:
        #     os.remove(output_path("unittests.parquet.gzip"))
        # except FileNotFoundError as e:
        #     pass

        # try:
        #     os.remove(output_path("unittests_str.parquet.gzip"))
        # except FileNotFoundError as e:
        #     pass

    def test_scale_time__default(self):
        """"""

        # TODO
        # input_path("filename")
        # scale_time()
        assert 4 == 4

