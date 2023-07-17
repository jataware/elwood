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
# from pandas.util.testing import assert_frame_equal, assert_dict_equal
# import pandas as pd

from elwood.transformations.clipping import clip_dataframe
from elwood.transformations.geo_utils import calculate_boundary_box
from elwood.transformations.regridding import regrid_dataframe
from elwood.transformations.scaling import scale_time

logger = logging.getLogger(__name__)


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

    def test_clip_dataframe__default(self):
        """"""

        # TODO
        # input_path("filename")
        # clip_dataframe()
        assert 2 == 2



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

