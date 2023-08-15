---
layout: default
title: Tests
nav_order: 5
has_toc: true
---

# Testing Elwood

Tests are available in the `tests` directory. They can be run with:

```
cd tests
python3 -m unittest tests/test_standardization.py -v
python3 -m unittest tests/test_transformations.py -v
```

It is suggested to run these tests in the Elwood docker container so that all dependencies are installed correctly.

You may also use `pytest`. To run all tests, from the root directory, use:

```
pytest -vs
```

To run specific tests with pytest, use:

```
pytest -vs tests/test_transformations.py::TestRegridding::test_regrid_dataframe__default
```
