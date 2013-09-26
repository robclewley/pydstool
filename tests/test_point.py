#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from numpy import array, float64, int32
from numpy.testing import assert_almost_equal, assert_warns

from PyDSTool.Points import Point
from PyDSTool.common import _all_numpy_int, _all_numpy_float


#
# Testing Point creating from coordict
#
def test_unnamed_dict():
    """Create Point from name/value dict as a positional argument"""
    p = Point({'x': 1.0, 'y': -1.0})
    assert_almost_equal(p.toarray(), array([1.0, -1.0], dtype=float64))
    assert p.coordnames == ['x', 'y']


def test_named_argument():
    """Create Point from name/value dict as a named argument"""
    p = Point(coorddict={'x': 1.0, 'y': -1.0})
    assert_almost_equal(p.toarray(), array([1.0, -1.0], dtype=float64))
    assert p.coordnames == ['x', 'y']


def test_named_dict():
    """Create Point from dict with 'coorddict' key"""
    p = Point({'coorddict': {'x': -1.0, 'y': 1.0}})
    assert_almost_equal(p.toarray(), array([-1.0, 1.0], dtype=float64))
    assert p.coordnames == ['x', 'y']


def test_dict_with_list_values():
    """Lists are truncated to first element"""
    p = Point({'x': [1.0], 'y': [-1.0, 2.0]})
    assert_almost_equal(p.toarray(), array([1.0, -1.0], dtype=float64))


def test_dict_with_array_values():
    """Arrays are truncated to first element"""
    p = Point({'x': array([1.0]), 'y': array([-1.0, 2.0])})
    assert_almost_equal(p.toarray(), array([1.0, -1.0], dtype=float64))


def test_dict_empty_list():
    """Empty lists are forbidden"""
    with pytest.raises(ValueError):
        Point({'x': [], 'y': 0.0})


def test_dict_empty_array():
    """Empty arrays are forbidden"""
    with pytest.raises(ValueError):
        Point({'x': array([]), 'y': 0.0})


def test_warn_truncation():
    """Issue warning when provided value is truncated"""
    assert_warns(UserWarning, Point, {'coorddict': {'x': [0.0, 1.0]}})


def test_nonchar_coordnames():
    """Non-char coordnames are converted to strings"""
    p = Point({'coorddict': {11: 1.0, 100: -1.0}})
    assert_almost_equal(p.toarray(), array([-1.0, 1.0], dtype=float64))
    assert p.coordnames == ['100', '11']


#
# Test Point creating from coordarray
#
def test_coordarray_empty_list():
    """Empty lists are forbidden"""
    with pytest.raises(ValueError):
        Point({'coordarray': []})


def test_coordarray_empty_array():
    """Empty arrays are forbidden"""
    with pytest.raises(ValueError):
        Point({'coordarray': array([])})


#
# Test type checking
#
def test_type_checking_scalar():
    """Using 'float' value for 'int' Point is forbidden"""
    with pytest.raises(AssertionError):
        Point({'coorddict': {'x': 0.0}, 'coordtype': int})


def test_type_checking_list_for_coord():
    """Using list of 'float' values for 'int' Point is forbidden"""
    with pytest.raises(AssertionError):
        Point({'coorddict': {'x': [0.0]}, 'coordtype': int})


def test_type_checking_list_for_coordarray():
    """Build 'int' Point from list of 'float' values is forbidden"""
    with pytest.raises(AssertionError):
        Point({'coordarray': [0.0, 1.0, 2.0], 'coordtype': int})


def test_type_checking_array_for_coordarray():
    """Build 'int' Point from 'float' array is forbidden"""
    with pytest.raises(AssertionError):
        Point({'coordarray': array([0.0, 1.0, 2.0], float), 'coordtype': int})


def test_type_checking_exotic_values():
    """Exotic values are caught by numpy"""
    with pytest.raises(TypeError):
        Point({'x': 'x', 'y': 'y'})


def test_type_checking_int_to_float_coersion():
    """It's possible to make 'float' Point from 'int' values"""

    src = [
        [0.0, 1, 2],        # list with 'int' values
        array([0, 1, 2])    # array of 'int' type
    ]

    for s in src:
        p = Point(coordarray=s, coordtype=float)
        assert_almost_equal(p.toarray(), array([0.0, 1.0, 2.0], dtype=float64))


#
# Test Point coord type
#
def _check_point(p, expected_type):
    assert p.toarray().dtype.type == p.coordtype
    assert expected_type == p.coordtype


def test_dict_default():
    """Use 'float64' by default for Point from dict"""
    _check_point(
        Point({'x': 1.0, 'y': -1.0}),
        float64
    )


def test_dict_explicit_float():
    """Convert 'float' to 'float64'"""
    _check_point(
        Point(coorddict={'x': 1.0, 'y': -1.0}, coordtype=float),
        float64
    )


def test_dict_explicit_int():
    """Convert 'int' to 'int32'"""
    _check_point(
        Point(coorddict={'x': 1, 'y': -1}, coordtype=int),
        int32
    )


def test_list_default():
    """Use 'float64' by default when 'coordtype' is omitted"""
    _check_point(Point(coordarray=[0.0, 1.0, 2.0]), float64)


def test_list_explicit_float():
    """Convert 'float' to 'float64'"""
    _check_point(
        Point(coordarray=[0.0, 1.0, 2.0], coordtype=float),
        float64
    )


def test_list_explicit_int():
    """Convert 'int' to 'int32'"""
    _check_point(
        Point(coordarray=[0, 1, 2], coordtype=int),
        int32
    )


def test_array_float():
    """All float arrays are coerced to float64"""

    for ftype in _all_numpy_float:
        _check_point(
            Point(coordarray=array([0.0, 1.0, 2.0], dtype=ftype)),
            float64
        )


def test_array_int():
    """All int arrays are coerced to int32"""

    for itype in _all_numpy_int:
        _check_point(
            Point(coordarray=array([0, 1, 2]), dtype=itype),
            int32
        )


#
# Test Point sanity checks
#
def test_duplicated_coordnames():
    """Duplicated coord names are forbidden"""
    with pytest.raises(AssertionError):
        Point({'coordarray': [0.0, 1.0], 'coordnames': 'xx'})


def test_non_matching_lengths():
    """Length of values sequence must be equal to the length of names sequence"""
    with pytest.raises(ValueError):
        Point({'coordarray': [0.0, 1.0], 'coordnames': 'xyz'})


def test_wrong_rank():
    """Values must be scalar or one-dimension array"""
    with pytest.raises(ValueError):
        Point({'coordarray': [[0.0, 1.0], [2.0, 3.0]], 'coordnames': 'xyz'})


def test_missing_coord_data():
    """Coord values are required argument"""
    with pytest.raises(ValueError):
        Point({'coordnames': 'xyz'})
