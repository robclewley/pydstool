#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy import array
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest

from PyDSTool import PyDSTool_BoundsError
from PyDSTool.Generator import (
    InterpolateTable,
)


def test_can_build_interpolate_table_and_use_it_for_unknown_values():
    """Functional (a.k.a acceptance) test for InterpolateTable"""

    # John prepares data to be interpolated
    ts = array([0.1, 1.1, 2.1])
    x1 = array([10.2, -1.4, 4.1])
    x2 = array([0.1, 0.01, 0.4])

    # John calculates "trajectory" for his data
    table = InterpolateTable({
        'name': 'interp',
        'tdata': ts,
        'ics': dict(zip(['x1', 'x2'], [x1, x2]))
    })
    traj = table.compute('itable')

    # John checks interpolation for prepared values
    for i, t in enumerate(ts):
        assert_array_almost_equal(traj(t), [x1[i], x2[i]])

    # John calculates values for interesting time values ...
    # ... for all dependent variables
    assert_array_almost_equal(traj(1.5), [0.8, 0.166])
    # ... or for concrete variables
    assert_almost_equal(traj(0.4, 'x1'), 6.72)
    assert_array_almost_equal(traj(1.4, ['x1', 'x2']), [0.25, 0.127])

    # John can extract interpolation for the concrete variable
    # ... from table
    x1 = table.variables['x1']
    assert_almost_equal(x1(0.75), traj(0.75, 'x1'))
    # ... or from trajectory
    x2 = traj.variables['x2']
    assert_almost_equal(x2(1.5), traj(1.5, 'x2'))

    # John can't calculate values outside of the domain for the independent variable
    with pytest.raises(PyDSTool_BoundsError):
        traj(0.0)
    with pytest.raises(PyDSTool_BoundsError):
        traj(2.2)
    with pytest.raises(PyDSTool_BoundsError):
        x1(0.0)
    with pytest.raises(PyDSTool_BoundsError):
        x2(2.2)
