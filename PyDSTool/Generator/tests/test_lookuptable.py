#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy import array
import pytest

from PyDSTool import Point
from PyDSTool.Generator import LookupTable


def test_can_build_lookup_table_and_use_it_for_known_values():
    """Functional (a.k.a acceptance) test for LookupTable"""

    # John prepares data to be looked up
    ts = array([0.1, 1.1, 2.1])
    x1 = array([10.2, -1.4, 4.1])
    x2 = array([0.1, 0.01, 0.4])

    # John calculates "trajectory" for his data
    table = LookupTable({
        'name': 'lookup',
        'tdata': ts,
        'ics': dict(zip(['x1', 'x2'], [x1, x2])),
        })

    traj = table.compute('ltable')

    # Now John can retrieve his values from table
    for i, t in enumerate(ts):
        assert traj(t) == Point({'coordnames': ['x1', 'x2'], 'coordarray': [x1[i], x2[i]]})
        assert traj(t, 'x1') == Point({'x1': x1[i]})
        assert traj(t, 'x2') == Point({'x2': x2[i]})

    # John can get only those values, that he has previously inserted
    with pytest.raises(ValueError):
        traj(0.4)
    with pytest.raises(ValueError):
        traj(0.4, 'x1')
    with pytest.raises(ValueError):
        traj(0.4, 'x2')
