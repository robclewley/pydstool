#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from PyDSTool import (
    args,
    Point,
    PyDSTool_ValueError,
)

from PyDSTool.Toolbox.dssrt import (
    transition_psi,
    transition_tau
)


@pytest.mark.xfail(reason="FIXME: fails with AttributeError")
def test_dssrt():
    ep1 = args(actives=['x', 'y'], modulatory=['z', 'w'], sigma=2, relative_ratios=True)
    res1 = transition_psi(ep1, Point({'x': 1, 'y': 0.499, 'z': 0.2, 'w': 0.1}), 0.01)
    assert res1 == ('leave', 'y')
    res2 = transition_psi(ep1, Point({'x': 1, 'y': 0.601, 'z': 0.3, 'w': 0.1}), 0.01)
    assert res2 == ('join', 'z')
    ep2 = args(actives=['x'], modulatory=['z', 'w'], sigma=2, relative_ratios=True)
    res3 = transition_psi(ep2, Point({'x': 1, 'z': 0.1, 'w': 0.501}), 0.01)
    assert res3 == ('join', 'w')
    with pytest.raises(PyDSTool_ValueError):
        transition_psi(ep2, Point({'x': 1, 'z': 2, 'w': 0.1}), 0.01)
    ep1 = args(actives=['x', 'y'], modulatory=['z', 'w'], sigma=2, relative_ratios=False)
    res1 = transition_psi(ep1, Point({'x': 1, 'y': 0.499, 'z': 0.2, 'w': 0.1}), 0.01)
    assert res1 == ('leave', 'y')
    res2 = transition_psi(ep1, Point({'x': 1, 'y': 0.601, 'z': 0.499, 'w': 0.1}), 0.01)
    assert res2 == ('join', 'z')
    ep2 = args(actives=['x'], modulatory=['z', 'w'], sigma=2, relative_ratios=False)
    res3 = transition_psi(ep2, Point({'x': 1, 'z': 0.1, 'w': 0.501}), 0.01)
    assert res3 == ('join', 'w')
    with pytest.raises(PyDSTool_ValueError):
        transition_psi(ep2, Point({'x': 1, 'z': 2, 'w': 0.1}), 0.01)
    ep3 = args(actives=['x', 'y'], modulatory=[], sigma=2, relative_ratios=False)
    res5 = transition_psi(ep3, Point({'x': 1, 'y': 0.501}), 0.01)
    assert res5 == ('leave', 'y')
    ep1 = args(fast=['x', 'y'], slow=['w', 'z'], order1=['v'], gamma=2)
    res1 = transition_tau(ep1, Point({'x': 0.3, 'y': 0.2, 'v': 1, 'w': 2.005, 'z': 3}), 0.01)
    assert res1 == ('slow_leave', 'w')
