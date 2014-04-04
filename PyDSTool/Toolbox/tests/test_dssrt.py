from PyDSTool import (
    Point,
    PyDSTool_ValueError,
    parseUtils,
    common
)

from PyDSTool.Toolbox.dssrt import (
    transition_psi,
    transition_tau,
    indent
)

import pytest


def setup_module(PyDSTool):
    # Tests with dummy epoch stubs
    namemap = parseUtils.symbolMapClass({'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z'})
    ep1 = common.args(actives=['x','y'],modulatory=['z','w'],sigma=2,relative_ratios=True,
                    inv_psi_namemap=namemap, inv_tau_namemap=namemap)
    res1 = transition_psi(ep1, Point({'x': 1, 'y': 0.499, 'z': 0.2, 'w': 0.1}), 0.01)
    res2 = transition_psi(ep1, Point({'x': 1, 'y': 0.601, 'z': 0.3, 'w': 0.1}), 0.01)

    ep2 = common.args(actives=['x'],modulatory=['z','w'],sigma=2,relative_ratios=True,
                    inv_psi_namemap=namemap, inv_tau_namemap=namemap)
    res3 = transition_psi(ep2, Point({'x': 1, 'z': 0.1, 'w': 0.501}), 0.01)

    ep3 = common.args(actives=['x','y'],modulatory=['z','w'],sigma=2,relative_ratios=False,
                    inv_psi_namemap=namemap, inv_tau_namemap=namemap)
    res5 = transition_psi(ep3, Point({'x': 1, 'y': 0.499, 'z': 0.2, 'w': 0.1}), 0.01)
    res6 = transition_psi(ep3, Point({'x': 1, 'y': 0.601, 'z': 0.499, 'w': 0.1}), 0.01)

    ep4 = common.args(actives=['x'],modulatory=['z','w'],sigma=2,relative_ratios=False,
                            inv_psi_namemap=namemap, inv_tau_namemap=namemap)
    res7 = transition_psi(ep4, Point({'x': 1, 'z': 0.1, 'w': 0.501}), 0.01)

    ep5 = common.args(actives=['x','y'],modulatory=[],sigma=2,relative_ratios=False,
                            inv_psi_namemap=namemap, inv_tau_namemap=namemap)
    res9 = transition_psi(ep5, Point({'x': 1, 'y': 0.501}), 0.01)

    ep6 = common.args(fast=['x','y'],slow=['w','z'],order1=['v'],gamma=2,
                    inv_psi_namemap=namemap, inv_tau_namemap=namemap)
    res10 = transition_tau(ep6, Point({'x': 0.3, 'y': 0.2, 'v':1, 'w': 2.005, 'z': 3}), 0.01)

    PyDSTool.TestDSSRT.ep1 = ep1
    PyDSTool.TestDSSRT.ep2 = ep2
    PyDSTool.TestDSSRT.ep3 = ep3
    PyDSTool.TestDSSRT.ep4 = ep4
    PyDSTool.TestDSSRT.ep5 = ep5
    PyDSTool.TestDSSRT.ep6 = ep6

    PyDSTool.TestDSSRT.res1 = res1
    PyDSTool.TestDSSRT.res2 = res2
    PyDSTool.TestDSSRT.res3 = res3
    # no 4
    PyDSTool.TestDSSRT.res5 = res5
    PyDSTool.TestDSSRT.res6 = res6
    PyDSTool.TestDSSRT.res7 = res7
    # no  8
    PyDSTool.TestDSSRT.res9 = res9
    PyDSTool.TestDSSRT.res10 = res10


class TestDSSRT:

    def test_one(self):
        assert self.res1 == ('leave', 'y')

    def test_two(self):
        assert self.res2 == ('join', 'z')

    def test_three(self):
        assert self.res3 == ('join', 'w')

    def test_four(self):
        pytest.raises(PyDSTool_ValueError, "transition_psi(self.ep2, Point({'x': 1, 'z': 2, 'w': 0.1}), 0.01)")

    def test_five(self):
        assert self.res5 == ('leave', 'y')

    def test_six(self):
        assert self.res6 == ('join', 'z')

    def test_seven(self):
        assert self.res7 == ('join', 'w')

    def test_eight(self):
        pytest.raises(PyDSTool_ValueError, "transition_psi(self.ep4, Point({'x': 1, 'z': 2, 'w': 0.1}), 0.01)")

    def test_nine(self):
        assert self.res9 == ('leave', 'y')

    def test_ten(self):
        assert self.res10 == ('slow_leave', 'w')


def test_indent_regression():
    table = [
        ['Name', 'Value', 'Comment'],
        ['x', '0.0', ''],
        ['y', '1.0', 'Multi line\ncomment'],
        ['z', '2.0', 'Z value'],
    ]

    assert  [
        '-------------------------',
        'Name | Value | Comment   ',
        '-------------------------',
        'x    | 0.0   |           ',
        '-------------------------',
        'y    | 1.0   | Multi line',
        '     |       | comment   ',
        '-------------------------',
        'z    | 2.0   | Z value   ',
        '-------------------------',
        '',
    ] == indent(table, hasHeader=True, separateRows=True).split('\n')
