"""
Tests and demonstration of Point, Pointset, and their label classes
"""

from numpy import (
    array,
    float64,
)
from PyDSTool import (
    comparePointCoords,
    Point,
    Pointset,
    pointsToPointset,
)
import pytest


def test_one():
    x = Point({
        'coorddict': {'x0': [1.123456789], 'x1': [-0.4], 'x2': [4000]},
        'coordtype': float})
    x[1] = -0.45
    x[['x0', 'x1']] = [4.11103, -0.56]
    assert comparePointCoords(x, (x + 0) * 1, fussy=True)


def test_two():
    k = Pointset({
        'coordarray': array(0.1),
        'coordnames': 'k0',
        'indepvarname': 't',
        'indepvararray': array(0.0)})
    assert k.dimension == 1


class TestPointSet(object):
    def setup(self):
        # self.y = Point({'y': 4})
        # self.v = Pointset({'coorddict': {'x0': 0.2, 'x1': -1.2},
        #                    'indepvardict': {'t': 0.01},
        #                    'coordtype': float64,
        #                    'indepvartype': float64})
        # self.u = Pointset(
        #     {'coordarray': array([10., 20., 30., 40.])})
        # self.w_x0 = self.wp['x0']
        # self.w_at_1 = self.wp(1.).toarray()
        self.wp = Pointset(
            {'coordarray': array([	[4.456, 2.34634, 7.3431, 5.443],
                                    [-10.0336, -5.2235, -3.23221, -0.01],
                                    [3e5, 3.1e5, 3.3e5, 2.8e5]], float64),
                'coordnames': ['x0', 'x1', 'x2'],
                'indepvarname': 't',
                'indepvararray': array([0.0, 1.0, 2.0, 3.0], float64)})
        self.wp_ins = Pointset(
            {'coorddict': {'x0': [-2.1, -4., -5., -4.5], 'x1': [50., 51., 52., 54.], 'x2': [0.01, 0.02, 0.4, 0.9]},
             'indepvardict': {'t': [1.5, 5.2, 9., 10.]},
             'coordtype': float64,
             'indepvartype': float64,
             'labels': {2: 'b', 3: {'a': {'bif': 'H'}}}
             })
        self.wp2 = Pointset(
            {'coorddict': {'x0': [-4.5, 2, 3], 'x1': [54, 62, 64], 'x2': [0.9, 0.8, 0.2]},
             'indepvardict': {'t': [10, 11, 12]},
             'coordtype': float64,
             'indepvartype': float64,
             'labels': {0: {'a_different': {'bif': 'H'}}, 2: 'd'}
             })

    def test_three(self):
        assert type(self.wp.coordarray) == type(array([1, 2], float64))

    def test_four(self):
        vw = Pointset(
            {'coorddict': {'x0': [0.1, 0.15], 'x1': [100., 102], 'x2': [0.2, 0.1]},
             'indepvardict': {'t': [4.5, 5.0]},
             'coordtype': float64,
             'indepvartype': float64,
             'labels': {1: 'c'}})
        self.wp.append(vw)
        self.wp.append(Point({'coorddict': {'t': 6.5, 'x0': 2, 'x1': -300, 'x2': -0.9997}}))
        assert type(self.wp.coordarray) == type(array([1, 2], float64))

    def test_five(self):
        self.wp.toarray()
        self.wp.labels[3] = ('a', {'bif': 'SN'})
        self.wp.insert(self.wp_ins)
        wp_part = self.wp[3:5]
        assert wp_part.labels[0] == self.wp.labels[3]

    def test_six(self):
        self.wp.labels[3] = ('a', {'bif': 'SN'})
        wpt = self.wp(3.)
        assert wpt.labels == {'a': {'bif': 'SN'}}

    def test_seven(self):
        self.pointlist = []
        self.wp.insert(self.wp_ins)
        for t in self.wp['t']:
            self.pointlist.append(self.wp(t))
        w_reconstructed = pointsToPointset(
            self.pointlist, 't', self.wp['t'])
        with pytest.raises(ValueError):
            w_double = w_reconstructed.append(w_reconstructed)

    def test_eight(self):
        # wnp = pointsToPointset(self.pointlist)
        # wnp.labels[0] = ('b', {})
        # wnp.addlabel(4, 'c', {'bif': 'H'})  # preferred syntax
        vw = Pointset(
            {'coorddict': {'x0': [0.1, 0.15], 'x1': [100., 102], 'x2': [0.2, 0.1]},
             'indepvardict': {'t': [4.5, 5.0]},
             'coordtype': float64,
             'indepvartype': float64,
             'labels': {1: 'c'}})
        self.wp.append(vw)
        self.wp.append(Point({'coorddict': {'t': 6.5, 'x0': 2, 'x1': -300, 'x2': -0.9997}}))
        self.wp.insert(self.wp_ins)
        self.wp.append(self.wp2, skipMatchingIndepvar=True)
        assert len(self.wp) == 13

    def test_nine(self):
        self.wp.insert(self.wp_ins)
        assert self.wp.bylabel('b')['t'][0] == 9.0

    def test_ten(self):
        self.wp.labels[3] = ('a', {'bif': 'SN'})
        self.wp.insert(self.wp_ins)
        assert all(self.wp.bylabel('a')['t'] == array([3., 10.]))

    def test_eleven(self):
        self.wp.append(self.wp2, skipMatchingIndepvar=True)
        assert self.wp.bylabel('d')['t'][0] == 12.0

    def test_twelve(self):
        self.wp.append(self.wp2, skipMatchingIndepvar=True)
        assert all(self.wp.bylabel('a_different')['t'] == array([10.]))

    def test_thirteen(self):
        self.wp.insert(self.wp_ins)
        self.wp.append(self.wp2, skipMatchingIndepvar=True)
        z = self.wp[-5:]
        assert z.labels.getIndices() == [1, 2, 4]
