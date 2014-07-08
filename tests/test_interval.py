from __future__ import division

from PyDSTool import *
import pytest


def setup_module(PyDSTool):
    PyDSTool.TestInterval.a = Interval('a', float, [-1, 1], abseps=1e-5)
    PyDSTool.TestInterval.b = Interval('b', float, [-1., 1.])
    PyDSTool.TestInterval.c = Interval('c', float, [-3., -2.])
    PyDSTool.TestInterval.m = Interval('test1', float, (0, 1))
    PyDSTool.TestInterval.n = Interval('test2', float, (0, 0.4))
    PyDSTool.TestInterval.s = Interval('a_singleton', float, 0.4)
    PyDSTool.TestInterval.r = Interval('another_singleton', float, 0.0)
    PyDSTool.TestInterval.b = Interval('b', int, 0)
    PyDSTool.TestInterval.i = Interval('i', int, (0, 1))
    PyDSTool.TestInterval.ii = PyDSTool.TestInterval.i + 3
    PyDSTool.TestInterval.iii = 4 * PyDSTool.TestInterval.i
    PyDSTool.TestInterval.iiii = 2 - PyDSTool.TestInterval.i
    PyDSTool.TestInterval.iiiii_1 = 1 / PyDSTool.TestInterval.i
    PyDSTool.TestInterval.iiiii_2 = -1 / PyDSTool.TestInterval.i
    PyDSTool.TestInterval.i_neg = Interval('i_neg', float, (-1, 1))
    PyDSTool.TestInterval.ii_neg = 1 / PyDSTool.TestInterval.i_neg
    PyDSTool.TestInterval.j = Interval('test3', float, (0, 0.999999999))
    PyDSTool.TestInterval.q = PyDSTool.TestInterval.m.contains(0.9)
    PyDSTool.TestInterval.i2 = Interval('i2', int, (0, 10))
    PyDSTool.TestInterval.i3 = Interval('i3', float, (0., 0.4))
    PyDSTool.TestInterval.inf1 = Interval('inf1', float, [0, Inf], abseps=0)
    PyDSTool.TestInterval.inf2 = Interval('inf2', float, [-Inf, Inf])
    PyDSTool.TestInterval.inf3 = Interval('inf3', float, [-Inf, 0])
    PyDSTool.TestInterval.inf_int = Interval('inf3', int, [-Inf, 0])
    PyDSTool.TestInterval.i4 = Interval('i4', int, [-5, 5])
    PyDSTool.TestInterval.i5 = Interval('i5', int, [4, 4])
    PyDSTool.TestInterval.i5._abseps = 0


class TestInterval:

    def test_one(self):
        pytest.raises(PyDSTool_TypeError, "self.b.contains(self.a)")

    def test_two(self):
        assert -2 < self.a

    def test_three(self):
        assert self.a > -2

    def test_four(self):
        assert self.a < 1 + 2 * self.a._abseps

    def test_five(self):
        assert not (self.a < 1 + 0.5 * self.a._abseps)

    def test_six(self):
        assert 1 + 2 * self.a._abseps > self.a

    def test_seven(self):
        assert not (1 + 0.5 * self.a._abseps > self.a)

    def test_eight(self):
        assert self.c < self.b

    def test_nine(self):
        assert self.c < self.a

    def test_ten(self):
        assert self.a > self.c

    def test_eleven(self):
        assert ([-5, 0, -1] < self.a) == [True, False, False]

    def test_twelve(self):
        assert (self.a > array([-5, -4, -1])) == [True, True, False]

    def test_thirteen(self):
        pytest.raises(PyDSTool_UncertainValueError, "self.n in self.m")

    def test_fourteen(self):
        assert self.s in self.m

    def test_fifteen(self):
        pytest.raises(PyDSTool_UncertainValueError, "self.b in self.m")

    def test_sixteen(self):
        assert self.b in self.i

    def test_seventeen(self):
        assert self.ii[0] == self.i[0] + 3

    def test_eighteen(self):
        assert self.ii[1] == self.i[1] + 3

    def test_ninteen(self):
        assert self.iii[0] == 4 * self.i[0]

    def test_twenty(self):
        assert self.iii[1] == 4 * self.i[1]

    def test_twentyone(self):
        assert self.iiii[0] == 2 - self.i[1]

    def test_twentytwo(self):
        assert self.iiii[1] == 2 - self.i[0]

    def test_twentythree(self):
        assert self.iiiii_1[0] == 1

    def test_twentyfour(self):
        assert self.iiiii_1[1] == Inf

    def test_twentyfive(self):
        assert self.iiiii_2[0] == -Inf

    def test_twentysix(self):
        assert self.iiiii_2[1] == -1

    def test_twentyseven(self):
        assert self.i.contains(1) is contained

    def test_twentyeight(self):
        assert self.ii_neg.get() == [-1, 1]

    def test_twentynine(self):
        assert self.q is contained

    def test_thirty(self):
        assert not(self.q is True)

    def test_thirtyone(self):
        assert len(self.i3.sample(0.36, strict=True)) == 3

    def test_thirtytwo(self):
        pytest.raises(PyDSTool_TypeError, "self.inf_int.contains(self.inf3)")

    def test_thirtythree(self):
        assert self.inf_int.contains(-Inf)

    def test_thirtyfour(self):
        pytest.raises(PyDSTool_TypeError, "self.inf3.intersect(self.i4).get()")

    def test_thirtyfive(self):
        pytest.raises(PyDSTool_TypeError, "self.j.intersect(self.i2).get()")

    def test_thirtyfive(self):
        pytest.raises(PyDSTool_TypeError, "self.j.intersect(self.i2).get()")

    def test_thirtysix(self):
        assert self.i5.issingleton

    def test_thirtyseven(self):
        self.i5._abseps = 0
        assert 4 in self.i5

    def test_thirtyeight(self):
        self.i5._abseps = 0
        assert 4.0 in self.i5

    def test_thirtynine(self):
        assert 4.0 in self.i5
