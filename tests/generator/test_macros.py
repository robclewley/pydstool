#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import pytest
from numpy import array, pi, allclose

from PyDSTool import args
from PyDSTool.Generator import (
    Euler_ODEsystem,
    Vode_ODEsystem,
    Dopri_ODEsystem,
    Radau_ODEsystem,
)

from .helpers import clean_files


@pytest.fixture
def fnspecs():
    return {
        'testif': (['x'], 'if(x<0.0,0.0,x)'),
        'testmin': (['x', 'y'], 'min(x,y)'),
        'testmax': (['x', 'y'], 'max(x,y)'),
        'testmin2': (['x', 'y'], '1/(2+min(1+(x*3),y))+y'),
        'indexfunc': (['x'], 'pi*x')
    }


def test_macros_euler(fnspecs):
    """Test if, min, max, & for macro"""

    _run_check_macros_1(Euler_ODEsystem, fnspecs)
    _run_check_macros_2(Euler_ODEsystem, fnspecs)
    _run_check_macros_3(Euler_ODEsystem, fnspecs)


def test_macros_vode(fnspecs):
    """Test if, min, max, & for macro"""

    _run_check_macros_1(Vode_ODEsystem, fnspecs)
    _run_check_macros_2(Vode_ODEsystem, fnspecs)
    _run_check_macros_3(Vode_ODEsystem, fnspecs)


def test_macros_dopri(fnspecs):

    _run_check_macros_1(Dopri_ODEsystem, fnspecs)
    _run_check_macros_2(Dopri_ODEsystem, fnspecs)


def test_macros_radau(fnspecs):

    _run_check_macros_1(Radau_ODEsystem, fnspecs)
    _run_check_macros_2(Radau_ODEsystem, fnspecs)


def _run_check_macros_1(ode, fnspecs):
    DSargs = args(
        name='test',
        pars={'p0': 0, 'p1': 1, 'p2': 2},
        # 'special_erfc' is not available for non-Python generators
        varspecs={
            'z[j]': 'for(j, 0, 1, 2*z[j+1] + p[j])',
            'z2': '-z0 + p2 + special_erfc(2.0)' if ode is Vode_ODEsystem else '-z0 + p2 + 1',
        },
        fnspecs=fnspecs
    )

    tmm = ode(DSargs)
    # test user interface to aux functions and different combinations of embedded
    # macros
    assert tmm.auxfns.testif(1.0) == 1.0
    assert tmm.auxfns.testmin(1.0, 2.0) == 1.0
    assert tmm.auxfns.testmax(1.0, 2.0) == 2.0
    assert tmm.auxfns.testmin2(1.0, 2.0) == 2.25
    assert tmm.Rhs(0, {'z0': 0.5, 'z1': 0.2, 'z2': 2.1})[1] == 5.2


def _run_check_macros_2(ode, fnspecs):

    DSargs2 = args(name='test2',
                   pars={'p0': 0, 'p1': 1, 'p2': 2},
                   varspecs={
                       'y[i]': 'for(i, 0, 1, y[i]+if(y[i+1]<2, 2+p[i]+getbound(%s,0), indexfunc([i])+y[i+1]) - 3)' % ('y2' if ode in [Vode_ODEsystem, Euler_ODEsystem] else '"y2"'),
                       'y2': '0'
                   },
                   xdomain={'y2': [-10, 10]},
                   fnspecs=fnspecs,
                   ics={'y0': 0, 'y1': 0, 'y2': 0.1}
                   )
    tm2 = ode(DSargs2)
    tm2.set(tdata=[0, 10])
    tm2.compute('test')
    assert allclose(
        tm2.Rhs(0, {'y0': 0, 'y1': 0.3, 'y2': 5}), array([-11., 2.3 + pi, 0.]))


def _run_check_macros_3(ode, fnspecs):
    # show example of summing where i != p defines the sum range, and p is a
    # special value (here, 2)
    DSargs3 = args(name='test3',
                   pars={'p0': 0, 'p1': 1, 'p2': 2},
                   varspecs={
                        'x': 'sum(i, 0, 4, sum(j, 0, 1, if([i]==2, 0, indexfunc([j] + p[j]))))'},
                   fnspecs=fnspecs
                   )
    tm3 = ode(DSargs3)
    tm3.set(tdata=[0, 10], ics={'x': 1})
    tm3.compute('test')
    assert allclose(tm3.Rhs(0, {'x': 0}), 8 * pi)


def teardown_module():
    clean_files(['test', 'test2'])
