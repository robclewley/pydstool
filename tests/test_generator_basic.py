#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
from numpy import array, float64, pi, allclose
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
)

from PyDSTool import (
    makeDataDict,
    Interval,
    Events,
    makeZeroCrossEvent,
    PyDSTool_BoundsError,
    args,
)
from PyDSTool.Generator import (
    InterpolateTable,
    LookupTable,
    Vode_ODEsystem,
    ExplicitFnGen,
    Dopri_ODEsystem,
)


@pytest.fixture
def tb_args():
    xnames = ['x1', 'x2']
    timeData = array([0.1, 1.1, 2.1])
    x1data = array([10.2, -1.4, 4.1])
    x2data = array([0.1, 0.01, 0.4])
    xData = makeDataDict(xnames, [x1data, x2data])
    itableArgs = {}
    itableArgs['tdata'] = timeData
    itableArgs['ics'] = xData
    itableArgs['name'] = 'interp'
    return itableArgs


def test_interpolatetable(tb_args):
    interptable = InterpolateTable(tb_args)
    itabletraj = interptable.compute('itable')
    assert_almost_equal(itabletraj(0.4, 'x1'), 6.72)
    assert_array_almost_equal(itabletraj(1.1), [-1.4, 0.01])


def test_lookuptable(tb_args):
    tb_args['name'] = 'lookup'
    lookuptable = LookupTable(tb_args)
    ltabletraj = lookuptable.compute('ltable')
    assert_array_almost_equal(ltabletraj(1.1), [-1.4, 0.01])
    with pytest.raises(ValueError):
        ltabletraj(0.4)


def test_ode_system(tb_args):
    fvarspecs = {
        "w": "k*w + a*itable + sin(t) + myauxfn1(t)*myauxfn2(w)",
        'aux_wdouble': 'w*2 + globalindepvar(t)',
        'aux_other': 'myauxfn1(2*t) + initcond(w)'
    }
    fnspecs = {
        'myauxfn1': (['t'], '2.5*cos(3*t)'),
        'myauxfn2': (['w'], 'w/2')
    }
    DSargs = {
        'tdomain': [0.1, 2.1],
        'pars': {'k': 2, 'a': -0.5},
        'inputs': {'itable': InterpolateTable(tb_args).variables['x1']},
        'auxvars': ['aux_wdouble', 'aux_other'],
        'algparams': {'init_step': 0.01, 'strict': False},
        'checklevel': 2,
        'name': 'ODEtest',
        'fnspecs': fnspecs,
        'varspecs': fvarspecs
    }
    testODE = Vode_ODEsystem(DSargs)
    assert testODE.pars == DSargs['pars']
    assert (not testODE.defined)
    testODE.set(
        ics={'w': 3.0},
        tdata=[0.11, 2.1]
    )
    testtraj = testODE.compute('test1')
    assert testODE.defined
    assert_almost_equal(testtraj(0.5, 'w'), 6.05867901304, 4)
    assert_almost_equal(testtraj(0.2, 'aux_other'), 3.90581993688, 4)
    assert testODE.indepvariable.depdomain == Interval(
        't', float64, [0.11, 2.1])
    assert testODE.diagnostics.hasWarnings()
    # FIXME: next fails with ValueError
    # assert testODE.diagnostics.findWarnings(21) != []

    # Now adding a terminating co-ordinate threshold event...
    ev_args = {
        'name': 'threshold',
        'eventtol': 1e-4,
        'eventdelay': 1e-5,
        'starttime': 0,
        'active': True,  # = default
        'term': True,
        'precise': True  # = default
    }
    thresh_ev = Events.makePythonStateZeroCrossEvent('w', 20, 1, ev_args)
    testODE.eventstruct.add(thresh_ev)
    traj2 = testODE.compute('test2')
    assert testODE.diagnostics.hasWarnings()
    assert testODE.diagnostics.findWarnings(10) != []
    print testODE.diagnostics.showWarnings()
    assert_almost_equal(traj2.getEventTimes()['threshold'][0], 1.51449456, 4)
    assert testODE.indepvariable.depdomain == Interval(
        't', float64, [0.11, 2.1])


def test_explicit_functional_trajectory():
    """Explicit functional trajectory 'sin_gen' computes sin(t*speed)"""

    sine_time_ev = makeZeroCrossEvent(
        't-2',
        1,
        {'name': 'sine_time_test', 'term': True}
    )

    # Make 'xdomain' argument smaller than known limits for sine wave:
    # [-1.001, 0.7]
    ef_args = {
        'tdomain': [-50, 50],
        'pars': {'speed': 1},
        'xdomain': {'s': [-1., 0.7]},
        'name': 'sine',
        'globalt0': 0.4,
        'pdomain': {'speed': [0, 200]},
        'varspecs': {'s': "sin(globalindepvar(t)*speed)"},
        'events': sine_time_ev
    }
    sin_gen = ExplicitFnGen(ef_args)
    sintraj = sin_gen.compute('sinewave')
    assert sintraj(0.0, checklevel=2)['s'] - 0.38941834 < 1e-7

    # Expect problem calling at t=0.8...
    with pytest.raises(PyDSTool_BoundsError):
        sintraj(0.8, checklevel=2)
    sin_gen.set(xdomain={'s': [-1., 1.]})
    sintraj2 = sin_gen.compute('sinewave2')
    # this doesn't raise an exception now
    sintraj2(0.8, checklevel=2)
    evts = sintraj.getEventTimes()
    assert len(evts) == 1


def test_macros_vode():
    """Test if, min, max, & for macro"""
    fnspecs = {'testif': (['x'], 'if(x<0.0,0.0,x)'),
               'testmin': (['x', 'y'], 'min(x,y)'),
               'testmax': (['x', 'y'], 'max(x,y)'),
               'testmin2': (['x', 'y'], '1/(2+min(1+(x*3),y))+y'),
               'indexfunc': (['x'], 'pi*x')
               }

    DSargs = args(name='test',
                  pars={'p0': 0, 'p1': 1, 'p2': 2},
                  varspecs={'z[j]': 'for(j, 0, 1, 2*z[j+1] + p[j])',
                            'z2': '-z0 + p2 + special_erfc(2.0)'},
                  fnspecs=fnspecs
                  )
    tmm = Vode_ODEsystem(DSargs)

    # test user interface to aux functions and different combinations of embedded
    # macros

    assert tmm.auxfns.testif(1.0) == 1.0
    assert tmm.auxfns.testmin(1.0, 2.0) == 1.0
    assert tmm.auxfns.testmax(1.0, 2.0) == 2.0
    assert tmm.auxfns.testmin2(1.0, 2.0) == 2.25
    assert tmm.Rhs(0, {'z0': 0.5, 'z1': 0.2, 'z2': 2.1})[1] == 5.2

    DSargs2 = args(name='test2',
                   pars={'par0': 0, 'par1': 1, 'par2': 2},
                   varspecs={
                        'y[i]': "for(i, 0, 1, y[i]+if(y[i+1]<2, 2+par[i]+getbound(y2,0), indexfunc([i])+y[i+1]) - 3)",
                        'y2': '0'
                   },
                   xdomain={'y2': [-10, 10]},
                   fnspecs=fnspecs,
                   ics={'y0': 0, 'y1': 0, 'y2': 0.1}
                   )
    tm2 = Vode_ODEsystem(DSargs2)
    tm2.set(tdata=[0, 10])
    tm2.compute('test')
    assert allclose(
        tm2.Rhs(0, {'y0': 0, 'y1': 0.3, 'y2': 5}), array([-11., 2.3 + pi, 0.]))

    # show example of summing where i != p defines the sum range, and p is a
    # special value (here, 2)
    DSargs3 = args(name='test3',
                   pars={'p0': 0, 'p1': 1, 'p2': 2},
                   varspecs={
                        'x': 'sum(i, 0, 4, sum(j, 0, 1, if([i]==2, 0, indexfunc([j] + p[j]))))'},
                   fnspecs=fnspecs
                   )
    tm3 = Vode_ODEsystem(DSargs3)
    tm3.set(tdata=[0, 10], ics={'x': 1})
    tm3.compute('test')
    assert allclose(tm3.Rhs(0, {'x': 0}), 8 * pi)


def test_macros_dopri():
    fnspecs = {'testif': (['x'], 'if(x<0.0,0.0,x)'),
               'testmin': (['x', 'y'], 'min(x,y)'),
               'testmax': (['x', 'y'], 'max(x,y)'),
               'testmin2': (['x', 'y'], '1/(2+min(1+(x*3),y))+y'),
               'indexfunc': (['x'], 'pi*x')
               }
    DSargs = args(name='test',
                  pars={'p0': 0, 'p1': 1, 'p2': 2},
                  varspecs={'z[j]': 'for(j, 0, 1, 2*z[j+1] + p[j])',
                            'z2': '-z0 + p2 + 1'},
                  fnspecs=fnspecs
                  )
    tmm = Dopri_ODEsystem(DSargs)
    # test user interface to aux functions and different combinations of embedded
    # macros
    assert tmm.auxfns.testif(1.0) == 1.0
    assert tmm.auxfns.testmin(1.0, 2.0) == 1.0
    assert tmm.auxfns.testmax(1.0, 2.0) == 2.0
    assert tmm.auxfns.testmin2(1.0, 2.0) == 2.25
    assert tmm.Rhs(0, {'z0': 0.5, 'z1': 0.2, 'z2': 2.1})[1] == 5.2
    DSargs2 = args(name='test2',
                   pars={'p0': 0, 'p1': 1, 'p2': 2},
                   varspecs={
                       'y[i]': 'for(i, 0, 1, y[i]+if(y[i+1]<2, 2+p[i]+getbound("y2",0), indexfunc([i])+y[i+1]) - 3)',
                       'y2': '0'
                   },
                   xdomain={'y2': [-10, 10]},
                   fnspecs=fnspecs,
                   ics={'y0': 0, 'y1': 0, 'y2': 0.1}
                   )
    tm2 = Dopri_ODEsystem(DSargs2)
    tm2.set(tdata=[0, 10])
    tm2.compute('test')
    assert allclose(
        tm2.Rhs(0, {'y0': 0, 'y1': 0.3, 'y2': 5}), array([-11., 2.3 + pi, 0.]))
    # cleaning up
    files = [
        'dop853_test_vf.py',
        'dop853_test_vf.pyc',
        '_dop853_test_vf.so',
        'dop853_test2_vf.py',
        'dop853_test2_vf.pyc',
        '_dop853_test2_vf.so',
    ]
    for f in files:
        os.remove(f)
