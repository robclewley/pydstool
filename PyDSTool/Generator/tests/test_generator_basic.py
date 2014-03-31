#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import pytest
from numpy import array, float64
from numpy.testing import (
    assert_almost_equal,
)

from PyDSTool import (
    Interval,
    Events,
    makeZeroCrossEvent,
    PyDSTool_BoundsError,
)
from PyDSTool.Generator import (
    InterpolateTable,
    Vode_ODEsystem,
    ExplicitFnGen,
    Dopri_ODEsystem,
    Radau_ODEsystem,
)


@pytest.fixture
def tb_args():
    xnames = ['x1', 'x2']
    timeData = array([0.1, 1.1, 2.1])
    x1data = array([10.2, -1.4, 4.1])
    x2data = array([0.1, 0.01, 0.4])
    xData = dict(zip(xnames, [x1data, x2data]))
    itableArgs = {}
    itableArgs['tdata'] = timeData
    itableArgs['ics'] = xData
    itableArgs['name'] = 'interp'
    return itableArgs


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
    assert_almost_equal(testtraj(0.5, 'w'), 6.05867901304, 3)
    assert_almost_equal(testtraj(0.2, 'aux_other'), 3.90581993688, 3)
    assert testODE.indepvariable.depdomain == Interval(
        't', float64, [0.11, 2.1])
    assert testODE.diagnostics.hasWarnings()
    assert testODE.diagnostics.findWarnings(21) != []

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
    print(testODE.diagnostics.showWarnings())
    assert_almost_equal(traj2.getEventTimes()['threshold'][0], 1.51449456, 3)
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
