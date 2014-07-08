#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy import (
    all,
    allclose,
    array,
    linspace,
    sin,
    sort,
)

from numpy.testing import assert_almost_equal
import pytest

from PyDSTool import (
    Events,
    args,
    makeMultilinearRegrFn,
)

from PyDSTool.Generator import (
    Euler_ODEsystem,
    InterpolateTable,
    Vode_ODEsystem,
)


def test_vode_events_compare_with_euler():
    """
        Test terminal and non-terminal event testing with VODE integrator,
        including some comparisons and tests of Euler integrator too.
    """
    DSargs = args(varspecs={'w': 'k*sin(2*t) - w'}, name='ODEtest')
    DSargs.tdomain = [0, 10]
    DSargs.pars = {'k': 1, 'p_thresh': -0.25}
    DSargs.algparams = {'init_step': 0.001, 'atol': 1e-12, 'rtol': 1e-13}
    DSargs.checklevel = 2
    DSargs.ics = {'w': -1.0}
    DSargs.tdata = [0, 10]
    ev_args_nonterm = {'name': 'monitor',
                       'eventtol': 1e-4,
                       'eventdelay': 1e-5,
                       'starttime': 0,
                       'active': True,
                       'term': False,
                       'precise': True}
    thresh_ev_nonterm = Events.makeZeroCrossEvent('w', 0,
                                                  ev_args_nonterm, varnames=['w'])
    ev_args_term = {'name': 'threshold',
                    'eventtol': 1e-4,
                    'eventdelay': 1e-5,
                    'starttime': 0,
                    'active': True,
                    'term': True,
                    'precise': True}
    thresh_ev_term = Events.makeZeroCrossEvent('w-p_thresh',
                                               -1, ev_args_term, varnames=['w'], parnames=['p_thresh'])
    DSargs.events = [thresh_ev_nonterm, thresh_ev_term]
    testODE = Vode_ODEsystem(DSargs)
    # diagnostics and other possible user-defined python functions
    # for python solvers only (currently only Euler)
    # def before_func(euler):
    # print(euler.algparams['init_step'])
    #
    # def after_func(euler):
    # print(euler._solver.y)
    #
    ##DSargs.user_func_beforestep = before_func
    ##DSargs.user_func_afterstep = after_func
    testODE_Euler = Euler_ODEsystem(DSargs)
    traj = testODE.compute('traj')
    traj2 = testODE_Euler.compute('traj')
    pts = traj.sample()
    testODE.diagnostics.showWarnings()
    mon_evs_found = testODE.getEvents('monitor')
    term_evs_found = testODE.getEvents('threshold')
    # test Euler
    assert allclose(array(testODE.getEventTimes('monitor')),
                    array(traj2.getEventTimes('monitor')), atol=1e-3)
    assert all(traj.getEvents('monitor') == mon_evs_found)
    assert all(traj.getEventTimes('threshold')
               == testODE.getEventTimes('threshold'))
    term_evs_found.info()
    # Alternative way to extract events: they are labelled in the
    # pointset! These return dictionaries indexing into the pointset.
    mon_evs_dict = pts.labels.by_label['Event:monitor']
    mon_ev_points = pts[sort(list(mon_evs_dict.keys()))]
    assert len(mon_evs_found) == len(mon_ev_points) == 2
    assert all(mon_evs_found == mon_ev_points)


@pytest.fixture
def my_input():
    timeData = linspace(0, 10, 20)
    sindata = sin(20 * timeData)
    xData = {'example_input': sindata}
    return InterpolateTable({
        'tdata': timeData,
        'ics': xData,
        'name': 'interp1d',
        'method': 'linear',
        'checklevel': 1,
        'abseps': 1e-5
    }).compute('interp')


def test_vode_events_with_external_input(my_input):
    """
        Test Vode_ODEsystem with events involving external inputs.
        Robert Clewley, September 2006.
    """
    xs = ['x1', 'x2', 'x3']
    ys = [0, 0.5, 1]
    fvarspecs = {"w": "k*w  + pcwfn(sin(t)) + myauxfn1(t)*myauxfn2(w)",
                 'aux_wdouble': 'w*2 + globalindepvar(t)',
                 'aux_other': 'myauxfn1(2*t) + initcond(w)'}
    fnspecs = {'myauxfn1': (['t'], '2.5*cos(3*t)'),
               'myauxfn2': (['w'], 'w/2'),
               'pcwfn': makeMultilinearRegrFn('x', xs, ys)}
    # targetlang is optional if the default python target is desired
    DSargs = args(fnspecs=fnspecs, name='ODEtest')
    DSargs.varspecs = fvarspecs
    DSargs.tdomain = [0.1, 2.1]
    DSargs.pars = {'k': 2, 'a': -0.5, 'x1': -3, 'x2': 0.5, 'x3': 1.5}
    DSargs.vars = 'w'
    DSargs.inputs = {'in': my_input.variables['example_input']}
    DSargs.algparams = {'init_step': 0.01}
    DSargs.checklevel = 2
    testODE = Vode_ODEsystem(DSargs)
    assert not testODE.defined
    testODE.set(ics={'w': 3.0},
                tdata=[0.11, 2.1])
    traj1 = testODE.compute('traj1')
    assert testODE.defined
    assert_almost_equal(traj1(0.5, 'w'), 8.9771499, 5)
    assert not testODE.diagnostics.hasWarnings()
    assert_almost_equal(traj1(0.2, ['aux_other']), 3.905819936, 5)
    print("\nNow adding a terminating co-ordinate threshold event")
    print(" and non-terminating timer event")
    # Show off the general-purpose, language-independent event creator:
    #  'makeZeroCrossEvent'
    ev_args_nonterm = {'name': 'monitor',
                       'eventtol': 1e-4,
                       'eventdelay': 1e-5,
                       'starttime': 0,
                       'active': True,
                       'term': False,
                       'precise': True}
    thresh_ev_nonterm = Events.makeZeroCrossEvent('in', 0,
                                                  ev_args_nonterm, inputnames=['in'])
    # Now show use of the python-target specific creator:
    #  'makePythonStateZeroCrossEvent', which is also only
    #  able to make events for state variable threshold crossings
    ev_args_term = {'name': 'threshold',
                    'eventtol': 1e-4,
                    'eventdelay': 1e-5,
                    'starttime': 0,
                    'active': True,
                    'term': True,
                    'precise': True}
    thresh_ev_term = Events.makePythonStateZeroCrossEvent('w',
                                                          20, 1, ev_args_term)
    testODE.eventstruct.add([thresh_ev_nonterm, thresh_ev_term])
    print("Recomputing trajectory:")
    print("traj2 = testODE.compute('traj2')")
    traj2 = testODE.compute('traj2')
    print("\ntestODE.diagnostics.showWarnings() => ")
    testODE.diagnostics.showWarnings()
    print("\ntraj2.indepdomain.get() => ", traj2.indepdomain.get())
    indep1 = traj2.indepdomain[1]
    assert indep1 < 1.17 and indep1 > 1.16
    mon_evs_found = testODE.getEvents('monitor')
    assert len(mon_evs_found) == 1
