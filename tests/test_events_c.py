#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest

from numpy import linspace, sin
from PyDSTool import (
    args,
    Events,
    makeDataDict,
)
from PyDSTool.Generator import (
    Dopri_ODEsystem,
    InterpolateTable,
    Radau_ODEsystem,
)


@pytest.fixture
def dsargs():
    timeData = linspace(0, 10, 20)
    sindata = sin(20 * timeData)
    xData = makeDataDict(['in'], [sindata])
    my_input = InterpolateTable({
        'tdata': timeData,
        'ics': xData,
        'name': 'interp1d',
        'method': 'linear',
        'checklevel': 1,
        'abseps': 1e-5
    }).compute('interp')

    fvarspecs = {
        "w": "k*w  + sin(t) + myauxfn1(t)*myauxfn2(w)",
        'aux_wdouble': 'w*2 + globalindepvar(t)',
        'aux_other': 'myauxfn1(2*t) + initcond(w)'
    }
    fnspecs = {
        'myauxfn1': (['t'], '2.5*cos(3*t)'),
        'myauxfn2': (['w'], 'w/2')
    }
    # targetlang is optional if the default python target is desired
    DSargs = args(fnspecs=fnspecs, name='event_test')
    DSargs.varspecs = fvarspecs
    DSargs.tdomain = [0.1, 2.1]
    DSargs.pars = {'k': 2, 'a': -0.5}
    DSargs.vars = 'w'
    DSargs.ics = {'w': 3}
    DSargs.inputs = {'in': my_input.variables['in']}
    DSargs.algparams = {'init_step': 0.01}
    DSargs.checklevel = 2
    ev_args_nonterm = {
        'name': 'monitor',
        'eventtol': 1e-4,
        'eventdelay': 1e-5,
        'starttime': 0,
        'active': True,
        'term': False,
        'precise': True
    }
    thresh_ev_nonterm = Events.makeZeroCrossEvent(
        'in',
        0,
        ev_args_nonterm,
        inputnames=['in'],
        targetlang='c'
    )

    ev_args_term = {
        'name': 'threshold',
        'eventtol': 1e-4,
        'eventdelay': 1e-5,
        'starttime': 0,
        'active': True,
        'term': True,
        'precise': True
    }

    thresh_ev_term = Events.makeZeroCrossEvent(
        'w-20',
        1,
        ev_args_term,
        ['w'],
        targetlang='c'
    )
    DSargs.events = [thresh_ev_nonterm, thresh_ev_term]

    return DSargs


def test_dopri_event(dsargs):
    """
        Test Dopri_ODEsystem with events involving external inputs.

        Robert Clewley, September 2006.
    """

    _run_checks(Dopri_ODEsystem(dsargs))

    _clean_files([
        'dop853_event_test_vf.py',
        'dop853_event_test_vf.pyc',
        '_dop853_event_test_vf.so',
    ])


def test_radau_event(dsargs):
    """
        Test Radau_ODEsystem with events involving external inputs.

        Robert Clewley, September 2006.
    """

    _run_checks(Radau_ODEsystem(dsargs))

    _clean_files([
        'radau5_event_test_vf.py',
        'radau5_event_test_vf.pyc',
        '_radau5_event_test_vf.so',
    ])


def _run_checks(ode):

    print "Computing trajectory:"
    print "traj = testODE.compute('traj')"
    traj = ode.compute('traj')

    print "\ntestODE.diagnostics.showWarnings() => "
    ode.diagnostics.showWarnings()
    print "\ntraj.indepdomain.get() => ", traj.indepdomain.get()
    indep1 = traj.indepdomain[1]
    assert indep1 < 1.15 and indep1 > 1.13
    mon_evs_found = ode.getEvents()['monitor']
    assert len(mon_evs_found) == 1

    pts = traj.sample()
    for t in linspace(0, 10, 20):
        if t <= pts['t'][-1] and t >= pts['t'][0]:
            assert t in pts['t'], "t=%f not present in output!" % t


def _clean_files(files):
    for f in files:
        os.remove(f)