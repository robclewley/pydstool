#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import pytest

from numpy import linspace, sin
from numpy.testing import assert_almost_equal
from PyDSTool import (
    args,
    Events
)
from PyDSTool.Generator import (
    Dopri_ODEsystem,
    Euler_ODEsystem,
    InterpolateTable,
    Radau_ODEsystem,
    Vode_ODEsystem,
)

from .helpers import clean_files


@pytest.fixture
def dsargs():
    timeData = linspace(0, 10, 20)
    sindata = sin(20 * timeData)
    xData = {'in': sindata}
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


def test_radau_event(dsargs):
    """
        Test Radau_ODEsystem with events involving external inputs.

        Robert Clewley, September 2006.
    """

    _run_checks(Radau_ODEsystem(dsargs))


def test_vode_event(dsargs):
    """
        Test Vode_ODEsystem with events involving external inputs.
    """

    _run_checks(Vode_ODEsystem(dsargs))


def test_euler_event(dsargs):
    """
        Test Euler_ODEsystem with events involving external inputs.
    """

    dsargs.algparams['init_step'] = 0.0001
    _run_checks(Euler_ODEsystem(dsargs))


def _run_checks(ode):

    traj = ode.compute('traj')

    assert ode.diagnostics.hasWarnings()
    assert ode.diagnostics.findWarnings(10) != []
    assert ode.diagnostics.findWarnings(20) != []

    assert_almost_equal(traj.indepdomain[1], 1.14417, 3)
    assert_almost_equal(traj.getEventTimes()['monitor'][0], 0.80267, 4)


def teardown_module():
    clean_files(['event_test'])
