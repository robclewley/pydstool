#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
    Simple forward integration test for ODE generators

    Comparing numerical results with exact solution

    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)

    Taken from SciPy tests for scipy.integrate
"""


from __future__ import absolute_import, print_function

from numpy import linspace, allclose, array, cos, sin, sqrt

from PyDSTool.Generator import (
    Euler_ODEsystem,
    Vode_ODEsystem,
    Radau_ODEsystem,
    Dopri_ODEsystem,
)

from .helpers import clean_files


def oscillator(t):
    """
        DS description for generators
    """

    k = 4.0
    m = 1.0
    z0 = array([1.0, 0.1], float)

    dsargs = {
        'tdomain': [t[0], t[-1]],
        'pars': {'k': k, 'm': m},
        'algparams': {'max_step': 0.01},
        'name': 'Oscillator',
        'varspecs': {
            'x': 'xdot',
            'xdot': '- k / m * x',
        },
        'ics': {'x': z0[0], 'xdot': z0[1]},
    }

    # Expected result
    omega = sqrt(k / m)
    u = z0[0] * cos(omega * t) + z0[1] * sin(omega * t) / omega

    return dsargs, u


def test_euler():
    _check_generator(Euler_ODEsystem)


def test_vode():
    _check_generator(Vode_ODEsystem)


def test_radau():
    _check_generator(Radau_ODEsystem)


def test_dopri():
    _check_generator(Dopri_ODEsystem)


def _check_generator(generator):

    t = linspace(0.0, 1.0)
    dsargs, expected = oscillator(t)
    if generator is Euler_ODEsystem:
        dsargs['algparams']['init_step'] = 1e-3
        atol = 1e-2
    else:
        atol = 1e-4
    ode = generator(dsargs)

    assert ode.pars == dsargs['pars']
    assert (not ode.defined)

    traj = ode.compute('traj')

    assert ode.defined
    assert allclose(expected, traj(t)['x'], atol=atol, rtol=1e-5)


def teardown_module():
    clean_files(['Oscillator'])
