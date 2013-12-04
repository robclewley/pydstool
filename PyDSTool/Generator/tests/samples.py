#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Sample systems of ODEs for testing purposes
"""

from __future__ import absolute_import, print_function

from numpy import array, cos, sin, sqrt


def oscillator(t):
    """
    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)

    Taken from SciPy tests for scipy.integrate
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
