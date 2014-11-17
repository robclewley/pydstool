#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Sample systems of ODEs for testing purposes
"""

from __future__ import absolute_import, print_function


def vanDerPol():
    """van der Pol equation"""
    pars = {'eps': 1.0, 'a': 0.5}

    dsargs = {
        'name': 'vanDerPol',
        'pars': pars,
        'varspecs': {
            'x': '(y - (x * x * x / 3 - x)) / eps',
            'y': 'a - x',
        },
        'ics': {
            'x': pars['a'],
            'y': pars['a'] - pow(pars['a'], 3) / 3,
        },
        'algparams': {'max_step': 1e-2, 'max_pts': 30000}
    }

    # TODO: add expected result

    return dsargs, None


def dae():
    """System with mass-matrix, taken from examples/DAE_example.py"""

    dsargs = {}
    dsargs['name'] = "DAE_test"
    dsargs['pars'] = {}
    dsargs['vars'] = ['x', 'y']
    dsargs['varspecs'] = {'y': '-1', 'x': 'y - x*x'}
    dsargs['algparams'] = {
        'init_step': 0.05,
        'refine': 0,
        'max_step': 0.1,
        'rtol': 1e-4,
        'atol': 1e-4
    }
    dsargs['checklevel'] = 1
    dsargs['ics'] = {'y': 4, 'x': 2}

    # 0 in the (x,x) entry of the mass matrix indicates that this is the
    # algebraic equation ( 0 . dx/dt = y - x*x )
    #
    # 1 in the (y,y) entry indicates that the 'y' varspec is a regular
    # differential equation.
    #
    # 0 in the (x,y) and (y,x) entries just says that there's no interaction
    # between the equations apart from what's explicitly given in the right-hand
    # sides.
    dsargs['fnspecs'] = {'massMatrix': (['t', 'x', 'y'], '[[0, 0],[0, 1]]')}

    return dsargs
