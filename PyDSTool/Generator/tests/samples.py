#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Sample systems of ODEs for testing purposes
"""

from __future__ import absolute_import, print_function

from numpy import array, cos, sin, sqrt


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
