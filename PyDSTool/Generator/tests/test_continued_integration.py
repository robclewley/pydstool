#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import platform

from numpy.testing import assert_array_almost_equal
import pytest

from PyDSTool.Generator import (
    Dopri_ODEsystem,
    Radau_ODEsystem,
    Vode_ODEsystem,
)

from PyDSTool.Generator.tests.helpers import clean_files


def dsargs():
    """van der Pol equation"""
    pars = {'eps': 1.0, 'a': 0.5, 'y1': -0.708}

    return {
        'name': 'vanDerPol',
        'tdomain': [0, 200],
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


def test_vode():
    _check_continued_integration(Vode_ODEsystem)


@pytest.mark.skipif("platform.system() == 'FreeBSD' and '10.' in platform.release()")
def test_dopri():
    _check_continued_integration(Dopri_ODEsystem)


@pytest.mark.skipif("platform.system() == 'FreeBSD' and '10.' in platform.release()")
def test_radau():
    _check_continued_integration(Radau_ODEsystem)


def _check_continued_integration(generator):

    ode = generator(dsargs())

    # two step integration
    ode.set(tdata=[0, 100])
    left = ode.compute('first_half_traj')
    ode.set(tdata=[100, 200])
    right = ode.compute('second_half_traj', 'c')

    assert left(100.0) == right(100.0)

    # one step integration
    ode.set(tdata=[0, 200])
    full = ode.compute('full_traj')

    t = [0.0, 25.0, 50.0, 75.0, 100.0]
    assert_array_almost_equal(left(t, 'x').toarray(), full(t, 'x').toarray(), 4)
    assert_array_almost_equal(left(t, 'y').toarray(), full(t, 'y').toarray(), 4)

    t = [100.0, 125.0, 150.0, 175.0, 200.0]
    assert_array_almost_equal(right(t, 'x').toarray(), full(t, 'x').toarray(), 4)
    assert_array_almost_equal(right(t, 'y').toarray(), full(t, 'y').toarray(), 4)


def teardown_module():
    clean_files(['vanDerPol'])
