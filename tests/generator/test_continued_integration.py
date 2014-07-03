#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy.testing import assert_array_almost_equal

from PyDSTool.Generator import (
    Dopri_ODEsystem,
    Radau_ODEsystem,
    Vode_ODEsystem,
)

from .helpers import clean_files
from .samples import vanDerPol


def test_vode():
    _check_continued_integration(Vode_ODEsystem)


def test_dopri():
    _check_continued_integration(Dopri_ODEsystem)


def test_radau():
    _check_continued_integration(Radau_ODEsystem)


def _check_continued_integration(generator):

    dsargs, _ = vanDerPol()
    dsargs['name'] += '_continued'
    ode = generator(dsargs)

    # two step integration
    ode.set(tdomain=[0, 200])
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
    clean_files(['vanDerPol_continued'])
