#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy import linspace
from numpy.testing import assert_array_almost_equal

from PyDSTool.Generator import (
    Euler_ODEsystem,
    Dopri_ODEsystem,
    Radau_ODEsystem,
    Vode_ODEsystem,
)

from .helpers import clean_files
from .samples import vanDerPol


def test_euler_vode():
    _cross_check_forward_integration(Euler_ODEsystem, Vode_ODEsystem)


def test_vode_radau():
    _cross_check_forward_integration(Vode_ODEsystem, Radau_ODEsystem)


def test_vode_dopri():
    _cross_check_forward_integration(Vode_ODEsystem, Dopri_ODEsystem)


def test_radau_dopri():
    _cross_check_forward_integration(Radau_ODEsystem, Dopri_ODEsystem)


def _cross_check_forward_integration(generator, other):

    dsargs, _ = vanDerPol()
    dsargs['name'] += '_forward'
    if Euler_ODEsystem in [generator, other]:
        dsargs['algparams']['init_step'] = 1e-3
        decimal = 2
    else:
        decimal = 4

    t = linspace(0.0, 20.0)
    dsargs['tdata'] = [t[0], t[-1]]

    left = generator(dsargs).compute(str(generator) + '_traj')
    right = other(dsargs).compute(str(other) + '_traj')
    print(str(generator))

    assert_array_almost_equal(left(t, 'x').toarray(), right(t, 'x').toarray(), decimal)
    assert_array_almost_equal(left(t, 'y').toarray(), right(t, 'y').toarray(), decimal)


def teardown_module():
    clean_files(['vanDerPol_forward'])
