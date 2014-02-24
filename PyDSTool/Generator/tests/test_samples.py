#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy import linspace, allclose
import platform
import pytest

from PyDSTool.Generator import (
    Euler_ODEsystem,
    Vode_ODEsystem,
    Radau_ODEsystem,
    Dopri_ODEsystem,
)

from PyDSTool.Generator.tests import samples
from PyDSTool.Generator.tests.helpers import clean_files


def test_euler():
    _check_generator(Euler_ODEsystem)


def test_vode():
    _check_generator(Vode_ODEsystem)


@pytest.mark.skipif("platform.system() == 'FreeBSD' and int(platform.release()[:2].replace('.', '')) >= 10")
def test_radau():
    _check_generator(Radau_ODEsystem)


@pytest.mark.skipif("platform.system() == 'FreeBSD' and int(platform.release()[:2].replace('.', '')) >= 10")
def test_dopri():
    _check_generator(Dopri_ODEsystem)


def _check_generator(generator):

    t = linspace(0.0, 1.0)
    problems = [samples.oscillator(t)]
    for dsargs, expected in problems:
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
