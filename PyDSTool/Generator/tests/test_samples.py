#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import linspace, allclose

from PyDSTool.Generator import (
    Vode_ODEsystem,
    Radau_ODEsystem,
    Dopri_ODEsystem,
)

import samples


def test_vode():
    _check_generator(Vode_ODEsystem)


def test_radau():
    _check_generator(Radau_ODEsystem)


def test_dopri():
    _check_generator(Dopri_ODEsystem)


def _check_generator(generator):

    t = linspace(0.0, 1.0)
    problems = [samples.oscillator(t)]
    for dsargs, expected in problems:
        ode = generator(dsargs)

        assert ode.pars == dsargs['pars']
        assert (not ode.defined)

        traj = ode.compute('traj')

        assert ode.defined
        assert allclose(expected, traj(t)['x'], atol=1e-4, rtol=1e-5)
