#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy.testing import assert_allclose
import pytest

from PyDSTool.Generator import (
    Dopri_ODEsystem,
    Radau_ODEsystem,
    Vode_ODEsystem,
)

from .helpers import clean_files
from .samples import vanDerPol


def test_vode():
    with pytest.raises(NotImplementedError):
        _check_backward_integration(Vode_ODEsystem)


def test_dopri():
    _check_backward_integration(Dopri_ODEsystem)


def test_radau():
    _check_backward_integration(Radau_ODEsystem)


def _check_backward_integration(generator):

    dsargs, _ = vanDerPol()
    dsargs['name'] += '_backward'
    ode = generator(dsargs)

    # forward integration
    ode.set(tdomain=[0, 20])
    ode.set(tdata=[0, 20])
    forward = ode.compute('fwd')

    # backward integration
    ode.set(tdata=[0, 20], ics=forward(20.0).todict())
    backward = ode.compute('bwd', 'b')

    t = [0.0, 5.0, 10.0, 15.0, 20.0]
    for v in ode.variables:
        assert_allclose(
            forward(t, v).toarray(),
            backward(t, v).toarray(),
            rtol=1e-7,
            atol=0 if generator != Radau_ODEsystem else 1e-2
        )


def teardown_module():
    clean_files(['vanDerPol_backward'])
