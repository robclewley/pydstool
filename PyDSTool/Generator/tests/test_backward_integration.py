#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import platform

from numpy.testing import assert_allclose
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
        'tdomain': [0, 20],
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
    with pytest.raises(NotImplementedError):
        _check_backward_integration(Vode_ODEsystem)


@pytest.mark.skipif("platform.system() == 'FreeBSD' and '10.' in platform.release()")
def test_dopri():
    _check_backward_integration(Dopri_ODEsystem)


@pytest.mark.skipif("platform.system() == 'FreeBSD' and '10.' in platform.release()")
def test_radau():
    _check_backward_integration(Radau_ODEsystem)


def _check_backward_integration(generator):

    ode = generator(dsargs())

    # forward integration
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
    clean_files(['vanDerPol'])
