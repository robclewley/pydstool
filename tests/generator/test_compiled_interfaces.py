#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

""" Test compiled interfaces """


import pytest
from numpy import array, allclose

from PyDSTool import args
from PyDSTool.Generator import (
    Radau_ODEsystem,
)

from .helpers import clean_files


@pytest.fixture(scope="module")
def ode():
    """Dummy system with all functions set"""

    DSargs = args(
        name='test_interfaces',
        fnspecs={
            'Jacobian': (['t', 'y0', 'y1', 'y2'],
                         """[[-0.04,  1e4*y2       ,  1e4*y1 ],
                         [ 0.04, -1e4*y2-6e7*y1, -1e4*y1 ],
                         [ 0.0 ,  6e7*y1       ,  0.0    ]]"""),
            'Jacobian_pars': (['t', 'p1', 'p2', 'p3'],
                              "[[1, 0, p1], [-1, 0, p2], [0, 0, p3]]"),
            'ydot0': (['y0', 'y1', 'y2'], "-0.04*y0 + 1e4*y1*y2"),
            'ydot2': (['y0', 'y1', 'y2'], "3e7*y1*y1"),
            'massMatrix': (['t', 'y0', 'y1', 'y2'],
                         """[[-0.04,  1e4*y2       ,  1e4*y1 ],
                         [ 0.04, -1e4*y2-6e7*y1, -1e4*y1 ],
                         [ 0.0 ,  6e7*y1       ,  0.0    ]]"""),
        },
        varspecs={"y0": "ydot0(y0,y1,y2)",
                  "y2": "ydot2(y0,y1,y2)",
                  "y1": "-ydot0(y0,y1,y2)-ydot2(y0,y1,y2)",
                  'aux0': 'y0 + 2 * y1 - t',
                  'aux1': 'y2 - y1 - 2 * y0',
                  },
        auxvars=['aux0', 'aux1'],
        pars={'p1': 0.01, 'p2': 0.02, 'p3': 0.03},
        tdomain=[0., 1e20],
        ics={'y0': 1.0, 'y1': 0., 'y2': 0.},
        algparams={
            'init_step': 0.4,
            'rtol': 1e-4, 'atol': [1e-8, 1e-14, 1e-6]},
        checklevel=2,
    )

    return Radau_ODEsystem(DSargs)


def test_rhs(ode):
    assert allclose(
        ode.Rhs(0, {'y0': 1.0, 'y1': 0., 'y2': 0.}),
        array([-0.04, 0.04, 0]))


def test_auxvar(ode):
    assert allclose(
        ode.AuxVars(0, {'y0': 1.0, 'y1': 0., 'y2': 0.}),
        array([1.0, -2.]))


def test_jacobian(ode):
    assert allclose(
        ode.Jacobian(0, {'y0': 1.0, 'y1': 0., 'y2': 0.}),
        array([
            [-0.04, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]))


def test_jacobian_param(ode):
    assert allclose(
        ode.JacobianP(0, {'y0': 1.0, 'y1': 0., 'y2': 0.}),
        array([
            [1, 0, 0.01],
            [-1, 0, 0.02],
            [0, 0, 0.03],
        ]))


def test_mass_matrix(ode):
    assert allclose(
        ode.MassMatrix(0, {'y0': 1.0, 'y1': 0., 'y2': 0.}),
        array([
            [-0.04, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]))


def teardown_module():
    clean_files(['test_interfaces'])
