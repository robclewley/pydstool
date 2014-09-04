#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest

from PyDSTool import PyDSTool_ValueError, PyDSTool_TypeError
from PyDSTool.Generator import Vode_ODEsystem


@pytest.fixture()
def ode():
    return Vode_ODEsystem({
        'name': 'ode',
        'vars': ['x'],
        'pars': {'p': 0.5},
        'varspecs': {'x': 'x+p'},
        'pdomain': {'p': [0,1]}
    })


def test_setting_invalid_key(ode):
    with pytest.raises(KeyError):
        ode.set(invalid_key='')


def test_setting_globalt0(ode):
    ode.set(globalt0=11.0)
    assert_almost_equal(11.0, ode.globalt0)
    assert ode._extInputsChanged


def test_setting_checklevel(ode):
    ode.set(checklevel=10)
    assert ode.checklevel == 10


def test_setting_abseps(ode):
    ode.set(abseps=0.001)
    assert_almost_equal(1e-3, ode._abseps)


def test_setting_ics(ode):
    ode.set(ics={'x': -1.0})
    assert_almost_equal(-1.0, ode.initialconditions['x'])


def test_setting_ics_raises_exception_for_illegal_varname(ode):
    with pytest.raises(ValueError):
        ode.set(ics={'y': 0.0})


def test_setting_tdata(ode):
    ode.set(tdata=[0, 10])
    assert_array_almost_equal([0, 10], ode.tdata)


def test_setting_tdomain(ode):
    ode.set(tdomain=[0, 20])
    assert_array_almost_equal([0, 20], ode.tdomain)


def test_setting_tdata_respects_domain(ode):
    ode.set(tdomain=[0, 20])
    ode.set(tdata=[-10, 30])
    assert_array_almost_equal([0, 20], ode.tdata)


def test_setting_xdomain(ode):
    ode.set(xdomain={'x': [0, 20]})
    assert_array_almost_equal([0, 20], ode.variables['x'].depdomain.get())


def test_setting_xdomain_using_single_value(ode):
    ode.set(xdomain={'x': 0})
    assert_array_almost_equal([0, 0], ode.variables['x'].depdomain.get())


def test_setting_xdomain_raises_exception_for_illegal_varname(ode):
    with pytest.raises(ValueError):
        ode.set(xdomain={'y': []})


def test_setting_xdomain_raises_exception_for_nondictionary_value(ode):
    with pytest.raises(AttributeError):
        ode.set(xdomain=('x', []))


def test_setting_xdomain_raises_exception_for_wrongly_sorted_values(ode):
    with pytest.raises(PyDSTool_ValueError):
        ode.set(xdomain={'x': [20, 0]})


def test_settting_xdomain_raises_exception_for_nonsequence_value(ode):
    with pytest.raises(PyDSTool_TypeError):
        ode.set(xdomain={'x': {}})



def test_setting_pdomain(ode):
    ode.set(pdomain={'p': [0, 20]})
    assert_array_almost_equal([0, 20], ode.parameterDomains['p'].get())


def test_setting_pdomain_using_single_value(ode):
    ode.set(pdomain={'p': 0})
    assert_array_almost_equal([0, 0], ode.parameterDomains['p'].get())


def test_setting_pdomain_raises_exception_for_illegal_parname(ode):
    with pytest.raises(ValueError):
        ode.set(pdomain={'q': []})


def test_setting_pdomain_raises_exception_for_nondictionary_value(ode):
    with pytest.raises(AttributeError):
        ode.set(pdomain=('p', []))


def test_setting_pdomain_raises_exception_for_wrongly_sorted_values(ode):
    with pytest.raises(PyDSTool_ValueError):
        ode.set(pdomain={'p': [20, 0]})


def test_settting_pdomain_raises_exception_for_nonsequence_value(ode):
    with pytest.raises(PyDSTool_TypeError):
        ode.set(pdomain={'p': {}})
