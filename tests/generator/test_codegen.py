#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import os
from tempfile import mkstemp

import pytest

from .samples import vanDerPol, dae

from PyDSTool import Events
from PyDSTool.Generator import (
    Dopri_ODEsystem,
    Radau_ODEsystem,
)


@pytest.fixture
def ode():
    dsargs, _ = vanDerPol()
    dsargs['nobuild'] = True
    return Radau_ODEsystem(dsargs)


@pytest.fixture
def tmpfile():
    _, fname = mkstemp()
    return fname


def test_user_module_code(ode, tmpfile):
    ode.makeLibSource(fname=tmpfile)
    refname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_module.out")
    with open(refname) as f:
        ref = f.readlines()
        with open(tmpfile) as g:
            for i, s in enumerate(g):
                assert s == ref[i], 'line %d: %r != %r' % (i + 1, s, ref[i])


def test_adding_more_includes(ode, tmpfile):
    ode.makeLibSource(fname=tmpfile, include=['myheader.h'])
    with open(tmpfile) as g:
        assert '#include "myheader.h"\n' in g.readlines()


def test_warning_about_duplicated_include(ode, tmpfile, capsys):
    ode.makeLibSource(fname=tmpfile, include=['Python.h'])
    out, _ = capsys.readouterr()
    assert 'Warning: library \'Python.h\' already appears in list' in out


def test_dopri_does_not_support_mass_matrix(tmpfile):
    dsargs = dae()
    dsargs['nobuild'] = True
    with pytest.raises(ValueError):
        ode = Dopri_ODEsystem(dsargs)
        ode.makeLibSource(fname=tmpfile)


def test_radau_support_mass_matrix(tmpfile):
    dsargs = dae()
    dsargs['nobuild'] = True
    ode = Radau_ODEsystem(dsargs)
    ode.makeLibSource(fname=tmpfile)
    mm_src = '\n'.join([
        'f_[0][0] = 0;',
        'f_[0][1] = 0;',
        'f_[1][0] = 0;',
        'f_[1][1] = 1;'
    ])

    with open(tmpfile) as g:
        assert mm_src in g.read()


def test_correct_integrator_name_in_generated_file(ode, tmpfile):
    ode.makeLibSource(fname=tmpfile)
    with open(tmpfile) as g:
        assert 'for Radau integrator' in next(g)


def test_correct_dopri_integrator_name_in_generated_file(tmpfile):
    dsargs, _ = vanDerPol()
    dsargs['nobuild'] = True
    ode =  Dopri_ODEsystem(dsargs)
    ode.makeLibSource(fname=tmpfile)
    with open(tmpfile) as f:
        assert 'for Dopri853 integrator' in next(f)


def test_events_code_generating(tmpfile):
    dsargs, _ = vanDerPol()
    dsargs['nobuild'] = True
    ev_args = {
        'name': 'monitor',
        'eventtol': 1e-4,
        'eventdelay': 1e-5,
        'starttime': 0,
        'active': True,
        'term': False,
        'precise': True
    }
    ev = Events.makeZeroCrossEvent('y', 0, ev_args, ['y'], targetlang='c')
    dsargs['events'] = [ev]

    ode = Radau_ODEsystem(dsargs)
    ode.makeLibSource(fname=tmpfile)

    with open(tmpfile) as g:
        code = g.read()
        assert 'double monitor(unsigned n_, double t, double *Y_, double *p_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_);' in code
        assert 'int N_EVENTS = 1;' in code
        assert 'void assignEvents(EvFunType *events){\n events[0] = &monitor;\n\n}' in code
        assert '\n'.join([
        'double monitor(unsigned n_, double t, double *Y_, double *p_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {',
        'return  y; ',
        '}']) in code
