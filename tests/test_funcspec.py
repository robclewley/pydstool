#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test FuncSpec for python and C right-hand sides.
"""

import os
import re

import pytest

from PyDSTool import (
    FuncSpec,
    RHSfuncSpec,
    PyDSTool_KeyError,
    args
)


def test_funcspec_raises_exception_if_there_are_invalid_keys():
    args = {
        'name': 'fs_with_invalid_key',
        'invalid_key': 'dummy',
        'myvars': ['a', 'b'],
    }

    with pytest.raises(PyDSTool_KeyError):
        FuncSpec(args)


def test_funcspec_uses_default_name_if_not_set():
    args = {
        'vars': ['x'],
        'varspecs': {'x': 'x + 1'},
    }

    fs = FuncSpec(args)
    assert 'untitled' == fs.name


def test_funcspec_raises_exception_if_vars_key_missed():
    with pytest.raises(PyDSTool_KeyError):
        FuncSpec({})

    with pytest.raises(PyDSTool_KeyError):
        FuncSpec({'name': 'test'})


def test_funcspec_raises_exception_if_both_varspecs_and_spec_key_missed():
    with pytest.raises(PyDSTool_KeyError):
        FuncSpec({'vars': ['x']})


def test_funcspec_raises_exception_if_both_varspecs_and_spec_keys_in_input():
    with pytest.raises(PyDSTool_KeyError):
        FuncSpec({'vars': [], 'spec': {}, 'varspecs': {}})


def test_funcspec_raises_exception_for_non_string_targetlang():
    with pytest.raises(TypeError):
        FuncSpec({'vars': [], 'varspecs': {}, 'targetlang': 1})


def test_funcspec_raises_exception_for_not_supported_langs():
    with pytest.raises(ValueError):
        FuncSpec({'vars': [], 'varspecs': {}, 'targetlang': 'ruby'})


def test_funcspec_uses_python_as_default_target():
    fs = FuncSpec({'vars': ['x'], 'varspecs': {'x': 'x + 1'}})
    assert 'python' == fs.targetlang


def test_funcspec_wraps_vars_string_to_list():
    fs = FuncSpec({'vars': 'x', 'varspecs': {'x': 'x + 1'}})
    assert ['x'] == fs.vars


def test_funcspec_makes_copy_of_input_vars_list():
    vars_ = ['x']
    fs = FuncSpec({'vars': vars_, 'varspecs': {'x': 'x + 1'}})
    assert id(fs.vars) != id(vars_)


def test_funcspec_raises_exception_if_vars_is_neither_str_nor_iterable():
    with pytest.raises(TypeError):
        FuncSpec({'vars': 1, 'varspecs': {}})


def test_funcspec_names_list_are_sorted():
    fs = FuncSpec({
        'vars': ['y', 'z', 'x'],
        'varspecs': {'x': 'y', 'y': 'z', 'z': 'x', 'aux2': 'x + y', 'aux11': 'z * x'},

        'auxvars': ['aux2', 'aux11'],
        'inputs': ['input_y', 'input_x'],
        'pars': ['k', 'o', 'v', 'j'],
    })
    assert sorted(fs.vars) == fs.vars
    assert sorted(fs.auxvars) == fs.auxvars
    assert sorted(fs.inputs) == fs.inputs
    assert sorted(fs.pars) == fs.pars


def test_funcspec_sum_macro_raises_value_error():
    with pytest.raises(ValueError):
        FuncSpec({
            'vars': 'z',
            'varspecs': {
                'z': 'sum(i, 1, 3, [i] / [i+1]**2)',
            },
        })


def test_dependencies():
    fs = FuncSpec({
        'vars': ['x', 'y', 'z'],
        'varspecs': {
            'x': 'y + myaux(x, z)',
            'y': 'z - input_y',
            'z': '1',
            'my': 'x + y + z',
        },
        'fnspecs': {
            'myaux': (['x', 'z'], 'x**2 + z^2'),
            'my': (['x', 'y'], 'x + y'),
        },
        'auxvars': ['my'],
        'inputs': ['input_y'],
        'targetlang': 'matlab',
    })

    assert ('x', 'y') in fs.dependencies
    assert ('x', 'x') in fs.dependencies
    assert ('x', 'z') in fs.dependencies
    assert ('y', 'z') in fs.dependencies
    assert ('y', 'input_y') in fs.dependencies
    assert ('my', 'x') in fs.dependencies
    assert ('my', 'y') in fs.dependencies
    assert ('my', 'z') in fs.dependencies
    assert all(v != 'z' for v, _ in fs.dependencies)


@pytest.fixture
def fsargs():
    fvarspecs = {
        'z[i]': 'for(i, 0, 1, 2*a+myauxfn1(2+t) - ' +
        'w*exp(-t)+ 3*z[i]**2 + z[i+1])',
        'z2': '2*a+myauxfn1(2+t) - w*exp(-t) + 3*z2**2 + z0',
        'w': 'k* w + a*sin(t)*itable + myauxfn1(t)*myauxfn2(w)/(a*sin(t))',
        'aux_wdouble': 'w*2 + globalindepvar(t)-0.5*sin(t)',
        'aux_other': 'myauxfn1(2*t) + initcond(w)',
        'aux_iftest': '1+if(z1>0,(1e-2+w), k+0.4)'
    }
    fnspecs = {
        'simpfn': ([''], '1+a'),
        'myauxfn1': (['t'], '2.5*cos(3*t)*simpfn()+sin(t)*2.e-3'),
        'myauxfn2': (['z0', 'w'], 'if(-z0<0,w/2+exp(-w),0)'),
        'myauxfn3': (['w'], 'k*w/2+ pow(w,-3.0+k)'),
        # not supposed to be the actual Jacobian!
        'Jacobian': (['t', 'z0', 'z1', 'z2', 'w'],
                     """[[2e5*z0+.1, -w/2, 0, 1.],
                     [-3.e-2 - a*w+z1, 1., z0+1, z0+sin(t)],
                     [k, w*z1+exp(-t), 0., z2/3+w],
                     [0, z1+a, 1., z0-w/3]]"""),
        # not supposed to be the actual Jacobian_pars!
        'Jacobian_pars': (['t', 'k', 'a'],
                          """[[0., 2.],
                          [z0, 1],
                          [3*w, 0.],
                          [2-z1, sin(t)*z2]]""")
    }
    return {
        'name': 'xfn',
        # vars is always unrolled by Gen base class if there are FOR loop macros
        'vars': ['w', 'z0', 'z1', 'z2'],
        'auxvars': ['aux_wdouble', 'aux_other', 'aux_iftest'],
        'pars': ['k', 'a'],
        'inputs': 'itable',
        'varspecs': fvarspecs,
        'fnspecs': fnspecs,
        # In practice, _for_macro_info created automatically by Generator base class
        '_for_macro_info': args(
            numfors=1,
            totforvars=2,
            varsbyforspec={
                'z[i]': ['z0', 'z1'],
                'w': ['w'],
                'z2': ['z2'],
                'aux_wdouble': ['aux_wdouble'],
                'aux_other': ['aux_other'],
                'aux_iftest': ['aux_iftest']
            }
        ),
        'reuseterms': {'a*sin_t': 'ast',
                       'exp(-t)': 'expmt',
                       'sin(t)': 'sin_t',
                       'myauxfn1(2+t)': 'afn1call'},
        'codeinsert_end': """    print('code inserted at end')""",
        'codeinsert_start': """    print('code inserted at start')""",
        'targetlang': 'python'
    }


def _compare_with_file(specstr, filename):

    spec = specstr.split('\n')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)) as f:
        for i, s in enumerate(f):
            s = s.replace('\n', '')
            if i != len(spec) - 1:
                assert s == spec[i], 'line %d: %r != %r' % (i + 1, s, spec[i])
            else:
                # special check for dependencies list
                extract_deps = lambda s: re.findall(r"\('\w+', '\w+'\)", s)
                assert set(extract_deps(s)) == set(extract_deps(spec[i]))


def test_funcspecs_python(fsargs):
    _compare_with_file(RHSfuncSpec(fsargs)._infostr(verbose=2), "funcspec_python.out")


def test_funcspec_recreate(fsargs):
    del fsargs['codeinsert_start']
    del fsargs['codeinsert_end']
    pyspec = RHSfuncSpec(fsargs)
    cspec_recreated = pyspec.recreate('c')

    fsargs['targetlang'] = 'c'
    cspec = RHSfuncSpec(fsargs)

    assert cspec._infostr(verbose=2) == cspec_recreated._infostr(verbose=2)


def test_funcspecs_c(fsargs):
    fsargs['targetlang'] = 'c'
    fsargs['codeinsert_start'] = "fprintf('code inserted at start\n')"
    fsargs['codeinsert_end'] = "fprintf('code inserted at end\n')"
    _compare_with_file(RHSfuncSpec(fsargs)._infostr(verbose=2), "funcspec_c.out")
