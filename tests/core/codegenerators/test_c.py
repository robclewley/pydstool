#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Tests for C code generator

"""

import pytest

from PyDSTool import FuncSpec
from PyDSTool.Generator import Dopri_ODEsystem


def test_c_funcspec_for_ds_with_single_var():
    args = {
        'name': 'single_var',
        'targetlang': 'c',
        'vars': ['x'],
        'varspecs': {'x': 'x + 1'},
    }
    fs = FuncSpec(args)
    assert fs.spec == (
        '\n'.join([
            'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
            '',
            'f_[0] = x+1;',
            '',
            '}',
            '\n',
        ]),
        'vfieldfunc'
    )


def test_c_funcspec_for_ds_with_two_vars():
    args = {
        'name': 'two_vars',
        'targetlang': 'c',
        'vars': ['x', 'y'],
        'varspecs': {
            'x': 'y + 1',
            'y': 'x - 1',
        },
    }

    fs = FuncSpec(args)
    assert fs.spec == (
        '\n'.join([
            'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
            '',
            'f_[0] = y+1;',
            'f_[1] = x-1;',
            '',
            '}',
            '\n',
        ]),
        'vfieldfunc'
    )


def test_c_funcspec_for_ds_with_single_var_and_single_param():
    args = {
        'name': 'fun_with_var_and_par',
        'targetlang': 'c',
        'vars': ['x'],
        'pars': ['p'],
        'varspecs': {'x': 'p * x - 1'},
        'fnspecs': {'myaux': (['x'], 'x**2 + p')},
    }
    fs = FuncSpec(args)
    assert fs.spec == (
        '\n'.join([
            'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
            '',
            'f_[0] = p*x-1;',
            '',
            '}',
            '\n',
        ]),
        'vfieldfunc'
    )

    assert fs.auxfns['myaux'][0].split('\n') == [
        'double myaux(double __x__, double *p_, double *wk_, double *xv_) {',
        '',
        '',
        'return pow(__x__,2)+p ;',
        '',
        '}'
    ]

    assert all(fn in fs._pyauxfns.keys() for fn in [
        'getbound',
        'getindex',
        'globalindepvar',
        'heav',
        'if',
        'initcond',
    ])

    assert 'myaux' in fs._pyauxfns.keys()


def test_c_funcspec_with_jacobian_and_auxfunc():
    args = {
        'name': 'ds_with_jac',
        'targetlang': 'c',
        'vars': ['y0', 'y1', 'y2'],
        'varspecs': {
            "y0": "ydot0(y0,y1,y2)",
            "y2": "ydot2(y0,y1,y2)",
            "y1": "-ydot0(y0,y1,y2)-ydot2(y0,y1,y2)"
        },
        'fnspecs': {
            'Jacobian': (
                ['t', 'y0', 'y1', 'y2'],
                """[[-0.04,  1e4*y2       ,  1e4*y1 ],
                [ 0.04, -1e4*y2-6e7*y1, -1e4*y1 ],
                [ 0.0 ,  6e7*y1       ,  0.0    ]]"""
            ),
            'ydot0': (['y0', 'y1', 'y2'], "-0.04*y0 + 1e4*y1*y2"),
            'ydot2': (['y0', 'y1', 'y2'], "3e7*y1*y1")
        },
    }

    fs = FuncSpec(args)

    assert fs.spec[0].split('\n') == [
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
        '',
        'f_[0] = ydot0(y0,y1,y2, p_, wk_, xv_);',
        'f_[1] = -ydot0(y0,y1,y2, p_, wk_, xv_)-ydot2(y0,y1,y2, p_, wk_, xv_);',
        'f_[2] = ydot2(y0,y1,y2, p_, wk_, xv_);',
        '',
        '}',
        '',
        ''
    ]

    assert fs.auxfns['Jacobian'][0].split('\n') == [
        'void jacobian(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {',
        '',
        '',
        'f_[0][0] = -0.04;',
        'f_[0][1] = 0.04;',
        'f_[0][2] = 0.0;',
        'f_[1][0] = 1e4*y2;',
        'f_[1][1] = -1e4*y2-6e7*y1;',
        'f_[1][2] = 6e7*y1;',
        'f_[2][0] = 1e4*y1;',
        'f_[2][1] = -1e4*y1;',
        'f_[2][2] = 0.0;',
        '',
        ' ;',
        '',
        '}'
    ]

    assert fs.auxfns['ydot0'][0].split('\n') == [
        'double ydot0(double __y0__, double __y1__, double __y2__, double *p_, double *wk_, double *xv_) {',
        '',
        '',
        'return -0.04*__y0__+1e4*__y1__*__y2__ ;',
        '',
        '}'
    ]

    assert fs.auxfns['ydot2'][0].split('\n') == [
        'double ydot2(double __y0__, double __y1__, double __y2__, double *p_, double *wk_, double *xv_) {',
        '',
        '',
        'return 3e7*__y1__*__y1__ ;',
        '',
        '}'
    ]


def test_c_funcspec_for_ds_with_if_builtin():
    args = {
        'name': 'single_var',
        'targetlang': 'c',
        'vars': ['x'],
        'varspecs': {'x': 'if(x < 0, x, x**3)'},
    }
    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
        '',
        'f_[0] = __rhs_if(x<0,x,pow(x,3), p_, wk_, xv_);',
        '',
        '}',
        '',
        ''
    ]


def test_c_funcspec_with_reuseterms():
    args = {
        'name': 'fs_with_reuseterms',
        'targetlang': 'c',
        'vars': ['x', 'y'],
        'varspecs': {'x': 'cy + sx', 'y': 'sx * cy'},
        'reuseterms': {'cos(y)': 'cy', 'sin(x)': 'sx', 'sx * cy': 'sc'},
    }

    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
        '/* reused term definitions */',
        'double cy = cos(y);',
        'double sx = sin(x);',
        'double sc = sx*cy;',
        '',
        'f_[0] = cy+sx;',
        'f_[1] = sc;',
        '',
        '}',
        '',
        ''
    ]


def test_c_funcspec_with_massmatrix():
    args = {
        'name': 'fs_with_massmatrix',
        'targetlang': 'c',
        'vars': ['x', 'y'],
        'varspecs': {'y': '-1', 'x': 'y - x * x'},
        'fnspecs': {'massMatrix': (['t', 'x', 'y'], '[[0,0],[0,1]]')},
    }
    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
        '',
        'f_[0] = y-x*x;',
        'f_[1] = -1;',
        '',
        '}',
        '',
        '',
    ]

    assert fs.auxfns['massMatrix'][0].split('\n') == [
        'void massMatrix(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {',
        '',
        '',
        'f_[0][0] = 0;',
        'f_[0][1] = 0;',
        'f_[1][0] = 0;',
        'f_[1][1] = 1;',
        '',
        ' ;',
        '',
        '}',
    ]


def test_c_funcspec_with_loop():
    args = {
        'name': 'fs_with_loop',
        'varspecs': {
            'z[i]': 'for(i, 1, 6, t + [i]/2)',  # FIXME: using 't**[i]' or 't^[i]' here results in RuntimeError
        },
        'nobuild': True,
    }

    # XXX: FuncSpec doesn't support 'for' loop directly
    fs = Dopri_ODEsystem(args).funcspec
    assert fs.spec[0].split('\n') == [
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
        '',
        'f_[0] = t+1/2;',
        'f_[1] = t+2/2;',
        'f_[2] = t+3/2;',
        'f_[3] = t+4/2;',
        'f_[4] = t+5/2;',
        'f_[5] = t+6/2;',
        '',
        '}',
        '',
        '',
    ]


def test_c_funspec_for_macro_raises_exception():
    with pytest.raises(ValueError):
        FuncSpec({
            'vars': ['z1', 'z2', 'z3'],
            'targetlang': 'c',
            'varspecs': {
                'z[i]': 'for(i, 1, 3, z[i]**2)',
            },
        })


def test_c_funcspec_has_python_user_auxfn_interface():
    args = {
        'name': 'test_user_auxfn_interface',
        'targetlang': 'c',
        'vars': ['x'],
        'pars': ['p'],
        'varspecs': {'x': 'p * x - 1'},
        'fnspecs': {'myaux': (['x'], 'x**2 + p')},
    }
    fs = FuncSpec(args)

    assert fs._user_auxfn_interface['myaux'].split('\n') == [
        'def myaux(self,x,__parsinps__=None):',
        '\tif __parsinps__ is None:',
        '\t\t__parsinps__=self.map_ixs(self.genref)',
        '\treturn self.genref._auxfn_myaux(__parsinps__,x)',
        ''
    ]


def test_c_funcspec_inserts_additional_code_in_vfield():
    start = r'fprintf("START\n");'
    end = r'fprintf("END\n");'
    args = {
        'name': 'test_codeinsert',
        'vars': ['x'],
        'pars': ['p'],
        'varspecs': {'x': 'p * x - 1'},
        'fnspecs': {'myaux': (['x'], 'x**2 + p')},
        'codeinsert_start': start,
        'codeinsert_end': end,
        'targetlang': 'c',
    }

    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
        '/* Verbose code insert -- begin */',
        start,
        '/* Verbose code insert -- end */',
        '',
        '',
        'f_[0] = p*x-1;',
        '',
        '/* Verbose code insert -- begin */',
        end,
        '/* Verbose code insert -- end */',
        '',
        '}',
        '',
        '',
    ]

    assert fs.auxfns['myaux'][0].split('\n') == [
        'double myaux(double __x__, double *p_, double *wk_, double *xv_) {',
        '',
        '',
        'return pow(__x__,2)+p ;',
        '',
        '}'
    ]


def test_c_funspec_with_powers():
    """Regression test for https://github.com/robclewley/pydstool/issues/90"""
    args = {
        'name': 'fs_with_powers',
        'targetlang': 'c',
        'vars': ['x', 'y'],
        'varspecs': {'y': 'x', 'x': 'y + x**3 - x^4'},
    }
    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){',
        '',
        'f_[0] = y+pow(x,3)-pow(x,4);',
        'f_[1] = x;',
        '',
        '}',
        '',
        '',
    ]
