#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Tests for Python code generator

"""

from PyDSTool import FuncSpec
from PyDSTool.Generator import Vode_ODEsystem


def test_python_funcspec_for_ds_with_single_var():
    args = {
        'name': 'single_var',
        'vars': ['x'],
        'varspecs': {'x': 'x + 1'},
    }
    fs = FuncSpec(args)
    assert fs.spec == (
        '\n'.join([
            'def _specfn(ds, t, x, parsinps):',
            '    xnew0 = x[0] + 1 ',
            '    return array([xnew0])\n'
        ]),
        '_specfn'
    )


def test_python_funcspec_for_ds_with_two_vars():
    args = {
        'name': 'two_vars',
        'vars': ['x', 'y'],
        'varspecs': {
            'x': 'y + 1',
            'y': 'x - 1',
        },
    }

    fs = FuncSpec(args)
    assert fs.spec == (
        '\n'.join([
            'def _specfn(ds, t, x, parsinps):',
            '    xnew0 = x[1] + 1 ',
            '    xnew1 = x[0] - 1 ',
            '    return array([xnew0, xnew1])\n'
        ]),
        '_specfn'
    )


def test_python_funcspec_for_ds_with_single_var_and_single_param():
    args = {
        'name': 'fun_with_var_and_par',
        'vars': ['x'],
        'pars': ['p'],
        'varspecs': {'x': 'p * x - 1'},
        'fnspecs': {'myaux': (['x'], 'x**2 + p')},
    }
    fs = FuncSpec(args)
    assert fs.spec == (
        '\n'.join([
            'def _specfn(ds, t, x, parsinps):',
            '    xnew0 = parsinps[0] * x[0] - 1 ',
            '    return array([xnew0])\n'
        ]),
        '_specfn'
    )

    assert fs.auxfns['myaux'][0].split('\n') == [
        'def _auxfn_myaux(ds, parsinps, x):',
        '    return math.pow(x,2)+parsinps[0]'
    ]


def test_python_funcspec_with_jacobian_and_auxfunc():
    args = {
        'name': 'ds_with_jac',
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
        'def _specfn(ds, t, x, parsinps):',
        '    xnew0 = ds._auxfn_ydot0(parsinps, x[0],x[1],x[2])',
        '    xnew1 = -ds._auxfn_ydot0(parsinps, x[0],x[1],x[2])-ds._auxfn_ydot2(parsinps, x[0],x[1],x[2])',
        '    xnew2 = ds._auxfn_ydot2(parsinps, x[0],x[1],x[2])',
        '    return array([xnew0, xnew1, xnew2])',
        '',
    ]

    assert fs.auxfns['Jacobian'][0].split('\n') == [
        'def _auxfn_Jac(ds, t, x, parsinps):',
        '    xjac0 = [-0.04,1e4*x[2],1e4*x[1]] ',
        '    xjac1 = [0.04,-1e4*x[2]-6e7*x[1],-1e4*x[1]] ',
        '    xjac2 = [0.0,6e7*x[1],0.0] ',
        '    return array([xjac0, xjac1, xjac2])',
    ]

    assert fs.auxfns['ydot0'][0].split('\n') == [
        'def _auxfn_ydot0(ds, parsinps, y0, y1, y2):',
        '    return -0.04*y0 + 1e4*y1*y2'
    ]

    assert fs.auxfns['ydot2'][0].split('\n') == [
        'def _auxfn_ydot2(ds, parsinps, y0, y1, y2):',
        '    return 3e7*y1*y1'
    ]


def test_python_funcspec_for_ds_with_if_builtin():
    args = {
        'name': 'single_var',
        'vars': ['x'],
        'varspecs': {'x': 'if(x < 0, x, x**3)'},
    }
    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'def _specfn(ds, t, x, parsinps):',
        '    xnew0 = ds._auxfn_if(parsinps, x[0]<0,x[0],math.pow(x[0],3))',
        '    return array([xnew0])',
        ''
    ]


def test_python_funcspec_with_reuseterms():
    args = {
        'name': 'fs_with_reuseterms',
        'vars': ['x', 'y'],
        'varspecs': {'x': 'cy', 'y': 'sx'},
        'reuseterms': {'cos(y)': 'cy', 'sin(x)': 'sx'},
    }

    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'def _specfn(ds, t, x, parsinps):',
        '    cy = math.cos(x[1])',
        '    sx = math.sin(x[0])',
        '    xnew0 = cy ',
        '    xnew1 = sx ',
        '    return array([xnew0, xnew1])',
        ''
    ]


def test_python_funcspec_with_massmatrix():
    args = {
        'name': 'fs_with_massmatrix',
        'vars': ['x', 'y'],
        'varspecs': {'y': '-1', 'x': 'y - x * x'},
        'fnspecs': {'massMatrix': (['t', 'x', 'y'], '[[0,0],[0,1]]')},
    }
    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'def _specfn(ds, t, x, parsinps):',
        '    xnew0 = x[1] - x[0] * x[0] ',
        '    xnew1 = -1 ',
        '    return array([xnew0, xnew1])',
        '',
    ]

    assert fs.auxfns['massMatrix'][0].split('\n') == [
        'def _auxfn_massMatrix(ds, t, x, parsinps):',
        '    xmat0 = [0,0] ',
        '    xmat1 = [0,1] ',
        '    return array([xmat0, xmat1])'
        '',
    ]


def test_python_funcspec_with_for_loop():
    args = {
        'name': 'fs_with_loop',
        'varspecs': {'z[i]': 'for(i, 1, 6, t**[i]/2)'},
    }

    # XXX: FuncSpec doesn't support 'for' loop directly
    fs = Vode_ODEsystem(args).funcspec
    assert fs.spec[0].split('\n') == [
        'def _specfn(ds, t, x, parsinps):',
        '    xnew0 = t/2 ',
        '    xnew1 = math.pow(t,2)/2 ',
        '    xnew2 = math.pow(t,3)/2 ',
        '    xnew3 = math.pow(t,4)/2 ',
        '    xnew4 = math.pow(t,5)/2 ',
        '    xnew5 = math.pow(t,6)/2 ',
        '    return array([xnew0, xnew1, xnew2, xnew3, xnew4, xnew5])',
        '',
    ]


def test_python_funspec_ignoring_for_macro():
    fs = FuncSpec({
        'vars': 'z[i]',
        'varspecs': {
            'z[i]': 'for(i, 1, 3, z[i]**2)',
        },
    })

    assert fs.vars == ['z[i]']
    assert fs.spec[0].split('\n') == [
        'def _specfn(ds, t, x, parsinps):',
        '    return array([])',
        '',
    ]


def test_python_funcspec_has_python_user_auxfn_interface():
    args = {
        'name': 'test_user_auxfn_interface',
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


def test_python_funcspec_inserts_additional_code_in_vfield():
    start = "    print('START')"
    end = "    print('END')"
    args = {
        'name': 'test_codeinsert',
        'vars': ['x'],
        'pars': ['p'],
        'varspecs': {'x': 'p * x - 1'},
        'fnspecs': {'myaux': (['x'], 'x**2 + p')},
        'codeinsert_start': start,
        'codeinsert_end': end,
    }

    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'def _specfn(ds, t, x, parsinps):',
        start,
        '    xnew0 = parsinps[0] * x[0] - 1 ',
        end,
        '    return array([xnew0])',
        '',
    ]

    assert fs.auxfns['myaux'][0].split('\n') == [
        'def _auxfn_myaux(ds, parsinps, x):',
        '    return math.pow(x,2)+parsinps[0]'
    ]
