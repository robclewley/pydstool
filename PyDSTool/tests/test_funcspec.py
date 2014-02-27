"""Test FuncSpec for python, Matlab and C right-hand sides.
"""

import platform

import pytest
from PyDSTool import (
    FuncSpec,
    RHSfuncSpec,
    wrapArgInCall,
    addArgToCalls,
    PyDSTool_KeyError,
)
from PyDSTool.Generator import Vode_ODEsystem, Dopri_ODEsystem
from PyDSTool.parseUtils import proper_match


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


def test_funcspec_raises_exception_if_both_varspecs_and_spec_key_missed():
    with pytest.raises(PyDSTool_KeyError):
        FuncSpec({'vars': ['x']})


def test_funspec_raises_exception_for_non_string_targetlang():
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
        _ = FuncSpec({'vars': 1, 'varspecs': {}})


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
        'vars': ['w', 'z[i]', 'z2'],
        'auxvars': ['aux_wdouble', 'aux_other', 'aux_iftest'],
        'pars': ['k', 'a'],
        'inputs': 'itable',
        'varspecs': fvarspecs,
        'fnspecs': fnspecs,
        'reuseterms': {'a*sin_t': 'ast',
                       'exp(-t)': 'expmt',
                       'sin(t)': 'sin_t',
                       'myauxfn1(2+t)': 'afn1call'},
        'codeinsert_end': """    print 'code inserted at end'""",
        'codeinsert_start': """    print 'code inserted at start'""",
        'targetlang': 'python'
    }


@pytest.mark.xfail(reason="FIXME: fails with ValueError")
def test_funcspecs_python(fsargs):
    DSfuncspec = RHSfuncSpec(fsargs)
    print DSfuncspec._infostr(verbose=2)
    print "\nDSfuncspec.auxfns['Jacobian'] =>", DSfuncspec.auxfns['Jacobian'], "\n"


@pytest.mark.xfail(reason="FIXME: fails with ValueError")
def test_funcspecs_c(fsargs):
    DSfuncspec = RHSfuncSpec(fsargs)
    fsargs['targetlang'] = 'c'
    fsargs['codeinsert_start'] = "fprintf('code inserted at start\n')"
    fsargs['codeinsert_end'] = "fprintf('code inserted at end\n')"
    del fsargs['codeinsert_start']
    del fsargs['codeinsert_end']
    DSfuncspec_C = RHSfuncSpec(fsargs)
    print '\nC version of same specifications:\n', DSfuncspec_C._infostr(verbose=2)
    print "Dependencies are not calculated for C code in FuncSpec. If you use ModelSpec"
    print "to generate your models you will have that information there."
    print "\nTesting re-targetting of funcspec using 'recreate' method..."
    dsc = DSfuncspec.recreate('c')
    assert dsc.spec[0] == DSfuncspec_C.spec[0], " - FAILED"
    print " - PASSED.\n"
    print "\n============================================================="
    print "Test: wrapping delimiters around call arguments"
    print """  ... around first argument: wrapArgInCall(fs, 'initcond', '"')"""
    fs = 'initcond(v,p)'
    print fs, " -- wrapped to: ", wrapArgInCall(fs, 'initcond', '"'), "\n"
    print """  ... around second argument: wrapArgInCall(fs,'initcond','"',argnums=[1])"""
    print fs, " -- wrapped to: ", wrapArgInCall(fs, 'initcond', '"', argnums=[1]), "\n"
    print """  ... extra braces to both arguments: wrapArgInCall(fs,'initcond','[',']',[0,1])"""
    print fs, " -- wrapped to: ", wrapArgInCall(fs, 'initcond', '[', ']', [0, 1]), "\n"
    print "\nTest combo of addArgToCalls and wrapArgInCall with embedded calls:"
    fs2 = "1/zeta(y_rel(y,initcond(y)+initcond(z)),z-initcond(x))+zeta(0.)"
    print fs2, "\n"
    fs2_p = addArgToCalls(fs2, ['zeta', 'y_rel', 'initcond', 'nothing'], "p")
    print " ** becomes **"
    print fs2_p, "\n"
    fs2_p = wrapArgInCall(fs2_p, 'initcond', '"')
    print " ** becomes **"
    print fs2_p
    s = '1 +abc13 + abc'
    assert proper_match(s, 'abc')
    assert not proper_match(s[:10], 'abc')


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
        '    sx = math.sin(x[0])',
        '    cy = math.cos(x[1])',
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
        'void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){', '', 'f_[0] = ydot0(y0,y1,y2, p_, wk_, xv_);',
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
        'f_[0] = __rhs_if(x<0,x,x**3, p_, wk_, xv_);',
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
        'double sx = sin(x);',
        'double cy = cos(y);',
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


@pytest.mark.skipif("platform.system() == 'FreeBSD' and int(platform.release()[:2].replace('.', '')) >= 10")
def test_c_funcspec_with_loop():
    args = {
        'name': 'fs_with_loop',
        'varspecs': {
            'z[i]': 'for(i, 1, 6, t + [i]/2)',  # FIXME: using 't**[i]' or 't^[i]' here results in RuntimeError
        },
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

    from PyDSTool.Generator.tests.helpers import clean_files
    clean_files(['fs_with_loop'])


def test_matlab_funcspec_for_ds_with_single_var_and_single_param():
    args = {
        'name': 'fun_with_var_and_par',
        'targetlang': 'matlab',
        'vars': ['x'],
        'pars': ['p'],
        'varspecs': {'x': 'p * x - 1'},
        'fnspecs': {'myaux': (['x'], 'x**2 + p')},
    }
    fs = FuncSpec(args)
    assert fs.spec[0].split('\n') == [
        'function [vf_, y_] = vfield(vf_, t_, x_, p_)',
        '% Vector field definition for model fun_with_var_and_par',
        '% Generated by PyDSTool for ADMC++ target',
        '',
        '',
        '% Parameter definitions',
        '',
        '\tp = p_(1);',
        '',
        '% Variable definitions',
        '',
        '\tx = x_(1);',
        '',
        '',
        'y_(1) = p * x - 1;',
        '',
        '',
        '',
    ]

    assert fs.auxfns['myaux'][0].split('\n') == [
        'function y_ = myaux(x__,  p_)',
        '% Auxilliary function myaux for model fun_with_var_and_par',
        '% Generated by PyDSTool for ADMC++ target',
        '',
        '',
        '% Parameter definitions',
        '',
        '\tp = p_(1);',
        ' ',
        '',
        '',
        'y_ = x__^2+p;',
        '',
        ''
    ]
