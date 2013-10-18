"""Test FuncSpec for python and C right-hand sides.
"""

import pytest
from PyDSTool import (
    RHSfuncSpec,
    wrapArgInCall,
    addArgToCalls,
)
from PyDSTool.parseUtils import proper_match


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
