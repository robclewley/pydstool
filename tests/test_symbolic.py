#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import (
    exp,
    cosh,
    sqrt,
    e,
    pi,
)
from numpy.random import rand
try:
    from numpy.testing import assert_approx_equal
except ImportError: 
    # For backwards compatibility
    from numpy.testing.utils import assert_approx_equal
from copy import copy

from PyDSTool.parseUtils import (
    doneg,
    dosub,
    replaceCallsWithDummies,
)

from PyDSTool import (
    args,
    Component,
    convertPowers,
    expr2fun,
    Fun,
    info,
    LeafComponent,
    Par,
    QuantSpec,
    Sin,
    targetLangs,
    Var,
)

import pytest


class myLeaf1(LeafComponent):
    compatibleGens=('AGenerator',)
    targetLangs=targetLangs

class myLeaf2(LeafComponent):
    compatibleGens=('AGenerator',)
    targetLangs=targetLangs

class myNode(Component):
    compatibleGens=('AGenerator','BGenerator')
    targetLangs=targetLangs


def test_symbolic():
    assert doneg('-x-y') == 'x+y'
    assert doneg('(-x-y)') == '(x+y)'
    assert doneg('-(-x-y)') == '(-x-y)'
    assert dosub('1', '-x-y') == '(1+x+y)'

    g2 = expr2fun('1-max([0., -a+b*x])', **{'a': 3, 'b': 1.5})
    assert g2._args == ['x']
    assert g2(1) == 1.0
    assert g2(10) == -11.0

    ds = {'a': 3, 'bbb': 1}
    f=expr2fun('1+ds["a"]')
    assert f._args == ['ds']
    assert f(ds) == 4
    f2=expr2fun('1+ds["a"]')
    assert f2(**{'ds':ds}) == 4
    assert f2._args == ['ds']
    g=expr2fun('1+ds["bbb"]', ds=ds)
    assert g() == 2
    # g must be dynamic and not based on static eval of ds on initialization
    ds['bbb'] = 2
    assert g._args == []
    assert g() == 3

    m = args(pars=copy(ds))
    h = expr2fun('m.pars["a"]+c', m=m, c=1)
    assert h() == 4
    assert h._args == []
    h2 = expr2fun('1 + m.pars["a"]/2.', m=m)
    assert h2() == 2.5
    assert h2._args == []

    def func(x, y):
        return x * (y+1)

    m.func = func
    i = expr2fun('1+func(x,y)+b', func=m.func, b=0.5)
    assert 1+func(2,3)+0.5 == i(2,3)

    j = expr2fun('i(x,func(2,y))*2', i=i, func=m.func)
    assert j(1,0) == 9

    fnspec = {'f': (['x','y'], 'x+1+2*y-a')}
    # a is expected to be in scope like a FuncSpec parameter
    # so can't use the above method of providing explicit functions
    k = expr2fun('-f(c,d)+b', f=fnspec['f'], b=0.5, a=1)
    assert k(1,2) == -4.5

    # alternate scope and testing embedded dynamic pars in both
    # main expression and sub-expressions like f()
    l = expr2fun('-f(c,d)+b', ensure_dynamic={'a':1, 'c':2},
                 **fnspec)
    assert l._args == ['d', 'b']
    assert l._call_spec == "-self.f(self._pardict['c'],d)+b"
    assert l(1,2) == -2

    s='1+a/(f(x,y)-3)+h(2)'
    t=s.replace('y','g(x,z)')
    u=t.replace('z','f(z)')
    r1, d1 = replaceCallsWithDummies(s, ['f','g','h'])
    r2, d2 = replaceCallsWithDummies(t, ['f','g','h'])
    r3, d3 = replaceCallsWithDummies(u, ['f','g','h'])

    assert r1 == '1+a/(__dummy1__-3)+__dummy2__'
    assert len(d1) == 2
    assert r2 == '1+a/(__dummy2__-3)+__dummy3__'
    assert len(d2) == 3
    assert r2 == '1+a/(__dummy2__-3)+__dummy3__'
    assert len(d3) == 4

    ps = 'abs((HB9_fs_Vq-HB9_fs_V)*(-((HB9_fs_Lk_g*(HB9_fs_V-HB9_fs_Lk_vrev))+(-HB9_fs_Iapp_Ibias)+((HB9_fs_Na_g*(1.0/(1.0+exp((HB9_fs_V-HB9_fs_Na_theta_m)/HB9_fs_Na_k_m)))*(1.0/(1.0+exp((HB9_fs_V-HB9_fs_Na_theta_m)/HB9_fs_Na_k_m)))*(1.0/(1.0+exp((HB9_fs_V-HB9_fs_Na_theta_m)/HB9_fs_Na_k_m)))*(1-HB9_fs_K_n))*(HB9_fs_V-HB9_fs_Na_vrev))+(HB9_fs_K_g*HB9_fs_K_n*HB9_fs_K_n*HB9_fs_K_n*HB9_fs_K_n*(HB9_fs_V-HB9_fs_K_vrev))+(HB9_fs_isyn_g*(HB9_fs_V-HB9_fs_isyn_vrev))+(HB9_fs_esyn_g*(HB9_fs_V-HB9_fs_esyn_vrev)))/HB9_fs_C)+(HB9_fs_Knq-HB9_fs_K_n)*(((1.0/(1.0+exp((HB9_fs_V-HB9_fs_K_theta_n)/HB9_fs_K_k_n)))-HB9_fs_K_n)/(HB9_fs_K_taun_bar/cosh((HB9_fs_V-HB9_fs_K_theta_n)/(2*HB9_fs_K_k_n)))))/(sqrt(HB9_fs_Vq*HB9_fs_Vq+HB9_fs_Knq*HB9_fs_Knq)+sqrt(pow((-((HB9_fs_Lk_g*(HB9_fs_V-HB9_fs_Lk_vrev))+(-HB9_fs_Iapp_Ibias)+((HB9_fs_Na_g*(1.0/(1.0+exp((HB9_fs_V-HB9_fs_Na_theta_m)/HB9_fs_Na_k_m)))*(1.0/(1.0+exp((HB9_fs_V-HB9_fs_Na_theta_m)/HB9_fs_Na_k_m)))*(1.0/(1.0+exp((HB9_fs_V-HB9_fs_Na_theta_m)/HB9_fs_Na_k_m)))*(1-HB9_fs_K_n))*(HB9_fs_V-HB9_fs_Na_vrev))+(HB9_fs_K_g*HB9_fs_K_n*HB9_fs_K_n*HB9_fs_K_n*HB9_fs_K_n*(HB9_fs_V-HB9_fs_K_vrev))+(HB9_fs_isyn_g*(HB9_fs_V-HB9_fs_isyn_vrev))+(HB9_fs_esyn_g*(HB9_fs_V-HB9_fs_esyn_vrev)))/HB9_fs_C),2)+pow((((1.0/(1.0+exp((HB9_fs_V-HB9_fs_K_theta_n)/HB9_fs_K_k_n)))-HB9_fs_K_n)/(HB9_fs_K_taun_bar/cosh((HB9_fs_V-HB9_fs_K_theta_n)/(2*HB9_fs_K_k_n)))),2)))'
    parnames = ['HB9_fs_Vq', 'HB9_fs_Lk_g', 'HB9_fs_Lk_vrev', 'HB9_fs_Iapp_Ibias', 'HB9_fs_Na_g', 'HB9_fs_Na_theta_m', 'HB9_fs_Na_k_m', 'HB9_fs_Na_vrev', 'HB9_fs_K_g', 'HB9_fs_K_vrev', 'HB9_fs_isyn_g', 'HB9_fs_isyn_vrev', 'HB9_fs_esyn_g', 'HB9_fs_esyn_vrev', 'HB9_fs_C', 'HB9_fs_Knq', 'HB9_fs_K_theta_n', 'HB9_fs_K_k_n', 'HB9_fs_K_taun_bar']
    varnames = ['HB9_fs_V', 'HB9_fs_K_n']
    ps2 = convertPowers(ps, 'pow')
    for n in parnames+varnames:
        v=rand(1)[0]+1e-5
        ps2=ps2.replace(n,str(v))
        ps=ps.replace(n,str(v))
    eps=eval(ps)
    eps2=eval(ps2)
    assert eps==eps2

    a = Par('3.5', 'a')
    qa = Var(['a*3', 'b'], 'array_test')
    assert str(qa.eval(a=1)) == '[3,b]'
    # explicit exporting 'a' to globals to make this work as expected
    globals()['a'] = a
    assert str(qa.eval()) == '[10.5,b]'

    testq = QuantSpec('d', 'a')
    testq.simplify()
    assert testq() == 'a'
    assert str(testq.eval(a=3)) == '3'
    q = QuantSpec('q', 'zeta(yrel(y,initcond(y)),z)-1')
    print(q.eval({}))
    assert 'initcond' in str(q.eval({}))
    q2=QuantSpec('q','Exp(-spikeTable+b)/k')
    assert 'spikeTable' in q2.freeSymbols

    #    x = Var('x')
    #    print x.isDefined()
    #    xs = QuantSpec('x', '1-rel - 2*x + cos(z) + 2e10', 'RHSfuncSpec')
    #    x.bindSpec(xs)
    #    print x.isDefined(),"\n"

    x = Var(QuantSpec('x', '1-rel - 2*x + cos(z) + 2e10', 'RHSfuncSpec'))
    p = Par('p')
    az = Var(QuantSpec('z', 'myfunc(0,z)+abs(x+1)', 'RHSfuncSpec'))
    w = Var('x-1/w[i]', 'w[i,0,1]', specType='RHSfuncSpec')



    myLeaf1.compatibleContainers=(myNode,)
    myLeaf2.compatibleContainers=(myNode,)
    myNode.compatibleSubcomponents=(myLeaf1,myLeaf2)

    c = myLeaf1('leaf1')

    assert c.isDefined() == False
    c.add(x)
    print(c.freeSymbols, c.isDefined())
    c.add(az)
    print(c.freeSymbols, c.isDefined())
    c.add(w)
    print(c.freeSymbols, c.isDefined())

    c.compileFuncSpec()
    print(c.funcSpecDict)

    empty_fn = Fun('1+exp(1)', [], 'dumb_fn')
    print(empty_fn())

    q = Par('qpar')
    y = Var(QuantSpec('rel', 'v+p'), domain=[0,1])
    g = Fun(QuantSpec('qfunc', '-1.e-05+sin(qpar)*(10.e-5-xtol)'), ['xtol'])
    d = myLeaf2('leaf2')
    ##    d.add(y)
    q_dummy = Var(QuantSpec('q_notpar', '-2+sin(30)'))
    g_dummy = Fun(QuantSpec('qfunc_dummy', 'sin(q_notpar)*(10.e-5-xtol)'), ['xtol'])
    d.add([q_dummy, g_dummy])  # will delete these later
    d.add([q,g])

    d2 = myLeaf2('leaf3')
    d2.add([q,g])

    v = Var(QuantSpec('v', 'v * myfunc(rel,v) - sin(p)*t', 'RHSfuncSpec'))
    # p is a global parameter so this is ok in v
    f = Fun(QuantSpec('myfunc', '2.0+s-t+exp(p)'), ['s','t'])
    # t is just a local argument here, so it won't clash with its
    # occurrence in v (which we'll see is declared as a global
    # when we call flattenSpec()).
    ipar = Par('ipar')
    z = Var('z[i]+v/(i*ipar)', 'z[i,0,5]', specType='RHSfuncSpec')
    a = myNode('sys1')

    a.add([f,p,y])
    print(a.isDefined(True))
    a.add(c)
    print(a.freeSymbols, a.isDefined(), a.isComplete())
    a.add(d)
    print(a.freeSymbols, a.isDefined(), a.isComplete())
    a.add(d2)
    print(a.freeSymbols, a.isDefined(), a.isComplete())
    a.add(v)
    print("Added v")
    print(a.freeSymbols, a.isDefined(), a.isComplete())
    print("Removed v")
    a.remove(v)
    print(a.freeSymbols, a.isDefined(), a.isComplete())
    a.add([z,ipar])
    print(a.freeSymbols, a.isDefined(), a.isComplete())
    print("\na._registry -->  ")
    print(a._registry)
    print("Re-added v")
    a.add(v)
    print(a.freeSymbols, a.isDefined(), a.isComplete())
    print("\nv in a -->", v in a)

    print("\n")
    with pytest.raises(TypeError):
        a.compileFuncSpec()
    a.remove(['leaf2.qfunc_dummy', 'leaf2.q_notpar'])

    print("---------  sys1: funcSpecDict ---------------------")
    a.compileFuncSpec()
    info(a.funcSpecDict)

    print("\n\n-------------  Flatten spec with unravelling\n")
    print("\n\ninfo(a.flattenSpec()) --> \n")
    info(a.flattenSpec(globalRefs=['t']), "Model specification")
    print("\n\n-------------  Flatten spec with no unravelling\n")
    print("\n\ninfo(a.flattenSpec(False, globalRefs=['t'])) --> \n")
    info(a.flattenSpec(False, globalRefs=['t']), "Model specification")

    print("\n\nDemos for functions (results are strings):\n")
    h = f(p, -x)
    z = QuantSpec('zero','0')
    print("h = f(p, -x) --> ", h)
    print("z = QuantSpec('zero','0') --> ", z)
    print("f(g(3)*1,h) --> ", f(g(3)*1,h))
    print("f(g(p),h) --> ", f(g(p),h))
    print("f(g(p),0*h) --> ", f(g(p),0*h))
    print("f(g(x),h+z) --> ", f(g(x),h+z))
    # e is the math constant, but it doesn't evaluate to a float!
    print("f(g(x()),(e+h)/2) --> ", f(g(x()),(e+h)/2))
    print("f(g(x()),-h) --> ", f(g(x()),-h))
    print("f(g(x()),.5-h+0) --> ", f(g(x()),.5-h+0))
    print("Sin(pi+q) --> ", Sin(pi+q))
    qsin=QuantSpec('qsin','zv-sin(beta)')
    assert str(qsin.eval()) == 'zv-sin(beta)'

    print("\n\nDemos for local scope evaluation and **:\n")
    print("q=Var('xv+1','qv')")
    print("x=Var('3','xv')")
    q=Var('xv+1','qv')
    x=Var('3','xv')
    globals()['x'] = x
    globals()['q'] = q
    sc1 = str(q.eval()) == '4'
    print("q.eval() == 4? ", sc1)
    assert sc1
    print("a=x/q")
    a=x/q
    sc2 = str(a) == 'xv/qv'
    print("a == xv/qv? ", sc2)
    assert sc2
    sc3 = str(a.eval())=='0.75'
    print("a.eval() == 0.75? ", sc3)
    assert sc3
    sc4 = str(a.eval(xv=5))=='5/qv'
    print("a.eval(xv=5) == 5/q? ", sc4)
    assert sc4
    sc5 = (str(a.eval(xv=5,qv=q())),'0.83333333333333337')
    assert_approx_equal(*sc5)
    print("assert_approx_equal(%s,%s)" % sc5)
    sc6 = (str(a.eval({'xv': 10, 'qv': q()})),'0.90909090909090906')
    print("assert_approx_equal(%s,%s)" % sc6)
    assert_approx_equal(*sc6)

    print("qs=QuantSpec('qsv','xsv+1')")
    print("xs=QuantSpec('xsv','3')")
    qs=QuantSpec('qsv','xsv+1')
    xs=QuantSpec('xsv','3')
    globals()['qs'] = qs
    globals()['xs'] = xs
    qse = qs.eval()
    qt1 = str(qse) == '4'
    print("qs.eval() == 4? ", qt1)
    assert qt1
    assert qse.tonumeric() == 4
    print("asq = xs/qs")
    asq=xs/qs
    qt2 = str(asq) == '3/(xsv+1)'
    print("asq == 3/(xsv+1)? ", qt2)
    assert qt2
    qt3 = str(asq.eval()) == '0.75'
    print("as.eval() == 0.75? ", qt3)
    assert qt3
    ps = asq**xs
    print("ps = as**xs")
    qt4 = str(ps) == 'Pow(3/(xsv+1),3)'
    print("ps == Pow(3/(xsv+1),3)? ", qt4)
    assert qt4
    qt5 = str(ps.eval()) == str(0.75**3)
    print("ps.eval() == 0.421875? ", qt5)
    assert qt5

    print("sq=QuantSpec('sv','sin(xsv)')")
    print("s2q=QuantSpec('s2v','Sin(xv)')")
    sq=QuantSpec('sv','sin(xsv)')
    s2q=QuantSpec('s2v','Sin(xv)')
    print("sq.eval() --> ", sq.eval())
    print("s2q.eval() --> ", s2q.eval())
    assert sq.eval().tonumeric() == s2q.eval().tonumeric()
    assert sq[:] == ['sin','(','xsv',')']

    print("\n\nDemos for multiple quantity definitions:\n")
    mp=QuantSpec('p','a + 3*z[4*i-2]')
    m=Var(mp, 'z[i,2,5]', specType='RHSfuncSpec')
    v=Var('3*z[i-1]+z4-i', 'z[i,1,5]', specType='RHSfuncSpec')
    print("mp=QuantSpec('p','a + 3*z[4*i-2]')")
    print("m=Var(mp, 'z[i,2,5]', specType='RHSfuncSpec')")
    print("v=Var('3*z[i-1]+z4-i', 'z[i,1,5]', specType='RHSfuncSpec')")
    print("v[3] -->", v[3])
    assert str(v[3])=='z3'
    print("v.freeSymbols -->", v.freeSymbols)
    assert v.freeSymbols == ['z0']
    print("\nModelSpec a already contains 'z0', which was defined as part of")
    print("a multiple quantity definition, so check that attempting to add")
    print("v to a results in an error ...")
    with pytest.raises(AttributeError):
        a.add(v)
    print("\nTest of eval method, e.g. on a function f(s,t)...")
    print("f.eval(s='1', t='t_val') -->", f.eval(s='1', t='t_val'))
    print("f.eval(s=1, t='t_val', p=0.5) -->", f.eval(s=1, t='t_val', p=0.5))
    print("\nTesting convertPowers():")
    cp_tests = ["phi1dot^m3", "1+phi1dot^m3*s",
                "phi1dot**m3", "1+phi1dot**m3*s",
                "sin(x^3)**4", "(2/3)^2.5", "3^cos(x)-pi",
                "3^(cos(x)-pi)", "2^(sin(y**p))"]
    for spec in cp_tests:
        print(spec, " --> ", convertPowers(spec))


    globals().pop('a')
    qc=QuantSpec('t', "a+coot+b/'coot'")
    assert str(qc.eval()) == 'a+coot+b/"coot"'
    coot=QuantSpec('coot', "1.05")
    globals()['coot'] = coot
    assert str(qc.eval()) == 'a+1.05+b/"coot"'

    print("\nTest of function calling with argument names that clash with")
    print("bound names inside the function.")
    x0=Var('x0')
    x1=Var('x1')
    x2=Var('x2')
    F=Fun([x0*x2,x0*5,x2**0.5], [x0,x1,x2], 'F')
    print("F=Fun([x0*x2,x0*5,x2**0.5], [x0,x1,x2], 'F')")
    print("F(3,2,Sin(x0))) = [3*Sin(x0),15,Pow(Sin(x0),0.5)] ...")
    print("  ... even though x0 is a bound name inside definition of F")
    assert str(F(3,2,Sin(x0)))=='[3*Sin(x0),15,Pow(Sin(x0),0.5)]'

    # moved from test_symbolic_diff.py
    p0 = Var('p0')
    p1 = Var('p1')

    pv = Var([p0, p1], 'p')
    assert str(pv()) == '[p0,p1]'
    assert str(pv.eval()) == '[p0,p1]'

    u = Var('Pi/(2*Sin(Pi*t/2))', 'u')
    assert u.eval(t=1).tonumeric() == pi / 2
