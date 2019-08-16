"""Tests on symbolic differentiation using ModelSpec Quantity objects.

Robert Clewley, September 2005.

"""
from __future__ import print_function

from math import pi

from numpy import ndarray
from numpy.linalg import norm
try:
    from numpy.testing import assert_approx_equal, assert_allclose, assert_almost_equal
except ImportError:
    # For backwards compatibility
    from numpy.testing.utils import assert_approx_equal, assert_allclose, assert_almost_equal

from PyDSTool import Diff, DiffStr, Var, Pow, QuantSpec, Par, Fun, Exp, Sin, expr2fun, remain
from PyDSTool.Toolbox.phaseplane import prepJacobian


def test_diffstr():
    assert DiffStr('x-(4*x*y)/(1+x*x)', 'x') == \
        '1-4*(y)/(1+x*x)+(4*x*y)*2*x*pow((1+x*x),-2)'


def test_symbolic_diff():
    """Showing the variety of ways that symbolic Diff() can be used."""

    x = Var('x')
    y = Var('y')
    xx = QuantSpec('dummy', 'x')
    function_variants = ('[-3*x**2+2*(x+y),-y/2]', ['-3*x**2+2*(x+y)', '-y/2'],
                         [-3 * Pow(x, 2) + 2 * (x + y), -y / 2])
    for f in function_variants:
        for v in ('x', x, xx):
            assert str(Diff(f, v)) == '[-6*x+2,0]'

    for f in function_variants:
        for v in (['x', 'y'], [x, y], [xx, y]):
            assert str(Diff(f, v)) == '[[-6*x+2,2],[0,-0.5]]'


def test_complex():
    t = Var('t')
    s = Var('s')

    assert str(Diff(Pow((t * 5), 2), t)) != '0'

    # XXX: doesn't work without global
    global p
    p = Par('3.', 'p')
    f = Fun(QuantSpec('f', str(2.0 + s - 10 * (t ** 2) + Exp(p))), ['s', 't'])
    assert str(2 * f.eval(s=3, t=t)) == '2*((2.0+3)-10*Pow(t,2)+Exp(p))'
    assert str(Diff('-10*Pow(t,2)', 't')) == '-20*t'
    assert str(Diff(2 * f.eval(s=3, t=t), t)) == '2*-20*t'
    assert str(Diff(3 + t * f.eval(s=3,
                                   t=t),
                    t)) == '((2.0+3)-10*Pow(t,2)+Exp(p))+t*(-10*2*t)'
    assert str(Diff(3 + t * f(s, t),
                    t).eval(s=3,
                            t=1,
                            p=p)) == '((2.0+3)-10*Pow(1,2)+Exp(p))+1*(-10*2*1)'
    # FIXME: segmentation fault!
    # assert_almost_equal(Diff(3 + t * f(s, t), t).eval(s=3,
    #                                                   t=1,
    #                                                   p=p()),
    #                     -4.914463076812332)
    assert Diff(str(f(s, t)), 't') == Diff(f(s, t), t)
    q1 = Diff(f(s, t), t)
    q2 = Diff(str(f(s, t)), t)
    assert q1 == q2

    assert str(Diff(f(t, s), t)) == '1'
    assert str(Diff(2 * f(3, t * 5), t)) == '2*-100*t*5'
    assert str(Diff(2 * f(3, t * 5), t)) != str(0)
    assert f(s, t) != f(t, s)

    assert str(f(s, t).eval()) == '(2.0+s)-10*Pow(t,2)+Exp(3.0)'
    q = f(s, t)
    assert str(q.eval()) == '(2.0+s)-10*Pow(t,2)+Exp(3.0)'

    assert str(Diff('g(s)', s)) == 'g_0(s)'
    assert str(Diff('g(s)', s).eval()) == 'g_0(s)'
    # XXX: doesn't work without global
    global dg_dt
    dg_dt = Fun(QuantSpec('g_0', '2-Sin(t/2)'), ['t'])
    assert str(Diff('g(t)', t).eval()) == '2-Sin(t/2)'
    assert str(Diff('g(s)', s)) == 'g_0(s)'
    assert str(Diff('g(s)', s).eval()) == '2-Sin(s/2)'

    g = Fun('', [t], 'g')  # declare empty function
    assert str(g(t)) == 'g(t)'
    assert str(Diff(g(s), s).eval()) == '2-Sin(s/2)'

    assert eval(str(Diff('pow(1,2)*t', 't'))) == 1
    assert eval(str(Diff(Pow(1, 2) * t, t))) == 1
    assert str(Diff(Sin(Pow(t, 1)), t)) == 'Cos(t)'

    q = QuantSpec('q', '-0+3+pow(g(x)*h(y,x),1)*1')
    assert str(Diff(q, 'x')) == '(g_0(x)*h(y,x)+g(x)*h_1(y,x))'
    # BROKEN in this version (need to move to SymPy)
    # print Diff(q,'x').eval()
    # assert str(Diff(q,'x').eval()) == '(2-Sin(x/2))*h(y,x)+g(x)*h_1(y,x)'


def test_diff_functions():
    funcs = ['x', 'a*x', 'a', 'a(x)', 'a(x)*x', 'a(x)*b(x)', 'a+x', 'a(x)+x',
             'a(x)+b(x)+c(x)', 'a(x)*b(x)*c(x)', 'a(x)*(b(x)+c(x))',
             '(a(x)+b(x))*c(x)', 'a(x)/b(x)', '(a(x))**b(x)', 'x**n', 'x**5',
             'x**-5', 'sin(x)', 'cos(x)', 'exp(x)', 'ln(x)', 'log(x)',
             'log10(x)', 'asin(x)', 'sinh(x)']
    derivs = ['1', 'a', '0', 'a_0(x)', 'a_0(x)*x+a(x)',
              'a_0(x)*b(x)+a(x)*b_0(x)', '1', 'a_0(x)+1',
              'a_0(x)+b_0(x)+c_0(x)',
              'a_0(x)*b(x)*c(x)+a(x)*(b_0(x)*c(x)+b(x)*c_0(x))',
              'a_0(x)*(b(x)+c(x))+a(x)*(b_0(x)+c_0(x))',
              '(a_0(x)+b_0(x))*c(x)+(a(x)+b(x))*c_0(x)',
              'a_0(x)/b(x)-a(x)*b_0(x)*Pow(b(x),-2)',
              '(Log(a(x))*b_0(x)+a_0(x)*b(x)/(a(x)))*Pow(a(x),b(x))',
              'n*Pow(x,n-1)', '5*Pow(x,4)', '-5*Pow(x,-6)', 'Cos(x)',
              '-Sin(x)', 'Exp(x)', '1.0/x', '1.0/x', 'Log(10)/x',
              '1.0/Sqrt(1-Pow(x,2))', 'Cosh(x)']
    for f, d in zip(funcs, derivs):
        assert str(Diff(f, 'x')) == d


def test_diff_integer_powers():
    funcs = ['x**-1', '-Pow(x,-2)', '2*Pow(x,-3)']
    for f, d in zip(funcs[:-1], funcs[1:]):
        assert str(Diff(f, 'x')) == d


def test_diff_float_powers():
    f, d = 'pow(x,-1.5)', '-1.5*Pow(x,-2.5)'
    assert str(Diff(f, 'x')) == d


def test_symbolic_vector():
    # XXX: doesn't work without global
    global q0, q1
    p0 = Var('p0')
    q0 = Var(p0 + 3, 'q0')
    q1 = Var(Diff(1 + Sin(Pow(p0, 3) + q0), p0), 'q1')

    qv = Var([q0, q1], 'q')
    assert str(qv()) == '[q0,q1]'
    assert str(qv.eval()) == '[(p0+3),(3*Pow(p0,2)*Cos(Pow(p0,3)+(p0+3)))]'

    v = Var('v')
    w = Var('w')
    f = Var([-3 * Pow((2 * v + 1), 3) + 2 * (w + v), -w / 2], 'f')

    df = Diff(f, [v, w])
    assert str(df) == '[[-3*6*Pow((2*v+1),2)+2,2],[0,-0.5]]'
    dfe = df.eval(v=3, w=10).tonumeric()
    assert_allclose(dfe, [[-880.0, 2.0], [0.0, -0.5]])
    assert isinstance(dfe, ndarray)
    assert isinstance(df.fromvector(), list)

    y0 = Var('y0')
    y1 = Var('y1')
    y2 = Var('y2')
    t = Var('t')

    ydot0 = Fun(-0.04 * y0 + 1e4 * y1 * y2, [y0, y1, y2], 'ydot0')
    ydot2 = Fun(3e7 * y1 * y1, [y0, y1, y2], 'ydot2')
    ydot1 = Fun(-ydot0(y0, y1, y2) - ydot2(y0, y1, y2), [y0, y1, y2], 'ydot1')

    F = Fun([ydot0(y0, y1, y2), ydot1(y0, y1, y2), ydot2(y0, y1, y2)],
            [y0, y1, y2], 'F')
    assert F.dim == 3
    DF = Diff(F, [y0, y1, y2])
    DF0, DF1, DF2 = DF.fromvector()
    assert_approx_equal(DF0.fromvector()[0].tonumeric(), -0.04)
    # str(Diff(F,[y0,y1,y2])) should be (to within numerical rounding errors):
    # '[[-0.04,10000*y2,10000*y1],[0.040000000000000001,(-10000*y2)-30000000*2*y1,-10000*y1],[0,30000000*2*y1,0]]')

    jac = Fun(Diff(F, [y0, y1, y2]), [t, y0, y1, y2], 'Jacobian')
    assert jac(t, 0.1, y0 + 1, 0.5).eval(y0=0) == jac(t, 0.1, 1 + y0,
                                                      0.5).eval(y0=0)
    assert jac(t, 0.1, y0, 0.5) == jac(t, 0.1, 0 + y0, 0.5)

    x = Var('x')
    y = Var('y')

    f1 = Fun([-3 * x ** 3 + 2 * (x + y), -y / 2], [x, y], 'f1')
    f2 = ['-3*x**3+2*(x+y)', '-y/2']
    f3 = [-3 * x ** 3. + 2 * (x + y), -y / 2.]
    assert str(f1) == 'f1'
    assert str(f2) == '[\'-3*x**3+2*(x+y)\', \'-y/2\']'
    assert str(
        f3) == '[QuantSpec __result__ (ExpFuncSpec), QuantSpec __result__ (ExpFuncSpec)]'

    f4 = [-3 * Pow((2 * x + 1), 3) + 2 * (x + y), -y / 2]
    xx = QuantSpec('dummy', 'x')
    f5 = Var([-3 * Pow((2 * x + 1), 3) + 2 * (x + y), -y / 2], 'f5')

    assert Diff(f1, x) == Diff(f1, 'x')
    assert str(Diff(f1, x)) == '[-3*3*Pow(x,2)+2,0]'
    assert str(Diff(f3, x)) == '[-3*3*Pow(x,2)+2,0]'
    assert str(Diff(f3, xx)) == '[-3*3*Pow(x,2)+2,0]'
    assert str(Diff(f4, x)) == '[-3*6*Pow((2*x+1),2)+2,0]'
    assert str(Diff(f4, xx)) == '[-3*6*Pow((2*x+1),2)+2,0]'

    # Examples of Jacobian Diff(f, [x,y])...
    assert Diff(f1, [x, y]) == Diff(f1, ['x', 'y']) == Diff(f1(x, y), [x, y])
    assert str(Diff(f2, ['x', 'y'])) == '[[-3*3*Pow(x,2)+2,2],[0,-0.5]]'
    assert str(Diff(f3, ['x', 'y'])) == '[[-3*3*Pow(x,2)+2,2],[0,-0.5]]'
    assert str(Diff(f1, [xx, y])) == '[[-3*3*Pow(x,2)+2,2],[0,-0.5]]'
    assert str(Diff(f1, [xx, 'y'])) == '[[-3*3*Pow(x,2)+2,2],[0,-0.5]]'
    assert str(Diff(f2, [x, y])) == '[[-3*3*Pow(x,2)+2,2],[0,-0.5]]'
    assert str(Diff(f3, [x, y])) == '[[-3*3*Pow(x,2)+2,2],[0,-0.5]]'
    assert str(Diff(f4, [x, y])) == '[[-3*6*Pow((2*x+1),2)+2,2],[0,-0.5]]'
    df5 = Diff(f5, [x, y])
    assert str(df5) == '[[-3*6*Pow((2*x+1),2)+2,2],[0,-0.5]]'
    assert_allclose(df5.eval(x=3,
                             y=10).tonumeric(), [[-880.0, 2.0], [0.0, -0.5]])
    # FIXME: segmentation fault!
    # assert_allclose(df5.eval(x=3,y=10).fromvector(0), [-880.0,2.0])
    assert str(df5.eval(x=3, y=10).fromvector(0)) == '[-880.0,2]'
    assert str(df5.fromvector(0)) == '[-3*6*Pow((2*x+1),2)+2,2]'
    assert isinstance(df5.fromvector(), list)
    a = df5.fromvector(0).eval(x=3, y=10).tonumeric()
    b = df5.eval(x=3, y=10).tonumeric()[0]
    assert a[0] == b[0] and a[1] == b[1]


def test_nested_functions_diff():
    """Examples of differentiation using nested functions.

    This functionality is built in to Symbolic.prepJacobian

    """

    func_ma_spec = (['p', 'v'], '0.32*(v+54)/(1-exp(-p*(v+54)/4))')
    ma = Fun(func_ma_spec[1], func_ma_spec[0], 'ma')
    ma_1 = Fun(Diff(ma, 'v'), ['v'], 'ma_1')
    func_mb_spec = (['v'], '0.28*(v+27)/(exp((v+27)/5)-1)')
    mb = Fun(func_mb_spec[1], func_ma_spec[0], 'mb')
    mb_0 = Fun(Diff(mb, 'v'), ['v'], 'mb_0')
    func_ma_1_spec = (['p', 'v'], str(ma_1.spec))
    func_mb_0_spec = (['v'], str(mb_0.spec))

    # artificial example to introduce time dependence
    rhs_m = 'exp(-2*t)*(100-v) + ma(1, v)*(1-m)-mb(v)*m'
    jac_part = Diff(rhs_m, ['v', 'm'])

    f = expr2fun(jac_part,
                 ma_1=func_ma_1_spec,
                 mb_0=func_mb_0_spec,
                 ma=func_ma_spec,
                 mb=func_mb_spec)
    assert remain(f._args, ['t', 'v', 'm']) == []
    assert abs(norm(f(0, -50, 0.2)) - 8.565615) < 1e-6
    # alternative usage syntax
    assert abs(norm(f(**{'t': 0, 'v': -50, 'm': 0.2})) - 8.565615) < 1e-6


def test_prep_Jacobian():
    vf1 = Fun(['-3*x+y*y', '-y/2'], ('x', 'y'), name='f')
    vf2 = Fun(['-3*x+y*y+sin(t)', '-y/2+x*cos(t)'], ('t', 'x', 'y'), name='f')

    J1, fs1 = prepJacobian(vf1, ('x','y'))
    J2, fs2 = prepJacobian(vf2, ('y','x')) # order of x, y don't matter

    assert str(J1) == '[[-3,2*y],[0,-0.5]]'
    assert str(J2) == '[[-3,2*y],[Cos(t),-0.5]]'
