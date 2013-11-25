"""Testing numerical differentiation using diff.

   Robert Clewley, September 2005.
"""

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy import (
    abs,
    allclose,
    array,
    diag,
    ones,
    sqrt,
    zeros,
)

from numpy.random import random

from PyDSTool import (
    diff,
    Diff,
    Fun,
    Point,
    simplifyMatrixRepr,
    Var,
)


def test_diff_scalar():
    def f(x):
        """f: R->R"""
        return 3 * x * x

    def df(x):
        return 2 * 3 * x

    for i in range(0, 51):
        assert_almost_equal(diff(f, 0.1 * i), df(0.1 * i), 5)


def test_diff_scalar_on_vector_2D():
    def f(x):
        "f: R^2 -> R"
        return array([
            x[0] * x[1] + 2,
            x[1] * x[1]]
        )

    x0 = array([1., 3.])
    df = diff(f, x0, vars=[1])
    assert simplifyMatrixRepr(diff(f, x0, vars=[1], axes=[0])) == 1.0


def test_diff_scalar_on_vector_4D():

    def f(x):
        """f: R^4->R"""
        return sqrt(sum((x - array([1., 0., 1., 1.])) ** 2))

    # Compare the 1st order Taylor expansion of f2 at x0+dx = x0+[dx1,dx2,0,0]
    # in the first argument
    x0 = zeros(4)
    dx1 = 0.25
    dx2 = 0.25
    actual = f(x0 + [dx1, dx2, 0., 0.])
    approx = simplifyMatrixRepr(f(x0) + diff(f, x0, [0]) * dx1)
    assert actual - approx < 0.033
    assert actual - approx > 0.0324


def test_diff_vector_3D_simple():
    def f(x):
        "f: R^3->R^3"
        return x ** 2 - array([1., 0., 1.])

    def jac(x):
        return diag(2 * x)

    for i in range(10):
        x = i * random(3)
        assert_array_almost_equal(diff(f, x), jac(x), 4)


def test_diff_vector_3D_complex():
    def f(x):
        """f: R^3->R^3"""
        return array([
            x[0] * x[2],
            x[0] * 5,
            x[2] * 0.5
        ])

    def jac(x):
        return array([
            [x[2], 0.0, x[0]],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.5]
        ])

    for i in range(10):
        x = i * random(3)
        assert_array_almost_equal(diff(f, x), jac(x))
        assert_array_almost_equal(
            simplifyMatrixRepr(diff(f, x, axes=[0])), jac(x)[0])
        assert_array_almost_equal(
            simplifyMatrixRepr(diff(f, x, axes=[1])), jac(x)[1])
        assert_array_almost_equal(
            simplifyMatrixRepr(diff(f, x, axes=[2])), jac(x)[2])

    # Comparing 1st order Taylor series for nearby point to actual value:
    x0 = zeros(3)
    dx = array([0.2, 0., -.2])
    actual = f(x0 + dx)
    approx = f(x0) + simplifyMatrixRepr(diff(f, x0)).dot(dx)
    assert allclose((actual - approx), array([-0.04,  0.,  0.]))


def test_diff_point_3D():
    x0 = Var('x0')
    x1 = Var('x1')
    x2 = Var('x2')
    f5_x0 = Fun(x0 * x2, [x0, x1, x2], 'f5_x0')
    f5_x1 = Fun(x0 * 5, [x0, x1, x2], 'f5_x1')
    f5_x2 = Fun(x2 * 0.5, [x0, x1, x2], 'f5_x2')
    z0 = Point({
        'coordarray': [3., 2., 1.],
        'coordnames': ['x0', 'x1', 'x2']
    })
    # could also have defined F directly from f5_x[i] definitions
    F1 = Fun(
        [
            f5_x0(x0, x1, x2),
            f5_x1(x0, x1, x2),
            f5_x2(x0, x1, x2)
        ],
        [x0, x1, x2],
        'F'
    )
    F2 = [
        f5_x0(x0, x1, x2),
        f5_x1(x0, x1, x2),
        f5_x2(x0, x1, x2)
    ]
    assert Diff(F1, [x0, x1, x2]) == Diff(F2, [x0, x1, x2])
    assert_array_almost_equal(
        Diff(F1, [x0, x1, x2]).eval(z0).tonumeric(),
        array([[1.0, 0, 3.0], [5, 0, 0], [0, 0, 0.5]])
    )
    assert_array_almost_equal(
        Diff(F2, [x0, x1, x2]).eval(z0).tonumeric(),
        array([[1.0, 0, 3.0], [5, 0, 0], [0, 0, 0.5]])
    )
    def F3(z):
        return Point({
            'coorddict': {
                'x0': z('x0') * z('x2'),
                'x1': z('x0') * 5.0,
                'x2': z('x2') * 0.5
            }
        })
    assert_array_almost_equal(
        simplifyMatrixRepr(diff(F3, z0, axes=['x1', 'x2'])),
        Diff(F1, [x0, x1, x2]).eval(z0).tonumeric()[1:]
    )
    # Comparing 1st order Taylor series for nearby point to actual value:
    z1 = Point({
        'coordarray': array([3.1, 2., .94]),
        'coordnames': ['x0', 'x1', 'x2']
    })
    actual = F3(z1)
    approx = F3(z0) + simplifyMatrixRepr(diff(F3, z0)).dot(z1 - z0)
    assert all([err < 0.01 for err in abs(approx - actual)])
