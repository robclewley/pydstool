"""Testing numerical differentiation using diff.

   Robert Clewley, September 2005.
"""

from PyDSTool import *
from PyDSTool import _num_types
from numpy import all

print "Testing numerical differentiation using diff..."

# Test scalar f

def f1(x):
    if isinstance(x, _num_types):
        return 3*x*x
    else:
        print type(x)
        raise TypeError

def f2(x):
    assert type(x) == ndarray and len(x) == 4
    return sqrt(sum((x-array([1.,0.,1.,1.]))**2))

def f2_2(x):
    return array([x[0]*x[1]+2, x[1]*x[1]])

# f1: R -> R
df1 = diff(f1, 0.5)
# df1 should be 3
assert df1 - 3 < 1e7

# "f2: R^4 -> R"
x0 = array([0., 0., 0., 0.])
df2 = diff(f2, x0)
dx1 = 0.25
dx2 = 0.25
# Compare the 1st order Taylor expansion of f2 at x0+dx = x0+[dx1,dx2,0,0] in the first argument
approx = simplifyMatrixRepr(f2(x0) + diff(f2, x0, vars=[0])*dx1)
actual = f2(array([dx1,dx2,0.,0.])+x0)
assert actual - approx < 0.033
assert actual - approx > 0.0324

x0 = array([1.,3.])
df2_2 = diff(f2_2, x0, vars=[1])
assert simplifyMatrixRepr(diff(f2_2, x0, vars=[1], axes=[0])) == 1.0

# 3-dimensional f3
def f3(y):
    assert len(y) == 3
    return y**2-array([1.,0.,1.])
y0 = array([3., 2., 1.])
df3 = diff(f3, y0)
#print simplifyMatrixRepr(df3)

# Test vector f

# f4 : R^3 -> R^3
def f4(y):
    assert len(y) == 3
    return array([y[0]*y[2], y[0]*5, y[2]*0.5])

df4 = diff(f4, y0, axes=[1])
assert (simplifyMatrixRepr(df4) == array([ 5.,  0.,  0.])).all()
dy = array([0.2, 0., -.2])

# Comparing 1st order Taylor series for nearby point to actual value:
actual = simplifyMatrixRepr(f4(y0+dy))
approx = simplifyMatrixRepr(mat(f4(y0)) + transpose(diff(f4, y0)*transpose(mat(dy))))
assert numpy.allclose( (actual - approx), array([-0.04,  0.  ,  0.  ]) )

# Test Point f

x0=Var('x0')
x1=Var('x1')
x2=Var('x2')
f5_x0 = Fun(x0*x2, [x0,x1,x2], 'f5_x0')
f5_x1 = Fun(x0*5, [x0,x1,x2], 'f5_x1')
f5_x2 = Fun(x2**0.5, [x0,x1,x2], 'f5_x2')
y0pt = Point({'coordarray': y0, 'coordnames': ['x0', 'x1', 'x2']})
y1pt = Point({'coordarray': array([3.1, 2., .94]), 'coordnames': ['x0', 'x1', 'x2']})

# could also have defined F directly from f5_x[i] definitions
F=Fun([f5_x0(x0,x1,x2),f5_x1(x0,x1,x2),f5_x2(x0,x1,x2)],[x0,x1,x2], 'F')

# f5: R^3 -> R^3 defined as
assert (Diff([f5_x0(x0,x1,x2),f5_x1(x0,x1,x2),f5_x2(x0,x1,x2)],[x0,x1,x2]).eval(y0pt).tonumeric() == array([[1.0,0,3.0],[5,0,0],[0,0,0.5]])).all()

def f5(z):
    assert isinstance(z, Point)
    return Point({'coorddict': {'x0': z('x0')*z('x2'),
                                'x1': z('x0')*5,
                                'x2': z('x2')**0.5}})
df5 = diff(f5, y0pt, axes=['x1','x2'])

# Comparing 1st order Taylor series for nearby point to actual value:
df5_taylor = array(f5(y1pt))
df5_diffd = array(f5(y0pt)) + simplifyMatrixRepr(diff(f5, y0pt)*matrix(y1pt-y0pt).T)
assert alltrue([err < 0.01 for err in abs(df5_taylor-df5_diffd)])

print "  ...passed"

