"""
    Tests for the Generator class.  #4.
    Test Implicit Function generator

    Robert Clewley, June 2005.
"""

from PyDSTool import *
from numpy.linalg import norm
from copy import copy

print "Test implicit function as variable"
# 1D example: a half-circle using newton's method (secant method for estimating derivative)
# Change sign of 'y' initial condition to solve for other half-circle
fvarspecs = {"y": "t*t+y*y-r*r",
                "x": "t"
                }
##fnspecs = {'myauxfn': (['t'], '.5*cos(3*t)')}
dsargs=args()
##dsargs.fnspecs = fnspecs
dsargs.varspecs = fvarspecs
dsargs.algparams = {'solvemethod': 'newton',
                     'atol': 1e-4}
dsargs.xdomain = {'y': [-2,2]}
dsargs.ics = {'y': 0.75}
dsargs.tdomain = [0,2]
dsargs.pars = {'r':2}
dsargs.vars = ['y']
dsargs.checklevel = 1
dsargs.name = 'imptest'

testimp = ImplicitFnGen(dsargs)
assert not testimp.defined
traj1 = testimp.compute('traj1')
assert testimp.defined
assert allclose(traj1(1.5)['y'], 1.3228755105)
assert traj1.dimension == 2

# 2D example: a quarter-sphere with linear constraint y = 3z,
# using MINPACK's fsolve
radius = 2.
fvarspecs2d = {"y": "t*t+y*y+z*z-r*r",
                  "z": "y-3.*z",
                  "x": "t"
                  }
args2d = {'varspecs': fvarspecs2d,
          'algparams': {'solvemethod': 'fsolve',
                        'atol': 1e-3},
          'xdomain': {'y': [-2,2], 'z': [-2,2]},
          'ics': {'y': 0.75, 'z': 0.9},
          'tdomain': [-2,2],
          'pars': {'r':radius},
          'vars': ['y', 'z'],
          'checklevel': 1,
          'name': 'imptest2d'
          }
testimp2d = ImplicitFnGen(args2d)

traj2 = testimp2d.compute('traj2')
p = traj2(1.5)
assert allclose(norm(p), radius)
assert allclose(p('y') - 3*p('z'), 0)
assert traj2.dimension == 3
assert isparameterized(traj2)

# Test bounds checking
try:
    traj2(3.)
except ValueError:
    pass
else:
    raise AssertionError("Bounds test failed")

saveObjects([testimp2d, traj2], 'temp_imp.pkl', force=True)

print "   ...passed"
