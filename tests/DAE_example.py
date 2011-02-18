"""Simple example of solving a differential-algebraic equation.
Here, the solution is constrained to lie on the right branch of y = x^2,
but note that the constraint becomes inconsistent with the flow at x=y=0.

(We could add an event to detect this point and stop the integration,
creating a hybrid system.)

    Robert Clewley, February 2007.
"""

from PyDSTool import *

DSargs = args()
DSargs['varspecs'] = {'y': '-1', 'x': 'y - x*x'}
DSargs['algparams'] = {'init_step': 0.05, 'refine': 0, 'max_step': 0.1,
                           'rtol': 1e-4, 'atol': 1e-4}
DSargs['checklevel'] = 1
DSargs['ics'] = {'y': 4, 'x': 2}

# 0 in the (x,x) entry of the mass matrix indicates that this is the
# algebraic equation ( 0 . dx/dt = y - x*x )
#
# 1 in the (y,y) entry indicates that the 'y' varspec is a regular
# differential equation.
#
# 0 in the (x,y) and (y,x) entries just says that there's no interaction
# between the equations apart from what's explicitly given in the right-hand
# sides.
DSargs['fnspecs'] = {'massMatrix': (['t','x','y'], '[[0,0],[0,1]]')}

DSargs['pars'] = {}
DSargs['vars'] = ['x','y']

DSargs['name'] = "DAE_test"

dae = Generator.Radau_ODEsystem(DSargs)

dae.set(tdata=[0, 4])
# if you integrate any longer there will be an error because the constraint
# will become inconsistent with the flow at (0,0)

traj = dae.compute('test')
pd = traj.sample(dt=0.05)
plot(pd['x'], pd['y'])

show()
