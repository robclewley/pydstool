"""
    Version of Vode_withJac_test.py, but using symbolic expressions and
     differentiation to derive Jacobian from the vector field.

    Demonstrates easier syntax expression for defining the Generator using
     the symbolic objects (especially the auxiliary functions).

    Robert Clewley, September 2005.
"""
from __future__ import print_function
from PyDSTool import *

# declarations of symbolic objects
y0 = Var('y0')
y1 = Var('y1')
y2 = Var('y2')
p = Par('p')
t = Var('t')

# definitions of right-hand sides for each variable, as functions of all variables
ydot0 = Fun(-0.04*y0 + p*y1*y2, [y0, y1, y2], 'ydot0')
ydot2 = Fun(3e7*y1*y1, [y0, y1, y2], 'ydot2')
ydot1 = Fun(-ydot0(y0,y1,y2)-ydot2(y0,y1,y2), [y0, y1, y2], 'ydot1')

# the whole vector field as one nonlinear multi-variate function: R^3 -> R^3
F = Fun([ydot0(y0,y1,y2),ydot1(y0,y1,y2),ydot2(y0,y1,y2)], [y0,y1,y2], 'F')

# Diff is the symbolic derivative operator, and returns a QuantSpec object.
# It must be named "Jacobian" for PyDSTool to recognize it as such.
jac = Fun(Diff(F,[y0,y1,y2]), [t, y0, y1, y2], 'Jacobian')
# This defines a function of arguments (t, y0, y1, y2) obtained by differentiating F in all three
# variables, y0, y1, and y2.

# The user could inspect this expression, jac, and make some simplifications
# by hand, to help optimize the speed of its evaluation. Here, we'll
# just use it as is.

DSargs = args(name='jactest', checklevel=2)
DSargs.fnspecs = [jac, ydot0, ydot2]
DSargs.varspecs = {y0: ydot0(y0,y1,y2),
                   y2: ydot2(y0,y1,y2),
                   y1: -ydot0(y0,y1,y2)-ydot2(y0,y1,y2)}
DSargs.tdomain = [0.,1e20]
DSargs.pars = {p: 1e4}
DSargs.ics = {y0: 1.0, y1: 0., y2: 0.}
DSargs.algparams = {'init_step':0.4, 'strictdt': True, 'stiff': True,
                    'rtol': 1e-4, 'atol': [1e-8,1e-14,1e-6]}

DSargs.events = makeZeroCrossEvent(y0-0.001, -1, {'name': 'thresh_ev',
                       'eventtol': 10,
                       'bisectlimit': 20,
                       'eventinterval': 500,
                       'eventdelay': 0,  #otherwise cannot catch event with only one step per run
                       'term': True}, [y0])
testODE = Vode_ODEsystem(DSargs)

print("Defined the following internal Python function for Jacobian:")
print(testODE.funcspec.auxfns['Jacobian'][0])

print("\nOutput at exponentially larger subsequent time steps (with event detection!):\n")
tvals = [0.4*10**i for i in range(0,12)]
t0 = 0.

for t1 in tvals:
    dt = t1-t0
    print("\n===============================================\nAt t=", end=' ')
    print(t1, "using dt =", dt)
    testODE.set(tdata=[t0,t1], algparams={'init_step': dt})
    if t0 >0.:
        print(testODE._solver.y)
    traj = testODE.compute('test', 'c')  # c for continue (acts as f first time)
    try:
        print(traj(t1))
    except ValueError:
        print("\n----------")
        testODE.diagnostics.showWarnings()
        print(traj(traj.indepdomain[1]))
        print("----------\n")
        t0 = traj.indepdomain[1]
    else:
        t0 = t1

print("\nCompare results with the output directly from the scipy_ode.py test")
print("(up to the point where the terminal event was found).")
print("The values from a test integration performed with scipy_ode.py " \
      + "are listed in the comments at the end of this script")

## At t=0.0  y=[ 1.  0.  0.]
## At t=0.4  y=[ 9.85172114e-001  3.38639538e-005  1.47940221e-002]
## At t=4.0  y=[ 9.05518679e-001  2.24047569e-005  9.44589166e-002]
## At t=40.0  y=[ 7.15827070e-001  9.18553479e-006  2.84163745e-001]
## At t=400.0  y=[ 4.50518662e-001  3.22290136e-006  5.49478115e-001]
## At t=4000.0  y=[ 1.83202242e-001  8.94237031e-007  8.16796863e-001]
## At t=40000.0  y=[ 3.89833646e-002  1.62176779e-007  9.61016473e-001]
## At t=400000.0  y=[ 4.93828363e-003  1.98499788e-008  9.95061697e-001]
## At t=4000000.0  y=[ 5.16819063e-004  2.06833253e-009  9.99483179e-001]
## At t=40000000.0  y=[ 5.20200703e-005  2.08090952e-010  9.99947980e-001]
## At t=400000000.0  y=[ 5.21215549e-006  2.08487297e-011  9.99994788e-001]
## At t=4000000000.0  y=[ 5.25335126e-007  2.10134087e-012  9.99999475e-001]
## At t=40000000000.0  y=[ 5.63729748e-008  2.25491907e-013  9.99999944e-001]

# check that we can set ICs and params using symbolic objects
testODE.set(ics={y0: 1.25}, pars={p: 1e4+1},
            tdata=[0,1], algparams={'init_step': 0.1})

# and this way
traj2 = testODE.compute('test2', ics={y1: 0.01})
