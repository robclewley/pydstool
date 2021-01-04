"""Using PyCont to do path following to find a zero level set of a nonlinear
function. Example is an ellipse.

Robert Clewley, August 2008.
"""

from PyDSTool import *
from numpy.linalg import norm

# f(y) = 0 is the required form
# where f(y) = x^2 /2 + y^2 - 1
DSargs = args(name='ellipse')

# treat one of the coordinates as a parameter for PyCont to use
DSargs.pars = {'x': 0}

# the other coordinate is a 'variable'
DSargs.varspecs = {'y': 'x*x/2.+y*y-1'}

# A starting point on the ellipse, if known. Let's pretend it's a tougher
# problem and we only know a point that's close to it: (x,y) = (0, 1.1)
# PyCont will find the closest point that's actually on the curve before
# doing the path following
DSargs.ics = {'y': 1.1}

# Define an initial value problem (ODE) with f(x) as the right hand side.
# For this example, it is just a formality to present PyCont with f(x).
# In examples involving fixed points or limit cycles of dynamical systems,
# this ODE will have a practical meaning.
testODE = Vode_ODEsystem(DSargs)

# Create an instance of PyCont
P = ContClass(testODE)

# EP-C = equilibrium point curve
PCargs = args(name='test', type='EP-C')
PCargs.freepars = ['x']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 800
PCargs.MaxStepSize = 1e-2

# Declare a new curve based on the above criteria
P.newCurve(PCargs)

# Do path following in the 'forward' direction. Max points is large enough
# to ensure we go right around the ellipse (PyCont automatically stops when
# we return to the initial point - unless MaxNumPoints is reached first.)
P['test'].forward()

sol = P['test'].sol

print("There were %i points computed" % len(sol))
# solution points:
print(sol)

print("\nLabels for each point in sol pointset give diagnostic information about")
print("first derivative of the point along the curve ('V' entry) and the arc ")
print("length parameter shows distance along the curve so far ('ds' entry).")

print("\nVelocity around curve is always 1, e.g. look at 100th point")
print("norm(Point(sol[100].labels['EP']['data'].V)) =", \
      norm(Point(sol[100].labels['EP']['data'].V)))

print("... at which we have travelled distance ds =", \
      sol[100].labels['EP']['data'].ds)

print("\nThis distance is equal to 100 * the max step size (0.01) which PyCont")
print("deemed sufficient for the desired accuracy.")

# easy way to plot the result
P.display(curves=['test'], coords=('x','y'))
