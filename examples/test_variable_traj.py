"""
Tests and demonstration of Variable class
"""

from PyDSTool import *

w_pts = Pointset({'coordarray': array([4.456, 2.34634, 7.3431, 5.443], float64),
              'indepvararray': array([0.0, 1.0, 2.0, 3.0], float64)})
w_var = Variable(w_pts)
print("w_var(0.0) => ", w_var(0.0))

print("\n")
f2 = interp1d([0., 1., 2.], [5,6,7])
v_var = Variable(f2, 't', 'x')
print("Use optional 'checklevel' argument to specify degree of bounds/domain checking")
print("This is useful mainly during initial computation of a variable")
print("v_var(0.01) => ", v_var(0.01, 2))
print("\nv_var(array([0.24,2.566]), 2) =>\n", end='')
try:
    print(v_var(array([0.24,2.566]), 2))
except ValueError as e:
    print(" ",e)
print("\nv_var =>\n", end='')
print(v_var)
print("v_var.getDataPoints() => ", v_var.getDataPoints())

# ------------------------------

print("""Test of domain checking... By making depdomain to be a
set of integers, and the coordarray to be non-integers, this
object can only be called for indepvar values in [0, 0.5, 1, 1.5,
..., 4.5]""")

print("\n")
v_int = Variable(Pointset({'coordarray': array(list(range(10)), float64)*0.1,
                        'indepvararray': array(list(range(10)), float64)*0.5
                        }))
print("v_int(0.5) => ", v_int(0.5, 2))
print("v_int(0.4) => ")
try:
    v_int(0.4, 2)
except ValueError as e:
    print("Successfully checked domain validation with v_int:")
    print(" ... error was "+str(e))
print("v_int(0) => ", v_int(0))

# ------------------------------

print("\nTest simple functions in Variable object")
exp_str = """exp_var = Variable(math.exp, 'x', Interval('y', float, [0,Inf]))"""
print(exp_str)
exec(exp_str)
print("exp_var(0.5) => ", exp_var(0.5))

print("\nTest wrapped functions in OutputFn class")
print("""The optional Intervals specify the "trajectory" range, but are for
informational purposes only! They are not checked anywhere.""")
sin_str = """sin_opfunc = OutputFn(math.sin, (Interval('t', float, [0,Inf]),
                                 Interval('x', float, [-1.,1.])))"""
print(sin_str)
exec(sin_str)
print("""\nThese Intervals specify the valid domains of the indep and dep var
Deliberately only allow +ve angles (unless specified here they won't be checked)""")
sin_str2 = """sin_var = Variable(sin_opfunc, Interval('t', float, [0,Inf]),
                Interval('x', float, [-1,1]))"""
print(sin_str2)
exec(sin_str2)
print("sin_var(math.pi) => ", sin_var(math.pi, 2))
print("sin_var(-math.pi/2) =>")
try:
    sin_var(-math.pi/2, 2)
except ValueError as e:
    print(" ", e)

print("sin_var([0., 0.5*math.pi, math.pi]) => ", sin_var([0., 0.5*math.pi,
                                                          math.pi]))

# ------------------------------

from PyDSTool.utils import makeImplicitFunc
print("\nTest implicit function routine on half-circle of radius 2")
cf = """def circ_formula(x,y):
    return x*x+y*y-4"""

print(cf)
exec(cf)

nf_str = """newton_halfcirc_fn = makeImplicitFunc(circ_formula, x0=0.75, solmethod='newton')"""
print(nf_str)
exec(nf_str)

impl_str = """implicit_halfcirc = OutputFn(newton_halfcirc_fn, (Interval('t',
float, (0,2)), Interval('x', float, (-2,2))))"""
print(impl_str)
exec(impl_str)

print("tval = -1.3")
tval = -1.3
xval = implicit_halfcirc(tval)
print("xval = implicit_halfcirc(tval) => ", xval)
print("math.sqrt(xval*xval + tval*tval) => ", math.sqrt(xval*xval + \
                                                        tval*tval), " = radius")
print("As it stands, the OutputFn doesn't understand the bounds on the variables:")
print("implicit_halfcirc(3.) =>")
try:
    implicit_halfcirc(3.)
except RuntimeError as e:
    print(" ... returns error: ", e)
except AssertionError as e:
    print(" ... returns error: ", e)
print("""\nSo we can embed this OutputFn into a variable for x as
a function of t, with enforcable bounds (using checklevel > 0 second call argument)""")

implicit_hc_varstr = """implicit_halfcirc_var = Variable(implicit_halfcirc, Interval('t', float, (0,2)),
Interval('x', float, (-2,2)))"""

print(implicit_hc_varstr)
exec(implicit_hc_varstr)

print(implicit_halfcirc_var(tval), ", as before")
print("implicit_halfcirc_var(3., 2) =>")
try:
    implicit_halfcirc_var(3., 2)
except ValueError as e:
    print(" ... returns error: ", e)

print("Tests passed")


print("\n\nTest regular Trajectory object")
v1 = Variable(Pointset({'coordarray': array(list(range(10)), float)*0.1,
                        'indepvararray': array(list(range(10)), float)*0.5
                        }), name='v1')
v2 = Variable(Pointset({'coordarray': array(list(range(10)), float64)*0.25+1.0,
                        'indepvararray': array(list(range(10)), float64)*0.5
                        }), name='v2')
traj = Trajectory('test1', [v1,v2])
print("print traj(0.5, checklevel=2) => ", traj(0.5, checklevel=2))
print("print traj([0., 0.5]) => ", traj([0., 0.5], 'v1'))
print("traj(0.4, 0, checklevel=2) =>")
try:
    traj(0.4, 0, checklevel=2)
except ValueError as e:
    print(" ... raised error: ", e)

print("Tests passed.")
