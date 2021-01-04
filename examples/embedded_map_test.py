"""
    Embedded map generator. Analyses a Poincare map of a van der Pol
    oscillator's limit cycle using PyCont.

    Robert Clewley, August 2006.
"""

from PyDSTool import *
from copy import copy, deepcopy
from scipy.optimize import minpack, optimize
from numpy.linalg import eigvals, det, norm
from PyDSTool.PyCont import *

# van der Pol type oscillator
vspec = {'x':'m*(y+x-x*x*x/3)',
         'y':'n*y-x'}
pars = {'m':1, 'n': 0.1}

ev = Events.makeZeroCrossEvent("x", -1,
                            {'eventtol': 1e-5,
                             'eventdelay': 0.05,
                             'starttime': 0,
                             'term': True,
                             'name': 'x_ev'}, ['x'], [],
                            targetlang='c')


## ev defines a Poincare section across the limit cycle (at x=y=0)
# use 'stiff' option for Vode integrator
mapsys = Generator.Dopri_ODEsystem(args(varspecs=vspec,
                                    pars=pars,
                                    algparams={'init_step':0.05}, #, 'stiff': True},
                                    tdata=[0,60],
                                    ics={'x':0, 'y':0},
                                    name='embeddedsys',
                                    checklevel=2,
                                    events=ev))

sys=deepcopy(mapsys)

def getTraj(ics=None,pars=None,t1=50,termFlag=True,trajname='test',pplane=True):
    if ics is not None:
        sys.set(ics=ics)
    if pars is not None:
        sys.set(pars=pars)
    sys.set(tdata=[0,t1])
    sys.eventstruct.setTermFlag('x_ev', termFlag)
    traj = sys.compute(trajname)
    pts = traj.sample()
    if pplane:
        plot(pts['x'],pts['y'])
    else:
        plot(pts['t'],pts['x'])
        plot(pts['t'],pts['y'])
    return traj, pts


##traj,pts=getTraj({'x':-2,'y':0},termFlag=False)
##1/0


map_ics = {'y': -2.1}
plot([0,0], [-3,0], 'k-', linewidth=2)

map_args = args(name='pdc_map')
map_args.varspecs = {'y': 'res["y"]'}
map_args.vfcodeinsert_start = """
    embeddedsys.set(ics={'y': y, 'x': 0}, pars={'m': m, 'n': n}, tdata=[0,20])
    traj=embeddedsys.compute('pdc')
##    print 't =', traj.indepdomain.get()
    ev = embeddedsys.getEvents()['x_ev']
##    print ev['t'][0]
    res = {'y':ev[0]['y']}"""
##    if ev is None:
##        raise RuntimeError('No return to cross-section!')
map_args.ignorespecial = ['res']
map_args.system = mapsys
map_args.ttype = int
map_args.ics = map_ics
map_args.pars = pars
map_args.tdomain = [0,1]
map = MapSystem(map_args)

map.showSpec()

test_traj = map.compute('test', map_ics)
print(test_traj.sample(), "\n")

def return_map(ic):
    return map.compute('pdc_shoot', ic)(1)


# Find fixed point of map
def fp_residual(ic):
    y0 = ic[0]
##    print "\nfp_residual: y0 = ", y0
    v = return_map({'y': y0})
##    plot(0, y0, 'k.')
    print(v)
    return v-ic[0]


print("Finding fixed point of return map as function of initial condition")

# Equally efficient alternatives to shoot for f.p. solution
y0_guess = map_ics['y']
sol_pdc = minpack.fsolve(fp_residual, array([y0_guess]), xtol=1e-6)
print("sol_pdc = ", sol_pdc)

print("Plotting (x,y) limit cycle of oscillator with Poincare section")
print(" given by x=0")
traj, pts = getTraj({'x':0, 'y':-sol_pdc}, {'m':1}, t1=10, termFlag=False)
traj, pts = getTraj({'x':0, 'y':-sol_pdc}, {'m':14}, t1=10, termFlag=False)


map.set(ics={'y':sol_pdc})

print("Initializing PyCont to follow x=0 crossing as parameter m varied:")

pc=ContClass(map)
pcargs=args(name='FPu', type='FP-C',
  freepars = ['m'],
  StepSize = 3e-1,
  MaxNumPoints = 30, # 40 causes IndexError in PyCont/BifPoint.py, line 149
  MaxStepSize = 5e-1,
  LocBifPoints = 'all',
  verbosity = 3
 )
pc.newCurve(pcargs)

print("\nRunning PyCont forward")

pc['FPu'].forward()

print("\n\nSolution is:\n", end='')
print(repr(pc['FPu'].sol))
s=pc['FPu'].sol
print("Plotting line of fixed points of x vs parameter m ...")
figure()
plot(s['m'],s['y'])

show()
