"""Test pickling for saving and loading various PyDSTool objects"""

from PyDSTool import *
import os

# --------------------------------------------------------------------

print "Test pickling for saving and loading various PyDSTool objects"

# array
a=array([1,Inf])
b=[Inf,0]

saveObjects([a,b], 'temp_objects.pkl', True)
loadedObjs = loadObjects('temp_objects.pkl')
assert a[0]==loadedObjs[0][0]
assert a[1]==loadedObjs[0][1]
assert b[0]==loadedObjs[1][0]

# Interval
m=Interval('test1', float, (-Inf,1))
s=Interval('a_singleton', float, 0.4)
saveObjects([m,s], 'temp_objects.pkl', True)
objs_ivals = loadObjects('temp_objects.pkl')
assert objs_ivals[0].get(1) == 1

# Try loading partial list from a larger file
objs_part = loadObjects('temp_objects.pkl', ['a_singleton'])

# Point
x = Point(coorddict = {'x0': [1.123456789], 'x1': [-0.4], 'x2': [4000]},
       coordtype = float64)

v = Pointset(coorddict = {'x0': 0.2, 'x1': -1.2},
            indepvardict = {'t': 0.01},
            coordtype = float,
            indepvartype = float)

saveObjects([x,v], 'temp_objects.pkl', True)
objs_pts = loadObjects('temp_objects.pkl')
assert objs_pts[0] == x

# Simple Variable
var1 = Variable(Pointset(coordarray = array(range(10), float)*0.1,
                         indepvararray = array(range(10), float)*0.5
                        ), name='v1')
saveObjects(var1, 'temp_objects.pkl', True)
obj_var = loadObjects('temp_objects.pkl')[0]
assert obj_var(1.5) == var1(1.5)

# Trajectory
var2 = Variable(Pointset(coordarray = array(range(10), float)*0.25+1.0,
                         indepvararray = array(range(10), float)*0.5
                        ), name='v2')
traj = Trajectory('traj1', [var1,var2])
saveObjects(traj, 'temp_objects.pkl', True)
traj_loaded = loadObjects('temp_objects.pkl')[0]
assert traj_loaded(2.0) == traj(2.0)


# Interpolated table Generator
xnames = ['x1', 'x2']
timeData = array([0.1, 1.1, 2.1])
x1data = array([10.2, -1.4, 4.1])
x2data = array([0.1, 0.01, 0.4])
xData = makeDataDict(xnames, [x1data, x2data])
itableArgs = {}
itableArgs['tdata'] = timeData
itableArgs['ics'] = xData
itableArgs['name'] = 'interp'
interptable = InterpolateTable(itableArgs)
itabletraj = interptable.compute('itable')
saveObjects(itabletraj, 'temp_objects.pkl', True)
obj_itab = loadObjects('temp_objects.pkl')
assert obj_itab[0](0.6) == itabletraj(0.6)


# Vode object with event and external input trajectory (defined earlier)
fvarspecs = {"w": "k*w + a*itable + sin(t) + myauxfn1(t)*myauxfn2(w)",
               'aux_wdouble': 'w*2 + globalindepvar(t)',
               'aux_other': 'myauxfn1(2*t) + initcond(w)'}
fnspecs = {'myauxfn1': (['t'], '2.5*cos(3*t)'),
             'myauxfn2': (['w'], 'w/2')}
ev_args = {'name': 'threshold',
           'eventtol': 1e-4,
           'eventdelay': 1e-5,
           'starttime': 0,
           'term': True,
           }
thresh_ev = Events.makePythonStateZeroCrossEvent('w', 20, 1, ev_args)
DSargs = {'tdomain': [0.1,2.1],
          'tdata': [0.11, 2.1],
          'ics': {'w': 3.0},
          'pars': {'k':2, 'a':-0.5},
          'inputs': {'itable' : interptable.variables['x1']},
          'auxvars': ['aux_wdouble', 'aux_other'],
          'algparams': {'init_step':0.01, 'strict':False},
          'events': thresh_ev,
          'checklevel': 2,
          'name': 'ODEtest',
          'fnspecs': fnspecs,
          'varspecs': fvarspecs
          }
testODE = Vode_ODEsystem(DSargs)
odetraj = testODE.compute('testode')
saveObjects([odetraj, testODE], 'temp_objects.pkl', True)
objs_ode = loadObjects('temp_objects.pkl')
objs_ode[1].diagnostics.clearWarnings()
assert len(objs_ode[1].diagnostics.warnings) == 0
odetraj2 = objs_ode[1].compute('testode2')
assert odetraj2(0.6) == odetraj(0.6)
assert len(objs_ode[1].diagnostics.warnings) == 1

# ExplicitFnGen
args = {'tdomain': [-50, 50],
        'pars': {'speed': 1},
        'xdomain': {'s': [-1., 1.]},
        'name': 'sine',
        'globalt0': 0.4,
        'pdomain': {'speed': [0, 200]},
        'varspecs': {'s': "sin(globalindepvar(t)*speed)"}}
sin_gen = ExplicitFnGen(args)
sintraj1 = sin_gen.compute('sine1')
sin_gen.set(pars={'speed': 2})
sintraj2 = sin_gen.compute('sine2')
saveObjects([sin_gen,sintraj1,sintraj2], 'temp_objects.pkl', True)
objs_sin = loadObjects('temp_objects.pkl')
assert sintraj1(0.55) == objs_sin[1](0.55)
assert sintraj2(0.55) == objs_sin[2](0.55)


# ImplicitFnGen
fvarspecs = {"y": "t*t+y*y-r*r",
                "x": "t"
                }
argsi = {}
argsi['varspecs'] = fvarspecs
argsi['algparams'] = {'solvemethod': 'newton',
                     'atol': 1e-4}
argsi['xdomain'] = {'y': [-2,2]}
argsi['ics'] = {'y': 0.75}
argsi['tdomain'] = [-2,0]
argsi['pars'] = {'r':2}
argsi['vars'] = ['y']
argsi['checklevel'] = 2
argsi['name'] = 'imptest'

testimp = ImplicitFnGen(argsi)
traj1 = testimp.compute('traj1')
saveObjects([testimp, traj1], 'temp_imp.pkl', True)
objs_imp = loadObjects('temp_imp.pkl')
assert objs_imp[0].xdomain['y'] == [-2,2]
assert traj1(-0.4) == objs_imp[1](-0.4)

# Model
# not present

# delete temp file
os.remove('temp_objects.pkl')

print "   ...passed"
