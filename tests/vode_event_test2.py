"""
    Test Vode_ODEsystem with events involving external inputs.

    Robert Clewley, September 2006.
"""

from PyDSTool import *

xs = ['x1','x2','x3']
ys = [0, 0.5, 1]

timeData = linspace(0, 10, 20)
sindata = sin(20*timeData)
xData = makeDataDict(['example_input'], [sindata])
my_input = InterpolateTable({'tdata': timeData,
                              'ics': xData,
                              'name': 'interp1d',
                              'method': 'linear',
                              'checklevel': 1,
                              'abseps': 1e-5
                              }).compute('interp')

fvarspecs = {"w": "k*w  + pcwfn(sin(t)) + myauxfn1(t)*myauxfn2(w)",
           'aux_wdouble': 'w*2 + globalindepvar(t)',
           'aux_other': 'myauxfn1(2*t) + initcond(w)'}
fnspecs = {'myauxfn1': (['t'], '2.5*cos(3*t)'),
           'myauxfn2': (['w'], 'w/2'),
           'pcwfn': makeMultilinearRegrFn('x',xs,ys)}
# targetlang is optional if the default python target is desired
DSargs = args(fnspecs=fnspecs, name='ODEtest')
DSargs.varspecs = fvarspecs
DSargs.tdomain = [0.1,2.1]
DSargs.pars = {'k':2, 'a':-0.5, 'x1': -3, 'x2': 0.5, 'x3': 1.5}
DSargs.vars = 'w'
DSargs.inputs = {'in': my_input.variables['example_input']}
DSargs.algparams = {'init_step':0.01}
DSargs.checklevel = 2
testODE = Vode_ODEsystem(DSargs)
print "params set => ", testODE.pars
print "DS defined? => ", testODE.defined
print "testODE.set(...)"
testODE.set(ics={'w':3.0},
                tdata=[0.11,2.1])
print "traj1 = testODE.compute('traj1')"
traj1 = testODE.compute('traj1')
print "DS defined now? => ", testODE.defined
print "traj1(0.5) => ", traj1(0.5)
print "testODE.diagnostics.showWarnings() => "
testODE.diagnostics.showWarnings()
print "\ntraj1.indepdomain => ", traj1.indepdomain
print "traj1(0.2, ['aux_other']) => ", traj1(0.2, ['aux_other'])

print "\nNow adding a terminating co-ordinate threshold event"
print " and non-terminating timer event"
# Show off the general-purpose, language-independent event creator:
#  'makeZeroCrossEvent'
ev_args_nonterm = {'name': 'monitor',
           'eventtol': 1e-4,
           'eventdelay': 1e-5,
           'starttime': 0,
           'active': True,
           'term': False,
           'precise': True}
thresh_ev_nonterm = Events.makeZeroCrossEvent('in', 0,
                        ev_args_nonterm, inputnames=['in'])

# Now show use of the python-target specific creator:
#  'makePythonStateZeroCrossEvent', which is also only
#  able to make events for state variable threshold crossings
ev_args_term = {'name': 'threshold',
           'eventtol': 1e-4,
           'eventdelay': 1e-5,
           'starttime': 0,
           'active': True,
           'term': True,
           'precise': True}
thresh_ev_term = Events.makePythonStateZeroCrossEvent('w',
                        20, 1, ev_args_term)
testODE.eventstruct.add([thresh_ev_nonterm,thresh_ev_term])
print "Recomputing trajectory:"
print "traj2 = testODE.compute('traj2')"
traj2 = testODE.compute('traj2')
print "\ntestODE.diagnostics.showWarnings() => "
testODE.diagnostics.showWarnings()
print "\ntraj2.indepdomain.get() => ", traj2.indepdomain.get()
indep1 = traj2.indepdomain[1]
assert indep1 < 1.17 and indep1 > 1.16
mon_evs_found = testODE.getEvents()['monitor']
assert len(mon_evs_found) == 1
