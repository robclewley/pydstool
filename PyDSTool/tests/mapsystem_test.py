"""
    Tests for the Generator class.  #3.
    Test MapSystem with external input (an InterpolateTable) and events.

    Robert Clewley, June 2005.
"""

from PyDSTool import *
from time import clock

# datafn provides an external input to the map system -- this could have
# been from an experimentally measured signal
datafn = Generator.InterpolateTable({'name':'datafunction',
                                   'tdata': [-30., 0., 5., 10., 30., 100., 180., 400.],
                                   'ics': {'x':[4., 1., 0., 1., 2., 3., 4., 8.]}
                                   })

fvarspecs = {"w": "15-a*w + 2*x",
                "v": "1+k*w/10",
           'aux_wdouble': 'w*2 + globalindepvar(t)',
           'aux_other': 'myauxfn(2*t) + initcond(w)'}
fnspecs = {'myauxfn': (['t'], '.5*cos(3*t)')}
# targetlang is optional if default=python is OK
DSargs = args(name='maptest',fnspecs=fnspecs)
DSargs.varspecs = fvarspecs
DSargs.tdomain = [0,400]
DSargs.pars = {'k':2.1, 'a':-0.5}
DSargs.vars = ['w', 'v']
DSargs.ttype = int  # force independent variable type to be integer
DSargs.checklevel = 2
DSargs.inputs = datafn.variables
testmap = MapSystem(DSargs)
print "params set => ", testmap.pars
print "DS defined? => ", testmap.defined
print "testmap.set(...)"
testmap.set(ics={'w': 3.0, 'v': 2.},
                tdata=[10,400])
print "traj1 = testmap.compute('traj1')"
traj1 = testmap.compute('traj1')
print "DS defined now? => ", testmap.defined
print "traj1(25) => ", traj1(25)
print "testmap.diagnostics.showWarnings() => "
testmap.diagnostics.showWarnings()
print "\ntraj1.indepdomain => ", traj1.indepdomain
print "traj1(30, ['aux_other']) => ", traj1(30, ['aux_other'])
print "traj1(range(10,40)) => ", traj1(range(10,40))

print "\nNow adding a terminating co-ordinate threshold event at w=58..."
ev_args = {'name': 'threshold',
           'eventtol': 1e-4,
           'eventdelay': 1e-5,
           'starttime': 0,
           'active': True,
           'term': True,
           'precise': False}
thresh_ev = Events.makePythonStateZeroCrossEvent('w', 58, 1, ev_args)
testmap.eventstruct.add(thresh_ev)
print "Recomputing trajectory:"
print "traj2 = testmap.compute('traj2')"
t0=clock()
traj2 = testmap.compute('traj2')
print "Elapsed time =", clock()-t0
assert testmap.getEventTimes()['threshold'] == [347.]
print "\ntestmap.diagnostics.showWarnings() => "
testmap.diagnostics.showWarnings()
print "\ntraj2.indepdomain.get() => ", traj2.indepdomain.get()
print "traj2(traj2.indepdomain[1],'w') => ", traj2(traj2.indepdomain[1],'w')
print "Tests passed."
