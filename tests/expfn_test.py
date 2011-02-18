"""
    Tests for the Generator class.  #5.
    This script tests:
    (1) the Explicit Function generator,
        its use of "global time", and its deletion.
    (2) Also test Implicit Function generator's deletion

    Robert Clewley, June 2005.
"""

from PyDSTool import *

print "Test of Explicit and Implicit Function generators, global time, and deletion"

ev_args = {'name': 'threshold',
           'eventtol': 1e-4,
           'eventdelay': 1e-5,
           'starttime': 0,
           'active': True,
           'term': True,
           'precise': True}
thresh_ev = Events.makePythonStateZeroCrossEvent('t', 20, 1, ev_args)

DSargs = {'tdomain': [-50, 50],
        'pars': {'speed': 1},
        'xdomain': {'s': [-1., 1.]},
        'name': 'sine',
        'globalt0': 0.4,
        'pdomain': {'speed': [0, 200]},
        'varspecs': {'s': "sin(globalindepvar(t)*speed)"},
        'events': thresh_ev}
sin_gen = ExplicitFnGen(DSargs)
sintraj1 = sin_gen.compute('sine1')
assert sintraj1.globalt0 == 0.4
assert sintraj1(0) == sin(0.4)

sin_gen.set(pars={'speed': 2})
sintraj2 = sin_gen.compute('sine2')

# sintraj2 independent variable domain truncated at terminal event
assert allclose(sin_gen.getEventTimes()['threshold'], 20+sintraj1.globalt0)
assert sintraj2.indepdomain.get() == [-50, 50]

# Simple implicit function
fvarspecs = {"y": "t*t+y*y-r*r",
             "x": "t"}
argsi = args()
argsi.varspecs = fvarspecs
argsi.algparams = {'solvemethod': 'newton',
                   'atol': 1e-4}
argsi.xdomain = {'y': [-2,2]}
argsi.ics = {'y': 0.75}
argsi.tdomain = [-2,0]
argsi.pars = {'r':2}
argsi.vars = ['y']
argsi.checklevel = 2
argsi.name = 'imptest'

testimp = ImplicitFnGen(argsi)
imptraj1 = testimp.compute('traj1')

dv=dir(Variable)
del sintraj1
assert dv != dir(Variable)

print "   ...passed"
