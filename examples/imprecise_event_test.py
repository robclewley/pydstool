"""A demonstration of an imprecise event detected using dopri.

   Robert Clewley, August 2005.
"""

from PyDSTool import *
from time import perf_counter


# ------------------------------------------------------------

thresh_ev_prec = Events.makeZeroCrossEvent('x-40', 1,
                                {'name': 'thresh_ev_prec',
                                 'eventtol': 1e-4,
                                 'term': False}, ['x'],
                                    targetlang='c')
thresh_ev_imprec = Events.makeZeroCrossEvent('x-40', 1,
                                {'name': 'thresh_ev_imprec',
                                 'eventtol': 1e-4,
                                 'precise': False,
                                 'term': False}, ['x'],
                                targetlang='c')

DSargs = {}
DSargs['varspecs'] = {'x': 'k*(a-x)'}
DSargs['pars'] = {'k': 1., 'a': 50.}
DSargs['xdomain'] = {'x': [0, 50]}
DSargs['algparams'] = {'init_step': 0.1, 'refine': 0, 'max_step': 5.}
DSargs['checklevel'] = 0
DSargs['ics'] = {'x': 0.1}
DSargs['name'] = 'imprecise_ode'
DSargs['events'] = thresh_ev_imprec
ode = Generator.Dopri_ODEsystem(DSargs)
ode.set(tdata=[0, 80])
traj = ode.compute('test')

print('Preparing plot')

plotData = traj.sample(dt=0.1)
yaxislabelstr = 'x'
plt.ylabel(yaxislabelstr)
plt.xlabel('t')
vline=plot(plotData['t'], plotData['x'])
evt=ode.getEventTimes()['thresh_ev_imprec'][0]
print("Event at", evt, "where x has value", traj(evt))
if traj(evt,'x') != 40:
    print("Event occurred away from precise threshold: Test PASSED")
else:
    print("Event occurred precisely on threshold value: Test FAILED")
    raise RuntimeError
plot(evt, traj(evt, 'x'), 'ro')
show()
