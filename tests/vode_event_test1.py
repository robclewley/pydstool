"""
    Test terminal and non-terminal event testing with VODE integrator.
"""

from PyDSTool import *

DSargs = args(varspecs={'w': 'k*sin(2*t) - w'}, name='ODEtest')
DSargs.tdomain = [0,10]
DSargs.pars = {'k':1, 'p_thresh': -0.25}
DSargs.algparams = {'init_step':0.001, 'atol': 1e-12, 'rtol': 1e-13}
DSargs.checklevel = 2
DSargs.ics={'w':-1.0}
DSargs.tdata=[0, 10]

ev_args_nonterm = {'name': 'monitor',
           'eventtol': 1e-4,
           'eventdelay': 1e-5,
           'starttime': 0,
           'active': True,
           'term': False,
           'precise': True}
thresh_ev_nonterm = Events.makeZeroCrossEvent('w', 0,
                        ev_args_nonterm, varnames=['w'])

ev_args_term = {'name': 'threshold',
           'eventtol': 1e-4,
           'eventdelay': 1e-5,
           'starttime': 0,
           'active': True,
           'term': True,
           'precise': True}
thresh_ev_term = Events.makeZeroCrossEvent('w-p_thresh',
                        -1, ev_args_term, varnames=['w'], parnames=['p_thresh'])

DSargs.events = [thresh_ev_nonterm,thresh_ev_term]

testODE = Vode_ODEsystem(DSargs)
traj = testODE.compute('traj')
pts = traj.sample()
plot(pts['t'],pts['w'])
testODE.diagnostics.showWarnings()

mon_evs_found = testODE.getEvents('monitor')
term_evs_found = testODE.getEvents('threshold')

assert all(traj.getEvents('monitor') == mon_evs_found)
assert all(traj.getEventTimes('threshold') == testODE.getEventTimes('threshold'))
term_evs_found.info()

# Alternative way to extract events: they are labelled in the
# pointset! These return dictionaries indexing into the pointset.
mon_evs_dict = pts.labels.by_label['Event monitor']
mon_ev_points = pts[sort(mon_evs_dict.keys())]

assert len(mon_evs_found) == len(mon_ev_points) == 2
assert numpy.all(mon_evs_found == mon_ev_points)

