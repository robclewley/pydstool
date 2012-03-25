import PyDSTool as dst
from PyDSTool import args
from matplotlib import pyplot as plt
import numpy as np


pars = {'eps': 1e-2, 'a': 0.5}
icdict = {'x': pars['a'],
          'y': pars['a'] - pars['a']*pars['a']*pars['a']/3}
xstr = '(y - (x*x*x/3 - x))/eps'
ystr = 'a - x'

event_x_a = dst.makeZeroCrossEvent('x-a', 0,
                            {'name': 'event_x_a',
                             'eventtol': 1e-6,
                             'term': False,
                             'active': True},
                    varnames=['x'], parnames=['a'],
                    targetlang='python')  # targetlang is redundant (defaults to python)

DSargs = args(name='vanderpol')  # struct-like data
DSargs.events = [event_x_a]
DSargs.pars = pars
DSargs.tdata = [0, 3]
DSargs.algparams = {'max_pts': 3000, 'init_step': 0.02, 'stiff': True}
DSargs.varspecs = {'x': xstr, 'y': ystr}
DSargs.xdomain = {'x': [-2.2, 2.2], 'y': [-2, 2]}
DSargs.fnspecs = {'Jacobian': (['t','x','y'],
                                """[[(1-x*x)/eps, 1/eps ],
                                    [ -1,  0 ]]""")}
DSargs.ics = icdict
vdp = dst.Vode_ODEsystem(DSargs)

traj = vdp.compute('test_traj')
pts = traj.sample()
evs = traj.getEvents('event_x_a')

plt.figure(1)
plt.plot(pts['t'], pts['x'], 'b', linewidth=2)
plt.plot(pts['t'], pts['y'], 'r', linewidth=2)

plt.figure(2)
plt.plot(pts['x'], pts['y'], 'k-o', linewidth=2)
plt.plot(evs['x'], evs['y'], 'rs')

from PyDSTool.Toolbox import phaseplane as pp

pp.plot_PP_vf(vdp, 'x', 'y', scale_exp=-1)

# only one fixed point, hence [0] at end
fp_coord = pp.find_fixedpoints(vdp, n=4, eps=1e-8)[0]
fp = pp.fixedpoint_2D(vdp, dst.Point(fp_coord), eps=1e-8)

nulls_x, nulls_y = pp.find_nullclines(vdp, 'x', 'y', n=3, eps=1e-8, max_step=0.1,
                             fps=[fp_coord])

pp.plot_PP_fps(fp)

plt.plot(nulls_x[:,0], nulls_x[:,1], 'b')
plt.plot(nulls_y[:,0], nulls_y[:,1], 'g')
plt.title('Phase plane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()