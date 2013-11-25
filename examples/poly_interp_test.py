"""Polynomial interpolation test using van der Pol oscillator.
"""

from PyDSTool import *

def create_system(sysname='vanderPol', eps=1.0):
    pars = {'eps': eps, 'a': 0.5, 'y1': -0.708}

    icdict = {'x': pars['a'], 'y': pars['a'] - pars['a']*pars['a']*pars['a']/3}

    # Set up models
    xstr = '(y - (x*x*x/3 - x))/eps'
    ystr = 'a - x'

    DSargs = args(name=sysname)
    """ (function, direction, arguments, variables, language) """
    event_x_a = Events.makeZeroCrossEvent('x-a', 0,
                                {'name': 'event_x_a',
                                 'eventtol': 1e-10,
                                 'term': False,
                                 'active': True}, varnames=['x'],
                                parnames=['a'])
    DSargs.events = [event_x_a]
    DSargs.pars = pars
    DSargs.tdata = [0, 50]
    DSargs.algparams = {'max_pts': 300000, 'init_step': 0.2, 'poly_interp': True}
    DSargs.varspecs = {'x': xstr, 'y': ystr}
    DSargs.ics = icdict
    DSargs.pdomain = {'a': [0, 1.2]}

    return Generator.Vode_ODEsystem(DSargs)

s = create_system()
traj = s.compute('test')
pts = traj.sample()
t = pts.indepvararray
x = pts['x']
plot(t, x)
# for now, no derivative information stored in traj, so have to fetch that ourselves
dx = array([s.Rhs(0, state, s.pars) for state in pts])[:,0]

poly = PiecewisePolynomial(t, array([x, dx]).T, 2)

xp = poly(t)

fine_t = linspace(t[0], t[50], 500)
xp_fine = poly(fine_t)
xp_check = traj(fine_t)['x']
assert sum(xp_fine-xp_check) == 0, 'Polynomial interpolation problem'

plot(t, xp, 'bo')
plot(fine_t, xp_fine, 'r.')
plot(fine_t, xp_check, 'k.')