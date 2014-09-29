""" EXAMPLE: Forced spring (as 2nd order ODE with step forcing function)

    We show two ways to achieve this functionality.

    Rob Clewley, March 2007
"""

from PyDSTool import *

pars = {'m': 5., 'c': 2, 'k': 1, 't_on': 7.08, 't_off': 8.5, 'f_mag': 3}

icdict = {'x': 3., 'y': 0.}

## Set up model
# m*x_ddot+c*x_dot+k*x=F where F = 0 until t > t_on
# i.e. this is a hybrid system
#
# Higher-order systems of ODEs must be written in first-order form
# i.e. x_dot = dx/dt = y, x_ddot = dy/dt = function of x, y

# integration time
t_end = 50

## Simple implementation that doesn't use proper event detection
# ... instead, F will switch on at a multiple of the integration time step
x_dot = 'y'
x_ddot = '(F(t)-c*y-k*x)/m'

DSargs = args(name='forced_spring')
DSargs.pars = pars
DSargs.varspecs = {'x': x_dot, 'y': x_ddot}
DSargs.fnspecs = {'F': (['t'], "if(t>=t_on,f_mag,0)*if(t>t_off,0,1)")}
# alternative definition of F: "heav(t>=t_on)*heav(t<t_off)*f_mag"
# or: "heav(t-t_on)*heav(t-t_off)*f_mag"
DSargs.ics = icdict
DSargs.tdata = [0,t_end]
DSargs.algparams = {'init_step': 0.2}

simpleDS = Generator.Vode_ODEsystem(DSargs)

traj_on = simpleDS.compute('force')
pts_on = traj_on.sample()  # get all points in trajectory mesh

simpleDS.set(pars={'t_on':t_end+1})  # no forcing for integration time interval given
traj_off = simpleDS.compute('noforce')
pts_off = traj_off.sample()

plot(pts_on['t'], pts_on['x'], 'g--')
plot(pts_off['t'], pts_off['x'], 'r--')

# accuracy of event detection
evtol = 1e-4

## Proper implementation as a hybrid system

on_event_args = {'name': 'sw_ON',
               'eventtol': evtol,
               'term': True
                }

off_event_args = {'name': 'sw_OFF',
               'eventtol': evtol,
               'term': True
                }

# Define threshold events for step forcing to come on and off
# Must use global time otherwise t = local time for each vector field
switch_on = Events.makeZeroCrossEvent('globalindepvar(t)-t_on', 1,
                                      on_event_args, parnames=['t_on'])
switch_off = Events.makeZeroCrossEvent('globalindepvar(t)-t_off', 1,
                                      off_event_args, parnames=['t_off'])

# f_state is a dummy variable (constant) holding on/off state
varspecs = {'x': x_dot, 'y': x_ddot, 'f_state': '0'}

# Define each vector field (one with forcing 'on', the other 'off')
DSargs_off = args(name='off')
DSargs_off.pars = pars
DSargs_off.varspecs = varspecs
DSargs_off.fnspecs = {'F': (['t'], '0')}
DSargs_off.events = [switch_on]
DSargs_off.algparams = {'init_step': 0.2}
DSargs_off.xdomain = {'f_state': 0}  # only the singleton value 0 is valid
ics = copy(icdict)
ics['f_state'] = 0
DSargs_off.ics = ics
DSargs_off.tdata = [0, t_end]
DS_off = embed(Generator.Vode_ODEsystem(DSargs_off), name='off')

DSargs_on = args(name='on')
DSargs_on.pars = pars
DSargs_on.varspecs = varspecs
DSargs_on.fnspecs = {'F': (['t'], 'f_mag')}
DSargs_on.events = [switch_off]
DSargs_on.algparams = {'init_step': 0.2}
DSargs_on.xdomain = {'f_state': 1}  # only the singleton value 1 is valid
ics = copy(icdict)
ics['f_state'] = 1
DSargs_on.ics = ics
DSargs_on.tdata = [0, t_end]
DS_on = embed(Generator.Vode_ODEsystem(DSargs_on), name='on')

# hybrid state switching rules -- includes event transition functions that
# map f_state discretely from 0 <--> 1
allnames = ['on','off']
DS_on_MI = intModelInterface(DS_on)
DS_off_MI = intModelInterface(DS_off)
evmap_on = EvMapping({"f_state": "1"}, model=DS_on)
evmap_off = EvMapping({"f_state": "0"}, model=DS_off)
g1Info = makeModelInfoEntry(DS_on_MI, allnames, [('sw_OFF', ('off', evmap_off))])
g2Info = makeModelInfoEntry(DS_off_MI, allnames, [('sw_ON', ('on', evmap_on))])
modelInfoDict = makeModelInfo([g1Info, g2Info])

# add state variable to initial conditions
icdict_hybrid = copy(icdict)
icdict_hybrid['f_state'] = 0

# make hybrid model object from the two generators + event transition information
m = Model.HybridModel(name='spring_model', modelInfo=modelInfoDict)
# No longer make this an internal variable
#m.forceIntVars(['f_state'])

# Model objects store trajectories internally, so have a different signature for 'compute'
m.compute(trajname='force', tdata=[0,t_end], ics=icdict_hybrid)
p=m.query('pars')
pts_on2 = m.sample('force')
m.compute(trajname='noforce', tdata=[0,t_end], pars={'t_on':t_end+1}, ics=icdict_hybrid)
pts_off2 = m.sample('noforce')
plot(pts_on2['t'], pts_on2['x'], 'g')
plot(pts_off2['t'], pts_off2['x'], 'r')

evts = m.getTrajEvents('force')
assert len(evts['sw_OFF']) == len(evts['sw_ON']) == 1
assert abs(evts['sw_ON']['t'][0] - p['t_on']) < evtol, "Event detection was inaccurate"
show()

print("Red trajectory shows unforced behavior. Green trajectory shows brief forcing...")
print("Now zoom in closely, to roughly the interval t = [%.3f, %.3f]"%(pars['t_on']-0.1,pars['t_on']+0.1))
print("The solid trajectory switches within +/-%.5f of the intended switch time\nt_on=%.2f"%(on_event_args['eventtol'],pars['t_on']))
