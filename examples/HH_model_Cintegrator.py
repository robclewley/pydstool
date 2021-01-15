"""A demonstration of a model for a single Hodgkin-Huxley membrane
potential for an oscillatory cortical neuron. (Includes demo of saving
a Model object that uses a Dopri integrator).

Inclusion of additional libraries in C code is also tested.

   Robert Clewley, June 2005.
"""

# textually substitute 'Dopri' for 'Radau' in this file to use Radau

from PyDSTool import *
from time import perf_counter


# ------------------------------------------------------------


def makeHHneuron(name, par_args, ic_args, evs=None, extra_terms='', nobuild=False):
    # extra_terms must not introduce new variables!
    vfn_str = '(I'+extra_terms+'-ionic(v,m,h,n))/C'
    mfn_str = 'ma(v)*(1-m)-mb(v)*m'
    nfn_str = 'na(v)*(1-n)-nb(v)*n'
    hfn_str = 'ha(v)*(1-h)-hb(v)*h'
    aux_str = 'm*m*m*h'

    auxdict = {'ionic': (['vv', 'mm', 'hh', 'nn'],
                              'gna*mm*mm*mm*hh*(vv-vna) + gk*nn*nn*nn*nn*(vv-vk) + gl*(vv-vl)'),
               'ma': (['v'], '0.32*(v+54)/(1-exp(-(v+54)/4))'),
               'mb': (['v'], '0.28*(v+27)/(exp((v+27)/5)-1)'),
               'ha': (['v'], '.128*exp(-(50+v)/18)'),
               'hb': (['v'], '4/(1+exp(-(v+27)/5))'),
               'na': (['v'], '.032*(v+52)/(1-exp(-(v+52)/5))'),
               'nb': (['v'], '.5*exp(-(57+v)/40)')}

    DSargs = {}
    DSargs['varspecs'] = {'v': vfn_str, 'm': mfn_str,
                             'h': hfn_str, 'n': nfn_str
                             }
    DSargs['pars'] = par_args
    DSargs['fnspecs'] = auxdict
    DSargs['xdomain'] = {'v': [-130, 70], 'm': [0,1], 'h': [0,1], 'n': [0,1]}
    DSargs['algparams'] = {'init_step': 0.1, 'refine': 0, 'max_step': 0.5}
    DSargs['checklevel'] = 0
    DSargs['ics'] = ic_args
    DSargs['name'] = name
    # nobuild=True so that we can test additional library inclusion
    if nobuild:
        DSargs['nobuild']=True
    if evs is not None:
        DSargs['events'] = evs
    return Generator.Dopri_ODEsystem(DSargs)


# ------------------------------------------------------------


print('-------- Test: Hodgkin-Huxley system')
par_args = {'gna': 100, 'gk': 80, 'gl': 0.1,
            'vna': 50, 'vk': -100, 'vl': -67,
            'I': 1.75, 'C': 1.0}
ic_args = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}

# test single terminal event first
print("Testing single terminal event and its sampling")

thresh_ev = Events.makeZeroCrossEvent('v', 1,
                                {'name': 'thresh_ev',
                                 'eventtol': 1e-4,
                                 'term': False}, ['v'],
                                      targetlang='c')
thresh_ev_term = Events.makeZeroCrossEvent('v', 1,
                              {'name': 'thresh_ev',
                               'eventtol': 1e-4,
                               'term': True}, ['v'],
                                    targetlang='c')

HH_term = makeHHneuron('HHtest', par_args, ic_args, [thresh_ev_term],
                       nobuild=True)
# test inclusion of other libraries in C file (not used in this example!)
HH_term.makeLib(include=['limits.h'])
print("Successfully tested inclusion of additional C libraries into vector")
print("field definition code.")
HH_term.set(tdata=[0, 25])
start = perf_counter()
HHtraj_term = HH_term.compute('test_term')
print('Computed trajectory in %.3f seconds.\n' % (perf_counter()-start))
trajdata = HHtraj_term.sample(dt=1.0)
print("Sampled this data at dt=1.0 up to the event", HH_term.getEventTimes(), ":")
print(trajdata['v'], "\n")

# HH is a "Generator" object (an ODE in this case)
# (Generator is the new name for the DynamicalSystem class, because some
# subclasses, for instance ones for trajectories given by explicit
# equations, are not dynamical systems!)
HH = makeHHneuron('HH_model_test', par_args, ic_args, [thresh_ev])
HH.set(tdata=[0, 50])

print('Integrating...')
start = perf_counter()
HHtraj = HH.compute('test')
print('  ... finished in %.3f seconds.\n' % (perf_counter()-start))
plotData = HHtraj.sample(dt=0.1)
evt=HH.getEventTimes()['thresh_ev']

print('Saving Model and Trajectory...')
saveObjects([HH, HHtraj], 'temp_HH_Cintegrator.pkl', True)

print('Testing continued integration')
new_t0 = HHtraj.indepdomain[1]
HH.set(tdata=[new_t0,new_t0+20])
HHtraj2 = HH.compute('test_cont', 'c')
print("Non-terminal events found:", HH.getEvents())

print('Preparing plot')
plotData2 = HHtraj2.sample(dt=0.1)
evt2=HH.getEventTimes()['thresh_ev']
yaxislabelstr = 'v'
plt.ylabel(yaxislabelstr)
plt.xlabel('t')
vline=plot(plotData['t'], plotData['v'])
vline2=plot(plotData2['t'], plotData2['v'])
plot(evt, HHtraj(evt, 'v'), 'ro')
plot(evt2, HHtraj2(evt2, 'v'), 'ro')

show()
