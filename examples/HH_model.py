"""A demonstration of a model for a single Hodgkin-Huxley membrane
potential for an oscillatory cortical neuron. (Includes demo of saving
a Model object). Uses direct FuncSpec specification with strings.
See ModelSpec_test.py for example of using the neuralcomp toolbox.

   Robert Clewley, June 2005.
"""
from __future__ import print_function

from PyDSTool import *
from time import clock
from copy import copy

# ------------------------------------------------------------


def makeHHneuron(name, par_args, ic_args, evs=None, extra_terms='',
                 gentype='vode'):
    # extra_terms must not introduce new variables!
    vfn_str = '(Iapp'+extra_terms+'-ionic(v,m,h,n))/C'
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
               'nb': (['v'], '.5*exp(-(57+v)/40)'),
               'ptest': (['p'], '1+p+ma(-50)+C'),
               'atest': (['q'], 'q+mb(-40)')}  # tests fn cross-calling

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'm': mfn_str,
                       'h': hfn_str, 'n': nfn_str,
                       'v_bd0': 'getbound("v",0)',  # demo of referencing bounds
                       'v_bd1': 'getbound("v",1)'}
    DSargs.pars = par_args
    DSargs.auxvars = ['v_bd0','v_bd1']
    DSargs.fnspecs = auxdict
    DSargs.xdomain = {'v': [-130, 70], 'm': [0,1], 'h': [0,1], 'n': [0,1]}
    DSargs.algparams = {'init_step':0.03}
    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = name
    if evs is not None:
        DSargs.events = evs
    if gentype == 'vode':
        return Generator.Vode_ODEsystem(DSargs)
    elif gentype == 'radau':
        return Generator.Radau_ODEsystem(DSargs)
    elif gentype == 'dopri':
        return Generator.Dopri_ODEsystem(DSargs)
    else:
        raise NotImplementedError("Unknown ODE system type: %s"%gentype)


# ------------------------------------------------------------

if __name__=='__main__':
    # need the __main__ to use above functions as imports without running this part
    print('-------- Test: Hodgkin-Huxley system')
    par_args = {'gna': 100, 'gk': 80, 'gl': 0.1,
                'vna': 50, 'vk': -100, 'vl': -67,
                'Iapp': 1.75, 'C': 1.0}
    ic_args = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}

    # test single terminal event first
    print("Testing single terminal event and its sampling")

    thresh_ev = Events.makePythonStateZeroCrossEvent('v', 0, 1,
                                       {'name': 'thresh_ev',
                                        'eventtol': 1e-4,
                                        'precise': True,
                                        'term': False})

    thresh_ev_term = Events.makePythonStateZeroCrossEvent('v', 0, 1,
                                       {'name': 'thresh_ev',
                                        'eventdelay': 1e-4,
                                        'eventtol': 1e-4,
                                        'precise': True,
                                        'term': True})

    HH_term = makeHHneuron('HHtest', par_args, ic_args, [thresh_ev_term])
    HH_term.set(tdata=[0, 25])
    start = clock()
    HHtraj_term = HH_term.compute('test_term')
    print('Computed trajectory in %.3f seconds.\n' % (clock()-start))
    trajdata = HHtraj_term.sample(dt=1.0, precise=True)
    print("sampled this data up until the event", HH_term.getEventTimes(), ":")
    print(trajdata['v'], "\n")


    HH = makeHHneuron('HHtest', par_args, ic_args, [thresh_ev])

    # test user interface to auxiliary functions
    assert HH.auxfns.ptest(4) == 1+4+HH.auxfns.ma(-50)+par_args['C'], "Failure of user interface to auxiliary functions"
    HH.set(pars={'C':101})
    assert HH.auxfns.ptest(4) == 1+4+HH.auxfns.ma(-50)+101, "Failure of user interface to auxiliary functions"
    assert HH.auxfns.atest(2.4) == 2.4+HH.auxfns.mb(-40), "Failure of user interface to auxiliary functions"
    HH.set(pars={'C':par_args['C']})

    # HH is a "Generator" object (an ODE in this case)
    # (Generator is the new name for the DynamicalSystem class, because some
    # subclasses, for instance ones for trajectories given by explicit
    # equations, are not dynamical systems!)
    HH.set(tdata=[0, 6.797])

    print('Integrating...')
    start = clock()
    HHtraj = HH.compute('test')
    print('  ... finished in %.3f seconds.\n' % (clock()-start))
    evt = HH.getEventTimes()['thresh_ev']
    evt_same = HH.getEventTimes('thresh_ev')
    assert evt == [] == evt_same

    print('Saving Model and Trajectory...')
    saveObjects([HH, HHtraj], 'temp_HH.pkl', True)

    # try a longer run
    print("Trying a longer run")
    HH.set(tdata=[0, 40])
    HHtraj2 = HH.compute('test_long')
    evts=HH.getEvents()
    HH.set(tdata=[40, 60])
    HHtraj3 = HH.compute('test_long_c','c')
    print("In 50ms, found the following events:")
    evts_c=HH.getEvents()
    all_evts = copy(evts)
    for k, a in list(evts_c.items()):
        if k in evts and a is not None:
            all_evts[k].append(a)
        else:
            all_evts[k] = a
    print(all_evts)
    assert len(all_evts['thresh_ev']) == 4, "Problem with ODE events"
    assert allclose(all_evts['thresh_ev']['t'][3], 56.218, 1e-3), "Problem with ODE events"

    plotData = HHtraj.sample()  # works but not VODE accurate for non-terminal events
    yaxislabelstr = 'v'
    plt.ylabel(yaxislabelstr)
    plt.xlabel('t')
    vline = plt.plot(plotData['t'], plotData['v'])
    # pick out first event
    evt = evts['thresh_ev']['t'][0]
    plt.plot(evt, HHtraj(evt, 'v'), 'ro')
    print("Showing longer trajectory with +10mV offset, using the syntax")
    print(">>> plotData2['v'] += 10")
    plotData2 = HHtraj2.sample()
    plotData3 = HHtraj3.sample()
    plotData2['v'] += 10  # could have plotted plotData2['v']+10
    plotData3['v'] += 10
    vline2 = plt.plot(plotData2['t'], plotData2['v'])
    vline3 = plt.plot(plotData3['t'], plotData3['v']-10)
    plt.show()
