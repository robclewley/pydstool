"""A demonstration of a model for a single Hodgkin-Huxley membrane
potential for an oscillatory cortical neuron. Tests bounds events
and dynamic update of bounds between integration runs.

   Robert Clewley, September 2006.
"""

# textually substitute 'Dopri' for 'Radau' in this file to use Radau

from PyDSTool import *
from time import perf_counter


# ------------------------------------------------------------


def makeHHneuron(name, par_args, ic_args, evs=None, extra_terms='',
                 nobuild=False):
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
                          'h': hfn_str, 'n': nfn_str,
                          'v_bd0': 'getbound("v",0)',
                          'v_bd1': 'getbound("v",1)',
                          'h_bd0': 'getbound("h",0)'
                             }
    DSargs['pars'] = par_args
    DSargs['auxvars'] = ['v_bd0','v_bd1','h_bd0']
    DSargs['fnspecs'] = auxdict
    DSargs['xdomain'] = {'v': [-100, 50], 'm': [0,1], 'h': [0,1], 'n': [0,1]}
    DSargs['algparams'] = {'init_step': 0.1, 'refine': 0, 'max_step': 0.5}
    DSargs['checklevel'] = 0
    DSargs['ics'] = ic_args
    DSargs['name'] = name
    DSargs['enforcebounds'] = True
    DSargs['activatedbounds'] = {'v': [1,1]}
    # nobuild=True so that we can test additional library inclusion
    if nobuild:
        DSargs['nobuild']=True
    if evs is not None:
        DSargs['events'] = evs
    return Generator.Dopri_ODEsystem(DSargs)


# ------------------------------------------------------------


if __name__=='__main__':
    print('-------- Test: Hodgkin-Huxley system')
    par_args = {'gna': 100, 'gk': 80, 'gl': 0.1,
                'vna': 50, 'vk': -100, 'vl': -67,
                'I': 1.75, 'C': 1.0}
    ic_args = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}

    # test single terminal event first
    print("Testing bounds terminal event and its sampling")

    HH = makeHHneuron('HHtest_bdev', par_args, ic_args, [])
    HH.set(tdata=[0, 25])
    HHtraj = HH.compute('test_term')
    trajdata = HHtraj.sample(dt=0.05, tlo=1.3, thi=20)
    assert trajdata['h_bd0'][0]==0
    assert trajdata['v_bd1'][0]==50
    assert trajdata['v_bd0'][0]==-100

    print('Testing continued integration from t=25, having now set')
    print('voltage domain to be [-100,20]')
    HH.set(xdomain={'v':[-100, 20]}, tdata=[0, 50])
    HHtraj2 = HH.compute('test_cont', 'c')
    assert len(HH.getEventTimes()['v_domlo']) == 0
    assert len(HH.getEventTimes()['v_domhi']) > 0
    print("Sampled this data up until the event at t={}:".format(
          HH.getEventTimes()['v_domhi'][0]
    ))

    plotData = HHtraj2.sample(dt=0.1)
    evt = HH.getEventTimes()['v_domhi']
    yaxislabelstr = 'v'
    plt.ylabel(yaxislabelstr)
    plt.xlabel('t')
    vline = plot(plotData['t'], plotData['v'])
    plot(evt, HHtraj2(evt, 'v'), 'ro')
    show()
