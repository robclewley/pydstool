"""A demonstration of a PRC calculation for a Hodgkin-Huxley
   oscillator using AUTO. Model is defined according to
   Govaerts and Sautois, "Computation of the phase response
   curve: a direct approach", Neural Computation, 2006.

   Robert Clewley, Jun 2007.
"""

from PyDSTool import *
from PyDSTool.Toolbox.adjointPRC import *
from copy import copy

# ------------------------------------------------------------


def makeHHneuron(name, dt, par_args, ic_args, evs=None,
                 special=None):
    vfn_str = '(I-ionic(v,m,h,n))/C'
    mfn_str = 'ma(v)*(1-m)-mb(v)*m'
    nfn_str = 'na(v)*(1-n)-nb(v)*n'
    hfn_str = 'ha(v)*(1-h)-hb(v)*h'
    aux_str = 'm*m*m*h'

    auxdict = {'ionic': (['vv', 'mm', 'hh', 'nn'],
        'gna*mm*mm*mm*hh*(vv-vna) + gk*nn*nn*nn*nn*(vv-vk) + gl*(vv-vl)'),
               'ma': (['v'], 'psi_m(v)/(exp(psi_m(v))-1)'),
               'mb': (['v'], '4*exp(-v/18.)'),
               'ha': (['v'], '0.07*exp(-v/20.)'),
               'hb': (['v'], '1/(1+exp((30-v)/10.))'),
               'na': (['v'], '0.1*psi_n(v)/(exp(psi_n(v))-1)'),
               'nb': (['v'], '0.125*exp(-v/80.)'),
               'psi_m': (['v'], '(25-v)/10.'),
               'psi_n': (['v'], '(10-v)/10.')}#,
#               'phi': ([], '3.**((T-6.3)/10)')}

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'm': mfn_str,
                       'h': hfn_str, 'n': nfn_str}
    DSargs.pars = par_args
    DSargs.fnspecs = auxdict
    DSargs.xdomain = {'v': [-130, 130], 'm': [0,1], 'h': [0,1], 'n': [0,1]}
    DSargs.algparams = {'init_step':dt, 'max_step': dt*5,
                        'max_pts': 300000, 'refine': 1}
    if special is not None:
        DSargs.algparams['use_special'] = True
        DSargs.algparams['specialtimes'] = special
##        DSargs.algparams['max_step'] = max(special)
##        DSargs.algparams['init_step'] = special[1]-special[0]
##        DSargs.algparams['refine'] = 0
##        DSargs.algparams['fac1'] = 1
##        DSargs.algparams['fac2'] = 1
    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = name
    if evs is not None:
        DSargs.events = evs
    return Generator.Dopri_ODEsystem(DSargs)


# ------------------------------------------------------------


print('-------- Test: PRC on Hodgkin-Huxley system')
par_args = {'gna': 120, 'gk': 36, 'gl': 0.3,
            'vna': 115, 'vk': -12, 'vl': 10.559,
            'T': 6.3, 'I': 12, 'C': 1.0}
# very close to periodic orbit
ic_args = {'h': 0.295581037525, 'v': -8.00000000013,
           'm': 0.0193125517095, 'n': 0.538645970873}

thresh_ev = Events.makeZeroCrossEvent('v+8', 1,  # increasing direction only
                                   {'name': 'thresh_ev',
                                    'eventdelay': 1e-3,
                                    'eventtol': 1e-8,
                                    'precise': True,
                                    'term': False}, ['v'],
                                      targetlang='c')

print("Making HH neuron")
HH = makeHHneuron('HH_PRCtest', 0.01, par_args, ic_args, [thresh_ev],
                  special=linspace(0,100,100/0.001))
HH.set(tdata=[0, 100])

print("Computing long orbit to converge to limit cycle")
HHtraj = HH.compute('test')
evt=HH.getEventTimes()['thresh_ev']
HHpts = HHtraj.sample()

assert len(evt) > 2
##    T=evt[-1]-evt[-2]
ix_lo = HHpts.findIndex(evt[-2])
ix_hi = HHpts.findIndex(evt[-1])
try:
    # ensure only
    po = HHpts[ix_lo:ix_hi+1]  # include last point
except ValueError:
    raise RuntimeError("Probably not enough max_pts for integrator, or not"
                       " enough time integrated to end in a complete cycle")
po[-1] = po[0]
po.indepvararray -= HHpts['t'][ix_lo]

##dts=[po['t'][i]-po['t'][i-1] for i in range(1,len(po))]
##print "dt stats: ", min(dts), max(dts), mean(dts), std(dts)

plot(po['t'],po['v'])
show()

print("Calling adjoint PRC calculator")
PRCdata = adjointPRC(HH, po, 'v', 'I', numIntervals=300, numCollocation=5,
                     spike_est=10, doPlot=True, saveData=False, verbosity=1)
