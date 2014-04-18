"""A demonstration of a PRC calculation for a Morris-Lecar
   oscillator using AUTO. Model is defined according to
   Govaerts and Sautois, "Computation of the phase response
   curve: a direct approach", Neural Computation, 2006.

   Robert Clewley, Jun 2007.
"""

from PyDSTool import *
from PyDSTool.Toolbox.adjointPRC import *
from copy import copy

# ------------------------------------------------------------


def makeMLneuron(name, dt, par_args, ic_args, evs=None,
                 special=None):
    vfn_str = '(I-ionic(v,n))/C'
    nfn_str = 'tau_n(v)*(n_inf(v)-n)'

    auxdict = {'ionic': (['vv', 'nn'],
        'gca*m_inf(vv)*(vv-vca) + gk*nn*(vv-vk) + gl*(vv-vl)'),
               'tau_n': (['v'], 'cosh(0.5*(v-v3)/v4)/15'),
               'm_inf': (['v'], '0.5*(1+tanh((v-v1)/v2))'),
               'n_inf': (['v'], '0.5*(1+tanh((v-v3)/v4))')}

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'n': nfn_str}
    DSargs.pars = par_args
    DSargs.fnspecs = auxdict
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
    return Generator.Radau_ODEsystem(DSargs)


# ------------------------------------------------------------


print('-------- Test: PRC on Morris-Lecar system')
par_args = {'gca': 4, 'gk': 8, 'gl': 2,
            'vca': 120, 'vk': -80, 'vl': -60,
            'v1': -1.2, 'v2': 18, 'v3': 4, 'v4': 17.4,
            'I': 45, 'C': 5.0}
# very close to periodic orbit
ic_args = {'v': -8.00000000687, 'n': 0.0397970934768}

thresh_ev = Events.makeZeroCrossEvent('v+50', 1,  # increasing direction only
                                   {'name': 'thresh_ev',
                                    'eventdelay': 1e-3,
                                    'eventtol': 1e-8,
                                    'precise': True,
                                    'term': False}, ['v'],
                                      targetlang='c')

print("Making ML neuron")
ML = makeMLneuron('ML_PRCtest', 0.01, par_args, ic_args, [thresh_ev])
ML.set(tdata=[0, 500])

print("Computing long orbit to converge to limit cycle")
MLtraj = ML.compute('test')
evt=ML.getEventTimes()['thresh_ev']
MLpts = MLtraj.sample()

assert len(evt) > 2
##    T=evt[-1]-evt[-2]
ix_lo = MLpts.findIndex(evt[-2])
ix_hi = MLpts.findIndex(evt[-1])
try:
    # ensure only
    po = MLpts[ix_lo:ix_hi+1]  # include last point
except ValueError:
    raise RuntimeError("Probably not enough max_pts for integrator, or not"
                       " enough time integrated to end in a complete cycle")
po[-1] = po[0]
po.indepvararray -= MLpts['t'][ix_lo]

##dts=[po['t'][i]-po['t'][i-1] for i in range(1,len(po))]
##print "dt stats: ", min(dts), max(dts), mean(dts), std(dts)

plot(po['t'],po['v'])
show()

print("Calling adjoint PRC calculator")
PRCdata = adjointPRC(ML, po, 'v', 'I', numIntervals=200, numCollocation=5,
                     spike_est=50, doPlot=True, saveData=False, verbosity=1)
