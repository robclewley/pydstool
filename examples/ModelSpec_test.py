"""ModelSpec and neural computation toolbox test -- point neurons.
    Demonstrates noisy external input signal and a calcium-based channel.

 Robert Clewley, September 2005
"""

from PyDSTool import *
from PyDSTool.Toolbox.neuralcomp import *
from copy import copy

targetGen = 'Vode_ODEsystem'
noauxs = True  # would use True to create auxiliaries for dominant scale analysis

# -------------------------------------------------------------------------

if targetGen == 'Vode_ODEsystem':
    targetlang='python'
else:
    targetlang='c'

# -------------------------------------------------------------------------

print("ModelSpec and neural computation toolbox tests...")

def make_noise_signal(dt, t_end, mean, stddev, num_cells, seed=None):
    """Helper function: Gaussian white noise at sample rate = dt for 1 or more cells,
    for a duration of t_end."""
    if seed is not None:
        np.random.seed(seed)
    N = int(ceil(t_end*1./dt))
    t = linspace(0, t_end, N)
    coorddict = {}
    for cellnum in range(num_cells):
        coorddict['noise%i' % (cellnum+1)] = np.random.normal(0, stddev, N)
    vpts = Pointset(coorddict=coorddict, indepvararray=t)
    return pointset_to_vars(vpts, discrete=False)

def make_biexp(tau1, tau2):
    if tau1 == tau2:
        def s(t):
            return t*exp(-t/tau1)
    else:
        def s(t):
            return (exp(-t/tau1) - exp(-t/tau2))/(tau1-tau2)
    return s

def convolve_biexp(sfunc, ixs, train, ts, dt, tau1, tau2):
    tau = max([tau1, tau2])
    max_t = 10*tau
    sig = zeros((len(ts),),float)
    #sig_ts = linspace(0, max(train), len(sig))
    for i, t in enumerate(train):
        ix = ixs[i]
        tmax = min([t+max_t, max(ts)])
        last_ix = argwhere(ts >= tmax)[0][0]
        trange = ts[ix:last_ix] - t
        gs = sfunc(trange)
        sig[ix:ix+len(gs)] += gs
    return sig

def make_Poisson_signal(dt, t_end, mean, tau1, tau2, num_cells, seed=None):
    """Helper function: biexponential post-synaptic conductance for a
    Poisson spike distribution at sample rate = dt for 1 or more cells,
    for a duration of t_end. The mean is expected number of spikes per
    time unit."""
    if seed is not None:
        np.random.seed(seed)
    N = ceil(t_end*1./dt)
    t = linspace(0, t_end, N)
    sfunc = make_biexp(tau1, tau2)
    coorddict = {}
    for cellnum in range(num_cells):
        rand_vals = np.random.random_sample(N)
        spike_ixs = argwhere(rand_vals < mean*dt).flatten()
        spike_ts = t[spike_ixs]
        f = convolve_biexp(sfunc, spike_ixs, spike_ts, t, dt, tau1, tau2)
        coorddict['postsyn%i' % (cellnum+1)] = f
    vpts = Pointset(coorddict=coorddict, indepvararray=t)
    return pointset_to_vars(vpts, discrete=False)


# largest time ever needed in model
t_max = 200  # ms

v = Var(voltage)
ma = 0.32*(v+54)/(1-Exp(-(v+54.)/4))
mb = 0.28*(v+27)/(Exp((v+27.)/5)-1)
ha = .128*Exp(-(50.+v)/18)
hb = 4/(1+Exp(-(v+27.)/5))

# gamma specifies the input set contributors for each variable involved,
# and can be used for dominant scale analysis (see HH_DSSRT_test.py), although
# is unused here.
# gamma1 is for terms whose dynamics depend on itself, gamma2 is for the rest.
# ie. in RHS for v, m appears in a voltage-dependent term, g*m^3*h*(v-vrev)
# so m appears in gamma1 for v.
# As m' depends on v and m, v appears in gamma1 for m.
channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                'h', False, ha, hb, 1, vrev=50, g=100,
                                noauxs=noauxs, gamma1=args(voltage=('m',),
                                                           m=(voltage,)))

na = .032*(v+52)/(1-Exp(-(v+52.)/5))
nb = .5*Exp(-(57.+v)/40)
channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4,
                               vrev=-100, g=99,
                               noauxs=noauxs, gamma1=args(voltage=('n',),
                                                          n=(voltage,)))

# bias current input to v's RHS does not depend on v, so goes in gamma2.
# this is the name of the parameter in this case.
channel_Ib1 = makeBiasChannel('Ib', 2.1, noauxs=noauxs,
                              gamma2=args(voltage=('Ibias',)))

# leak current input to v's RHS does depend on v, so the input goes in gamma1.
# this is the name of a dummy 'activation'-like variable for leak (always = 1).
channel_Lk1 = makeChannel_rates('Lk', vrev=-67, g=0.1, noauxs=noauxs,
                                gamma1=args(voltage=('leak',)))

# Set up an intracellular Calcium store and associated slow current
# (this is basically made up, but follows the formalism and kinetics of some
# invertebrate models)
ECa_fun = Fun('12.2396*log(13000./ca)', ['ca'], 'ECa')
ca = Var('Ca_conc')
ca_tau = Par('500', 'ca_tau')

# ca_inf is only being used to make a symbolic expression in the first version
ca_inf = Fun('1./(1+exp(-(x+25.)/7.2))', ['x'], 'ca_inf')

# this one is being declared to the channel, and is used at runtime
ICa_fun = Fun('g*mx*mx*mx*(vx-vc)', ['g', 'mx', 'vx', 'vc'], 'ica_fun')

Ca_conc_RHS = Var( (ca_inf(ICa_fun('g', 'm', v, 'ECa(Ca_conc)')) - ca) / ca_tau,
                   name=ca.name, domain=[0,500], specType='RHSfuncSpec')

##     --------------------------------------------------------------------
## For educational purposes, here are two other ways to specify the same meaning, using
##   the symbolic objects differently ...

## Equivalent syntax, with the results of ca_inf and ICa_fun substituted into a string
##   literal...
#Ca_conc_RHS = Var( '(1./(1+exp(-(g*m*m*m*(V-ECa(Ca_conc))+25.)/7.2)) - Ca_conc) / ca_tau',
#                   name=ca.name, domain=[0,500], specType='RHSfuncSpec')

## Equivalent meaning, again using only a string literal, but using the functions to be
##   eval'd at runtime: requires ca_inf and ICa_fun python objects to be added to
##   parlist in makeChannel_halfact call below, i.e.
##      parlist=[Ca_conc_RHS, ca_inf, ICa_fun, ECa_fun, ca_tau])
#Ca_conc_RHS = Var( '(ca_inf(ica_fun(g, m, V, ECa(Ca_conc))) - Ca_conc) / ca_tau',
#                   name=ca.name, domain=[0,500], specType='RHSfuncSpec')
##     --------------------------------------------------------------------

minf = 1./(1+Exp(-(v+22.)/8.5))
taum = 16-13.1/(1+Exp(-(v+25.1)/26.4))
channel_ICa1 = makeChannel_halfact('Ca', voltage, 'm', False,
                                   minf, taum, vrev=ECa_fun('Ca_conc'), g=0.2,
                                   parlist=[Ca_conc_RHS, ECa_fun, ca_tau],
                                   noauxs=noauxs, gamma1=args(voltage=('m'),
                                                              m=(voltage,'Ca_Conc')))

## Alternative type of external input
#noisy_current = makeExtInputCurrentChannel('noise', noauxs=noauxs,
#                       gamma2=args(voltage=('noise',)))
#Isignal_vardict = make_signal(0.1, t_max, 0, 4, 2)

noisy_current = makeExtInputConductanceChannel('noise', vrev=-70, g=3,
                               noauxs=noauxs, gamma2=args(voltage=('noise',)))
Isignal_vardict = make_noise_signal(0.1, t_max, 0.5, 0.25, 2)

## Example Poisson signal with on average 0.05 spikes per ms
#Isignal_vardict = make_Poisson_signal(0.1, t_max, 0.05, 0.5, 5, 1)
#v=Isignal_vardict['postsyn1']
#pts = v.getDataPoints()
#plot(pts['t'],pts['postsyn1'])


HHcell1 = makeSoma('cell1', v, channelList=[channel_Lk1, channel_Ib1,
                   channel_Na1, channel_K1, channel_ICa1, noisy_current], C=1.5,
                   noauxs=noauxs)

# copy instead of recreating everything again
HHcell2 = copy(HHcell1)
HHcell2.rename('cell2')

syn12 = connectWithSynapse('s12', 'inh', HHcell1, HHcell2, g=1)
syn21 = connectWithSynapse('s21', 'inh', HHcell2, HHcell1, g=0.5)

# Each half of the gap junction will reside with each cell. As such a
# synapse is inherently bidirectional, the two halves are made, with
# names '_1' and '_2' appended. There are no gating variables created
# so this call returns nothing.
# The conductance parameter g will exist in each cell separately.
connectWithSynapse('gapjunc', 'gap', HHcell1, HHcell2, g=0.1)

HHnet = makePointNeuronNetwork('HHnet', [HHcell1, HHcell2, syn12, syn21])

# ensure HHnet has a copy of its flattened spec for use in making event
# (otherwise it is not necessary)
HHnet.flattenSpec(ignoreInputs=True)

# build an event that picks out when RHS of cell1's Voltage eqn is 0
# i.e. when dV/dt=0

stat_ev_args = {'name': 'cell1_stat',
               'eventtol': 1e-3,
               'eventdelay': 1e-3,
               'starttime': 0,
               'term': False
                }
# stationary event => dv/dt = 0
stat_ev = Events.makeZeroCrossEvent(HHnet.flatSpec['vars']['cell1_V'],
                        0, stat_ev_args, targetlang=targetlang,
                        flatspec=HHnet.flatSpec)

alg_args = {'init_step':0.15}
# Naming is hierarchical based on the structure of the HHnet ModelSpec
# but the outermost component name ('HHnet') is redundant and is not used.
ic_args_net = {'cell1.V':-68.0, 'cell1.Na.m': 0.2,
               'cell1.Na.h': 1, 'cell1.K.n': 0,
               'cell1.Ca.m': 0, 'cell1.Ca.Ca_conc': 0.5,
               'cell2.V':-79.0, 'cell2.Na.m': 0,
               'cell2.Na.h': 1, 'cell2.K.n': 0.3,
               'cell2.Ca.m': 0, 'cell2.Ca.Ca_conc': 1.5,
               's12.s_cell1_cell2': 0, 's21.s_cell2_cell1': 0}
modelC_net = ModelConstructor('HHnet_model',
                          generatorspecs={HHnet.name: {'modelspec': HHnet,
                                                'target': targetGen,
                                                'algparams': alg_args}},
                          indepvar=('t',[0,t_max]),
                          inputs={'cell1.noise.ext_input': Isignal_vardict['noise1'],
                                  'cell2.noise.ext_input': Isignal_vardict['noise2']},
                          parvalues={'cell1.s21.g': 0.3,
                                     'cell2.s12.g': 0.35},
                          eventtol=1e-5)
modelC_net.addEvents('HHnet', stat_ev)
HHmodel_net = modelC_net.getModel()

### This test just checks that a correct model is built

verboselevel = 0
# print "Computing trajectory using verbosity level %d..."%verboselevel
## don't extend tdata past [0, t_max]
HHmodel_net.compute(trajname='test',
                     tdata=[0, 60],
                     ics=ic_args_net,
                     verboselevel=verboselevel)

v_dat = HHmodel_net.sample('test')
plt.figure()
v1line = plt.plot(v_dat['t'], v_dat['cell1.V'])
v2line = plt.plot(v_dat['t'], v_dat['cell2.V'])
plt.show()

print("   ... passed")
