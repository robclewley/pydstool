"""
Demonstration of cell types and use of neurocomp.py toolbox templates for
a spinal cord neural model or commisural interneurons.

Robert Clewley, Erik Sherwood, Oct 2005
"""


from PyDSTool import *
from PyDSTool.Toolbox.neuralcomp import *
from time import perf_counter

# --------------------------------------------------------------------------

class RGNcell(soma):
    pass

class MNcell(soma):
    pass

class CINcell(soma):
    pass

celltypes=[RGNcell, MNcell, CINcell]

# test of inherited sub-components
c=MNcell('c')
print("Inherited compatible sub-components of an MN cell are:")
allcompats = [className(csub) for csub in c._allSubcomponentTypes]
print(allcompats, "\n")
assert allcompats==['channel', 'Par', 'Var', 'Input', 'Fun', 'Component']

def gaussianICs(varnames, mu, sigma):
    return dict(zip(varnames, [random.gauss(mu, sigma) for v in varnames]))


def uniformICs(varnames, a, b):
    return dict(zip(varnames, [random.uniform(a, b) for v in varnames]))


defICs_soma = {
    RGNcell: {'V': -45, 'Na.h': 1, 'K.n': 0, 'NaP.mp': 0, 'NaP.hp': 0.5,
              'Na.m': 0},
    MNcell: {'V': -62, 'Na.h': 1, 'K.n': 0, 'Na.m': 0},
    CINcell: {'V': -62, 'Na.h': 1, 'K.n': 0}
    }

def defaultICs(cell):
    defICs = defICs_soma[type(cell)]
    ics = {}
    for k, v in defICs.items():
        ics[cell.name+'.'+k] = v
    return ics


# --------------------------------------------------------------------------


def Sigma(vo, theta, k):
    return 1.0 / ( 1.0 + Exp( ( vo - theta ) / k ) )

def Beta(vo, theta, k, c):
    return c / ( Cosh((vo-theta)/(2*k)) )

# switch to include auxiliary variables that measure timescales and
# asymptotic targets of all variables (slows down computations)
noauxs = True
auxwarning_part="Building model with auxiliary variables"
if noauxs:
    print("Not "+auxwarning_part.lower())
else:
    print(auxwarning_part)

targetGen = 'Dopri_ODEsystem'
print("Target integrator is "+targetGen)
print("------------------------------------")

# measure time to build model
t0 = perf_counter()

# Motor cell
MNtheta_m = makePar('theta_m', -52)
MNk_m = makePar('k_m', -5)
MNminf = Sigma(V, MNtheta_m, MNk_m)
MNhinf = '1-K.n'
channel_Na_MN = makeChannel_halfact('Na', voltage, s='m', isinstant=True,
                                    sinf=MNminf, spow=3,
                                    s2='h', isinstant2=True,
                                    sinf2=MNhinf,
                                    vrev=55, g=28, noauxs=noauxs,
                                    parlist=[MNtheta_m, MNk_m])

MNtheta_n = makePar('theta_n', -43)
MNk_n = makePar('k_n', -4)
MNninf = Sigma(V, MNtheta_n, MNk_n)
MNtaun_bar = makePar('taun_bar', 10)
MNtaun = Beta(V, MNtheta_n, MNk_n, MNtaun_bar)
channel_K_MN = makeChannel_halfact('K', voltage,
                                   s='n', sinf=MNninf, taus=MNtaun, spow=4,
                                   vrev=-85, g=11.2, noauxs=noauxs,
                                   parlist=[MNtheta_n, MNk_n, MNtaun_bar])

channel_Iapp_MN = makeBiasChannel('Iapp', 0, noauxs=noauxs)
channel_Lk_MN = makeChannel_rates('Lk', vrev=-65, g=2.8, noauxs=noauxs)

MNcell1 = makeSoma('MNcell1', channelList=[channel_Lk_MN, channel_Iapp_MN,
                   channel_Na_MN, channel_K_MN], C=21, noauxs=noauxs,
                   subclass=MNcell)

# RGN cell
RGNtheta_m = makePar('theta_m', -34)
RGNk_m = makePar('k_m', -5)
RGNminf = Sigma(V, RGNtheta_m, RGNk_m)
RGNhinf = '1-K.n'
channel_Na_RGN = makeChannel_halfact('Na', voltage,
                               s='m', isinstant=True, sinf=RGNminf, spow=3,
                               s2='h', isinstant2=True, sinf2=RGNhinf,
                               vrev=55, g=28, noauxs=noauxs,
                               parlist=[RGNtheta_m, RGNk_m])

RGNtheta_n = makePar('theta_n', -29)
RGNk_n = makePar('k_n', -4)
RGNninf = Sigma(V, RGNtheta_n, RGNk_n)
RGNtaun_bar = makePar('taun_bar', 10)
RGNtaun = Beta(V, RGNtheta_n, RGNk_n, RGNtaun_bar)
channel_K_RGN = makeChannel_halfact('K', voltage,
                                    s='n', sinf=RGNninf, taus=RGNtaun, spow=4,
                                    vrev=-85, g=11.2, noauxs=noauxs,
                                    parlist=[RGNtheta_n, RGNk_n, RGNtaun_bar])


channel_Iapp_RGN = makeBiasChannel('Iapp', 5, noauxs=noauxs)
channel_Lk_RGN = makeChannel_rates('Lk', vrev=-60, g=2.8, noauxs=noauxs)

RGNtheta_mp = makePar('theta_mp', -40)
RGNk_mp = makePar('k_mp', -6.3)
RGNmpinf = Sigma(V, RGNtheta_mp, RGNk_mp)
RGNtheta_hp = makePar('theta_hp', -48)
RGNk_hp = makePar('k_hp', 9)
RGNhpinf = Sigma(V, RGNtheta_hp, RGNk_hp)
RGNtauhp_bar = makePar('tauhp_bar', 7000)
RGNtauhp = Beta(V, RGNtheta_hp, RGNk_hp, RGNtauhp_bar)
channel_NaP_RGN = makeChannel_halfact('NaP', voltage,
                              s='mp', isinstant=True, sinf=RGNmpinf,
                              s2='hp', sinf2=RGNhpinf, taus2=RGNtauhp,
                              vrev=50, g=2.8, noauxs=noauxs,
                              parlist=[RGNtheta_mp, RGNk_mp, RGNtheta_hp,
                                         RGNk_hp, RGNtauhp_bar])

RGNcell1 = makeSoma('RGNcell1', channelList=[channel_Lk_RGN, channel_Iapp_RGN,
                   channel_Na_RGN, channel_K_RGN, channel_NaP_RGN], C=21,
                   noauxs=noauxs, subclass=RGNcell)


Tmax = makePar('Tmax', 3.2)
Vp = makePar('Vp', -20)
Kp = makePar('Kp', 4)
Tinf = ('vo', Tmax*Sigma('-vo', Vp, Kp))
synRGN_MN = connectWithSynapse('sRM', '', RGNcell1, MNcell1,
                               threshfun=Tinf, alpha=1, beta=0.2,
                               threshfun_d=Tinf, alpha_d=5, beta_d=0.0002,
                               adapt_typestr='f', vrev=0, g=10, noauxs=noauxs)
synRGN_MN.add([Tmax,Vp,Kp])

# Bring network components together into one large ModelSpec component: HHnet
HHnet = makePointNeuronNetwork('HHnet', [RGNcell1, MNcell1, synRGN_MN])

t1=perf_counter()
print("Compiled network specification in %f seconds"%(t1-t0))

## uncomment for easy interactive reference at prompt
#flatspec = HHnet.flattenSpec()

t0 = perf_counter()
alg_args = {'init_step':0.5, 'max_pts': 40000}
ic_args_net = {}
ic_args_net.update(defaultICs(RGNcell1))
ic_args_net.update(defaultICs(MNcell1))
ic_args_net.update({'sRM.s_RGNcell1_MNcell1': 0,
                    'sRM.d': 0.2})
modelC_net = ModelConstructor('HHnet_model',
                          generatorspecs={'HHnet': {'modelspec': HHnet,
                                                'target': targetGen,
                                                'algparams': alg_args}},
                          indepvar=('t',[0,100000]),
                          eventtol=1e-5, withStdEvts={'HHnet': True})
CB = modelC_net.getModel()
# checklevel = 0 switches off all bounds checking if not needed (faster!)
CB.set(ics=ic_args_net, checklevel=0)

t1=perf_counter()
print("Instantiated model for target ODE solver %s in %f seconds"%(targetGen, t1-t0))

def getTraj(dt_plot=0.25, tend=1000):
    # dt_plot only controls resolution of plot data, not integration
    # timestep (which is set using algparams option)
    CB.compute(trajname='test', force=True,
                   tdata=[0, tend])
    return CB.sample('test')#, dt=dt_plot)


print("\nParameters defined: check using who(Par)")
who(Par)

print("\nCells defined: check using who(soma, deepSearch=True)")
who(soma, deepSearch=True)

# try out conversion of expression to a function of the free names
RGNtaun_fun = expr2fun(RGNtaun, theta_n=-52, k_n=-5, taun_bar=10)
print("\nRGNtaun_fun is a Python function created from the ModelSpecs using expr2fun()")
print("RGNtaun_fun(-10.) =", RGNtaun_fun(-10.))

# --------------------------------------------------------------------

# utility functions

def plotActivation(expr, Vrange, parlist=[], dV=1.):
    """parlist is a list of parameter objects or dictionary of name-bindings
    that resolve expr to a function of a single variable -- the declared
    voltage variable."""
    subs_expr = subs(expr, parlist)
    fun = expr2fun(subs_expr)
    if [voltage] != subs_expr.freeSymbols:
        raise ValueError("Expression not solely a function of voltage variable %s"%str(v))
    vvals = arange(Vrange[0], Vrange[1], dV)
    avals = array([fun(val) for val in vvals])
    plot(vvals, avals)
    show()

def plotVoltages(dataset):
    plt.figure()
    allVlist = CB.searchForVars('soma.V')
    for vname in allVlist:
        plot(dataset['t'], dataset[vname])
    show()

def plotCurrents(cell_name, dataset):
    plt.figure()
    allIlist = CB.searchForVars(cell_name+'.channel.I')
    for Iname in allIlist:
        plot(dataset['t'], dataset[Iname])
    show()

# --------------------------------------------------------------------

print("\n\nComputing trajectory and getting plot data...")
t0=perf_counter()
v_dat = getTraj()
t1=perf_counter()
print("... finished in %f seconds\n"%(t1-t0))

#plotActivation(MNtaun, [-100,30], [MNtheta_n, MNk_n, MNtaun_bar])

plotVoltages(v_dat)
plt.figure()
print("Plotting synaptic facilitation variable, sRM.d ...")
plot(v_dat['t'], v_dat['sRM.d'])

print("Setting other terminal events ...")
CB.setDSEventActive(target='HHnet', eventTarget='RGNcell1_K_n_stat_dec_evt',
                    flagVal=True)
CB.setDSEventTerm(target='HHnet', eventTarget='RGNcell1_K_n_stat_dec_evt',
                  flagVal=True)
CB.compute(trajname='test2', tdata=[0,300])
print("Computed additional trajectory.")

###### Other useful operations with CIN

##plotCurrents('RGNcell', v_dat)
##
##info(CB.query('pars'))
##
CB.showDef('HHnet', 'spec')
##
CB.showDef('HHnet', 'modelspec')
##
##CB.setPars('soma.Na.g', -1000)
##
##info(CB.query('pars'))
##
##CB.setPars('soma.channel.g', -1000)
##
##info(CB.query('pars'))
##
##CB.setICs('soma.V', -60.6)
##
##info(CB.query('initialconditions'))
##
##searchModelSpec(HHnet, 'RGNcell.Par')
##
##searchModelSpec(HHnet, 'synapse')
##
##CB.searchForNames('MNcell1.Var')
##
##synlist = CB.searchForVars('synapse.Var')
##syns_from_RGNcell1 = matchSubName(synlist, 'RGNcell1', 1, 1)

show()
