"""Integrate-and-Fire network with delayed inhibitory pulse coupling.
  (Demonstrates stable synchronous solution in these circumstances.)

   Uses event queues and event-activating events to create an "event-based"
   simulation of delayed delta-function pulse coupling between neurons.

   Effect of the coupling on a target neuron is an instantaneous
   offset of the target variable's value at the arrival time of the
   source neuron's spike event.

   PLEASE NOTE: Future releases will improve the ease of specification for
   spike delayed events. This script is merely a proof-of-concept!!

   Robert Clewley, October 2005.
"""

from PyDSTool import *
from time import perf_counter

# ---------------------------------------------------------------------------

restart(2)

targetGen = 'Vode_ODEsystem'

# -------------------------------------------------------------------------

if targetGen == 'Vode_ODEsystem':
    targetlang='python'
else:
    targetlang='c'

class IFnetwork(Component):
    compatibleGens=('Vode_ODEsystem', 'Dopri_ODEsystem')
    targetLangs=(targetlang,)

class IF(LeafComponent):
    compatibleGens=('Vode_ODEsystem', 'Dopri_ODEsystem')
    targetLangs=(targetlang,)

IFnetwork.compatibleSubcomponents=(IF,)
IF.compatibleContainers=(IFnetwork,)

def makeIFneuron(name, par_args):
    v=Var("(I - gl*(V-vl))/C", 'V', domain=[-200,-50], specType='RHSfuncSpec')
    for parname in ['vl', 'gl', 'I', 'C']:
        assert parname in par_args, "Essential pars missing"
    gl = Par(str(par_args['gl']), 'gl')
    vl = Par(str(par_args['vl']), 'vl')
    I = Par(str(par_args['I']), 'I')
    C = Par(str(par_args['C']), 'C')
    # spike threshold event
    thresh_ev_args = {'name': name+'_thresh',
               'eventtol': 1e-4,
               'eventdelay': 1e-5,
               'starttime': 0,
               'term': True,
               'evpars': {'connected': []}
                }
    thresh_ev = Events.makeZeroCrossEvent(name+'_V-threshval', 1,
                                          thresh_ev_args,
                                          varnames=[name+'_V'],
                                          parnames=['threshval'],
                                          targetlang=targetlang)
    thresh_ev.createQ(name)
    # spike arrival event
    sparr_ev_args = {'name': name+'_sparr',
               'eventtol': 1e-4,
               'eventdelay': 1e-5,
               'starttime': 0,
               'active': False,
               'term': True
                }
    sparr_ev = Events.makeZeroCrossEvent("globalindepvar(t)-" + name + \
                                                "_nextspikearr",
                                         1, sparr_ev_args,
                                         parnames=[name+"_nextspikearr"],
                                         targetlang=targetlang)
    sparr_ev.createQ(name)
    nextT = Par('0', "nextspikearr")

    IFneuron = IF(name)
    IFneuron.add([v, gl, vl, I, C, nextT])
    IFmodelC.addEvents('IFnet', [thresh_ev, sparr_ev])
    return IFneuron


def connectIFneurons(source, target):
    # add to 'connected' list of source gen's thresh event's evpars dict
    for ev in IFmodelC._events['IFnet']:
        if ev.name == source.name+'_thresh':
            ev.evpars['connected'].append(target.name)
    thresh_epmapping=EvMapping(defString="""xdict['"""+source.name+"""_V']=pdict['Vreset']
targets = estruct.events['"""+source.name+"""_thresh'].evpars['connected']
for targ in targets:
  estruct.setActiveFlag(targ+'_sparr', True)
  estruct.events[targ+'_sparr'].addToQ(targ, t+pdict['delay'])
  pdict[targ+'_nextspikearr'] = estruct.events[targ+'_sparr'].queues[targ][0]""")
    sparr_epmapping=EvMapping(defString="""xdict['"""+target.name+"""_V']=xdict['"""+target.name+"""_V']+pdict['syn_sign']*pdict['syn_strength']
try:
  estruct.events['"""+target.name+"""_sparr'].popFromQ('"""+target.name+"""')
  pdict['"""+target.name+"""_nextspikearr'] = estruct.events['"""+target.name+"""_sparr'].queues['"""+target.name+"""'][0]
except IndexError:
  estruct.setActiveFlag('"""+target.name+"""_sparr', False)""")

    IFmodelC.mapEvent('IFnet', source.name+'_thresh', 'IFnet', thresh_epmapping)
    IFmodelC.mapEvent('IFnet', target.name+'_sparr', 'IFnet', sparr_epmapping)

# -------------------------------------------------------------------------

print('-------- IF network test with delayed pulse coupling')

print("Building coupled IF model\n")

delay = Par('9', 'delay')
vreset = Par('-90', 'Vreset')
threshval = Par('-58', 'threshval')
syn_sign = Par('1', 'syn_sign')
syn_strength = Par('0.2', 'syn_strength')

IFmodelC = ModelConstructor('IFmodel', indepvar=('t',[0,10000]),
                          eventtol=1e-2)


IFmspec = IFnetwork('IFnet')
IFmspec.add([delay, vreset, threshval, syn_sign, syn_strength])

genAlgPars = {'rtol': 1e-2, 'atol': 1e-3, 'init_step': 0.01}
if targetGen != 'Vode_ODEsystem':
    genAlgPars['max_step'] = 0.5

IFmodelC.addModelInfo(IFmspec, targetGen,
                    genAlgPars=genAlgPars)

par_args = {'vl': -67, 'gl': 0.1, 'C': 1, 'I': 1.3}
IF1 = makeIFneuron('IF1', par_args)
IF2 = makeIFneuron('IF2', par_args)
connectIFneurons(IF1, IF2)
connectIFneurons(IF2, IF1)
IFmspec.add([IF1, IF2])
IFmodel=IFmodelC.getModel()

# -------------------------------------------------------------------------

print("Computing trajectory...\n")
icdict = {'IF1.V': -75, 'IF2.V': -85}
def test():
    IFmodel.compute(trajname='test',
                    tdata=[0, 300],
                    ics=icdict,
                    verboselevel=0)
start = perf_counter()
test()
print('... finished in %.3f seconds.\n' % (perf_counter()-start))

# -------------------------------------------------------------------------

plotData = IFmodel.sample('test', dt=0.1)

# make sure these are both defined by touching them
IFmodel.getTrajEventTimes('test')['IF1_thresh']
IFmodel.getTrajEventTimes('test')['IF2_thresh']

plt.ylabel('V')
plt.xlabel('t')
vline1 = plt.plot(plotData['t'], plotData['IF1.V'])
vline1 = plt.plot(plotData['t'], plotData['IF2.V'])
plt.show()
