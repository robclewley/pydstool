"""Integrate-and-Fire network with delayed inhibitory pulse coupling.
   (Demonstrates stable synchronous solution in these circumstances.)
   This '_syn' version uses more realistic post-synaptic currents.

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

compatGens = ('Vode_ODEsystem', 'Dopri_ODEsystem', 'Radau_ODEsystem')

class IFnetwork(Component):
    compatibleGens=compatGens
    targetLangs=(targetlang,)

class IF(Component):
    compatibleGens=compatGens
    targetLangs=(targetlang,)

class synchannel(LeafComponent):
    compatibleGens=compatGens

IFnetwork.compatibleSubcomponents=(IF,)
IF.compatibleContainers=(IFnetwork,)
IF.compatibleSubcomponents=(synchannel,)
synchannel.compatibleContainers=(IF,)

def makeIFneuron(name, par_args, IFmodelC):
    # if internal variable 'excited' is 0 (not excited) then be leaky
    # integrator, otherwise stay dormant (RHS=0) so that events can
    # set square pulse spike.
    v=Var("if( excited==0, (I - gl*(V-vl) - for(synchannel,I,+))/C, 0 )",
          'V', domain=[-120,50], specType='RHSfuncSpec')
    exc=Var("0", 'excited', domain=[0,1], specType='RHSfuncSpec')
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
    thresh_ev = Events.makeZeroCrossEvent(name+'_V-threshval', 1, thresh_ev_args,
                                                     varnames=[name+'_V'],
                                                     parnames=['threshval'],
                                          targetlang=targetlang)
    thresh_ev.createQ(name)

    # end of spike event (in "explicit function" phase)

    spend_ev_args = {'name': name+'_spend',
                     'eventtol': 1e-4,
                     'eventdelay': 1e-5,
                     'starttime': 0,
                     'active': False,
                     'term': True
                     }
    spend_ev = Events.makeZeroCrossEvent("t-tspike", 1, spend_ev_args,
                                         parnames=["tspike"],
                                         targetlang=targetlang)

    # spike arrival event
    sparr_ev_args = {'name': name+'_sparr',
               'eventtol': 1e-4,
               'eventdelay': 1e-5,
               'starttime': 0,
               'active': False,
               'term': True
                }
    sparr_ev = Events.makeZeroCrossEvent("globalindepvar(t)-"+name+"_nextspikearr", 1, sparr_ev_args,
                                         parnames=[name+"_nextspikearr"],
                                         targetlang=targetlang)
    sparr_ev.createQ(name)
    nextT = Par('0', "nextspikearr")

    IFneuron = IF(name)
    IFneuron.add([v, gl, vl, I, C, nextT, exc])

    spend_epmapping=EvMapping(defString="""estruct.setActiveFlag('"""+name+"""_spend', False)
xdict['"""+name+"""_excited'] = 0
xdict['"""+name+"""_V'] = pdict['Vreset']""")

    IFmodelC.addEvents('IFnet_delay', [thresh_ev, sparr_ev, spend_ev])
    IFmodelC.mapEvent('IFnet_delay', name+'_spend', 'IFnet_delay', spend_epmapping)
    del thresh_ev, spend_ev, sparr_ev
    return IFneuron


def connectIFneurons(source, target, g, vrev, IFmodelC):
    # add to 'connected' list of source gen's thresh event's evpars dict
    for ev in IFmodelC._events['IFnet_delay']:
        if ev.name == source.name+'_thresh':
            ev.evpars['connected'].append(target.name)
    V=Var("V")
    s=Var("-b_syn*s", 's', domain=[0,1], specType='RHSfuncSpec')
    g=Par(str(g), 'g')
    vrev=Par(str(vrev), 'vrev')
    I=Var(g*s*(V-vrev), "I", specType='ExpFuncSpec')
    syn=synchannel('syn_'+source.name+"_"+target.name)
    syn.add([s,I,g,vrev])
    target.add(syn)

    # These event mappings work because the vector field is always the same
    # -- it's always the same Model object used each time after an event.
    # If vector fields change then the target estruct etc. have to be held
    # in an evpars field so as to be able to access them here. (estruct is only
    # the ending Model's)

    thresh_epmapping=EvMapping(defString="""xdict['"""+source.name+"""_V']=pdict['Vspike']
targets = estruct.events['"""+source.name+"""_thresh'].evpars['connected']
for targ in targets:
  estruct.setActiveFlag(targ+'_sparr', True)
  estruct.setActiveFlag('"""+source.name+"""_spend', True)
  estruct.events[targ+'_sparr'].addToQ(targ, t+pdict['delay'])
  xdict['"""+source.name+"""_excited'] = 1
  pdict[targ+'_nextspikearr'] = estruct.events[targ+'_sparr'].queues[targ][0]""")

    sparr_epmapping=EvMapping(defString="""xdict['"""+target.name+"_"+syn.name+"""_s']=1
try:
  estruct.events['"""+target.name+"""_sparr'].popFromQ('"""+target.name+"""')
  pdict['"""+target.name+"""_nextspikearr'] = estruct.events['"""+target.name+"""_sparr'].queues['"""+target.name+"""'][0]
except IndexError:
  estruct.setActiveFlag('"""+target.name+"""_sparr', False)""")

    IFmodelC.mapEvent('IFnet_delay', source.name+'_thresh', 'IFnet_delay', thresh_epmapping)
    IFmodelC.mapEvent('IFnet_delay', target.name+'_sparr', 'IFnet_delay', sparr_epmapping)

# -------------------------------------------------------------------------

print('-------- IF network test with delayed pulse coupling')

print("Building coupled IF model\n")

delay = Par('6', 'delay')
vreset = Par('-90', 'Vreset')
vspike = Par('50', 'Vspike')
tspike = Par('1', 'tspike')
threshval = Par('-55', 'threshval')
b_syn=Par('0.5', 'b_syn')  # decay rate of synaptic pulse

IFmodelC = ModelConstructor('IFmodel', indepvar=('t',[0,10000]),
                          eventtol=1e-3)

IFmspec = IFnetwork('IFnet_delay')
IFmspec.add([delay, vreset, threshval, b_syn, vspike, tspike])

genAlgPars={'rtol': 1e-2, 'atol': 1e-3, 'max_step': 0.25,
                                'init_step': 0.05}
if targetGen == 'Vode_ODEsystem':
    # Vode fails if don't use stiff option
    genAlgPars['stiff']=True

IFmodelC.addModelInfo(IFmspec, targetGen,
                    genAlgPars=genAlgPars)

par_args = {'vl': -67, 'gl': 0.1, 'C': 1, 'I': 1.5}
IF1 = makeIFneuron('IF1', par_args, IFmodelC)
IF2 = makeIFneuron('IF2', par_args, IFmodelC)
connectIFneurons(IF1, IF2, 1.2, -80, IFmodelC)
connectIFneurons(IF2, IF1, 1.2, -80, IFmodelC)
IFmspec.add([IF1, IF2])
IFmodel=IFmodelC.getModel()

# -------------------------------------------------------------------------

print("Computing trajectory...\n")
icdict = {'IF1.V': -76, 'IF2.V': -66, 'IF1.excited': 0, 'IF2.excited': 0,
          'IF1.syn_IF2_IF1.s': 0, 'IF2.syn_IF1_IF2.s': 0}
start = perf_counter()
IFmodel.compute(trajname='test',
                    tdata=[0, 1000],
                    ics=icdict,
                    verboselevel=0)
print('... finished in %.3f seconds.\n' % (perf_counter()-start))

# -------------------------------------------------------------------------

print("Testing synch orbits")
icdict['IF1.V'] = icdict['IF2.V']
IFmodel.compute(trajname='synch_test', tdata=[0,30], ics=icdict)
assert IFmodel.getTrajEventTimes('synch_test')['IF1_thresh'][-1] - \
       IFmodel.getTrajEventTimes('synch_test')['IF2_thresh'][-1] == 0, \
         "Trajectories should have stayed identical"

plotData = IFmodel.sample('test', dt=0.1)
plt.ylabel('V')
plt.xlabel('t')
vline1 = plt.plot(plotData['t'], plotData['IF1.V'])
vline1 = plt.plot(plotData['t'], plotData['IF2.V'])
plt.show()
