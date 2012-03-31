"""Multi-compartment demonstration of neural computation toolbox, part 1
    (single cell).
   Uses Vode integrator.

 Robert Clewley, January 2011
"""

from PyDSTool import *
from PyDSTool.Toolbox.neuralcomp import *
from copy import copy

targetGen = 'Vode_ODEsystem'

# -------------------------------------------------------------------------

if targetGen == 'Vode_ODEsystem':
    targetlang='python'
else:
    targetlang='c'

# -------------------------------------------------------------------------


# largest time ever needed in model
t_max = 200

v = Var(voltage)
ma = 0.32*(v+54)/(1-Exp(-(v+54.)/4))
mb = 0.28*(v+27)/(Exp((v+27.)/5)-1)
ha = .128*Exp(-(50.+v)/18)
hb = 4/(1+Exp(-(v+27.)/5))
channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                'h', False, ha, hb, 1, vrev=50, g=100)

na = .032*(v+52)/(1-Exp(-(v+52.)/5))
nb = .5*Exp(-(57.+v)/40)
channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-100, g=99)

channel_Ib1 = makeBiasChannel('Ib', 2.1)
channel_Lk1 = makeChannel_rates('Lk', vrev=-67, g=0.1)


soma1 = makeSoma('soma1', channelList=[channel_Lk1, channel_Ib1,
                   channel_Na1, channel_K1], C=1.5)

channel_Lk2 = makeChannel_rates('Lk', vrev=-68, g=0.05)
dend2 = makeDendrite('dend2', channelList=[channel_Lk2], C=0.2)
connectWithSynapse('gapjunc', 'gap', soma1, dend2, g=0.1)

cell = neuron('HHcell')
cell.add([soma1,dend2])


####### Instantiate the model

# build an event that picks out when RHS of cell1's Voltage eqn is 0
# i.e. when dV/dt=0

stat_ev_args = {'name': 'cell1_stat',
               'eventtol': 1e-3,
               'eventdelay': 1e-3,
               'starttime': 0,
               'term': False
                }
# stationary event => dv/dt = 0
cell.flattenSpec(ignoreInputs=True)
stat_ev = Events.makeZeroCrossEvent(cell.flatSpec['vars']['soma1_V'],
                        0, stat_ev_args, targetlang=targetlang,
                        flatspec=cell.flatSpec)

alg_args = {'init_step':0.15}
ic_args_net = {'soma1.V':-68.0, 'soma1.Na.m': 0.2,
               'soma1.Na.h': 1, 'soma1.K.n': 0,
               'dend2.V':-79.0} #,
#               's12.s_cell1_cell2': 0, 's21.s_cell2_cell1': 0}
modelC_net = ModelConstructor('HH_model',
                          generatorspecs={cell.name: {'modelspec': cell,
                                                'target': targetGen,
                                                'algparams': alg_args}},
                          indepvar=('t',[0,t_max]),
#                          parvalues={'cell.dend2.gapjunc.g': 0.3},
                          eventtol=1e-5)
modelC_net.addEvents('HHcell', stat_ev)
HHmodel_net = modelC_net.getModel()

verboselevel = 2
HHmodel_net.compute(trajname='test',
                     tdata=[0, 60],
                     ics=ic_args_net,
                     verboselevel=verboselevel)

v_dat = HHmodel_net.sample('test')
plt.figure()
v1line = plt.plot(v_dat['t'], v_dat['soma1.V'])
v2line = plt.plot(v_dat['t'], v_dat['dend2.V'])
plt.show()
