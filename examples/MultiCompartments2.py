"""Multi-compartment demonstration of neural computation toolbox, part 2
     (small network).
   Uses Dopri integrator.

Robert Clewley, January 2011
"""

from PyDSTool import *
from PyDSTool.Toolbox.neuralcomp import *
from copy import copy

#FIXME: leads to *** Error in `python': malloc(): memory corruption: ***
#targetGen = 'Dopri_ODEsystem'
targetGen = 'Vode_ODEsystem'

# -------------------------------------------------------------------------

if targetGen == 'Vode_ODEsystem':
    targetlang='python'
else:
    targetlang='c'

# -------------------------------------------------------------------------


# largest time ever needed in model
t_max = 200000

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

channel_Ib1 = makeBiasChannel('Ib', 2.2)
channel_Lk1 = makeChannel_rates('Lk', vrev=-67, g=0.1)


soma = makeSoma('soma', channelList=[channel_Lk1, channel_Ib1,
                   channel_Na1, channel_K1], C=1.5)

channel_Lk2 = makeChannel_rates('Lk', vrev=-68, g=0.05)
dend = makeDendrite('dend', channelList=[channel_Lk2], C=0.2)
connectWithSynapse('gapjunc', 'gap', soma, dend, g=0.2)

cell1 = neuron('cell1')
cell1.add([soma,dend])

cell2 = copy(cell1)
cell2.rename('cell2')

syn12 = connectWithSynapse('s12', 'inh', cell1, cell2, 'dend', g=1.9)
syn21 = connectWithSynapse('s21', 'inh', cell2, cell1, 'dend', g=1.5)

net = makeNeuronNetwork('HHnet', [cell1, cell2])


####### User demonstration stuff
print("*** Example of hierarchical referencing to components, etc.")
print("Na vrev Par object is given by >>> net['cell2.soma.Na.vrev']  = \n  ", \
      repr(net['cell2.soma.Na.vrev']))
print("which is equivalent to >>> net.components['cell2'].components['soma'].components['Na'].pars['vrev']")
print("Na vrev value is given by >>> net['cell2.soma1.Na.vrev']()  = \n  ", \
      net['cell2.soma.Na.vrev']())
print("Could delete this Par in place using >>> del net['cell2.soma.Na.vrev']  (not done here!)")



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
net.flattenSpec(ignoreInputs=True)  # needed for event specs only
stat_ev1 = Events.makeZeroCrossEvent(net.flatSpec['vars']['cell1_soma_V'],
                        0, stat_ev_args, targetlang=targetlang,
                        flatspec=net.flatSpec)

alg_args = {'init_step':0.15,
            'max_pts': 50000}
ic_args_net = {'cell1.soma.V':-68.0, 'cell1.soma.Na.m': 0.05,
               'cell1.soma.Na.h': 1, 'cell1.soma.K.n': 0,
               'cell1.dend.V':-65.0,
               'cell2.soma.V':-80.0, 'cell2.soma.Na.m': 0,
               'cell2.soma.Na.h': 1, 'cell2.soma.K.n': 0,
               'cell2.dend.V':-79.0,
               'cell1.s12.s_cell1_soma_cell2_dend': 0,
               'cell2.s21.s_cell2_soma_cell1_dend': 0}

modelC_net = ModelConstructor('HH_model',
                          generatorspecs={net.name: {'modelspec': net,
                                                'target': targetGen,
                                                'algparams': alg_args}},
                          checklevel=0,
                          indepvar=('t',[0,t_max]),
                          # make cell2's bias current different
                          parvalues={'cell2.soma.Ib.Ibias': 2.1},
                          eventtol=1e-5)
modelC_net.addEvents('HHnet', stat_ev1)
HHmodel_net = modelC_net.getModel()

verboselevel = 0
HHmodel_net.compute(trajname='test',
                     tdata=[0, 400],
                     ics=ic_args_net,
                     verboselevel=verboselevel)

v_dat = HHmodel_net.sample('test')
plt.figure()
vs1line = plt.plot(v_dat['t'], v_dat['cell1.soma.V'], 'g')
vd1line = plt.plot(v_dat['t'], v_dat['cell1.dend.V'], 'g--')
vs2line = plt.plot(v_dat['t'], v_dat['cell2.soma.V'], 'r')
vd2line = plt.plot(v_dat['t'], v_dat['cell2.dend.V'], 'r--')
plt.show()
