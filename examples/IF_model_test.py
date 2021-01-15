"""Integrate-and-Fire neuron model test.

The model exhibits a square-pulse spike on reaching a firing
threshold. The model utilizes an 'absolute refractory period' and a
discontinuous state reset from refractoriness using features of the
hybrid system simulator.

Uses events to detect threshold crossing accurately, and separate
"vector fields" are simulated for sub-threshold and spiking behaviour,
using individual Generators. The Generators are brought together using
the Model class as a hybrid dynamical system.

    Robert Clewley, March-August 2005.
"""

from PyDSTool import *
from time import perf_counter

# ---------------------------------------------------------------------------

print('-------- Model test')
all_model_names = ['leak', 'spike']

# 'excited' is an internal variable of the model, and is used to
# ensure that the compute() method can determine which Generator
# to start the calculation with
leak_event_args = {'name': 'threshold',
                   'eventtol': 1e-3,
                   'eventdelay': 1e-5,
                   'starttime': 0,
                   'active': True,
                   'term': True,
                   'precise': True}
leak_thresh_ev = Events.makePythonStateZeroCrossEvent('V', -60, 1,
                                                 leak_event_args)
leak_args = {'pars': {'I': 1.3, 'gl': 0.1, 'vl': -67},
              'xdomain': {'V': [-100,50], 'excited': 0},
              'xtype': {'excited': int},
              'varspecs': {'V':       "I - gl*(V-vl)",
                              'excited': "0"
                              },
              'algparams': {'init_step': 0.02},
              'events': leak_thresh_ev,
              'abseps': 1.e-7,
              'name': 'leak'}

ics = {'V': -80., 'excited': 0.}

DS_leak = embed(Generator.Vode_ODEsystem(leak_args), icdict=ics, tdata=[0, 30])
DS_leak_MI = intModelInterface(DS_leak)

# spike length parameter 'splen' must be contained within 'tdomain' in
# order to get a fully-formed square-pulse `spike`
spike_args = {'tdomain': [0.0, 1.5],
                'varspecs': {'V': "if(t<splen,50,-95)", 'excited': "1."},
                'pars': {'splen': 0.75},
                'xdomain': {'V': [-97, 51], 'excited': 1},
                'xtype': {'excited': int},
                'name': 'spike'}
DS_spike = embed(Generator.ExplicitFnGen(spike_args), icdict=ics, tdata=[0, 30])

DS_spike_MI = intModelInterface(DS_spike)

# test discrete state mapping that is used at changes of vector field
# after terminal events
# must set excited to 0 in order to meet bounds requirements of IF gen
epmapping = EvMapping({"V": "V+15",
                       "excited": "0"}, model=DS_spike)

# build model object from individual DS's
DS_leak_info = makeModelInfoEntry(DS_leak_MI, all_model_names,
                                  [('threshold', 'spike')])
DS_spike_info = makeModelInfoEntry(DS_spike_MI, all_model_names,
                                 [('time', ('leak', epmapping))])
modelInfoDict = makeModelInfo([DS_leak_info, DS_spike_info])

IFmodel = Model.HybridModel({'name': 'IF_fit', 'modelInfo': modelInfoDict})

print("Computing trajectory...\n")
start = perf_counter()
IFmodel.compute(trajname='onespike', tdata=[0,30], ics=ics, verboselevel=2, force=True)
print('... finished in %.3f seconds.\n' % (perf_counter()-start))

print('Preparing plot to show non-identity mapping of epoch state transitions')
plotData = IFmodel.sample('onespike', ['V'], 0.02)
plt.ylabel('V')
plt.xlabel('t')
vline = plt.plot(plotData['t'], plotData['V'])

print("\n\nInformation about Model's components:\n")
info(IFmodel.query('submodels'))

plt.show()
