"""
    Parameter Estimation tests #2.

    Robert Clewley, March 2005.
"""
from __future__ import print_function

# PyDSTool imports
from PyDSTool import *
from PyDSTool.Toolbox.ParamEst import BoundMin, L2_feature_1D
from PyDSTool.common import metric_float_1D
import HH_model, IF_squarespike_model

# ----------------------------------------------------------------

trange = [0, 15]

par_args_HH = {'gna': 100, 'gk': 80, 'gl': 0.1,
            'vna': 50, 'vk': -100, 'vl': -67,
            'Iapp': 1.35, 'C': 1.0}
# deliberately set Iapp not quite 1.3, as used for IF neuron
ic_args_HH = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}
HH = HH_model.makeHHneuron('goalHH', par_args_HH, ic_args_HH)

HH.set(tdata=trange)

HH_traj = HH.compute('test')
HH_sampleData = {}
HH_sampleData['t'] = []
HH_sampleData['v'] = []
sample_dt = 0.06
count = 0
countlim = 5
print("Generating non-uniform samples from HH orbit...")
tsamples = arange(0, 14, sample_dt)
vsamples = HH_traj(tsamples, ['v']).toarray()
for i in range(len(tsamples)):
    t = tsamples[i]
    v = vsamples[i]
    if v > -57:
        HH_sampleData['t'].append(t)
        HH_sampleData['v'].append(v)
    else:
        # reduce sample rate for non-spiking region
        count += 1
        if count == countlim:
            HH_sampleData['t'].append(t)
            HH_sampleData['v'].append(v)
            count = 0
print("... done")


tableArgs = {'tdata': HH_sampleData['t'],
             'ics': {'v': HH_sampleData['v']},
             'name': 'HH_data'}
HH_DataTable = Generator.LookupTable(tableArgs)
tmesh_par = HH_sampleData['t']

par_args_linear = {'Iapp': 1.3, 'gl': 0.1, 'vl': -67, 'threshval': -60, 'C': 1.0}
par_args_spike = {'splen': 1.0}

## Parameter estimation for firing threshold
icdict = {'v': -70.0, 'excited': 0}
IFmodel_thr = IF_squarespike_model.makeIFneuron('IF_thr_fit', par_args_linear,
                                par_args_spike, icdict=icdict)

# un-fitted IF trajectory
IFmodel_thr.compute(trajname='orig', tdata=[0, tmesh_par[-1]],
                        ics={'v':-70, 'excited':0}, verboselevel=2)
orig_pdata = IFmodel_thr.sample('orig', ['v'], 0.1)


HH_event_args = {'name': 'HH_zerothresh',
               'eventtol': 1e-2,
               'eventdelay': 1e-3,
               'starttime': 0,
               'active': True}
HH_thresh_ev = Events.makePythonStateZeroCrossEvent('v', 0, 1, HH_event_args,
                                               HH_traj.variables['v'])


result = HH_thresh_ev.searchForEvents((0, 15))
HH_spike_t = result[0][0]
print("HH spike time found at ", HH_spike_t)

class IF_spike_feat(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_float_1D()
        self.metric_len = 1

    def evaluate(self, target):
        tparts = target.test_traj.timePartitions
        if len(tparts) == 1:
            spike_t = 1000
        else:
            spike_t = tparts[1][1]
        return self.metric(self.ref_traj.sample()[0], spike_t)

# set tdata here so that it persists beyond any one call to compute
IFmodel_thr.set(tdata=[0, 15])

feat = IF_spike_feat('t_similar', pars=args(debug=True))

pest_condition = condition({feat: True})

class ext_iface(extModelInterface):
    # holds the data (external from the model)
    pass

class int_iface(intModelInterface):
    # holds the test model
    pass

pest_data_interface = ext_iface(numeric_to_traj([[HH_spike_t]], 'ref', ['st'],
                                                indepvarname='ix',
                                                indepvar=[0]),
                                 pest_condition)

pest_context = context([ (pest_data_interface, int_iface) ])

pest_thr = BoundMin(freeParams=['threshval'],
                 testModel=IFmodel_thr,
                 context=pest_context
                )

pestData_thr = pest_thr.run(parConstraints=[-65,-57],
                            xtol=5e-3,
                            verbose=True)


## Parameter estimation for spike length
print("\nParam est. for spike length ...")
if not pestData_thr['success']:
    raise RuntimeError("Failure: will not continue")

thresh_fit = pestData_thr['pars_sol']['threshval']

par_args_linear = {'Iapp': 1.3, 'gl': 0.1, 'vl': -67, 'threshval': thresh_fit,
                   'C': 1.0}
par_args_spike = {'splen': 1.0}

HH_datatable_traj = HH_DataTable.compute('goaltraj')

# find closest (t, v) point for i.c. near spike
ic_not_found = True
tmesh_ic = []
for t in HH_sampleData['t']:
    if t >= 7.0 and t < 11:
        tmesh_ic.append(t)
        if ic_not_found:
            t_ic = t
            v_ic = HH_datatable_traj(t, ['v'])
            ic_not_found = False
    if t >= 11:
        break


IFmodel_splen = IF_squarespike_model.makeIFneuron('IF_splen_fit', par_args_linear,
                                par_args_spike, icdict={'v':-70, 'excited':0})

## test IF trajectory
IFmodel_splen.compute(trajname='test', tdata=[0, t_ic])
IF_ic = IFmodel_splen('test', t_ic, ['v'])
IFmodel_splen.set(tdata=[t_ic, 12])

print("\n----------------------")
IFmodel_splen.set(ics={'v': IF_ic})

splen_feat = L2_feature_1D('splen', pars=args(t_samples=tmesh_ic,
                                           coord='v',
                                           tol=1e-3))

splen_condition = condition({splen_feat: True})
splen_data_interface = ext_iface(HH_datatable_traj,
                                 splen_condition)

splen_context = context([ (splen_data_interface, int_iface) ])


pest_splen = BoundMin(freeParams=['splen'],
                 testModel=IFmodel_splen,
                 context=splen_context
                )

pestData_splen = pest_splen.run(xtol=0.01, parConstraints=[0.2,1.0],
                                verbose=True)

IFmodel_splen.set(pars={'splen': pestData_splen['pars_sol']['splen'],
                        'threshval': thresh_fit})

IFmodel_splen.compute(trajname='disp',
                      tdata=[0,15],
                      ics={'v':-70, 'excited':0})

## Plot data
print("Acquiring plot data")
origline=plot(orig_pdata['t'], orig_pdata['v'], label="Un-fitted IF orbit")
IF_sampleData = []
for t in HH_sampleData['t']:
    IF_sampleData.append(IFmodel_splen('disp', t, ['v']))
plt.ylabel('w')
plt.xlabel('t')
goalline=plot(HH_sampleData['t'], HH_sampleData['v'], 'bo',
              label='HH reference')
estline_splen = plot(HH_sampleData['t'], IF_sampleData, 'k-',
                     linewidth=2, label='IF spike thresh \& width fitted')
plt.legend(loc='lower left')
show()
