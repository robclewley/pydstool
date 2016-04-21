"""
    Parameter Estimation tests #3.

    Robert Clewley, March 2005.
"""
from __future__ import print_function

# PyDSTool imports
from PyDSTool import *
from PyDSTool.Toolbox.ParamEst import *
from PyDSTool.Toolbox.neuro_data import *
import HH_model_Cintegrator as HH_model

# Other imports
from numpy.linalg import norm
from time import clock

# ----------------------------------------------------------------

tdata = [0, 15]

par_args_HH_goal = {'gna': 100, 'gk': 80, 'gl': 0.1,
            'vna': 50, 'vk': -100, 'vl': -67,
            'I': 1.3, 'C': 1.0}
ic_args_HH = {'v':-68, 'm': 0, 'h': 1, 'n': 0}

HH_event_args = args(name='HH_zerothresh',
               eventtol=1e-4,
               eventdelay=1e-3,
               starttime=0,
               active=True)
HH_thresh_ev = Events.makeZeroCrossEvent('v', 1, HH_event_args, ['v'],
                                         targetlang='c')

HH_goal = HH_model.makeHHneuron('goalHH', par_args_HH_goal, ic_args_HH,
            evs=HH_thresh_ev,
            extra_terms='-0.04*(sin(9.1*t)*cos(2.6*t)+sin(5.1119*t+2))*(v-60)')

# extra terms simulate low frequency "noise"

HH_goal.set(tdata=tdata,
            algparams={'init_step':0.1})

goaltraj = HH_goal.compute('goalHHtraj')


HH_spike_t = HH_goal.getEventTimes()['HH_zerothresh'][0]
print("HH spike time found at ", HH_spike_t)


## Set up test HH model
par_args_HH_test = {'gna': 100, 'gk': 80, 'gl': 0.12,
            'vna': 50, 'vk': -100, 'vl': -70,
            'I': 1.34, 'C': 1.0}
# Note that I is not the same as that for goal, even though we're not
# optimizing this parameter. Increasing I from original 1.3 to 1.34
# causes slow convergence.

DS_event_args = args(name='threshold',
           eventtol=1e-4,
           eventdelay=1e-3,
           starttime=0,
           active=True,
           term=False,
           precise=True)
thresh_ev = Events.makeZeroCrossEvent('v', 1, DS_event_args, ['v'],
                                      targetlang='c')
HH_test = HH_model.makeHHneuron('testHH', par_args_HH_test, ic_args_HH,
                                thresh_ev)

HH_test.set(tdata=[0,15], algparams={'atol':1e-9,'rtol':1e-8,
                                     'min_step': 1e-5})



## Set up external interface for the reference trajectory based on spike time

tmesh = goaltraj.indepdomain.sample(dt=(tdata[1]-tdata[0])/100.,
                                    avoidendpoints=True)

## DATA SPIKE ===================

# quantitative feature
sp_feat = spike_feature('spike_feat', pars=args(tol=0.6))

spike_condition = condition({sp_feat: True})

# one interface for judging the spike (uses a qual feature to process the ref
# trajectory)
is_spike = get_spike_data('is_spike', pars=args(height_tol=1.,
                                                fit_width_max=1.,
                                                weight=0,
                                                width_tol=10,
                                                noise_tol=0.5,
                                                thresh_pc=0.15,
                                                eventtol=1e-5,
                                                coord='v',
                                                tlo=tdata[0],
                                                thi=tdata[1]))

class ext_spike_iface(extModelInterface):
    def postprocess_test_traj(self, traj):
        # convert traj to individual spike time, value pair
        assert is_spike(traj)
        spike_time = is_spike.results.spike_time
        spike_height = is_spike.results.spike_val
        return numeric_to_traj([[spike_time], [spike_height]], self._trajname,
                               ['sptime','spval'],
                               indepvar=[0])

spike_interface = ext_spike_iface(goaltraj,
            conditions=spike_condition,
            compatibleInterfaces=['int_spike_iface'])

## DATA GEOM ===================

geom_feat = geom_feature('geom_feat', pars=args(tol=10,
                                                tmesh=tmesh,
                                                depvar='v'))

geom_condition = condition({geom_feat: True})

# one interface for judging the shape of the V trajectory
class ext_geom_iface(extModelInterface):
    pass

geom_interface = ext_geom_iface(goaltraj,
                        conditions=geom_condition,
                        compatibleInterfaces=['int_geom_iface'])

# Make model out of HH DS
HH_test_model = embed(HH_test, ic_args_HH)
HH_test_model.compute(trajname='orig')

class int_spike_iface(intModelInterface):
    def postprocess_test_traj(self, traj):
        evpts = traj.getEvents('threshold')
        # catch "broken" output and penalize
        if evpts is None:
            ev_t = [300]
            ev_v = [300]
        elif len(evpts) != 1:
            ev_t = [300]
            ev_v = [300]
        else:
            ev_t = evpts['t']
            ev_v = evpts['v']
        return numeric_to_traj([ev_t, ev_v], self._trajname,
                               ['sptime', 'spval'],
                               indepvar=[0])

class int_geom_iface(intModelInterface):
    def postprocess_test_traj(self, traj):
        # use tmesh of data points only (may not be the same mesh as was used by
        # this model traj, that's why we have to resample
        varray = traj(tmesh)['v']
        return numeric_to_traj([varray], self._trajname, ['v'],
                               indepvar=tmesh)


pest_context = context([ (spike_interface, int_spike_iface),
                         (geom_interface, int_geom_iface) ])

print("Feature evaluation on initial set-up: ", pest_context.evaluate(HH_test_model))
print("geom feat residual: ", norm(geom_feat.metric.results))
pts1=geom_feat.ref_traj(tmesh,coords=['v'])
pts2=HH_test_model('test_iface_traj', tmesh, coords=['v'])
#plot(tmesh, pts1['v'])
#plot(tmesh, pts2['v'])
print("\nResidual norm before feature weighting:")
print(norm(pest_context.residual(HH_test_model)))
#pest_context.set_weights({geom_interface: 0.005})
pest_context.set_weights({geom_interface: 0.005, spike_interface: 0.25})
print("Residual norm after feature weighting:")
print(norm(pest_context.residual(HH_test_model)))

## Parameter estimation
print('\nEstimating pars gl and vl for fit')
print('Goal values are vl =', par_args_HH_goal['vl'], ', gl = ', \
            par_args_HH_goal['gl'], ' ...')

pnames = ['vl', 'gl']
parscales = {'vl': 0.1, 'gl': 0.01}
parseps = {'vl': 3e-2, 'gl': 1e-3}
pest1, opt = make_opt(pnames, residual_fn_context, HH_test_model, pest_context,
                      parscales=parscales, parseps=parseps)

#pest_pars = LMpest(freeParams=['vl', 'gl'],
#                 testModel=HH_test_model,
#                 context=pest_context,
#                 verbose_level=2
#                )

start = clock()
#pestData_par = pest_pars.run(parDict={'ftol':1e-4,
#                                      'xtol':1e-4,
#                                      },
#                             verbose=True)

opt.iterate()
pest_context.set_weights({geom_interface: 0.1, spike_interface: 0.1})
pest2, opt = make_opt(pnames, residual_fn_context, HH_test_model, pest_context,
                      parscales=parscales, parseps=parseps)
opt.iterate()

print('... finished in %.3f seconds\n' % (clock()-start))


log_ix = pest2.find_logs()[0]
sol_pars = pest2.log[log_ix].pars
HH_test_model.set(pars=sol_pars) #pestData_par['pars_sol'])
print("Feature evaluation on solution set-up: ", \
      pest_context.evaluate(HH_test_model))
print("geom feat residual: ", norm(geom_feat.metric.results))

# solution trajectory involving voltage happens to be the first of the
# two trajectories stored in each log (one for each model interface, and
# stored in order of the names of the interfaces).
sol_traj = pest2.log[log_ix].trajectories[0]

## Finish preparing plots
print('\nPreparing plots')
figure()
disp_dt = 0.05
plotData_orig = HH_test_model.sample('orig', ['v'], disp_dt, precise=True)
plotData_goal = goaltraj.sample(['v'], disp_dt, precise=True)
plotData_par = sol_traj.sample(['v'])

plt.ylabel('v')
plt.xlabel('t')
goalline=plot(plotData_goal['t'], plotData_goal['v'], label="v goal")
origline = plot(plotData_orig['t'], plotData_orig['v'], label="v initial")
estline = plot(plotData_par['t'], plotData_par['v'], label='v estimated')

plt.legend(loc='lower left')
show()
