"""
    Parameter Estimation tests #4.

    Variant of #3 but with sodium conductance pars being optimized.
    Requires Dopri integrator because more computational power is needed!

    Robert Clewley, March 2005.
"""
from __future__ import print_function

# PyDSTool imports
from PyDSTool import *
from PyDSTool.Toolbox.ParamEst import LMpest
from PyDSTool.Toolbox.neuro_data import *
import HH_model

from time import clock

# print "This test runs much more efficiently using the Dopri integrator"
gentype = 'dopri'
genlang = 'c'

# ----------------------------------------------------------------

tdata = [0, 20]

par_args_HH_goal = {'gna': 100, 'gk': 80, 'gl': 0.1,
            'vna': 50, 'vk': -100, 'vl': -67,
            'Iapp': 1.3, 'C': 1.0}
ic_args_HH = {'v':-70, 'm': 0, 'h': 1, 'n': 0}
HH_goal = HH_model.makeHHneuron('goalHH', par_args_HH_goal, ic_args_HH,
             extra_terms='-0.03*(sin(9.1*t)*cos(2.6*t)+sin(5.1119*t+2))*(v-60)')
HH_goal.set(tdata=tdata, algparams={'init_step':0.1})
goaltraj = HH_goal.compute('goalHHtraj')

HH_event_args = args(name='HH_zerothresh',
               eventtol=1e-3,
               eventdelay=1e-3,
               starttime=0,
               active=True,
               precise=True)
HH_thresh_ev = Events.makePythonStateZeroCrossEvent('v', 0, 1,
                                   HH_event_args, goaltraj.variables['v'])


result = HH_thresh_ev.searchForEvents(tuple(tdata))
HH_spike_t = result[0][0]
print("True HH spike time based on threshold event is at ", HH_spike_t)
print("but assume the traj is real data so that we have to find the spike")
print("directly from the noisy data")

## Set up external interface for the reference trajectory based on spike time

tmesh = goaltraj.indepdomain.sample(dt=(tdata[1]-tdata[0])/100.,
                                    avoidendpoints=True)

## DATA SPIKE ===================

# quantitative feature
sp_feat = spike_feature('spike_feat', pars=args(tol=0.6))

spike_condition = condition({sp_feat: True})

# one interface for judging the spike (uses a qual feature to process the ref
# trajectory)
is_spike = get_spike_data('is_spike', pars=args(height_tol=2.,
                                                fit_width_max=1.,
                                                weight=0,
                                                width_tol=15,
                                                noise_tol=0.5,
                                                thresh_pc=0.15,
                                                eventtol=1e-4,
                                                coord='v',tlo=tdata[0],
                                                thi=tdata[1]))

assert is_spike(goaltraj)

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

## ----------------------------------------------------------------------


## Set up test HH model
par_args_HH_test = {'gna': 95, 'gk': 82, 'gl': 0.12,
            'vna': 48, 'vk': -95, 'vl': -67.5,
            'Iapp': 1.32, 'C': 1.0}

# Note that these params are not the same as that for goal, even though we're not
# optimizing them

DS_event_args = args(name='threshold',
           eventtol=5e-3,
           eventdelay=1e-3,
           starttime=0,
           active=True,
           term=False,
           precise=True)
thresh_ev = Events.makeZeroCrossEvent('v', 1, DS_event_args, varnames=['v'],
                                      targetlang=genlang)
HH_test = HH_model.makeHHneuron('testHH2', par_args_HH_test, ic_args_HH,
                                thresh_ev, gentype=gentype)

if genlang == 'python':
    # need to force more accuracy b/c less efficient integrator
    init_step = 1e-3
else:
    init_step = 1e-2
HH_test.set(tdata=tdata, algparams={'atol':1e-9,'rtol':1e-8, 'init_step': init_step,
                                               'min_step':1e-5})

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

## Parameter estimation
print('Estimating pars gna and vl for fit to non-identical HH cell')
print('Goal values are gna =', par_args_HH_goal['gna'], ', gl =', \
            par_args_HH_goal['gl'], ' ...')


pest_pars = LMpest(freeParams=['gna', 'gl'],
                 testModel=HH_test_model,
                 context=pest_context,
                 verbose_level=2
                )

# In case finite difference stepsize needs adjusting
pest_pars.fn.eps=1e-5

pest_context.set_weights({spike_interface: {sp_feat: 10},
                          geom_interface: {geom_feat: 0.2}})

t0=clock()
pestData_par_phase1 = pest_pars.run(parDict={'ftol':1e-5,
                                      'xtol':1e-6
                                      },
                             verbose=True)

HH_test_model.set(pars=pestData_par_phase1['pars_sol'])
pest_context.set_weights({spike_interface: {sp_feat: 5},
                          geom_interface: {geom_feat: 0.7}})

pestData_par = pest_pars.run(parDict={'ftol':1e-5,
                                      'xtol':1e-6
                                      },
                             verbose=True)

print('... finished in %.4f seconds\n'%(clock()-t0))


## Finish preparing plots
print('\nPreparing plots')
figure()
disp_dt = 0.05
plotData_orig = HH_test_model.sample('orig', ['v'], disp_dt, precise=True)
origleg = "v initial"
plotData_goal = goaltraj.sample(['v'], disp_dt)
goalleg = "v goal"
plotData_par = HH_test_model.sample('test_iface_traj', ['v'], disp_dt)

plt.ylabel('v')
plt.xlabel('t')
goalline = plt.plot(plotData_goal['t'], plotData_goal['v'])
origline = plt.plot(plotData_orig['t'], plotData_orig['v'])
estline = plt.plot(plotData_par['t'], plotData_par['v'])
estleg = 'v estimated'

plt.legend([origline, goalline, estline],
             [origleg, goalleg, estleg],
             'lower left')
show()
