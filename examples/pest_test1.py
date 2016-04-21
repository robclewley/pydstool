"""
    Parameter Estimation tests #1.

    Robert Clewley, February 2005.
"""
from __future__ import print_function

# PyDSTool imports
from PyDSTool import *
from PyDSTool.Toolbox.ParamEst import LMpest, L2_feature

# Other imports
import time
from copy import copy

# ----------------------------------------------------------------


## Initialize some values
trange = [0,2.5]
xic = {'w':3.0}
refpars = {'k':2, 'a':25}

## Prepare sine-wave input to represent 'noise' in the goal trajectory
fnArgs = args(varspecs={'s': "sin(t*speed)"},
          xdomain={'s': [-1, 1]},
          pars={'speed': 18},
          name='sine')
sin_input=Generator.ExplicitFnGen(fnArgs)
sin_input_traj = sin_input.compute('noise')


## Prepare goal ODE trajectory
print('Preparing goal trajectory from perturbed system (with k = ', \
    refpars['k'], ') ...')

xfn_refstr = "50-k*w+a*(2-t)*sin_input"

refDSargs = args(algparams={'init_step':0.02, 'strictdt':True},
              tdata=trange,
              pars=refpars,
              varspecs={'w':xfn_refstr},
              xdomain={'w':[0, 1000]},
              inputs={'sin_input' : sin_input_traj.variables['s']},
              checklevel=1,
              ics=xic,
              name='ODEtest'
              )
refODE = Generator.Vode_ODEsystem(refDSargs)
reftraj = refODE.compute('goal')
print('... finished.\n')


## Get plot data for goal orbit
tplotData = linspace(trange[0], trange[1], 50)
wplotData = reftraj(tplotData,['w'])
refleg = 'w'+' for k = '+str(refpars['k'])


## Parameter estimation
print('Estimating parameter k for fit (assuming initial condition for w)')
print('Goal value is k = ', refpars['k'], " ...")

xfn_str = "50-k*w"
testDSargs = args(algparams={'init_step':0.02, 'strictopt':True},
              varspecs={'w':xfn_str},
              xdomain={'w':[0, 1000]},
              tdata=trange,
              pars={'k':0.1},
              checklevel=2,
              ics=xic,
              name='test_model_par'
              )

ftol=3e-3
# use L2-norm of data (sum of squares)
L2_similarity_w = L2_feature('L2_similar', pars=args(t_samples=tplotData,
                                                   trange=trange,
                                                   coord='w',
                                                   tol=ftol,
                                                   debug=True))

pest_condition_w = condition({L2_similarity_w : True})

class ext_iface(extModelInterface):
    # holds the data (external from the model)
    pass

class int_iface(intModelInterface):
    # holds the test model
    pass

pest_data_interface_w = ext_iface(reftraj,
                   pest_condition_w)

c = context([ (pest_data_interface_w, int_iface) ])

testModel_par = embed(Generator.Vode_ODEsystem(testDSargs))
pest_pars = LMpest(freeParams=['k'],
                 testModel=testModel_par,
                 context=c
                )

start_time = time.clock()
pestData_par = pest_pars.run(parDict={'ftol': ftol,
                                      'xtol':1e-3},
                             verbose=True)
print('... finished in %.4f seconds\n' % (time.clock()-start_time))

bestFitModel_par = pestData_par['sys_sol']

## Initial condition estimation
print("Estimating initial condition for w (assuming k is correct)")
print("Goal value is w(0) = ", xic['w'], " ...")

modelArgs_ic = copy(testDSargs)
modelArgs_ic['ics'] = {'w': 0.0}
modelArgs_ic['pars'] = {'k': pestData_par['pars_sol']['k']}
modelArgs_ic['name'] = 'test_model_ic'

testModel_ic = embed(Generator.Vode_ODEsystem(modelArgs_ic))

pest_ic = LMpest(freeParams=['w'],
                 testModel=testModel_ic,
                 context=c
                 )
pestData_ic = pest_ic.run(parDict={'ftol': ftol,
                                   'xtol':1e-3},
                          verbose=True)
bestFitModel_ic = pestData_ic['sys_sol']
print('... finished')


## Finish preparing plots
print('\nPreparing plots')
west_plotData_par = bestFitModel_par.sample('test_iface_traj', dt=0.02,
                                                           tlo=min(trange),
                                                           thi=max(trange),
                                                precise=True)
west_plotData_ic = bestFitModel_ic.sample('test_iface_traj', dt=0.02,
                                                         tlo=min(trange),
                                                         thi=max(trange),
                                                precise=True)

plt.ylabel('w')
plt.xlabel('t')
refline=plot(tplotData, wplotData['w'], label=refleg)
estleg_par = 'w est. for k = %.4f'%pestData_par['pars_sol']['k']
estline_par = plot(west_plotData_par['t'], west_plotData_par['w'],
                   label=estleg_par)
estleg_ic = 'w est. for w(0) = %.4f'%pestData_ic['pars_sol']['w']
estline_ic = plot(west_plotData_ic['t'], west_plotData_ic['w'],
                  label=estleg_ic)
plt.legend(loc='upper left')
show()
