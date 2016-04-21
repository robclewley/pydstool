"""A parameter estimation problem inspired by Joe Tien's research.

   Requires files 'ttraj_prebotfast.dat', 'xtraj_prebotfast.dat'.
   ttraj_prebotfast (time points) and xtraj_prebotfast (state variables)
   contain a reference (target) trajectory, with corresponding parameter
   values h=0.3, gl=40, vl=-38., gnap=2.8. We optimize over two parameters,
   gl and vl, comparing voltage traces only. The parameter values for the
   initial guess are h=0.3, gl=38, vl=-35, gnap=2.8.

   Although this setup is overkill for such a simple problem, it is a
   tutorial in what the minimal setup is using the ModelInterface classes
   with the ParamEst module.

   Robert Clewley, March 2005, updated 2008 for ModelInterface classes.
"""
from __future__ import print_function

from PyDSTool import *
from PyDSTool.Toolbox.ParamEst import LMpest, L2_feature

import random
from copy import copy
from time import clock


# ------------------------------------------------------------


def makeTestDS(name, par_args, ic_args, tdomain, evs=None):

    # Variables: v, n
    # Parameters: h, gl, vl, gnap

    fnspecs = {'m': (['v'], '1.0/(1.0+exp((v-th_m)/s_m))'),
      'an': (['v'], '1.0/(2.0*tau_n)*exp(-(v-th_n)/(2.0*s_n))'),
      'bn': (['v'], '1.0/(2.0*tau_n)*exp((v-th_n)/(2.0*s_n))'),
      'mnap': (['v'], '1.0/(1.0+exp((v-th_mnap)/s_mnap))')
      }

    vstr = '-1.0/C*(gna*m(v)*m(v)*m(v)*(1-n)*(v-vna)+gk*n*n*n*n*(v-vk)+' +\
                       'gnap*mnap(v)*h*(v-vna)+gl*(v-vl))'
    nstr = 'an(v)*(1.0-n)-bn(v)*n'

    DSargs = args()
    DSargs.tdomain = tdomain
    DSargs.pars = par_args
    DSargs.varspecs = {'v': vstr, 'n': nstr}
    DSargs.fnspecs = fnspecs
    DSargs.xdomain = {'v': [-130, 70], 'n': [0,1]}
    DSargs.algparams = {'init_step':0.02}
    DSargs.ics = ic_args
    DSargs.checklevel = 0
    DSargs.name = name
    return Generator.Vode_ODEsystem(DSargs)


# ------------------------------------------------------------

pars = {'C': 21.0, 'gna': 28.0, 'gk': 11.2, 'vna': 50.0, 'th_m': -34.0,
      's_m': -5.0, 'vk': -85.0, 'th_n': -29.0, 's_n': -4.0, 'tau_n': 10.0,
      'th_mnap': -40.0, 's_mnap': -6.0}

icdict = {'v': -24.228, 'n': 0.43084}

# ---- make reference system
# in data system version, these "known" pars are not used except to
# show the goal values to the user. they are not part of the param. est. calc.
est_pars_ref = {'h':0.3, 'gl': 40, 'vl': -38., 'gnap': 2.8}
pars_ref = copy(pars)
pars_ref.update(est_pars_ref)


allValDict = importPointset('xtraj_prebotfast.dat',
                            t='ttraj_prebotfast.dat')
xDataDict = dict(list(zip(['v', 'n'], allValDict['vararray'])))
tmesh = allValDict['t'].tolist()
tdomain = [tmesh[0], tmesh[-1]]
refDS = Generator.LookupTable({'tdata': tmesh,
                                'ics': xDataDict,
                                'name': 'ref'
                                })
reftraj = refDS.compute('ref')

# ---- make test system
est_pars_test = {'h':0.3, 'gl': 38, 'vl': -35, 'gnap': 2.8}
pars_test = copy(pars)
pars_test.update(est_pars_test)
testDS = makeTestDS('ref', pars_test, icdict, tdomain)
testModel = embed(testDS, icdict)

# ---- make features for measuring similarity
# feature residual tolerance
ftol=3e-3
# use L2-norm of data (sum of squares)
L2_similarity = L2_feature('L2_similar', pars=args(t_samples=tmesh,
                                                   trange=tdomain,
                                                   coord='v',
                                                   tol=ftol))
# condition for feature is that it is present in data (True)
# (could also specify that it is not present using False)
pest_condition = condition({L2_similarity : True})

# trivial sub-classes of external and internal model interfaces
# that will interact with each other
class ext_iface(extModelInterface):
    # holds the data (external from the model)
    pass

class int_iface(intModelInterface):
    # holds the test model
    pass

# embed the reference trajectory and measurement condition/features
# into an instance of the external interface
pest_data_interface = ext_iface(reftraj,
                   pest_condition)

# generate a context for the model between the external interface
# instance and the internal interface class (not an instance yet)
c = context([ (pest_data_interface, int_iface) ])

# specify which parameters we will fit
est_parnames = ['gl', 'vl']

# parameter estimation
print('Starting Least Squares parameter estimation')
print('Goal pars are gl = ', est_pars_ref['gl'], ' vl = ', est_pars_ref['vl'])
pest_pars = LMpest(freeParams=est_parnames,
                 testModel=testModel,
                 context=c
                )

start = clock()

pestData_par = pest_pars.run(parDict={'ftol':ftol,
                                      'xtol':1e-3},
                             verbose=True
                             )
print('  ... finished in %.3f seconds.\n' % (clock()-start))

# Prepare plots
print('\nPreparing plots')
disp_dt = 0.05
##plotData_goal = reftraj.sample(['v'], disp_dt)
plotData_par = testModel.sample('test_iface_traj', ['v'], disp_dt, precise=True)

plt.ylabel('v')
plt.xlabel('t')
##goalline=plot(plotData_par['t'], plotData_goal['v'])
goal_v = reftraj(tmesh, 'v')
goalline = plot(tmesh, goal_v, 'ok', label='v original')
estline = plot(plotData_par['t'], plotData_par['v'], label='v estimated')

plt.legend(loc='lower left')
show()
