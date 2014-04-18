""" EXAMPLE: Hybrid system continuation -- a simple example
      Continue period of an integrate-and-fire with square spike oscillator
    in two parameters: drive current Iapp and leak conductance gl.

    Change PCargs.MaxNumPoints to e.g. 10 for a more serious use (but it's slow)

    Robert Clewley, January 2011.
"""

from PyDSTool import *
from IF_squarespike_model import makeIFneuron

# ensure I is large enough to make spikes
par_args_linear = {'Iapp': 1.5, 'gl': 0.1, 'vl': -67, 'threshval': -65, 'C': 1}
par_args_spike = {'splen': 0.75}

IFmodel = makeIFneuron('IF_bif', par_args_linear, par_args_spike, evtol=1e-8)
# set IC right before a spike
icdict = {'v': -66, 'excited': 0}
IFmodel.set(ics=icdict, tdata=[0,100])

def get_cycle(DS):
    DS.compute(trajname='cont_traj', force=True)
    thresh_ts = DS.getTrajEventTimes('cont_traj')['threshold']
    return array([thresh_ts[-1]-thresh_ts[-2]], float)


T_target = get_cycle(IFmodel)[0]

# Set up continuation class
PyCont = ContClass(IFmodel)

def cont_func(C, pt, pars):
    DS = C.model
    DS.set(pars={'gl': pt['gl'],
                 'Iapp': pars['Iapp']})
    try:
        F = get_cycle(DS)
    except PyDSTool_ExistError:
        print('Problem computing orbit')
        C._userdata.problem = True
        return array([0], float)
    else:
        return F - T_target


PCargs = args(name='UD1', type='UD-C')
PCargs.description = """User-defined continuation of period"""
PCargs.uservars = ['gl']
PCargs.userpars = PyCont.gensys.query('pars')
PCargs.userfunc = cont_func
PCargs.FuncTol = 1e-6
PCargs.VarTol = 1e-6
PCargs.freepars = ['Iapp']
PCargs.StepSize = 3e-2
PCargs.MaxStepSize = 5e-2
PCargs.MaxNumPoints = 4 # Make larger for a more serious run (but slow)
PCargs.SaveJacobian = True
PCargs.verbosity = 4
PCargs.initpoint = {'gl': PyCont.gensys.query('pars')['gl']}
PyCont.newCurve(PCargs)

print('Computing curve...')
PyCont['UD1'].forward()
PyCont['UD1'].backward()

# Plot
PyCont.display(('Iapp','gl'))
show()
