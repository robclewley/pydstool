"""
    Version of 2D spring-loaded inverted pendulum (SLIP)
    using the VODE/Dopri/Radau integrators.

    Robert Clewley, July 2005.

(Adapted from Justin Seipel's Matlab code)
"""

from PyDSTool import *
from copy import copy

# --------------------------------------------------------------------------

dt_global=1e-4
abseps_global=1e-9

allgen_names = ['stance', 'flight']

def getStanceArgs(grav_in_stance, algparams, liftoff_ev, pars, abseps):
    if grav_in_stance:
        grav_str = "-g"
    else:
        grav_str = ""
    return {'pars': pars,
            'fnspecs': {'zeta': (['y', 'z'], 'sqrt(y*y+z*z)'),
##                          initial conditions cannot be accessed in event
##                          'yrel': (['v'], 'v-initcond(y)+cos(beta)')},
                          'yrel': (['y','yic'], 'y-yic-cos(beta)')},
            'varspecs': {'y':    "ydot",
                            'ydot': "k*(1/zeta(yrel(y,initcond(y)),z)-1)*yrel(y,initcond(y))",
                            'z':    "zdot",
                            'zdot': "k*(1/zeta(yrel(y,initcond(y)),z)-1)*z"+grav_str,
                            'zetaval': "zeta(yrel(y,initcond(y)),z)",
                            'incontact': "0"
                            },
            'auxvars': ['zetaval'], #, 'incontact'],
            'xdomain': {'y': [0, Inf], 'z': [0,Inf], 'zetaval': [0,1],
                        'incontact': 1},
            'pdomain': {'beta': [0.0, pi/2]},
            'ics': {'zetaval': 1., 'incontact': 1},
            'xtype': {'incontact': int},
            'algparams': algparams,
            'events': liftoff_ev,
            'abseps': abseps,
            'name': 'stance'}


def getFlightArgs(algparams, touchdown_ev, pars, abseps):
    return {'pars': pars,
           #'fnspecs': {'zeta': (['y', 'z'], 'sqrt(y*y+z*z)')},
           'varspecs': {'y': "initcond(ydot)",
                           'ydot': "0",
                           'z':    "zdot",
                           'zdot': "-g",
                           'zetaval': "1",
                           'incontact': "0"},
           'auxvars': ['zetaval'], #, 'incontact'],
           'xdomain': {'y': [0, Inf], 'z': [0,Inf], 'zetaval': 1,
                       'incontact': 0},
           'pdomain': {'beta': [0.0, pi/2]},
           'ics': {'zetaval': 1., 'incontact': 0},
           'xtype': {'incontact': int},
           'algparams': algparams,
           'events': touchdown_ev,
           'abseps': abseps,
           'name': 'flight'}

def makeInterface(DS, incontact):
    # intial conditions are dummies
    # end time 100 is arbitrary but much longer than possibly needed
    y_dummy = 0.3
    z_dummy = sqrt(1 - y_dummy*y_dummy)
    return intModelInterface(embed(DS,
                                icdict={'zetaval': 1,
                                        'incontact':int(incontact),
                                        'y': y_dummy, 'z': z_dummy,
                                        'ydot': 0.6, 'zdot': 0.3},
                                tdata=[0,100]))

def makeSLIP2D_Vode(pars, dt=dt_global, abseps=abseps_global, grav_in_stance=True,
                    stop_at_TD=False, stop_at_LO=False):
    stance_args, flight_args = makeDS_parts(pars, dt, abseps, grav_in_stance)

    stanceDS = Generator.Vode_ODEsystem(stance_args)
    flightDS = Generator.Vode_ODEsystem(flight_args)
    stanceMI = makeInterface(stanceDS, True)
    flightMI = makeInterface(flightDS, False)

    return makeSLIPModel(stanceMI, flightMI, stop_at_TD, stop_at_LO)


def makeSLIP2D_Dopri(pars, dt=dt_global, abseps=abseps_global, grav_in_stance=True,
                     stop_at_TD=False, stop_at_LO=False):
    stance_args, flight_args = makeDS_parts(pars, dt, abseps, grav_in_stance, 'c')
    stanceDS = Generator.Dopri_ODEsystem(stance_args)
    flightDS = Generator.Dopri_ODEsystem(flight_args)
    stanceMI = makeInterface(stanceDS, True)
    flightMI = makeInterface(flightDS, False)

    return makeSLIPModel(stanceMI, flightMI, stop_at_TD, stop_at_LO)


def makeSLIP2D_Radau(pars, dt=dt_global, abseps=abseps_global, grav_in_stance=True,
                     stop_at_TD=False, stop_at_LO=False):
    stance_args, flight_args = makeDS_parts(pars, dt, abseps, grav_in_stance, 'c')
    stanceDS = Generator.Radau_ODEsystem(stance_args)
    flightDS = Generator.Radau_ODEsystem(flight_args)
    stanceMI = makeInterface(stanceDS, True)
    flightMI = makeInterface(flightDS, False)

    return makeSLIPModel(stanceMI, flightMI, stop_at_TD, stop_at_LO)


def makeSLIPModel(stanceMI, flightMI, stop_at_TD, stop_at_LO):

    if stop_at_TD:
        flightMI_info = makeModelInfoEntry(flightMI, allgen_names,
                                    [('touchdown', 'terminate')])
    else:
        flightMI_info = makeModelInfoEntry(flightMI, allgen_names,
                    [('touchdown', ('stance', EvMapping({"incontact": "1"},
                                                        model=flightMI.model)))])
    if stop_at_LO:
        stanceMI_info = makeModelInfoEntry(stanceMI, allgen_names,
                                    [('liftoff', 'terminate')])
    else:
        stanceMI_info = makeModelInfoEntry(stanceMI, allgen_names,
                    [('liftoff', ('flight', EvMapping({"incontact": "0"},
                                                      model=stanceMI.model)))])
    modelInfoDict = makeModelInfo([stanceMI_info, flightMI_info])

    SLIP = Model.HybridModel({'name': 'SLIP', 'modelInfo': modelInfoDict})
    # promote aux vars from Generators to "vars" in the hybrid model
    SLIP.forceIntVars(['zetaval']) #, 'incontact'])
    return SLIP


def makeDS_parts(pars, dt, abseps, grav_in_stance=True,
                 targetlang='python'):
    assert dt < 1, "dt should be less than 1.0"
    assert abseps < 0.2, "abseps should be less than 0.2"
    if targetlang == 'python':
        max_step = dt
    else:
        max_step = dt*10
    algparams = {'init_step': dt, 'max_step': max_step, 'max_pts': 100000}
    liftoff_args = {'eventtol': abseps/10,
               'eventdelay': abseps*10,
               'eventinterval': abseps*10,
               'active': True,
               'term': True,
               'precise': True,
               'name': 'liftoff'}
    liftoff_ev = Events.makeZeroCrossEvent('zeta(yrel(y,initcond(y)),z)-1', 1,
                          liftoff_args, ['y','z'], ['beta'],
                          fnspecs={'zeta': (['y', 'z'], 'sqrt(y*y+z*z)'),
                                     'yrel': (['y','yic'], 'y-yic-cos(beta)')},
                          targetlang=targetlang)

    stance_args = getStanceArgs(grav_in_stance, algparams, liftoff_ev, pars,
                                abseps)

    touchdown_args = {'eventtol': abseps/10,
               'eventdelay': abseps*10,
               'eventinterval': abseps*10,
               'active': True,
               'term': True,
               'precise': True,
               'name': 'touchdown'}
    touchdown_ev = Events.makeZeroCrossEvent('z-sin(beta)', -1, touchdown_args,
                                             ['z'], ['beta'],
                                             targetlang=targetlang)

    flight_args = getFlightArgs(algparams, touchdown_ev, pars, abseps)
    return (stance_args, flight_args)

