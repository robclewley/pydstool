"""
PySCes interface code for systems biology modeling and SBML model markup.

This toolbox assumes you have PySCes installed.

R. Clewley, 2012
"""
from __future__ import division, absolute_import

from PyDSTool import Generator
from PyDSTool.common import args, remain
from PyDSTool.common import _seq_types, _num_types

import numpy as np
from scipy import linspace, isfinite, sign, alltrue, sometrue

import copy, os, sys


# ----------------------------------------------------------------------------

_functions = ['get_pysces_model', ]

_classes = []

_features = []

__all__ = _functions + _classes + _features

# ----------------------------------------------------------------------------

try:
    import pysces
except ImportError:
    raise ImportError("PySCes is needed for this toolbox to work")

def make_varspecs(m, fnspecs):
    # derived from PySCes.PyscesModel module, class PysMod.showODE method
    maxmetlen = 0
    for x in m.__species__:
        if len(x) > maxmetlen:
            maxmetlen = len(x)

    maxreaclen = 0
    for x in m.__reactions__:
        if len(x) > maxreaclen:
            maxreaclen = len(x)
    odes = {}

    for x in range(m.__nmatrix__.shape[0]):
        odestr = ''
        beginline = 0
        for y in range(m.__nmatrix__.shape[1]):
            reaction = m.__reactions__[y]
            reaction_args = '(' + ','.join(fnspecs[reaction][0]) + ')'
            if abs(m.__nmatrix__[x,y]) > 0.0:
                if m.__nmatrix__[x,y] > 0.0:
                    if beginline == 0:
                        odestr +=  repr(abs(m.__nmatrix__[x,y])) + '*' +  reaction + reaction_args
                        beginline = 1
                    else:
                        odestr +=  ' + ' + repr(abs(m.__nmatrix__[x,y])) + '*' +  reaction + reaction_args
                else:
                    if beginline == 0:
                        odestr +=  ' -' + repr(abs(m.__nmatrix__[x,y])) + '*' +  reaction + reaction_args
                    else:
                        odestr +=  ' - ' + repr(abs(m.__nmatrix__[x,y])) + '*' +  reaction + reaction_args
                    beginline = 1
        odes[m.__species__[x]] = odestr

    if m.__HAS_RATE_RULES__:
        for rule in m.__rate_rules__:
            odes[rule] = m.__rules__[rule]['formula'] #.replace('()','')

    return odes

def get_pysces_model(filename, target='Vode'):
    path, fname = os.path.split(filename)
    m = pysces.model(fname, dir=path)

    max_t = np.Inf

    parlist = m.__fixed_species__ + m.__parameters__
    pardict = dict([(pname, p['initial']) for pname, p in m.__pDict__.items()])
    varlist = m.__species__ # list ['s0', 's1', 's2']

    icdict = dict([(vname, v['initial']) for vname, v in m.__sDict__.items() if not v['fixed']])
    fixed_species = dict([(pname, p['initial']) for pname, p in m.__sDict__.items() if p['fixed']])
    pardict.update(fixed_species)

    fnspecs = {}
    for R in m.__reactions__: # list ['R1', 'R2', 'R3', 'R4']
        R_info = m.__nDict__[R]
        #assert R_info['Modifiers'] == []
        assert R_info['Type'] == 'Rever'
        arglist = []
        for reagent in R_info['Reagents']:
            r = reagent.replace('self.','')
            if r in varlist:
                arglist.append(r)
        arglist.sort()
        fnspecs[R] = (arglist, R_info['RateEq'].replace('self.',''))

    varspecs = make_varspecs(m, fnspecs)

    for fname, fspec in m.__userfuncs__.items():
        # Don't know how these are implemented yet
        fnspec[fname] = fspec

    dsargs = args(name=fname[:-3],
                  varspecs=varspecs,
                  fnspecs=fnspecs,
                  pars=pardict,
                  ics=icdict,
                  tdata=[0, max_t])

    genclassname = target + '_ODEsystem'
    try:
        genclass = getattr(Generator, genclassname)
    except AttributeError:
        raise TypeError("Invalid ODE solver type")
    return genclass(dsargs)


##################################################
#  CODE NOTES FOR FUTURE DEVELOPMENT
##################################################

#m.__events__ # list of ?
# compartments will be ModelSpec objects

#m.__compartments__ # dict of ?
#m.__eDict__ # dict of events

"""
m.__nDict__ # dict
  = {'R1': {'Modifiers': [],
        'Params': ['self.k1', 'self.x0', 'self.k2'],
        'RateEq': 'self.k1*self.x0-self.k2*self.s0',
        'Reagents': {'self.s0': 1.0, 'self.x0': -1.0},
        'Type': 'Rever',
        'compartment': None,
        'name': 'R1'},
 'R2': {'Modifiers': [],
        'Params': ['self.k3', 'self.k4'],
        'RateEq': 'self.k3*self.s0-self.k4*self.s1',
        'Reagents': {'self.s0': -1.0, 'self.s1': 1.0},
        'Type': 'Rever',
        'compartment': None,
        'name': 'R2'},
 'R3': {'Modifiers': [],
        'Params': ['self.k5', 'self.k6'],
        'RateEq': 'self.k5*self.s1-self.k6*self.s2',
        'Reagents': {'self.s1': -1.0, 'self.s2': 1.0},
        'Type': 'Rever',
        'compartment': None,
        'name': 'R3'},
 'R4': {'Modifiers': [],
        'Params': ['self.k7', 'self.k8', 'self.x3'],
        'RateEq': 'self.k7*self.s2-self.k8*self.x3',
        'Reagents': {'self.s2': -1.0, 'self.x3': 1.0},
        'Type': 'Rever',
        'compartment': None,
        'name': 'R4'}}

m.__pDict__ # dict of param values (not fixed species)
 = {'k1': {'initial': 10.0, 'name': 'k1'},
 'k2': {'initial': 1.0, 'name': 'k2'},
 'k3': {'initial': 5.0, 'name': 'k3'},
 'k4': {'initial': 1.0, 'name': 'k4'},
 'k5': {'initial': 3.0, 'name': 'k5'},
 'k6': {'initial': 1.0, 'name': 'k6'},
 'k7': {'initial': 2.0, 'name': 'k7'},
 'k8': {'initial': 1.0, 'name': 'k8'}}

m.__sDict__ # dict of species descriptions (variable and fixed)
 = {'s0': {'compartment': None,
        'fixed': False,
        'initial': 1.0,
        'isamount': False,
        'name': 's0'},
 's1': {'compartment': None,
        'fixed': False,
        'initial': 1.0,
        'isamount': False,
        'name': 's1'},
 's2': {'compartment': None,
        'fixed': False,
        'initial': 1.0,
        'isamount': False,
        'name': 's2'},
 'x0': {'compartment': None,
        'fixed': True,
        'initial': 10.0,
        'isamount': False,
        'name': 'x0'},
 'x3': {'compartment': None,
        'fixed': True,
        'initial': 1.0,
        'isamount': False,
        'name': 'x3'}}

m.__uDict__ # dict of units
  =  {'area': {'exponent': 2, 'kind': 'metre', 'multiplier': 1.0, 'scale': 0},
 'length': {'exponent': 1, 'kind': 'metre', 'multiplier': 1.0, 'scale': 0},
 'substance': {'exponent': 1, 'kind': 'mole', 'multiplier': 1.0, 'scale': 0},
 'time': {'exponent': 1, 'kind': 'second', 'multiplier': 1.0, 'scale': 0},
 'volume': {'exponent': 1, 'kind': 'litre', 'multiplier': 1.0, 'scale': 0}}
    """
#m.__userfuncs__ # dict of ?
#m.__functions__ # dict of ?
#m.__piecewises__ # dict of ?
#m.__rate_rules__ # list of ?
#m.__rules__ # dict of ?
