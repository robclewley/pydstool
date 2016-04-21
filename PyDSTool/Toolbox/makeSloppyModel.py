# makeSloppyModel: conversion from sloppy cell model description to PyDSTool
# Robert Clewley, Oct 2005

from __future__ import absolute_import, print_function

from PyDSTool import *
from PyDSTool.parseUtils import symbolMapClass
from copy import copy

allODEgens = findGenSubClasses('ODEsystem')

class sloppyModel(LeafComponent):
    compatibleGens=allODEgens
    targetLangs=targetLangs

def makeSloppyModel(modelName, modelDict, targetGen, globalRefs=None,
                    algParams=None, silent=False, containsRHSdefs=False):
    """
    containsRHSdefs: A boolean indicating whether or not ODEs contain symbols
    indicating that they should include the right-hand sides of other ODEs.
    To indicate that a right hand side of an ODE should be included in another ODE,
    the following symbol should be used: _var_RHS .
    For example
      dy/dt = -k*x/m      |  'y': '-k*x/m'
      dx/dt = y + (dy/dt) |  'x': 'y + _y_RHS'
    """
    if targetGen not in allODEgens:
        print('Valid target ODE solvers: ' + ", ".join(allODEgens))
        raise ValueError('Invalid target ODE solver')
    sModelSpec = sloppyModel(modelName)
    if not silent:
        print("Building sloppy model '%s'"%modelName)

    # first pass to collect names
    varnames = []
    for odeName in modelDict['odes'].keys():
        varnames.append(odeName)
    parnames = []
    if 'parameters' in modelDict:
        for parName in modelDict['parameters'].keys():
            parnames.append(parName)

    pdomains = {}
    xdomains = {}
    if 'domains' in modelDict:
        for name, dom in modelDict['domains'].items():
            if name in parnames:
                pdomains[name] = dom
            elif name in varnames:
                xdomains[name] = dom
            else:
                raise ValueError("Name %s unknown in domain specs"%name)

    if 'derived_params' in modelDict:
        paramMapping = {}
        for assgnName, expr in modelDict['derived_params'].items():
            if not silent:
                print('Adding derived parameters: ', assgnName)
            paramMapping[assgnName] = '(%s)' % expr
        derivedParamsMap = symbolMapClass(paramMapping)
    else:
        derivedParamsMap = None

    odeItems = list(modelDict['odes'].items())
    if containsRHSdefs:
        # the sentinal chosen to indicate a RHS is '_var_RHS'
        _ode_map = {}
        for k,v in odeItems:
            _ode_map['_%s_RHS' % k] = '(%s)' % v
        odeRHSMap = symbolMapClass(_ode_map)

    for odeName, expr in odeItems:
        if not silent:
            print('Adding ODE: ', odeName)
        if odeName in xdomains:
            odeRHS = Var(expr, odeName, specType='RHSfuncSpec', domain=xdomains[odeName])
        else:
            odeRHS = Var(expr, odeName, specType='RHSfuncSpec')

        if containsRHSdefs:
            if not silent:
                print('Making substitutions based on potential right-hand-side usage in the ODE: ', odeName)
            odeRHS.mapNames(odeRHSMap)

        if derivedParamsMap:
            # Incorporate the derived parameter mappings into the odes.
            # Make this substitution twice because the derived parameters
            # may have internal inter-dependencies. It would be better to
            # have a helper function to resolve these inter-dependencies ahead of time.
            if not silent:
                print('Making derived parameter substitutions in the ODE: ', odeName)
            odeRHS.mapNames(derivedParamsMap)
            odeRHS.mapNames(derivedParamsMap)

        sModelSpec.add(odeRHS)

    auxvarnames = []
    if 'assignments' in modelDict:
        for assgnName, expr in modelDict['assignments'].items():
            if not silent:
                print('Adding assignment: ', assgnName)
            sModelSpec.add(Var(expr, assgnName, specType='ExpFuncSpec'))
            auxvarnames.append(assgnName)

    if 'parameters' in modelDict:
        for parName, val in modelDict['parameters'].items():
            if not silent:
                print('Adding parameter: ', parName, "=", val)
            if parName in pdomains:
                sModelSpec.add(Par(str(val), parName, domain=pdomains[parName]))
            else:
                sModelSpec.add(Par(str(val), parName))

    auxfndict = {}
    for funSig, expr in modelDict['functions'].items():
        assert ')' == funSig[-1]
        assert '(' in funSig
        major = funSig.replace(')','').replace(' ','').split('(')
        args = major[1].split(',')
        name = major[0]
        if not silent:
            print('Adding function: ', name, " of arguments:", args)
        sModelSpec.add(Fun(expr, args, name))
        auxfndict[name] = (args, expr)

    if globalRefs is None:
        globalRefs = []
    if not sModelSpec.isComplete(globalRefs):
        print("Model retains free names: " + ", ".join(sModelSpec.freeSymbols))
        print("These must be resolved in the specification before continuing.")
        print("If one of these is time, then include it explicitly as an")
        print("entry in the argument list ('globalRefs' key)")
        raise ValueError('Incomplete model specification')
    targetlang = theGenSpecHelper(targetGen).lang
    # single-generator model so give both same name
    if algParams is None:
        algParams = {}
    genName = modelName
    sModel = ModelConstructor(modelName,
                            generatorspecs={genName: {'modelspec': sModelSpec,
                                                      'target': targetGen,
                                                      'algparams': algParams}})
    if not silent:
        print("Adding events with default tolerances...")
    argDict={'precise': True, 'term': True}
    evcount = 0
    if 'events' in modelDict:
        for evspec, mappingDict in modelDict['events'].items():
            if evspec[:2] == 'lt':
                dircode = -1
            elif evspec[:2] == 'gt':
                dircode = 1
            else:
                raise ValueError("Invalid event specification: use 'lt' and 'gt'")
            assert '(' == evspec[2], 'Invalid event specification'
            evspec_parts = evspec[3:].replace(')','').replace(' ','').split(',')
            evexpr = evspec_parts[0]
            threshval = evspec_parts[1]
            evname = 'Event'+str(evcount)
            argDict['name'] = evname
            ev = makeZeroCrossEvent(evexpr, dircode, argDict,
                                 varnames, parnames, [], auxfndict, targetlang)
            evcount += 1
            sModel.addEvents(genName, ev)
            evmap = EvMapping(mappingDict,
                              infodict={'vars': varnames+auxvarnames,
                                        'pars': parnames})
            sModel.mapEvent(genName, evname, genName, evmap)
    if not silent:
        print("Building target model with default settings")
    return sModel.getModel()

