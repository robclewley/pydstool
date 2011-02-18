# makeSloppyModel: conversion from sloppy cell model description to PyDSTool
# Robert Clewley, Oct 2005

from PyDSTool import *
from copy import copy

allODEgens = findGenSubClasses('ODEsystem')

class sloppyModel(LeafComponent):
    compatibleGens=allODEgens
    targetLangs=targetLangs

def makeSloppyModel(modelName, modelDict, targetGen, globalRefs=[],
                    algParams={}, silent=False):
    if targetGen not in allODEgens:
        print 'Valid target ODE solvers: ' + ", ".join(allODEgens)
        raise ValueError('Invalid target ODE solver')
    sModelSpec = sloppyModel(modelName)
    if not silent:
        print "Building sloppy model '%s'"%modelName
    varnames = []
    for odeName, expr in modelDict['odes'].iteritems():
        if not silent:
            print 'Adding ODE: ', odeName
        sModelSpec.add(Var(expr, odeName, specType='RHSfuncSpec'))
        varnames.append(odeName)
    auxvarnames = []
    for assgnName, expr in modelDict['assignments'].iteritems():
        if not silent:
            print 'Adding assignment: ', assgnName
        sModelSpec.add(Var(expr, assgnName, specType='ExpFuncSpec'))
        auxvarnames.append(assgnName)
    parnames = []
    for parName, val in modelDict['parameters'].iteritems():
        if not silent:
            print 'Adding parameter: ', parName, "=", val
        sModelSpec.add(Par(str(val), parName))
        parnames.append(parName)
    auxfndict = {}
    for funSig, expr in modelDict['functions'].iteritems():
        assert ')' == funSig[-1]
        assert '(' in funSig
        major = funSig.replace(')','').replace(' ','').split('(')
        args = major[1].split(',')
        name = major[0]
        if not silent:
            print 'Adding function: ', name, " of arguments:", args
        sModelSpec.add(Fun(expr, args, name))
        auxfndict[name] = (args, expr)
    if sModelSpec.freeSymbols != []:
        print "Model retains free names: " + ", ".join(sModelSpec.freeSymbols)
        print "These must be resolved in the specification before continuing."
        print "If one of these is time, then include it explicitly as an"
        print "entry in the argument list ('globalRefs' key)"
        raise ValueError('Incomplete model specification')
    targetlang = theGenSpecHelper(targetGen).lang
    # single-generator model so give both same name
    genName = modelName
    sModel = ModelConstructor(modelName,
                            generatorspecs={genName: {'modelspec': sModelSpec,
                                                      'target': targetGen,
                                                      'algparams': algParams}})
    if not silent:
        print "Adding events with default tolerances..."
    argDict={'precise': True, 'term': True}
    evcount = 0
    for evspec, mappingDict in modelDict['events'].iteritems():
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
        evmap = makeEvMapping(mappingDict, varnames+auxvarnames, parnames)
        sModel.mapEvent(genName, evname, genName, evmap)
    if not silent:
        print "Building target model with default settings"
    return sModel.getModel()

