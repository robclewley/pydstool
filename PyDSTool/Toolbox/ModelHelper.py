"""A set of functions to help build ODE models with complete
  sets of standard events, etc. based on function specifications
  provided in a format similar to that for sloppyModels.

  Eric Sherwood.


  2008: This is superceded by GDescriptor class. See PyDSTool/tests/
"""

from PyDSTool import *
from copy import copy
from time import perf_counter

class SpiffyODEModel(LeafComponent):
    pass

# Set of all compatible generators
allODEgens = findGenSubClasses('ODEsystem')

def makeSpiffyODEModel(**kw):

    if 'modelName' in kw:
        modelName = kw['modelName']
    else:
        raise KeyError('No modelName specified')

    if 'modelDict' in kw:
        modelDict = kw['modelDict']
    else:
        raise KeyError('No modelDict specified')

    if 'targetGen' in kw:
        targetGen = kw['targetGen']
    else:
        # Default to Dopri_ODEsystem
        targetGen = 'Dopri_ODEsystem'

    if 'globalRefs' in kw:
        globalRefs = kw['globalRefs']
    else:
        globalRefs = []
        kw['globalRefs'] = globalRefs

    if 'algParams' in kw:
        algParams = kw['algParams']
    else:
        algParams = {'init_step':0.05, 'max_pts':5000000, 'atol': 1e-12, 'rtol': 1e-12}
        kw['algParams'] = algParams

    if 'eventDefaults' in kw:
        eventDefaults = kw['eventDefaults']
    else:
        eventDefaults = {
            'starttime': 0,
            'term': False}
        kw['eventDefaults'] = eventDefaults

    if 'eventTol' in kw:
        eventTol = kw['eventTol']
    else:
        eventTol = 1e-9
        kw['eventTol'] = eventTol

    if 'indepVar' in kw:
        indepVar = kw['indepVar']
    else:
        indepVar = ('t', [0, 100000])
        kw['indepVar'] = indepVar

    if 'withEvents' in kw:
        withEvents = kw['withEvents']
    else:
        withEvents = True
        kw['withEvents'] = withEvents

    if 'withJac' in kw:
        withJac = kw['withJac']
    else:
        withJac = False
        kw['withJac'] = withJac

    if 'withJacP' in kw:
        withJacP = kw['withJacP']
    else:
        withJacP = False
        kw['withJacP'] = withJacP

    if 'silent' in kw:
        silent = kw['silent']
    else:
        silent = True
        kw['silent'] = silent

    if 'buildModel' in kw:
        buildModel = kw['buildModel']
    else:
        buildModel = False
        kw['buildModel'] = buildModel

    if 'buildCont' in kw:
        buildCont = kw['buildCont']
    else:
        buildCont = False
        kw['buildCont'] = buildCont



    makeJac = withJac
    makeJacP = withJacP

    if targetGen not in allODEgens:
        # Print a list of the valid targets
        print('Valid target ODE solvers: ' + ", ".join(allODEgens))
        raise ValueError('Invalid target ODE solver')

    # Empty model spec to be populated with the information in modelDict
    theModelSpec = SpiffyODEModel(modelName, compatibleGens=allODEgens)

    if not silent:
        print("Building model: '%s'"%modelName)

    # Build the ODE RHS, save list of ODEs
    varnames = []
    if 'odes' in modelDict.keys():
        for odeName, expr in modelDict['odes'].items():
            theModelSpec.add(Var(expr, odeName, specType='RHSfuncSpec'))
            varnames.append(odeName)
            if not silent:
                print('Added ODE: ', odeName)

    # Build the aux variables, save list of them
    auxvarnames = []
    if 'auxs' in modelDict.keys():
        for auxName, expr in modelDict['auxs'].items():
            theModelSpec.add(Var(expr, auxName, specType='ExpFuncSpec'))
            auxvarnames.append(auxName)
            if not silent:
                print('Added aux variable: ', auxName)

    # Build the list of RHS specific parameters, save list of them
    parnames = []
    if 'params' in modelDict.keys():
        for parName, val in modelDict['params'].items():
            theModelSpec.add(Par(str(val), parName))
            parnames.append(parName)
            if not silent:
                print('Added parameter: ', parName, "=", val)

    # Build the inputs, save list of them
    inputnames = []
    inputdict = {}

    if 'inputs' in modelDict.keys():
        for inputName, val in modelDict['inputs'].items():
            theModelSpec.add(Input(str(inputName)))
            inputnames.append(inputName)
            #if not silent:
            #    print val.variables.keys()
            inputdict[inputName] = val.variables[inputName]
            if not silent:
                print('Added input: ', inputName, "=", val)

    # Build the dict of extra functions, save list of them
    auxfndict = {}
    if 'functions' in modelDict.keys():
        for funSig, expr in modelDict['functions'].items():
            # check syntax; this could be taken care of by Lex-built
            # syntax checking function
            assert ')' == funSig[-1]
            assert '(' in funSig
            # Not sure what this replacement is doing?
            major = funSig.replace(')', '').replace(' ', '').split('(')
            args = major[1].split(',')
            name = major[0]

            theModelSpec.add(Fun(expr, args, name))
            auxfndict[name] = (args, expr)
            # Jacobians are special and might otherwise be handled separately
            if name == 'Jacobian':
                makeJac = False
            if name == 'Jacobian_pars':
                makeJacP = False
            if not silent:
                print('Added function: ', name, " of arguments: ", args)


    # Make symbolic jacobian if requested
    if makeJac:
        if 'Jacobian' not in modelDict.keys():
            varnames.sort()
            dotlist = [modelDict['odes'][x] for x in varnames]
            F = Fun(dotlist, varnames, 'F')
            Jac = Fun(Diff(F, varnames), ['t'] + varnames, 'Jacobian')
            theModelSpec.add(Jac)
            if not silent:
                print("Added jacobian")

    if makeJacP:
        if 'Jacobian_pars' not in modelDict.keys():
            varnames.sort()
            parnames.sort()
            dotlist = [modelDict['odes'][x] for x in varnames]
            G = Fun(dotlist, varnames + parnames, 'G')
            JacP = Fun(Diff(G, varnames + parnames), ['t'] + varnames + parnames, 'Jacobian_pars')
            theModelSpec.add(JacP)
            if not silent:
                print("Added jacobian w.r.t parameters")


    # Construct the model
    targetLang = theGenSpecHelper(targetGen).lang
    # single-generator model so give both same name
    genName = modelName

    theModel = ModelConstructor(modelName,
                                generatorspecs={genName: {'modelspec': theModelSpec,
                                                          'target': targetGen,
                                                          'algparams': algParams
                                                          }},
                                indepvar=indepVar,
                                inputs=inputdict,
                                eventtol=eventTol, withStdEvts={genName: withEvents})

    return theModel
