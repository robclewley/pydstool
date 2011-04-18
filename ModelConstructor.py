"""Model Constructor classes.

   Instantiate abstract Model Specifications into concrete simulation code.

   Robert Clewley, September 2005.


Overview of steps that ModelConstructor takes:

1. Build Generators for each vector field
 * Take flattened spec and select a compatible Generator type and language
   * include check that compatibleGen selection actually exists,
      and is compatible with specType/domain, and with cts/discrete
      time variables)
   * presently, each v.f. must have all the same Gen type
 * Map things like x^y and x**y power specs -> pow(x,y)
    and abs() function -> fabs() in C.

2. Given the specific target lang/Generator with a ModelSpec vector field
 * Associate additional events, code inserts, reused vars
 * Choose which domains map to domain verification events (certainly none that
    are infinite -- although semi-infinite can be checked in one direction)
 * All non-infinite domain ends would have an event created for them, but
    default is for them to be inactive.

 The 'mspec' argument to the GeneratorConstructor class must be a complete
 ModelSpec (up to introduction of global references including definitions of
 external inputs).

 The resulting model output from getModel contains the ModelSpec mspec in
 order to use its structure in resolving information about the relationship
 between variables.
"""

# PyDSTool imports
from errors import *
from common import *
from utils import info, remain, intersect
import Model, Generator, ModelSpec, Symbolic, Events, MProject
from parseUtils import symbolMapClass, NAMESEP, isNumericToken

# Other imports
from numpy import Inf, NaN, isfinite,  array, \
     arange, zeros, ones, concatenate, swapaxes, take, \
     sometrue, alltrue, any, all
import numpy, scipy, math # for access by user-defined code of EvMapping
import sys, types, copy

# Exports
__all__ = ['GeneratorConstructor', 'ModelConstructor', 'makeModelInfo',
           'makeModelInfoEntry', 'embed', 'EvMapping', 'makeEvMapping',
           'GDescriptor', 'MDescriptor']

# -----------------------------------------------------------------------------

mathNameMap = dict(zip(Symbolic.allmathnames_symbolic,
                       Symbolic.allmathnames))


class Descriptor(args):
    """Abstract class for model and generator descriptors"""
    _validKeys = ()
    _defaults = {}  # for values other than None
    _checkKeys = ()  # values that must be set in order for the generator to be instantiable

    def __init__(self, **kw):
        self.__dict__ = filteredDict(kw, self._validKeys)
        if remain(kw.keys(), self._validKeys) != []:
            print "Valid keys: ", self._validKeys
            raise ValueError("Invalid keys provided for Model Descriptor")
        done_defs = []
        for def_key in remain(self._defaults.keys(), kw.keys()):
            def_value = self._defaults[def_key]
            done_defs.append(def_key)
            self.__dict__[def_key] = def_value
        for key in remain(self._validKeys, kw.keys()+done_defs):
            self.__dict__[key] = None

    def validate(self):
        raise NotImplementedError("Defined in concrete sub-class")

    def __repr__(self):
        return className(self)

    __str__ = __repr__



class GDescriptor(Descriptor):
    """All-in-one descriptor class for single Generators, and information
    necessary to be able to build a Model object using a ModelConstructor call
    -- i.e. for forming a valid 'generatorspecs' field.
    """
    _validKeys = ('changelog', 'orig_name', 'modelspec', 'description',
                  'algparams', 'target', 'withStdEvts', 'stdEvtArgs', 'withJac',
                  'withJacP', 'reuseTerms', 'eventPars', 'unravelInfo',
                  'userEvents', 'userFunctions', 'userEventMaps', 'options')
    _defaults = {'description': '', 'withStdEvts': False, 'withJac': False,
                 'withJacP': False, 'unravelInfo': True, 'options': {}}
    _checkKeys = ('target')

#    def __init__(self, **kw):
#        Descriptor.__init__(self, **kw)
#        self.component_hierarchy = GTree(self.modelspec)

    def __getitem__(self, hier_name):
        """Return object in model spec named using the hierarchical
        naming format.
        """
        return self.modelspec[hier_name]

    def search(self, hier_name):
        return self.modelspec.search(hier_name)

    def validate(self):
        validated = isinstance(self.modelspec, ModelSpec.ModelSpec) and \
                  self.modelspec.isComplete() and \
                  self.modelspec.isDefined(ignoreInputs=True)
        freeSymbols = self.modelspec.freeSymbols
        return (validated, freeSymbols)

    def isinstantiable(self):
        return self.validate()[0] and self.target is not None



class MDescriptor(Descriptor):
    """All-in-one descriptor class for hybrid model definitions and information
    necessary to be able to build a Model object using a ModelConstructor call.

    generatorspecs should be a dictionary of gen modelspec names -> modelspecs.
    """
    _validKeys = ('changelog', 'orig_name', 'name', 'generatorspecs',
                  'description', 'eventtol', 'abseps', 'activateAllBounds',
                  'checklevel', 'tdata', 'indepvar', 'icvalues', 'parvalues',
                  'inputs', 'unravelInfo')
    _defaults = {'description': '', 'indepvar': ('t', [-Inf,Inf]),
                 'checklevel': 2, 'activateAllBounds': False,
                 'generatorspecs': {}, 'icvalues': {}, 'parvalues': {},
                 'inputs': {}, 'unravelInfo': True}
    _checkKeys = ('icvalues', 'parvalues', 'inputs')

    def validate(self):
        assert hasattr(self.generatorspecs, 'values') and \
               hasattr(self.generatorspecs, 'keys')
        validated = alltrue([isinstance(gd, GDescriptor) for \
                             gd in self.generatorspecs.values()])
        # check for consistency of internal interfaces
        inconsistencies = []
        return (validated, inconsistencies)

    def isinstantiable(self):
        valid = self.validate()[0]
        vars_i_all = True
        pars_i_all = True
        inps_i_all = True
        for ms in self.generatorspecs.values():
            all_vars = ms.modelspec.search('Var')
            dyn_vars = [v for v in all_vars if ms.modelspec._registry[v].obj.specType == 'RHSfuncSpec']
            vars_i = alltrue([varname in self.icvalues for \
                              varname in dyn_vars])
            pars_i = alltrue([(parname in self.parvalues or \
                               par.spec.specStr !='') \
                          for parname, par in ms.modelspec.pars.items()])
            inps_i = alltrue([inpname in self.inputs for \
                              inpname in ms.modelspec.inputs])
            vars_i_all = vars_i_all and vars_i
            pars_i_all = pars_i_all and pars_i
            inps_i_all = inps_i_all and inps_i
        return valid and vars_i_all and pars_i_all and inps_i_all

    def get_desc(self, name):
        if name in self.generatorspecs:
            return self.generatorspecs[name]
        else:
            raise KeyError('Generator %s does not exist in registry'%name)

    def add(self, gd):
        self.generatorspecs[gd.modelspec.name] = gd


# ------------------------------------------------


class GeneratorConstructor(object):
    def __init__(self, mspec=None, userevents=None, userfns=None,
                 unravelInfo=True, inputs=None, checklevel=2,
                 activateAllBounds=False, activatedBounds=None,
                 targetGen="", algparams=None, indepvar=('t',[-Inf,Inf]),
                 tdata=None, parvalues=None, icvalues=None, reuseTerms=None,
                 options=None, abseps=None, eventPars=None, preReuse=False,
                 preReuseTerms=None, preFlat=False):
        """Notes for initialization arguments:

        mspec : corresponding ModelSpec, for reference

        userevents : list of Event objects
        userfns : dictionary of named user functions specs
        inputs : dictionary of Variable objects
        algparams : dictionary of algorithmic parameters for Generator
        parvalues : dictionary of parameter values
        icvalues : dictionary of initial condition values
        reuseterms : dictionary of reused terms in specifications

        targetGen : STRING name of any compatible Generator class, e.g. 'Vode_ODEsystem'

        eventPars : list of parameter names associated solely with events

        options : Internal use by ModelConstructor
        preReuse : Internal use
        preFlat : Internal use

        RETURNS: getGenerator method returns a Generator of the specified class
        """
        self.mspec = mspec
        # user events are additional to the intrinsic constraint events
        # that are made automatically from the variables' bounds information
        if userevents is None:
            self.userevents = []
        else:
            self.userevents = copy.copy(userevents)
        if userfns is None:
            self.userfns = {}
        else:
            # ensure a list of Symbolic defs is converted to
            # the dictionary of string signatures and definitions format
            self.userfns = Symbolic.ensureStrArgDict(copy.copy(self.userfns))
        self.unravelInfo = unravelInfo   # presently just a Boolean
        if type(targetGen)==str:
            self.targetGen = targetGen
        else:
            raise TypeError("targetGen argument must be a string")
        if algparams is None:
            self.algparams = {}
        else:
            self.algparams = copy.copy(algparams)
        self.indepvarname = indepvar[0]
        self.indepvardomain = indepvar[1]
        self.tdata = tdata
        if inputs is None:
            self.inputs = {}
        else:
            self.inputs = copy.copy(inputs)
        if parvalues is None:
            self.parvalues = {}
        else:
            self.parvalues = copy.copy(parvalues)
        if icvalues is None:
            self.icvalues = {}
        else:
            self.icvalues = copy.copy(icvalues)
        self.checklevel = checklevel
        self.forcedAuxVars = []
        if options is None:
            self.optDict = {}
        else:
            self.optDict = copy.copy(options)
        if reuseTerms is None:
            self.reuseTerms = {}
        else:
            self.reuseTerms = copy.copy(reuseTerms)
        self.vfcodeinsert_start = ""
        self.vfcodeinsert_end = ""
        if activatedBounds is None:
            self.activatedBounds = {}
        else:
            self.activatedBounds = copy.copy(activatedBounds)
        self.activateAllBounds = activateAllBounds  # overrides activatedBounds
        if abseps is None:
            self.abseps = 1e-13
        else:
            self.abseps = abseps
        # List of parameter names associated solely with events
        if eventPars is None:
            self.eventPars = []
        else:
            self.eventPars = copy.copy(eventPars)
        self.preReuse = preReuse
        if preReuseTerms is None:
            self.preReuseTerms = {}
        else:
            self.preReuseTerms = copy.copy(preReuseTerms)
        self.preFlat = preFlat


    def setForcedAuxVars(self, vlist):
        self.forcedAuxVars = vlist

    def setReuseTerms(self, rdict):
        self.reuseTerms = rdict
        self.preReuse = False

    def setVfCodeInsertStart(self, codestr):
        self.vfcodeinsert_start = codestr

    def setVfCodeInsertEnd(self, codestr):
        self.vfcodeinsert_end = codestr

    def setOptions(self, optDict):
        # e.g. for 'nobuild' option for C-based ODE integrators
        self.optDict = optDict

    def addEvents(self, evtarg, eventPars=None):
        if isinstance(evtarg, list):
            self.userevents.extend(evtarg)
        elif isinstance(evtarg, Events.Event):
            self.userevents.append(evtarg)
        else:
            raise TypeError("Invalid event or event list")
        # Use this list to determine whether parameters are event specific
        if eventPars is not None and eventPars != [] and eventPars != '':
            if isinstance(eventPars, list):
                self.eventPars.extend(eventPars)
            elif isinstance(eventPars, str):
                self.eventPars.append(eventPars)


    def addFunctions(self, fnarg):
        self.userfns.update(Symbolic.ensureStrArgDict(copy.copy(fnarg)))

    def activateBounds(self, varname=None, which_bounds='all'):
        """which_bounds argument is either 'lo', 'hi', or a pair ('lo', 'hi').
        Calling with no arguments activates all bounds."""
        if varname is None and which_bounds=='all':
            self.activateAllBounds = True
        else:
            entry = [False,False]
            if 'hi' in which_bounds:
                entry[1] = True
            if 'lo' in which_bounds:
                entry[0] = True
            self.activatedBounds[varname] = entry


    def getGenerator(self):
        """Build and return a Generator instance from an abstract
        specification."""
        ### Instantiate (flatten) target model structured specification
        ## Flatten ModelSpec self.mspec using indepvarname global and inputs
        # and using connectivity bindings (latter not yet implemented)
        globalRefs = [self.indepvarname] + self.inputs.keys()
        self.mspec.eventPars = copy.copy(self.eventPars)

        if not self.preFlat:
            try:
                flatspec = self.mspec.flattenSpec(multiDefUnravel=self.unravelInfo,
                                                  globalRefs=globalRefs,
                                                  ignoreInputs=True)
            except KeyboardInterrupt:
                raise
            except:
                print "Problem flattening Model Spec '%s'"%self.mspec.name
                print "Global refs: ", globalRefs
                raise
        else:
            flatspec = self.mspec.flatSpec

        FScompatibleNames = flatspec['FScompatibleNames']
        FScompatibleNamesInv = flatspec['FScompatibleNamesInv']
        ## Check target Generator info
        if self.targetGen in self.mspec.compatibleGens \
                   and self.targetGen in Generator.theGenSpecHelper:
            gsh = Generator.theGenSpecHelper(self.targetGen)
            if gsh.lang not in self.mspec.targetLangs:
                raise ValueError("Incompatible target language between supplied"
                                 " ModelSpec and target Generator")
        else:
            print "ModelSpec's compatible Generators:", \
                  ", ".join(self.mspec.compatibleGens)
            print "ModelConstructor target Generator:", self.targetGen
            raise ValueError("Target Generator mismatch during generator "
                             "construction")
        self.targetLang = gsh.lang
        ## Make Generator initialization argument dictionary
        a = args(abseps=self.abseps)
        a.pars = {}
        parnames = flatspec['pars'].keys()
        for p, valstr in flatspec['pars'].iteritems():
            if valstr == '':
                if FScompatibleNamesInv(p) not in self.parvalues:
                    raise ValueError("Parameter %s is missing a value"%FScompatibleNamesInv(p))
            else:
                if valstr == p:
                    # placeholder
                    a.pars[p] = 0
                    #raise NotImplementedError
                else:
                    try:
                        a.pars[p] = float(valstr)
                    except ValueError:
                        raise ValueError("Invalid parameter value set in ModelSpec"
                                         " for '%s', value: %s"%(p,valstr))
        # override any par vals set in ModelSpec with those explicitly set
        # here
        for p, val in self.parvalues.iteritems():
            try:
                pr = FScompatibleNames(p)
            except KeyError:
                raise NameError("Parameter '%s' missing from ModelSpec"%p)
            if pr not in flatspec['pars']:
                raise NameError("Parameter '%s' missing from ModelSpec"%p)
            a.pars[pr] = val
        if self.icvalues != {}:
            a.ics = {}
            for v, val in self.icvalues.iteritems():
                try:
                    vr = FScompatibleNames(v)
                except KeyError:
                    raise NameError("Variable '%s' missing from ModelSpec"%v)
                if vr not in flatspec['vars']:
                    raise NameError("Variable '%s' missing from ModelSpec"%v)
                a.ics[vr] = val
        a.tdomain = self.indepvardomain
        if self.tdata is not None:
            a.tdata = self.tdata
        # a.ttype = float or int ?
        a.inputs = self.inputs
        a.name = self.mspec.name
        xdomain = {}
        pdomain = {}
        for k, d in flatspec['domains'].iteritems():
            # e.g. d == (float, Continuous, [-Inf, Inf])
            if k in flatspec['vars']:
                assert len(d[2]) == 2, "Domain spec must be a valid interval"
                xdomain[k] = d[2]
            elif k in flatspec['pars']:
                assert len(d[2]) == 2, "Domain spec must be a valid interval"
                pdomain[k] = d[2]
            if d[1] != gsh.domain:
                raise AssertionError("Domain mismatch (%s) with target Generator's (%s)"%(d[1],gsh.domain))
        a.xdomain = xdomain
        a.pdomain = pdomain
        exp_vars = [v for (v,t) in flatspec['spectypes'].items() \
                         if t == 'ExpFuncSpec']
        rhs_vars = [v for (v,t) in flatspec['spectypes'].items() \
                         if t == 'RHSfuncSpec']
        imp_vars = [v for (v,t) in flatspec['spectypes'].items() \
                         if t == 'ImpFuncSpec']
        if gsh.specType == 'RHSfuncSpec':
            assert imp_vars == [], "Cannot use implicitly defined variables"
 #           assert self.forcedAuxVars == [], "Cannot force auxiliary variables"
            varnames = rhs_vars
            auxvarnames = exp_vars
        elif gsh.specType == 'ExpFuncSpec':
            assert imp_vars == [], "Cannot use implicitly defined variables"
            assert rhs_vars == [], "Cannot use RHS-type variables"
            varnames = exp_vars
            invalid_auxvars = remain(self.forcedAuxVars, varnames)
            if invalid_auxvars == []:
                # then all forced aux varnames were legitimate
                # so remove them from varnames and put them in auxvarnames
                varnames = remain(varnames, self.forcedAuxVars)
                auxvarnames = self.forcedAuxVars
            else:
                print "Invalid auxiliary variable names:"
                print invalid_auxvars
#               raise ValueError("Forced auxiliary variable names were invalid")
        elif gsh.specType == 'ImpFuncSpec':
            assert rhs_vars == [], "Cannot use RHS-type variables"
            varnames = imp_vars
            auxvarnames = exp_vars
        # search for explicit variable interdependencies and resolve by
        # creating 'reuseterms' declarations, substituting in the cross-ref'd
        # definitions
        # e.g. state variables v and w, and explicit aux vars are given by:
        #     x = 1+v
        #     y = f(x) + w
        # Here, y illegally depends on x, so define a 'reused' temporary
        # definition, and re-write in terms of that:
        #     temp = 1+v
        #     x = temp
        #     y = f(temp) + w
        # e.g. state variables v and w, aux var x:
        #     v' = 1-v -f(x)
        # Here, v illegally uses an auxiliary variable on the RHS, so make
        # a 'reused' substitution as before
        #
        # first pass to find which substitutions are needed

        # DO THIS PART AGAIN ONLY IF BOOLEAN IS SET
        if not self.preReuse:
            reuseTerms, subsExpr = processReused(varnames+auxvarnames, auxvarnames,
                                                 flatspec, self.mspec._registry,
                                                 FScompatibleNames, FScompatibleNamesInv)

            clash_reused = intersect(reuseTerms.keys(), self.reuseTerms.keys())
            if clash_reused != []:
                print "Clashing terms:", clash_reused
                raise ValueError("User-supplied reused terms clash with auto-"
                                 "generated terms")
            # second pass, this time to actually make the substitutions
            for v in subsExpr:
                flatspec['vars'][v] = subsExpr[v]
            reuseTerms.update(self.reuseTerms)
            a.reuseterms = reuseTerms

        else:
            a.reuseterms = self.preReuseTerms

        a.varspecs = dict(zip(varnames+auxvarnames, [flatspec['vars'][v] \
                                      for v in varnames+auxvarnames]))
        a.auxvars = auxvarnames
        a.fnspecs = self.userfns
        try:
            a.fnspecs.update(flatspec['auxfns'])
        except KeyError:
            # no aux fns defined in flat spec!
            pass
        a.checklevel = self.checklevel
        a.algparams = self.algparams
        if self.vfcodeinsert_start != "":
            a.vfcodeinsert_start = self.vfcodeinsert_start
        if self.vfcodeinsert_end != "":
            a.vfcodeinsert_end = self.vfcodeinsert_end
        ## Events
        events = []
        # make events from bound constraints (activated accordingly)
        # (parameter bounds only useful for continuation with PyCont)
        nonEvtParnames = remain(parnames, self.eventPars)
        domnames = varnames+nonEvtParnames
        for xname in domnames:
            hier_name_lo = FScompatibleNamesInv(xname)+"_domlo"
            FScompatibleNames[hier_name_lo] = xname+"_domlo"
            FScompatibleNamesInv[xname+"_domlo"] = hier_name_lo
            hier_name_hi = FScompatibleNamesInv(xname)+"_domhi"
            FScompatibleNames[hier_name_hi] = xname+"_domhi"
            FScompatibleNamesInv[xname+"_domhi"] = hier_name_hi
        if self.activateAllBounds:
            a.activatedbounds = {}.fromkeys(domnames,(True,True))
        else:
            a.activatedbounds = self.activatedBounds
        a.enforcebounds = True  # not currently manipulated, used in Generator baseclasses
        # add events from user events
        for e in self.userevents:
            if e not in events:
                events.append(e)
            else:
                raise ValueError("Repeated event definition!")

        a.events = events
        # Add any additional special options (e.g. 'nobuild' directive)
        for k,v in self.optDict.iteritems():
            if hasattr(a, k):
                raise KeyError("'%s' already exists as a Generator argument"%k)
            a.k = v
        a.FScompatibleNames = FScompatibleNames
        a.FScompatibleNamesInv = FScompatibleNamesInv

        # Parameters solely associated with events -- don't make domain events for them
        a.eventPars = self.eventPars

        # keep a copy of the arguments in self for users to see what was done
        self.conargs = copy.copy(a)

        ## Make Generator
        try:
            return gsh.genClass(a)
        except:
            print "Problem initializing target Generator '%s'"%self.targetGen
            raise


    def __del__(self):
        del self.userevents
        if hasattr(self, 'conargs'):
            try:
                del self.conargs.events
            except AttributeError:
                pass


# -----------------------------------------------------------------------------


class ModelConstructor(object):
    def __init__(self, name, userevents=None, userfns=None, unravelInfo=True,
                 inputs=None, checklevel=2, activateAllBounds=False,
                 generatorspecs=None, indepvar=('t',[-Inf,Inf]),
                 parvalues=None, icvalues=None, tdata=None, reuseTerms=None,
                 withJac=None, withJacP=None, featureDicts=None,
                 abseps=None, eventtol=None, eventPars=None,
                 withStdEvts=None, stdEvtArgs=None):
        """Notes for initialization arguments.

        name : string name of this ModelConstructor

        ** The following are applied to all Generators

        activateAllBounds : Boolean
        checklevel : integer
        indepvar : pair of (independent var name, pair giving domain interval)

        ** The following are dictionaries keyed by Generator name, with values:

        generatorspecs : ModelSpecs
        userevents : list of Event objects
        userfns : dictionary of named user functions specs
        inputs : dictionary of Variable objects
        algparams : dictionary of algorithmic parameters for Generator
        parvalues : dictionary of parameter values
        icvalues : dictionary of initial condition values
        reuseterms : dictionary of reused terms in specifications
        eventPars : list of parameter names associated solely with events
        withStdEvts : Boolean for making standard events (bounds & turning points)
        stdEvtArgs : arguments for the standard events
        featureDicts : dictionary of Features for making each Model Interface
        withJac : Boolean for making Jacobian
        withJacP : Boolean for making Jacobian with respect to parameters

        RETURNS: getModel method returns a Model object
        """
        self.name = name
        self.forcedIntVars = []
        if generatorspecs is None:
            self._generators = {}
        else:
            self._generators = copy.copy(generatorspecs)
        if userevents is None:
            self._events = {}
        else:
            self._events = copy.copy(userevents)
        if userfns is None:
            self._funcs = {}
        else:
            self._funcs = copy.copy(userfns)
        self.indepvar = indepvar
        self.indepvarname = self.indepvar[0]
        self.eventmaps = {}
        if reuseTerms is None:
            self.reuseTerms = {}
        else:
            self.reuseTerms = copy.copy(reuseTerms)
        if inputs is None:
            self.inputs = {}
        else:
            self.inputs = copy.copy(inputs)
        if parvalues is None:
            self.parvalues = {}
        else:
            self.parvalues = copy.copy(parvalues)
        if icvalues is None:
            self.icvalues = {}
        else:
            self.icvalues = copy.copy(icvalues)
        self.tdata = tdata
        self.checklevel = checklevel
        self.unravelInfo = unravelInfo   # presently just a Boolean
        self.activatedBounds = {}
        self.activateAllBounds = activateAllBounds  # overrides activatedBounds
        if abseps is None:
            abseps = 1e-13
        self.abseps = abseps
        if eventtol is None:
            eventtol = abseps * 1e3
        self.eventtol = eventtol
        if withJac is None:
            self.withJac = {}
        else:
            self.withJac = copy.copy(withJac)
        if withJacP is None:
            self.withJacP = {}
        else:
            self.withJacP = copy.copy(withJacP)
        if withStdEvts is None:
            self.withStdEvts = {}
        else:
            self.withStdEvts = copy.copy(withStdEvts)
        if stdEvtArgs is None:
            self.stdEvtArgs = {}
        else:
            self.stdEvtArgs = copy.copy(stdEvtArgs)
        if featureDicts is None:
            self.featureDicts = {}
        else:
            self.featureDicts = copy.copy(featureDicts)

        # At this point, process our reuse terms, so we don't have to do it
        # again. Includes flattening the spec?
        self.preFlat = {} # dictionary keyed by generator; whether it's been flattened already (change in addEvents)
        self.preReuse = {} # dictionary keyed by generator; whether it's reused terms have been processed already
        self.preReuseTerms = {}
        for g in self._generators:
            self.preFlat[g] = False
            self.preReuse[g] = False
            self.preReuseTerms[g] = {}

        if self.withJac != {}:
            self.createJac()
        if self.withJacP != {}:
            self.createJacP()

        # dictionary of lists of parameter names associated solely with events;
        # keyed by generator name
        if eventPars is None:
            self._eventPars = {}
        else:
            self._eventPars = copy.copy(eventPars)

        if self.withStdEvts != {}:
            self.preprocessFlatten()
            self.preprocessReuseTerms()
            self.createStdEvts()


    def __repr__(self):
        return "ModelConstructor %s"%self.name

    def preprocessFlatten(self):
        globalRefs = [self.indepvarname] + self.inputs.keys()
        for g in self._generators:
            gspec = self._generators[g]['modelspec']
            try:
                flatspec = gspec.flattenSpec(multiDefUnravel=self.unravelInfo, globalRefs=globalRefs,
                                             ignoreInputs=True)
            except KeyboardInterrupt:
                raise
            except:
                print "Problem flattening Model Spec %s"%self.mspec.name
                print "Global refs: ", globalRefs
                raise
            self.preFlat[g] = True

    def preprocessReuseTerms(self):
        for g in self._generators:
            gspec = self._generators[g]['modelspec']
            assert self.preFlat[g]
            flatspec = gspec.flatSpec
            gsh = Generator.theGenSpecHelper(self._generators[g]['target'])

            FScompatibleNames = flatspec['FScompatibleNames']
            FScompatibleNamesInv = flatspec['FScompatibleNamesInv']

            exp_vars = [v for (v,t) in flatspec['spectypes'].items() \
                        if t == 'ExpFuncSpec']
            rhs_vars = [v for (v,t) in flatspec['spectypes'].items() \
                        if t == 'RHSfuncSpec']
            imp_vars = [v for (v,t) in flatspec['spectypes'].items() \
                        if t == 'ImpFuncSpec']
            if gsh.specType == 'RHSfuncSpec':
                assert imp_vars == [], "Cannot use implicitly defined variables"
#                assert self.forcedAuxVars == [], "Cannot force auxiliary variables"
                varnames = rhs_vars
                auxvarnames = exp_vars
            elif gsh.specType == 'ExpFuncSpec':
                assert imp_vars == [], "Cannot use implicitly defined variables"
                assert rhs_vars == [], "Cannot use RHS-type variables"
                varnames = exp_vars
                #invalid_auxvars = remain(self.forcedAuxVars, varnames)
                #if invalid_auxvars == []:
                    ## then all forced aux varnames were legitimate
                    ## so remove them from varnames and put them in auxvarnames
                    #varnames = remain(varnames, self.forcedAuxVars)
                    #auxvarnames = self.forcedAuxVars
                #else:
                    #print "Invalid auxiliary variable names:"
                    #print invalid_auxvars
                    #raise ValueError("Forced auxiliary variable names were invalid")
            elif gsh.specType == 'ImpFuncSpec':
                assert rhs_vars == [], "Cannot use RHS-type variables"
                varnames = imp_vars
                auxvarnames = exp_vars

            reuseTerms, subsExpr = processReused(varnames+auxvarnames, auxvarnames,
                                                 flatspec, gspec._registry,
                                                 FScompatibleNames, FScompatibleNamesInv)

            clash_reused = intersect(reuseTerms.keys(), self.reuseTerms.keys())
            if clash_reused != []:
                print "Clashing terms:", clash_reused
                raise ValueError("User-supplied reused terms clash with auto-"
                                 "generated terms")

            # second pass, this time to actually make the substitutions
            for v in subsExpr:
                flatspec['vars'][v] = subsExpr[v]
            reuseTerms.update(self.reuseTerms)

            # Need to make reuseterms universally available
            self.preReuse[g] = True
            self.preReuseTerms[g] = reuseTerms

    def addModelInfo(self, genSpec, genTarg, genAlgPars={}, unravelInfo={},
                   genOpts={}):
        """genSpec can either be a complete ModelSpec description or a
        string-based dictionary of definitions.
        """
        if isinstance(genSpec, dict):
            genSpec = args(**genSpec)
        if len(genAlgPars)==0:
            # in case user gave a string-based definition, algparams
            # may already be given in that definition.
            if hasattr(genSpec, 'algparams'):
                genAlgPars = genSpec['algparams']
        if hasattr(genSpec, 'events'):
            self.addEvents(genSpec.name, genSpec.events)
        self._generators[genSpec.name] = args(modelspec=genSpec,
                                          target=genTarg,
                                          algparams=copy.copy(genAlgPars),
                                          unravelInfo=copy.copy(unravelInfo),
                                          options=copy.copy(genOpts))

    def createStdEvts(self):
        evtArgsDefaults =  {'eventtol': self.eventtol,
                            'eventdelay': self.eventtol*1e4,
                            'starttime': 0,
                            'term': False,
                            'active': False}

        rhsEvtTypeList = ['val', 'deriv', 'stat']
        expEvtTypeList = ['val']
        withEvtParList = ['val', 'deriv']
        evtDirList = [('inc', 1), ('dec', -1), ('neut', 0)]
        specList = ['auxfns', 'vars']
        evtParList = []
        for g in self._generators:
            targetLang = Generator.theGenSpecHelper(self._generators[g]['target']).lang
            evtList = []
            try:
                makeEvts = self.withStdEvts[g]
            except KeyError:
                makeEvts = False
            if makeEvts:
                gspec = self._generators[g]['modelspec']
                if not self.preFlat[g]:
                    print "Flattening"
                    gspec.flattenSpec()
                fspec = gspec.flatSpec
                # name maps
                FScNM = fspec['FScompatibleNames']
                FScNMInv = fspec['FScompatibleNamesInv']
                # temp dict to store new event par name mappings
                FSc_update_dict = {}
                FScInv_update_dict = {}
                try:
                    stdEvtArgs = self.stdEvtArgs[g]
                except KeyError:
                    stdEvtArgs = evtArgsDefaults

                # Make event functions for auxfns
                evtTypeList = expEvtTypeList
                for s in specList:
                    if s not in fspec.keys():
                        continue

                    # auxfns are only explicit types
                    if s == 'auxfns':
                        evtTypeList = expEvtTypeList
                        checkEvtType = False
                    else:
                        evtTypeList = []
                        checkEvtType = True

                    for f in fspec[s].keys():
                        if checkEvtType:
                            if fspec['spectypes'][f] == 'ExpFuncSpec':
                                evtTypeList = expEvtTypeList
                            elif fspec['spectypes'][f] == 'RHSfuncSpec':
                                evtTypeList = rhsEvtTypeList
                            else:
                                raise PyDSTool_ValueError("Don't know this "
                                                          "spec type.")

                            # val, deriv, stat
                            for evtType in evtTypeList:
                                # inc, dec, neut
                                for i in range(len(evtDirList)):
                                    # make event, parameter names (auxfns can only hit values, not test derivs)
                                    evtName = f + '_'+ evtType + '_' + evtDirList[i][0] + '_evt'
                                    evtNameFSInv = FScNMInv(f) + '_'+ evtType + '_' + evtDirList[i][0] + '_evt'
                                    # If there is an event parameter associated with this kind of event
                                    if evtType in withEvtParList:
                                        parname = evtName+'_p'
                                        FScInv_update_dict[parname] = evtNameFSInv+'_p'
                                        FSc_update_dict[evtNameFSInv+'_p'] = parname
                                        # default param value is 0
                                        par = Symbolic.Par(str(0), parname)
                                        par.compatibleGens = gspec.compatibleGens
                                        # add parameter to modelspec pars
                                        # add parameters names, values to flattened spec
                                        gspec.pars[parname] = par
                                        fspec['pars'][parname] = 0 # default value is 0

                                    # make the associated event
                                    # Error correction: var val events are on the variable value, not the deriv. value
                                    if s == 'vars' and evtType == 'val':
                                        evtStr = f + ' - '  + parname
                                    elif evtType in withEvtParList:
                                        evtStr = fspec[s][f] + '-' + parname
                                    else:
                                        evtStr = fspec[s][f]
                                    # Adding the event is the same for all cases
                                    evtDir = evtDirList[i][1]
                                    evtArgs = stdEvtArgs
                                    evtArgs['name'] = evtName
                                    evtSuccess = True
                                    # Some events can't be made if they are ill-formed (currently arises
                                    # with the neural toolbox auxilliary variables)
                                    try:
                                        if self.preReuse[g]:
                                            # This has a conflict with LowLevelEvent class which expects
                                            # there to just be a return string -- fix later
                                            #       reuseterms = self.preReuseTerms[g]
                                            reuseterms = {}
                                        else:
                                            reuseterms = {}
                                        theEvt = Events.makeZeroCrossEvent(expr=evtStr,
                                                                           dircode=evtDir,
                                                                           argDict=evtArgs,
                                                                           targetlang=targetLang,
                                                                           flatspec=fspec,
                                                                           reuseterms=reuseterms)
                                    except ValueError, errinfo:
                                        evtSuccess = False
                                        #print "Warning: Could not make standard event " + evtName + " with definition " + evtStr
                                        #print "  Original problem: ", errinfo
                                        #print "  Skipping this event."
                                    if evtSuccess:
                                        evtList.append(theEvt)
                                    # Add the event parameter to evtParList even if building event was
                                    # a failure, since we have already made the parameter and added it to the
                                    # flatspec
                                    if evtType in withEvtParList:
                                        evtParList.append(parname)

            # Do something with the events that are made
            if evtList != []:
                self.addEvents(g, evtList)

            # Do something with the event par lists
            if evtParList != []:
                # add event par name mappings
                FScNM.update(FSc_update_dict)
                FScNMInv.update(FScInv_update_dict)
                if g in self._eventPars.keys():
                    self._eventPars[g].extend(evtParList)
                else:
                    self._eventPars[g] = evtParList

    def createJacP(self):
        for g in self._generators:
            if self.withJac[g]:
                gspec = self._generators[g]['modelspec']
                # haven't made generator yet so don't know which are the
                # regular RHS variables
                candidate_vars = gspec.funcSpecDict['vars']  # Quantity objects
                vars = {}
                auxvars = {}
                for v in candidate_vars:
                    vname = str(v).replace('.','_')
                    if v.specType == 'RHSfuncSpec':
                        vars[vname] = gspec.flatSpec['vars'][vname]
                    elif v.specType == 'ExpFuncSpec':
                        auxvars[vname] = gspec.flatSpec['vars'][vname]
                varnames = vars.keys()
                varnames.sort()
                # RHS specs may contain aux vars, so need to substitute their
                # definitions from flatSpec
                varspecs = {}
                for vn in varnames:
                    q = ModelSpec.QuantSpec('__temp__', vars[vn])
                    varspecs[vn] = str(q.eval(auxvars))

                # Find parameters with w.r.t which to take derivs
                candidate_pars = gspec.funcSpecDict['pars']  # Quantity objects
                parnames = []

                try:
                    evtPars = self._eventPars[g]
                except KeyError:
                    evtPars = []

                for p in candidate_pars:
                    pname_with_dot = str(p)
                    pname_no_dot = str(p).replace('.','_')
                    if pname_with_dot in evtPars or pname_no_dot in evtPars:
                        pass
                    else:
                        parnames.append(pname_no_dot)
                parnames.sort()

                jacP = Symbolic.Diff([varspecs[vn] for vn in varnames],
                           parnames).renderForCode()
                self.addFunctions(g, Symbolic.Fun(jacP, ['t'] + varnames,
                                                  'Jacobian_pars'))
    def createJac(self):
        for g in self._generators:
            if self.withJac[g]:
                gspec = self._generators[g]['modelspec']
                # haven't made generator yet so don't know which are the
                # regular RHS variables
                candidate_vars = gspec.funcSpecDict['vars']  # Quantity objects
                vars = {}
                auxvars = {}
                for v in candidate_vars:
                    vname = str(v).replace('.','_')
                    if v.specType == 'RHSfuncSpec':
                        vars[vname] = gspec.flatSpec['vars'][vname]
                    elif v.specType == 'ExpFuncSpec':
                        auxvars[vname] = gspec.flatSpec['vars'][vname]
                varnames = vars.keys()
                varnames.sort()
                # RHS specs may contain aux vars, so need to substitute their
                # definitions from flatSpec
                varspecs = {}
                for vn in varnames:
                    q = ModelSpec.QuantSpec('__temp__', vars[vn])
                    varspecs[vn] = str(q.eval(auxvars))
                jac = Symbolic.Diff([varspecs[vn] for vn in varnames],
                           varnames).renderForCode()
                self.addFunctions(g, Symbolic.Fun(jac, ['t'] + varnames,
                                                  'Jacobian'))

    def createGenerators(self):
        """Create the generators from the source specs, either in the form
        of dicts or args objects, or as a GDescriptor.

        Still some teething trouble getting expected types neat and tidy.
        """
        # 1. build constituent generators from whichever source
        # 2. combine all generators' FScompatibleNames symbol maps
        FScompatibleNames = {}
        FScompatibleNamesInv = {}
        genObjs = {}
        for gname, geninfo in self._generators.iteritems():
            if isinstance(geninfo, args):
                if isinstance(geninfo.modelspec, args):
                    # assume geninfo is traditional string definition
                    gen = self._genFromStrings(geninfo)
                else:
                    # convert ModelSpec to GDescriptor
                    gen = self._genFromMSpec(GDescriptor(**geninfo.__dict__))
            elif isinstance(geninfo, dict):
                gen = self._genFromMSpec(GDescriptor(**geninfo))
            else:
                # GDescriptor already
                gen = self._genFromMSpec(geninfo)
            if gname != gen.name:
                print gname, " vs.", gen.name
                raise ValueError("Generator name mismatch in gen descriptor")
            genObjs[gen.name] = gen
            # assume that there won't be any name clashes (there shouldn't be)
            FScompatibleNames.update(gen._FScompatibleNames.lookupDict)
            FScompatibleNamesInv.update(gen._FScompatibleNamesInv.lookupDict)
        return genObjs, genObjs.keys(), FScompatibleNames, FScompatibleNamesInv

    def _genFromStrings(self, geninfodesc):
        genStrings = geninfodesc['modelspec']
        # don't include event-related info in attrs because it's used
        # for event mappings
        attrs = [self.preReuse, self.preReuseTerms, self._funcs,
                 self.preFlat, self.parvalues, self.icvalues]
        if sometrue([len(a) > 0 for a in attrs]):
            raise ValueError("Can't mix string-based generator specs "
                             "with spec info added to ModelConstructor "
                             "object")
        gsh = Generator.theGenSpecHelper(geninfodesc['target'])
        return gsh.genClass(genStrings)

    def _genFromMSpec(self, geninfodesc):
        genSpec = geninfodesc.modelspec
        genTarg = geninfodesc.target
        genAlgPars = geninfodesc.algparams
        if self.inputs is not None:
            genInputs = self.inputs
        else:
            genInputs = {}
        genUnravelInfo = geninfodesc.unravelInfo
        genOpts = geninfodesc.options
        try:
            genEvents = self._events[genSpec.name]
        except KeyError:
            genEvents = []
        else:
            if not isinstance(genEvents, list):
                genEvents = [genEvents]
        try:
            genEventPars = self._eventPars[genSpec.name]
        except KeyError:
            genEventPars = []
        else:
            if not isinstance(genEventPars, list):
                genEventPars = [genEventPars]

        try:
            genFns = self._funcs[genSpec.name]
        except KeyError:
            genFns = None
        try:
            preReuse = self.preReuse[genSpec.name]
        except KeyError:
            self.preReuse[genSpec.name] = False
            preReuse = False
        try:
            preReuseTerms = self.preReuseTerms[genSpec.name]
        except KeyError:
            self.preReuseTerms[genSpec.name] = {}
            preReuseTerms = {}
        try:
            preFlat = self.preFlat[genSpec.name]
        except KeyError:
            self.preFlat[genSpec.name] = False
            preFlat = False

        # extract par values and ic values relevant to this generator
        genPars = {}
        for p, val in self.parvalues.iteritems():
            # don't bother to check that p is a valid param name
            # for this generator -- that will be checked by
            # GeneratorConstructor
            if p in genSpec._registry:
                genPars[p] = val
        genICs = {}
        for v, val in self.icvalues.iteritems():
            # don't bother to check that v is a valid variable name
            # for this generator -- that will be checked by
            # GeneratorConstructor
            if v in genSpec._registry:
                genICs[v] = val
        genCon = GeneratorConstructor(genSpec, checklevel=self.checklevel,
                                      userevents=genEvents,
                                      userfns=genFns,
                                      targetGen=genTarg,
                                      algparams=genAlgPars,
                                      tdata=self.tdata,
                                      indepvar=self.indepvar,
                                      parvalues=genPars,
                                      inputs=genInputs,
                                      icvalues=genICs,
                                      options=genOpts,
                                      unravelInfo=genUnravelInfo,
                                      reuseTerms=self.reuseTerms,
                                      abseps=self.abseps,
                                      activateAllBounds=self.activateAllBounds,
                                      activatedBounds=self.activatedBounds,
                                      eventPars=genEventPars,
                                      preReuse=preReuse,
                                      preReuseTerms=preReuseTerms,
                                      preFlat=preFlat)
        return genCon.getGenerator()

    def getModel(self):
        """Build and return (hybrid) model made up of declared Generators and
        the mappings between events used to change vector fields in a hybrid
        system.
        """
        # 1. create generators
        genObjs, allGenNames, FScompatibleNames, FScompatibleNamesInv \
                   = self.createGenerators()
        # 2. build event mappings
        modelInfoEntries = {}
        modelInterfaces = {}
        allModelNames = allGenNames
        # TO DO: implement global consistency conditions

        # hack to allow test trajectories for one-gen models to avoid needing
        # pre-computation in order to test a trivial condition
        test_trajs = {}
        for genname, gen in genObjs.iteritems():
            test_trajs[genname] = None
        if len(genObjs)==1:
            # singleton generator may need non-hybrid Model class unless
            # it contains discrete event state changes that map to itself
            useMI = False  # initial value
            genname = genObjs.keys()[0]
            if genname in self.eventmaps:
                for emap in self.eventmaps[genname]:
                    if emap[1] != 'terminate':
                        # then needs a hybrid model class
                        useMI = True
                        break
            if useMI and genname not in self.featureDicts:
                # then user didn't provide a feature to make a
                # condition from. need to fill a default one in
                # (simple, because there's only one model)
                self.featureDicts = {genname: {MProject.always_feature('always'): True}}
                # 1 is not None so GenInterface._ensure_test_traj()
                # will think that a test traj has already been computed
                test_trajs[genname] = 1
        else:
            useMI = True
        for hostGen, genObj in genObjs.iteritems():
            if useMI:
                m = embed(genObj,
                      tdata=genObj.indepvariable.depdomain.get())
                try:
                    DSi = MProject.intModelInterface(m,
                                MProject.condition(self.featureDicts[hostGen]),
                                test_traj=test_trajs[hostGen])
                except KeyError:
                    # no corresponding features to use
                    DSi = MProject.intModelInterface(m)
                allDSnames = allModelNames
            else:
                DSi = MProject.GeneratorInterface(genObj)
                allDSnames = allGenNames
            allGenTermEvents = genObj.eventstruct.getTermEvents()
            allGenTermEvNames = [e[0] for e in allGenTermEvents]
            if hostGen in self.eventmaps:
                genMaps = self.eventmaps[hostGen]
                genMapNames = []
                for gmix, gmtuple in enumerate(genMaps):
                    genMapNames.append(gmtuple[0])
                    if isinstance(gmtuple[1], tuple):
                        # make mapping use model name
                        genMaps[gmix] = (gmtuple[0],
                                         (gmtuple[1][0],
                                          gmtuple[1][1]))
                for evname in remain(allGenTermEvNames, genMapNames):
                    genMaps.append((evname, 'terminate'))
                modelInfoEntries[hostGen] = makeModelInfoEntry(DSi,
                                                           allDSnames,
                                                           genMaps)
            else:
                # default for a generator without an event mapping is to
                # terminate when its time is up.
                genMaps = [('time', 'terminate')]
                for evname in allGenTermEvNames:
                    genMaps.append((evname, 'terminate'))
                if not isfinite(genObj.indepvariable.depdomain[1]):
                    print "Warning: Generator %s has no termination event"%genObj.name
                    print "because it has an non-finite end computation time..."
                modelInfoEntries[hostGen] = makeModelInfoEntry(DSi,
                                                           allDSnames,
                                                           genMaps)
        modelInfoDict = makeModelInfo(modelInfoEntries.values())
        # 3. build model
        mod_args = {'name': self.name,
                    'modelInfo': modelInfoDict,
                    'mspecdict': copy.copy(self._generators),
                    'eventPars': copy.copy(self._eventPars)}
        if self.tdata is not None:
            mod_args['tdata'] = self.tdata
        if useMI:
            model = Model.HybridModel(mod_args)
        else:
            model = Model.NonHybridModel(mod_args)
        if self.forcedIntVars != []:
            model.forceIntVars(self.forcedIntVars)
        if self.icvalues != {}:
            model.set(ics=self.icvalues)
        if self.parvalues != {}:
            model.set(pars=self.parvalues)
        del genObjs
        del modelInfoEntries
        del modelInfoDict
        return model

    def addFeatures(self, hostGen, featDict):
        """Update with feature -> Bool mapping dictionaries
        for a host generator.
        """
        if hostGen not in self.featureDicts:
            self.featureDicts[hostGen] = {}
        if isinstance(featDict, dict):
            self.featureDicts[hostGen].update(featDict)
        else:
            raise TypeError("Invalid feature dictionary")

    def addEvents(self, hostGen, evTarg, eventPars=None):
        if hostGen not in self._events:
            self._events[hostGen] = []
        if hostGen not in self._eventPars:
            self._eventPars[hostGen] = []
        if isinstance(evTarg, (list, tuple)):
            self._events[hostGen].extend(evTarg)
        elif isinstance(evTarg, Events.Event):
            self._events[hostGen].append(evTarg)
        else:
            raise TypeError("Invalid event or event list")
        # Use this list to determine whether parameters are event specific
        if eventPars is not None and eventPars != [] and eventPars != '':
            if isinstance(eventPars, list):
                self._eventPars[hostGen].extend(eventPars)
            elif isinstance(eventPars, str):
                self._eventPars[hostGen].append(eventPars)
            self._generators[hostGen].addEvtPars(eventPars)


    def addFunctions(self, hostGen, fnTarg):
        if hostGen not in self._funcs:
            self._funcs[hostGen] = []
        if isinstance(fnTarg, list):
            self._funcs[hostGen].extend(fnTarg)
        elif isinstance(fnTarg, dict):
            # for compatibility with list style of _funcs for symbolic Fun
            # objects, convert the string defs to symbolic form
            for k, v in fnTarg.items():
                self._funcs[hostGen].append(Symbolic.Fun(v[1], v[0], k))
        else:
            self._funcs[hostGen].append(fnTarg)


    def setReuseTerms(self, rdict):
        self.reuseTerms = rdict
        for g in self._generators:
            self.preReuse[g] = False

    def activateBounds(self, varname=None, which_bounds='all'):
        """which_bounds argument is either 'all', 'lo', 'hi', or a pair ('lo', 'hi').
        Calling with no arguments defaults to activating all bounds."""
        if varname is None and which_bounds=='all':
            self.activateAllBounds = True
        else:
            entry = [0,0]
            if 'hi' in which_bounds:
                entry[1] = 1
            if 'lo' in which_bounds:
                entry[0] = 1
            self.activatedBounds[varname] = entry

    def setInternalVars(self, arg):
        if isinstance(arg, list):
            self.forcedIntVars = arg
        elif isinstance(arg, str):
            self.forcedIntVars = [arg]
        # !! Should check whether these are valid variable names of model

    def mapEvent(self, hostGen, eventname, target, eventmapping=None):
        """eventmapping may be a dictionary or an EvMapping product.
        You must have declared all generators before calling this function!
        """
        allGenNames = []
        for gname, geninfo in self._generators.iteritems():
            # geninfo may be an args(dict) type or a GDescriptor
            if isinstance(geninfo, GDescriptor):
                allGenNames.append(geninfo.modelspec.name)
            else:
                allGenNames.append(geninfo['modelspec'].name)
        if target not in allGenNames and target != 'terminate':
            raise ValueError("Unknown target Generator %s"%target)
        if hostGen not in self._generators:
            raise ValueError("Unknown host Generator %s"%hostGen)
        try:
            genEvs = self._events[hostGen]
        except KeyError:
            genEvs = []
        # hack to allow reference to domain bounds hi and lo events before
        # their creation
        is_domev = eventname[-6:] in ['_domlo', '_domhi'] and len(eventname) > 6
        evNames = [ev.name for ev in genEvs]
        if eventname not in evNames and eventname != 'time' and not is_domev:
            raise ValueError("Unknown event '%s' for host Generator"
                             " '%s'"%(eventname, hostGen))
        if eventmapping is None:
            evm = EvMapping()
        elif isinstance(eventmapping, dict):
            evm = EvMapping(eventmapping)
        else:
            evm = eventmapping
        if hostGen in self.eventmaps:
            self.eventmaps[hostGen].append((eventname, (target, evm)))
        else:
            self.eventmaps[hostGen] = [(eventname, (target, evm))]


# ---------------------------------------------------------------------------

## Utility functions
def embed(gen, icdict=None, name=None, tdata=None,
          make_copy=True):
    """Only use this function for building non-hybrid models with single
    Generators. Otherwise, use the ModelConstructor class.

    NB The supplied Generator is *copied* into the model unless
    optional make_copy argument is False."""
    assert isinstance(gen, Generator.Generator), ("gen argument "
                                        "must be a Generator object")
    if name is None:
        name = gen.name
    if make_copy:
        g = copy.copy(gen)
    else:
        g = gen
    modelInfoEntry = makeModelInfoEntry(MProject.GeneratorInterface(g),
                                [g.name])
    modelArgs = {'name': name,
                 'modelInfo': makeModelInfo([modelInfoEntry])}
    modelArgs['ics'] = g.get('initialconditions')
    if icdict is not None:
        # allows for partial specification of ICs here
        modelArgs['ics'].update(icdict)
    if tdata is not None:
        modelArgs['tdata'] = tdata
    elif g.tdata is not None:
        modelArgs['tdata'] = g.tdata
    return Model.NonHybridModel(modelArgs)


def makeModelInfo(arg):
    if len(arg) == 1 and isinstance(arg, dict):
        dsList = [arg]
    else:
        dsList = arg
    allDSNames = []
    returnDict = {}
    for infodict in dsList:
        assert len(infodict) == 1, \
                   "Incorrect length of info dictionary"
        dsName = infodict.keys()[0]
        if dsName not in allDSNames:
            allDSNames.append(dsName)
            returnDict.update(infodict)
        else:
            raise ValueError("clashing DS names in info "
                             "dictionaries")
        try:
            assert remain(infodict.values()[0].keys(), ['dsi',
                    'swRules', 'globalConRules', 'domainTests']) == []
        except AttributeError:
            raise TypeError("Expected dictionary in modelInfo entry")
        except AssertionError:
            raise ValueError("Invalid keys in modelInfo entry")
    return returnDict


class EvMapping(object):
    """Event mapping class, for use by makeModelInfoEntry and, when
    instantiated, the Model class.

    defStrings (list of statements) overrides assignDict if supplied at
    initialization, to permit full flexibility in the contents of the
    event mapping function.

    Use activeDict to map event names to new states."""

    def __init__(self, assignDict=None, defString="",
                 activeDict=None):
        if assignDict is None:
            assignDict = {}
        if activeDict is None:
            activeDict = {}
        self.assignDict = assignDict.copy()
        self.defString = defString
        self.activeDict = activeDict.copy()
        self.makeCallFn()


    def __cmp__(self, other):
        try:
            return alltrue([self.assignDict==other.assignDict,
                            self.defString==other.defString,
                            self.activeDict==other.activeDict])
        except AttributeError:
            return False

    def makeCallFn(self):
        indent = "  "
        fnString = """def evmapping(self, xdict, pdict, idict, estruct, t):"""
        if self.defString == "" and self.assignDict == {} and self.activeDict == {}:
            # default is the "identity mapping" (do nothing)
            fnString += indent + "pass\n"
        elif len(self.defString) >= 13 and self.defString[:13] == "def evmapping":
            # already defined, probably rebuilding after save/load object
            fnString = self.defString
        else:
            if len(self.assignDict) > 0:
                for lhs, rhs in self.assignDict.iteritems():
                    if not(type(lhs)==type(rhs)==str):
                        raise TypeError("Assignment dictionary for event "
                                        "mapping must consist of strings for "
                                        "both keys and values")
                fnString += "\n" + indent + ("\n"+indent).join(["%s = %s"%(l,r) \
                                    for l, r in self.assignDict.items()])
            if len(self.defString) > 0:
                fnString += "\n" + indent + ("\n"+indent).join(self.defString.split("\n"))
            if len(self.activeDict) > 0:
                for evname, state in self.activeDict.iteritems():
                    if not(type(evname)==str and type(state)==bool):
                        raise TypeError("Invalid types given for setting "
                                        "active events")
                fnString += "\n" + indent + \
                         ("\n"+indent).join(["estruct.setActiveFlag('%s',%s)"%(evname,str(state)) \
                                    for evname, state in self.activeDict.items()])
            self.defString = fnString
        try:
            exec fnString
        except:
            print 'Invalid function definition for event mapping:'
            print fnString
            raise
        setattr(self, 'evmapping', types.MethodType(locals()['evmapping'],
                                                      self, self.__class__))

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        try:
            del d['evmapping']
        except KeyError:
            print "'evmapping' local function not in self.__dict__"
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.makeCallFn()


def makeEvMapping(mappingDict, varnames, parnames):
    evMapDict = {}
    namemap = {}
    for varname in varnames:
        namemap[varname] = "xdict['"+varname+"']"
    for parname in parnames:
        namemap[parname] = "pdict['"+parname+"']"
    for k, v in mappingDict.iteritems():
        v_dummyQ = Symbolic.QuantSpec('dummy', v)
        v_dummyQ.mapNames(namemap)
        evMapDict["xdict['%s']"%k] = v_dummyQ()
    return EvMapping(evMapDict)

def validateTransitionName(name, special_reasons):
    if sometrue([name == r for r in special_reasons + ['time', 'terminate']]):
        raise ValueError("Name %s is reserved:\n"%name + \
            "Cannot use variable names or internal names 'time' and 'terminate'")

def makeModelInfoEntry(dsi, allModelNames=None, swmap_list=None,
                       globcon_list=None, nonevent_reasons=None):
    """Create an entry for the modelInfo attribute of a Model or Generator,
    already wrapped in a dsInterface object. Specify the list of non-event based
    reasons which can be generated by global consistency checks."""

    if allModelNames is None:
        allModelNames = []
    if swmap_list is None:
        swmap_list = []
    if globcon_list is None:
        globcon_list = []
    if nonevent_reasons is None:
        nonevent_reasons = []
    assert isinstance(allModelNames, list), \
                             "'allModelNames' argument must be a list"
    assert isinstance(swmap_list, list), \
                             "'swmap_list' argument must be a list"
    assert isinstance(globcon_list, list), \
                             "'globcon_list' argument must be a list of ModelInterfaces"
    assert isinstance(nonevent_reasons, list), \
                             "'nonevent_reasons' argument must be a list"
#    assert remain(globcon_list, allModelNames) == [], \
#               "global consistency list must consist of declared model names only"
    doms = {}
    if isinstance(dsi, MProject.GeneratorInterface):
        assert allModelNames == [dsi.model.name], \
               "Cannot use non-embedded Generators in hybrid system"
        if swmap_list != []:
            for (name, target) in swmap_list:
                if isinstance(target, str):
                    if target != 'terminate':
                        print name, target
                        raise AssertionError("Generators can only be used "
                                             "directly for non-hybrid systems")
                else:
                    # had better be a pair with first element == name
                    try:
                        assert target[0] != name
                    except (TypeError, AssertionError):
                        # type error if not subscriptable
                        print name, target
                        raise AssertionError("Generators can only be used "
                                             "directly for non-hybrid systems")
        for vname, var in dsi.model.variables.iteritems():
            if alltrue(var.depdomain.isfinite()):
                doms[vname] = Model.domain_test(vname+'_domtest',
                        pars=args(coordname=vname,
                                  derivname='D_'+vname,
                            interval=var.depdomain, verbose_level=0))
        # domain tests here for event-based tests?
        return {dsi.model.name: {'dsi': dsi, 'domainTests': doms,
                               'swRules': {}, 'globalConRules': globcon_list}}
    elif isinstance(dsi, MProject.ModelInterface):
        model = dsi.model
    else:
        raise TypeError("Invalid type for DS interface: "
                        "must be a GeneratorInterface or ModelInterface")
    # continue here only for ModelInterface
    for vname, dom in model.query('vardomains').iteritems():
        if alltrue(dom.isfinite()):
            #vname_compat = model._FScompatibleNames(vname)
            doms[vname] = Model.domain_test(vname+'_domtest',
                    pars=args(coordname=vname,
                              derivname='D_'+vname,
                        interval=dom, verbose_level=0))
    # domain tests here for event-based tests?
    special_reasons = ['time'] + model.query('variables') + nonevent_reasons
    validateTransitionName(model.name, special_reasons)
    try:
        # BUG !!! should only collect terminal events
        allEndReasonNames = model.query('events').keys() \
                            + special_reasons
    except AttributeError:
        # no events associated with the model
        allEndReasonNames = special_reasons
    if model.name not in allModelNames:
        print model.name, allModelNames
        raise ValueError('Sub-model`s name not in list of all '
                                         'available names!')
    if not alltrue([name not in allEndReasonNames for name in allModelNames]):
        print model.name, allModelNames
        raise ValueError('Sub-model names overlapped with event or '
                         'variable names')
    allTargNames = allModelNames + ['terminate']
    # if no event map function specified, assume the identity fn
    seenReasons = []
    swmap_pairs = []
    if swmap_list == []:
        raise ValueError("There must be an event mapping "
                  "specified when the model is hybrid")
    for mapentry in swmap_list:
        # check the entries of swmap_list and turn into a
        # (reason, infopair) pair, adding a default event map function
        # to some entries
        reason = mapentry[0]
        mapping_info = mapentry[1]
        if len(mapentry) > 2:
            raise ValueError("mapping entry must be (reason, info-pair) tuple")
        if isinstance(mapping_info, tuple):
            targetName = mapping_info[0]
            numargs = len(mapping_info)
        elif isinstance(mapping_info, str):
            targetName = mapentry[1]
            numargs = 1
        else:
            raise TypeError("Invalid event mapping entry")
        if numargs == 2:
            epmap = mapping_info[1]
            assert isinstance(epmap, EvMapping), "Must supply EvMapping class"
            swmap_pairs.append((reason, mapping_info))
        elif numargs == 1:
            # use default identity mapping fn for event
            # and make this entry into a three-tuple
            swmap_pairs.append((reason, (targetName, EvMapping())))
        else:
            raise ValueError("Expected 2 or 3 arguments to model "
                             "switch map entry")
        assert reason not in seenReasons, ('reason cannot appear more than'
                                           ' once in map domain')
        seenReasons.append(reason)
        if reason not in allEndReasonNames:
            print "Model %s:"%model.name
            print allEndReasonNames
            raise ValueError("name '"+reason+"' in map "
                                            "domain is missing")
        if targetName not in allTargNames:
            print "Model %s:"%model.name
            print allTargNames
            raise ValueError("name '"+targetName+"' in "
                                            "map range is missing")
    unseen_sr = remain(allEndReasonNames, seenReasons)
    if unseen_sr != []:
        # then there are 'end reasons' that do not have switch rules,
        # so give them defaults (terminate) - must use empty EvMapping
        # to match how the others will be created internally
        for r in unseen_sr:
            swmap_pairs.append((r, ('terminate', EvMapping())))
    if len(swmap_pairs) != len(allEndReasonNames):
        info(dict(swmap_pairs))
        print "(%i in total), versus:"%len(swmap_pairs)
        print allEndReasonNames, "(%i in total)"%len(allEndReasonNames)
        sw_keys = dict(swmap_pairs).keys()
        print remain(sw_keys, allEndReasonNames)
        print remain(allEndReasonNames, sw_keys)
        raise ValueError('Incorrect number of map pairs given in argument')
    return {model.name: {'dsi': dsi, 'domainTests': doms,
                'swRules': dict(swmap_pairs), 'globalConRules': globcon_list}}


def processReused(sourcenames, auxvarnames, flatspec, registry,
                  FScompatibleNames, FScompatibleNamesInv):
    """Find and process reused terms in abstract specification. To avoid
    RHS specs depending on auxiliary variables, temp variables will be declared
    in FuncSpec.py and used in both the RHS and auxiliary variables in the
    target language specification.
    """
    reuseTerms={}
    subsExpr={}
    num_reused=0
    # auxvarnames are those that sourcename definitions cannot use
    # build auxiliary token map to get rid of auxvar - auxvar inter-
    # dependencies
    u_subsMap = {}
    for auxtok in auxvarnames:
        tokobj = registry[FScompatibleNamesInv(auxtok)].obj
        addtokbraces = tokobj.spec.isCompound()
#        u_new_reusedname = "__"+auxtok+str(num_reused)
        FScompat_spec = "".join(FScompatibleNames(tokobj.spec[:]))
        u_subsMap[auxtok] = "("*addtokbraces + \
                       FScompat_spec + ")"*addtokbraces
#        u_subsMap[auxtok] = "".join(FScompatibleNames( \
#                                     tokobj.spec[:]))
    # some of these u_subsMap targets may contain auxiliary variables
    # themselves, so we must purge them now in repeated passes to u_subsMap.
    # put in a trap for infinite loop of inter-dependencies!
    loopCount = 0
    loopMaxDepth = 15
    purgeDone = {}.fromkeys(auxvarnames, False)
    while not all(purgeDone.values()) and loopCount < loopMaxDepth:
        loopCount += 1
#        print "Loop count: ", loopCount
        tempMap = {}
        for auxtok, sx in u_subsMap.iteritems():
#            print "** ", auxtok
            if purgeDone[auxtok]:
#                print "  Continue 1"
                continue
            dummyQ = Symbolic.QuantSpec('dummy', sx)
            if not any([auxname in dummyQ \
                            for auxname in auxvarnames]):
                # no auxvar names appear in the subs expr, so this is cleared
                purgeDone[auxtok] = True
#                print "  Continue 2"
                continue
            dummyQ.mapNames(u_subsMap)
            tempMap[auxtok] = dummyQ()
        # update name map with any new substitutions
#        if tempMap != {}:
#            info(tempMap)
        u_subsMap.update(tempMap)
    if not purgeDone and len(auxvarnames)>0:
        # then must have maxed out
        print "Declared auxilary variables:", auxvarnames
        raise RuntimeError("You probably have an infinite loop of auxiliary "
                       "variable inter-dependencies: recursion depth of "
                       "more than %i encountered during model build"%loopCount)
    for v in sourcenames:
        if v not in flatspec['vars']:
            # v could be a parameter, a function name, or a constant (in a
            # recursive call), so ignore
            continue
        subsMap = {}
        dummyQ = Symbolic.QuantSpec('dummy', flatspec['vars'][v])
        for u in dummyQ.usedSymbols:
            if u in auxvarnames:
                new_reusedname = "__"+u
                if new_reusedname in reuseTerms.values():
                    # simple way to avoid name clashes
                    new_reusedname += '_'+str(num_reused)
                    num_reused += 1
                spec_text = flatspec['vars'][u]
                testQ = Symbolic.QuantSpec('dummy', spec_text)
                testQ.mapNames(mathNameMap)
                # add test for unary minus otherwise no braces around
                # testQ will lead to both signs disappearing on reuseTerm
                # substitution, leaving two symbols adjoined without any
                # operator!
                addbraces = testQ.isCompound() or testQ()[0] == '-'
                # no subs expression for auxvar that points to a constant
                noSubsExpr = not addbraces and \
                      (FScompatibleNamesInv(spec_text) in registry \
                                          or isNumericToken(spec_text))
                # make substitutions for any aux vars appearing in
                # spec_text (testQ)
                testQ.mapNames(u_subsMap)
                # update addbraces after mapping
                addbraces = testQ.isCompound() or testQ()[0] == '-'
                #testQ.simplify()
                spec_text_new = "("*addbraces + testQ() + ")"*addbraces
#                spec_text_new = testQ()
                if not noSubsExpr:
                    if u in subsExpr:
                        # putting braces around auxtok in u_subsMap means
                        # that some of the expressions won't have the same
                        # bracketing as spec_text_new, so don't bother with
                        # this check
                        pass
#                        if subsExpr[u] != spec_text_new:
#                            print subsExpr[u]
#                            print spec_text_new
#                            raise RuntimeError("Different subs expr for %s in subsExpr"%u)
                    else:
                        subsExpr[u] = spec_text_new
                    if testQ()[0] == '-':
                        reuse_term = spec_text_new
                    else:
                        reuse_term = testQ()
                    if reuse_term not in reuseTerms:
                        reuseTerms[reuse_term] = new_reusedname
                if u in subsMap:
                    raise RuntimeError("%s already in subsMap!"%u)
                else:
                    subsMap[u] = spec_text_new
        # use QuantSpec's inbuilt tokenized version of exp_var definition
        # to make substitutions using the name mapping subsMap
        dummyQ.mapNames(subsMap)
        #dummyQ.simplify()
        # uses addvbraces is use addbraces above, otherwise get clash
##        addvbraces = dummyQ.isCompound()
##        subsExpr[v] = "("*addvbraces + dummyQ() + ")"*addvbraces
        dummyQ.mapNames(mathNameMap)
        subsExpr[v] = dummyQ()
    return reuseTerms, subsExpr
