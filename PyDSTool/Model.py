"""General purpose (hybrid) model class, and associated hybrid trajectory
and variable classes.

   Robert Clewley, March 2005.

A Model object's hybrid trajectory can be treated as a curve, or as
a mapping. Call the model object with the trajectory name, time(s), and
set the 'asmap' argument to be True to use an integer time to select the
trajectory segment. These are numbered from zero.

A trajectory value in a Model object's 'trajectories' dictionary
attribute is a HybridTrajectory object, having the following
attributes (among others):

    timeInterval is the entire time interval for the trajectory.

    timePartitions is a sequence of time_intervals (for each trajectory
        segment in trajseq), and

    trajSeq is a list of epoch or regime trajectory segments [traj_0, traj_1,
        ..., traj_(R-1)],

        where traj_i is a callable Trajectory or HybridTrajectory object.

    eventStruct is the event structure used to determine that trajectory.

    events is a dictionary of event names -> list of times at which that
      event took place.

    modelNames is a list of the generators used for the trajectory (one per
                                                             partition).

    variables is a dictionary that mimics the variables of the trajectory.
"""

# ----------------------------------------------------------------------------


## PyDSTool imports
from . import Generator, Events, ModelContext
from .utils import *
from .common import *
from .errors import *
from .Interval import *
from .Trajectory import *
from .Variable import *
from .Points import *
from .ModelSpec import *
from .Symbolic import isMultiRef
from .parseUtils import isHierarchicalName, NAMESEP, mapNames, symbolMapClass

## Other imports
import math, sys
from numpy import isfinite, sign, abs, array, arange, \
     zeros, concatenate, transpose, shape
from numpy import sometrue, alltrue, any, all
import numpy as np
import copy
from time import perf_counter
import pprint

__all__ = ['Model', 'HybridModel', 'NonHybridModel',
           'boundary_containment', 'boundary_containment_by_postproc',
           'boundary_containment_by_event',
           'domain_test']

# ----------------------------------------------------------------------------


class boundary_containment(ModelContext.qt_feature_leaf):
    # not implemented using metrics because the metrics are trivial
    # and cause a lot of overhead for this often-evaluated feature
    def __init__(self, name, description='', pars=None):
        ModelContext.qt_feature_leaf.__init__(self, name, description, pars)
        try:
            pars.thresh
        except AttributeError:
            raise ValueError("Missing threshold specification")
        try:
            # tolerance for small rounding errors
            pars.abseps
        except AttributeError:
            self.pars.abseps = 0
        try:
            assert pars.interior_dirn in [-1, 0, 1]
        except AttributeError:
            raise ValueError("Missing interior direction specification")
        except AssertionError:
            raise ValueError("Invalid interior direction specification value "
                " use -1 for 'below', 1 for 'above', or 0 for 'discrete domain'")
        try:
            self.pars.coordname
        except AttributeError:
            # test all coords at once
            self.pars.coordname = None
        else:
            assert isinstance(self.pars.coordname, str), \
                   "Coordinate name must be a string"

        def evaluate(self, target):
            raise NotImplementedError("Only call this method on a concrete "
                    "sub-class")


class boundary_containment_by_event(boundary_containment):
    def __init__(self, name, description='', pars=None):
        boundary_containment.__init__(self, name, description, pars)
        try:
            self.pars.bd_eventname
        except AttributeError:
            raise ValueError("Missing boundary event name")
        # assume that the supplied model will correspond to the source of
        # trajectories in evaluate method
        try:
            self.pars.model
        except AttributeError:
            raise ValueError("Missing model associated with event")
        # have not verified that event is present in model

    def evaluate(self, traj):
        # verify whether event exists and was flagged in associated model
        try:
            evpts = traj.getEvents(self.pars.bd_eventname)
        except ValueError as errinfo:
            print(errinfo)
            raise RuntimeError("Could not find flagged events for this trajectory")
        try:
            evpt = evpts[self.pars.coordname]
        except KeyError:
            raise ValueError("No such coordinate %s in the defined event"%self.pars.coordname)
        except TypeError:
            # no events of this kind were found, so passed feature eval test
            # dereferencing None (unsubscriptable object)
            if self.pars.abseps > 0:
                # would like to re-evaluate event at its threshold+abseps, but
                # leave for now
                print("Warning -- Boundary containment feature %s:"%self.name)
                print(" Check for uncertain case using events not implemented")
                self.results.output = None
                self.results.uncertain = False
            else:
                self.results.output = None
                self.results.uncertain = False
            return True
        else:
            # event found
            if self.pars.abseps > 0:
                # would like to re-evaluate event at its threshold+abseps, but
                # leave for now
                print("Warning -- Boundary containment feature %s:"%self.name)
                print(" Check for uncertain case using events not implemented")
                self.results.output = evpt[0]  # only use first event (in case not Terminal)
                self.results.uncertain = False
            else:
                self.results.output = evpt[0]  # only use first event (in case not Terminal)
                self.results.uncertain = False
            return False

    def _find_idx(self):
        """Helper function for finding index in trajectory meshpoints
        at which containment first failed."""
        if self.results.satisfied:
            # Trajectory satisfied constraint!
            return None
        return len(self.results.output)


class boundary_containment_by_postproc(boundary_containment):
    def evaluate(self, traj):
        diffs = [p - self.pars.thresh for p in \
                 traj.sample(coords=self.pars.coordname)]
        if self.pars.verbose_level > 1:
            print("%s diffs in coord %s ="%(self.name,self.pars.coordname) + ", %s" % diffs)
        res_strict = array([sign(d) \
                     == self.pars.interior_dirn for d in diffs])
        satisfied_strict = alltrue(res_strict)
        if self.pars.abseps > 0:
            if self.pars.interior_dirn == 0:
                # especially for discrete domains
                res_loose = array([abs(d) < self.pars.abseps for d in diffs])
            else:
                res_loose = array([sign(d + self.pars.interior_dirn*self.pars.abseps) \
                                   == self.pars.interior_dirn for d in diffs])
            satisfied_loose = alltrue(res_loose)
            self.results.output = res_loose
            # if p values are *outside* thresh by up to abseps amount
            # then flag this as 'uncertain' for use by domain_test class's
            # transversality testing
            self.results.uncertain = satisfied_loose and not satisfied_strict
            return satisfied_loose
        else:
            self.results.output = res_strict
            self.results.uncertain = sometrue(array(diffs)==0)
            return alltrue(res_strict)

    def _find_idx(self):
        """Helper function for finding index in trajectory meshpoints
        at which containment first failed"""
        if self.results.satisfied:
            # Trajectory satisfied constraint!
            return None
        res = self.results.output
        if res[0] == -1:
            adjusted_res = list((res + 1) != 0)
        elif res[0] == 1:
            adjusted_res = list((res - 1) != 0)
        else:
            # starts with 0 already
            adjusted_res = list(res != 0)
        # find first index at which value is non-zero
        # should never raise ValueError because this method is
        # only run if there was a sign change found
        return adjusted_res.index(True)


class domain_test(ModelContext.qt_feature_node):
    def __init__(self, name, description='', pars=None):
        ModelContext.qt_feature_node.__init__(self, name, description, pars)
        try:
            self.pars.interval
        except AttributeError:
            raise ValueError("Missing domain interval specification")
        try:
            # tolerance for small rounding errors
            self.pars.abseps
        except AttributeError:
            if isinstance(self.pars.interval, Interval):
                # Interval type passed, inherit its abseps
                self.pars.abseps = self.pars.interval._abseps
            else:
                self.pars.abseps = 0
                if not isinstance(self.pars.interval, (tuple, list)):
                    # assume a singleton numeric type passed
                    self.pars.interval = [self.pars.interval, self.pars.interval]
                elif len(self.pars.interval)==1:
                    # singleton passed, so copy the value for both
                    # "endpoints" so that xlo_bc etc. below will work
                    self.pars.interval = [self.pars.interval[0],
                                          self.pars.interval[0]]
        self.isdiscrete = self.pars.interval[0] == self.pars.interval[1]
        try:
            self.pars.coordname
        except AttributeError:
            raise ValueError("Missing coordinate name")
        try:
            self.pars.derivname
        except AttributeError:
            raise ValueError("Missing coordinate derivative name")

        # multiply interior directions by the integer value of not self.isdiscrete
        # in order to set them to zero when the "interval" is actually a singleton
        # value for a discrete domain. That fixes the boundary containment evaluation
        # code which compares the sign of coord differences with that interior
        # direction value.
        xlo_bc = boundary_containment_by_postproc('x_test_lo',
                             description='Test x lower bound',
                             pars=args(thresh=self.pars.interval[0],
                                       interior_dirn=1*int(not self.isdiscrete),
                                       abseps=self.pars.abseps,
                                       coordname=self.pars.coordname))
        xhi_bc = boundary_containment_by_postproc('x_test_hi',
                             description='Test x upper bound',
                             pars=args(thresh=self.pars.interval[1],
                                       interior_dirn=-1*int(not self.isdiscrete),
                                       abseps=self.pars.abseps,
                                       coordname=self.pars.coordname))
        dxlo_bc = boundary_containment_by_postproc('dx_test_lo',
                             description='Test dx at lower x bound',
                             pars=args(thresh=0,
                                       interior_dirn=1*int(not self.isdiscrete),
                                       abseps=0,
                                       coordname=self.pars.derivname))
        dxhi_bc = boundary_containment_by_postproc('dx_test_hi',
                             description='Test dx at upper x bound',
                             pars=args(thresh=0,
                                       interior_dirn=-1*int(not self.isdiscrete),
                                       abseps=0,
                                       coordname=self.pars.derivname))
        self.subfeatures = {'x_test_lo': xlo_bc,
                            'dx_test_lo': dxlo_bc,
                            'x_test_hi': xhi_bc,
                            'dx_test_hi': dxhi_bc}

    def evaluate(self, traj):
        # Arg is a traj!
        for sf in self.subfeatures.values():
            self.propagate_verbosity(sf)
            sf.super_pars = self.pars
            sf.super_results = self.results
        xlo_bc = self.subfeatures['x_test_lo']
        xlo_test = xlo_bc(traj)
        if xlo_bc.results.uncertain:
            if self.pars.verbose_level > 0:
                print("Lo bd uncertain")
            if self.isdiscrete:
                # accept uncertain case for discrete domain
                xlo_test = True
            else:
                # check transversality at critical (boundary) value of domain
                xlo_test = self.subfeatures['dx_test_lo'](traj)
        xhi_bc = self.subfeatures['x_test_hi']
        xhi_test = xhi_bc(traj)
        if xhi_bc.results.uncertain:
            if self.pars.verbose_level > 0:
                print("Hi bd uncertain")
            if self.isdiscrete:
                # accept uncertain case for discrete domain
                xhi_test = True
            else:
                # check transversality at critical (boundary) value of domain
                xhi_test = self.subfeatures['dx_test_hi'](traj)
        for sf in self.subfeatures.values():
            self.results[sf.name] = sf.results
        return xlo_test and xhi_test

    def _find_idx(self):
        """Helper function for finding lowest index in trajectory meshpoints
        at which domain test first failed"""
        if self.results.satisfied:
            # Trajectory satified domain conditions!
            return None
        lowest_idx = np.Inf
        for sfname, sf in self.subfeatures.items():
            if self.pars.verbose_level > 0:
                print("\n %s %r" % (sfname, sf.results))
            try:
                res = list(self.results[sfname].output)
            except AttributeError:
                # dxdt transversality test was not run so ignore
                continue
            if sf.results.satisfied:
                continue
            if self.pars.verbose_level > 0:
                print(res)
            # Find first index at which value is non-zero.
            # Will not raise ValueError because test satisfaction
            # was already checked, so must have a zero crossing
            idx = res.index(False)
            if idx < lowest_idx:
                lowest_idx = idx
        return lowest_idx


# -------------------


class Model(object):
    """
    General-purpose Hybrid and Non-Hybrid Model abstract class.

    """
    _needKeys = ['name', 'modelInfo']
    _optionalKeys = ['ics', 'mspecdict', 'verboselevel',
                    'norm', 'tdata', 'eventPars', 'abseps']
    # query keys for 'query' method
    _querykeys = ['pars', 'parameters', 'events', 'submodels',
                  'ics', 'initialconditions', 'vars', 'variables',
                  'auxvariables', 'auxvars', 'vardomains', 'pardomains',
                  'abseps']
    # valid keys for 'set' method
    _setkeys = ['pars', 'algparams', 'checklevel', 'abseps',
                'ics', 'inputs', 'tdata', 'restrictDSlist',
                'globalt0', 'verboselevel', 'inputs_t0']

    def __init__(self, legit, *a, **kw):
        # legit is a way to ensure that instances of this abstract class
        # are not created directly
        if not legit==True:
            # use explicit comparison to True otherwise kw argument will
            # eval to True, which is not what we want
            raise RuntimeError("Only use HybridModel or NonHybridModel classes")
        if len(a) > 0:
            if len(a) == 1 and isinstance(a[0], dict):
                if intersect(a[0].keys(),kw.keys()) != []:
                    raise ValueError("Cannot have initialization keys "
                        "common to both dictionary and keyword arguments")
                kw.update(a[0])
            else:
                raise ValueError("Non-keyword arguments must be a single "
                                   "dictionary")
        try:
            self.name = kw['name']
            # modelInfo is a dict mapping model names --> a dict of:
            #   'dsi': GeneratorInterface or ModelInterface object
            #   'swRules': dict of switching rules (transitions between
            #      trajectory segments)
            #   'globalConRules': list of global consistency DS names
            #   'domainTests': dictionary of variable name -> domain
            self.modelInfo = kw['modelInfo']
        except KeyError:
            raise KeyError('Necessary keys missing from argument')
        _foundKeys = len(self._needKeys)

        # by default, 'observables' are all variables that are common
        # to the Model/Generator objects in self.modelInfo, and the
        # 'internals' are those remaining generate obsvars, intvars...
        # aux vars are those aux vars common to *all* sub-models.
        # create self.obsvars, self.intvars, self.auxvars
        self.defaultVars()

        # trajectories is a dict of trajectory segments (in sequence)
        # ... can only add one at a time!
        self.trajectories = {}
        self.trajectory_defining_args = {}
        # registry of generators or models (depending on sub-class)
        # Using registry provides a shortcut for accessing a sub-model regardless
        # of whether it's a Generator or a Model class
        self.registry = {}
        for name, infodict in self.modelInfo.items():
            # set super model tag of ds object (which is either a
            # ModelInterface or Generator)
            try:
                infodict['dsi']._supermodel = self.name
            except KeyError:
                raise TypeError("Invalid modelInfo entry found with name %s"%name)
            self.registry[name] = infodict['dsi'].model

        self.diagnostics = Diagnostics()
        # set initial conditions if specified already (if not, they must
        # be specified before or during when compute() is called)
        self.icdict = {}
        if 'ics' in kw:
            self.icdict = dict(kw['ics'])
            _foundKeys += 1
        else:
            self.icdict = {}.fromkeys(self.allvars, np.NaN)
        if 'tdata' in kw:
            self.tdata = kw['tdata']
            _foundKeys += 1
        else:
            self.tdata = None
        if 'norm' in kw:
            self._normord = kw['norm']
            _foundKeys += 1
        else:
            self._normord = 2
        if 'abseps' in kw:
            # impose this absolute epsilon (small scale) on all components
            self._abseps = kw['abseps']
            _foundKeys += 1
        else:
            # All components use their defaults
            self._abseps = self.query('abseps')
        if 'verboselevel' in kw:
            if kw['verboselevel'] in [0,1,2]:
                self.verboselevel = kw['verboselevel']
            else:
                raise ValueError("Verbosity level value must be 0, 1, or 2")
            _foundKeys += 1
        else:
            self.verboselevel = 0
        if 'mspecdict' in kw:
            self._mspecdict = kw['mspecdict']
            _foundKeys += 1
        else:
            self._mspecdict = None
        if 'eventPars' in kw:
            self._eventPars = kw['eventPars']
            _foundKeys += 1
        else:
            self._eventPars = {}
        if _foundKeys < len(kw):
            raise KeyError('Invalid keys found in arguments')
        # If not already created, a True result for
        # self.haveJacobian() means that the Model will have a
        # Jacobian method made during compute(), which
        # references the appropriate _auxfn_Jac function in the Generator
        # objects.
        #
        # Use this dict to record if any external input t0 time shift
        # values are linked to a parameter value, for automatic
        # updating before trajectory computation. (Internal use)
        self._inputt0_par_links = {}


    def __len__(self):
        """Return number of sub-models"""
        return len(self.registry)

    def sub_models(self):
        """Return a list of all sub-model instances (model interfaces or generators)"""
        return list(self.registry.values())

    def _makeDefaultVarNames(self):
        """Return default observable, internal, and auxiliary variable names
        from modelInfo."""
        obsvars = []
        auxvars = []
        all_known_varnames = []
        for infodict in self.modelInfo.values():
            varnames = infodict['dsi'].query('variables')
            all_known_varnames.extend(varnames)
            auxvarnames = infodict['dsi'].query('auxvariables')
            if auxvars == []:
                # first ds to have auxvars, so just add them all
                auxvars.extend(auxvarnames)
            else:
                auxvars = intersect(auxvars, auxvarnames)
            if obsvars == []:
                # first ds, so add them all
                obsvars = varnames
            else:
                obsvars = intersect(obsvars, varnames)
        intvars = remain(all_known_varnames, obsvars)
        return (obsvars, intvars, auxvars)

    def _generateParamInfo(self):
        """Record parameter info locally, for future queries.
        Internal use only.
        """
        # use query method in case model in registry is a wrapped Generator
        # that uses _ versions of hierarchical names that are used natively
        # here.
        self.pars = {}
        for model in self.registry.values():
            try:
                self.pars.update(model.query('pars'))
            except AttributeError:
                # no pars present
                pass

    def showDef(self, target=None, type=''):
        """type = 'spec', 'auxspec', 'auxfnspec', 'events', or 'modelspec'
        (leave blank for the first *four* together).

        'spec', 'auxspec' and 'auxfnspec', 'events' refer to the compiled
        target language code for the specifications. 'modelspec' refers to
        the pre-compiled abstract specifications of the model."""
        if target is None:
            print("Use showInfo() to find names of defined sub-models")
            return
        else:
            showAll = type==''
            try:
                if type=='spec' or showAll:
                    self.registry[target].showSpec()
                if type=='auxspec' or showAll:
                    self.registry[target].showAuxSpec()
                if type=='auxfnspec' or showAll:
                    self.registry[target].showAuxFnSpec()
                if type=='events' or showAll:
                    self.registry[target].showEventSpec()
                if type=='modelspec':
                    if self._mspecdict is None:
                        raise PyDSTool_ExistError("Cannot use this function "
                            "for models not defined through ModelSpec")
                    info(self._mspecdict[target]['modelspec'].flattenSpec(\
                                [self.modelInfo[target]['dsi'].get('indepvariable').name]))
            except KeyError:
                raise ValueError("Model named %s is not known"%target)

    def showSpec(self):
        for ds in self.registry.values():
            ds.showSpec()

    def showAuxSpec(self):
        for ds in self.registry.values():
            ds.showAuxSpec()

    def showAuxFnSpec(self):
        for ds in self.registry.values():
            ds.showAuxFnSpec()

    def showEventSpec(self):
        for ds in self.registry.values():
            ds.showEventSpec()

    def current_defining_args(self):
        return args(pars=self.pars, ics=self.icdict,
                        tdata=self.tdata)

    def has_exact_traj(self, trajname, info):
        """Compare self.pars, self.icdict and self.tdata
        against what's stored for a previously computed trajectory,
        so that re-computation can be avoided.
        """
        try:
            return info == self.trajectory_defining_args[trajname]
        except KeyError:
            # not even a trajectory of this name
            return False

    def _prepareCompute(self, trajname, **kw):
        foundKeys = 0
        if 'verboselevel' in kw:
            self.set(verboselevel=kw['verboselevel'])
            foundKeys += 1
        else:
            self.set(verboselevel=0)
        if 'ics' in kw:
            self.icdict = dict(kw['ics'])
            foundKeys += 1
        if 'pars' in kw:
            self.set(pars=kw['pars'])
            foundKeys += 1
        if 'tdata' in kw:
            tdata = kw['tdata']
            foundKeys += 1
#            print "tdata in kw of %s (%s):"%(self.name, type(self)), tdata
        else:
            tdata = self.tdata
#            print "tdata from self of %s (%s):"%(self.name, type(self)), tdata
        if 'force' in kw:
            force_overwrite = kw['force']
            foundKeys += 1
        else:
            force_overwrite = False
        if len(kw) != foundKeys:
            raise PyDSTool_KeyError('Invalid argument keys passed to compute()')
        if tdata is None:
            raise ValueError("tdata must be specified")
        if len(tdata) == 1:
            assert isinstance(tdata, float) or isinstance(tdata, int), \
                   'tdata must be either a single number or a pair'
            t0_global = tdata[0]
            t1_global = np.Inf
        elif len(tdata) == 2:
            t0_global = tdata[0]
            t1_global = tdata[1]
        else:
            raise ValueError('tdata argument key may be either a single '
                               'float or a pair of floats')
        if not force_overwrite:
            assert trajname not in self.trajectories, \
                                   'Trajectory name already exists'
        assert self.modelInfo != {}, \
               'No Generator or Model objects defined for this model'
        return tdata, t0_global, t1_global, force_overwrite

    def query(self, querykey=''):
        """Return info about Model set-up.
        Valid query key: 'pars', 'parameters', 'events', 'submodels',
         'ics', 'initialconditions', 'vars', 'variables',
         'auxvars', 'auxvariables', 'vardomains', 'pardomains', 'abseps'
         """
        assert isinstance(querykey, str), \
                       ("Query argument must be a single string")
        if querykey not in self._querykeys:
            print('Valid query keys are: %r' % self._querykeys)
            print("('events' key only queries model-level events, not those")
            print(" inside sub-models)")
            if querykey != '':
                raise TypeError('Query key '+querykey+' is not valid')
        if querykey in ['pars', 'parameters']:
            result = copy.copy(self.pars)
        elif querykey in ['ics', 'initialconditions']:
            result = copy.copy(self.icdict)
        elif querykey == 'events':
            result = {}
            for dsName, model in self.registry.items():
                try:
                    result.update(model.eventstruct.events)
                except AttributeError:
                    # ds is a ModelInterface, not a Generator
                    result.update(model.query('events'))
        elif querykey == 'submodels':
            result = self.registry
        elif querykey in ['vars', 'variables']:
            result = copy.copy(self.allvars)
        elif querykey in ['vardomains', 'xdomains']:
            result = {}
            # accumulate domains from each sub-model for regular variables
            for model in self.registry.values():
                vardoms = model.query('vardomains')
                if len(result)==0:
                    result.update(vardoms)
                else:
                    for vname, vdom in result.items():
                        if vdom.issingleton:
                            # singleton
                            vdom_lo = vdom.get()
                            vdom_hi = vdom_lo
                        else:
                            # range
                            vdom_lo = vdom[0]
                            vdom_hi = vdom[1]
                        if vardoms[vname][0] < vdom[0]:
                            vdom[0] = vardoms[vname][0]
                        if vardoms[vname][1] > vdom[1]:
                            vdom[1] = vardoms[vname][1]
                        if vdom._abseps < result[vname]._abseps:
                            # have to keep abseps the tightest
                            # of any of the instances for safety
                            result[vname]._abseps = vdom._abseps
                        result[vname] = vdom
            # remaining vars are promoted aux vars
            for vname in remain(self.allvars, result.keys()):
                result[vname] = Interval(vname, float, [-np.Inf, np.Inf])
        elif querykey in ['pardomains', 'pdomains']:
            result = {}
            # accumulate domains from each sub-model for regular variables
            for model in self.registry.values():
                pardoms = model.query('pardomains')
                if len(result)==0:
                    result.update(pardoms)
                else:
                    for pname, pdom in result.items():
                        if pdom.issingleton:
                            # singleton
                            pdom_lo = pdom.get()
                            pdom_hi = pdom_lo
                        else:
                            # range
                            pdom_lo = pdom[0]
                            pdom_hi = pdom[1]
                        if pardoms[pname][0] < pdom[0]:
                            pdom[0] = pardoms[pname][0]
                        if pardoms[pname][1] > pdom[1]:
                            pdom[1] = pardoms[pname][1]
                        if pdom._abseps < result[pname]._abseps:
                            # have to keep abseps the tightest
                            # of any of the instances for safety
                            result[pname]._abseps = pdom._abseps
                        result[pname] = pdom
        elif querykey in ['auxvars', 'auxvariables']:
            result = copy.copy(self.auxvars)
        elif querykey == 'abseps':
            result = min([ds.query('abseps') for ds in self.registry.values()])
        return result

    def getEventMappings(self, dsName):
        try:
            return self.modelInfo[dsName]['swRules']
        except KeyError:
            raise NameError("Sub-model %s not found in model"%dsName)

    def setPars(self, p, val):
        # process multirefs first, then hierarchical names
        # calls itself recursively to resolve all names
        if isMultiRef(p):
            # e.g to change a group of numerically indexed parameter names
            # such as p0 - p9 or comp1.p0.g - comp1.p9.g
            # extract numeric range of pars
            # [ and ] are guaranteed to be present, from isMultiRef()
            lbrace = p.find('[')
            rbrace = p.find(']')
            if rbrace < lbrace:
                raise ValueError("Invalid multiple reference to pars")
            rootname = p[:lbrace]
            rangespec = p[lbrace+1:rbrace].split(',')
            try:
                remainder = p[rbrace+1:]
            except KeyError:
                # no more of p after this multireference
                remainder = ''
            if len(rangespec) != 2:
                raise ValueError("Invalid multiple reference to pars")
            loix = int(rangespec[0])
            hiix = int(rangespec[1])
            if loix >= hiix:
                raise ValueError("Invalid multiple reference to pars")
            # call setPars for each resolved name (these may include further
            # multi references or hierarchical names
            for ix in range(loix, hiix+1):
                self.setPars(rootname+str(ix), val)
        elif isHierarchicalName(p):
            if self._mspecdict is None:
                raise PyDSTool_ExistError("Cannot use this functionality for"
                                    " models not defined through ModelSpec")
            # find out: is root of p a valid 'type' in model spec?
            # find all occurrences of last p
            allFoundNames = []
            for mspecinfo in self._mspecdict.values():
                foundNames = searchModelSpec(mspecinfo['modelspec'], p)
                # don't add duplicates
                allFoundNames.extend(remain(foundNames,allFoundNames))
            # if allFoundNames == [] then either the hierarchical name was
            # a specific reference, or the type matching is invalid.
            # either way, we can just call set(p) and let that resolve the issue
            # (note that all multi-refs will have been dealt with by this point)
            if allFoundNames == []:
                self.set(pars={p: val})
            else:
                self.set(pars={}.fromkeys(allFoundNames, val))
        else:
            self.set(pars={p: val})


    def setICs(self, p, val):
        # process multirefs first, then hierarchical names
        # calls itself recursively to resolve all names
        if isMultiRef(p):
            # e.g to change a group of numerically indexed parameter names
            # such as p0 - p9 or comp1.p0.g - comp1.p9.g
            # extract numeric range of pars
            # [ and ] are guaranteed to be present, from isMultiRef()
            lbrace = p.find('[')
            rbrace = p.find(']')
            if rbrace < lbrace:
                raise ValueError("Invalid multiple reference to initial conditions")
            rootname = p[:lbrace]
            rangespec = p[lbrace+1:rbrace].split(',')
            try:
                remainder = p[rbrace+1:]
            except KeyError:
                # no more of p after this multireference
                remainder = ''
            if len(rangespec) != 2:
                raise ValueError("Invalid multiple reference to initial conditions")
            loix = int(rangespec[0])
            hiix = int(rangespec[1])
            if loix >= hiix:
                raise ValueError("Invalid multiple reference to initial conditions")
            # call setICs for each resolved name (these may include further
            # multi references or hierarchical names
            for ix in range(loix, hiix+1):
                self.setICs(rootname+str(ix), val)
        elif isHierarchicalName(p):
            if self._mspecdict is None:
                raise PyDSTool_ExistError("Cannot use this functionality for"
                                    " models not defined through ModelSpec")
            # find out: is root of p a valid 'type' in model spec?
            # find all occurrences of last p
            allFoundNames = []
            for mspecinfo in self._mspecdict.values():
                foundNames = searchModelSpec(mspecinfo['modelspec'], p)
                # don't add duplicates
                allFoundNames.extend(remain(foundNames,allFoundNames))
            # if allFoundNames == [] then either the hierarchical name was
            # a specific reference, or the type matching is invalid.
            # either way, we can just call set(p) and let that resolve the issue
            # (note that all multi-refs will have been dealt with by this point)
            if allFoundNames == []:
                self.set(ics={p: val})
            else:
                self.set(ics={}.fromkeys(allFoundNames, val))
        else:
            self.set(ics={p: val})


    def set(self, **kw):
        """Set specific parameters of Model. These will get passed on to
        all Generators/sub-models that support these keys unless the
        restrictDSList argument is set (only applies to keys algparams,
        checklevel, and abseps).

        Permitted keys: 'pars', 'algparams', 'checklevel', 'abseps',
                        'ics', 'inputs', 'tdata', 'restrictDSlist',
                        'globalt0', 'verboselevel', 'inputs_t0'
        """
        for key in kw:
            if key not in self._setkeys:
                raise KeyError("Not a permitted parameter argument: %s"%key + \
                                    ". Allowed keys: "+str(self._setkeys))
        if 'restrictDSlist' in kw:
            restrictDSlist = kw['restrictDSlist']
        else:
            restrictDSlist = []
        # Handle initial conditions here, because compute will pass
        # the values on to the appropriate sub-models when they are called.
        if 'ics' in kw:
            self.icdict.update(filteredDict(dict(kw['ics']), self.icdict.keys()))
        if 'tdata' in kw:
            self.tdata = kw['tdata']
        if 'abseps' in kw:
            self._abseps = kw['abseps']
        if 'inputs_t0' in kw:
            for model in self.registry.values():
                # Propagate values to sub-models.
                # If any values are strings they must refer to a parameter, so
                # here we evaluate them. We keep a record of these for later
                # automated updating.
                t0val_dict = kw['inputs_t0']
                for inp, val in t0val_dict.items():
                    if isinstance(val, str):
                        try:
                            new_val = self.pars[val]
                        except KeyError:
                            raise ValueError("Input t0 to parameter link "
                                             "invalid: no such parameter "+val)
                        t0val_dict[inp] = new_val
                        self._inputt0_par_links.update({inp: val})
                    elif isinstance(val, _num_types):
                        if inp in self._inputt0_par_links:
                            # no longer parameter-linked
                            del self._inputt0_par_links[inp]
                    else:
                        raise TypeError("Invalid type for input t0 value")
                try:
                    model.set(inputs_t0=t0val_dict)
                except AssertionError:
                    # generator doesn't involve inputs
                    pass
        if 'verboselevel' in kw:
            if kw['verboselevel'] in [0,1,2]:
                self.verboselevel = kw['verboselevel']
            else:
                raise ValueError("Verbosity level value must be 0, 1, or 2")
            for model in self.registry.values():
                # propagate to sub-models
                try:
                    model.set(verboselevel=self.verboselevel)
                except KeyError:
                    # generator doesn't support verboselevel
                    pass
        if restrictDSlist == []:
            restrictDSlist = list(self.registry.keys())
        # For the remaining keys, must propagate parameter changes to all
        # sub-models throughout modelInfo structure.
        #
        # Changed by WES 10FEB06 to handle problem of 'new' pars being
        # added if the parameter names do not exist in any generators
        dsis = list(self.modelInfo.values())
        numDSs = len(dsis)
        # loop over keywords
        for key, value in filteredDict(kw, ['ics', 'tdata',
                            'inputs_t0', 'restrictDSlist', 'globalt0'],
                                neg=True).items():
            # keep track of the number of errors on this keyword
            if isinstance(value, dict):
                # keep track of entry errors for this key
                entry_err_attr = {}
                entry_err_val = {}
                for entrykey, entryval in value.items():
                    entry_err_attr[entrykey] = 0
                    entry_err_val[entrykey] = 0
            else:
                entry_err_attr = 0
                entry_err_val = 0
            # try setting pars in each sub-model
            for infodict in dsis:
                callparsDict = {}
                # select out the ones relevant to this sub-model
                ds = infodict['dsi'].model
                if hasattr(ds, '_validKeys'):
                    options = ds._validKeys
                elif hasattr(ds, '_setkeys'):
                    options = ds._setkeys
                if key in options:
                    if key in ['algparams', 'checklevel', 'abseps'] and \
                        ds.name not in restrictDSlist:
                            # only apply these keys to the restricted list
                            continue
                    if isinstance(value, dict):
                        for entrykey, entryval in value.items():
                            try:
                                ds.set(**{key:{entrykey:entryval}})
                            except PyDSTool_AttributeError:
                                entry_err_attr[entrykey] += 1
                            except PyDSTool_ValueError:
                                entry_err_val[entrykey] += 1
                            except AssertionError:
                                # key not valid for this type of ds
                                pass
                    else:
                        try:
                            ds.set(**{key: value})
                        except PyDSTool_AttributeError:
                            entry_err_attr += 1
                        except PyDSTool_ValueError:
                            entry_err_val += 1
                        except AssertionError:
                            # key not valid for this type of ds
                            pass
            # Check that none of the entries in the dictionary caused errors
            # in each sub-model
            if isinstance(value, dict):
                for entrykey, entryval in value.items():
                    if entry_err_attr[entrykey] == numDSs:
                        raise PyDSTool_AttributeError('Parameter does not' +\
                              ' exist in any sub-model: %s = %f'%(entrykey,
                                                                  entryval))
                    if entry_err_val[entrykey] == numDSs:
                        raise PyDSTool_ValueError('Parameter value error in' +\
                              ' every sub-model: %s = %f'%(entrykey, entryval))
                    else:
                        # can't think of other ways for this error to crop up
                        pass
            else:
                if entry_err_attr == numDSs:
                    raise PyDSTool_AttributeError('Parameter does not exist' +\
                             ' in any sub-model: %s'%key)
                if entry_err_val == numDSs:
                    raise PyDSTool_ValueError('Parameter value error in' +\
                              ' every sub-model: %s'%key)
            del(entry_err_attr)
            del(entry_err_val)
        self._generateParamInfo()

    def __getitem__(self, trajname):
        try:
            return self.trajectories[trajname]
        except KeyError:
            raise ValueError('No such trajectory.')

    def __delitem__(self, trajname):
        self._delTraj(trajname)

    def _delTraj(self, trajname):
        """Delete a named trajectory from the database."""
        try:
            traj = self.trajectories[trajname]
        except KeyError:
            # a trajectory piece may have been created without
            # the top-level trajectory ever being completed
            # (e.g. after an unxpected error or ^C interruption)
            ##raise ValueError('No such trajectory.')
            l = len(trajname)
            for m in self.registry.values():
                # delete all matching pieces (of form trajname + '_' + <digits>)
                for n in m.trajectories.keys():
                    if n[:l] == trajname and n[l] == '_' and n[l+1:].isdigit():
                        m._delTraj(trajname)
        else:
            # propagate deletions down through registry
            if not isinstance(traj.modelNames, str):
                for i, model_name_i in enumerate(traj.modelNames):
                    del(self.registry[model_name_i][trajname+'_%i'%i])
            del(self.trajectories[trajname])


    def __call__(self, trajname, t, coords=None, asGlobalTime=True,
                 asmap=False):
        """Evaluate position of hybrid trajectory at time t.
        if optional argument asmap == True then t must be an integer in
        [0, #segments].
        """
        try:
            traj = self.trajectories[trajname]
        except KeyError:
            raise ValueError("trajectory '"+trajname+"' unknown")
        else:
            return traj(t, coords=coords, asGlobalTime=asGlobalTime,
                        asmap=asmap)


    def getEndPoint(self, trajname, end=1):
        """Returns endpoint of specified trajectory as Point.
        trajname: name of selected trajectory
        end: (default=1) index of trajectory endpoint.
        0 => first, 1 => last
        """
        xdict = {}
        if end not in [0,1]:
            raise ValueError("end must be 0, 1")
        endtraj = self.trajectories[trajname].trajSeq[-end]
        for xname in endtraj.coordnames:
            try:
                xdict[endtraj._FScompatibleNamesInv(xname)] = \
                     endtraj.variables[xname].output.datapoints[1][-end]
            except KeyError:
                # auxiliary var didn't need calling
                pass
            except AttributeError:
                # non-point based output attributes of a Variable need
                # to be called ...
                tend = endtraj.indepdomain[end]
                xdict[endtraj._FScompatibleNamesInv(xname)] = \
                     endtraj.variables[xname](tend)
            except PyDSTool_BoundsError:
                print("Value out of bounds in variable call:")
                print("  variable '%s' was called at time %f"%(xname, tend))
                raise
        return Point({'coorddict': xdict,
              'coordnames': endtraj._FScompatibleNamesInv(endtraj.coordnames),
              'coordtype': float,
              'norm': self._normord})

    def getEndTime(self, trajname, end=1):
        """Returns end time of specified trajectory.
        trajname: name of selected trajectory
        end: (default=1) index of trajectory endpoint.
        0 => first, 1 => last
        """
        if end not in [0,1]:
            raise ValueError("end must be 0, 1")
        endtraj = self.trajectories[trajname].trajSeq[-end]
        tend = endtraj.indepdomain[end]
        return tend

    def _validateVarNames(self, names):
        """Check types and uniqueness of variable names."""
        namelist = []  # records those seen so far
        for vname in names:
            assert isinstance(vname, str), \
                   'variable name must be a string'
            assert vname not in namelist, ('variable names must be unique for'
                                           ' model')
            namelist.append(vname)


    def forceObsVars(self, varnames):
        """Force variables to be the observables in the Model.
        May also promote auxiliary variables."""
        r = remain(varnames, self.obsvars+self.intvars+self.auxvars)
        if len(r) > 0:
            # then there are names given that are not known as
            # obs, int, or aux
            raise ValueError("Unknown variable names: "+str(r))
        for v in remain(varnames, self.obsvars):
            # only include names that are not already observables
            self.obsvars.append(v)
        # remove any vars that are now observables
        self.intvars = remain(self.intvars, varnames)
        self.auxvars = remain(self.auxvars, varnames)
        self.obsvars.sort()
        self.intvars.sort()
        self.auxvars.sort()
        self.allvars = self.obsvars + self.intvars
        self.allvars.sort()
        self.dimension = len(self.allvars)


    def forceIntVars(self, varnames):
        """Force variables to become internal variables in the Model.
        May also promote auxiliary variables."""
        r = remain(varnames, self.obsvars+self.intvars+self.auxvars)
        if len(r) > 0:
            # then there are names given that are not known as
            # obs, int, or aux
            raise ValueError("Unknown variable names: "+str(r))
        for v in remain(varnames, self.intvars):
            # only include names that are not already internals
            self.intvars.append(v)
        # remove any vars that are now internals
        self.obsvars = remain(self.obsvars, varnames)
        self.auxvars = remain(self.auxvars, varnames)
        self.obsvars.sort()
        self.intvars.sort()
        self.auxvars.sort()
        self.allvars = self.obsvars + self.intvars
        self.allvars.sort()
        self.dimension = len(self.allvars)


    def defaultVars(self):
        """(Re)set to default observable and internal variable names."""
        obsvars, intvars, auxvars = self._makeDefaultVarNames()
        self._validateVarNames(obsvars + intvars + auxvars)
        # OK to store these permanently after varname validation
        self.obsvars = obsvars
        self.intvars = intvars
        self.auxvars = auxvars
        self.obsvars.sort()
        self.intvars.sort()
        self.auxvars.sort()
        self.allvars = self.obsvars + self.intvars
        self.allvars.sort()
        self.dimension = len(self.allvars)


    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))


    def __repr__(self):
        return self._infostr(verbose=0)

    __str__ = __repr__


    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def renameTraj(self, trajname, newname, force=False):
        """Rename stored trajectory. Force option (default False)
        will overwrite any existing trajectory with the new name.
        """
        try:
            traj = self.trajectories[trajname]
        except KeyError:
            raise ValueError('No such trajectory name %s'%trajname)
        if trajname != newname:
            if newname not in self.trajectories or force:
                self.trajectories[newname] = traj
                del self.trajectories[trajname]
                traj.name = newname
            else:
                raise ValueError("Name %s already exists"%newname)

    def getTrajModelName(self, trajname, t=None):
        """Return the named trajectory's associated sub-model(s) used
        to create it, specific to time t if given."""
        try:
            modelNames = self.trajectories[trajname].modelNames
        except KeyError:
            raise ValueError('No such trajectory name %s'%trajname)
        if t is None:
            # return list of Generators for associated hybrid trajectory
            return modelNames
        else:
            parts = self.getTrajTimePartitions(trajname)
            pix = 0
            for pinterval in parts:
                if pinterval.contains(t) is not notcontained:
                    return modelNames[pix]
                else:
                    pix += 1

    def getTrajEventTimes(self, trajname, events=None):
        """Return the named trajectory's Generator-flagged event times.

        events argument can be singleton string name of an event,
        returning the event data, or events can be a list of event names,
        returning a dictionary of event name -> event data.
        Event names should use hierarchical naming convention, if
        applicable."""
        try:
            return self.trajectories[trajname].getEventTimes(events)
        except KeyError:
            raise
            #raise ValueError('No such trajectory name')

    def getTrajEvents(self, trajname, events=None):
        """Return the named trajectory's Generator-flagged events.

        events argument can be singleton string name of an event,
        returning the event data, or events can be a list of event names,
        returning a dictionary of event name -> event data.
        Event names should use hierarchical naming convention, if
        applicable."""
        try:
            return self.trajectories[trajname].getEvents(events)
        except KeyError:
            raise ValueError('No such trajectory name')

    def getTrajEventStruct(self, trajname):
        """Return the named trajectory's model event structure (representing
        external constraints), as present when it was computed.
        """
        try:
            return self.trajectories[trajname].modelEventStructs
        except KeyError:
            raise ValueError('No such trajectory name')

    def getTrajTimeInterval(self, trajname):
        """Return the named trajectory's time domain,
        over which it is defined, in a single interval."""
        try:
            return self.trajectories[trajname].indepdomain
        except KeyError:
            raise ValueError('No such trajectory name')

    def getTrajTimePartitions(self, trajname):
        """Return the named trajectory's time domain,
        over which it is defined, as a list of time interval partitions."""
        try:
            return self.trajectories[trajname].timePartitions
        except KeyError:
            raise ValueError('No such trajectory name')

    def getDSAlgPars(self, target, par, idx=None):
        """
        Returns value of given algorithmic parameter for selected sub-model.
        target -- name of sub-model in model (cannot be list).
        par -- name of algorithmic parameter.
        idx -- (optional) index into value if algorithmic parameter val is a
          list of values.
        """
        if target in self.registry.keys():
            algpars = self.registry[target].get('algparams')
            if par in algpars.keys():
                if isinstance(algpars[par], list):
                    if idx is not None:
                        if isinstance(idx, list):
                            val = [algpars[par][x] for x in idx]
                        else:
                            val = algpars[par][idx]
                    else:
                        val = algpars[par]
                else:
                    val = algpars[par]
            else:
                val = None
        else:
            raise ValueError("Target sub-model name not found")
        return val

    def setDSAlgPars(self, target, par, val):
        """
        Set value of algorithmic parameter in a specific generator.
        target -- name or list of generators in model.
        par -- name of algorithmic parameter is to be set.
        val -- value to which the algorithmic parameter is to be set.

        if target is a list, then algorithmic pararameter 'par' is
        set to 'val' for every generator in the list, if par exists for that
        generator.

        WARNING: THIS FUNCTION IS NOT 'SAFE' -- IT DOES NOT CHECK THAT VALS
          ARE APPROPRIATE TO PARAMETERS!!!
        """

        if isinstance(target, list):
            subModelList = target
        else:
            subModelList = [target]
        for dsName in subModelList:
            algpars = self.registry[dsName].get('algparams')
            if par in algpars.keys():
                # If target is a list, make sure that the input list is of the
                # appropriate length, otherwise warn and skip
                if isinstance(algpars[par], list):
                    if isinstance(val, list):
                        if len(algpars[par]) != len(val):
                            print("Warning: par %s list len (%d) in generator %s doesn't match val list len (%d). Skipping."%(par, len(algpars[par]) + "%s, %d" % (dsName, len(val))))
                            continue
                        else:
                            algpars[par] = val
                    else:
                        # Set every member of the list to that value
                        for x in range(len(algpars[par])):
                            algpars[par][x] = val
                else:
                    if isinstance(val, list):
                        print("Warning: par %s type (%s) in generator %s doesn't match val type (%s). Skipping."%(par, type(algpars[par]) + "%s %d" % (dsName, type(val))))
                    else:
                        algpars[par] = val
            else:
                pass


    def sample(self, trajname, coords=None, dt=None,
                   tlo=None, thi=None, doEvents=True, precise=False):
        """Uniformly sample the named trajectory over range indicated,
        including any event points. (e.g. use this call for plotting purposes.)
        Outputs a Pointset from the trajectory over a given range.

        Arguments:

          trajname   Name of stored trajectory to sample
          coords     (optional) list of variable names to include in output
          dt         (optional) step size to use in sampling of independent variable
                     If not given, the underlying time mesh is used, if available.
          tlo        (optional) Start value for independent variable, default 0
          thi        (optional) End value for independent variable, default last value
          doEvents   (optional) include any event points in output, default True
          precise    (optional) The default value, False, causes an attempt to use
                     the underlying mesh of the trajectory to return a Pointset more
                     quickly. Currently, this can only be used for trajectories that
                     have a single segment (non-hybrid).
        """
        try:
            return self.trajectories[trajname].sample(coords, dt, tlo, thi,
                                                  doEvents, precise)
        except KeyError:
            raise


    def _infostr(self, dsName=None, verbosity=0):
        """Return string information about named sub-model (if given by dsName)
        at given verbosity level (default 0)"""
        pp = pprint.PrettyPrinter(indent=3)
        if dsName is None:
            result = {}
            res_str = "Sub-models defined in model %s:\n" % self.name
            for name, infodict in self.modelInfo.items():
                result[name] = infodict['dsi'].model._infostr(verbosity)
            return res_str + pp.pprint(result)
        else:
            # return more information for a single sub-model
            try:
                result = {dsName: self.modelInfo[dsName]['dsi'].model._infostr(verbosity)+"\n"+\
                      "Event mapping info:\n" + str(self.modelInfo[dsName]['swRules'])}
            except KeyError:
                raise NameError("Sub-model %s not found in model"%dsName)
            return "Sub-model %s:\n" % (dsName) + pp.pprint(result)

    def showDSEventInfo(self, target, verbosity=1, ics=None, t=0):
        # call to eventstruct prints info to stdout
        if ics is None:
            ics = self.icdict
        estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
        estruct(verbosity)

    def getDSEvent(self, target, evname, ics=None, t=0):
        """
        Return Event object from target (name of generator/sub-model) with name
        evname
        """
        if ics is None:
            ics = self.icdict
        estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
        # getAllEvents returns a list of pairs (name, Event)
        all_ev_dict = dict(estruct.getAllEvents())
        try:
            return all_ev_dict[evname]
        except KeyError:
            raise ValueError("No such event found in target sub-model")

    def getDSEventTerm(self, target, flagVal=True, ics=None, t=0):
        """
        List of events in target which are terminal/non-terminal according to
        value of flagVal.
        target -- name of generator/sub-model in model (cannot be list)
        flagVal -- True (terminal) or False (non-terminal)
        """
        if ics is None:
            ics = self.icdict
        estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
        if isinstance(target, str):
            if flagVal:
                return [x[0] for x in estruct.getTermEvents()]
            else:
                return [x[0] for x in estruct.getNonTermEvents()]
        else:
            return []

    def setDSEventTerm(self, target, eventTarget, flagVal, ics=None, t=0):
        """
        Set event in a specific generator to be (non)terminal.
        target -- name or list of names of generators/sub-models in model.
        eventTarget -- name or list of names of events in specified generator(s)
        flagVal -- True (event terminal) or False (event non-terminal)

        flagVal is applied to every listed event in every listed generator, if
          events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setTermFlag(eventTarget, flagVal)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setTermFlag(eventTarget, flagVal)

    def getDSEventActive(self, target, flagVal=True, ics=None, t=0):
        """
        List of events in target which are active/inactive according to
        value of flagVal.
        target -- name of generator/sub-model in model (cannot be list)
        flagVal -- True (active) or False (inactive)
        """
        if ics is None:
            ics = self.icdict
        estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
        if isinstance(target, str):
            if flagVal:
                return [x[0] for x in estruct.getActiveEvents()]
            else:
                return [x[0] for x in estruct.getNonActiveEvents()]
        else:
            return []

    def setDSEventActive(self, target, eventTarget, flagVal, ics=None, t=0):
        """
        Set event in a specific generator to be (in)active.
        target -- name or list of names of generators/sub-models in model.
        eventTarget -- name or list of names of events in specified generator(s)
        flagVal -- True (event active) or False (event inactive)

        flagVal is applied to every listed event in every listed generator, if
        events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setActiveFlag(eventTarget, flagVal)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setActiveFlag(eventTarget, flagVal)

    def setDSEventICs(self, target, eventTarget, val, ics=None, t=0):
        """
        Set event in a specific generator to have specified initial conditions.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        val -- dictionary of varnames and initial condition values

        val is applied to every listed event in every listed generator, if
        events and generators exist.
        If a varname in the val dict does not exist in a specified
        generator/event, then it is skipped.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setICs(eventTarget, val)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setICs(eventTarget, val)

    def setDSEventPrecise(self, target, eventTarget, flagVal, ics=None, t=0):
        """
        Set event in a specific generator to be (im)precise.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        flagVal -- True (event precise) or False (event imprecise)

        flagVal is applied to every listed event in every listed generator, if events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setPreciseFlag(eventTarget, flagVal)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setPreciseFlag(eventTarget, flagVal)

    def setDSEventDelay(self, target, eventTarget, val, ics=None, t=0):
        """
        Set event in a specific generator to have specified event delay.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        val -- event delay (int, float >= 0)

        val is applied to every listed event in every listed generator, if events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setEventDelay(eventTarget, val)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setEventDelay(eventTarget, val)

    def setDSEventInterval(self, target, eventTarget, val, ics=None, t=0):
        """
        Set event in a specific generator to have specified event interval.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        val -- event interval (int, float >= 0)

        val is applied to every listed event in every listed generator, if events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setEventInterval(eventTarget, val)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setEventInterval(eventTarget, val)

    def setDSEventTol(self, target, eventTarget, val, ics=None, t=0):
        """
        Set event in a specific generator to have specified event tolerance.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        val -- event tolerance (int, float >= 0)

        val is applied to every listed event in every listed generator, if events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setEventTol(eventTarget, val)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setEventTol(eventTarget, val)

    def setDSEventDir(self, target, eventTarget, val, ics=None, t=0):
        """
        Set event in a specific generator to have specified direction code.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        val -- direction code (-1: decreasing, 1: increasing, 0: either direction)

        val is applied to every listed event in every listed generator, if events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setEventDir(eventTarget, val)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setEventDir(eventTarget, val)

    def setDSEventStartTime(self, target, eventTarget, val, ics=None, t=0):
        """
        Set event in a specific generator to have specified start time.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        val -- start time (float, int)

        val is applied to every listed event in every listed generator, if events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setStartTime(eventTarget, val)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setStartTime(eventTarget, val)

    def setDSEventBisect(self, target, eventTarget, val, ics=None, t=0):
        """
        Set event in a specific generator to have specified bisect limit.
        target -- name or list of names of generators in model.
        eventTarget -- name or list of names of events in specified generator(s)
        val -- bisect limit (int > 0)

        val is applied to every listed event in every listed generator, if events and generators exist.
        """
        if ics is None:
            ics = self.icdict
        if isinstance(target, list):
            for targName in target:
                estruct = self.modelInfo[targName]['dsi'].get('eventstruct', ics, t)
                estruct.setBisect(eventTarget, val)
        else:
            estruct = self.modelInfo[target]['dsi'].get('eventstruct', ics, t)
            estruct.setBisect(eventTarget, val)

    def resetEventTimes(self):
        """Internal method"""
        for ds in self.registry.values():
            ds.resetEventTimes()

    def _set_for_hybrid_DS(self, state):
        """Internal method to set all sub-models flag for being part of a hybrid
        trajectory computation.
        Useful to indicate that high level events should not be reset when a
        Generator is reused between hybrid trajectory segments."""
        for ds in self.registry.values():
            ds._set_for_hybrid_DS(state)

    def searchForNames(self, template):
        """Find parameter / variables names matching template in
        the model's generators or sub-models: the returned result is a list."""
        if self._mspecdict is None:
            raise PyDSTool_ExistError("Cannot use this function for models"
                                      " not defined through ModelSpec")
        result = {}
        for modelName, mspecinfo in self._mspecdict.items():
            # HACK to force compatibility of mspecinfo being a dict/args vs. a GDescriptor
            if isinstance(mspecinfo, dict):
                foundNames = searchModelSpec(mspecinfo['modelspec'], template)
            else:
                # args or GDescriptor
                foundNames = searchModelSpec(mspecinfo.modelspec, template)
            result[modelName] = foundNames
        return result


    def searchForVars(self, template):
        """Find variable and auxiliary variable names that have to be in every
        generator or sub-model: the returned result is a list."""
        if self._mspecdict is None:
            raise PyDSTool_ExistError("Cannot use this function for models"
                                      " not defined through ModelSpec")
        # all sub-models must define the same variables, so need only
        # to look at one
        a_ds_name = list(self._mspecdict.keys())[0]
        a_ds_mspec = self._mspecdict[a_ds_name]['modelspec']
        foundNames = searchModelSpec(a_ds_mspec, template)
        try:
            fspec = self.registry[a_ds_name].get('funcspec')
        except AttributeError:
            # for non-hybrid models
            fspec = list(self.registry[a_ds_name].registry.values())[0].get('funcspec')
        # funcspec won't have its internal names converted to
        # hierarchical form, so do it here on the attributes we
        # need
        return intersect(self.registry[a_ds_name]._FScompatibleNamesInv(fspec.vars + fspec.auxvars),
                             foundNames)




class NonHybridModel(Model):
    def __init__(self, *a, **kw):
        Model.__init__(self, True, *a, **kw)
        # collect parameter info from all modelInfo objects
        self._generateParamInfo()
        self._validateRegistry(self.obsvars, self.intvars)

    def _findTrajInitiator(self, end_reasons, partition_num, t0, xdict,
                              gi=None, swRules=None):
        # return GeneratorInterface object and include return
        # elements for compatibility with HybridModel._findTrajInitiator:
        # swRules, globalConRules, nextModelName, reused (True b/c always same),
        #       epochStateMaps, notDone
        infodict = list(self.modelInfo.values())[0]
        return infodict['dsi'], infodict['swRules'], \
               infodict['globalConRules'], infodict['dsi'].model.name, \
               True, None, True

    def cleanupMemory(self):
        """Clean up memory usage from past runs of a solver that is interfaced through
        a dynamic link library. This will prevent the 'continue' integration option from
        being accessible and will delete other data about the last integration run."""
        list(self.registry.values())[0].cleanupMemory()

    def haveJacobian(self):
        """Returns True iff all objects in modelInfo have
        defined Jacobians."""
        return list(self.registry.values())[0].haveJacobian()

    def haveJacobian_pars(self):
        """Returns True iff all objects in modelInfo have
        defined Jacobians."""
        return list(self.registry.values())[0].haveJacobian_pars()

    def Rhs(self, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's Rhs function.
        Parameters:

          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        # get Generator as 'ds'
        ds = list(self.registry.values())[0]
        # filer to ensure only variable names get transformed
        fscm_dict = filteredDict(ds._FScompatibleNames.lookupDict, xdict.keys())
        fscm = symbolMapClass(fscm_dict)
        fscmInv = fscm.inverse()
        vars = ds.get('funcspec').vars   # FS compatible
        x_fs = filteredDict(fscm(xdict), vars)
        # in case ds i.c.'s are different or not set yet (NaN's)
        # for purposes of auxiliary function 'initcond' calls during Rhs
        old_ics = ds.initialconditions.copy()
        ds.initialconditions.update(x_fs)
        x_fs.update(ds.initialconditions)
        if pdict is None:
            pdict = self.pars
        if asarray:
            # some may not be in the same order, so must inefficiently ensure
            # re-ordered according to alphabetical order of non-FS compatible names
            rhsval = array(ds.Rhs(t, x_fs, pdict))[fscm.reorder()]
        else:
            rhsval = Point({'coorddict': dict(zip(fscmInv(vars),
                                              ds.Rhs(t, x_fs, pdict))),
                      'coordtype': float,
                      'norm': self._normord})
        ds.initialconditions.update(old_ics)
        return rhsval

    def Jacobian(self, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's Jacobian function (if defined).

        Arguments:

          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        ds = list(self.registry.values())[0]
        if not ds.haveJacobian():
            raise PyDSTool_ExistError("Jacobian not defined")
        fscm = ds._FScompatibleNames
        fscmInv = ds._FScompatibleNamesInv
        vars = ds.get('funcspec').vars   # FS compatible
        x_fs = filteredDict(fscm(xdict), vars)
        # in case ds i.c.'s are different or not set yet (NaN's)
        # for purposes of auxiliary function 'initcond' calls during Rhs
        old_ics = ds.initialconditions.copy()
        ds.initialconditions.update(x_fs)
        x_fs.update(ds.initialconditions)
        if pdict is None:
            pdict = self.pars
        if asarray:
            J = array(ds.Jacobian(t, x_fs, pdict))[fscm.reorder()]
        else:
            J = Pointset({'coorddict': dict(zip(fscmInv(vars),
                                         ds.Jacobian(t, x_fs, pdict))),
                         'coordtype': float,
                         'norm': self._normord})
        ds.initialconditions.update(old_ics)
        return J

    def JacobianP(self, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's JacobianP function (if defined).

        Arguments:

          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        ds = list(self.registry.values())[0]
        if not ds.haveJacobian_pars():
            raise PyDSTool_ExistError("Jacobian w.r.t. pars not defined")
        fscm = ds._FScompatibleNames
        fscmInv = ds._FScompatibleNamesInv
        x_fs = filteredDict(fscm(xdict), ds.get('funcspec').vars)
        # in case ds i.c.'s are different or not set yet (NaN's)
        # for purposes of auxiliary function 'initcond' calls during Rhs
        old_ics = ds.initialconditions.copy()
        ds.initialconditions.update(x_fs)
        x_fs.update(ds.initialconditions)
        if pdict is None:
            pdict = self.pars
        if asarray:
            Jp = array(ds.JacobianP(t, x_fs, pdict))[fscm.reorder()]
        else:
            Jp = Pointset({'coorddict': dict(zip(fscmInv(ds.get('funcspec').pars),
                                          ds.JacobianP(t, x_fs, pdict))),
                         'coordtype': float,
                         'norm': self._normord})
        ds.initialconditions.update(old_ics)
        return Jp

    def MassMatrix(self, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's MassMatrix function (if defined).

        Arguments:

          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        ds = list(self.registry.values())[0]
        if not ds.haveMass():
            raise PyDSTool_ExistError("Mass matrix not defined")
        fscm = ds._FScompatibleNames
        fscmInv = ds._FScompatibleNamesInv
        vars = ds.get('funcspec').vars   # FS compatible
        x_fs = filteredDict(fscm(xdict), vars)
        # in case ds i.c.'s are different or not set yet (NaN's)
        # for purposes of auxiliary function 'initcond' calls during Rhs
        old_ics = ds.initialconditions.copy()
        ds.initialconditions.update(x_fs)
        x_fs.update(ds.initialconditions)
        if pdict is None:
            pdict = self.pars
        if asarray:
            M = array(ds.MassMatrix(t, x_fs, pdict))[fscm.reorder()]
        else:
            M = Point({'coorddict': dict(zip(fscmInv(vars),
                                        ds.MassMatrix(t, x_fs, pdict))),
                      'coordtype': float,
                      'norm': self._normord})
        ds.initialconditions.update(old_ics)
        return M

    def AuxVars(self, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's auxiliary variables
        definition (if defined).

        Arguments:

          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        ds = list(self.registry.values())[0]
        fscm = ds._FScompatibleNames
        fscmInv = ds._FScompatibleNamesInv
        x_fs = filteredDict(fscm(xdict), ds.get('funcspec').vars)
        # in case ds i.c.'s are different or not set yet (NaN's)
        # for purposes of auxiliary function 'initcond' calls during Rhs
        old_ics = ds.initialconditions.copy()
        ds.initialconditions.update(x_fs)
        x_fs.update(ds.initialconditions)
        if pdict is None:
            pdict = self.pars
        if asarray:
            A = array(ds.AuxVars(t, x_fs, pdict))[fscm.reorder()]
        else:
            A = Point({'coorddict': dict(zip(fscmInv(ds.get('funcspec').auxvars),
                                ds.AuxVars(t, x_fs, pdict))),
                      'coordtype': float,
                      'norm': self._normord})
        ds.initialconditions.update(old_ics)
        return A


    def compute(self, trajname, **kw):
        """Compute a non-hybrid trajectory. Returns a Trajectory object.

        Arguments (non-keyword):
          trajname   Name of trajectory to create (string)

        Arguments (keyword only, all optional):
          force      (Bool, default False) - force overwrite of any trajectory
                     stored in this object with the same name
          verboselevel  (int, default 0)
          ics        initial conditions dict or Point
          pars       parameters dict or Point
          tdata      time data (interval as sequence of 2 numeric values)
        """
        tdata, t0_global, t1_global, force_overwrite = \
                Model._prepareCompute(self, trajname, **kw)
        if self.icdict == {}:
            # just get i.c. from the single generator,
            # making sure it's non-empty
            self.icdict = list(self.registry.values())[0].get('initialconditions')
            if self.icdict == {}:
                # gen's i.c.s were empty too
                raise PyDSTool_ExistError("No initial conditions specified")
        xdict = {}
        for xname, value in self.icdict.items():
            #if xname not in self.obsvars+self.intvars:
            #    raise ValueError("Invalid variable name in initial "
            #                       "conditions: " + xname)
            xdict[xname] = ensurefloat(value)
        # clean up self.icdict
        self.icdict = xdict.copy()

        ## compute trajectory segment until an event or t1 is reached
        gen = list(self.registry.values())[0]     # sole entry, a Generator
#        print "N-h model before fiddling: t0_global =", t0_global, "t1_global =", t1_global
        t0_global += gen.globalt0
        t1_global += gen.globalt0
        t0 = t0_global    # initial value
        if t1_global > gen.indepvariable.indepdomain[1]+t0:
            if isfinite(gen.indepvariable.indepdomain[1]):
                if self.verboselevel > 0:
                    print("Warning: end time was truncated to max size of specified independent variable domain")
                t1 = gen.indepvariable.indepdomain[1]+t0
            else:
                t1 = t1_global
        else:
            t1 = t1_global

        # For sub-models that support setPars() and
        # compute() methods, set the sub-model's global time
        # reference (in case it needs it), and verify that t=0 is valid
        # in the sub-model's time domain
#        print "\nNonhybrid model %s 'compute'"%self.name
#        print "  generator global t0 =", gen.globalt0
#        print "  model t0_global=", t0_global, "t1_global=",t1_global
#        print "  original model tdata = [%f,%f], gen tdata = [0,%f]"%(tdata[0],tdata[1],t1-t0), " and globalt0=",t0

        # add remaining pars for system
        setup_pars = {'ics': xdict, 'tdata': [0, t1-t0]}
        if self._abseps is not None:
            setup_pars['abseps'] = self._abseps
        if hasattr(gen, 'algparams'):
            setup_pars['algparams'] = {'verbose': self.verboselevel}
            try:
                if gen.algparams['init_step'] > t1-t0:
                    if self.verboselevel > 0:
                        print("Warning: time step too large for remaining time"\
                              + " interval. Temporarily reducing time step to " \
                              + "1/10th of its previous value")
                    setup_pars['algparams'].update({'init_step': (t1-t0)/10})
            except (AttributeError, KeyError):
                # system does not support this integration parameter
                pass
        # if external inputs used, make sure any parameter-bound
        # global t0 offsets are updated in case epoch state map changed them
        if self._inputt0_par_links != {}:
            t0val_dict = {}
            for inp, val in self._inputt0_par_links.items():
                t0val_dict[inp] = self.pars[val]
            gen.set(inputs_t0=t0val_dict)
        gen.set(**setup_pars)
        gen.diagnostics.clearWarnings()
        gen.diagnostics.clearErrors()

        #### Compute a trajectory segment as a NonHybrid Trajectory
        #print "Entering generator %s at t0=%f, t1=%f"%(gen.name, t0, t1)
        try:
            traj = gen.compute(trajname+'_0')
        except PyDSTool_ValueError as e:
            print("\nError in Generator:%s" % gen.name)
            gen.diagnostics.showWarnings()
            gen.diagnostics.showErrors()
            print("Are the constituent generator absolute epsilon ")
            print(" tolerances too small? -- this abseps =%f" % gen._abseps)
            raise
        except KeyboardInterrupt:
            raise
        except:
            print("\nError in Generator:%s" % gen.name)
            gen.diagnostics.showWarnings()
            gen.diagnostics.showErrors()
            self.diagnostics.traceback = {}
            for k,v in gen.diagnostics.traceback.items():
                if isinstance(v, dict):
                    dentry = {}
                    for vk, vv in v.items():
                        dentry[gen._FScompatibleNamesInv(vk)] = vv
                else:
                    dentry = v
                self.diagnostics.traceback[k] = dentry
            if self.diagnostics.traceback != {}:
                print("Traceback dictionary copied to model")
            raise
        if traj is None:
            raise ValueError("Generator %s failed to create a trajectory"%gen.name)
        else:
            if self._abseps is not None:
                assert traj.indepdomain._abseps == self._abseps
                for v in traj.depdomain.values():
                    assert v._abseps == self._abseps
        if not isparameterized(traj):
            raise ValueError("Generator " + gen.name + " produced a " \
                             + " non-parameterized Trajectory")
        # update original copy of the generator with new event times during
        # this trajectory, for high level event detection to be able to check
        # eventinterval for terminal events between hybrid steps
        # ?????
        ##self.modelInfo[gen.name]['dsi'].eventstruct = copy.copy(gen.eventstruct)

        #### Post-process trajectory segment:
        # look at warnings etc. for any terminating events that occurred
        if gen.diagnostics.hasErrors() and self.verboselevel > 0:
            for e in gen.diagnostics.errors:
                print('Generator ' + gen.name + \
                                    ' in trajectory segment had errors')
                gen.diagnostics.showErrors()
        end_reasons = ['time']
        # NB. end reason will always be 'time' for ExplicitFnGen
        # Default time interval after Generator completed,
        # in case it was truncated by a terminal event
        time_interval = traj.indepdomain
        ti_1 = time_interval[1]
        if gen.diagnostics.hasWarnings():
            if self.verboselevel >= 1:
                gen.diagnostics.showWarnings()
            for w in gen.diagnostics.warnings:
                if w[0] == Generator.W_TERMEVENT or \
                   w[0] == Generator.W_TERMSTATEBD:
                    if self.verboselevel > 1:
                        print("Time:", (ti_1+t0))
                        print('Generator ' + gen.name + \
                            ' had a terminal event. Details (in local ' + \
                            'system time) ...\n ' + str(w[1]))
                    # w[1] always has t as first entry, and
                    # either a state variable name
                    # or a list of terminal event names as second.
                    if isinstance(w[1][1], list):
                        if len(w[1][1]) > 1:
                            if self.verboselevel > 0:
                                print('Warning: More than one terminal event found.')
                                print('  Consider altering event pars.')
                        end_reasons = [w[1][1][ri] \
                               for ri in range(len(w[1][1]))]
                    else:
                        end_reasons = [w[1][1]]
                    #print "End reason: ", end_reasons[0]
            self.diagnostics.update(gen.diagnostics)

        # Apply global consistency conditions, and truncate
        # trajectory to the last OK index in the independent variable
        # if any fail, for those conditions that do not apply to the
        # whole trajectory in an accept/reject fashion. If any fail
        # without specifying an index then the whole trajectory is being
        # rejected.
        global_end_reasons = {}
        global_end_ixs = []
        dsi = list(self.modelInfo.values())[0]['dsi']
        for globalDS in list(self.modelInfo.values())[0]['globalConRules']:
            if globalDS(dsi):
                global_end_reasons[np.Inf] = \
                        globalDS.conditions.collate_results('reasons',
                                                        merge_lists=True)
                global_end_ixs.append(np.Inf)
            else:
                # if global consistency fails then the features *must* provide
                # the last OK position in the trajectory to truncate to,
                # otherwise the whole trajectory FAILs.
                try:
                    final_ok_idx = globalDS.conditions._find_idx()
                except (AttributeError, RuntimeError):
                    print("Trajectory creation failed...")
                    globalDS.conditions.results.info()
                    raise PyDSTool_ExistError("Global consistency checks failed for"
                            " model interface %s for trajectory %s"%(str(globalDS),
                                                                 traj.name))
                else:
                    if final_ok_idx is not None:
                        global_end_ixs.append(final_ok_idx)
                    global_end_reasons[final_ok_idx] = \
                            globalDS.conditions.collate_results('reasons',
                                                        merge_lists=True)
        if len(global_end_reasons) > 0:
            smallest_ok_idx = min(global_end_ixs)
            # this idx may be Inf if globalDS passed all points OK
            # - but it may have still changed (refined) the end reason
            if isfinite(smallest_ok_idx):
                # truncate in place
                traj.truncate_to_idx(smallest_ok_idx)
            # overwrite current end reason
            end_reasons = global_end_reasons[smallest_ok_idx]
        if self.verboselevel > 1:
            print("End reason in %s was %s"%(self.name,str(end_reasons)))
        # DEBUG
#        print "Nonhybrid traj info:", traj.indepdomain.get(), traj.globalt0, type(traj)
#        print "traj vars:", traj.variables.keys(), "\n"

        #### Clean up
        if ti_1 > t1_global-t0:
            print("Warning: Generator time interval exceeds prescribed limits:")
            print(" ... %f > %f" % (ti_1, t1_global-t0))
        epochEvents = {}
        try:
            epochEvents.update(gen.getEvents(asGlobalTime=True))
        except (AttributeError, TypeError):
            # this Generator has no eventstruct
            pass

        self.trajectories[trajname] = traj


    def _validateRegistry(self, obsvars, intvars):
        """Validate Model's modelInfo attribute."""
        # ensure that modelInfo is a single Generator object only
        assert len(self.modelInfo) == 1, \
               "Non-hybrid model must contain a single Generator"
        # Avoids circular dependence
        #infodict = self.modelInfo.values()[0]
        #if not isinstance(infodict['dsi'], ModelContext.GeneratorInterface):
        #    raise TypeError("Must provide a single Generator object"
        #                    " wrapped in a GeneratorInterface")

    def _infostr(self, verbose=1):
        if verbose > 0:
            outputStr = 'Non-Hybrid Model '+self.name+" containing components:"
            print("Observable variables:%r" % self.obsvars)
            print("Internal variables:%r" % self.intvars)
            print("Auxiliary variables:%r" % self.auxvars)
            name, infodict = list(self.modelInfo.items())[0]
            outputStr += "\n--- Generator: "+name
            outputStr += "\n  "+infodict['dsi'].model._infostr(verbose-1)
        else:
            outputStr = 'Non-Hybrid Model '+self.name
        return outputStr



class HybridModel(Model):
    """
    obsvars specifies the observable
    variables for this model, which must be present in all
    trajectory segments (regardless of which other variables are
    specified in those segments). intvars specifies
    non-observable (internal) variables for the model, which
    may or may not be present in trajectory segments, depending
    on reductions, etc., that are present in the sub-model (e.g.
    generator) that determines the trajectory segment.
    """
    def __init__(self, *a, **kw):
        Model.__init__(self, True, *a, **kw)
        # collect parameter info from all modelInfo objects
        self._generateParamInfo()
        # Ensure all Generators provided to build trajectory share the same
        # observables, and that modelInfo switch rules mappings are OK
        self._validateRegistry(self.obsvars, self.intvars)
        # Check that all pars of the same name are equal
        self._validateParameters()

    def _prepareICs(self, xdict, traj, ds, t0, ti_1):
        # ds_old no longer needed
        # get new initial conditions for this epoch (partition)
        # DEBUG
#        print "\n\n_prepareICs for HybridModel %s and old traj %s"%(self.name,traj.name)
#        print "t0=%f, ti_1=%f"%(t0,ti_1)
#        print "old traj globalt0 =", traj.globalt0
#        print "t domain =", traj.indepdomain.get()
#        print dict(traj(traj.indepdomain[1], asGlobalTime=False)), "\n"
        for xname in self.allvars+self.auxvars:
            # update final condition of epoch to be used to generate
            # next initial condition when we re-enter the loop
            xname_compat = traj._FScompatibleNames(xname)
            try:
                xdict[xname] = traj.variables[xname_compat].output.datapoints[1][-1]
            except KeyError:
                # e.g. auxiliary var didn't need calling
                pass
            except AttributeError:
                # non-point based output attributes of a Variable need
                # to be 'called' ...
                try:
                    xdict[xname] = traj.variables[xname_compat](ti_1)
                except RuntimeError:
                    # traj is a hybrid traj
                    print("**********")
                    print(xname + " dep domain:%r" % traj.variables[xname_compat].depdomain.get())
                    print(traj.depdomain[xname_compat].get())
                    print("indep domain:%r" % traj.variables[xname_compat].indepdomain.get())
                    print("Var val at t-eps %f = %r" % (
                        ti_1-1e-4, traj.variables[xname_compat](ti_1-1e-4)))
                    raise
                except:
                    print("**********")
                    print(xname + " dep domain:%r" % traj.variables[xname_compat].depdomain.get())
                    print(traj.depdomain[xname_compat].get())
                    print(traj.trajSeq[-1].variables[xname_compat].depdomain.get())
                    raise
            except PyDSTool_BoundsError:
                print("Value out of bounds in variable call:")
                print("  variable '%s' was called at time %f"%(xname, ti_1))
                raise


    def _applyStateMap(self, epochStateMaps, model_interface,
                       next_model_interface, traj, xdict, t0):
        # the mapping might be the ID fn
        num_maps = len(epochStateMaps)
        # apply all maps to the initial condition
        # but check that the ordering doesn't matter (they are
        # commutative), otherwise tell the user we can't safely
        # proceed [FOR NOW, JUST DON'T ALLOW SIMULTANEOUS EVENTS]
        if num_maps > 1:
            # ensure that all the maps are the same for multiple
            # simultaneous events that point to the same generator
            # (the latter was already verified above)
            for mapix in range(1,num_maps):
                if epochStateMaps[0] != epochStateMaps[mapix]:
                    # FIXME: this check clashes with rich comparison in
                    # `EvMapping` class.  Needs to be investigated deeper as
                    # also has broken logic. See discussion in issue 29
                    # (https://github.com/robclewley/pydstool/issues/29)
                    # For now simply suppress exception raising to make
                    # 'example/IF_delaynet_syn.py' to pass.
                    # raise RuntimeError("PyDSTool does not yet allow "
                    #      "truly simultaneous events that do not point "
                    #      "to the same model with the same mapping")
                    pass
        # Now work out new inputs state
        dsinps = model_interface.get('inputs', xdict, t0)
        if dsinps != {}:
            # build dictionary of initial values for the
            # external inputs
            extinputs_ic = {}.fromkeys(dsinps)
            try:
                for k in extinputs_ic:
                    extinputs_ic[k] = dsinps[k](t0)
            except PyDSTool_BoundsError:
                print("Cannot proceed - sub-model '" + model_interface.model.name \
                      + "' had an external input undefined " \
                      + "at t =%f" % t0)
                raise
        else:
            extinputs_ic = {}
        eventstruct = model_interface.get('eventstruct', xdict, t0)
        pars = model_interface.get('pars', xdict, t0)
        pars_temp = traj._FScompatibleNames(pars)
        xdict_temp = traj._FScompatibleNames(xdict)
        for epochStateMap in epochStateMaps:
            # epochStateMap might update xdict_temp and pars_dict
            epochStateMap.evmapping(xdict_temp, pars_temp,
                            extinputs_ic, eventstruct, t0)
        # use update method to change dicts *in place*
        pars.update(traj._FScompatibleNamesInv(pars_temp))
        # update next MI's parameters if mapped by the event mapping
        # (the one corresponding to the original xdict)
        try:
            next_model_interface.set('pars', pars, xdict, t0)
        except (PyDSTool_AttributeError, PyDSTool_ValueError):
            # next model may not have these params, so just ignore
            pass
        xdict.update(traj._FScompatibleNamesInv(xdict_temp))
        # No longer need the aux vars
        for xname in self.auxvars:
            try:
                del xdict[xname]
            except KeyError:
                # aux var value never put here
                pass

    def cleanupMemory(self):
        """Clean up memory usage from past runs of a solver that is interfaced
        through a dynamic link library. This will prevent the 'continue'
        integration option from being accessible and will delete other data
        about the last integration run.
        """
        for MI in self.registry.values():
            MI.model.cleanupMemory()

    def _findTrajInitiator(self, end_reasons, partition_num, t0, xdict,
                              mi=None, swRules=None):
        # Initial values for these
        epochStateMaps = None
        notDone = True
        reused = False
        if end_reasons is None:
            # first time in the while loop, so set up xnames
            assert partition_num == 0, ("end_reasons was None on a "
                                        "non-initial epoch")
            xnames = list(self.icdict.keys())
            xnames.sort()
            # only for the initial Generator of a trajectory
            try:
                infodict = findTrajInitiator(self.modelInfo, t0, xdict,
                                        self.pars, self.intvars,
                                        self.verboselevel)
            except PyDSTool_ValueError as errinfo:
                print(errinfo)
                raise PyDSTool_ExistError('No unique eligible Model found:'
                  ' cannot continue (check active terminal event definitions'
                  ' or error message above)')
            mi = infodict['dsi']
            swRules = infodict['swRules']
            globalConRules = infodict['globalConRules']
            missing = remain(mi.get('variables', ics=xdict, t0=t0).keys(),
                          xnames+self.auxvars+self.intvars)
            if missing != []:
                raise AssertionError('Missing initial condition specifications'
                                     ' for %s' % str(missing))
            for entry, val in xdict.items():
                if not isfinite(val):
                    print("Warning: %s initial condition for "%mi.model.name \
                           + str(entry) + " = " + str(val))
            if self.verboselevel > 1:
                print("\nStarting partition 0")
                print("Chose initiator '%s'"%mi.model.name)
            nextModelName = None
        else:
            # find new Generator using switching rules
            num_reasons = len(end_reasons)
            # num_reasons > 1 will occur most often
            # for discrete time systems
            if num_reasons > 1:
                # then all next generators must be the same!
                candidateNextModel = ""
                mapinfos = []
                for ri in range(num_reasons):
                    try:
                        mapinfos.append(swRules[end_reasons[ri]])
                    except KeyError:
                        # no rule for this termination reason
                        continue
                    nextModelName = mapinfos[-1][0]
                    if candidateNextModel == "":
                        candidateNextModel = nextModelName
                    else:
                        if candidateNextModel != nextModelName:
                            raise RuntimeError("The multiple reasons "
                              "for terminating last trajectory segment"
                              " did not point to the same generator to"
                              " continue the calculation.")
                if len(mapinfos) == 0:
                    # No next generators found
                    nextModelName = 'terminate'
                else:
                    epochStateMaps = [m[1] for m in mapinfos]
            else:
                if end_reasons[0] in swRules:
                    mapinfo = swRules[end_reasons[0]]
                    nextModelName = mapinfo[0]
                    epochStateMaps = [mapinfo[1]]
                else:
                    nextModelName = 'terminate'
                    epochStateMaps = None   # will not be needed
            if nextModelName == 'terminate':
                if self.verboselevel > 0:
                    print('Trajectory calculation for Model `'+mi.model.name\
                          +'` terminated without a DS specified in switch'\
                          +' rules with which to continue.')
                globalConRules = None
                notDone = False   # terminates compute method
            elif nextModelName == mi.model.name:
                # Reuse the same model copy (take advantage of
                # not having to re-copy the model from modelInfo)
                # Currently we can't use the continue integration
                # option of ODE solvers because of the way that each
                # new hybrid partition is solved in "local time", i.e.
                # from t=0. So this would mess up the solvers`
                # 'continue' option.
                reused = True   # not used in compute
                # swRules is untouched
                # return same globalConRules as had before
                globalConRules = self.modelInfo[nextModelName]['globalConRules']
            else:
                next = self.modelInfo[nextModelName]
                mi = next['dsi']
                swRules = next['swRules']
                globalConRules = next['globalConRules']
        if self.verboselevel > 1:
            print("\nStarting partition #%i"%partition_num)
            print("Chose model '%s'"%mi.model.name)
        return mi, swRules, globalConRules, nextModelName, reused, \
               epochStateMaps, notDone

    def _addTraj(self, trajname, trajseq, epochEvents,
                 modelNames, force_overwrite=False):
        """Add a computed trajectory to database."""

        if not force_overwrite:
            assert trajname not in self.trajectories, \
                   'Trajectory name already used'
        #genEvStructs = dict(zip(modelNames,
        #                [copy.copy(self.modelInfo[gn]['dsi'].eventstruct) \
        #                           for gn in modelNames]))
        # TEMP
        #print "Model: avoid copying eventstructs?"
        genEvStructs = dict(zip(modelNames,
                        [self.modelInfo[gn]['dsi'].eventstruct \
                                   for gn in modelNames]))
        # time_range is whole time range over sequence
        time_range = [None, None]
        indepvarname = ''
        genEventTimes = {}   # keyed by event name
        genEvents = {}    # keyed by event name
        # collate all FScompatible name maps from trajectories
        FScompatibleNames = symbolMapClass()
        FScompatibleNamesInv = symbolMapClass()
        for traj_ix, traj in enumerate(trajseq):
            assert isinstance(traj, Trajectory), \
                   ('traj must contain trajectories')
            if indepvarname == '':
                indepvarname = traj.indepvarname
##            else:
##                if indepvarname != traj.indepvarname:
##                    print indepvarname, traj.indepvarname
##                    raise ValueError("Independent "
##                                        "variable name mismatch in trajectory")
            FScompatibleNames.update(traj._FScompatibleNames)
            FScompatibleNamesInv.update(traj._FScompatibleNamesInv)
            for evname in epochEvents[traj_ix]:
                if evname in genEvents and genEvents[evname] is not None:
                    val = mapNames(traj._FScompatibleNamesInv,
                                       epochEvents[traj_ix][evname])
                    if val is not None:
                        try:
                            genEvents[evname].append(val)
                        except ValueError:
                            # in extreme cases with highest precision events,
                            # may have an "equal" time being added
                            if genEventTimes[evname][-1] - val['t'][0] > 0:
                                # some other ValueError
                                raise
                            else:
                                # don't add this point, it's identical
                                pass
                        else:
                            try:
                                genEventTimes[evname].extend( \
                                    val.indepvararray.tolist())
                            except AttributeError:
                                pass
                else:
                    genEvents[evname] = mapNames(traj._FScompatibleNamesInv,
                                          epochEvents[traj_ix][evname])
                    try:
                        genEventTimes[evname] = \
                            epochEvents[traj_ix][evname].indepvararray.tolist()
                    except AttributeError:
                        pass
            if time_range[0] is None:
                # first partition of sequence
                DS_t_numtype = traj.indepdomain.type
                if isinputcts(traj):
                    DS_t_typestr = 'continuous'
                    # continuous time must be float
                    if not compareNumTypes(DS_t_numtype, _all_float):
                        raise TypeError('continuous time inconsistent with '
                                        'non-float type')
                else:
                    DS_t_typestr = 'discrete'
                    # discrete time can be floats or ints
                traj_vars = traj._FScompatibleNamesInv(traj.variables)
                assert intersect(self.allvars, traj_vars.keys()), \
                   "variable name for traj not present in sub-model's variables"
                # Pick one observable variable ('varname') to test with -- they
                # must all have the same type. This variable must be present in
                # all trajectory segments.
                varname = self.obsvars[0]
                if varname in traj_vars:
                    DS_x_numtype = traj_vars[varname].depdomain.type
                else:
                    raise ValueError("varname not known")
##                assert DS_x_numtype == self.obsvars[varname], \
##                       ('Mismatch between declared type of variable and '
##                        'that found in trajectory segment')
                time_range = traj.indepdomain.get()
                try:
                    time_range[0] += traj.globalt0
                    time_range[1] += traj.globalt0
                except IndexError:
                    raise ValueError('time interval of segment in hybrid '
                                            'trajectory cannot be singleton')
                # DEBUG
                # temp
#                print "\n---- %s _addTraj case 1: "%self.name
#                print traj.indepdomain.get()
#                print time_range
#                print traj.globalt0
                time_partitions = [(traj.indepdomain, \
                                    traj.globalt0, \
                                    traj.checklevel)]  # initial value
            else:
                # remove the following line to support trajectories that
                # only have partially-defined variable information, esp.
                # if part is a map (vars defined only at discrete times)
                if not compareNumTypes(DS_t_numtype, traj.indepdomain.type):
                    raise TypeError('Mismatched time types for hybrid '
                                    'trajectory sequence')
#                print "\n---- %s _addTraj case 2: "%self.name
#                print traj.indepdomain.get()
#                temp = copy.copy(time_range)
#                temp[1] += traj.indepdomain[1]
#                print temp
#                print traj.globalt0
                if not traj.indepdomain.atEndPoint(time_range[1] \
                                                    - traj.globalt0, 'lo'):
                    # temp
#                    print "\n***", time_range
#                    print traj.indepdomain.get()
#                    print traj.globalt0
                    raise ValueError('Hybrid trajectory sequence time intervals'
                                     ' must be contiguous: ' + \
                                     str(traj.indepdomain[0]) + ' vs. ' + \
                                     str(time_range[1]-traj.globalt0))
                time_range[1] += traj.indepdomain[1]
                time_partitions.append((traj.indepdomain, \
                                        traj.globalt0, \
                                        traj.checklevel))
        # Full time interval of the trajectory
        time_interval = Interval(indepvarname, DS_t_numtype, time_range,
                                 abseps=self._abseps)
        # Add to trajectory dictionary, using hierarchical event names for
        # genEvents and genEventTimes (which are called directly by user methods
        # of Model)
        self.trajectories.update({trajname: \
                HybridTrajectory(trajname, trajseq,
                                 timePartitions=time_partitions,
                                 timeInterval=time_interval,
                                 eventTimes=genEventTimes,
                                 events=genEvents,
                                 modelEventStructs=genEvStructs,
                                 modelNames=modelNames,
                                 FScompatibleNames=FScompatibleNames,
                                 FScompatibleNamesInv=FScompatibleNamesInv,
                                 abseps=self._abseps,
                                 globalt0=0, norm=self._normord)
                                  })

    def haveJacobian(self):
        """Returns True iff all objects in modelInfo have
        defined Jacobians."""
        result = True
        for model in self.registry.values():
            result = result and model.haveJacobian()
        return result

    def haveJacobian_pars(self):
        """Returns True iff all objects in modelInfo have
        defined Jacobians."""
        result = True
        for model in self.registry.values():
            result = result and model.haveJacobian_pars()
        return result

    def Rhs(self, dsName, t, xdict, pdict=None, asarray=False):
        """Direct access to a sub-model generator's Rhs function.
        Arguments:

          dsName   Name of a sub-model
          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        try:
            dsi = self.modelInfo[dsName]['dsi']
        except KeyError:
            raise ValueError("No DS named %s was found"%dsName)
        if pdict is None:
            pdict = self.pars
        if asarray:
            return dsi.Rhs(t, xdict, pdict, asarray=True)
        else:
            # get returns FS compatible names
            varnames = dsi.get('funcspec', xdict, t).vars
            return Point({'coorddict': dict(zip(varnames,
                                dsi.Rhs(t, xdict, pdict))),
                      'coordtype': float,
                      'norm': self._normord})

    def Jacobian(self, dsName, t, xdict, pdict=None, asarray=False):
        """Direct access to a sub-model generator's Jacobian function (if defined).
        Arguments:

          dsName   Name of a sub-model
          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        try:
            dsi = self.modelInfo[dsName]['dsi']
        except KeyError:
            raise ValueError("No DS named %s was found"%dsName)
        if pdict is None:
            pdict = self.pars
        if dsi.haveJacobian():
            if asarray:
                return dsi.Jacobian(t, xdict, pdict, asarray=True)
            else:
                varnames = dsi.get('funcspec', xdict, t).vars
                return Pointset({'coorddict': dict(zip(varnames,
                                              dsi.Jacobian(t, xdict, pdict))),
                             'coordtype': float,
                             'norm': self._normord})
        else:
            raise PyDSTool_ExistError("Jacobian not defined")

    def JacobianP(self, dsName, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's JacobianP function (if defined).
        Arguments:

          dsName   Name of a sub-model
          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        try:
            dsi = self.modelInfo[dsName]['dsi']
        except KeyError:
            raise ValueError("No DS named %s was found"%dsName)
        if dsi.haveJacobian_pars():
            if pdict is None:
                pdict = self.pars
            if asarray:
                return dsi.JacobianP(t, xdict, pdict, asarray=True)
            else:
                parnames = dsi.get('funcspec', xdict, t).pars
                return Pointset({'coorddict': dict(zip(parnames,
                                    dsi.JacobianP(t, xdict, pdict))),
                             'coordtype': float,
                             'norm': self._normord})
        else:
            raise PyDSTool_ExistError("Jacobian w.r.t. pars not defined")

    def MassMatrix(self, dsName, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's MassMatrix function (if defined).
        Arguments:

          dsName   Name of a sub-model
          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        try:
            dsi = self.modelInfo[dsName]['dsi']
        except KeyError:
            raise ValueError("No DS named %s was found"%dsName)
        if pdict is None:
            pdict = self.pars
        if asarray:
            dsi.MassMatrix(t, xdict, pdict, asarray=True)
        else:
            varnames = dsi.get('funcspec', xdict, t).vars
            return Point({'coorddict': dict(zip(varnames,
                            dsi.MassMatrix(t, xdict, pdict))),
                      'coordtype': float,
                      'norm': self._normord})

    def AuxVars(self, dsName, t, xdict, pdict=None, asarray=False):
        """Direct access to a generator's auxiliary variables
        definition (if defined).

        Arguments:

          dsName   Name of a sub-model
          t        time (can use 0 for an autonomous system)
          xdict    state dictionary or Point.
          pdict    parameter dictionary or Point
                   (optional, default current parameters)
          asarray  (Bool, optional, default False) If true, will return an array
                   in state name alphabetical order, else a Point
        """
        try:
            dsi = self.modelInfo[dsName]['dsi']
        except KeyError:
            raise ValueError("No DS named %s was found"%dsName)
        if pdict is None:
            pdict = self.pars
        if asarray:
            return dsi.AuxVars(t, xdict, pdict, asarray=True)
        else:
            auxvarnames = dsi.get('funcspec', xdict, t).auxvars
            return Point({'coorddict': dict(list(zip(auxvarnames,
                                dsi.AuxVars(t, xdict, pdict)))),
                      'coordtype': float,
                      'norm': self._normord})

    def compute(self, trajname, **kw):
        """Compute a hybrid trajectory and store it internally in the 'trajectories'
        attribute.

        Arguments (non-keyword):
          trajname   Name of trajectory to create (string)

        Arguments (keyword only, all optional):
          force      (Bool, default False) - force overwrite of any trajectory
                     stored in this object with the same name
          verboselevel  (int, default 0)
          ics        initial conditions dict or Point
          pars       parameters dict or Point
          tdata      time data (interval as sequence of 2 numeric values)
        """
        # initially expect to compute over a global time interval,
        # [t0_global, t1_global], which is truncated if it extends beyond
        # DS's independent variable domain to give [t0, t1].
        # DS tdata is then set to compute over relative time interval,
        # [0, t1-t0].
        tdata, t0_global, t1_global, force_overwrite = \
                Model._prepareCompute(self, trajname, **kw)
        # Set initial reason for an epoch to end to None to initiate
        # search for eligible Generator or Model for first epoch
        end_reasons = None
        # check initial condition specification in icdict
        if self.icdict == {}:
            raise PyDSTool_ExistError("No initial conditions specified")

        xdict = {}
        for xname, value in self.icdict.items():
            # ensure string in case Symbolic
            xname = str(xname)
            if xname not in self.allvars:
                raise ValueError("Invalid variable name in initial "
                                   "conditions: " + xname)
            xdict[xname] = ensurefloat(value)
        # clean up self.icdict
        self.icdict = xdict.copy()

        # initial values
        notDone = True  # finished computing trajectory segment?
        t0 = t0_global
        partition_num = 0
        trajseq = []
        traj = None
        ti_1 = None
        modelNames = []
        epochEvents = []
        MI_prev = None
        MI = None
        swRules = None
        # flag for re-use of a model from one hybrid segment to the next
        reused = False
        # reset persistent storage of event times
        self.resetEventTimes()

        # From t0 (if non-autonomous system), icdict, and switching rules
        # (self-consistency validity conditions), determine which model
        # applies for the initial portion of the trajectory
        while notDone:
            # find appropriate model to compute trajectory segment
            # (reused flag indicates whether MI is the same as on previous loop
            # but is not used at the moment)
            MI, swRules, globalConRules, nextModelName, reused, \
                epochStateMaps, notDone = self._findTrajInitiator(end_reasons,
                                                      partition_num, t0, xdict,
                                                      MI, swRules)
            if not notDone:
                continue

            # convenient shorthand
            model = MI.model

            if partition_num > 0:
                ## map previous ds state
                # to new ds state using epoch mapping
                # and current ds's pars and external inputs
                # (previous) traj is well defined from previous run
                pars_copy = copy.copy(self.pars)
                self._applyStateMap(epochStateMaps, MI_prev, MI, traj, xdict, t0)
                ## if changed last MI's event delay, then change it back.
                for event, delay in list(event_delay_record.items()):
                    event.eventdelay = delay

                ## temporarily set terminal events from previous in this MI to have
                # event_delay = event_interval because event finding code
                # does not know that an event just occurred in previous MI and
                # partition.
                event_delay_record = {}
                dummy1, dummy2, targetMI = MI._get_initiator_cache(xdict, t0)
                evstruct = targetMI.get('eventstruct', xdict, t0)
                # record change for later resetting
                for reason in end_reasons:
                    # should only be one!
                    try:
                        event = evstruct.events[reason]
                    except KeyError:
                        pass
                    else:
                        event_delay_record[event] = event.eventdelay
                        event.eventdelay = event.eventinterval
            else:
                # way to force partition 0 to definitely update any
                # parameter-linked input t0's
                pars_copy = {}
                event_delay_record = {}

            # if external inputs used, make sure any parameter-bound
            # global t0 offsets are updated in case epoch state map changed them
            if self._inputt0_par_links != {} and pars_copy != self.pars:
                t0val_dict = {}
                for inp, val in list(self._inputt0_par_links.items()):
                    t0val_dict[inp] = self.pars[val]
                model.set(inputs_t0=t0val_dict)

            ## compute trajectory segment until an event or t1 is reached
            indepvar = MI.get('indepvariable', xdict, t0)
            if t1_global > indepvar.indepdomain[1]+t0:
                if isfinite(indepvar.indepdomain[1]):
                    if self.verboselevel > 0:
                        print("Warning: end time was truncated to max size " + \
                              "of specified independent variable domain")
                    t1 = indepvar.indepdomain[1]+t0
                else:
                    t1 = t1_global
            else:
                t1 = t1_global

            # For Models that support setPars() and
            # compute() methods, set the global time
            # reference (in case it needs it), and verify that t=0 is valid
            # in the time domain
            ## next line redundant (and changed) by setup_pars['tdata'] below
#            print "\nHybrid model %s:  set tdata=[0, %f] in %s (type %s)"%(self.name, t1-t0, MI.model.name, str((type(MI), type(MI.model))))
#            print "  H model t0, t1 =", t0, t1
#            print "  and globalt0 = ", t0
            MI.set('tdata', [0, t1-t0], xdict, t0)
            MI.set('globalt0', t0, xdict, t0)

            # add remaining pars for system
            setup_pars = {'ics': xdict, 'algparams': {}}
            #if self._abseps is not None:
            #    setup_pars['abseps'] = self._abseps
            #    setup_pars['algparams'].update({'abseps': self._abseps})
            try:
                if MI.get('algparams', xdict, t0)['init_step'] > t1-t0:
                    if self.verboselevel > 0:
                        print("Warning: time step too large for remaining time"\
                            + " interval. Temporarily reducing time step to " \
                            + "1/10th of its previous value")
                    setup_pars['algparams'].update({'init_step': (t1-t0)/10})
            except (AttributeError, KeyError, ValueError):
                # system does not support this integration parameter
                pass
            else:
                # we know there's compatibility to set verbose level
                setup_pars['algparams']['verbose'] = self.verboselevel
            model.set(**setup_pars)
            # ensure that if reusing same model as previous segment, that
            # any high level events are not reset: especially for VODE and map system
            # (!!! unsure why not needed for Dopri & Radau)
            try:
                if modelNames[-1] == model.name:
                    # reusing same model again
                    model._set_for_hybrid_DS(True)
            except IndexError:
                # first hybrid partition, so doesn't apply
                pass
            model.diagnostics.clearWarnings()
            model.diagnostics.clearErrors()
#            assert model._abseps == self._abseps

            #### Compute a trajectory segment (as a Hybrid Trajectory)
#            print "DEBUG: Entering model %s at t0=%f, t1=%f"%(model.name, t0, t1)
            try:
                traj = MI.get_test_traj(force=True)
            except PyDSTool_ValueError as e:
                print("\nError in Model:" + model.name)
                model.diagnostics.showWarnings()
                model.diagnostics.showErrors()
                print("Are the constituent generator absolute epsilon ")
                print(" tolerances too small? -- this abseps =%f" % model._abseps)
                raise
            except KeyboardInterrupt:
                raise
            except:
                print("\nError in Model:" + model.name)
                self.diagnostics.traceback = {}
                for k,v in model.diagnostics.traceback.items():
                    if isinstance(v, dict):
                        dentry = {}
                        for vk, vv in v.items():
                            dentry[traj._FScompatibleNamesInv(vk)] = vv
                    else:
                        dentry = v
                    self.diagnostics.traceback[k] = dentry
                if self.diagnostics.traceback != {}:
                    print("Traceback dictionary copied to model")
                raise
            # ensure this is reset to False
            model._set_for_hybrid_DS(False)
            if traj is None:
                raise ValueError("Model %s failed to create a trajectory"%model.name)
            else:
                # rename from ModelInterface default name
                model.renameTraj(ModelContext.ModelInterface._trajname,
                                 trajname+'_'+str(partition_num),
                                 force=force_overwrite)
#                if self._abseps is not None:
#                    assert traj.indepdomain._abseps == self._abseps
#                    for v in traj.depdomain.values():
#                        assert v._abseps == self._abseps
                if not isparameterized(traj):
                    raise ValueError("Model " + model.name + " produced a " \
                                     + " non-parameterized Trajectory")
                time_interval = traj.indepdomain
                # ti_1 means time interval index 1 (i.e., the endpoint)
                # this is a local time
                ti_1 = time_interval[1]
                if self.verboselevel > 1:
                    print("\nt0=", t0, "traj.globalt0=", traj.globalt0)
                    print("traj.indepdomain =  %r" % traj.indepdomain.get())
                    for vn, v in traj.variables.items():
                        print(vn + ": %s %s" % (v.trajirange, v.indepdomain.get()))
                    print("Last point of traj %s was: "%traj.name, traj(ti_1))
            # update original copy of the model with new event times during
            # this trajectory, for high level event detection to be able to check
            # eventinterval for terminal events between hybrid steps
            # ?????
            ##MI.eventstruct = copy.copy(model.eventstruct)

            # append record of the sequence of used generators
            modelNames.append(model.name)

            #### Post-process trajectory segment:
            # look at warnings etc. for any terminating events that occurred
            if model.diagnostics.hasErrors() and self.verboselevel > 0:
                for e in model.diagnostics.errors:
                    print('Model ' + model.name + \
                                        ' in trajectory segment had errors')
                    model.diagnostics.showErrors()
            # default end reason is time (may be overwritten)
            end_reasons = ['time']
            # Default time interval after traj computation completed,
            # in case it was truncated by a terminal event
            # DEBUG
#            print "Hybrid Traj %s t0, ti_1 ="%traj.name, traj.indepdomain[0], ti_1
            if model.diagnostics.hasWarnings():
                if self.verboselevel >= 1:
                    model.diagnostics.showWarnings()
                for w in model.diagnostics.warnings:
                    if w[0] == Generator.W_TERMEVENT or \
                       w[0] == Generator.W_TERMSTATEBD:
                        if self.verboselevel > 1:
                            print("Time:", (ti_1+t0))
                            print('Model ' + model.name + \
                                ' had a terminal event. Details (in local ' + \
                                'system time) ...\n ' + str(w[1]))
                        # w[1] always has t as first entry, and
                        # either a state variable name
                        # or a list of terminal event names as second.
                        if isinstance(w[1][1], list):
                            if len(w[1][1]) > 1:
                                if self.verboselevel > 0:
                                    print('Warning: More than one terminal event found.')
                                    print('  Consider altering event pars.')
                            end_reasons = [w[1][1][ri] \
                                   for ri in range(len(w[1][1]))]
                        else:
                            end_reasons = [w[1][1]]
                        #print "End reason: ", end_reasons[0]
                self.diagnostics.update(model.diagnostics)

            # Apply global consistency conditions, and truncate
            # trajectory to the last OK index in the independent variable
            # if any fail, for those conditions that do not apply to the
            # whole trajectory in an accept/reject fashion. If any fail
            # without specifying an index then the whole trajectory is
            # rejected.
            global_end_reasons = {}
            global_end_ixs = []
            for globalDS in globalConRules:
                if globalDS(MI):
                    global_end_reasons[np.Inf] = \
                            globalDS.conditions.collate_results('reasons',
                                                            merge_lists=True)
                    global_end_ixs.append(np.Inf)
                else:
                    # if global consistency fails then the features *must* provide
                    # the last OK position in the trajectory to truncate to,
                    # otherwise the whole trajectory FAILs.
                    try:
                        final_ok_idx = globalDS.conditions._find_idx()
                    except (AttributeError, RuntimeError):
                        print("Trajectory creation failed...")
                        globalDS.conditions.results.info()
                        raise PyDSTool_ExistError("Global consistency checks failed for"
                                " model interface %s for trajectory %s"%(str(globalDS),
                                                                     traj.name))
                    else:
                        if final_ok_idx is None:
                            final_ok_idx = np.Inf
                        global_end_ixs.append(final_ok_idx)
                        global_end_reasons[final_ok_idx] = \
                                globalDS.conditions.collate_results('reasons',
                                                            merge_lists=True)
            if len(global_end_reasons) > 0:
                smallest_ok_idx = min(global_end_ixs)
                # this idx may be Inf if globalDS passed all points OK
                # - but it may have still changed (refined) the end reason
                if isfinite(smallest_ok_idx):
                    # truncate in place
                    traj.truncate_to_idx(smallest_ok_idx)
                # overwrite current end reason only if it's non-empty
                new_reason = global_end_reasons[smallest_ok_idx]
                if new_reason != []:
                    end_reasons = new_reason
            if self.verboselevel > 1:
                print("End reason in %s was %s"%(self.name,str(end_reasons)))

            # DEBUG
#            print "\nTraj segment %i info:"%partition_num, traj.indepdomain.get(), traj.globalt0, type(traj)
#            print "\n"

            #### Prepare for next partition & clean up
            partition_num += 1
            if time_interval.atEndPoint(t1_global-t0, 'hi'):
                # we reached end of desired computation interval
                notDone = False
            if ti_1 > t1_global-t0:
                # We computed too far for the time interval
                print("Warning: Segment time interval exceeds prescribed limits:")
                print(" ... %f > %f" % (ti_1, t1_global-t0))
                notDone = False
            # last Model's time == t1 should hold after successful traj computation
            # otherwise t1 needs to be updated according to last terminating
            # event time point
            if ti_1 < t1-t0:
                if end_reasons[0] not in swRules:
                    # there's prescribed time left to compute but no
                    # eligible DS to transfer control to
                    print('Trajectory calculation for Model `'+model.name+'` ' \
                          +'terminated without a DS specified in the ' \
                          +'switching rules from which to continue.')
                    print('Perhaps a variable went out of bounds:')
                    model.diagnostics.showWarnings()
                    print('Last reasons for stopping were: %s' % end_reasons)
                    print("Trajectory calculated: ")
                    dt_sample = (ti_1-time_interval[0])/10.
                    print(traj.sample(dt=dt_sample))
                    notDone = False
                    raise PyDSTool_ValueError("Premature termination of "
                                              "trajectory calculation")
                elif swRules[end_reasons[0]][0] == 'terminate':
                    if self.verboselevel > 0:
                        print('Trajectory calculation for Model `'+model.name+'` ' \
                              +'terminated.')
                    notDone = False
                    t1 = ti_1+t0
                else:
                    t1 = ti_1+t0
            # update xdict with final point in traj
            self._prepareICs(xdict, traj, MI, t0, ti_1)
            # store trajectory segment in the sequence
            trajseq.append(traj)
            # before moving on, update t0 for next epoch / regime
            t0 = t1
            try:
                epochEvents.append(traj.getEvents())
            except AttributeError:
                # no eventstruct present
                epochEvents.append({})
            MI_prev = MI
        # end while loop

        # Take a copy of the events as they were at the time that the
        # trajectory was computed, for future reference
        self._addTraj(trajname, trajseq,
                      epochEvents, modelNames, force_overwrite)
        # Record defining data for this trajectory (especially useful for interfaces)
        self.trajectory_defining_args[trajname] = args(ics=copy.copy(self.icdict),
                                              pars=copy.copy(self.pars),
                                              tdata=copy.copy(tdata))

    def _validateRegistry(self, obsvars, intvars):
        """Validate Model's modelInfo attribute.
        Now that a HybridModel is invariably built using ModelConstructor,
        which already has validation built in, and also has more complex ways
        of construction, this method is defunct."""
        pass

##        # ensure that modelInfo consists of more than one ModelInterface object
##        # unless it's a single model punctuated by discrete events
##        if len(self.modelInfo) == 1:
##            infodict = self.modelInfo.values()[0]
##            swmapping = infodict['swRules']
##            if all([outcome[0] == 'terminate' \
##                   for (reason, outcome) in swmapping.iteritems()]):
##                # then all point to 'terminate' and user needed a non-hybrid
##                # model class, otherwise has discrete event mappings
##                # that make the model technically "hybrid"
##                raise AssertionError("Use a non-hybrid Model class")
##        # Avoid circular import for access to ModelInterface
##        #for infodict in self.modelInfo.values():
##        #    if not isinstance(infodict['dsi'], ModelContext.ModelInterface):
##        #        raise TypeError("Must provide ModelInterface objects")
##        allDSnames = self.modelInfo.keys()
###        print "\n_validateRegistry: ", obsvars, intvars
##        for modelName, infodict in self.modelInfo.iteritems():
##            # dynamical system for trajectory
##            miface = infodict['dsi']
##            # dict of reason -> (ds_name, epmapping) pairs
##            swmapping = infodict['swRules']
##            allvarnames = miface.model.query('variables')
###            print "Model %s had vars "%modelName, allvarnames
##            special_reasons = ['time'] + allvarnames
##            assert miface.model.name == modelName, ('DS`s name does not '
##                                'correspond to the key in the modelInfo dict')
##            assert miface.model.name in allDSnames, ('DS`s name not in list '
##                                        'of all available DS names!')
##            assert len(intersect(obsvars, allvarnames)) ==\
##                len(obsvars), 'DS `'+miface.model.name+'` did not contain ' \
##                                      + ' required observable variables'
##            assert len(intersect(intvars, allvarnames)) ==\
##                len(intvars), 'DS '+miface.model.name+' did not contain ' \
##                                      + ' required internal variables'
##            try:
##                allEndReasonNames = miface.model.query('events').keys() \
##                                      + special_reasons
##            except AttributeError:
##                allEndReasonNames = special_reasons
##            seenReasons = []
##            # can't assert this when non-event based reasons are used
##            # (e.g. for DSSRT post-processing which creates the reason based on
##            # generic events)
###            assert len(swmapping) == len(allEndReasonNames), ('Incorrect number'
###                                            ' of map pairs given in argument')
##            for (reason, outcome) in swmapping.iteritems():
##                targetName = outcome[0]
##                epmapping = outcome[1]
##                # no checks are made on epmapping here
##                assert reason not in seenReasons, ('reasons cannot appear more'
##                                            ' than once in map domain')
##                seenReasons.append(reason)
##                assert reason in allEndReasonNames, ('name `'+reason+'` in map '
##                                                    'domain is invalid')
##                if targetName != 'terminate':
##                    assert targetName in allDSnames, ('name `'+targetName+\
##                                            '` in map range is invalid')



    def _validateParameters(self):
        pars_found = {}
        for infodict in self.modelInfo.values():
            try:
                pars_recognized = intersect(pars_found.keys(),
                                        infodict['dsi'].pars)
            except AttributeError:
                # gen doesn't have any pars
                continue  # for loop
            pars_unknown = remain(infodict['dsi'].pars, pars_found.keys())
            for parname in pars_recognized:
                if infodict['dsi'].pars[parname] != pars_found[parname]:
                    print("Inconsistent parameter values between model objects inside model")
                    print("probably means that they were changed during trajectory computation")
                    print("at discrete events: you should reset the pars when calling compute()")
                    raise ValueError("Inconsistency between parameter values"
                     " of same name ('%s') in different constiutent Generator"%parname +\
                     " objects: reset during call to compute")
            new_vals = dict(zip(pars_unknown, [infodict['dsi'].pars[p] \
                                               for p in pars_unknown]))
            pars_found.update(new_vals)


    def _infostr(self, verbose=1):
        if verbose > 0:
            outputStr = 'Hybrid Model '+self.name+" containing components:"
            outputStr += "Observable variables: " + ",".join(self.obsvars)
            outputStr += "Internal variables: " + ",".join(self.intvars)
            outputStr += "Auxiliary variables: " + ",".join(self.auxvars)
            for name, infodict in self.modelInfo.items():
                outputStr += "\n--- Sub-model: "+name
                outputStr += "\n  "+infodict['dsi'].model._infostr(verbose-1)
        else:
            outputStr = 'Hybrid Model '+self.name
        return outputStr


# -------------------------------------------------------------------------
#### Private functions

def getAuxVars(dsi, t, icdict, pardict):
    """Return auxiliary variable values evaluated at t, icdict,
    pardict.
    """
    # prepare auxiliary variables eval'd at i.c.
    icdict_local = copy.copy(icdict)
    # to restore these to their former values after domain tests are done,
    # (in case events refer to auxiliary function initcond()
    old_icdict = copy.copy(dsi.model.icdict)
    # update icdict with auxiliary variable vals eval'd at t, for
    # use in event function calls for domain test.
    dsi.model.icdict.update(icdict_local)
    icdict_local['t'] = t
    fspec = dsi.get('funcspec', icdict, t)
    if fspec.auxvars == []:
        auxdict = {}
    else:
        try:
            auxdict = dict(zip(
                fspec.auxvars,
                dsi.get('AuxVars', icdict, t)(*(t, icdict_local, pardict)) ))
        except (ValueError, TypeError):
            # no auxiliary variables for this model
            # e.g. ExplicitFnGen has its auxvars defined through the
            # main variables
            auxdict = {}
    # restore model's original initial conditions
    dsi.model.icdict = old_icdict
    return auxdict


def findTrajInitiator(modelInfo, t, vardict, pardict, intvars,
                         verboselevel=0):
    """Find eligible Model to begin computation of trajectory.
    Cannot depend on any internal variables (e.g. those not common to
    all sub-models).
    """
    eligibleMI = []
    if len(modelInfo) == 1:
        return list(modelInfo.values())[0]
    outcome = {}
    for infodict in modelInfo.values():
        MI = infodict['dsi']
        MI_vars = MI.query('vars')
        xdict = filteredDict(vardict, MI_vars)
        dxdt_zeros = dict(zip(xdict.keys(),[0.]*len(xdict)))
        domtests = infodict['domainTests']
        globalConRules = infodict['globalConRules']
        model = MI.model
        outcome[model.name] = {}
        t_test = MI.get('indepvariable', xdict, t).indepdomain.contains(t) \
               is not notcontained
        fs = MI.get('funcspec', xdict, t)
        allvars = fs.vars + fs.auxvars
        # if external inputs then include initial value of those in pardict
        dsinps = MI.get('inputs', xdict, t)
        if dsinps != {}:
            try:
                input_ics = [dsinps[in_name](t) for in_name in dsinps.keys()]
            except KeyboardInterrupt:
                raise
            except:
                raise RuntimeError("Error in evaluating inputs for generator at"
                                   " initial condition")
            pardict.update(dict(zip(dsinps.keys(), input_ics)))
        icdict = copy.copy(xdict)
        # don't overwrite any aux vars that are treated as regular variables
        # at this level, but fetch other aux vars and external inputs
        icdict.update(filteredDict(getAuxVars(MI, t, icdict, pardict),
                                   list(icdict.keys()), neg=True))
        # override icdict with any finite-valued generator
        # initial condition (deliberately preset in generator definition)
        # for non-internal variables, and any auxiliary variables that aren't
        # already defined in icdict from above call to getAuxVars (must do this
        # to prevent NaNs being passed to numeric_to_traj.
        for xname in allvars:
            if xname not in xdict or xname in intersect(intvars, xdict):
                if xname in xdict and xname in intvars:
                    # ignore
                    continue
                if xname in fs.auxvars and xname in icdict and isfinite(icdict[xname]):
                    continue
                try:
                    if isfinite(model.icdict[xname]):
                        icdict[xname] = model.icdict[xname]
                except KeyError:
                    # ic (for internal variable) not defined for this
                    # generator, so ignore
                    pass
        # set initial conditions of MI in case auxiliary function initcond
        # is called during Rhs call
        model.set(ics=icdict)
        x_test = True   # initial value
        try:
            # derivatives of variables are just Rhs values for ODE
            dxdt = dict(MI.Rhs(t, icdict, pardict))
        except AttributeError:
            # no Rhs for the underlying generator
            dxdt = dxdt_zeros
        else:
            for vname, val in dxdt.items():
                # hack to fix NaNs showing up for VODE Rhs function
                # but very large finite values come back from Dopri/Radau
                # and Nan's mess up call to numeric_to_traj below
                if not isfinite(val):
                    dxdt[vname] = 1e308
        for aname in fs.auxvars:
            # set derivatives for remaining auxvars to dummy value
            dxdt[aname] = 0
#        print "Temp in findTrajInitiator for %s: x, dxdt ="%MI.model.name
#        print icdict
#        print dxdt
        for xname, x in icdict.items():
            # don't count time 't' or any internal variables
            if xname == 't' or xname in intvars or xname not in domtests:
                # 't' is a special value inserted into icdict, for use
                # in auxiliary variable evaluation at initial time, but not
                # a regular variable so don't do domain check here.
                continue
            xtraj = numeric_to_traj([[x], [dxdt[xname]]], 'test', [xname, 'D_'+xname], t)
#            print "Dom test for %s in %s"%(xname, MI.model.name)
#            print "  discrete? ", domtests[xname].isdiscrete
            newtest = domtests[xname](xtraj)
            if verboselevel >=2:
                print("\nstate dom test for '%s' @ value %f:"%(xname, icdict[xname]))
                print("depdomain is  %r" % MI.get('variables', xdict, t)[xname].depdomain.get())
                print("-> test result is ", newtest)
            x_test = x_test and newtest
        g_test = True  # initial value
        xdict = filteredDict(icdict, intvars, neg=True)
        MI.test_traj = numeric_to_traj(array([list(xdict.values())]).T, 'ic_trajpt',
                                       list(xdict.keys()), t)
        # ensure that events are cleared in case global con rules in a MI
        # check for events (for when used after trajectories computed)
        # don't use MI.get('diagnostics', xdict, t) as that will return a copy
        # so that clearWarnings will not propagate to underlying sub-model
        xdict, t, I = MI._get_initiator_cache(xdict, t)
        while True:
            # descend into sub-models until leaf found
            if isinstance(I, ModelContext.GeneratorInterface):
                I.model.diagnostics.clearWarnings()
                break
            else:
                xdict, t, I = I._get_initiator_cache(xdict, t)

        # !!! Currently assumes globalConRules is a list, not a Context
        # (should be upgraded to that)
        for gc in globalConRules:
            try:
                globtest = gc(MI)
            except KeyboardInterrupt:
                raise
            except:
                globtest = False
            else:
                if verboselevel >= 2:
                    print("\nglobal con test for '%s' @ value %f:"%(str(gc), icdict[xname]))
                    print("-> test result is ", globtest)
                g_test = g_test and globtest
        if verboselevel >= 1:
            print("\nModel '%s' tests..."%model.name)
            print(" ...for initial time: " + str(t_test))
            print(" ...for initial state: " + str(x_test))
            print(" ...for global conditions: " + str(g_test))
        outcome[model.name]['time dom. test'] = t_test
        outcome[model.name]['state dom. test'] = x_test
        outcome[model.name]['global con. test'] = g_test
        if t_test and x_test and g_test:
            # this t, xdict is a valid initial state for this Generator
            eligibleMI.append(infodict)
    if eligibleMI == []:
        info(outcome, "\nOutcomes for eligibility tests, by model")
        raise PyDSTool_ValueError('No eligible models from'
                           ' which to begin trajectory computation')
    if len(eligibleMI) > 1:
        info(outcome, "\nOutcomes for eligibility tests, by model")
        raise PyDSTool_ValueError('Too many eligible models'
                           ' to start trajectory computation')
    # only remaining possibility is a single eligible model
    return eligibleMI[0]
