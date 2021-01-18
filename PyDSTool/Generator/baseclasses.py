# Generator base classes: Generator, ctsGen, discGen

from .allimports import *
from PyDSTool.utils import *
from PyDSTool.common import *
from PyDSTool.Symbolic import ensureStrArgDict, Quantity, QuantSpec, \
     mathNameMap, allmathnames
from PyDSTool.Trajectory import Trajectory
from PyDSTool.parseUtils import symbolMapClass, readArgs
from PyDSTool.Variable import Variable, iscontinuous
from PyDSTool.Points import Pointset
import PyDSTool.Events as Events

# Other imports
from numpy import isfinite, sometrue, alltrue
import numpy as np
import math, random
import os
from copy import copy, deepcopy

# -----------------------------------------------------------------------------

__all__ = ['ctsGen', 'discGen', 'theGenSpecHelper', 'Generator',
           'genDB', 'auxfn_container', '_pollInputs']

# -----------------------------------------------------------------------------

smap_mathnames = symbolMapClass(mathNameMap)

class genDBClass(object):
    """This class keeps a record of which non-Python Generators have been
    created in a session. A single global instance of this class is created,
    and prevents the user from re-using the name of a low-level
    DLL (such as that for a Dopri_ODEsystem vector field) unless the
    system is identical (according to its FuncSpec)."""

    def __init__(self):
        self.database = {}

    def check(self, gen):
        """Look up generator instance and return Boolean of whether
        it is already registered. Only non-python based generators
        are registered, so all others will return False.
        """
        if gen.funcspec.targetlang == 'python':
            # these do not need to be checked
            return False
        try:
            if gen._solver.rhs in self.database:
                entry = self.database[gen._solver.rhs]
                return className(gen) == entry['class'] \
                   and hash(gen.funcspec) == entry['hash']
            else:
                return False
        except AttributeError:
            # these do not need to be checked
            return False

    def unregister(self, gen):
        try:
            del self.database[gen._solver.rhs]
        except (AttributeError, KeyError):
            # invalid type of generator or not present
            # so do nothing
            return

    def register(self, gen):
        if gen.funcspec.targetlang == 'python':
            # these do not need to be checked
            return
        # verify name of vector field in database
        try:
            if gen._solver.rhs not in self.database:
                self.database[gen._solver.rhs] = {'name': gen.name,
                                                  'class': className(gen),
                                                  'hash': hash(gen.funcspec)}
            else:
                if hash(gen.funcspec) != self.database[gen._solver.rhs]['hash']:
                    raise PyDSTool_KeyError("Generator named %s already exists"%gen.name)
                # otherwise it's identical and can ignore
        except AttributeError:
            # these do not need to be checked
            return

    def __repr__(self):
        s = "Generator internal database class: "
        s += str(list(self.database.keys()))
        return s

    __str__ = __repr__


    def clearall(self):
        self.database = {}


# use single instance of nameResolver per session
global genDB
genDB = genDBClass()


# ----------------------------------------------------------------------

class auxfn_container(object):
    """
    Auxiliary function interface for python user
    """
    def __init__(self, genref):
        self.genref = genref
        self.map_ixs = ixmap

    def __getitem__(self, k):
        try:
            return self.__dict__[k]
        except KeyError:
            raise KeyError("Function %s not present"%k)

    def keys(self):
        return [k for k in self.__dict__.keys() \
                if k not in ['genref', 'map_ixs']]

    def __contains__(self, k):
        return k in self.keys()

    def values(self):
        return [self.__dict__[k] for k in self.__dict__.keys() \
                if k not in ['genref', 'map_ixs']]

    def items(self):
        return zip(self.keys(), self.values())

    def __repr__(self):
        return "Aux functions: " + ", ".join(self.keys())


class ixmap(dict):
    def __init__(self, genref):
        self.parixmap = {}
        self.pars = genref.pars
        i = list(genref.inputs.keys())
        i.sort()
        p = list(genref.pars.keys())
        p.sort()
        allnames = p + i
        for pair in enumerate(allnames):
            self.parixmap[pair[0]] = pair[1]

    def __getitem__(self, k):
        try:
            return self.pars[self.parixmap[k]]
        except:
            raise PyDSTool_KeyError("Cannot access external input values using ixmap class")

    def __repr__(self):
        return "Index mapping: " + str(self.parixmap)


class Generator(object):
    """
    Trajectory Generator abstract class.
    """
    # query keys for 'query' method
    _querykeys = ['pars', 'parameters', 'events', 'abseps',
                  'ics', 'initialconditions', 'vars', 'variables',
                  'auxvariables', 'auxvars', 'vardomains', 'pardomains']
    # initialization keyword keys
    _needKeys = ['name']
    _optionalKeys = ['globalt0', 'checklevel', 'model', 'abseps',
                              'eventPars', 'FScompatibleNames',
                              'FScompatibleNamesInv']

    def __init__(self, kw):
        # _funcreg stores function instances (and base classes) that are
        # passed from self.funcspec, as they are not defined in __main__.
        # A Generator object cannot be deep copied or
        # pickled without the additional __getstate__ and __setstate__
        # methods which know how to reconstruct these functions.
        self._funcreg = {}
        try:
            # sometimes certain keys are checked prior to calling this base
            # class so we don't want to tread on the feet of those __init__
            # methods
            dummy = self.foundKeys
            # if this works then we won't reset to zero
        except:
            self.foundKeys = 0
        try:
            self.name = kw['name']
            self.foundKeys += 1
        except KeyError:
            raise PyDSTool_KeyError("name must be supplied as a keyword in arguments")
        # dimension value is set later
        self.dimension = None
        # Regular Variable objects, and auxiliary variables.
        # These are callable and can even appear in
        # inputs. They are defined internally by the FuncSpec object.
        # These self.variables are generally placeholders containing domain
        # information only, except for the special Generator types LookupTable
        # and InterpTable. The actual variable contents from a computed
        # trajectory are created locally during computeTraj and immediately
        # exported to a Trajectory object, not affecting these variable objects.
        self.variables = {}
        # initial conditions for each regular and auxiliary variable
        self.initialconditions = {}
        # Generator's functional specification for variables, e.g.
        # right-hand side of ODE, or data lists for look-up table
        self.funcspec = None
        # Generator's internal pars - for the functional specification
        self.pars = {}
        # Place-holder for event structure, if used
        self.eventstruct = None
        # Place-holder for latest trajectory events
        self.trajevents = None
        # Autonomous external inputs (dictionary of callable Variables)
        self.inputs = {}
        # Local independent variable interval and validity range (relative to global t0)
        self.indepvariable = None
        # Sorted list of callable names
        self._callnames = []
        # Registry of object names and types
        self._registry = {}
        # Internal flag for whether the generator is being used as part of a hybrid DS
        # (useful for control over high level event resetting between trajectory
        # segments when generators are reused) -- HybridModel class must keep
        # track of this and set using the _set_for_hybrid_DS method
        self._for_hybrid_DS = False
        # Absolute tolerance for Interval endpoints
        if 'abseps' in kw:
            self._abseps = kw['abseps']
            self.foundKeys += 1
        else:
            self._abseps = 1e-13
        # `Checklevel` code determining response to uncertain float comparisons
        if 'checklevel' in kw:
            if kw['checklevel'] not in range(4):
                    raise ValueError('Invalid interval endpoint checking option')
            self.checklevel = kw['checklevel']
            self.foundKeys += 1
        else:
            # default: no checking
            self.checklevel = 0

        self.diagnostics = Diagnostics(errmessages, errorfields, warnmessages,
                                       warnfields, propagate_dict=self.inputs)

        # global independent variable reference (for non-autonomous systems,
        # especially when a Generator is embedded in a hybrid Model object)
        if 'globalt0' in kw:
            v = kw['globalt0']
            assert isinstance(v, _num_types), 'Incorrect type of globalt0'
            self.globalt0 = v
            self.foundKeys += 1
        else:
            self.globalt0 = 0

        # If part of a Model, keep a reference to which one this
        # Generator belongs to
        if 'model' in kw:
            assert isinstance(kw['model'], str), "model tag must be a string"
            self._modeltag = kw['model']
            self.foundKeys += 1
        else:
            self._modeltag = None
        # the default map allows title-cased quantity objects like
        # Pow, Exp, etc. to be used in FuncSpec defs, but no need
        # to keep an inverse map of these below
        self._FScompatibleNames = deepcopy(smap_mathnames)
        if 'FScompatibleNames' in kw:
            sm = kw['FScompatibleNames']
            if sm is not None:
                self._FScompatibleNames.update(sm)
            self.foundKeys += 1
        if 'FScompatibleNamesInv' in kw:
            sm = kw['FScompatibleNamesInv']
            if sm is None:
                sm = symbolMapClass()
            self._FScompatibleNamesInv = sm
            self.foundKeys += 1
        else:
            self._FScompatibleNamesInv = symbolMapClass()

        # If there are eventPars, keep track of them (name list)
        self._eventPars = []
        if 'eventPars' in kw:
            if isinstance(kw['eventPars'], list):
                self._eventPars = kw['eventPars']
            elif isinstance(kw['eventPars'], str):
                self._eventPars.append(kw['eventPars'])
            self.foundKeys += 1
            self._eventPars = self._FScompatibleNames(self._eventPars)
        # Indicator of whether a trajectory has been successfully computed
        self.defined = False


    def addEvtPars(self, eventPars):
        """Register parameter names as event specific parameters."""
        if isinstance(eventPars, list):
            self._eventPars.extend(self._FScompatibleNames(eventPars))
        elif isinstance(eventPars, str):
            self._eventPars.append(self._FScompatibleNames(eventPars))

    def getEvents(self, evnames=None, asGlobalTime=True):
        """Produce dictionary of pointsets of all flagged events' independent
        and dependent variable values, for each event (whether terminal or not).
        Times will be globalized if optional asGlobalTime argument is True
        (default behavior). If a single event name is passed, only the pointset
        is returned (not a dictionary).

        evnames may be a singleton string or list of strings, or left blank to
        return data for all events.

        The events are not guaranteed to be ordered by the value of the
        independent variable.
        """
        compat_evnames = self._FScompatibleNamesInv(list(self.trajevents.keys()))
        if evnames is None:
            evnames = compat_evnames
        if asGlobalTime:
            t_offset = self.globalt0
        else:
            t_offset = 0
        if isinstance(evnames, str):
            # singleton
            assert evnames in compat_evnames, "Invalid event name provided: %s"%evnames
            try:
                result = self.trajevents[self._FScompatibleNames(evnames)]
            except AttributeError:
                # empty pointset
                return None
            else:
                result.indepvararray += t_offset
                return result
        else:
            # assume a sequence of strings
            assert all([ev in compat_evnames for ev in evnames]), \
                   "Invalid event name(s) provided: %s"%str(evnames)
            result = {}
            for (evname, evptset) in self.trajevents.items():
                compat_evname = self._FScompatibleNamesInv(evname)
                if compat_evname not in evnames:
                    continue
                result[compat_evname] = copy(evptset)
                try:
                    result[compat_evname].indepvararray += t_offset
                except AttributeError:
                    # empty pointset
                    pass
            return result

    def getEventTimes(self, evnames=None, asGlobalTime=True):
        """Produce dictionary of lists of all flagged events' independent
        variable values, for each event (whether terminal or not).
        Times will be globalized if optional asGlobalTime argument is True
        (default behavior). If a single event name is passed, only the pointset
        is returned (not a dictionary).

        evnames may be a singleton string or list of strings, or left blank to
        return data for all events.

        The events are guaranteed to be ordered by the value of the
        independent variable.
        """
        result = {}
        if asGlobalTime:
            t_offset = self.globalt0
        else:
            t_offset = 0
        compat_evnames = self._FScompatibleNamesInv(list(self.trajevents.keys()))
        if evnames is None:
            evnames = compat_evnames
        if isinstance(evnames, str):
            # singleton
            assert evnames in compat_evnames, "Invalid event name provided: %s"%evnames
            try:
                return self.trajevents[self._FScompatibleNames(evnames)].indepvararray \
                                               + t_offset
            except AttributeError:
                # empty pointset
                return []
        else:
            # assume a sequence of strings
            assert all([ev in compat_evnames for ev in evnames]), \
                   "Invalid event name(s) provided: %s"%str(evnames)
            for (evname, evptset) in self.trajevents.items():
                compat_evname = self._FScompatibleNamesInv(evname)
                if compat_evname not in compat_evnames:
                    continue
                try:
                    result[compat_evname] = evptset.indepvararray + t_offset
                except AttributeError:
                    # empty pointset
                    result[compat_evname] = []
            return result


    def query(self, querykey=''):
        """Return info about Generator set-up.
        Valid query key: 'pars', 'parameters', 'pardomains', 'events',
         'ics', 'initialconditions', 'vars', 'variables',
         'auxvars', 'auxvariables', 'vardomains'
         """
        assert isinstance(querykey, str), \
                       ("Query argument must be a single string")
        if querykey not in self._querykeys:
            print('Valid query keys are: %r' % (self._querykeys, ))
            print("('events' key only queries model-level events, not those")
            print(" inside sub-models)")
            raise ValueError('Query key '+querykey+' is not valid')
        if querykey in ['pars', 'parameters']:
            result = self._FScompatibleNamesInv(self.pars)
        elif querykey in ['ics', 'initialconditions']:
            try:
                result = self._FScompatibleNamesInv(self.initialconditions)
            except AttributeError:
                result = None
        elif querykey == 'events':
            result = self.eventstruct.events
        elif querykey in ['vars', 'variables']:
            result = self._FScompatibleNamesInv(self.funcspec.vars)
        elif querykey in ['auxvars', 'auxvariables']:
            result = self._FScompatibleNamesInv(self.funcspec.auxvars)
        elif querykey == 'vardomains':
            result = {}
            for varname, var in self.variables.items():
                result[self._FScompatibleNamesInv(varname)] = \
                               var.depdomain
        elif querykey == 'pardomains':
            result = {}
            for parname, pardom in self.parameterDomains.items():
                result[self._FScompatibleNamesInv(parname)] = \
                               pardom
        elif querykey == 'abseps':
            result = self._abseps
        return result

    def get(self, key):
        """For API compatibility with ModelInterface: get will make a copy of
        the key and pass it through the inverse FuncSpec-compatible name map.
        """
        return self._FScompatibleNamesInv(getattr(self, key))


    def haveJacobian(self):
        """Default method. Can be overridden by subclasses."""
        return False


    def haveJacobian_pars(self):
        """Default method. Can be overridden by subclasses."""
        return False


    def info(self, verbose=1):
        print(self._infostr(verbose))


    def _kw_process_dispatch(self, keys, kw):
        # compile funcspec arguments by processing init keys
        # make ignorespecial initially empty so that it can safely be
        # extended by its own dispatch method and by process_system
        fs_args = {'name': self.name,
                   'ignorespecial': []}
        # make sure to do varspecs first in case of FOR macros
        self._kw_process_varspecs(kw, fs_args)
        for key in keys:
            if key == 'varspecs':
                # already did it
                continue
            f = getattr(self, '_kw_process_'+key)
            # f can update fs_args in place
            f(kw, fs_args)
        return fs_args

    def _kw_process_varspecs(self, kw, fs_args):
        if 'varspecs' in kw:
            for varname, varspec in kw['varspecs'].items():
                if not isinstance(varname, (str, QuantSpec,
                                            Quantity)):
                    print("Expected string, QuantSpec, or Quantity to name variable, got type %s" %(type(varname)))
                    raise PyDSTool_TypeError("Invalid type for Variable name: %s"%str(varname))
                if not isinstance(varspec, (str, QuantSpec,
                                            Quantity)):
                    print("Expected string, QuantSpec, or Quantity definition for %s, got type %s" %(varname, type(varspec)))
                    raise PyDSTool_TypeError("Invalid type for Variable %s's specification."%varname)
            self.foundKeys += 1
            fs_args['varspecs'] = \
                    self._FScompatibleNames(ensureStrArgDict(kw['varspecs']))
        else:
            raise PyDSTool_KeyError("Keyword 'varspecs' missing from "
                                    "argument")
        all_vars = []
        # record number of variables defined by macros for FuncSpec checks
        fs_args['_for_macro_info'] = args(totforvars=0, numfors=0, varsbyforspec={})
        for specname, specstr in fs_args['varspecs'].items():
            if not '[' in specname:
                # record non-FOR variables as identity mapping
                all_vars.append(specname)
                fs_args['_for_macro_info'].varsbyforspec[specname] = [specname]
                continue
            # assume this will be a FOR macro (FuncSpec will check properly later)
            assert specstr[:4] == 'for(', ('Expected `for` macro when '
                        'square brackets used in name definition')
            # read contents of braces
            ok, arglist, nargs = readArgs(specstr[3:])
            if not ok:
                raise ValueError('Error finding '
                                 'arguments applicable to `for` '
                                 'macro')
            rootstr = specname[:specname.find('[')]
            assert len(arglist) == 4, ('Wrong number of arguments passed '
                                   'to `for` macro. Expected 4')
            ilo = int(arglist[1])
            ihi = int(arglist[2])
            new_vars = [rootstr+str(i) for i in range(ilo,ihi+1)]
            all_vars.extend(new_vars)
            fs_args['_for_macro_info'].numfors += 1
            fs_args['_for_macro_info'].totforvars += (ihi-ilo+1)
            fs_args['_for_macro_info'].varsbyforspec[specname] = new_vars
        # Temporary record of all var names, will be deleted before finalizing
        # class initialization
        self.__all_vars = all_vars

    def _kw_process_tdomain(self, kw, fs_args):
        if 'tdomain' in kw:
            self.tdomain = kw['tdomain']
            if not self._is_domain_ordered(self.tdomain[0], self.tdomain[1]):
                print("Time domain specified: [%s, %s]"%(self.tdomain[0],
                                                         self.tdomain[1]))
                raise PyDSTool_ValueError("tdomain values must be in order of "
                                "increasing size")
            self.foundKeys += 1
        else:
            self.tdomain = [-np.Inf, np.Inf]

    def _kw_process_ttype(self, kw, fs_args):
        # e.g. for map system
        if 'ttype' in kw:
            try:
                self.indepvartype = _num_equivtype[kw['ttype']]
            except KeyError:
                raise TypeError('Invalid ttype: %s'%str(kw['ttype']))
            self.foundKeys += 1
        else:
            self.indepvartype = float

    def _kw_process_tdata(self, kw, fs_args):
        # set tdomain first
        if 'tdata' in kw:
            self.tdata = kw['tdata']
            if self.tdata[0] >= self.tdata[1]:
                raise PyDSTool_ValueError("tdata values must be in order of "
                                          "increasing size")
            # _tdata is made into list to be consistent with
            # other uses of it in other Generators...
            if self.tdomain[0] > self.tdata[0]:
                raise ValueError('tdata cannot be specified below smallest '\
                      'value in tdomain\n (possibly due to uncertain'\
                      'bounding)')
            if self.tdomain[1] < self.tdata[1]:
                raise ValueError('tdata cannot be specified above largest '\
                      'value in tdomain\n (possibly due to uncertain '\
                      'bounding)')
            self.foundKeys += 1
        else:
            self.tdata = self.tdomain  # default needed

    def _kw_process_tstep(self, kw, fs_args):
        # requires self.indepvartype (e.g. for map system)
        if 'tstep' in kw:
            self.tstep = kw['tstep']
            if self.tstep > self.tdata[1]-self.tdata[0]:
                raise PyDSTool_ValueError('tstep too large')
            if compareNumTypes(self.indepvartype, _all_int) and round(self.tstep) != self.tstep:
                raise PyDSTool_ValueError('tstep must be an integer for integer ttype')
            self.foundKeys += 1
        else:
            if compareNumTypes(self.indepvartype, _all_int):
                # default to 1 for integer types
                self.tstep = 1
            else:
                # no reasonable default - so raise error
                raise PyDSTool_KeyError('tstep key needed for float ttype')

    def _kw_process_inputs(self, kw, fs_args):
        if 'inputs' in kw:
            inputs = copy(kw['inputs'])
            if isinstance(inputs, Trajectory):
                for n in inputs.variables:
                    if n in mathNameMap or n in allmathnames:
                        raise ValueError("Input name {} clash with built-in math/scipy name".format(n))
                # extract the variables
                self.inputs.update(self._FScompatibleNames(inputs.variables))
            elif isinstance(inputs, Variable):
                if inputs.name in mathNameMap or input.name in allmathnames:
                    raise ValueError("Input name {} clash with built-in math/scipy name".format(n))
                self.inputs.update({self._FScompatibleNames(inputs.name): \
                                    inputs})
            elif isinstance(inputs, Pointset):
                # turn into Variables with linear interpoolation between
                # independent variable values
                for n in inputs.coordnames:
                    if n in mathNameMap or n in allmathnames:
                        raise ValueError("Input name {} clash with built-in math/scipy name".format(n))
                    x_array = inputs[n]
                    nFS = self._FScompatibleNames(n)
                    self.inputs[nFS] = \
                        Variable(interp1d(inputs.indepvararray,
                                                       x_array), 't',
                                         Interval(nFS, float, extent(x_array),
                                                  abseps=self._abseps),
                                         name=n)  # keep original name here
            elif isinstance(inputs, dict):
                for n in inputs.keys():
                    if n in mathNameMap or n in allmathnames:
                        raise ValueError("Input name {} clash with built-in math/scipy name".format(n))
                self.inputs.update(self._FScompatibleNames(inputs))
                # ensure values are Variables or Pointsets
                for k, v in self.inputs.items():
                    if not isinstance(v, Variable):
                        try:
                            self.inputs[k]=Variable(v)
                        except:
                            raise TypeError("Invalid specification of inputs")
            else:
                raise TypeError("Invalid specification of inputs")
            self._register(self.inputs)
            self.foundKeys += 1
            # only signal that _extInputsChanged if there are actually some
            # defined, e.g. inputs may be formally present in the keys but in
            # fact unused
            self._extInputsChanged = (self.inputs != {})
            fs_args['inputs'] = list(self.inputs.keys())
        else:
            self._extInputsChanged = False

    def _kw_process_ics(self, kw, fs_args):
        if 'ics' in kw:
            self._xdatadict = {}
            for k, v in dict(kw['ics']).items():
                self._xdatadict[self._FScompatibleNames(str(k))] = ensurefloat(v)
            self.initialconditions = self._xdatadict.copy()
            unspecd = remain(self._xdatadict.keys(), self.__all_vars)
            if unspecd != []:
                    # ics were declared for variables not in varspecs
                    raise ValueError("Missing varspec entries for declared ICs: " + str(unspecd))
            for name in remain(self.__all_vars,
                               self._xdatadict.keys()):
                self.initialconditions[name] = np.NaN
            self.foundKeys += 1
        else:
            self._xdatadict = {}
            for name in self.__all_vars:
                self.initialconditions[name] = np.NaN

    def _kw_process_allvars(self, kw, fs_args):
        for varname in self.__all_vars:
            if varname in mathNameMap or varname in allmathnames:
                raise ValueError("Var name {} clash with built-in math/scipy name".format(varname))
        if 'auxvars' in kw:
            assert 'vars' not in kw, ("Cannot use both 'auxvars' and 'vars' "
                                      "keywords")
            if isinstance(kw['auxvars'], list):
                auxvars = self._FScompatibleNames([str(v) for v in kw['auxvars']])
            else:
                auxvars = self._FScompatibleNames([str(kw['auxvars'])])
            vars = remain(self.__all_vars, auxvars)
            self.foundKeys += 1
        elif 'vars' in kw:
            assert 'auxvars' not in kw, \
                   "Cannot use both 'auxvars' and 'vars' keywords"
            if isinstance(kw['vars'], list):
                vars = self._FScompatibleNames([str(v) for v in kw['vars']])
            else:
                vars = self._FScompatibleNames([str(kw['vars'])])
            auxvars = remain(self.__all_vars, vars)
            self.foundKeys += 1
        else:
            # default is that all are considered regular vars
            auxvars = []
            vars = self.__all_vars
        # vars will never have any macro spec names
        fs_args['vars'] = vars
        self.dimension = len(vars)
        if auxvars != []:
            fs_args['auxvars'] = auxvars

    def _kw_process_xtype(self, kw, fs_args):
        # requires varspecs to have been set
        # default types are float
        self.xtype = {}
        if 'xtype' in kw:
            xts = kw['xtype']
            for name_temp, xt in dict(xts).items():
                if compareNumTypes(xt, _all_int):
                    xt_actual = int
                elif compareNumTypes(xt, _all_float):
                    xt_actual = float
                else:
                    raise TypeError("Invalid variable type %s"%str(xt))
                name = self._FScompatibleNames(name_temp)
                if name[-1] == ']':
                    # for macro -- FuncSpec.py will double check for correct syntax
                    base = name[:name.index('[')]
                    # pull out everything in parentheses
                    for_spec = fs_args['varspecs'][name][4:-1].replace(' ', '').split(',')
                    for name_i in range(int(for_spec[1]), int(for_spec[2])+1):
                        self.xtype[base+str(name_i)] = xt_actual
                else:
                    self.xtype[name] = xt_actual
            for name in remain(fs_args['varspecs'].keys(), self.xtype.keys()):
                self.xtype[name] = float
            self.foundKeys += 1
        else:
            for name in fs_args['varspecs']:
                if name[-1] == ']':
                    # for macro -- FuncSpec.py will double check for correct syntax
                    base = name[:name.index('[')]
                    # pull out everything in parentheses
                    for_spec = fs_args['varspecs'][name][4:-1].replace(' ', '').split(',')
                    for name_i in range(int(for_spec[1]), int(for_spec[2])+1):
                        self.xtype[base+str(name_i)] = float
                else:
                    self.xtype[name] = float

    def _is_domain_ordered(self, left_bound, right_bound):
        try:
            return left_bound <= right_bound
        except TypeError:
            # non-numeric types are unorderable
            if isinstance(left_bound, QuantSpec):
                return self._is_domain_ordered(
                    float(str(left_bound)), float(str(right_bound)))
            return True

    def _kw_process_xdomain(self, kw, fs_args):
        if 'xdomain' in kw:
            self.xdomain = {}
            for k, v in dict(kw['xdomain']).items():
                name = self._FScompatibleNames(str(k))
                if isinstance(v, _seq_types):
                    assert len(v) == 2, \
                           "Invalid size of domain specification for "+name
                    if self._is_domain_ordered(v[0], v[1]):
                        self.xdomain[name] = copy(v)
                    else:
                        raise PyDSTool_ValueError('xdomain values must be in'
                                                  'order of increasing size')
                elif isinstance(v, _num_types):
                    self.xdomain[name] = [v, v]
                else:
                    raise PyDSTool_TypeError('Invalid type for xdomain spec'
                                             ' '+name)
            for name in remain(fs_args['varspecs'].keys(), self.xdomain.keys()):
                if name[-1] == ']':
                    # for macro -- FuncSpec.py will double check for correct syntax
                    base = name[:name.index('[')]
                    # pull out everything in parentheses
                    for_spec = fs_args['varspecs'][name][4:-1].replace(' ', '').split(',')
                    for name_i in range(int(for_spec[1]), int(for_spec[2])+1):
                        self.xdomain[base+str(name_i)] = [-np.Inf, np.Inf]
                else:
                    self.xdomain[name] = [-np.Inf, np.Inf]
            self.foundKeys += 1
        else:
            self.xdomain = {}
            for name in fs_args['varspecs']:
                if name[-1] == ']':
                    # for macro -- FuncSpec.py will double check for correct syntax
                    base = name[:name.index('[')]
                    # pull out everything in parentheses
                    for_spec = fs_args['varspecs'][name][4:-1].replace(' ', '').split(',')
                    for name_i in range(int(for_spec[1]), int(for_spec[2])+1):
                        self.xdomain[base+str(name_i)] = [-np.Inf, np.Inf]
                else:
                    self.xdomain[name] = [-np.Inf, np.Inf]

    def _kw_process_reuseterms(self, kw, fs_args):
        if 'reuseterms' in kw:
            self.foundKeys += 1
            fs_args['reuseterms'] = kw['reuseterms']

    def _kw_process_ignorespecial(self, kw, fs_args):
        if 'ignorespecial' in kw:
            self.foundKeys += 1
            fs_args['ignorespecial'].extend(kw['ignorespecial'])

    def _kw_process_algparams(self, kw, fs_args):
        if 'algparams' in kw:
            self.algparams = copy(kw['algparams'])
            self.foundKeys += 1
        else:
            self.algparams = {}

    def _kw_process_pars(self, kw, fs_args):
        if 'pars' in kw:
            self.pars = {}
            if isinstance(kw['pars'], list):
                # may be a list of symbolic definitions
                for p in kw['pars']:
                    if p.name in mathNameMap or p.name in allmathnames:
                        raise ValueError("Param name {} clash with built-in math/scipy name".format(p.name))
                    try:
                        self.pars[self._FScompatibleNames(p.name)] = p.tonumeric()
                    except (AttributeError, TypeError):
                        raise TypeError("Invalid parameter symbolic definition")
            else:
                for k, v in dict(kw['pars']).items():
                    kstr = str(k)
                    if kstr in mathNameMap or kstr in allmathnames:
                        raise ValueError("Param name {} clash with built-in math/scipy name".format(k))
                    self.pars[self._FScompatibleNames(kstr)] = ensurefloat(v)
            fs_args['pars'] = list(self.pars.keys())
            self._register(self.pars)
            self.foundKeys += 1
        self.numpars = len(self.pars)

    def _kw_process_pdomain(self, kw, fs_args):
        if 'pdomain' in kw:
            if self.pars:
                self.pdomain = {}
                for k, v in dict(kw['pdomain']).items():
                    assert len(v) == 2, \
                               "Invalid size of domain specification for "+k
                    self.pdomain[self._FScompatibleNames(str(k))] = v
                for name in self.pdomain:
                    if not self._is_domain_ordered(self.pdomain[name][0], self.pdomain[name][1]):
                        raise PyDSTool_ValueError('pdomain values must be in order of increasing size')
                for name in remain(self.pars.keys(), self.pdomain.keys()):
                    self.pdomain[name] = [-np.Inf, np.Inf]
                self.foundKeys += 1
            else:
                raise ValueError('Cannot specify pdomain because no pars declared')
        else:
            if self.pars:
                self.pdomain = {}
                for pname in self.pars:
                    self.pdomain[pname] = [-np.Inf, np.Inf]
        if self.pars:
            self.parameterDomains = {}
            for pname in self.pdomain:
                self.parameterDomains[pname] = Interval(pname, float,
                                                        self.pdomain[pname],
                                                        self._abseps)
                try:
                    cval = self.parameterDomains[pname].contains(self.pars[pname])
                except KeyError:
                    raise ValueError("Parameter %s is missing a value"%pname)
                if self.checklevel < 3:
                    if cval is not notcontained:
                        if cval is uncertain and self.checklevel == 2:
                            print('Warning: Parameter value at bound')
                    else:
                        print("%r not in %r" % (self.pars[pname], self.parameterDomains[pname].get()))
                        raise PyDSTool_ValueError('Parameter %s: value out of bounds'%pname)
                else:
                    if cval is uncertain:
                        raise PyDSTool_UncertainValueError('Parameter %s: value at bound'%pname)
                    elif cval is notcontained:
                        raise PyDSTool_ValueError('Parameter %s: value out of bounds'%pname)

    def _kw_process_fnspecs(self, kw, fs_args):
        if 'fnspecs' in kw:
            fnspec_dict = ensureStrArgDict(kw['fnspecs'])
            for k, v in fnspec_dict.items():
                if k in mathNameMap or k in allmathnames:
                    raise ValueError("Aux function name {} clash with built-in math/scipy name".format(k))
            fs_args['fnspecs'] = fnspec_dict
            self.foundKeys += 1

    def _kw_process_target(self, kw, fs_args):
        fs_args['targetlang'] = theGenSpecHelper(self).lang
        if 'compiler' in kw:
            if fs_args['targetlang'] == 'python':
                print("Warning: redundant option 'compiler' for python target")
            self._compiler = kw['compiler']
            self.foundKeys += 1
        elif fs_args['targetlang'] != 'python':
            osname = os.name
            # os-specific defaults for C compiler
            if osname == 'nt':
                self._compiler = 'mingw32'
            elif osname == 'mac':
                self._compiler = 'mwerks'
            elif osname == 'posix' or osname == 'unix':
                self._compiler = 'unix'
            elif osname == 'os2emx':
                self._compiler = 'emx'
            else:
                self._compiler = ''
        else:
            self._compiler = ''

    def _kw_process_vfcodeinserts(self, kw, fs_args):
        if 'vfcodeinsert_start' in kw:
            fs_args['codeinsert_start'] = kw['vfcodeinsert_start']
            self.foundKeys += 1
        if 'vfcodeinsert_end' in kw:
            fs_args['codeinsert_end'] = kw['vfcodeinsert_end']
            self.foundKeys += 1

    def _kw_process_system(self, kw, fs_args):
        # for python-based solvers, esp. map system
        if 'system' in kw:
            self._solver = kw['system']
            try:
                fs_args['ignorespecial'].append(self._solver.name)
            except:
                raise TypeError("Invalid solver system provided")
            self.foundKeys += 1
            if self.pars:
                # automatically pass par values on to embedded system
                # when Rhs called
                parlist = list(self.pars.keys())
                parstr = "".join(["'%s': %s, "%(parname,parname) \
                                  for parname in parlist])
            else:
                parstr = ""
            if 'codeinsert_start' in fs_args:
                fs_args['codeinsert_start'] = \
                    '    %s.set(pars={%s})\n'%(self._solver.name, parstr) \
                    + fs_args['codeinsert_start']
            else:
                fs_args['codeinsert_start'] = \
                    '    %s.set(pars={%s})\n'%(self._solver.name, parstr)
        else:
            self._solver = None


    def _infostr(self, verbose=1):
        """Return detailed information about the Generator
        specification."""

        if verbose == 0:
            outputStr = "Generator "+self.name
        else:
            outputStr = '**************************************************'
            outputStr += '\n           Generator  '+self.name
            outputStr +='\n**************************************************'
            outputStr +='\nType : ' + className(self)
            outputStr +='\nIndependent variable interval: ' + str(self.indepvariable.depdomain)
            outputStr +='\nGlobal t0 = ' + str(self.globalt0)
            outputStr +='\nInterval endpoint check level = ' + str(self.checklevel)
            outputStr +='\nDimension = ' + str(self.dimension)
            outputStr +='\n'
            if isinstance(self.funcspec, FuncSpec):
                outputStr += self.funcspec._infostr(verbose)
        if verbose == 2:
            outputStr += '\nVariables` validity intervals:'
            for v in self.variables.values():
                outputStr += '\n  ' + str(v.depdomain)
            if self.eventstruct is not None:
                outputStr += '\nEvents defined:'
                outputStr += '\n  ' + str(list(self.eventstruct.events.keys()))
        if self._modeltag is not None:
            outputStr += '\nAssociated Model: ' + self._modeltag.name
        if verbose > 0:
            outputStr += '\n'
        return outputStr


    def showEventSpec(self):
        if self.eventstruct is not None:
            for evname, ev in self.eventstruct.events.items():
                print(evname + ":\n" + ev._funcstr)
                print("\n")


    def showSpec(self):
        print(self.funcspec.spec[0])


    def showAuxSpec(self):
        print(self.funcspec.auxspec[0])


    def showAuxFnSpec(self, auxfnname=None):
        if auxfnname is None:
            retdict = {}
            for aname, aspec in self.funcspec.auxfns.items():
                retdict[aname] = aspec[0]
            info(retdict)
        else:
            try:
                print(self.funcspec.auxfns[auxfnname][0])
            except KeyError:
                raise NameError("Aux function %s not found"%auxfnname)


    def __repr__(self):
        return self._infostr(verbose=0)


    __str__ = __repr__


    def validateSpec(self):
        try:
            assert self.funcspec
            assert self.dimension > 0
            try:
                num_auxvars = len(self.funcspec.auxvars)
            except AttributeError:
                # no auxvars
                num_auxvars = 0
            assert len(self.variables) == self.dimension + num_auxvars
            # don't assert self.pars because not all systems need them
            assert self.indepvariable.name == 't'
            assert self.checklevel in range(4)
            #  check that all names in individual dicts are all in _registry
            if self.pars:
                for name in self.pars:
                    assert isinstance(self.pars[name], _num_types)
                    assert type(self.pars[name]) == self._registry[name]
            if self.inputs:
                for subjectname, obj in self.inputs.items():
                    # test for containment of input's interval in independent
                    # variable interval
                    # (use checklevel = 1 for this o/w could get errors)
                    assert self.contains(obj.indepdomain,
                                         self.indepvariable.indepdomain, 0)
                    # test that types entered in registry are still correct
                    assert type(obj) == self._registry[subjectname]
            for name in self.variables:
                assert self.variables[name].__class__ == self._registry[name]
            dummy = self.indepvariable(self.tdata[0]) # exception if this call is ill-defined
            # check consistency with FuncSpec type of self.funcspec
            # (unnecessary for dictionary version of FuncSpec)
            if isinstance(self.funcspec, FuncSpec):
                varnames = list(self.variables.keys())
                try:
                    auxvars = self.funcspec.auxvars
                except AttributeError:
                    auxvars = []
                fsvars = self.funcspec.vars + auxvars
                if len(varnames) > 1:
                    varnames.sort()
                    fsvars.sort()
                    assert varnames == fsvars, ('Inconsistency with funcspec '
                                                'variable names')
                else:
                    assert varnames == fsvars
                parnames = list(self.pars.keys())
                fspars = self.funcspec.pars
                if len(parnames) > 1:
                    parnames.sort()
                    fspars.sort()
                    assert parnames == fspars, ('Inconsistency with funcspec '
                                                'parameter names')
                else:
                    assert parnames == fspars
                if self.inputs:
                    inputnames = list(self.inputs.keys())
                    fsinputs = self.funcspec.inputs
                    if len(inputnames) > 1:
                        inputnames.sort()
                        fsinputs.sort()
                        assert inputnames == fsinputs, ('Inconsistency with funcspec'
                                                        ' input names')
                    else:
                        assert inputnames == fsinputs
            else:
                assert len(self.funcspec) == self.dimension
            # check that all aux function specs in events made it into funcspec
            # (situation not caught on Vode, and leads to weird errors with C
            # integrators) -- only check names, not definitions
            try:
                fnspecs = self.funcspec.auxfns.keys()
            except AttributeError:
                # no aux fns declared
                pass
            else:
                for ename, ev in self.eventstruct.events.items():
                    for fname in remain(ev._fnspecs.keys(), ('if', 'initcond',
                                                     'heav', 'globalindepvar',
                                                     'getindex', 'getbound')):
                        assert fname in fnspecs, "All aux functions from " + \
                               "events must be declared to FuncSpec too"

        except:
            print('Invalid system specification')
            raise


    # call this after all expected keywords have been processed
    def checkArgs(self, kw):
        if len(kw) == self.foundKeys:
            for name in self._needKeys:
                if name not in kw:
                    raise PyDSTool_KeyError('Necessary key missing: ' + name)
            for name in kw:
                if name not in self._needKeys + self._optionalKeys:
                    raise PyDSTool_KeyError('Key name ' + name + ' is invalid')
        else:
            print('Keywords supplied:\n\t' + str(list(kw.keys())))
            print('# keywords found: ' + str(self.foundKeys))
            print('Needed:\n\t' + str(self._needKeys))
            print('Optional:\n\t' + str(self._optionalKeys))
            raise PyDSTool_KeyError('Invalid keyword arguments for this class')
        del self.foundKeys


    def _set_for_hybrid_DS(self, state):
        """Internal method for indicating whether this Generator is currently
        being used as part of a hybrid dybnamical system calculation"""
        self._for_hybrid_DS = state


    def _register(self, items):
        """_register names and types of sub-system variables (including
        Generator variables), pars and external inputs.

        Names must be unique for the Generator.
        """

        if isinstance(items, dict):
            # for parameter and variable dictionaries
            for name, v in items.items():
                if isinstance(self.funcspec, FuncSpec):
                    assert name in self.funcspec.vars \
                       or name in self.funcspec.auxvars \
                       or name in self.funcspec.pars \
                       or name in self.funcspec.inputs, \
                       ("Generator = '"
                       +self.name+"': name "+name+" not found in "
                       "functional specification declaration")
                if name not in self._registry:
                    self._registry[name] = type(v)
                else:
                    raise ValueError('The name `' + name + '` of type `'
                                       + type(v).__name__ +
                                       '` already exists in the registry')
                if isinstance(v, Variable) and \
                   name in self.variables or name in self.inputs:
                    self._callnames.append(name)
            self._callnames.sort()
        elif isinstance(items, Variable) and items.name == 't':
            # for self.indepvariable
            if items.name not in self._registry:
                self._registry[items.name] = type(items)
            else:
                raise ValueError('The reserved name `t` has already'
                                   ' been declared to the registry')
        else:
            raise TypeError('Expected dictionary or independent variable in '
                              'argument to _register()')

    def _kw_process_events(self, kw):
        # Only call once funcspec built
        #
        # Holder and interface for events
        self.eventstruct = EventStruct()
        if 'enforcebounds' in kw:
            if 'activatedbounds' in kw:
                ab = kw['activatedbounds']
                self.foundKeys += 1
            else:
                ab = None
            if kw['enforcebounds']:
                self._makeBoundsEvents(precise=True, activatedbounds=ab)
            self.foundKeys += 1
        if 'events' in kw:
            self._addEvents(kw['events'])
            self.foundKeys += 1


    def _addEvents(self, evs):
        if isinstance(evs, list):
            for e in copy(evs):
                self.eventstruct.add(e)
        elif isinstance(evs, Event):
            # singleton
            self.eventstruct.add(evs)
        else:
            raise TypeError('Unsupported type of argument for event '
                              'structure')


    def _makeBoundsEvents(self, precise=True, eventtol=1e-6,
                          activatedbounds=None):
        events = []
        # pars + vars (pars used only by PyCont during continuation)
        alldoms = copy(self.pdomain)
        alldoms.update(self.xdomain)
        if self._eventPars != []:
            # This exclusion doesn't work at the moment.
            #            nonEvtPars = remain(self.funcspec.pars, self._eventPars)
            nonEvtPars = self.funcspec.pars
        else:
            nonEvtPars = self.funcspec.pars
        allnames = self.funcspec.vars + nonEvtPars
        if activatedbounds in (None,{}):
            activatedbounds = {}.fromkeys(allnames, (False,False))
        for xname, xdom in alldoms.items():
            if xname not in allnames:
                # don't make bound constraints for non-state variables
                continue
            xdlo = xdom[0]
            xdhi = xdom[1]
            evname = xname+"_domlo"
            evargs = {'term': True,
                      'precise': precise,
                      'name': evname,
                      'eventtol': eventtol,
                      'eventdelay': eventtol*10,
                      'eventinterval': eventtol*20,
                      'xdomain': self.xdomain,
                      'pdomain': self.pdomain}
            try:
                evargs['active'] = activatedbounds[xname][0] and isfinite(xdlo)
            except KeyError:
                evargs['active'] = False
            evstr = xname + '-' + 'getbound("%s", 0)'%xname
            ev = Events.makeZeroCrossEvent(evstr, -1, evargs, [xname],
                                       targetlang=self.funcspec.targetlang,
                                       reuseterms=self.funcspec.reuseterms)
            events.append(ev)
            evname = xname+"_domhi"
            evargs = {'term': True,
                      'precise': precise,
                      'name': evname,
                      'eventtol': eventtol,
                      'eventdelay': eventtol*10,
                      'eventinterval': eventtol*20,
                      'xdomain': self.xdomain,
                      'pdomain': self.pdomain}
            try:
                evargs['active'] = activatedbounds[xname][1] and isfinite(xdhi)
            except KeyError:
                evargs['active'] = False
            evstr = xname + '-' + 'getbound("%s", 1)'%xname
            ev = Events.makeZeroCrossEvent(evstr, 1, evargs, [xname],
                                       targetlang=self.funcspec.targetlang,
                                       reuseterms=self.funcspec.reuseterms)
            events.append(ev)
        if events != []:
            self._addEvents(events)


    def set(self, **kw):
        """Set generic parameters."""
        # Note: globalt0 may be unused by many classes of Generator
        if len(kw) > 0:
            if 'globalt0' in kw:
                self.globalt0 = kw['globalt0']
                try:
                    self.eventstruct.setglobalt0(self.globalt0)
                except AttributeError:
                    # no events present
                    pass
            if 'checklevel' in kw:
                self.checklevel = kw['checklevel']
                if hasattr(self, 'algparams'):
                    self.algparams['checkBounds'] = kw['checklevel']
            if 'abseps' in kw:
                self._abseps = kw['abseps']
            if remain(kw.keys(), ['globalt0', 'checklevel', 'abseps']) != []:
                raise PyDSTool_KeyError('Invalid keywords passed')


    def setEventICs(self, ics, gt0=0):
        """Set initialconditions attribute of all generator's events, in
        case event uses auxiliary functions that access this information."""
        try:
            evs = list(self.eventstruct.events.values())
        except AttributeError:
            # no events present
            pass
        else:
            for ev in evs:
                ev.initialconditions = self._FScompatibleNames(ics)
                ev.globalt0 = gt0


    def resetEventTimes(self):
        try:
            self.eventstruct.resetEvtimes()
        except AttributeError:
            # no events present
            pass

    def resetEvents(self, state=None):
        """Reset any high level (Python) events in Generator"""
        try:
            # time is OK to be 0 here, it will be overwritten before use anyway
            # e.g. by VODE or map system
            self.eventstruct.resetHighLevelEvents(0, state=state)
        except AttributeError:
            # no events present
            pass

    # Auxiliary functions for user-defined code to call

    def _auxfn_globalindepvar(self, parsinps, t):
        return self.globalt0 + t

    def _auxfn_initcond(self, parsinps, varname):
        return self.initialconditions[varname]

    def _auxfn_heav(self, parsinps, x):
        if x>0:
            return 1
        else:
            return 0

    def _auxfn_if(self, parsinps, c, e1, e2):
        if c:
            return e1
        else:
            return e2

    def _auxfn_getindex(self, parsinps, varname):
        return self._var_namemap[varname]


    def _generate_ixmaps(self, gentypes=None):
        """Generate indices mapping.

        This creates a mapping from the names of variables,
        pars and inputs, to indices in the arrays used for
        refering to the internal (dynamic) call methods."""

        if gentypes is not None:
            if isinstance(gentypes, str):
                gentypes = [gentypes]
            for s in gentypes:
                assert s in ['variables', 'inputs', 'pars'], \
                       ('Incorrect type string for _generate_ixmaps')
        else:
            # default to all
            gentypes = ['variables', 'inputs', 'pars']
        # ixmap (list) : int -> str
        # namemap (dict) : str -> int
        if 'variables' in gentypes:
            self._var_ixmap = sortedDictKeys(self.variables,
                                             self.funcspec.vars)
            self._var_namemap = invertMap(self._var_ixmap)
        if 'pars' in gentypes:
            if self.pars:
                self._parameter_ixmap = sortedDictKeys(self.pars)
                self._parameter_namemap = invertMap(self._parameter_ixmap)
            else:
                self._parameter_ixmap = []
                self._parameter_namemap = {}
        if 'inputs' in gentypes:
            if self.inputs:
                self._inputs_ixmap = \
                    sortedDictKeys(self.inputs)
                self._inputs_namemap = invertMap(self._inputs_ixmap)
            else:
                self._inputs_ixmap = []
                self._inputs_namemap = {}


    def contains(self, interval, val, checklevel=2):
        """Interval containment test"""
        # NB. val may be another interval
        if checklevel == 0:
            # level 0 -- no bounds checking at all
            # code should avoid calling this function with checklevel = 0
            # if possible, but this case is left here for completeness and
            # consistency
            return True
        elif checklevel == 2:
            # level 2 -- warn on uncertain and continue
            testresult = interval.contains(val)
            if testresult is contained:
                return True
            elif testresult is uncertain:
                self.diagnostics.warnings.append((W_UNCERTVAL, (val,interval)))
                return True
            else:
                return False
        elif checklevel == 1:
            # level 1 -- ignore uncertain cases (treat as contained)
            if interval.contains(val) is not notcontained:
                return True
            else:
                return False
        else:
            # level 3 -- exception will be raised for uncertain case
            if val in interval:
                return True
            else:
                return False


    # Methods for pickling protocol
    def __getstate__(self):
        d = copy(self.__dict__)
        for fname, finfo in self._funcreg.items():
            try:
                del d[fname]
            except KeyError:
                pass
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._funcreg != {}:
            self.addMethods()


    def __del__(self):
        # delete object-specific class methods etc. before deleting
        # to avoid crowding namespace
        try:
            for fname, finfo in self._funcreg.items():
                try:
                    delattr(eval(finfo[0]), fname)
                except AttributeError:
                    # nothing to delete, fname no longer an attribute
                    pass
                except (NameError, SyntaxError):
                    # NameError:
                    # not sure what happens here, but some other names
                    # may be deleted before all references to them have
                    # been deleted, but it's very non-fatal to ignore.
                    # SyntaxError:
                    # Also unsure, but sometimes finfo[0] is None
                    # (rather than the usual 'self') so cannot evaluate
                    # to object.
                    pass
            if hasattr(self, 'eventstruct'):
                if self.eventstruct is not None:
                    self.eventstruct.__del__()
            if self.indepvariable is not None:
                del self.indepvariable
            for v in self.variables.values():
                v.__del__()
            if hasattr(self, 'inputs'):
                for v in self.inputs.values():
                    v.__del__()
        except AttributeError:
            # self does not have _funcreg
            pass
        except NameError:
            # see above notes for NameError catch
            pass


    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


#--------------------------------------------------------------------------


class ctsGen(Generator):
    "Abstract class for continuously-parameterized trajectory generators."

    def validateSpec(self):
        # only check that domain is cts, range may be a finite subset of points
        assert isinputcts(self.indepvariable), ("self.indepvariable must be continuously-"
                                         "defined for this class")
        Generator.validateSpec(self)

    def __del__(self):
        Generator.__del__(self)



class discGen(Generator):
    "Abstract class for discretely-parameterized trajectory generators."

    def validateSpec(self):
        assert isdiscrete(self.indepvariable), ("self.indepvariable must be discretely-"
                                         "defined for this class")
        Generator.validateSpec(self)

    def __del__(self):
        Generator.__del__(self)


#--------------------------------------------------------------------------


class GenSpecInfoObj(object):
    # empty class struct for GenSpecHelper
    pass


class GenSpecHelper(object):
    """Generator specification helper - abstract class.

    Used to help ModelConstructor translate abstract model specifications
    into concrete specifications specific to individual Generators."""

    def __init__(self):
        self.gshDB = {}

    def add(self, genClass, symbolMapDict, lang, specType='RHSfuncSpec'):
        genName = className(genClass)
        if genName in self.gshDB:
            raise ValueError("Generator %s has already been declared"%genName)
        else:
            infoObj = GenSpecInfoObj()
            infoObj.genClass = genClass
            infoObj.symbolMap = symbolMapClass(symbolMapDict)
            infoObj.lang = lang
            infoObj.specType = specType
            if issubclass(genClass, ctsGen):
                infoObj.domain = Continuous
            elif issubclass(genClass, discGen):
                infoObj.domain = Discrete
            else:
                raise TypeError("Invalid Generator class")
            self.gshDB[genName] = infoObj

    def __call__(self, subject):
        try:
            if isinstance(subject, str):
                return self.gshDB[subject]
            else:
                return self.gshDB[className(subject)]
        except KeyError:
            raise KeyError("Generator %s was not found in database"%str(subject))

    def __contains__(self, subject):
        return subject in self.gshDB or className(subject) in self.gshDB


def _pollInputs(inputVarList, t, checklevel):
    ilist = []
    try:
        for f in inputVarList:
            f.diagnostics.clearWarnings()
            ilist.append(f(t, checklevel))
    except AssertionError:
        print('External input call has t out of range: t = %f' % (t, ))
        print('Maybe checklevel is 3 and initial time is not', \
                    'completely inside valid time interval')
        raise
    except ValueError:
        print('External input call has value out of range: t = %f' % (t, ))
        print('Check beginning and end time of integration')
        for f in inputVarList:
            if f.diagnostics.hasWarnings():
                print('External input %s out of range:' % f.name)
                print('   t = %r, %s = %r' % (f.diagnostics.warnings[-1][0], f.name, f.diagnostics.warnings[-1][1]))
        raise
    return ilist

global theGenSpecHelper
theGenSpecHelper = GenSpecHelper()
