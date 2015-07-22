# ODEsystem base class
from __future__ import absolute_import, print_function

from .allimports import *
from .baseclasses import ctsGen, theGenSpecHelper, auxfn_container
from PyDSTool.utils import *
from PyDSTool.common import *
from PyDSTool.Variable import Variable, iscontinuous
from PyDSTool.Trajectory import Trajectory
from PyDSTool.Points import Pointset
from PyDSTool.Interval import uncertain

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, array, arange, \
     zeros, float64, int32, transpose, shape
import math, random
import types
from copy import copy, deepcopy
import six


class ODEsystem(ctsGen):
    """Abstract class for ODE system solvers."""
    _validKeys = ['globalt0', 'xdomain', 'tdata', 'tdomain', 'checklevel',
                  'ics', 'pars', 'algparams', 'inputs', 'pdomain', 'abseps',
                  'inputs_t0']
    _needKeys = ctsGen._needKeys + ['varspecs']
    _optionalKeys = ctsGen._optionalKeys + ['tdomain', 'xdomain', 'xtype',
                                            'inputs', 'tdata',
                          'ics', 'events', 'compiler', 'enforcebounds',
                          'activatedbounds', 'checklevel', 'algparams',
                          'auxvars', 'vars', 'pars', 'fnspecs', 'pdomain',
                          'reuseterms', 'vfcodeinsert_start',
                          'vfcodeinsert_end', 'ignorespecial']

    def __init__(self, kw):
        ctsGen.__init__(self, kw)
        self.diagnostics._errmessages[E_COMPUTFAIL] = 'Integration failed'
        # user auxiliary function interface
        self.auxfns = auxfn_container(self)
        dispatch_list = ['varspecs', 'tdomain', 'tdata', 'inputs',
                        'ics', 'allvars', 'xtype', 'xdomain',
                        'reuseterms', 'algparams', 'pars', 'pdomain',
                        'fnspecs', 'target', 'vfcodeinserts', 'ignorespecial']
        # process keys and build func spec
        self.funcspec = RHSfuncSpec(self._kw_process_dispatch(dispatch_list,
                                                              kw))
        self.indepvartype = float
        for v in self.inputs.values():
            if not iscontinuous(v):
                raise ValueError("External inputs for ODE system must be "
                       "continuously defined")
        self._kw_process_events(kw)
        self.checkArgs(kw)
        tindepdomain = Interval('t_domain', self.indepvartype, self.tdomain,
                                self._abseps)
        tdepdomain = Interval('t', self.indepvartype, self.tdata, self._abseps)
        self.indepvariable = Variable(listid, tindepdomain, tdepdomain, 't')
        self._register(self.indepvariable)
        for xname in self.funcspec.vars + self.funcspec.auxvars:
            # Add a temporary dependent variable domain, for validation testing
            # during integration
            self.variables[xname] = Variable(indepdomain=tdepdomain,
                                             depdomain=Interval(xname,
                                                          self.xtype[xname],
                                                          self.xdomain[xname],
                                                          self._abseps))
        self._register(self.variables)
        self._generate_ixmaps()
        # Introduce any python-specified code to the local namespace
        self.addMethods()
        # all registration completed
        self.validateSpec()


    def prepDirection(self, dirn):
        """Common pre-integration tasks go here"""
        if dirn in ['f', 'forward', 1]:
            self._dircode = 1
            continue_integ = False
        elif dirn in ['b', 'backward', -1]:
            self._dircode = -1
            continue_integ = False
        elif dirn in ['c', 'continue', 0]:
            if self.defined and self._solver is not None:
                continue_integ = True
            else:
                # just treat as initial forwards integration
                continue_integ = False
                self._dircode = 1
        else:
            raise ValueError('Invalid direction code in argument')
        return continue_integ


    def addMethods(self):
        """Add Python-specific functions to this object's methods."""

        # Add the auxiliary function specs to this Generator's namespace
        for auxfnname in self.funcspec._pyauxfns:
            fninfo = self.funcspec._pyauxfns[auxfnname]
            if not hasattr(self, fninfo[1]):
                # user-defined auxiliary functions
                # (built-ins are provided explicitly)
                try:
                    exec(fninfo[0])
                except:
                    print('Error in supplied auxiliary function code')
                self._funcreg[fninfo[1]] = ('self', fninfo[0])
                setattr(self, fninfo[1], six.create_bound_method(locals()[fninfo[1]], self))
                # user auxiliary function interface wrapper
                try:
                    uafi_code = self.funcspec._user_auxfn_interface[auxfnname]
                    try:
                        exec(uafi_code)
                    except:
                        print('Error in auxiliary function wrapper')
                        raise
                    setattr(self.auxfns, auxfnname,
                            six.create_bound_method(locals()[auxfnname], self.auxfns))
                    self._funcreg[auxfnname] = ('', uafi_code)
                except KeyError:
                    # not a user-defined aux fn
                    pass
        # Add the spec function to this Generator's namespace if
        # target language is python (otherwise integrator exposes it anyway)
        if self.funcspec.targetlang == 'python':
            fninfo = self.funcspec.spec
            try:
                exec(fninfo[0])
            except:
                print('Error in supplied functional specification code')
                raise
            self._funcreg[fninfo[1]] = ('self', fninfo[0])
            setattr(self, fninfo[1], six.create_bound_method(locals()[fninfo[1]], self))
            # Add the auxiliary spec function (if present) to this
            # Generator's namespace
            if self.funcspec.auxspec != '':
                fninfo = self.funcspec.auxspec
                try:
                    exec(fninfo[0])
                except:
                    print('Error in supplied auxiliary variable code')
                    raise
                self._funcreg[fninfo[1]] = ('self', fninfo[0])
                setattr(self, fninfo[1], six.create_bound_method(locals()[fninfo[1]], self))


    def haveJacobian(self):
        """Report whether ODE system has an explicit user-specified Jacobian
        associated with it."""
        return 'Jacobian' in self.funcspec.auxfns


    def haveJacobian_pars(self):
        """Report whether ODE system has an explicit user-specified Jacobian
        with respect to pars associated with it."""
        return 'Jacobian_pars' in self.funcspec.auxfns


    def haveMass(self):
        """Report whether ODE system has an explicit user-specified mass matrix
        associated with it."""
        return 'massMatrix' in self.funcspec.auxfns


    def checkInitialConditions(self, checkauxvars=False):
        for xname, val in self.initialconditions.items():
            if xname not in self.funcspec.vars and not checkauxvars:
                # auxvars do not need initial conditions unless
                # explicitly requested (e.g. for user call to RHS
                # function of C code vector field before trajectory
                # has been computed)
                continue
            try:
                if not isfinite(val):
                    raise ValueError("Initial condition for "+xname+" has been "
                                    "incorrectly initialized")
            except TypeError:
                print("Found: %r" % val)
                print("of type: %s" % type(val))
                raise TypeError("Invalid type for %s`s initial"%xname \
                                + "condition value")
            if not self.contains(self.variables[xname].depdomain,
                                 val, self.checklevel):
                print("Bounds: %r" % self.variables[xname].depdomain.get())
                print("Variable value: %s" % val)
                raise ValueError("Initial condition for "+xname+" has been "
                                   "set outside of prescribed bounds")


    def set(self, **kw):
        """Set ODE system parameters"""
        if remain(kw.keys(), self._validKeys) != []:
            raise KeyError("Invalid keys in argument")
        if 'globalt0' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, globalt0=kw['globalt0'])
            # set this True so that can adjust the input time arrays
            # to new globalt0
            self._extInputsChanged = True
        if 'checklevel' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, checklevel=kw['checklevel'])
        if 'abseps' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, abseps=kw['abseps'])
        # optional keys for this call are ['pars', 'tdomain', 'ics',
        #   'algparams', 'tdata', 'xdomain', 'pdomain', 'inputs']
        if 'ics' in kw:
            for k_temp, v in kw['ics'].items():
                # str() ensures that Symbolic objects can be passed
                k = str(self._FScompatibleNames(k_temp))
                if k in self.funcspec.vars+self.funcspec.auxvars:
                    self._xdatadict[k] = ensurefloat(v)
                else:
                    raise ValueError('Illegal variable name %s'%k)
            self.initialconditions.update(self._xdatadict)
        tchange = False
        if 'tdata' in kw:
            self.tdata = kw['tdata']
            tchange = True
        if 'tdomain' in kw:
            self.tdomain = kw['tdomain']
            self.indepvariable.indepdomain.set(self.tdomain)
            tchange = True
        if tchange:
            if self.tdomain[0] > self.tdata[0]:
                if self.indepvariable.indepdomain.contains(self.tdata[0]) == uncertain:
                    self.diagnostics.warnings.append((W_UNCERTVAL,
                                                      (self.tdata[0],self.tdomain)))
                else:
                    print('tdata cannot be specified below smallest '\
                          'value in tdomain\n (possibly due to uncertain bounding).'\
                          ' It has been automatically adjusted from %f to %f (difference of %f)\n' % (
                          self.tdata[0], self.tdomain[0], self.tdomain[0]-self.tdata[0]))
                    if self._modeltag:
                        print('Try reducing step size in model.')
                self.tdata[0] = self.tdomain[0]
            if self.tdomain[1] < self.tdata[1]:
                if self.indepvariable.indepdomain.contains(self.tdata[1]) == uncertain:
                    self.diagnostics.warnings.append((W_UNCERTVAL,
                                                      (self.tdata[1],self.tdomain)))
                else:
                    print('tdata cannot be specified above largest '\
                          'value in tdomain\n (possibly due to uncertain bounding).'\
                          ' It has been automatically adjusted from %f to %f (difference of %f)\n' % (
                          self.tdomain[1], self.tdomain[1], self.tdata[1]-self.tdomain[1]))
                    if self._modeltag:
                        print('Try reducing step size in model.')
                self.tdata[1] = self.tdomain[1]
            self.indepvariable.depdomain.set(self.tdata)
        if 'xdomain' in kw:
            for k_temp, v in kw['xdomain'].items():
                k = str(self._FScompatibleNames(k_temp))
                if k in self.funcspec.vars+self.funcspec.auxvars:
                    if isinstance(v, _seq_types):
                        assert len(v) == 2, \
                               "Invalid size of domain specification for "+k
                        if v[0] >= v[1]:
                            raise PyDSTool_ValueError('xdomain values must be'
                                                      'in order of increasing '
                                                      'size')
                        else:
                            self.xdomain[k] = copy(v)
                    elif isinstance(v, _num_types):
                        self.xdomain[k] = [v, v]
                    else:
                        raise PyDSTool_TypeError('Invalid type for xdomain spec'
                                                 ' '+k)
                else:
                    raise ValueError('Illegal variable name')
                try:
                    self.variables[k].depdomain.set(v)
                except TypeError:
                    raise TypeError('xdomain must be a dictionary of variable'
                                      ' names -> valid interval 2-tuples or '
                                      'singletons')
                try:
                    evs = self.eventstruct.events.values()
                except AttributeError:
                    evs = []
                for ev in evs:
                    ev.xdomain[k] = self.xdomain[k]
        if 'pdomain' in kw:
            for k_temp, v in kw['pdomain'].items():
                k = str(self._FScompatibleNames(k_temp))
                if k in self.funcspec.pars:
                    if isinstance(v, _seq_types):
                        assert len(v) == 2, \
                               "Invalid size of domain specification for "+k
                        if v[0] >= v[1]:
                            raise PyDSTool_ValueError('pdomain values must be'
                                                      'in order of increasing '
                                                      'size')
                        else:
                            self.pdomain[k] = copy(v)
                    elif isinstance(v, _num_types):
                        self.pdomain[k] = [v, v]
                    else:
                        raise PyDSTool_TypeError('Invalid type for pdomain spec'
                                                 ' '+k)
                else:
                    raise ValueError('Illegal parameter name')
                try:
                    self.parameterDomains[k].set(v)
                except TypeError:
                    raise TypeError('xdomain must be a dictionary of parameter'
                                      ' names -> valid interval 2-tuples or '
                                      'singletons')
                try:
                    evs = self.eventstruct.events.values()
                except AttributeError:
                    evs = []
                for ev in evs:
                    ev.pdomain[k] = self.pdomain[k]
        if 'pars' in kw:
            for k_temp, v in kw['pars'].items():
                k = str(self._FScompatibleNames(k_temp))
                if k in self.pars:
                    cval = self.parameterDomains[k].contains(v)
                    if self.checklevel < 3:
                        if cval is not notcontained:
                            self.pars[k] = ensurefloat(v)
                            if cval is uncertain and self.checklevel == 2:
                                print('Warning: Parameter %s: value at bound'%str(k))
                        else:
                            raise PyDSTool_ValueError('Parameter %s: value out of bounds'%str(k))
                    else:
                        if cval is contained:
                            self.pars[k] = ensurefloat(v)
                        elif cval is uncertain:
                            raise PyDSTool_UncertainValueError('Parameter %s: value at bound'%str(k))
                        else:
                            raise PyDSTool_ValueError('Parameter %s: value out of bounds'%str(k))
                else:
                    raise PyDSTool_ValueError('Illegal parameter name '+str(k))
        if 'algparams' in kw:
            for k, v in kw['algparams'].items():
                self.algparams[k] = v
                if k in ('eventActive', 'eventTol', 'eventDelay', 'eventDir', 'eventInt', 'eventTerm',
                         'maxbisect'):
                    raise ValueError("Use appropriate setX method in Generator.eventstruct")
        if 'inputs' in kw:
            assert self.inputs, ('Cannot provide inputs after '
                                'initialization without them')
            inputs = copy(kw['inputs'])
            _inputs = {}
            if isinstance(inputs, Trajectory):
                # extract the variables
                _inputs = self._FScompatibleNames(inputs.variables)
            elif isinstance(inputs, Variable):
                _inputs = {self._FScompatibleNames(inputs.name): inputs}
            elif isinstance(inputs, Pointset):
                # turn into Variables
                for n in inputs.coordnames:
                    x_array = inputs[n]
                    nFS = self._FScompatibleNames(n)
                    _input[nFS] = Variable(interp1d(inputs.indepvararray,
                                                    x_array), 't',
                                           Interval(nFS, float, extent(x_array),
                                                    abseps=self._abseps),
                                           name=n)
            elif isinstance(inputs, dict):
                _inputs = self._FScompatibleNames(inputs)
                # ensure values are Variables
                for v in _inputs.values():
                    if not isinstance(v, Variable):
                        raise TypeError("Invalid specification of inputs")
            else:
                raise TypeError("Invalid specification of inputs")
            for v in _inputs.values():
                if not iscontinuous(v):
                    raise ValueError("External inputs for ODE system must be "
                                     "continously defined")
            if _inputs:
                for i in _inputs:
                    assert i in self.inputs, 'Incorrect input name provided'
                    self.inputs[i] = _inputs[i]
                # re-calc inputs ixmaps
                self._generate_ixmaps('inputs')
            self._extInputsChanged = True
        if 'inputs_t0' in kw:
            assert self.inputs, ('Cannot provide inputs after '
                                'initialization without them')
            inputs_t0 = self._FScompatibleNames(kw['inputs_t0'])
            for iname, t0 in inputs_t0.items():
                self.inputs[iname]._internal_t_offset = t0
            self._extInputsChanged = True


    def compute(self, dirn):
        """This is an abstract class."""

        # Don't call this from a concrete sub-class
        if self.__class__ is ODEsystem:
            raise NotImplementedError('ODEsystem is an abstract class. '
                              'Use a concrete sub-class of ODEsystem.')
        else:
            raise NotImplementedError("This Generator does not support "
                              "calls to compute()")

    def cleanupMemory(self):
        """Clean up memory usage from past runs of a solver that is interfaced through
        a dynamic link library. This will prevent the 'continue' integration option from
        being accessible and will delete other data about the last integration run."""
        if hasattr(gen, '_solver'):
            # clean up memory usage after calculations in Dopri and Radau, or other
            # solvers that we have interfaced to
            try:
                gen._solver.CleanupEvents()
                gen._solver.CleanupInteg()
            except AttributeError:
                # incompatible solver, so no need to worry
                pass

    def validateICs(self):
        assert hasattr(self, 'initialconditions')
        if remain(self.initialconditions.keys(),
                      self.variables.keys()) != []:
            print("IC names defined: %r" % list(self.initialconditions.keys()))
            print("Varnames defined: %r" % list(self.variables.keys()))
            raise ValueError("Mismatch between initial condition and variable "
                             "names")
        for v in self.funcspec.vars:
            try:
                assert self.initialconditions[v] is not NaN, ('Must specify'
                                    ' initial conditions for all variables')
            except KeyError:
                raise KeyError("Variable name "+v+" not found in initial"
                                 " conditions")


    def Rhs(self, t, xdict, pdict=None, asarray=False):
        # Don't call this from a concrete sub-class
        if self.__class__ is ODEsystem:
            raise NotImplementedError('ODEsystem is an abstract class. '
                              'Use a concrete sub-class of ODEsystem.')
        else:
            raise NotImplementedError("This Generator does not support "
                              "calls to Rhs()")


    def Jacobian(self, t, xdict, pdict=None, asarray=False):
        # Don't call this from a concrete sub-class
        if self.__class__ is ODEsystem:
            raise NotImplementedError('ODEsystem is an abstract class. '
                              'Use a concrete sub-class of ODEsystem.')
        else:
            raise NotImplementedError("This Generator does not support "
                              "calls to Jacobian()")


    def JacobianP(self, t, xdict, pdict=None, asarray=False):
        # Don't call this from a concrete sub-class
        if self.__class__ is ODEsystem:
            raise NotImplementedError('ODEsystem is an abstract class. '
                              'Use a concrete sub-class of ODEsystem.')
        else:
            raise NotImplementedError("This Generator does not support "
                              "calls to JacobianP()")


    def AuxVars(self, t, xdict, pdict=None, asarray=False):
        # Don't call this from a concrete sub-class
        if self.__class__ is ODEsystem:
            raise NotImplementedError('ODEsystem is an abstract class. '
                              'Use a concrete sub-class of ODEsystem.')
        else:
            raise NotImplementedError("This Generator does not support "
                              "calls to AuxFunc()")


    # Methods for pickling protocol
    def __getstate__(self):
        d = copy(self.__dict__)
        for fname, finfo in self._funcreg.items():
            try:
                del d[fname]
            except KeyError:
                pass
        # delete user auxiliary function interface
        try:
            del d['auxfns']
        except KeyError:
            pass
        # can't pickle module objects, so ensure that _solver deleted:
        # Dopri and Radau don't have _solver in funcreg
        try:
            del d['_solver']
        except KeyError:
            pass
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        self._solver = None
        if self._funcreg != {}:
            self.auxfns = auxfn_container(self)
            self.addMethods()


    def __del__(self):
        ctsGen.__del__(self)


