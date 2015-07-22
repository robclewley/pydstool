# Embedded dynamical system generator
from __future__ import division, absolute_import, print_function

from .allimports import *
from .baseclasses import ctsGen, theGenSpecHelper
from PyDSTool.utils import *
from PyDSTool.common import *
from PyDSTool.Interval import uncertain

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, array, arange, \
     transpose, shape
import math, random
from copy import copy, deepcopy


class EmbeddedSysGen(ctsGen):
    """Embedded dynamical system form specifying a trajectory.

    The embedded system is assumed to be of type Model.
    """
    # inherit most of these from the embedded system
    _validKeys = ['globalt0', 'xdomain', 'tdata', 'tdomain',
                     'ics', 'pars', 'checklevel', 'pdomain', 'abseps']
    _needKeys = ctsGen._needKeys + ['specfn', 'system']
    _optionalKeys = ctsGen._optionalKeys + ['tdomain', 'pars', 'pdomain', 'xdomain',
                                  'ics', 'vars', 'tdata', 'enforcebounds',
                                  'activatedbounds']

    def __init__(self, kw):
        ctsGen.__init__(self, kw)
        dispatch_list = ['tdomain', 'tdata', 'xtype', 'xdomain',
                         'ics', 'pars', 'pdomain', 'system']
        if 'varspecs' in kw:
            raise PyDSTool_KeyError('varspecs option invalid for EmbeddedSysGen '
                                    'class')
        if 'inputs' in kw:
            raise PyDSTool_KeyError('inputs option invalid for EmbeddedSysGen '
                                    'class')
        try:
            kw['varspecs'] = kw['system'].query('vardomains')
        except (KeyError, AttributeError):
            raise PyDSTool_KeyError("Model-type system must be provided")
        self.funcspec = args(**self._kw_process_dispatch(dispatch_list, kw))
        self.funcspec.vars = list(kw['varspecs'].keys())
        self.funcspec.auxvars = []
        # varspecs not specified by user and must be removed for checkArgs()
        del kw['varspecs']
        self.indepvartype = float
        try:
            self._embed_spec = kw['specfn']
        except:
            raise PyDSTool_KeyError("Must provide a function for the specification of this system")
        else:
            self.foundKeys += 1
        self.eventstruct = EventStruct()
        self.checkArgs(kw)
        assert self.eventstruct.getLowLevelEvents() == [], \
               "Events are not supported for EmbeddedSysGen class"
        assert self.eventstruct.getHighLevelEvents() == [], \
               "Events are not supported for EmbeddedSysGen class"
        self.indepvariable = Variable(listid, Interval('t_domain',
                                                       self.indepvartype,
                                              self.tdomain, self._abseps),
                             Interval('t', self.indepvartype, self.tdata,
                                      self._abseps), 't')
        self._register(self.indepvariable)
        for x in self.xdomain.keys():
            # aux vars?
            try:
                xinterval=Interval(x, self.xtype[x], self.xdomain[x], self._abseps)
            except KeyError as e:
                raise PyDSTool_KeyError('Mismatch between declared variables '
                                 'and xspecs: ' + str(e))
            # placeholder variable so that this class can be
            # copied before it is defined (listid function is a dummy)
            self.variables[x] = Variable(None, self.indepvariable.depdomain,
                                         xinterval, x)
        # xdomain and pdomain ignored!


    def compute(self, trajname, ics=None):
        """
        """
        if ics is not None:
            self.set(ics=ics)
        self._solver.set(pars=self.pars,
                         globalt0=self.globalt0,
                         ics=self.initialconditions,
                         checklevel=self.checklevel,
                         abseps=self._abseps)
        self.diagnostics.clearWarnings()
        self.diagnostics.clearErrors()
        if not self.defined:
            self._register(self.variables)
#        self.validateSpec()
        try:
            traj = self._embed_spec(self._solver)
        except:
            print("Error in user-provided embedded system")
            raise
        self.defined = True
        traj.name = trajname
        return traj


    def haveJacobian_pars(self):
        """Report whether generator has an explicit user-specified Jacobian
        with respect to pars associated with it."""
        return self._solver.haveJacobian_pars()

    def haveJacobian(self):
        """Report whether generator has an explicit user-specified Jacobian
        associated with it."""
        return self._solver.haveJacobian()


    def set(self, **kw):
        """Set ExplicitFnGen parameters"""
        if remain(kw.keys(), self._validKeys) != []:
            raise KeyError("Invalid keys in argument")
        if 'globalt0' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, globalt0=kw['globalt0'])
        if 'checklevel' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, checklevel=kw['checklevel'])
        if 'abseps' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, abseps=kw['abseps'])
        # optional keys for this call are
        #   ['pars', 'tdomain', 'xdomain', 'pdomain']
        if 'xdomain' in kw:
            for k_temp, v in kw['xdomain'].items():
                # str() ensures that Symbolic objects can be passed
                k = str(self._FScompatibleNames(k_temp))
                if k in self.xdomain.keys():
                    if isinstance(v, _seq_types):
                        assert len(v) == 2, \
                               "Invalid size of domain specification for "+k
                        if v[0] >= v[1]:
                            raise PyDSTool_ValueError('xdomain values must be'
                                                      'in order of increasing '
                                                      'size')
                    elif isinstance(v, _num_types):
                        pass
                    else:
                        raise PyDSTool_TypeError('Invalid type for xdomain spec'
                                                 ' '+k)
                    self.xdomain[k] = v
                else:
                    raise ValueError('Illegal variable name')
                try:
                    self.variables[k].depdomain.set(v)
                except TypeError:
                    raise TypeError('xdomain must be a dictionary of variable'
                                      ' names -> valid interval 2-tuples or '
                                      'singletons')
        if 'pdomain' in kw:
            for k_temp, v in kw['pdomain'].items():
                k = str(self._FScompatibleNames(k_temp))
                if k in self.pars.keys():
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
                    raise TypeError('pdomain must be a dictionary of parameter'
                                      ' names -> valid interval 2-tuples or '
                                      'singletons')
        if 'tdata' in kw:
            self.tdata = kw['tdata']
        if 'tdomain' in kw:
            self.tdomain = kw['tdomain']
            self.indepvariable.indepdomain.set(self.tdomain)
        if self.tdomain[0] > self.tdata[0]:
            if self.indepvariable.indepdomain.contains(self.tdata[0]) == uncertain:
                self.diagnostics.warnings.append((W_UNCERTVAL,
                                                  (self.tdata[0],self.tdomain)))
            else:
                print('tdata cannot be specified below smallest '\
                      'value in tdomain\n (possibly due to uncertain bounding).'\
                      ' It has been automatically adjusted from\n %f to %f '\
                      '(difference of %f)' % (self.tdata[0], self.tdomain[0], self.tdomain[0]-self.tdata[0]))
            self.tdata[0] = self.tdomain[0]
        if self.tdomain[1] < self.tdata[1]:
            if self.indepvariable.indepdomain.contains(self.tdata[1]) == uncertain:
                self.diagnostics.warnings.append((W_UNCERTVAL,
                                                  (self.tdata[1],self.tdomain)))
            else:
                print('tdata cannot be specified above largest '\
                      'value in tdomain\n (possibly due to uncertain bounding).'\
                      ' It has been automatically adjusted from\n ', \
                      '%f to %f (difference of %f)' % (self.tdomain[1], self.tdomain[1], self.tdata[1]-self.tdomain[1]))
            self.tdata[1] = self.tdomain[1]
        self.indepvariable.depdomain.set(self.tdata)
        if 'ics' in kw:
            for k_temp, v in kw['ics'].items():
                k = str(self._FScompatibleNames(k_temp))
                if k in self.xdomain.keys():
                    self._xdatadict[k] = ensurefloat(v)
                else:
                    raise ValueError('Illegal variable name')
            self.initialconditions.update(self._xdatadict)
        if 'pars' in kw:
            if not self.pars:
                raise ValueError('No pars were declared for this object'
                                   ' at initialization.')
            for k_temp, v in kw['pars'].items():
                k = str(self._FScompatibleNames(k_temp))
                if k in self.pars:
                    cval = self.parameterDomains[k].contains(v)
                    if self.checklevel < 3:
                        if cval is not notcontained:
                            self.pars[k] = ensurefloat(v)
                            if cval is uncertain and self.checklevel == 2:
                                print('Warning: Parameter value at bound')
                        else:
                            raise PyDSTool_ValueError('Parameter value out of '
                                                      'bounds')
                    else:
                        if cval is contained:
                            self.pars[k] = ensurefloat(v)
                        elif cval is uncertain:
                            raise PyDSTool_UncertainValueError('Parameter value'
                                                               ' at bound')
                        else:
                            raise PyDSTool_ValueError('Parameter value out of'
                                                      ' bounds')
                else:
                    raise PyDSTool_AttributeError('Illegal parameter name')


    def validateSpec(self):
        ctsGen.validateSpec(self)
        try:
            for v in self.variables.values():
                assert isinstance(v, Variable)
            assert not self.inputs
        except AssertionError:
            print('Invalid system specification')
            raise


    def __del__(self):
        ctsGen.__del__(self)



# Register this Generator with the database

symbolMapDict = {}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(EmbeddedSysGen, symbolMapDict, 'python', None)
