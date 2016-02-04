# Implicit function generator
from __future__ import division, absolute_import, print_function

from .allimports import *
from .baseclasses import ctsGen, theGenSpecHelper
from PyDSTool.utils import *
from PyDSTool.common import *
from PyDSTool.Interval import uncertain

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, array, \
     transpose, shape
import math, random
from copy import copy, deepcopy
import six


class ImplicitFnGen(ctsGen):
    """Implicitly defined functional-form trajectory generator.
    """
    _validKeys = ['globalt0', 'xdomain', 'tdata', 'tdomain', 'checklevel',
                   'name', 'ics', 'pars', 'algparams', 'pdomain', 'abseps']
    _needKeys = ctsGen._needKeys + ['varspecs', 'ics']
    _optionalKeys = ctsGen._optionalKeys + ['tdomain', 'pars', 'pdomain', 'xdomain',
                                  'xtype', 'auxvars', 'vars', 'events',
                                  'algparams', 'fnspecs', 'tdata']

    def __init__(self, kw):
        ctsGen.__init__(self, kw)
        dispatch_list = ['varspecs', 'tdomain', 'tdata', 'xtype', 'xdomain',
                         'ics', 'allvars', 'pars', 'pdomain', 'fnspecs',
                         'algparams', 'target']
        if 'inputs' in kw:
            raise PyDSTool_KeyError("inputs option invalid for ImplicitFnGen "
                                    "class")
        self.funcspec = ImpFuncSpec(self._kw_process_dispatch(dispatch_list, kw))
        self.indepvartype = float
        for s in self.funcspec.spec[0]:
            if s.find('x[') > -1:
                raise ValueError('Variable values cannot depend on '
                            'other variables in implicit function specs -- '
                            'in function:\n'+s)
        if 'solvemethod' in self.algparams:
            if self.algparams['solvemethod'] not in _implicitSolveMethods:
                raise PyDSTool_ValueError('Invalid implicit solver type')
        # Holder and interface for events
        self.eventstruct = EventStruct()
        if 'events' in kw:
            raise PyDSTool_ValueError('ImplicitFnGen does not presently' \
                                      ' support events')
##            self._addEvents(kw['events'])
##            assert self.eventstruct.getLowLevelEvents() == [], \
##               "Can only pass high level events to ImplicitFnGen objects"
##            assert self.eventstruct.query(['highlevel', 'varlinked']) == [], \
##               "Only non-variable linked events are valid for this class"
##            self.foundKeys += 1
        self.checkArgs(kw)
        self.newTempVars()
        self._generate_ixmaps()


    def newTempVars(self):
        self.indepvariable = Variable(listid, Interval('t_domain',
                                                       self.indepvartype,
                                        self.tdomain, self._abseps),
                                      Interval('t', self.indepvartype,
                                               self.tdata, self._abseps),
                                      't')
        if not self.defined:
            self._register(self.indepvariable)
        for x in self.funcspec.vars + self.funcspec.auxvars:
            try:
                xinterval=Interval(x, self.xtype[x], self.xdomain[x], self._abseps)
            except KeyError as e:
                raise PyDSTool_KeyError('Mismatch between declared variables'
                                 ' and xspecs: ' + str(e))
            # placeholder variable so that this class can be
            # copied before it is defined (listid function is a dummy)
            self.variables[x] = Variable(None, self.indepvariable.depdomain,
                                         xinterval, x)


    def compute(self, trajname, ics=None):
        """Attach specification functions to callable interface."""

        assert self.funcspec.targetlang == 'python', \
               ('Wrong target language for functional specification. '
                'Python needed for this class')
        assert isinstance(self.funcspec, ImpFuncSpec), ('ImplicitFnGen'
                                    ' requires ImpFuncSpec type to proceed')
        # repeat this check (made in __init__) in case events were added since
        assert self.eventstruct.getLowLevelEvents() == [], \
               "Can only pass high level events to ImplicitFnGen objects"
        assert self.eventstruct.query(['highlevel', 'varlinked']) == [], \
               "Only non-variable linked events are valid for this class"
        if ics is not None:
            self.set(ics=ics)
        # set some defaults for implicit function
        if 'solvemethod' in self.algparams:
            if self.algparams['solvemethod'] not in _implicitSolveMethods:
                raise PyDSTool_ValueError('Invalid implicit solver type')
        else:
            self.algparams['solvemethod'] = 'fsolve'
        if self.algparams['solvemethod'] in _1DimplicitSolveMethods and \
           self.dimension > 1:
            raise PyDSTool_TypeError('Inappropriate implicit solver for '
                                     'non-scalar system')
        if 'atol' not in self.algparams:
            self.algparams['atol'] = 1e-8
        if 'maxnumiter' not in self.algparams:
            self.algparams['maxnumiter'] = 100
        if self.defined:
            # reset variables
            self.newTempVars()
        self.setEventICs(self.initialconditions, self.globalt0)
        tempfs = deepcopy(self.funcspec)
        tempvars = copyVarDict(self.variables)
        # make unique fn for this trajectory
        tempspec = makeUniqueFn(copy(tempfs.spec[0]), 7, self.name)
        tempfs.spec = tempspec
        # test supplied code
        try:
            six.exec_(tempspec[0], globals())
        except:
            print('Error in supplied functional specification code')
            raise
        # set up implicit function: utils.makeImplicitFunction gets
        # called finally in Variable.addMethods() method.
        tempfs.algparams.update(self.algparams)
        if self.haveJacobian():
            tempfs.algparams['jac'] = self.funcspec.auxfns['Jacobian']
        else:
            tempfs.algparams['jac'] = None
        tempfs.algparams['pars'] = sortedDictValues(self.pars)
        if self.dimension == 1 and \
           self.algparams['solvemethod'] in _1DimplicitSolveMethods:
            tempfs.algparams['x0'] = sortedDictValues(self.initialconditions,
                                                  self.funcspec.vars)[0]
        else:
            tempfs.algparams['x0'] = sortedDictValues(self.initialconditions,
                                                  self.funcspec.vars)
        tempfs.algparams['impfn_name'] = "impfn_" + timestamp(7)
##        tempfs.algparams['impfn_name'] = "impfn"
        # create wrapper functions around implicit function, for each variable
        for x in self.funcspec.vars:
            x_ix = self.funcspec.vars.index(x)
            funcname = "_mapspecfn_" + x + "_" + timestamp(7)
            funcstr = "def " + funcname + "(self, t):\n\treturn " \
                    + tempfs.algparams['impfn_name'] + "(self, t)"
            if self.dimension > 1:
                funcstr += "[" + str(x_ix) + "]\n"
            else:
                funcstr += "\n"
            # make output fn of each variable the entry in the output from the
            # same implicit function.
            # initial conditions aren't needed beyond algparams['x0']
            tempvars[x].setOutput((funcname,funcstr), tempfs,
                                  self.globalt0, self._var_namemap)
        if self.funcspec.auxvars != []:
            # make unique fn for this trajectory
            tempauxspec = makeUniqueFn(copy(tempfs.auxspec[0]), 7, self.name)
            tempfs.auxspec = tempauxspec
        for a in self.funcspec.auxvars:
            a_ix = self.funcspec.auxvars.index(a)
            funcname = "_mapspecfn_" + a + "_" + timestamp(7)
            funcstr = "def " + funcname + "(self, t):\n\treturn "
            if len(self.funcspec.auxvars) == 1:
                # we'll only go through this once!
                funcstr += tempauxspec[1] + "(self, t, [v(t) " \
                      + "for v in self._refvars], " \
                      + repr(sortedDictValues(self.pars)) \
                      + ")[0]\n"
            else:
                funcstr += tempauxspec[1] + "(self, t, [v(t) " \
                      + "for v in self._refvars], " \
                      + repr(sortedDictValues(self.pars)) \
                      + ")[" + str(a_ix) + "]\n"
            # initial conditions aren't needed beyond algparams['x0']
            tempvars[a].setOutput((funcname, funcstr), tempfs,
                                        self.globalt0, self.funcspec.auxvars,
                                        None,
                                        sortedDictValues(tempvars,
                                                         self.funcspec.vars))
        self.diagnostics.clearWarnings()
        self.diagnostics.clearErrors()
        if self.eventstruct.getHighLevelEvents():
            raise PyDSTool_ValueError('ImplicitFnGen does not presently' \
                                      ' support events')
##        # Find any events in tdomain, and adjust tdomain in case they
##        # are terminal
##        eventslist = self.eventstruct.query(['highlevel', 'active',
##                                             'notvarlinked'])
##        termevents = self.eventstruct.query(['term'], eventslist)
##        if eventslist != []:
##            Evts = []
##            for evix in xrange(len(eventslist)):
##                (evname, ev) = eventslist[evix]
##                evsfound = ev.searchForEvents(self.indepvariable.depdomain.get(),
##                                              parDict=self.pars,
##                                              vars=tempvars,
##                                              checklevel=self.checklevel)
##                Evts.append([evinfo[0] for evinfo in evsfound])
##        if eventslist != []:
##            self.eventstruct.resetHighLevelEvents(self.indepvariable.depdomain[0],
##                                                  eventslist)
##            self.eventstruct.validateEvents(self.funcspec.vars + \
##                                            self.funcspec.auxvars + \
##                                            ['t'], eventslist)
##            termevtimes = {}
##            nontermevtimes = {}
##            for evix in xrange(len(eventslist)):
##                numevs = shape(Evts[evix])[-1]
##                if numevs == 0:
##                    continue
##                if eventslist[evix][1].activeFlag:
##                    if numevs > 1:
##                        print "Event info:", Evts[evix]
##                    assert numevs <= 1, ("Internal error: more than one "
##                                     "terminal event of same type found")
##                    # For safety, we should assert that this event
##                    # also appears in termevents, but we don't
##                    evname = eventslist[evix][0]
##                    if Evts[evix][0] in termevtimes.keys():
##                        # append event name to this warning
##                        warning_ix = termevtimes[Evts[evix][0]]
##                        self.diagnostics.warnings[warning_ix][1][1].append(evname)
##                    else:
##                        # make new termevtime entry for the new warning
##                        termevtimes[Evts[evix][0]] = len(self.diagnostics.warnings)
##                        self.diagnostics.warnings.append((W_TERMEVENT,
##                                         (Evts[evix][0],
##                                         [eventslist[evix][0]])))
##                else:
##                    for ev in range(numevs):
##                        if Evts[evix][ev] in nontermevtimes.keys():
##                            # append event name to this warning
##                            warning_ix = nontermevtimes[Evts[evix][ev]]
##                            self.diagnostics.warnings[warning_ix][1][1].append(evname)
##                        else:
##                            # make new nontermevtime entry for the new warning
##                            nontermevtimes[Evts[evix][ev]] = \
##                                                        len(self.diagnostics.warnings)
##                            self.diagnostics.warnings.append((W_NONTERMEVENT,
##                                             (Evts[evix][ev],
##                                              [eventslist[evix][0]])))
##        termcount = 0
##        earliest_termtime = self.indepvariable.depdomain[1]
##        for (w,i) in self.diagnostics.warnings:
##            if w == W_TERMEVENT or w == W_TERMSTATEBD:
##                termcount += 1
##                if i[0] < earliest_termtime:
##                    earliest_termtime = i[0]
##        # now delete any events found after the earliest terminal event, if any
##        if termcount > 0:
##            warn_temp = []
##            for (w,i) in self.diagnostics.warnings:
##                if i[0] <= earliest_termtime:
##                    warn_temp.append((w,i))
##            self.diagnostics.warnings = warn_temp
##        self.indepvariable.depdomain.set([self.indepvariable.depdomain[0],
##            earliest_termtime])
##        for v in tempvars.values():
##            v.indepdomain.set(self.indepvariable.depdomain.get())
####                print 'Time interval adjusted according to %s: %s' % \
####                      (self._warnmessages[w], str(i[0])+", "+ str(i[1]))
        if not self.defined:
            self._register(self.variables)
        #self.validateSpec()
        self.defined = True
        return Trajectory(trajname, list(tempvars.values()),
                          abseps=self._abseps, globalt0=self.globalt0,
                          checklevel=self.checklevel,
                          FScompatibleNames=self._FScompatibleNames,
                          FScompatibleNamesInv=self._FScompatibleNamesInv,
                          modelNames=self.name,
                          modelEventStructs=self.eventstruct)

    def haveJacobian_pars(self):
        """Report whether generator has an explicit user-specified Jacobian
        with respect to pars associated with it."""
        return 'Jacobian_pars' in self.funcspec.auxfns

    def haveJacobian(self):
        """Report whether generator has an explicit user-specified Jacobian
        associated with it."""
        return 'Jacobian' in self.funcspec.auxfns


    def set(self, **kw):
        """Set ImplicitFnGen parameters"""
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
        # optional keys for this call are ['pars', 'tdomain', 'xdomain', 'pdomain']
        if 'xdomain' in kw:
            for k_temp, v in kw['xdomain'].items():
                # str() ensures that Symbolic objects can be passed
                k = str(self._FScompatibleNames(k_temp))
                if k in self.funcspec.vars+self.funcspec.auxvars:
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
                      ' It has been automatically adjusted from %f to %f (difference of %f)\n' % (
                          self.tdata[0], self.tdomain[0], self.tdomain[0]-self.tdata[0]))
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
            self.tdata[1] = self.tdomain[1]
        self.indepvariable.depdomain.set(self.tdata)
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
                    raise TypeError('pdomain must be a dictionary of parameter'
                                      ' names -> valid interval 2-tuples or '
                                      'singletons')
        if 'ics' in kw:
            for k_temp, v in kw['ics'].items():
                k = str(self._FScompatibleNames(k_temp))
                if k in self.funcspec.vars+self.funcspec.auxvars:
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
                            raise PyDSTool_ValueError('Parameter value out of bounds')
                    else:
                        if cval is contained:
                            self.pars[k] = ensurefloat(v)
                        elif cval is uncertain:
                            raise PyDSTool_UncertainValueError('Parameter value at bound')
                        else:
                            raise PyDSTool_ValueError('Parameter value out of bounds')
                else:
                    raise PyDSTool_AttributeError('Illegal parameter name')
        if 'algparams' in kw:
            for k, v in kw['algparams'].items():
                self.algparams[k] = v
        if 'solvemethod' in self.algparams:
            if self.algparams['solvemethod'] not in _implicitSolveMethods:
                raise PyDSTool_ValueError('Invalid implicit solver type')


    def validateSpec(self):
        ctsGen.validateSpec(self)
        try:
            for v in list(self.variables.values()):
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
theGenSpecHelper.add(ImplicitFnGen, symbolMapDict, 'python', 'ImpFuncSpec')
