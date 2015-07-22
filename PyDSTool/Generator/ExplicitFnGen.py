# Explicit function generator
from __future__ import division, absolute_import, print_function

from .allimports import *
from .baseclasses import ctsGen, theGenSpecHelper, \
     auxfn_container, _pollInputs
from PyDSTool.utils import *
from PyDSTool.common import *
from PyDSTool.Interval import uncertain

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, array, arange, \
     transpose, shape
import math, random, types
from copy import copy, deepcopy
import six


class ExplicitFnGen(ctsGen):
    """Explicit functional form specifying a trajectory.

    E.g. for an external input. This class allows parametric
    forms of the function, but with no dependence on x or its
    own external inputs."""
    _validKeys = ['globalt0', 'xdomain', 'tdata', 'tdomain',
                     'ics', 'pars', 'checklevel', 'pdomain', 'abseps']
    _needKeys = ctsGen._needKeys + ['varspecs']
    _optionalKeys = ctsGen._optionalKeys + ['tdomain', 'pars', 'pdomain', 'xdomain',
                                  'xtype', 'ics', 'auxvars', 'vars', 'events',
                                  'fnspecs', 'tdata', 'enforcebounds',
                                  'activatedbounds', 'reuseterms']

    def __init__(self, kw):
        ctsGen.__init__(self, kw)
        dispatch_list = ['varspecs', 'tdomain', 'tdata', 'xtype', 'xdomain',
                         'ics', 'allvars', 'reuseterms', 'pars', 'pdomain',
                         'fnspecs', 'target']
        # allow inputs only if it's empty, for compatibility with ModelConstructor
        # which might put an empty dictionary in for this key
        if 'inputs' in kw:
            if kw['inputs'] != {}:
                raise PyDSTool_KeyError('inputs option invalid for ExplicitFnGen '
                                    'class')
        self.funcspec = ExpFuncSpec(self._kw_process_dispatch(dispatch_list,
                                                              kw))
        self.indepvartype = float
        for s in self.funcspec.spec[0]:
            if s.find('x[') > -1:
                raise ValueError('Variable values cannot depend on '
                            'other variables in explicit function specs -- '
                            'in function:\n'+s)
        self._kw_process_events(kw)
        self.checkArgs(kw)
        self.indepvariable = Variable(listid, Interval('t_domain',
                                                       self.indepvartype,
                                              self.tdomain, self._abseps),
                             Interval('t', self.indepvartype, self.tdata,
                                      self._abseps), 't')
        self._register(self.indepvariable)
        for x in self.funcspec.vars + self.funcspec.auxvars:
            try:
                xinterval=Interval(x, self.xtype[x], self.xdomain[x], self._abseps)
            except KeyError as e:
                raise PyDSTool_KeyError('Mismatch between declared variables '
                                 'and xspecs: ' + str(e))
            # placeholder variable so that this class can be
            # copied before it is defined (listid function is a dummy)
            self.variables[x] = Variable(None, self.indepvariable.depdomain,
                                         xinterval, x)
        self._generate_ixmaps()
        self.auxfns = auxfn_container(self)
        self.addMethods()

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


    def compute(self, trajname, ics=None):
        """Attach specification functions to callable interface."""
        # repeat this check (made in __init__) in case events were added since
        assert self.eventstruct.getLowLevelEvents() == [], \
               "Can only pass high level events to ExplicitFnGen objects"
        assert self.eventstruct.query(['highlevel', 'varlinked']) == [], \
               "Only non-variable linked events are valid for this class"
##        icdict_local = copy(self.initialconditions)
##        t0 = self.indepvariable.depdomain[0]
##        icdict_local['t'] = t0
##        have_aux = len(self.funcspec.auxvars)>0
##        for a in self.funcspec.auxvars:
##            # these functions are intended to be methods in their target
##            # Variable object, so expect a first argument 'self'
##            exec self.funcspec.auxspec[0]
##        if have_aux:
##            self.initialconditions.update(dict(zip(self.funcspec.auxvars,
##                    apply(locals()[self.funcspec.auxspec[1]],
##                                  (None, t0, sortedDictValues(icdict_local),
##                                       sortedDictValues(self.pars))) )))
        if ics is not None:
            self.set(ics=ics)
        self.setEventICs(self.initialconditions, self.globalt0)
        tempfs = deepcopy(self.funcspec)
        tempvars = copyVarDict(self.variables)
        # make unique fn for this trajectory: function definition gets executed
        # finally in Variable.addMethods() method
        tempspec = makeUniqueFn(copy(tempfs.spec[0]), 7, self.name)
        tempfs.spec = tempspec
        for x in self.funcspec.vars:
            x_ix = self.funcspec.vars.index(x)
            funcname = "_mapspecfn_" + x + "_" + timestamp(7)
            funcstr = "def " + funcname + "(self, t):\n\treturn "
            if len(self.funcspec.vars) == 1:
                # this clause is unnecessary if [0] is ever dropped
                # i.e. if spec would return plain scalar in 1D case
                funcstr += tempfs.spec[1] + "(self, t, [0], " \
                       + repr(sortedDictValues(self.pars)) + ")[0]\n"
            else:
                funcstr += tempfs.spec[1] + "(self, t, [0], " \
                       + repr(sortedDictValues(self.pars)) + ")[" \
                       + str(x_ix) + "]\n"
            tempvars[x].setOutput((funcname, funcstr), tempfs,
                                        self.globalt0, self._var_namemap,
                                        copy(self.initialconditions))
        if self.funcspec.auxvars != []:
            # make unique fn for this trajectory
            tempauxspec = makeUniqueFn(copy(tempfs.auxspec[0]), 7, self.name)
            tempfs.auxspec = tempauxspec
        for a in self.funcspec.auxvars:
            a_ix = self.funcspec.auxvars.index(a)
            funcname = "_mapspecfn_" + a + "_" + timestamp(7)
            funcstr = "def " + funcname + "(self, t):\n\treturn "
            if len(self.funcspec.auxvars) == 1:
                # this clause is unnecessary if [0] is ever dropped
                # i.e. if auxspec would return plain scalar in 1D case
                funcstr += tempfs.auxspec[1] + "(self, t, [v(t) " \
                      + "for v in self._refvars], " \
                      + repr(sortedDictValues(self.pars)) \
                      + ")[0]\n"
            else:
                funcstr += tempfs.auxspec[1] + "(self, t, [v(t) " \
                      + "for v in self._refvars], " \
                      + repr(sortedDictValues(self.pars)) \
                      + ")[" + str(a_ix) + "]\n"
            tempvars[a].setOutput((funcname, funcstr), tempfs,
                                        self.globalt0, self.funcspec.auxvars,
                                        copy(self.initialconditions),
                                        sortedDictValues(tempvars,
                                                         self.funcspec.vars))
        self.diagnostics.clearWarnings()
        self.diagnostics.clearErrors()
        # Find any events in tdomain, and adjust tdomain in case they
        # are terminal
        eventslist = self.eventstruct.query(['highlevel', 'active',
                                             'notvarlinked'])
        termevents = self.eventstruct.query(['term'], eventslist)
        Evtimes = {}
        Evpoints = {}
        for evname, ev in eventslist:
            Evtimes[evname] = []
            Evpoints[evname] = []
        if eventslist != []:
            if self._for_hybrid_DS:
                # self._for_hybrid_DS is set internally by HybridModel class
                # to ensure not to reset events, because they may be about to
                # flag on first step if previous hybrid state was the same
                # generator and, for example, two variables are synchronizing
                # so that their events get very close together.
                # Just reset the starttimes of these events
                for evname, ev in eventslist:
                    ev.starttime = t0
            else:
                self.eventstruct.resetHighLevelEvents(self.indepvariable.depdomain[0],
                                                  eventslist)
                self.eventstruct.validateEvents(self.funcspec.vars + \
                                            self.funcspec.auxvars + \
                                            ['t'], eventslist)
            for evname, ev in eventslist:
                # select only continuous-valued variables for event detection
                # (in case of indicator variables used in hybrid systems)
                evsfound = ev.searchForEvents(self.indepvariable.depdomain.get(),
                                              parDict=self.pars,
                                              vars=copyVarDict(tempvars, only_cts=True),
                                              checklevel=self.checklevel)
                tvals = sortedDictValues(tempvars)
                for evinfo in evsfound:
                    Evtimes[evname].append(evinfo[0])
                    Evpoints[evname].append(array([v(evinfo[0]) for v in tvals]))
            self.eventstruct.resetHighLevelEvents(self.indepvariable.depdomain[0],
                                                  eventslist)
            self.eventstruct.validateEvents(self.funcspec.vars + \
                                            self.funcspec.auxvars + \
                                            ['t'], eventslist)
            termevtimes = {}
            nontermevtimes = {}
            for evname, ev in eventslist:
                numevs = shape(Evtimes[evname])[-1]
                if numevs == 0:
                    continue
                if ev.activeFlag:
                    if numevs > 1:
                        print("Event info: %r" % Evtimes[evname])
                    assert numevs <= 1, ("Internal error: more than one "
                                     "terminal event of same type found")
                    # For safety, we should assert that this event
                    # also appears in termevents, but we don't
                    if Evtimes[evname][0] in termevtimes.keys():
                        # append event name to this warning
                        warning_ix = termevtimes[Evtimes[evname][0]]
                        self.diagnostics.warnings[warning_ix][1][1].append(evname)
                    else:
                        # make new termevtime entry for the new warning
                        termevtimes[Evtimes[evname][0]] = \
                                   len(self.diagnostics.warnings)
                        self.diagnostics.warnings.append((W_TERMEVENT,
                                         (Evtimes[evname][0],
                                         [evname])))
                else:
                    for ev in range(numevs):
                        if Evtimes[evname][ev] in nontermevtimes.keys():
                            # append event name to this warning
                            warning_ix = nontermevtimes[Evtimes[evname][ev]]
                            self.diagnostics.warnings[warning_ix][1][1].append(evname)
                        else:
                            # make new nontermevtime entry for the new warning
                            nontermevtimes[Evtimes[evname][ev]] = \
                                                len(self.diagnostics.warnings)
                            self.diagnostics.warnings.append((W_NONTERMEVENT,
                                             (Evtimes[evname][ev],
                                              [evname])))
        termcount = 0
        earliest_termtime = self.indepvariable.depdomain[1]
        for (w,i) in self.diagnostics.warnings:
            if w == W_TERMEVENT or w == W_TERMSTATEBD:
                termcount += 1
                if i[0] < earliest_termtime:
                    earliest_termtime = i[0]
        # now delete any events found after the earliest terminal event, if any
        if termcount > 0:
            warn_temp = []
            for (w,i) in self.diagnostics.warnings:
                if i[0] <= earliest_termtime:
                    warn_temp.append((w,i))
            self.diagnostics.warnings = warn_temp
        self.indepvariable.depdomain.set([self.indepvariable.depdomain[0],
                                          earliest_termtime])
        for v in tempvars.values():
            v.indepdomain.set(self.indepvariable.depdomain.get())
##                print 'Time interval adjusted according to %s: %s' % \
##                      (self._warnmessages[w], str(i[0])+", "+ str(i[1]))
        # build event pointset information (reset previous trajectory's)
        self.trajevents = {}
        for (evname, ev) in eventslist:
            evpt = Evpoints[evname]
            if evpt == []:
                self.trajevents[evname] = None
            else:
                evpt = transpose(array(evpt))
                self.trajevents[evname] = Pointset({
                                'coordnames': sortedDictKeys(tempvars),
                                'indepvarname': 't',
                                'coordarray': evpt,
                                'indepvararray': Evtimes[evname],
                                'indepvartype': self.indepvartype})
        if not self.defined:
            self._register(self.variables)
        self.validateSpec()
        self.defined = True
        return Trajectory(trajname, list(tempvars.values()),
                          abseps=self._abseps, globalt0=self.globalt0,
                          checklevel=self.checklevel,
                          FScompatibleNames=self._FScompatibleNames,
                          FScompatibleNamesInv=self._FScompatibleNamesInv,
                          events=self.trajevents,
                          modelNames=self.name,
                          modelEventStructs=self.eventstruct)


    def AuxVars(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.AuxVars"""
        # also, ensure xdict doesn't contain elements like array([4.1]) instead of 4
        x = [float(val) for val in sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                                                 self.funcspec.vars))]
        if pdict is None:
            pdict = self.pars
            # internal self.pars already is FS-compatible
            p = sortedDictValues(pdict)
        else:
            p = sortedDictValues(self._FScompatibleNames(pdict))
        i = _pollInputs(sortedDictValues(self.inputs), t, self.checklevel)
        return getattr(self, self.funcspec.auxspec[1])(*[t, x, p+i])

    def haveJacobian_pars(self):
        """Report whether generator has an explicit user-specified Jacobian
        with respect to pars associated with it."""
        return 'Jacobian_pars' in self.funcspec.auxfns

    def haveJacobian(self):
        """Report whether generator has an explicit user-specified Jacobian
        associated with it."""
        return 'Jacobian' in self.funcspec.auxfns


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
                for ev in self.eventstruct.events.values():
                    ev.xdomain[k] = v
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
                for ev in self.eventstruct.events.values():
                    ev.pdomain[k] = v
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
theGenSpecHelper.add(ExplicitFnGen, symbolMapDict, 'python', 'ExpFuncSpec')
