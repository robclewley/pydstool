# MapSystem
# For the MapSystem class, maps have an explicit "time" step, generating
#  values for x at a specifc "time". Purely abstract time (i.e., iteration
#  steps) is represented using integers. (We could make LookupTable a
#  0-param sequence of maps with explicit time range?)
from __future__ import division, absolute_import, print_function

from .allimports import *
from .baseclasses import Generator, discGen, theGenSpecHelper, \
     auxfn_container, _pollInputs
from PyDSTool.utils import *
from PyDSTool.common import *
from PyDSTool.Variable import Variable
from PyDSTool.Trajectory import Trajectory
from PyDSTool.Points import Pointset
from PyDSTool.Interval import uncertain

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, array, transpose, \
     concatenate
import math, random, types
from copy import copy, deepcopy
import six


class MapSystem(discGen):
    """Discrete dynamical systems, as maps (difference equations).

    """
    _validKeys = ['globalt0', 'xdomain', 'tdata', 'tdomain', 'checklevel',
                     'ics', 'pars', 'inputs', 'pdomain', 'abseps']
    _needKeys = discGen._needKeys + ['varspecs']
    _optionalKeys = discGen._optionalKeys + ['tdomain', 'xdomain', 'inputs', 'tdata',
                          'ics', 'events', 'system', 'ignorespecial',
                          'auxvars', 'vars', 'fnspecs', 'ttype', 'xtype',
                          'tstep', 'checklevel', 'pars', 'pdomain',
                          'vfcodeinsert_start', 'vfcodeinsert_end',
                          'enforcebounds', 'activatedbounds', 'reuseterms']

    def __init__(self, kw):
        discGen.__init__(self, kw)
        self.diagnostics._errmessages[E_COMPUTFAIL] = 'Computation of trajectory failed'
        # user auxiliary function interface
        self.auxfns = auxfn_container(self)
        dispatch_list = ['varspecs', 'tdomain', 'ttype', 'tdata', 'tstep',
                          'inputs', 'ics', 'allvars', 'xtype', 'pars',
                          'xdomain', 'reuseterms', 'algparams', 'pdomain',
                          'system', 'fnspecs', 'vfcodeinserts', 'ignorespecial']
        # process keys and build func spec
        self.funcspec = RHSfuncSpec(self._kw_process_dispatch(dispatch_list,
                                                              kw))
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


    def addMethods(self):
        # Add the auxiliary function specs to this Generator's namespace
        for auxfnname in self.funcspec._pyauxfns:
            fninfo = self.funcspec._pyauxfns[auxfnname]
            if not hasattr(self, fninfo[1]):
                # user-defined auxiliary functions
                # (built-ins are provided explicitly)
                if self._solver:
                    fnstr = fninfo[0].replace(self._solver.name, 'ds._solver')
##                    self._funcreg[self._solver.name] = self._solver
                else:
                    fnstr = fninfo[0]
                try:
                    exec(fnstr)
                except:
                    print('Error in supplied auxiliary function code')
                self._funcreg[fninfo[1]] = ('self', fnstr)
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
        if self.funcspec.targetlang == 'python':
            # Add the spec function to this Generator's namespace
            fninfo = self.funcspec.spec
            if self._solver:
                fnstr = fninfo[0].replace(self._solver.name, 'ds._solver')
            else:
                fnstr = fninfo[0]
            try:
                exec(fnstr)
            except:
                print('Error in supplied functional specification code')
                raise
            self._funcreg[fninfo[1]] = ('self', fnstr)
            setattr(self, fninfo[1], six.create_bound_method(locals()[fninfo[1]], self))
            # Add the auxiliary spec function (if present) to this
            # Generator's namespace
            if self.funcspec.auxspec != '':
                fninfo = self.funcspec.auxspec
                if self._solver:
                    fnstr = fninfo[0].replace(self._solver.name, 'ds._solver')
                else:
                    fnstr = fninfo[0]
                try:
                    exec(fnstr)
                except:
                    print('Error in supplied auxiliary variable code')
                    raise
                self._funcreg[fninfo[1]] = ('self', fnstr)
                setattr(self, fninfo[1], six.create_bound_method(locals()[fninfo[1]], self))


    # Method for pickling protocol (setstate same as default)
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
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._funcreg != {}:
            self.auxfns = auxfn_container(self)
            self.addMethods()


    def haveJacobian_pars(self):
        """Report whether generator has an explicit user-specified Jacobian
        with respect to pars associated with it."""
        return 'Jacobian_pars' in self.funcspec.auxfns


    def haveJacobian(self):
        """Report whether map system has an explicit user-specified Jacobian
        associated with it."""
        return 'Jacobian' in self.funcspec.auxfns


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
                print("Variable value: %f" % val)
                raise ValueError("Initial condition for "+xname+" has been "
                                   "set outside of prescribed bounds")


    def set(self, **kw):
        """Set map system parameters"""
        if remain(kw.keys(), self._validKeys) != []:
            raise KeyError("Invalid keys in argument")
        if 'globalt0' in kw:
            # pass up to generic treatment for this
            discGen.set(self, globalt0=kw['globalt0'])
        if 'checklevel' in kw:
            # pass up to generic treatment for this
            discGen.set(self, checklevel=kw['checklevel'])
        if 'abseps' in kw:
            # pass up to generic treatment for this
            discGen.set(self, abseps=kw['abseps'])
        # optional keys for this call are ['pars', 'tdomain', 'ics',
        #   'algparams', 'tdata', 'xdomain', 'inputs', 'pdomain']
        if 'ics' in kw:
            for k_temp, v in kw['ics'].items():
                # str() ensures that Symbolic objects can be passed
                k = str(self._FScompatibleNames(k_temp))
                if k in self.funcspec.vars+self.funcspec.auxvars:
                    self._xdatadict[k] = ensurefloat(v)
                else:
                    raise ValueError('Illegal variable name, %s'%k)
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
                    self.xdomain[k] = v
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
            assert self.numpars > 0, ('No pars were declared for this '
                                      'model')
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
                    raise PyDSTool_ValueError('Illegal parameter name')
            # pass on parameter changes to embedded system, if present
            if self._solver:
                try:
                    shared_pars = intersect(kw['pars'].keys(), self._solver.pars)
                except AttributeError:
                    # no pars for this kind of solver
                    pass
                else:
                    if shared_pars != []:
                        self._solver.set(pars=filteredDict(kw['pars'], shared_pars))
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
                    _inputs[nFS] = \
                           Variable(interp1d(inputs.indepvararray,
                                                       x_array), 't',
                                         Interval(nFS, float, extent(x_array),
                                                  abseps=self._abseps),
                                         name=n)  # keep original name here
            elif isinstance(inputs, dict):
                _inputs = self._FScompatibleNames(inputs)
                # ensure values are Variables
                for v in _inputs.values():
                    if not isinstance(v, Variable):
                        raise TypeError("Invalid specification of inputs")
            else:
                raise TypeError("Invalid specification of inputs")
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


    def compute(self, trajname, ics=None):
        assert self.funcspec.targetlang == 'python', \
               ('Wrong target language for functional specification. '
                'Python needed for this class')
        assert isinstance(self.funcspec, RHSfuncSpec), ('Map system '
                                    'requires RHSfuncSpec type to proceed')
        self.diagnostics.clearWarnings()
        self.diagnostics.clearErrors()
        if ics is not None:
            self.set(ics=ics)
        xnames = self._var_ixmap  # ensures correct order
        # wrap up each dictionary initial value as a singleton list
        alltData = [self.indepvariable.depdomain[0]]
        allxDataDict = dict(zip(xnames, map(listid,
                                   sortedDictValues(self.initialconditions,
                                                    self.funcspec.vars))))
        rhsfn = getattr(self,self.funcspec.spec[1])
        # Check i.c.'s are well defined (finite)
        self.checkInitialConditions()
        self.setEventICs(self.initialconditions, self.globalt0)
        ic = sortedDictValues(self.initialconditions, self.funcspec.vars)
        plist = sortedDictValues(self.pars)
        extralist = copy(plist)
        ilist = []
        if self.inputs:
            # inputVarList is a list of Variables
            listend = self.numpars + len(self.inputs)
            inputVarList = sortedDictValues(self.inputs)
            try:
                for f in inputVarList:
                    f.diagnostics.clearWarnings()
                    ilist.append(f(alltData[0], self.checklevel))
            except AssertionError:
                print('External input call has t out of range: t = %f' % \
                    self.indepvariable.depdomain[0])
                print('Maybe checklevel is 3 and initial time is not', \
                            'completely inside valid time interval')
                raise
            except ValueError:
                print('External input call has value out of range: t = %f' % \
                      self.indepvariable.depdomain[0])
                for f in inputVarList:
                    if f.diagnostics.hasWarnings():
                        print('External input %s out of range:' % f.name)
                        print('   t = %r, %s, %r' % (repr(f.diagnostics.warnings[-1][0]),
                              f.name, repr(f.diagnostics.warnings[-1][1])))
                raise
        else:
            listend = self.numpars
            inputVarList = []
        extralist.extend(ilist)
        precevents = self.eventstruct.query(['precise'])
        if precevents != []:
            raise PyDSTool_ValueError('precise events are not valid for map systems')
        eventslist = self.eventstruct.query(['highlevel', 'active',
                                             'notvarlinked'])
        termevents = self.eventstruct.query(['term'], eventslist)
                # initialize event info dictionaries
        Evtimes = {}
        Evpoints = {}
        for (evname, ev) in eventslist:
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
                    ev.starttime = self.indepvariable.depdomain[0]
            else:
                self.eventstruct.resetHighLevelEvents(self.indepvariable.depdomain[0],
                                                  eventslist)
            self.eventstruct.validateEvents(self.funcspec.vars + \
                                            self.funcspec.auxvars + \
                                            ['t'], eventslist)

        # per-iteration storage of variable data (initial values are irrelevant)
        xDataDict = {}
        # storage of all auxiliary variable data
        allaDataDict = {}
        anames = self.funcspec.auxvars
        avals = getattr(self,self.funcspec.auxspec[1])(*[self.indepvariable.depdomain[0],
                       sortedDictValues(self.initialconditions,
                                        self.funcspec.vars),
                       extralist])
        for aix in range(len(anames)):
            aname = anames[aix]
            allaDataDict[aname] = [avals[aix]]
        # temp storage of first time at which terminal events found
        # (this is used for keeping the correct end point of new mesh)
        first_found_t = None
        tmesh = self.indepvariable.depdomain.sample(self.tstep,
                                        strict=False,
                                        avoidendpoints=self.checklevel>2)
        # Main loop
        breakwhile = False
        success = False
        x = ic
        notdone = True
        # did i=0 for initial condition already
        i = 1
        while notdone:
            t = tmesh[i]
            ## COMPUTE NEXT STATE y from x
            try:
                y = rhsfn(t, x, extralist)
            except:
                print("Error in calling right hand side function:")
                self.showSpec()
                raise
            for xi in range(self.dimension):
                xDataDict[xnames[xi]] = y[xi]
                if not self.contains(self.variables[xnames[xi]].depdomain,
                                 y[xi], self.checklevel):
                    self.diagnostics.warnings.append((W_TERMSTATEBD,
                                    (t, xnames[xi], y[xi],
                                     self.variables[xnames[xi]].depdomain)))
                    breakwhile = True
                    break  # for loop
            if breakwhile:
                notdone = False
                continue
            avals = getattr(self,self.funcspec.auxspec[1])(*[t,
                            sortedDictValues(xDataDict),
                            extralist])
            if eventslist != []:
                dataDict = copy(xDataDict)
                dataDict['t'] = t
                evsflagged = self.eventstruct.pollHighLevelEvents(None,
                                                            dataDict,
                                                            self.pars,
                                                            eventslist)
                termevsflagged = [e for e in termevents if e in evsflagged]
                nontermevsflagged = [e for e in evsflagged if e not in termevsflagged]
                # register any non-terminating events in the warnings list
                if len(nontermevsflagged) > 0:
                    evnames = [ev[0] for ev in nontermevsflagged]
                    self.diagnostics.warnings.append((W_NONTERMEVENT,
                                 (t, evnames)))
                    for evname in evnames:
                        Evtimes[evname].append(t)
                        xv = y
                        av = array(avals)
                        Evpoints[evname].append(concatenate((xv, av)))
                if termevsflagged != []:
                    # active terminal event flagged at this time point
                    # register the event in the warnings
                    evnames = [ev[0] for ev in termevsflagged]
                    self.diagnostics.warnings.append((W_TERMEVENT, \
                                             (t, evnames)))
                    for evname in evnames:
                        Evtimes[evname].append(t)
                        xv = y
                        av = array(avals)
                        Evpoints[evname].append(concatenate((xv, av)))
                    notdone = False
                    # ?? if continue here then won't add the event point to the
                    # trajectory values being constructed!
                    #continue
            alltData.append(t)
            for xi in range(self.dimension):
                allxDataDict[xnames[xi]].append(y[xi])
            for aix in range(len(anames)):
                aname = anames[aix]
                allaDataDict[aname].append(avals[aix])
            try:
                extralist[self.numpars:listend] = [f(*[t, self.checklevel]) \
                                              for f in inputVarList]
            except ValueError:
                print('External input call caused value out of range error:', \
                      't = %f' % t)
                for f in inputVarList:
                    if f.hasWarnings():
                        print('External input variable %s out of range:' % f.name)
                        print('   t = %r, %s, %r' % (repr(f.diagnostics.warnings[-1][0]),
                              f.name, repr(f.diagnostics.warnings[-1][1])))
                raise
            except AssertionError:
                print('External input call caused t out of range error: t = %f' % t)
                raise
            if i >= len(tmesh) - 1:
                notdone = False
            else:
                i += 1
                x = y
        # update success flag
        success = not notdone
        # Check that any terminal events found terminated the code correctly
        if first_found_t is not None:
            assert self.diagnostics.warnings[-1][0] == W_TERMEVENT, ("Event finding code "
                                        "for terminal event failed")
        # Package up computed trajectory in Variable variables
        # Add external inputs warnings to self.dignostics.warnings, if any
        for f in inputVarList:
            for winfo in f.diagnostics.warnings:
                self.diagnostics.warnings.append((W_NONTERMSTATEBD,
                                     (winfo[0], f.name, winfo[1],
                                      f.depdomain)))
        # check for non-unique terminal event
        termcount = 0
        for (w,i) in self.diagnostics.warnings:
            if w == W_TERMEVENT or w == W_TERMSTATEBD:
                termcount += 1
                if termcount > 1:
                    self.diagnostics.errors.append((E_NONUNIQUETERM,
                                                    (alltData[-1], i[1])))
##                print 'Time interval adjusted according to %s: %s' % \
##                      (self._warnmessages[w], str(i[0])+", "+ str(i[1]))
        # Create variables (self.variables contains no actual data)
        variables = copyVarDict(self.variables)
        # build event pointset information (reset previous trajectory's)
        self.trajevents = {}
        for (evname,  ev) in eventslist:
            evpt = Evpoints[evname]
            if evpt == []:
                self.trajevents[evname] = None
            else:
                evpt = transpose(array(evpt))
                self.trajevents[evname] = Pointset({'coordnames': xnames+anames,
                                'indepvarname': 't',
                                'coordarray': evpt,
                                'indepvararray': Evtimes[evname],
                                'indepvartype': self.variables[xnames[0]].indepvartype})
        for x in xnames:
            if len(alltData) > 1:
                variables[x] = Variable(Pointset({'coordnames': [x],
                               'coordarray': allxDataDict[x],
                               'coordtype': self.variables[x].coordtype,
                               'indepvarname': 't',
                               'indepvararray': alltData,
                               'indepvartype': self.variables[x].indepvartype}), 't', x, x)
            else:
                raise PyDSTool_ValueError("Fewer than 2 data points computed")
        for a in anames:
            if len(alltData) > 1:
                variables[a] = Variable(Pointset({'coordnames': [a],
                               'coordarray': allaDataDict[a],
                               'coordtype': self.variables[a].coordtype,
                               'indepvarname': 't',
                               'indepvararray': alltData,
                               'indepvartype': self.variables[a].indepvartype}), 't', a, a)
            else:
                raise PyDSTool_ValueError("Fewer than 2 data points computed")

        if success:
            #self.validateSpec()
            self.defined = True
            return Trajectory(trajname, list(variables.values()),
                              abseps=self._abseps, globalt0=self.globalt0,
                              checklevel=self.checklevel,
                              FScompatibleNames=self._FScompatibleNames,
                              FScompatibleNamesInv=self._FScompatibleNamesInv,
                              events=self.trajevents,
                              modelNames=self.name,
                              modelEventStructs=self.eventstruct)
        else:
            print('Trajectory computation failed')
            self.diagnostics.errors.append((E_COMPUTFAIL,
                                            (t, self._errorcodes[errcode])))
            self.defined = False


    def Rhs(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with Model.Rhs"""
        # must convert names to FS-compatible as '.' sorts before letters
        # while '_' sorts after!
        # also, ensure xdict doesn't contain elements like array([4.1]) instead of 4
        x = [float(val) for val in sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                                                 self.funcspec.vars))]
        if pdict is None:
            pdict = self.pars
            # internal self.pars already is FS-compatible
            p = sortedDictValues(pdict)
        else:
            p = sortedDictValues(self._FScompatibleNames(pdict))
        i = _pollInputs(sortedDictValues(self.inputs),
                        t, self.checklevel)
        return getattr(self,self.funcspec.spec[1])(*[t, x, p+i])


    def Jacobian(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.Jacobian"""
        if self.haveJacobian():
            x = [float(val) for val in sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                                                     self.funcspec.vars))]
            if pdict is None:
                pdict = self.pars
                # internal self.pars already is FS-compatible
                p = sortedDictValues(pdict)
            else:
                p = sortedDictValues(self._FScompatibleNames(pdict))
            i = _pollInputs(sortedDictValues(self.inputs),
                        t, self.checklevel)
            return getattr(self,self.funcspec.auxfns["Jacobian"][1])(*[t, x, p+i])
        else:
            raise PyDSTool_ExistError("Jacobian not defined")


    def JacobianP(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.JacobianP"""
        if self.haveJacobian_pars():
            x = [float(val) for val in sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                                                     self.funcspec.vars))]
            if pdict is None:
                pdict = self.pars
                # internal self.pars already is FS-compatible
                p = sortedDictValues(pdict)
            else:
                p = sortedDictValues(self._FScompatibleNames(pdict))
            i = _pollInputs(sortedDictValues(self.inputs),
                        t, self.checklevel)
            return getattr(self,self.funcspec.auxfns["Jacobian_pars"][1])(*[t, x, p+i])
        else:
            raise PyDSTool_ExistError("Jacobian w.r.t. parameters not defined")


    def AuxVars(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.AuxVars"""
        x = [float(val) for val in sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                                                 self.funcspec.vars))]
        if pdict is None:
            pdict = self.pars
            # internal self.pars already is FS-compatible
            p = sortedDictValues(pdict)
        else:
            p = sortedDictValues(self._FScompatibleNames(pdict))
        i = _pollInputs(sortedDictValues(self.inputs),
                        t, self.checklevel)
        return getattr(self,self.funcspec.auxspec[1])(*[t, x, p+i])


    def __del__(self):
        discGen.__del__(self)



# Register this Generator with the database

symbolMapDict = {}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(MapSystem, symbolMapDict, 'python')
