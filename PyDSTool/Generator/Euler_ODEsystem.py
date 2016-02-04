"""Euler integrator for ODE systems, with no step refinement for events.
"""
from __future__ import division, absolute_import, print_function

from .allimports import *
from PyDSTool.Generator import ODEsystem as ODEsystem
from .baseclasses import Generator, theGenSpecHelper, _pollInputs
from PyDSTool.utils import *
from PyDSTool.common import *

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, sign, all, any, \
     array, zeros, less_equal, transpose, concatenate, asarray, linspace
try:
    from numpy import unique
except ImportError:
    # older version of numpy
    from numpy import unique1d as unique
import math, random
import numpy as np
from copy import copy, deepcopy
import os, platform, shutil, sys, gc


class euler_solver(object):
    def __init__(self, rhs):
        self.f = rhs
        self.y = None
        self.t = None

    def set_initial_value(self, y0, t0):
        self.y = y0
        self.t = t0

    def set_f_params(self, extraparams=None):
        if extraparams is None:
            self.f_params = []
        else:
            self.f_params = extraparams

    def set_jac_params(self, extraparams=None):
        """Jacobian is not used"""
        pass

    def integrate(self, dt):
        """Single step"""
        self.t += dt
        self.y = self.y + dt*self.f(self.t, self.y, self.f_params)
        return 1

    def successful(self):
        return True

def _dummy_userfunc(euler):
    pass

class Euler_ODEsystem(ODEsystem):
    """Euler method. Fixed step.

    Uses Python target language only for functional specifications."""

    def __init__(self, kw):
        if 'user_func_beforestep' in kw:
            self.ufunc_before = kw['user_func_beforestep']
            # delete because not covered in ODEsystem
            del kw['user_func_beforestep']
        else:
            self.ufunc_before = _dummy_userfunc
        if 'user_func_afterstep' in kw:
            self.ufunc_after = kw['user_func_afterstep']
            # delete because not covered in ODEsystem
            del kw['user_func_afterstep']
        else:
            self.ufunc_after = _dummy_userfunc
        ODEsystem.__init__(self, kw)
        self._paraminfo = {'init_step': 'Fixed step size for time mesh.'}
        self.diagnostics._errorcodes = {1: 'Step OK'}
        self.diagnostics.outputStatsInfo = {'errorStatus': 'Error status on completion.'}
        algparams_def = {'poly_interp': False,
                         'init_step': 0.01,
                         'max_pts': 100000
                         }
        for k, v in algparams_def.items():
            if k not in self.algparams:
                self.algparams[k] = v


    def addMethods(self):
        # override to add _solver function
        ODEsystem.addMethods(self)
        # Jacobian ignored
        self._solver = euler_solver(getattr(self,self.funcspec.spec[1]))
        self._funcreg['_solver'] = ('self', 'euler_solver(getattr(self,' \
                                                  + 'self.funcspec.spec[1]))')


    def _debug_snapshot(self, solver, dt, inputlist):
        ivals = [i(solver.t) for i in inputlist]
        s = "\n***************\nNew t, x, inputs: " + " ".join([str(s) for s in (solver.t,solver.y,ivals)])
        s += "\ndt="+str(dt)+" f_params="+str(solver.f_params)+" dx/dt="
        s += str(solver.f(solver.t, solver.y, sortedDictValues(self.pars)+ivals))
        return s

    def compute(self, trajname, dirn='f', ics=None):
        continue_integ = ODEsystem.prepDirection(self, dirn)
        if self._dircode == -1:
            raise NotImplementedError('Backwards integration is not implemented')
        if ics is not None:
            self.set(ics=ics)
        self.validateICs()
        self.diagnostics.clearWarnings()
        self.diagnostics.clearErrors()
        pnames = sortedDictKeys(self.pars)
        xnames = self._var_ixmap  # ensures correct order
        # Check i.c.'s are well defined (finite)
        self.checkInitialConditions()
        haveJac = int(self.haveJacobian())
        indepdom0, indepdom1 = self.indepvariable.depdomain.get()
        if continue_integ:
            if indepdom0 > self._solver.t:
                print("Previous end time is %f"%self._solver.t)
                raise ValueError("Start time not correctly updated for "
                                 "continuing orbit")
            x0 = self._solver.y
            indepdom0 = self._solver.t
            self.indepvariable.depdomain.set((indepdom0,indepdom1))
        else:
            x0 = sortedDictValues(self.initialconditions,
                                        self.funcspec.vars)
        t0 = indepdom0
        dt = self.algparams['init_step']
        # speed up repeated access to solver by making a temp name for it
        solver = self._solver
        solver.set_initial_value(x0, t0)
        solver.dt = dt
        # wrap up each dictionary initial value as a singleton list
        alltData = [t0]
        allxDataDict = dict(zip(xnames, map(listid, x0)))
        plist = sortedDictValues(self.pars)
        extralist = copy(plist)
        if self.inputs:
            # inputVarList is a list of Variables
            inames = sortedDictKeys(self.inputs)
            listend = self.numpars + len(self.inputs)
            inputVarList = sortedDictValues(self.inputs)
            ilist = _pollInputs(inputVarList, alltData[0]+self.globalt0,
                               self.checklevel)
        else:
            ilist = []
            inames = []
            listend = self.numpars
            inputVarList = []
        extralist.extend(ilist)
        solver.set_f_params(extralist)
        if haveJac:
            solver.set_jac_params(extralist)
        do_poly = self.algparams['poly_interp']
        if do_poly:
            rhsfn = getattr(self, self.funcspec.spec[1])
            dx0 = rhsfn(t0, x0, extralist)
            alldxDataDict = dict(zip(xnames, map(listid, dx0)))
        auxvarsfn = getattr(self,self.funcspec.auxspec[1])
        # Make t mesh if it wasn't given as 'specialtimes'
        if not all(isfinite(self.indepvariable.depdomain.get())):
            print("Time domain was: %r" % self.indepvariable.depdomain.get())
            raise ValueError("Ensure time domain is finite")
        if dt == indepdom1 - indepdom0:
            # single-step integration required
            # special times will not have been set (unless trivially
            # they are [indepdom0, indepdom1])
            tmesh = [indepdom0, indepdom1]
        else:
            notDone = True
            while notDone:
                tmesh = self.indepvariable.depdomain.sample(dt,
                                          strict=True,
                                          avoidendpoints=True)
                notDone = False
        first_found_t = None
        eventslist = self.eventstruct.query(['active', 'notvarlinked'])
        termevents = self.eventstruct.query(['term'], eventslist)
        tmesh.pop(0)  # get rid of first entry for initial condition
        xnames = self.funcspec.vars
        # storage of all auxiliary variable data
        allaDataDict = {}
        anames = self.funcspec.auxvars
        avals = auxvarsfn(t0, x0, extralist)
        for aix in range(len(anames)):
            aname = anames[aix]
            try:
                allaDataDict[aname] = [avals[aix]]
            except IndexError:
                print("\nEuler generator: There was a problem evaluating " \
                      + "an auxiliary variable")
                print("Debug info: avals (length %d) was %r" % (len(avals), avals))
                print("Index out of range was %d" % aix)
                print(self.funcspec.auxspec[1])
                print(hasattr(self, self.funcspec.auxspec[1]))
                print("Args were: %r" % [t0, x0, extralist])
                raise
        # Initialize signs of event detection objects at IC
        self.setEventICs(self.initialconditions, self.globalt0)
        if self.inputs:
            parsinps = copy(self.pars)
            parsinps.update(dict(zip(inames,ilist)))
        else:
            parsinps = self.pars
        if eventslist != []:
            dataDict = copy(self.initialconditions)
            dataDict.update(dict(zip(anames, avals)))
            dataDict['t'] = t0
            evsflagged = self.eventstruct.pollHighLevelEvents(None,
                                                            dataDict,
                                                            parsinps,
                                                            eventslist)
            if len(evsflagged) > 0:
                raise RuntimeError("Some events flagged at initial condition")
            if continue_integ:
                # revert to prevprevsign, since prevsign changed after call
                self.eventstruct.resetHighLevelEvents(t0, eventslist, 'prev')
            elif self._for_hybrid_DS:
                # self._for_hybrid_DS is set internally by HybridModel class
                # to ensure not to reset events, because they may be about to
                # flag on first step if previous hybrid state was the same
                # generator and, for example, two variables are synchronizing
                # so that their events get very close together.
                # Just reset the starttimes of these events
                for evname, ev in eventslist:
                    ev.starttime = t0
            else:
                # default state is a one-off call to this generator
                self.eventstruct.resetHighLevelEvents(t0, eventslist, None)
                self.eventstruct.validateEvents(self.funcspec.vars + \
                                            self.funcspec.auxvars + \
                                            self.funcspec.inputs + \
                                            ['t'], eventslist)
        evnames = [ev[0] for ev in eventslist]
        lastevtime = {}.fromkeys(evnames, None)
        # initialize new event info dictionaries
        Evtimes = {}
        Evpoints = {}
        if continue_integ:
            for evname in evnames:
                try:
                    # these are in global time, so convert to local time
                    lastevtime[evname] = self.eventstruct.Evtimes[evname][-1] \
                                           - self.globalt0
                except (IndexError, KeyError):
                    # IndexError: Evtimes[evname] was None
                    # KeyError: Evtimes does not have key evname
                    pass
        for evname in evnames:
            Evtimes[evname] = []
            Evpoints[evname] = []
        # temp storage for repeatedly used object attributes (for lookup efficiency)
        depdomains = dict(zip(range(self.dimension),
                        [self.variables[xn].depdomain for xn in xnames]))
        # Main integration loop
        num_points = 0
        breakwhile = False
        while not breakwhile:
            try:
                new_t = tmesh.pop(0)  # this destroys tmesh for future use
            except IndexError:
                # empty
                break
            # optional user function (not a method)
            self.ufunc_before(self)
            try:
                errcode = solver.integrate(dt)
            except:
                print("Error calling right hand side function:")
                self.showSpec()
                print("Numerical traceback information (current state, " \
                      + "parameters, etc.)")
                print("in generator dictionary 'traceback'")
                self.traceback = {'vars': dict(zip(xnames,solver.y)),
                                  'pars': dict(zip(pnames,plist)),
                                  'inputs': dict(zip(inames,ilist)),
                                  self.indepvariable.name: new_t}
                raise
            avals = auxvarsfn(new_t, solver.y, extralist)
            # Uncomment the following assertion for debugging
#            assert all([isfinite(a) for a in avals]), \
#               "Some auxiliary variable values not finite"
            if eventslist != []:
                dataDict = dict(zip(xnames,solver.y))
                dataDict.update(dict(zip(anames,avals)))
                dataDict['t'] = new_t
                if self.inputs:
                    parsinps = copy(self.pars)
                    parsinps.update(dict(zip(inames,
                              _pollInputs(inputVarList, new_t+self.globalt0,
                                          self.checklevel))))
                else:
                    parsinps = self.pars
                evsflagged = self.eventstruct.pollHighLevelEvents(None,
                                                            dataDict,
                                                            parsinps,
                                                            eventslist)
##                print new_t, evsflagged
##                evsflagged = [ev for ev in evsflagged if solver.t-indepdom0 > ev[1].eventinterval]
                termevsflagged = [e for e in termevents if e in evsflagged]
                nontermevsflagged = [e for e in evsflagged if e not in termevsflagged]
                # register any non-terminating events in the warnings
                # list, unless they are 'precise' in which case flag
                # them to be resolved after integration completes
                if len(nontermevsflagged) > 0:
                    evnames = [ev[0] for ev in nontermevsflagged]
                    precEvts = self.eventstruct.query(['precise'],
                                                          nontermevsflagged)
                    # register both precise and non-precise events the same
                    # (Euler currently ignores precise events with its fixed time step)
                    nonprecEvts = self.eventstruct.query(['notprecise'],
                                                         nontermevsflagged) + precEvts
                    nonprec_evnames = [e[0] for e in nonprecEvts] + [e[0] for e in precEvts]
                    # only record events if they have not been previously
                    # flagged within their event interval
                    if nonprec_evnames != []:
                        temp_names = []
                        for evname, ev in nonprecEvts:
                            prevevt_time = lastevtime[evname]
                            if prevevt_time is None:
                                ignore_ev = False
                            else:
                                if solver.t-prevevt_time < ev.eventinterval:
                                    ignore_ev = True
                                else:
                                    ignore_ev = False
                            if not ignore_ev:
                                temp_names.append(evname)
                                lastevtime[evname] = solver.t
                        self.diagnostics.warnings.append((W_NONTERMEVENT,
                                     (solver.t, temp_names)))
                        for evname in temp_names:
                            Evtimes[evname].append(solver.t)
                            xv = solver.y
                            av = array(avals)
                            Evpoints[evname].append(concatenate((xv, av)))
                do_termevs = []
                if termevsflagged != []:
                    # only record events if they have not been previously
                    # flagged within their event interval
                    for e in termevsflagged:
                        prevevt_time = lastevtime[e[0]]
##                        print "Event %s flagged."%e[0]
##                        print "  ... last time was ", prevevt_time
##                        print "  ... event interval = ", e[1].eventinterval
##                        print "  ... t = %f, dt = %f"%(solver.t, dt)
                        if prevevt_time is None:
                            ignore_ev = False
                        else:
##                            print "  ... comparison = %f < %f"%(solver.t-dt-prevevt_time, e[1].eventinterval)
                            if solver.t-dt-prevevt_time < e[1].eventinterval:
                                ignore_ev = True
##                                print "Euler ignore ev"
                            else:
                                ignore_ev = False
                        if not ignore_ev:
                            do_termevs.append(e)
                if len(do_termevs) > 0:
                    # >= 1 active terminal event flagged at this time point
                    evnames = [ev[0] for ev in do_termevs]
                    self.diagnostics.warnings.append((W_TERMEVENT, \
                                         (solver.t, evnames)))
                    first_found_t = solver.t
                    for evname in evnames:
                        Evtimes[evname].append(solver.t)
                        xv = solver.y
                        av = array(avals)
                        Evpoints[evname].append(concatenate((xv, av)))
                    # break while loop after appending t, x
                    breakwhile = True
            # after events have had a chance to be detected at a state boundary
            # we check for any that have not been caught by an event (will be
            # much less accurately determined)
            if not breakwhile:
                # only here if a terminal event hasn't just flagged
                for xi in range(self.dimension):
                    if not self.contains(depdomains[xi],
                                     solver.y[xi],
                                     self.checklevel):
                        self.diagnostics.warnings.append((W_TERMSTATEBD,
                                    (solver.t,
                                     xnames[xi],solver.y[xi],
                                     depdomains[xi].get())))
                        breakwhile = True
                        break  # for loop
                if breakwhile:
                    break  # while loop
            alltData.append(solver.t)
            if do_poly:
                dxvals = rhsfn(solver.t, solver.y, extralist)
                for xi, xname in enumerate(xnames):
                    allxDataDict[xname].append(solver.y[xi])
                    alldxDataDict[xname].append(dxvals[xi])
            else:
                for xi, xname in enumerate(xnames):
                    allxDataDict[xname].append(solver.y[xi])
            for aix, aname in enumerate(anames):
                allaDataDict[aname].append(avals[aix])
            num_points += 1
            if not breakwhile:
                try:
                    extralist[self.numpars:listend] = [f(solver.t+self.globalt0,
                                                         self.checklevel) \
                                                  for f in inputVarList]
                except ValueError:
                    print('External input call caused value out of range error:',\
                          't = %f' % solver.t)
                    for f in inputVarList:
                        if f.diagnostics.hasWarnings():
                            print('External input variable %s out of range:'%f.name)
                            print('   t = %r, %s = %r' % (
                                repr(f.diagnostics.warnings[-1][0]),
                                f.name,
                                repr(f.diagnostics.warnings[-1][1])))
                    raise
                except AssertionError:
                    print('External input call caused t out of range error: t = %f' % solver.t)
                    raise
                solver.set_f_params(extralist)
                breakwhile = not solver.successful()
            # optional user function (not a method)
            self.ufunc_after(self)

        # Check that any terminal events found terminated the code correctly
        if first_found_t is not None:
            # ... then terminal events were found.
            try:
                if self.diagnostics.warnings[-1][0] not in [W_TERMEVENT,
                                                            W_TERMSTATEBD]:
                    print("t =%f" % solver.t)
                    print("state =%r" % dict(zip(xnames,solver.y)))
                    raise RuntimeError("Event finding code for terminal event "
                                       "failed in Generator " + self.name + \
                                       ": try decreasing eventdelay or "
                                       "eventinterval below eventtol, or the "
                                       "atol and rtol parameters")
            except IndexError:
                info(self.diagnostics.outputStats, "Output statistics")
                print("t =%f" % solver.t)
                print("x =%f" % solver.y)
                raise RuntimeError("Event finding failed in Generator " + \
                                   self.name + ": try decreasing eventdelay "
                                   "or eventinterval below eventtol")
        # Package up computed trajectory in Variable variables
        # Add external inputs warnings to self.diagnostics.warnings, if any
        for f in inputVarList:
            for winfo in f.diagnostics.warnings:
                self.diagnostics.warnings.append((W_NONTERMSTATEBD,
                                     (winfo[0], f.name, winfo[1],
                                      f.depdomain.get())))
        # check for non-unique terminal event
        termcount = 0
        for (w,i) in self.diagnostics.warnings:
            if w == W_TERMEVENT or w == W_TERMSTATEBD:
                termcount += 1
                if termcount > 1:
                    self.diagnostics.errors.append((E_NONUNIQUETERM,
                                                    (alltData[-1], i[1])))
        # uncomment the following lines for debugging
#        assert len(alltData) == len(allxDataDict.values()[0]) \
#             == len(allaDataDict.values()[0]), "Output data size mismatch"
#        for val_list in allaDataDict.values():
#            assert all([isfinite(x) for x in val_list])
        # Create variables (self.variables contains no actual data)
        # These versions of the variables are only final if no non-terminal
        # events need to be inserted.
        variables = copyVarDict(self.variables)
        for x in xnames:
            if len(alltData) > 1:
                if do_poly:
                    xvals = array([allxDataDict[x], alldxDataDict[x]]).T
                    interp = PiecewisePolynomial(alltData, xvals, 2)
                else:
                    xvals = allxDataDict[x]
                    interp = interp1d(alltData, xvals)
                variables[x] = Variable(interp, 't', x, x)
            else:
                print("Error in Generator:%s" % self.name)
                print("t = %r" % alltData)
                print("x = %r" % allxDataDict)
                raise PyDSTool_ValueError("Fewer than 2 data points computed")
        for a in anames:
            if len(alltData) > 1:
                variables[a] = Variable(interp1d(alltData, allaDataDict[a]),
                                        't', a, a)
            else:
                print("Error in Generator:%s" % self.name)
                print("t = %r" % alltData)
                print("x = %r" % allxDataDict)
                raise PyDSTool_ValueError("Fewer than 2 data points computed")
        self.diagnostics.outputStats = {'last_step': dt,
                            'num_fcns': num_points,
                            'num_steps': num_points,
                            'errorStatus': errcode
                            }
        if solver.successful():
            #self.validateSpec()
            for evname, evtlist in Evtimes.items():
                try:
                    self.eventstruct.Evtimes[evname].extend([et+self.globalt0 \
                                            for et in evtlist])
                except KeyError:
                    self.eventstruct.Evtimes[evname] = [et+self.globalt0 \
                                            for et in evtlist]
            # build event pointset information (reset previous trajectory's)
            self.trajevents = {}
            for (evname, ev) in eventslist:
                evpt = Evpoints[evname]
                if evpt == []:
                    self.trajevents[evname] = None
                else:
                    evpt = transpose(array(evpt))
                    self.trajevents[evname] = Pointset({'coordnames': xnames+anames,
                                               'indepvarname': 't',
                                               'coordarray': evpt,
                                               'indepvararray': Evtimes[evname],
                                               'indepvartype': float})
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
            try:
                self.diagnostics.errors.append((E_COMPUTFAIL, (solver.t,
                                    self.diagnostics._errorcodes[errcode])))
            except TypeError:
                # e.g. when errcode has been used to return info list
                print("Error information: %d" % errcode)
                self.diagnostics.errors.append((E_COMPUTFAIL, (solver.t,
                                    self.diagnostics._errorcodes[0])))
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
        return getattr(self, self.funcspec.spec[1])(*[t, x, p+i])


    def Jacobian(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.Jacobian"""
        if self.haveJacobian():
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
            return getattr(self, self.funcspec.auxfns["Jacobian"][1])(*[t, x, p+i])
        else:
            raise PyDSTool_ExistError("Jacobian not defined")


    def JacobianP(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.JacobianP"""
        if self.haveJacobian_pars():
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
            return getattr(self, self.funcspec.auxfns["Jacobian_pars"][1])(*[t, x, p+i])
        else:
            raise PyDSTool_ExistError("Jacobian w.r.t. parameters not defined")


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
        i = _pollInputs(sortedDictValues(self.inputs),
                        t, self.checklevel)
        return getattr(self, self.funcspec.auxspec[1])(*[t, x, p+i])


    def __del__(self):
        ODEsystem.__del__(self)



# Register this Generator with the database

symbolMapDict = {}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(Euler_ODEsystem, symbolMapDict, 'python')
