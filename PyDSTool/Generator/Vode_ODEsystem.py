"""VODE integrator for ODE systems, imported from a mild modification of
the scipy-wrapped VODE Fortran solver.
"""

import math
from copy import copy, deepcopy

import numpy as np
from scipy.integrate import ode

from .allimports import *
from .baseclasses import theGenSpecHelper, _pollInputs
from PyDSTool.Generator import ODEsystem as ODEsystem
from PyDSTool.common import *
from PyDSTool.utils import *


# Other imports
try:
    from numpy import unique
except ImportError:
    # older version of numpy
    from numpy import unique1d as unique


class Vode_ODEsystem(ODEsystem):

    """Wrapper for VODE, from SciPy.

    Uses Python target language only for functional specifications.

    """

    _paraminfo = {
        'init_step': 'Fixed step size for time mesh.',
        'strictdt': 'Boolean determining whether to evenly space time mesh '
            '(default=False), or to use exactly dt spacing.',
        'stiff': 'Boolean to activate the BDF method, otherwise Adams method '
            'used. Default False.',
        'use_special': 'Switch for using special times',
        'specialtimes': 'List of special times to use during integration',
    }

    def __init__(self, kw):
        ODEsystem.__init__(self, kw)

        self.diagnostics._errorcodes = \
            {0: 'Unrecognized error code returned (see stderr output)',
             -1: 'Excess work done on this call. (Perhaps wrong method MF.)',
             -2: 'Excess accuracy requested. (Tolerances too small.)',
             -3: 'Illegal input detected. (See printed message.)',
             -4: 'Repeated error test failures. (Check all input.)',
             -5: 'Repeated convergence failures. (Perhaps bad'
             ' Jacobian supplied or wrong choice of method MF or tolerances.)',
             -6: 'Error weight became zero during problem. (Solution'
             ' component i vanished, and ATOL or ATOL(i) = 0.)'
             }
        self.diagnostics.outputStatsInfo = {
            'errorStatus': 'Error status on completion.'
        }
        # note: VODE only supports array atol, not rtol.
        algparams_def = {
            'poly_interp': False,
            'rtol': 1e-9,
            'atol': [1e-12 for dimix in range(self.dimension)],
            'stiff': False,
            'max_step': 0.0,
            'min_step': 0.0,
            'init_step': 0.01,
            'max_pts': 1000000,
            'strictdt': False,
            'use_special': False,
            'specialtimes': []
        }
        for k, v in algparams_def.items():
            if k not in self.algparams:
                self.algparams[k] = v

    def addMethods(self):
        # override to add _solver function
        ODEsystem.addMethods(self)
        if self.haveJacobian():
            self._solver = ode(
                getattr(self, self.funcspec.spec[1]),
                getattr(self, self.funcspec.auxfns['Jacobian'][1]))
            self._funcreg['_solver'] = ('self',
                                        'ode(getattr(self,self.funcspec.spec[1]),'
                                        + 'getattr(self,self.funcspec.auxfns["Jacobian"][1]))')
        else:
            self._solver = ode(getattr(self, self.funcspec.spec[1]))
            self._funcreg['_solver'] = ('self', 'ode(getattr(self,'
                                        + 'self.funcspec.spec[1]))')

    def _debug_snapshot(self, solver, dt, inputlist):
        ivals = [i(solver.t) for i in inputlist]
        s = '\n***************\nNew t, x, inputs: ' + ' '.join(
            [str(s) for s in (solver.t, solver.y, ivals)])
        s += '\ndt=' + str(dt) + ' f_params=' + str(
            solver.f_params) + ' dx/dt='
        s += str(solver.f(solver.t, solver.y,
                          sortedDictValues(self.pars) + ivals))
        if solver.t > 7:
            s += '\nfirst, max, min steps =' + str(
                [solver.first_step, solver.max_step, solver.min_step])
        return s

    def compute(self, trajname, dirn='f', ics=None):
        continue_integ = ODEsystem.prepDirection(self, dirn)
        if self._dircode == -1:
            raise NotImplementedError(
                'Backwards integration is not implemented')
        if ics is not None:
            self.set(ics=ics)
        self.validateICs()
        self.diagnostics.clearWarnings()
        self.diagnostics.clearErrors()
        pnames = sortedDictKeys(self.pars)
        xnames = self._var_ixmap  # ensures correct order
        # Check i.c.'s are well defined (finite)
        self.checkInitialConditions()
        if self.algparams['stiff']:
            methstr = 'bdf'
            methcode = 2
        else:
            methstr = 'adams'
            methcode = 1
        haveJac = int(self.haveJacobian())
        if isinstance(self.algparams['atol'], list):
            if len(self.algparams['atol']) != self.dimension:
                raise ValueError('atol list must have same length as phase '
                                 'dimension')
        else:
            atol = self.algparams['atol']
            self.algparams['atol'] = [atol for dimix in range(self.dimension)]
        indepdom0, indepdom1 = self.indepvariable.depdomain.get()
        if continue_integ:
            if indepdom0 > self._solver.t:
                print('Previous end time is %f' % self._solver.t)
                raise ValueError('Start time not correctly updated for '
                                 'continuing orbit')
            x0 = self._solver.y
            indepdom0 = self._solver.t
            self.indepvariable.depdomain.set((indepdom0, indepdom1))
        else:
            x0 = sortedDictValues(self.initialconditions, self.funcspec.vars)
        t0 = indepdom0
        if self.algparams['use_special']:
            tmesh_special = list(self.algparams['specialtimes'])
            if continue_integ:
                if self._solver.t not in tmesh_special:
                    raise ValueError('Invalid time to continue integration:'
                                     "it is not in 'special times'")
                tmesh_special = tmesh_special[tmesh_special.index(
                    self._solver.t):]
            try:
                dt = min([tmesh_special[1] - t0, self.algparams['init_step']])
            except:
                raise ValueError('Invalid list of special times')
            if not isincreasing(tmesh_special):
                raise ValueError('special times must be given in increasing '
                                 'order')
            if self.indepvariable.depdomain.contains(t0) is notcontained or \
                    self.indepvariable.depdomain.contains(tmesh_special[-1]) is notcontained:
                raise PyDSTool_BoundsError(
                    'special times were outside of independent '
                    'variable bounds')
        else:
            tmesh_special = []
            dt = self.algparams['init_step']
        # speed up repeated access to solver by making a temp name for it
        solver = self._solver
        if getattr(solver, '_integrator', None) is None:
            # Banded Jacobians not yet supported
            #
            # start a new integrator, because method may have been
            # switched
            solver.set_integrator('vode',
                                  method=methstr,
                                  rtol=self.algparams['rtol'],
                                  atol=self.algparams['atol'],
                                  nsteps=self.algparams['max_pts'],
                                  max_step=self.algparams['max_step'],
                                  min_step=self.algparams['min_step'],
                                  first_step=dt / 2.,
                                  with_jacobian=haveJac)
        else:
            solver.with_jacobian = haveJac
            # self.mu = lband
            # self.ml = uband
            solver.rtol = self.algparams['rtol']
            solver.atol = self.algparams['atol']
            solver.method = methcode
            # self.order = order
            solver.nsteps = self.algparams['max_pts']
            solver.max_step = self.algparams['max_step']
            solver.min_step = self.algparams['min_step']
            solver.first_step = dt / 2.
        solver.set_initial_value(x0, t0)
        # Broken code for going backwards (doesn't respect 'continue' option
        # either)
        # if self._dircode == 1:
        #     solver.set_initial_value(x0, t0)
        # else:
        #     solver.set_initial_value(x0, indepdom1)
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
            ilist = _pollInputs(inputVarList, alltData[0] + self.globalt0,
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
        auxvarsfn = getattr(self, self.funcspec.auxspec[1])
        strict = self.algparams['strictdt']
        # Make t mesh if it wasn't given as 'specialtimes'
        if not np.all(np.isfinite(self.indepvariable.depdomain.get())):
            print('Time domain was: %f' % self.indepvariable.depdomain.get())
            raise ValueError('Ensure time domain is finite')
        if dt == indepdom1 - indepdom0:
            # single-step integration required
            # special times will not have been set (unless trivially
            # they are [indepdom0, indepdom1])
            tmesh = [indepdom0, indepdom1]
        else:
            notDone = True
            repeatTol = 10
            count = 0
            while notDone and count <= repeatTol:
                try:
                    tmesh = self.indepvariable.depdomain.sample(
                        dt,
                        strict=strict,
                        avoidendpoints=self.checklevel > 2)
                    notDone = False
                except AssertionError:
                    count += 1
                    dt = dt / 3.0
            if count == repeatTol:
                raise AssertionError('Supplied time step is too large for '
                                     'selected time interval')
            # incorporate tmesh_special, if used, ensuring uniqueness
            if tmesh_special != []:
                tmesh.extend(tmesh_special)
                tmesh = list(unique(tmesh))
                tmesh.sort()
            if len(tmesh) <= 2:
                # safety net, in case too few points in mesh
                # too few points unless we can add endpoint
                if tmesh[-1] != indepdom1:
                    # dt too large for tmesh to have more than one point
                    tmesh.append(indepdom1)
            if not strict:  # get actual time step used
                # don't use [0] in case avoided end points
                try:
                    dt = tmesh[2] - tmesh[1]
                except IndexError:
                    # can't avoid end points for such a small mesh
                    dt = tmesh[1] - tmesh[0]
        # if self.eventstruct.query(['lowlevel']) != []:
        #    raise ValueError("Only high level events can be passed to VODE")
        eventslist = self.eventstruct.query(['active', 'notvarlinked'])
        termevents = self.eventstruct.query(['term'], eventslist)
        # reverse time by reversing mesh doesn't work
        # if self._dircode == -1:
        #     tmesh.reverse()
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
                print('\nVODE generator: There was a problem evaluating '
                      + 'an auxiliary variable')
                print('Debug info: avals (length %d) was %r' %
                      (len(avals), avals))
                print('Index out of range was %d' % aix)
                print(self.funcspec.auxspec[1])
                print(hasattr(self, self.funcspec.auxspec[1]))
                print('Args were:%r' % [t0, x0, extralist])
                raise
        # Initialize signs of event detection objects at IC
        self.setEventICs(self.initialconditions, self.globalt0)
        if self.inputs:
            parsinps = copy(self.pars)
            parsinps.update(dict(zip(inames, ilist)))
        else:
            parsinps = self.pars
        if eventslist != []:
            dataDict = copy(self.initialconditions)
            dataDict.update(dict(zip(anames, avals)))
            dataDict['t'] = t0
            # Removed this "belt and braces" pre-check because it messes
            # up some events in the middle of hybrid regime switches.
            # Not sure yet what the problem is.  (Oct 2013)
            # evsflagged = self.eventstruct.pollHighLevelEvents(None,
            #                                                dataDict,
            #                                                parsinps,
            #                                                eventslist)
            # if len(evsflagged) > 0:
            #    raise RuntimeError("Some events flagged at initial condition")
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
                self.eventstruct.validateEvents(self.funcspec.vars +
                                                self.funcspec.auxvars +
                                                self.funcspec.inputs +
                                                ['t'], eventslist)
        # temp storage of first time at which terminal events found
        # (this is used for keeping the correct end point of new mesh)
        first_found_t = None
        # list of precise non-terminal events to be resolved after integration
        nontermprecevs = []
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
        # temp storage for repeatedly used object attributes (for lookup
        # efficiency)
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
            # a record of previous time for use by event detector
            last_t = solver.t
            try:
                y_ignored = solver.integrate(new_t)
            except:
                print('Error calling right hand side function:')
                self.showSpec()
                print('Numerical traceback information (current state, '
                      + 'parameters, etc.)')
                print("in generator dictionary 'traceback'")
                self.traceback = {
                    'vars': dict(zip(xnames, solver.y)),
                    'pars': dict(zip(pnames, plist)),
                    'inputs': dict(zip(inames, ilist)),
                    self.indepvariable.name: new_t
                }
                raise
            avals = auxvarsfn(new_t, solver.y, extralist)
            # Uncomment the following assertion for debugging
            # assert all([isfinite(a) for a in avals]), \
            #    "Some auxiliary variable values not finite"
            if eventslist != []:
                dataDict = dict(zip(xnames, solver.y))
                dataDict.update(dict(zip(anames, avals)))
                dataDict['t'] = new_t
                if self.inputs:
                    parsinps = copy(self.pars)
                    parsinps.update(dict(zip(inames,
                                             _pollInputs(inputVarList,
                                                         new_t + self.globalt0,
                                                         self.checklevel))))
                else:
                    parsinps = self.pars
                evsflagged = self.eventstruct.pollHighLevelEvents(
                    None, dataDict, parsinps, eventslist)
                # print new_t, evsflagged
                # evsflagged = [ev for ev in evsflagged if solver.t-indepdom0 > ev[1].eventinterval]
                termevsflagged = [e for e in termevents if e in evsflagged]
                nontermevsflagged = [e for e in evsflagged
                                     if e not in termevsflagged]
                # register any non-terminating events in the warnings
                # list, unless they are 'precise' in which case flag
                # them to be resolved after integration completes
                if len(nontermevsflagged) > 0:
                    evnames = [ev[0] for ev in nontermevsflagged]
                    precEvts = self.eventstruct.query(['precise'],
                                                      nontermevsflagged)
                    prec_evnames = [e[0] for e in precEvts]
                    # first register non-precise events
                    nonprecEvts = self.eventstruct.query(['notprecise'],
                                                         nontermevsflagged)
                    nonprec_evnames = [e[0] for e in nonprecEvts]
                    # only record events if they have not been previously
                    # flagged within their event interval
                    if nonprec_evnames != []:
                        temp_names = []
                        for evname, ev in nonprecEvts:
                            prevevt_time = lastevtime[evname]
                            if prevevt_time is None:
                                ignore_ev = False
                            else:
                                if solver.t - prevevt_time < ev.eventinterval:
                                    ignore_ev = True
                                else:
                                    ignore_ev = False
                            if not ignore_ev:
                                temp_names.append(evname)
                                lastevtime[evname] = solver.t
                        self.diagnostics.warnings.append((
                            W_NONTERMEVENT, (solver.t, temp_names)))
                        for evname in temp_names:
                            Evtimes[evname].append(solver.t)
                            xv = solver.y
                            av = np.array(avals)
                            Evpoints[evname].append(np.concatenate((xv, av)))
                    for evname, ev in precEvts:
                        # only record events if they have not been previously
                        # flagged within their event interval
                        prevevt_time = lastevtime[evname]
                        if prevevt_time is None:
                            ignore_ev = False
                        else:
                            if last_t - prevevt_time < ev.eventinterval:
                                ignore_ev = True
                            else:
                                ignore_ev = False
                        if not ignore_ev:
                            nontermprecevs.append((last_t, solver.t,
                                                   (evname, ev)))
                            # be conservative as to where the event is, so
                            # that don't miss any events.
                            lastevtime[evname] = last_t  # solver.t-dt
                        ev.reset()  # ev.prevsign = None #
                do_termevs = []
                if termevsflagged != []:
                    # only record events if they have not been previously
                    # flagged within their event interval
                    for e in termevsflagged:
                        prevevt_time = lastevtime[e[0]]
                        # print "Event %s flagged."%e[0]
                        # print "  ... last time was ", prevevt_time
                        # print "  ... event interval = ", e[1].eventinterval
                        # print "  ... t = %f, dt = %f"%(solver.t, dt)
                        if prevevt_time is None:
                            ignore_ev = False
                        else:
                            # print "  ... comparison = %f <
                            # %f"%(solver.t-dt-prevevt_time,
                            # e[1].eventinterval)
                            if last_t - prevevt_time < e[1].eventinterval:
                                ignore_ev = True
                                # print "VODE ignore ev"
                            else:
                                ignore_ev = False
                        if not ignore_ev:
                            do_termevs.append(e)
                if len(do_termevs) > 0:
                    # >= 1 active terminal event flagged at this time point
                    if all([not ev[1].preciseFlag for ev in do_termevs]):
                        # then none of the events specify greater accuracy
                        # register the event in the warnings
                        evnames = [ev[0] for ev in do_termevs]
                        self.diagnostics.warnings.append((W_TERMEVENT,
                                                          (solver.t, evnames)))
                        for evname in evnames:
                            Evtimes[evname].append(solver.t)
                            xv = solver.y
                            av = np.array(avals)
                            Evpoints[evname].append(np.concatenate((xv, av)))
                        # break while loop after appending t, x
                        breakwhile = True
                    else:
                        # find which are the 'precise' events that flagged
                        precEvts = self.eventstruct.query(['precise'],
                                                          do_termevs)
                        # these events have flagged once so eventdelay has
                        # been used. now switch it off while finding event
                        # precisely (should be redundant after change to
                        # eventinterval and eventdelay parameters)
                        evnames = [ev[0] for ev in precEvts]
                        if first_found_t is None:
                            # print "first time round at", solver.t
                            numtries = 0
                            first_found_t = solver.t
                            restore_evts = deepcopy(precEvts)
                            minbisectlimit = min([ev[1].bisectlimit
                                                  for ev in precEvts])
                            for ev in precEvts:
                                ev[1].eventdelay = 0.
                        else:
                            numtries += 1
                            # print "time round: ", numtries
                            if numtries > minbisectlimit:
                                self.diagnostics.warnings.append((
                                    W_BISECTLIMIT, (solver.t, evnames)))
                                breakwhile = True

                        # get previous time point
                        if len(alltData) >= 1:
                            # take one step back -> told, which will
                            # get dt added back to first new meshpoint
                            # (solver.t is the step *after* the event was
                            # detected)
                            # solver.t-dt without the loss of precision from
                            # subtraction
                            told = last_t
                        else:
                            raise ValueError('Event %s found too ' % evnames[0] +
                                             'close to local start time: try decreasing '
                                             'initial step size (current size is '
                                             '%f @ t=%f)' % (dt, solver.t + self.globalt0))

                        # absolute tolerance check on event function values
                        # (not t)
                        in_tols = [abs(e[1].fval) < e[1].eventtol
                                   for e in precEvts]
                        if all(in_tols):
                            # print "Registering event:", dt_min, dt
                            # register the event in the warnings
                            self.diagnostics.warnings.append((
                                W_TERMEVENT, (solver.t, evnames)))
                            for evname in evnames:
                                Evtimes[evname].append(solver.t)
                                xv = solver.y
                                av = np.array(avals)
                                Evpoints[evname].append(np.concatenate((xv, av)))
                            # Cannot continue -- dt_min no smaller than
                            # previous dt. If this is more than the first time
                            # in this code then have found the event to within
                            # the minimum 'precise' event's eventtol, o/w need
                            # to set eventtol smaller.
                            # Before exiting event-finding loop, reset prevsign
                            # of flagged events
                            self.eventstruct.resetHighLevelEvents(0, precEvts)
                            # while loop, but append point first
                            breakwhile = True
                        if not breakwhile:
                            dt_new = dt / 5.0
                            # calc new tmesh
                            trangewidth = 2 * dt  # first_found_t - told
                            numpoints = int(math.ceil(trangewidth / dt_new))
                            # choose slightly smaller dt to fit trange exactly
                            dt = trangewidth / numpoints

                            tmesh = [told + i * dt
                                     for i in range(1, numpoints + 1)]
                            # linspace version is *much* slower for numpoints ~ 10 and 100
                            #tmesh = list(told+linspace(dt, numpoints*dt, numpoints))

                            # reset events according to new time mesh,
                            # setting known previous event state to be their
                            # "no event found" state
                            self.eventstruct.resetHighLevelEvents(told,
                                                                  precEvts,
                                                                  state='off')
                            # build new ic with last good values (at t=told)
                            if len(alltData) > 1:
                                new_ic = [allxDataDict[xname][-1]
                                          for xname in xnames]
                            else:
                                new_ic = x0
                            # reset integrator
                            solver.set_initial_value(new_ic, told)
                            extralist[self.numpars:listend] = _pollInputs(
                                inputVarList, told + self.globalt0,
                                self.checklevel)
                            solver.set_f_params(extralist)
                            # continue integrating over new mesh
                            continue  # while
            # after events have had a chance to be detected at a state boundary
            # we check for any that have not been caught by an event (will be
            # much less accurately determined)
            if not breakwhile:
                # only here if a terminal event hasn't just flagged
                for xi in range(self.dimension):
                    if not self.contains(depdomains[xi], solver.y[xi],
                                         self.checklevel):
                        self.diagnostics.warnings.append((
                            W_TERMSTATEBD, (solver.t, xnames[xi], solver.y[xi],
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
                    extralist[self.numpars:listend] = [f(solver.t + self.globalt0,
                                                         self.checklevel)
                                                       for f in inputVarList]
                except ValueError:
                    print('External input call caused value out of range error:',
                          't = %f' % solver.t)
                    for f in inputVarList:
                        if f.diagnostics.hasWarnings():
                            print('External input variable %s out of range:' %
                                  f.name)
                            print('   t = %r, %s, %r' %
                                  (repr(f.diagnostics.warnings[-1][0]), f.name,
                                   repr(f.diagnostics.warnings[-1][1])))
                    raise
                except AssertionError:
                    print(
                        'External input call caused t out of range error: t = %f'
                        % solver.t)
                    raise
                solver.set_f_params(extralist)
                breakwhile = not solver.successful()
        # Check that any terminal events found terminated the code correctly
        if first_found_t is not None:
            # ... then terminal events were found. Those that were 'precise' had
            # their 'eventdelay' attribute temporarily set to 0. It now should
            # be restored.
            for evname1, ev1 in termevents:
                # restore_evts are copies of the originally flagged 'precise'
                # events
                for evname2, ev2 in restore_evts:
                    if evname2 == evname1:
                        ev1.eventdelay = ev2.eventdelay
            try:
                if self.diagnostics.warnings[-1][0] not in [W_TERMEVENT,
                                                            W_TERMSTATEBD]:
                    print('t =%f' % solver.t)
                    print('state =%r' % dict(zip(xnames, solver.y)))
                    raise RuntimeError('Event finding code for terminal event '
                                       'failed in Generator ' + self.name +
                                       ': try decreasing eventdelay or '
                                       'eventinterval below eventtol, or the '
                                       'atol and rtol parameters')
            except IndexError:
                info(self.diagnostics.outputStats, 'Output statistics')
                print('t =%f' % solver.t)
                print('x =%f' % solver.y)
                raise RuntimeError('Event finding failed in Generator ' +
                                   self.name + ': try decreasing eventdelay '
                                   'or eventinterval below eventtol, or the '
                                   'atol and rtol parameters')
        # Package up computed trajectory in Variable variables
        # Add external inputs warnings to self.diagnostics.warnings, if any
        for f in inputVarList:
            for winfo in f.diagnostics.warnings:
                self.diagnostics.warnings.append((W_NONTERMSTATEBD,
                                                  (winfo[0], f.name, winfo[1],
                                                   f.depdomain.get())))
        # check for non-unique terminal event
        termcount = 0
        for (w, i) in self.diagnostics.warnings:
            if w == W_TERMEVENT or w == W_TERMSTATEBD:
                termcount += 1
                if termcount > 1:
                    self.diagnostics.errors.append((E_NONUNIQUETERM,
                                                    (alltData[-1], i[1])))
        # uncomment the following lines for debugging
        # assert len(alltData) == len(allxDataDict.values()[0]) \
        #      == len(allaDataDict.values()[0]), "Output data size mismatch"
        # for val_list in allaDataDict.values():
        #     assert all([isfinite(x) for x in val_list])
        # Create variables (self.variables contains no actual data)
        # These versions of the variables are only final if no non-terminal
        # events need to be inserted.
        variables = copyVarDict(self.variables)
        for x in xnames:
            if len(alltData) > 1:
                if do_poly:
                    xvals = np.array([allxDataDict[x], alldxDataDict[x]]).T
                    interp = PiecewisePolynomial(alltData, xvals, 2)
                else:
                    xvals = allxDataDict[x]
                    interp = interp1d(alltData, xvals)
                variables[x] = Variable(interp, 't', x, x)
            else:
                print('Error in Generator:%s' % self.name)
                print('t = %r' % alltData)
                print('x = %r' % allxDataDict)
                raise PyDSTool_ValueError('Fewer than 2 data points computed')
        for a in anames:
            if len(alltData) > 1:
                variables[a] = Variable(interp1d(alltData, allaDataDict[a]),
                                        't', a, a)
            else:
                print('Error in Generator:%s' % self.name)
                print('t = %r' % alltData)
                print('x = %r' % allxDataDict)
                raise PyDSTool_ValueError('Fewer than 2 data points computed')
        # Resolve non-terminal 'precise' events that were flagged, using the
        # variables created. Then, add them to a new version of the variables.
        ntpe_tdict = {}
        for (et0, et1, e) in nontermprecevs:
            lost_evt = False
            # problem if eventinterval > et1-et0 !!
            # was: search_dt = max((et1-et0)/5,e[1].eventinterval)
            search_dt = (et1 - et0) / 5
            try:
                et_precise_list = e[1].searchForEvents(
                    trange=[et0, et1],
                    dt=search_dt,
                    checklevel=self.checklevel,
                    parDict=self.pars,
                    vars=variables,
                    inputs=self.inputs,
                    abseps=self._abseps,
                    eventdelay=False,
                    globalt0=self.globalt0)
            except (ValueError, PyDSTool_BoundsError):
                # dt too large for trange, e.g. if event tol is smaller than
                # time mesh
                et_precise_list = [(et0, (et0, et1))]
            if et_precise_list == []:
                lost_evt = True
            for et_precise in et_precise_list:
                if et_precise[0] is not None:
                    if et_precise[0] in ntpe_tdict:
                        # add event name at this time (that already exists in
                        # the dict)
                        ntpe_tdict[et_precise[0]].append(e[0])
                    else:
                        # add event name at this time (when time is not already
                        # in dict)
                        ntpe_tdict[et_precise[0]] = [e[0]]
                else:
                    lost_evt = True
            if lost_evt:
                print(
                    "Error: A non-terminal, 'precise' event was lost -- did you reset",
                    end=' ')
                print('events prior to integration?')
                raise PyDSTool_ExistError(
                    'Internal error: A non-terminal, '
                    "'precise' event '%s' was lost after integration!" % e[0])
        # add non-terminal event points to variables
        if ntpe_tdict != {}:
            # find indices of times at which event times will be inserted
            tix = 0
            evts = list(ntpe_tdict.keys())
            evts.sort()
            for evix in range(len(evts)):
                evt = evts[evix]
                evnames = ntpe_tdict[evt]
                self.diagnostics.warnings.append((W_NONTERMEVENT,
                                                  (evt, evnames)))
                xval = [variables[x](evt) for x in xnames]
                ilist = _pollInputs(inputVarList, evt + self.globalt0,
                                    self.checklevel)
                # find accurate dx and aux vars value at this time
                if do_poly:
                    dxval = rhsfn(evt, xval, plist + ilist)
                aval = list(auxvarsfn(evt, xval, plist + ilist))
                for evname in evnames:
                    Evtimes[evname].append(evt)
                    Evpoints[evname].append(np.array(xval + aval))
                tcond = np.less_equal(alltData[tix:], evt).tolist()
                try:
                    tix = tcond.index(0) + tix  # lowest index for t > evt
                    do_insert = (alltData[tix - 1] != evt)
                except ValueError:
                    # evt = last t value so no need to add it
                    do_insert = False
                if do_insert:
                    alltData.insert(tix, evt)
                    for ai, a in enumerate(anames):
                        allaDataDict[a].insert(tix, aval[ai])
                    if do_poly:
                        for xi, x in enumerate(xnames):
                            allxDataDict[x].insert(tix, xval[xi])
                            alldxDataDict[x].insert(tix, dxval[xi])
                    else:
                        for xi, x in enumerate(xnames):
                            allxDataDict[x].insert(tix, xval[xi])
            for x in xnames:
                if do_poly:
                    # use asarray in case points added to sequences as a list
                    xvals = np.array([np.asarray(allxDataDict[x]),
                                   np.asarray(alldxDataDict[x])]).T
                    interp = PiecewisePolynomial(alltData, xvals, 2)
                else:
                    xvals = allxDataDict[x]
                    interp = interp1d(alltData, xvals)
                variables[x] = Variable(interp, 't', x, x)
            for a in anames:
                variables[a] = Variable(interp1d(alltData, allaDataDict[a]),
                                        't', a, a)
        self.diagnostics.outputStats = {
            'last_step': dt,
            'num_fcns': num_points,
            'num_steps': num_points,
            'errorStatus': solver._integrator.success
        }
        if solver.successful():
            # self.validateSpec()
            for evname, evtlist in Evtimes.items():
                try:
                    self.eventstruct.Evtimes[evname].extend([et + self.globalt0
                                                             for et in evtlist])
                except KeyError:
                    self.eventstruct.Evtimes[evname] = [et + self.globalt0
                                                        for et in evtlist]
            # build event pointset information (reset previous trajectory's)
            self.trajevents = {}
            for (evname, ev) in eventslist:
                evpt = Evpoints[evname]
                if evpt == []:
                    self.trajevents[evname] = None
                else:
                    evpt = np.transpose(np.array(evpt))
                    self.trajevents[evname] = Pointset({
                        'coordnames': xnames + anames,
                        'indepvarname': 't',
                        'coordarray': evpt,
                        'indepvararray': Evtimes[evname],
                        'indepvartype': float
                    })
            self.defined = True
            return Trajectory(trajname,
                              list(variables.values()),
                              abseps=self._abseps,
                              globalt0=self.globalt0,
                              checklevel=self.checklevel,
                              FScompatibleNames=self._FScompatibleNames,
                              FScompatibleNamesInv=self._FScompatibleNamesInv,
                              events=self.trajevents,
                              modelNames=self.name,
                              modelEventStructs=self.eventstruct)
        else:
            try:
                errcode = solver._integrator.success  # integer
                self.diagnostics.errors.append((E_COMPUTFAIL, (
                    solver.t, self.diagnostics._errorcodes[errcode])))
            except TypeError:
                # e.g. when errcode has been used to return info list
                print('Error information: %d' % errcode)
                self.diagnostics.errors.append((E_COMPUTFAIL, (
                    solver.t, self.diagnostics._errorcodes[0])))
            self.defined = False

    def Rhs(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.Rhs."""
        # must convert names to FS-compatible as '.' sorts before letters
        # while '_' sorts after!
        # also, ensure xdict doesn't contain elements like array([4.1]) instead
        # of 4
        x = [float(val) for val in sortedDictValues(filteredDict(
            self._FScompatibleNames(xdict), self.funcspec.vars))]
        if pdict is None:
            pdict = self.pars
            # internal self.pars already is FS-compatible
            p = sortedDictValues(pdict)
        else:
            p = sortedDictValues(self._FScompatibleNames(pdict))
        i = _pollInputs(sortedDictValues(self.inputs), t, self.checklevel)
        return getattr(self, self.funcspec.spec[1])(*[t, x, p + i])

    def Jacobian(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.Jacobian."""
        if self.haveJacobian():
            # also, ensure xdict doesn't contain elements like array([4.1])
            # instead of 4
            x = [float(val) for val in sortedDictValues(filteredDict(
                self._FScompatibleNames(xdict), self.funcspec.vars))]
            if pdict is None:
                pdict = self.pars
                # internal self.pars already is FS-compatible
                p = sortedDictValues(pdict)
            else:
                p = sortedDictValues(self._FScompatibleNames(pdict))
            i = _pollInputs(sortedDictValues(self.inputs), t, self.checklevel)
            return getattr(self, self.funcspec.auxfns['Jacobian'][1])(*[t, x,
                                                                        p + i])
        else:
            raise PyDSTool_ExistError('Jacobian not defined')

    def JacobianP(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.JacobianP."""
        if self.haveJacobian_pars():
            # also, ensure xdict doesn't contain elements like array([4.1])
            # instead of 4
            x = [float(val) for val in sortedDictValues(filteredDict(
                self._FScompatibleNames(xdict), self.funcspec.vars))]
            if pdict is None:
                pdict = self.pars
                # internal self.pars already is FS-compatible
                p = sortedDictValues(pdict)
            else:
                p = sortedDictValues(self._FScompatibleNames(pdict))
            i = _pollInputs(sortedDictValues(self.inputs), t, self.checklevel)
            return getattr(self, self.funcspec.auxfns['Jacobian_pars'][1])(
                *[t, x, p + i])
        else:
            raise PyDSTool_ExistError('Jacobian w.r.t. parameters not defined')

    def AuxVars(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.AuxVars."""
        # also, ensure xdict doesn't contain elements like array([4.1]) instead
        # of 4
        x = [float(val) for val in sortedDictValues(filteredDict(
            self._FScompatibleNames(xdict), self.funcspec.vars))]
        if pdict is None:
            pdict = self.pars
            # internal self.pars already is FS-compatible
            p = sortedDictValues(pdict)
        else:
            p = sortedDictValues(self._FScompatibleNames(pdict))
        i = _pollInputs(sortedDictValues(self.inputs), t, self.checklevel)
        return getattr(self, self.funcspec.auxspec[1])(*[t, x, p + i])

    def __del__(self):
        ODEsystem.__del__(self)

# Register this Generator with the database

symbolMapDict = {}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(Vode_ODEsystem, symbolMapDict, 'python')
