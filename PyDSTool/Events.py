"""Event handling for python-based computations, and specification for
both python and externally compiled code. (Externally compiled code
may include its own event determination implementation.)

"High-level" events are built in native Python function format.
"Low-level" events are built for external platforms, e.g. C or Matlab code.

    Robert Clewley, October 2005.
"""
from __future__ import absolute_import, print_function

# PyDSTool imports
from .Variable import *
from .utils import *
from .common import *
from .parseUtils import *
from .errors import *
from .Interval import *
from .utils import info as utils_info
from .Symbolic import QuantSpec, Var, ensureStrArgDict
from . import FuncSpec

# Other imports
import scipy, numpy, scipy, scipy.special
import math, random
import copy, types
import six
from six.moves import reduce

__all__ = ['EventStruct', 'Event',
            'HighLevelEvent', 'LowLevelEvent', 'MatlabEvent',
            'makePythonStateZeroCrossEvent', 'makeZeroCrossEvent']


# helper functions
def _highlevel(arg):
    return isinstance(arg[1], HighLevelEvent)

def _lowlevel(arg):
    return isinstance(arg[1], LowLevelEvent)

def _term(arg):
    return arg[1].termFlag

def _nonterm(arg):
    return not arg[1].termFlag

def _active(arg):
    return arg[1].activeFlag

def _notactive(arg):
    return not arg[1].activeFlag

def _varlinked(arg):
    return arg[1].varlinked

def _notvarlinked(arg):
    return not arg[1].varlinked

def _precise(arg):
    return arg[1].preciseFlag

def _notprecise(arg):
    return not arg[1].preciseFlag


# ---------------------------------------------------------------------------

class EventStruct(object):
    """A data structure to store and interface with multiple events."""

    def __init__(self):
        self.events = {}
        self._keylist = ['term', 'nonterm', 'active', 'varlinked', 'precise',
                         'notprecise', 'notvarlinked', 'notactive', 'highlevel',
                         'lowlevel']
        self._makeFilterDict()
        # record of recent events (mainly for checking event interval
        # in high-level events)
        self.resetEvtimes()

    def _makeFilterDict(self):
        self._filterDict = {}
        self._filterDict['highlevel'] = _highlevel
        self._filterDict['lowlevel'] = _lowlevel
        self._filterDict['term'] = _term
        self._filterDict['nonterm'] = _nonterm
        self._filterDict['active'] = _active
        self._filterDict['notactive'] = _notactive
        self._filterDict['varlinked'] = _varlinked
        self._filterDict['notvarlinked'] = _notvarlinked
        self._filterDict['precise'] = _precise
        self._filterDict['notprecise'] = _notprecise

    def resetEvtimes(self):
        self.Evtimes = {}

    def __deepcopy__(self, dummy):
        # bug in deepcopy concerning _filterDict attribute, so work around
        # - self.events is the only thing to really deep copy anyway
        deepcopied_events = copy.deepcopy(self.events)
        dcopy = copy.copy(self)
        dcopy.events = deepcopied_events
        return dcopy

    def __del__(self):
        # not sure if this is necessary in order to evoke Event.__del__
        for ev in self.events:
            del ev

    def __setitem__(self, ev):
        if compareBaseClass(ev, Event):
            if ev.name not in self.events:
                self.events[ev.name] = ev
            else:
                print(self)
                raise ValueError("Event name '"+ev.name+"' already present "
                                    "in database")
        elif isinstance(ev,list):
            for ev_item in ev:
                self.__setitem__(ev_item)
        else:
            raise TypeError('Argument must be an Event: received type '
                            '%s'%str(type(ev)))

    add = __setitem__

    def __delitem__(self, ename):
        del(self.events[ename])

    delete = __delitem__

    def __getitem__(self, ename):
        return self.events[ename]

    def sortedEventNames(self, eventlist=None):
        if eventlist is None:
            return sortedDictKeys(self.events)
        else:
            eventlist.sort()
            return eventlist

    def getHighLevelEvents(self):
        hlList = []
        for epair in self.events.items():
            if isinstance(epair[1], HighLevelEvent):
                hlList.append(epair)
        return hlList

    def getLowLevelEvents(self):
        llList = []
        for epair in self.events.items():
            if isinstance(epair[1], LowLevelEvent):
                llList.append(epair)
        return llList

    def getAllEvents(self):
        return list(self.events.items())

    def getTermEvents(self):
        teList = []
        for epair in self.events.items():
            if epair[1].termFlag:
                teList.append(epair)
        return teList

    def getNonTermEvents(self):
        neList = []
        for epair in self.events.items():
            if not epair[1].termFlag:
                neList.append(epair)
        return neList

    def getActiveEvents(self):
        neList = []
        for epair in self.events.items():
            if epair[1].activeFlag:
                neList.append(epair)
        return neList

    def getNonActiveEvents(self):
        neList = []
        for epair in self.events.items():
            if not epair[1].activeFlag:
                neList.append(epair)
        return neList

    def getNonPreciseEvents(self):
        neList = []
        for epair in self.events.items():
            if not epair[1].preciseFlag:
                neList.append(epair)
        return neList

    def getPreciseEvents(self):
        neList = []
        for epair in self.events.items():
            if epair[1].preciseFlag:
                neList.append(epair)
        return neList

    def setglobalt0(self, t0):
        for epair in self.events.items():
            epair[1].globalt0 = t0

    def query(self, keylist, eventlist=None):
        """Return eventlist with results of queries corresponding
        to self._keylist keys.

        Multiple keys permitted in the query.
        """
        if not isinstance(keylist, list):
            raise TypeError('Query argument must be a list of keys')
        for key in keylist:
            if key not in self._keylist:
                raise TypeError('Query keys must be in _keylist attribute')
        filterFuncs = [self._filterDict[key] for key in keylist]
        if eventlist is None:
            if self.events == {}:
                return []
            else:
                eventlist = list(self.events.items())
        if filterFuncs == []:
            return []
        for f in filterFuncs:
            eventlist = list(filter(f, eventlist))
        return eventlist


    def __call__(self):
        # info is defined in utils.py
        utils_info(self.__dict__, "EventStruct")


    def info(self, verboselevel=0):
        if verboselevel > 0:
            # info is defined in utils.py
            utils_info(self.__dict__, "EventStruct",
                        recurseDepthLimit=1+verboselevel)
        else:
            print(self.__repr__())


    def __contains__(self, evname):
        return evname in self.events

    # If low level events are passed to this method, there will be an
    # exception because they do not have __call__ method defined!
    # This function is used for non-varlinked events, e.g. for use in
    # integration loops.
    # varDict must be supplied (not None) for this polling to work!
    def pollHighLevelEvents(self, tval=None, varDict=None, parDict=None,
                            eventlist=None):
        if eventlist is None:
            eventlist = self.query(['highlevel', 'active', 'notvarlinked'])
        return [(n, e) for n, e in eventlist if e(t=tval, varDict=varDict,
                                                  parDict=parDict)]


    def resetHighLevelEvents(self, t0, eventlist=None, state=None):
        if eventlist is None:
            eventlist = self.query(['highlevel'])
        if type(state)==list:
            for i, (ev, s) in enumerate(zip(eventlist, state)):
                ev[1].reset(s)
                ev[1].starttime = t0
        else:
            for ev in eventlist:
                ev[1].reset(state)
                ev[1].starttime = t0


    def validateEvents(self, database, eventlist):
        """validateEvents is only used for high level events."""
        assert eventlist != [], 'Empty event list passed to validateEvents'
        for ev in eventlist:
            if isinstance(ev[1], HighLevelEvent):
                if not all(key in database for key in ev[1].vars.keys()):
                    ek = list(ev[1].vars.keys())
                    print("Missing keys: %r" % (remain(ek, database), ))
                    raise RuntimeError("Invalid keys in event '%s'" % ev[0])
            #else:
            #    print "Warning: Low level events should not be passed to " \
            #            + "validateEvents()"
            #    print "   (event '", ev[0], "')"


    def setTermFlag(self, eventTarget, flagval):
        if isinstance(flagval, bool):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].termFlag = flagval
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].termFlag = flagval
        else:
            raise TypeError("Invalid flag type")


    def setActiveFlag(self, eventTarget, flagval):
        if isinstance(flagval, bool):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].activeFlag = flagval
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].activeFlag = flagval
        else:
            raise TypeError("Invalid flag type")

    def setPreciseFlag(self, eventTarget, flagval):
        if isinstance(flagval, bool):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].preciseFlag = flagval
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].preciseFlag = flagval
        else:
            raise TypeError("Invalid flag type")

    def setEventICs(self, eventTarget, val):
        if isinstance(val, dict):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    for name in val.keys():
                        if name in self.events[evTarg].initialconditions.keys():
                            self.events[evTarg].initialconditions[name] = val[name]
            else:
                if eventTarget in self.events.keys():
                    for name in val.keys():
                        if name in self.events[eventTarget].initialconditions.keys():
                            self.events[eventTarget].initialconditions[name] = val[name]
        else:
            raise TypeError("Invalid ICs type -- must be dict of varname, value pairs")


    def setEventDelay(self, eventTarget, val):
        if isinstance(val, int) or isinstance(val, float):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].eventdelay = val
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].eventdelay = val
        else:
            raise TypeError("Invalid eventdelay type/value")

    def setEventInterval(self, eventTarget, val):
        if isinstance(val, int) or isinstance(val, float):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].eventinterval = val
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].eventinterval = val
        else:
            raise TypeError("Invalid eventinterval type/value")

    def setEventTol(self, eventTarget, val):
        if val > 0:
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].eventtol = val
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].eventtol = val
        else:
            raise TypeError("Invalid eventtol type/value")

    def setEventDir(self, eventTarget, val):
        if val in [-1,0,1] and isinstance(val, int):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].dircode = val
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].dircode = val
        else:
            raise TypeError("Invalid eventdir type/value")

    def setStartTime(self, eventTarget, val):
        if isinstance(val, int) or isinstance(val, float):
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].starttime = val
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].starttime = val
        else:
            raise TypeError("Invalid starttime type")


    def setBisect(self, eventTarget, val):
        if isinstance(val, int) and val > 0:
            if isinstance(eventTarget, list):
                for evTarg in intersect(eventTarget, self.events.keys()):
                    self.events[evTarg].bisectlimit = val
            else:
                if eventTarget in self.events.keys():
                    self.events[eventTarget].bisectlimit = val
        else:
            raise TypeError("Invalid bisectlimit type/value")



class Event(object):
    """Generic Event.

    Possible keys in argument dictionary at initialization:
        name, eventtol, eventdelay, starttime, bisectlimit, term, active,
        precise, vars, expr.
    """

    def __init__(self, kw):
        if 'name' in kw:
            self.name = kw['name']
        else:
            raise KeyError('Name must be supplied to event')
        # absolute tolerance for event location (in dependent variable/expression)
        if 'eventtol' in kw:
            self.eventtol = kw['eventtol']
        else:
            self.eventtol = 1e-9
        # time interval before event detection begins on each run
        if 'eventdelay' in kw:
            self.eventdelay = kw['eventdelay']
        else:
            self.eventdelay = 1e-3
        # time interval between event detections restart
        if 'eventinterval' in kw:
            self.eventinterval = kw['eventinterval']
        else:
            self.eventinterval = 1e-3
        # number of bisection steps to take when finding events
        if 'bisectlimit' in kw:
            self.bisectlimit = kw['bisectlimit']
        else:
            self.bisectlimit = 100
        # terminating event flag
        if 'term' in kw:
            self.termFlag = kw['term']
        else:
            self.termFlag = False
        # active event flag
        if 'active' in kw:
            self.activeFlag = kw['active']
        else:
            self.activeFlag = True
        # determines whether event must be computed precisely
        if 'precise' in kw:
            self.preciseFlag = kw['precise']
        else:
            self.preciseFlag = True
        # store 'plain text' definition of event as string, if provided
        if 'expr' in kw:
            assert isinstance(kw['expr'], six.string_types), \
                    "Invalid type for event definition string"
            self._expr = kw['expr']
        else:
            self._expr = None
        # list of error structures
##        self.errors = []
        try:
            self.dircode = kw['dircode']
            assert self.dircode in [-1, 0, 1]  # direction codes
        except AssertionError:
            print('invalid value for direction code -- must be -1, 0, or 1')
            raise
        except KeyError:
            self.dircode = 0
        # effective time zero for searches (used for eventdelay)
        if 'starttime' in kw:
            self.starttime = kw['starttime']
        else:
            self.starttime = 0
        # optional variable and parameter bounds information, in case an event
        # wishes to refer to them
        if 'xdomain' in kw:
            assert type(kw['xdomain'])==dict, \
                    "Invalid type for variable bounds information"
            self.xdomain = kw['xdomain']
        else:
            self.xdomain = {}
        if 'pdomain' in kw:
            assert type(kw['pdomain'])==dict, \
                    "Invalid type for parameter bounds information"
            self.pdomain = kw['pdomain']
        else:
            self.pdomain = {}
        # var dictionary (can be just dict of keys for individual
        # value calls only) -- only for purely high level events
        if 'vars' in kw:
            self.vars = kw['vars']
            assert len(self.vars) > 0, 'vars dictionary must be non-empty'
            if all(isinstance(var, Variable) for var in self.vars.values()):
                self.varlinked = True
                # doesn't check that only argument is spec'd for _fn method
            else:
                self.varlinked = False
        else:
            raise KeyError('vars dictionary not present')
        # _funcreg is a register of dynamically created Event method names
        # in case of object copying (requiring destruction and re-creation
        # of dynamically created methods)
        self._funcreg = []
        self._funcstr = kw['funcspec'][0]
        self._funcname = kw['funcspec'][1]
        if 'auxfnspec' in kw:
            self._fnspecs = ensureStrArgDict(kw['auxfnspec'])
        else:
            self._fnspecs = {}
        if 'noHighLevel' in kw:
            # Boolean to indicate whether the non-Python event type can make
            # a high-level image of the event function
            if not kw['noHighLevel']:
                self.addMethods()
        else:
            # assume is high level
            self.addMethods()
        if 'prevsign_IC' in kw:
            # set initial value -- useful for one-off tests or as part of
            # map-based hybrid models with only one integer timestep
            self.prevsign = kw['prevsign_IC']
            self.prevsign_IC = self.prevsign
        else:
            self.prevsign = None
            self.prevsign_IC = None
        # additional history for previous sign, in case of resetting
        self.prevprevsign = None
        self.fval = None
        # placeholder for variables' initial conditions. If events use
        # auxiliary functions that access 'initcond' auxiliary function
        # then caller had better assign the current initial conditions
        # to this dictionary first.
        self.initialconditions = {}
        # placeholder for global independent variable value, for use in
        # hybrid systems
        self.globalt0 = 0
        # event queue information (used for discrete delays, recording
        # features, etc.)
        self.queues = {}
        self._sorted_queues = []
        # event internal parameter information (for temp usage by
        # event mappings)
        self.evpars = {}
        if 'evpars' in kw:
            self.evpars.update(kw['evpars'])
        # used for quadratic interpolation, if requested in searchForEvents
        self.quadratic = None  #fit_quadratic()


    # Queues are intended to be an ordered sequence of numeric types, e.g.
    # for times at which next terminal event should occur.
    # Queues are first in, first out, and may be sorted or not.
    # !! This feature is in development.
    def addToQ(self, qname, item):
        try:
            self.queues[qname].append(item)
        except KeyError:
            raise PyDSTool_ExistError("Queue %s was not declared"%qname)
        else:
            if qname in self._sorted_queues:
                self.queues[qname].sort()

    def createQ(self, qname, sorted=True, seq=None):
        """Also use to reset a queue."""
        if seq is None:
            self.queues[qname] = []
        else:
            # ensure list argument (for pop to work)
            self.queues[qname] = list(seq)
        if sorted:
            self._sorted_queues.append(qname)

    def popFromQ(self, qname):
        return self.queues[qname].pop(0)

    def deleteQ(self, qname):
        if qname in self._sorted_queues:
            i = self._sorted_queues.index(qname)
            del self._sorted_queues[i]
        try:
            del self.queues[qname]
        except KeyError:
            pass

    def _infostr(self, verbose=1):
        dirstr = ["decreasing", "either", "increasing"]
        if verbose <= 0:
            outputStr = "Event "+self.name
        elif verbose > 0:
            outputStr = "Event "+self.name+"\n" + \
                        "  active: " + str(self.activeFlag) + "\n" + \
                        "  terminal: " + str(self.termFlag) + "\n" + \
                        "  precise: " + str(self.preciseFlag) + "\n" + \
                        "  direction: " + dirstr[self.dircode+1] + "\n" + \
                        "  event tol: " + str(self.eventtol) + "\n" + \
                        "  definition: " + self._expr
            if verbose >= 2:
                outputStr += "\n  bisect limit: " + str(self.bisectlimit) \
                                + "\n" + \
                                "  event delay: " + str(self.eventdelay) + "\n" +\
                                "  event interval: " + str(self.eventinterval)
        return outputStr

    def __eq__(self, other):
        try:
            return self._infostr(2) == other._infostr(2)
        except AttributeError:
            return False

    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))


    def __repr__(self):
        return self._infostr(verbose=0)

    __str__ = __repr__


    def addMethods(self):
        # clean up FuncSpec usage of parsinps and x for pars/inputs and
        # variables
        try:
            exec(self._funcstr)
        except:
            print('Invalid event function definition:')
            print(self._funcstr)
            raise
        try:
            setattr(self, '_fn', six.create_bound_method(locals()[self._funcname], self))
        except KeyError:
            print('Must pass objective function for event at initialization')
            raise
        if '_fn' not in self._funcreg:
            self._funcreg.append('_fn')
        # _fnspecs keys are convenient, user-readable, short-hand names
        # for the functions, whose actual function names are the second entry
        # of the pair funcstr
        for funcpair in self._fnspecs.values():
            # clean up FuncSpec usage of parsinps and x for pars/inputs and
            # variables
            try:
                exec(funcpair[0])
            except:
                print('Invalid auxiliary function definition:')
                print(funcpair[0])
                raise
            try:
                setattr(self, funcpair[1], six.create_bound_method(locals()[funcpair[1]], self))
            except KeyError:
                print('Must pass objective function for event at initialization')
                raise
            if funcpair[1] not in self._funcreg:
                self._funcreg.append(funcpair[1])


    def reset(self, state=None):
        """Reset event`s prevsign attribute to a certain state (defaults to None)
        """
        self.fval = None
        if state is None or self.dircode == 0:
            if self.prevsign_IC is None and state is not None:
                self.prevsign = self.prevprevsign
            else:
                self.prevsign = self.prevsign_IC
        elif state == 'prev':
            self.prevsign = self.prevprevsign
        elif state == 'on':
            self.prevsign = self.dircode
        elif state == 'off':
            self.prevsign = -self.dircode
        else:
            raise ValueError("Invalid state passed to event reset method")


    def __call__(self, t=None, varDict=None, parDict=None):
        """Report on correct sign change.
        For external inputs, add input names and vales at time t to parDict
        """
        assert self.activeFlag, "Event cannot be called when inactivated"
        if varDict is None:
            if t is not None:
                assert self.varlinked, ('wrong type of call for non var-'
                                            ' linked event')
                if t < self.starttime + self.eventdelay:
                    return False
            else:
                raise ValueError('t must be specified for this type of call')
            if self.prevsign is None:
                try:
                    self.fval = self._fn(t, parDict)
                except:
                    print("Error in event %s" % (self.name, ))
                    info(parDict, "Parameters")
                    raise
                self.prevsign = scipy.sign(self.fval)
                self.prevprevsign = None
                return False
            else:
                try:
                    self.fval = self._fn(t, parDict)
                except:
                    print("Error in event %s" % (self.name, ))
                    info(parDict, "Parameters")
                    raise
                sval = scipy.sign(self.fval)
                if self.dircode == 0:
                    result = self.prevsign != sval
                else:
                    result = self.prevsign != sval and \
                            self.prevsign * self.dircode < 0
                self.prevprevsign = self.prevsign
                self.prevsign = sval
                return result
        else:
            # then calling as individual value, not var object
            varDict_temp = dict(varDict)
            if t is None:
                t = varDict['t']
            else:
                assert 't' not in varDict, ('`t` key already present in '
                                        'varDict argument. Cannot redefine.')
                varDict_temp['t'] = t
            if t < self.starttime + self.eventdelay:
                return False
            assert not self.varlinked, ('wrong type of call for var-linked'
                                        ' event')
            if self.prevsign is None:
                try:
                    self.fval = self._fn(varDict_temp, parDict)
                except:
                    print("Error in event %s" % (self.name, ))
                    info(parDict, "Parameters")
                    print("\n")
                    info(varDict_temp, "Variables")
                    raise
                self.prevsign = scipy.sign(self.fval)
                self.prevprevsign = None
                return False
            else:
                try:
                    self.fval = self._fn(varDict_temp, parDict)
                except:
                    print("Error in event %s" % (self.name, ))
                    info(parDict, "Parameters")
                    print("\n")
                    info(varDict_temp, "Variables")
                    raise
                sval = scipy.sign(self.fval)
                if self.dircode == 0:
                    result = self.prevsign != sval
                else:
                    result = self.prevsign != sval and \
                            self.prevsign * self.dircode < 0
                self.prevprevsign = self.prevsign
                self.prevsign = sval
                return result

    def __hash__(self):
        return hash((self.name, self._expr, self._funcstr))

    def searchForEvents(self, trange=None, dt=None, checklevel=2,
                        parDict=None, vars=None, inputs=None,
                        abseps=1e-13, eventdelay=True, globalt0=0):
        """Search a variable-linked event, or an event with supplied vars
        dictionary and relevant parameters, for zero crossings.

        (Variable-linked search not applicable to low level events.)

        trange=None, dt=None, checklevel=2, parDict=None, vars=None, inputs=None,
            abseps=1e-13, eventdelay=True -> (ev_t, (ev_tlo, ev_thi))
        where the lo-hi tuple is the smallest bound around ev_t (in
        case it is None because event was not found accurately).

        dt will default to 1e-3 * the time interval of the variables.
        'eventinterval' inherited from the event will be used to separate
        detected events.

        Only pass vars dictionary when event.varlinked is False.
        """

        # flags indicating whether continuous and discrete variables are present
        discretevars = False
        continuousvars = False
        if self.varlinked:
            assert vars is None, ("event is variable-linked already -- "
                                    "do not pass vars argument")
            varDict = self.vars
        else:
            assert vars is not None, ("event is not variable-linked "
                                        "already -- require a vars argument")
            varDict = vars
        precise = self.preciseFlag
        try:
            for var in varDict.values():
                if isdiscrete(var):
                    discretevars = True
                elif iscontinuous(var):
                    continuousvars = True
                else:
                    raise TypeError('varDict must consist of Variable objects')
        except AttributeError:
            print('event must contain a dictionary of vars')
            raise
        if discretevars and continuousvars:
            raise TypeError('Cannot mix discrete and continuous Variable types')
        if discretevars:
            if precise:
                print('argument precise cannot be used for discrete Variable objects')
                precise = False
            varnames = list(varDict.keys())
            if dt is not None:
                print('argument dt is unused for discrete Variable objects')
            if trange is not None:
                # trange had better be contained in all var.indepdomain ranges
                assert len(varDict) > 0, 'varDict was empty!'
                if not reduce(bool.__and__, [trange[0] in var.indepdomain \
                                and trange[1] in var.indepdomain for var \
                                in varDict.values()]):
                    raise ValueError('trange not contained in all var ranges')
            else:
                raise ValueError('trange must be defined for discrete'
                                    ' Variable objects')
            tlist = varDict[varnames[0]].indepdomain
            tlist = tlist[tlist.index(trange[0]):tlist.index(trange[1])]
        elif continuousvars:
            # trange had better be contained in all var.indepdomain ranges
            tlimits = None  # initial value
            for var in varDict.values():
                if tlimits is None:
                    tlimits = var.indepdomain.get()
                else:
                    temp = var.indepdomain.get()
                    if temp[0] < tlimits[0]:
                        tlimits[0] = temp[0]
                    if temp[1] > tlimits[1]:
                        tlimits[1] = temp[1]
            if trange is not None:
                # compare trange and tlimits using Interval inclusion,
                # so that can make use of uncertain values at boundaries
                trange_int = Interval('trange', float, trange, abseps)
                tlimits_int = Interval('tlimits', float, tlimits, abseps)
                if not self.contains(tlimits_int, trange_int, checklevel):
                    raise ValueError('trange not contained in all var ranges')
            else:
                trange = tlimits
            if dt is None:
                dt = max(self.eventtol, 1e-3*(trange[1]-trange[0]))
            if dt > trange[1]-trange[0]:
                raise ValueError('dt (eventtol if not specified) is too large'
                                 ' for trange in event %s'%self.name)
            ttemp = copy.copy(var.indepdomain)
            ttemp.set(trange)
            tlist = ttemp.sample(dt, avoidendpoints=True)
        else:
            raise TypeError('var must be a Variable object')
        if inputs is None:
            test_fn = self._fn
            self_caller = self.__call__
        else:
            def test_fn(ts, ps):
                if ps is None:
                    pidict = {}
                else:
                    pidict = copy.copy(ps)
                try:
                    for t in ts:
                        idict = dict([(n,i(t+globalt0)) for n,i in inputs.items()])
                        pidict.update(idict)
                        fvals.append(self._fn(t, pidict))
                except TypeError:
                    # Iteration over non sequence, so assume ts is just a single
                    # numeric value
                    try:
                        tvals = array(ts)+globalt0
                    except:
                        print("\n Found type %s" % (type(ts), ))
                        raise TypeError("t values invalid")
                    idict = dict([(n,i(tvals)) for n,i in inputs.items()])
                    pidict.update(idict)
                    fvals = self._fn(ts, pidict)
                return fvals
            def self_caller(t=None, varDict=None, parDict=None):
                if parDict is None:
                    pidict = dict([(n,i(t+globalt0)) for n,i in inputs.items()])
                else:
                    pidict = copy.copy(parDict)
                    pidict.update(dict([(n,i(t+globalt0)) for n,i in inputs.items()]))
                return self(t, varDict=varDict, parDict=pidict)
        # before search, check that start after 'eventdelay', if option set
        if eventdelay:
            while tlist[0] <= self.eventdelay:
                try:
                    tlist.pop(0)
                except IndexError:
                    raise ValueError('eventdelay too large -- '
                                        'no search interval')
        # now do the search
        try:
            if self.varlinked:
                try:
                    fvals = test_fn(tlist, parDict)
                    boollist = [False]  # initial point is always False
                    if self.dircode == 0:
                        boollist.extend([fvals[i] != fvals[i+1] for i in \
                                            range(len(fvals)-1)])
                    else:
                        boollist.extend([fvals[i] * fvals[i+1] < 0 and \
                            fvals[i] * self.dircode < 0 for i in \
                                            range(len(fvals)-1)])
                except:
                    # event does not support vectorized t calls
##                    print "Warning: event did not support vectorized t calls"
                    # set start time for self-calling purposes
                    self.starttime = trange[0]
                    boollist = [self_caller(t, parDict=parDict) for t in tlist]
                    # first bool may be True because event hasn't been called
                    # before and has prevsign unset (or mis-set)...
                    # we can safely overwrite it (bit of a hack) because
                    # it's always really False
                    boollist[0] = False
            else:
                # vallist is a list of 1D lists
                vallist = [v(tlist) for v in sortedDictValues(varDict)]
                varnames = sortedDictKeys(varDict)
                # set start time for self-calling purposes
                self.starttime = trange[0]
                if not eventdelay:
                    # switch off event delay temporarily
                    restore_val = self.eventdelay
                    self.eventdelay = 0.
                boollist = []
                for tix in range(len(tlist)):
                    boollist.append(self_caller(t=tlist[tix],
                                            varDict=dict(zip(varnames,
                                                            [vallist[i][tix] for\
                                                    i in range(len(vallist))])),
                                            parDict=parDict))
                # first bool may be True because event hasn't been called
                # before and has prevsign unset (or mis-set)...
                # we can safely overwrite it (bit of a hack) because
                # it's always really False
                boollist[0] = False
                if not eventdelay:
                    # restore value for future use
                    self.eventdelay = restore_val
        except KeyError as e:
            # for discretevars if not all var.indepdomains equal over trange
            if discretevars:
                print('Note: All discrete Variables must have identical ' \
                        'independent variable domains over trange')
            print("Check that all variable references in events are legitimate")
            raise
        # loop through boollist and find all events unless terminating at tpos!
        tpos = -1
        eventsfound = []
        t_last = -numpy.inf
        t_interval = self.eventinterval
        while True:
            try:
                tpos += boollist[tpos+1:].index(True)+1
            except (ValueError, IndexError):
                # no event found in this slice, at this dt
                break
            if precise and continuousvars:
                if self.varlinked:
                    result = findpreciseroot(self, tlist[tpos-1], tlist[tpos],
                                              parDict, inputs=inputs,
                                              globalt0=globalt0,
                                              quadratic_interp=self.quadratic)
                else:
                    result = findpreciseroot(self, tlist[tpos-1], tlist[tpos],
                                              parDict, varDict, inputs,
                                              globalt0=globalt0,
                                              quadratic_interp=self.quadratic)
                if result is not None and result[0] > t_last + t_interval:
                    eventsfound.append(result)
                    t_last = result[0]
            else:
                result = tlist[tpos]
                if result > t_last + t_interval:
                    eventsfound.append((result, (tlist[tpos-1], result)))
                    t_last = result
            if self.termFlag:  # quit searching now!
                break
        return eventsfound


    def contains(self, interval, val, checklevel=2):
        # NB. val may be another interval
        if checklevel == 0:
            # level 0 -- no bounds checking at all
            # code should avoid calling this function with checklevel = 0
            # if possible, but this case is left here for completeness and
            # consistency
            return True
        elif checklevel in [1,2]:
            # level 1 -- ignore uncertain cases (treat as contained)
            # level 2 -- warn on uncertain and continue (but warnings not
            #  used in Events, so treat as #1)
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

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        for fname in self._funcreg:
            del d[fname]
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.addMethods()

    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)



class HighLevelEvent(Event):
    """Event defined using python function code.

    First argument to function must always be 'self'
    """
    def __init__(self, kw):
        assert 'LLfuncspec' not in kw, \
            "'LLfuncspec' key is invalid for purely high level events"
        Event.__init__(self, kw)


class LowLevelEvent(Event):
    """Event defined using externally-compiled and linked function code (i.e. C)

    Specification of the function body as a string must include
    necessary temporary variable declarations and a return statement."""
    def __init__(self, kw):
        if 'vars' in kw:
            for v in kw['vars'].values():
                if v is not None:
                    raise TypeError("Low level events cannot be linked to a"
                                    " variable")
        Event.__init__(self, kw)
        assert isinstance(kw['LLfuncspec'], six.string_types), \
                          ("For low level events, must "
                           "pass string for 'LLfuncspec' in initialization")
        LLfuncstr = kw['LLfuncspec']
        dummyQ = QuantSpec('dummy', LLfuncstr, preserveSpace=True)
        dummyQ.mapNames({'abs': 'fabs', 'sign': 'signum', 'mod': 'fmod'})
        if dummyQ[0] != 'return':
            print("Found: %s" % (dummyQ[0], ))
            print("in event specification: %s" % (dummyQ(), ))
            raise ValueError("'return' must be first token in low level "
                                "event specification")
        # take out 'return' from tokenized because whitespace will be lost
        # (parser not designed for non-math expressions)
        self._LLfuncstr = "return " + "".join(dummyQ[1:])
        self._LLfuncname = self.name
        self._LLreturnstr = "double"
        self._LLargstr = "(unsigned n_, double t, double *Y_, double *p_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
        self.varlinked = False  # for compatibility with query calls


class MatlabEvent(LowLevelEvent):
    """Event defined using MATLAB syntax for use with ADMC++

    Specification of the function body as a string must include
    necessary temporary variable declarations."""
    def __init__(self, kw):
        if 'vars' in kw:
            for v in kw['vars'].values():
                if v is not None:
                    raise TypeError("Low level events cannot be linked to a variable")
        kw['noHighLevel'] = True
        Event.__init__(self, kw)
        assert isinstance(kw['Matlabfuncspec'], six.string_types), \
                             ("For low level events, must "
                              "pass string for 'LLfuncspec' in initialization")
        LLfuncstr = kw['Matlabfuncspec']
        dummyQ = QuantSpec('dummy', LLfuncstr, preserveSpace=True)
        if dummyQ[0] != 'return':
            print("Found: %s" % (dummyQ[0], ))
            print("in event specification: %s" % (dummyQ(), ))
            raise ValueError("'return' must be first token in low level "
                                "event specification")
        # take out 'return' from tokenized because whitespace will be lost
        # (parser not designed for non-math expressions)
        self._LLfuncstr = "Y_ =  " + "".join(dummyQ[1:])
        self._LLfuncname = self.name
        self._LLreturnstr = "Y_ = "
        self._LLargstr = "(t_, x_, p_)"
        self.varlinked = False  # for compatibility with query calls


# -------------------------------------------------------------

## Public exported functions
def makeZeroCrossEvent(expr, dircode, argDict, varnames=[], parnames=[],
                        inputnames=[], fnspecs={}, targetlang='python',
                        reuseterms={}, flatspec=None, extra_funcspec_args=None):
    """Target language-independent user-defined event involving coordinates,
    parameters, and time. Returns a non variable-linked event only.

    List of used variable, parameter, and input names defaults to
    the empty list and can be omitted if these are not referenced. If
    variable names are omitted then the expression must only depend on time
    't' and any declared parameters.

    Auxiliary function dictionary is required if the event accesses them.

    'targetlang' argument defaults to 'python'.
    The expression 'expr' may not use intermediate temporary variables (for
    which you should specify the body of the event function by hand.

    Optional argument reuseterms is a dictionary of terms used in expr
    that map to their definitions in terms of the state variables, time,
    parameters, and inputs.
    """
    # preprocess expression for varnames -> v['varname'] and similarly
    # for parameter names
    try:
        exprname = argDict['name']
    except KeyError:
        raise PyDSTool_KeyError("name key must be present in argDict")
    # support ModelSpec and Symbolic definitions, so take str() of expr
    try:
        expQS = QuantSpec('__ev_expr__', str(expr.renderForCode()))
    except AttributeError:
        expQS = QuantSpec('__ev_expr__', str(expr))
    # make copies of arguments to prevent defaults getting messed up
        # support ModelSpec and Symbolic definitions, so take str() of list elements
    varnames = [str(v) for v in varnames]
    parnames = [str(p) for p in parnames]
    inputnames = [str(i) for i in inputnames]
    auxfns = copy.copy(fnspecs)
    auxVarDefMap = {}
    if flatspec is not None:
        # transfer used definitions in flatpsec into varnames, parnames, auxfns
        # etc.
        assert varnames==parnames==[], 'flatspec argument requires that varnames and parnames lists are not used'
        assert fnspecs=={}, 'flatspec argument requires that fnspecs is not used'
        try:
            allvars = flatspec['vars']
        except KeyError:
            allvars = {}
        try:
            allinputs = flatspec['inputs']
        except KeyError:
            allinputs = {}
        try:
            allpars = flatspec['pars']
        except KeyError:
            allpars = {}

        try:
            allfuns = flatspec['auxfns']
        except KeyError:
            allfuns = {}
        collectReused(expQS, allvars, allpars, allinputs, allfuns,
                    varnames, parnames, inputnames, auxfns,
                    auxVarDefMap, flatspec)
##        for symb in expQS.freeSymbols:
##            if symb in allvars:
##                spectype = flatspec['spectypes'][symb]
##                if spectype == 'RHSfuncSpec':
##                    if symb not in varnames:
##                        varnames.append(symb)
##                elif spectype == 'ExpFuncSpec':
##                    auxVarDef = allvars[symb]
##                    auxVarDefMap[symb] = auxVarDef
##                    # only need to look one level deeper because auxvars
##                    # cannot depend on each other, in a well-defined system.
##                    symbQS = QuantSpec('__symb_expr__', auxVarDef)
##                    for auxsymb in symbQS.freeSymbols:
##                        if auxsymb in allvars:
##                            if auxsymb not in varnames:
##                                varnames.append(auxsymb)
##                                spectype = flatspec['spectypes'][auxsymb]
##                                if spectype == 'ExpFuncSpec':
##                                    auxVarDefMap[auxsymb] = allvars[auxsymb]
##                        elif auxsymb in allpars:
##                            if auxsymb not in parnames:
##                                parnames.append(auxsymb)
##                        elif auxsymb in allinputs:
##                            if auxsymb not in inputnames:
##                                inputnames.append(auxsymb)
##                        elif auxsymb in allfuns:
##                            if auxsymb not in auxfns:
##                                auxfns[auxsymb] = allfuns[auxsymb]
##            elif symb in allpars:
##                if symb not in parnames:
##                    parnames.append(symb)
##            elif symb in allinputs:
##                if symb not in inputnames:
##                    inputnames.append(symb)
##            elif symb in allfuns:
##                if symb not in auxfns:
##                    auxfns[symb] = allfuns[symb]
##                    # Add any dependencies of the function
##                    symbQS = QuantSpec('__symb_expr__', allfuns[symb][1])
##                    for fnsymb in symbQS.freeSymbols:
##                        if fnsymb in allpars:
##                            if fnsymb not in parnames:
##                                parnames.append(fnsymb)
##                        if fnsymb in allinputs:
##                            if fnsymb not in inputnames:
##                                inputnames.append(fnsymb)
##                        elif fnsymb in allfuns:
##                            raise "Ask Rob to add support for function-to-function calls which he didn't anticipate here!"
    expr = ''
    done = False
    while not done:
        expQS = expQS.eval(auxVarDefMap)
        expr_new = str(expQS)
        if expr_new == expr:
            done = True
        else:
            expr = expr_new
##    if inputnames != [] and targetlang != 'c':
##        raise NotImplementedError("Inputs to non C-based events are not "
##                                  "yet supported.")
    if exprname in varnames:
        raise ValueError("Expression name %s coincides with a variable name"%exprname)
    if exprname in parnames:
        raise ValueError("Expression name %s coincides with a parameter name"%exprname)
    if varnames == [] and inputnames == [] and 't' not in expQS:
        raise ValueError("In absence of variable or input names, time must appear in expression: %s"%expr)
    # zeros for t and varnames are unused placeholders. the names have
    # to be declared to funcspec with some value in order for the
    # object to be properly created. this is a kludge!
    specdict = {exprname: expr, 't': '0'}
    specdict.update({}.fromkeys(varnames, '0'))
    if 't' not in varnames:
        varnames.append('t')
    varnames.append(exprname)
    # Thus, if Jacobian etc. listed in funcspec then need to make a copy of the auxfns
    # and filter these out
    auxfns_temp = filteredDict(auxfns, ['Jacobian', 'Jacobian_pars', 'massMatrix'],
                               neg=True)
    # Hack: targetLang must be python if (actually) c or python; matlab otherwise
    if targetlang == 'matlab':
        dummytlang = 'matlab'
    else:
        dummytlang = 'python'
    dummyfs = {'varspecs': specdict,
               'name': exprname, 'vars': varnames,
               'fnspecs': auxfns_temp,
               'pars': parnames, 'inputs': inputnames,
               'targetlang': dummytlang
               }
    if extra_funcspec_args is not None:
        dummyfs.update(extra_funcspec_args)
    dummyfuncspec = FuncSpec.RHSfuncSpec(dummyfs)
    lstart = len(dummyfuncspec.codeinserts['start'])
    lend = len(dummyfuncspec.codeinserts['end'])
    if lstart > 0:
        # .strip() loses the indent (to replace with \t)
        start_code = dummyfuncspec._specStrParse(['inserts'],
                               {'inserts': dummyfuncspec.codeinserts['start']}, '',
                                noreturndefs=True, ignoreothers=True,
                                forexternal=True,
                                doing_inserts=True).strip() + '\n\t'
    else:
        start_code = ''
    if lend > 0:
        raise ValueError("End code inserts are not valid for Events")
    parsedstr = dummyfuncspec._specStrParse([exprname],
                                            dummyfuncspec.varspecs,
                                            noreturndefs=True,
                                            forexternal=True)
    # alter par and var name dictionaries used by Events.py
    parsedstr = parsedstr.replace("parsinps[","p[").replace("x[","v[")
    start_code = start_code.replace("parsinps[","p[").replace("x[","v[")
    if 'parsinps' in parsedstr:
        # then there are aux fns that use it
        funcstr = "parsinps=sortedDictValues(p,%s)+sortedDictValues(p,%s)"%(str(parnames),str(inputnames)) + \
                    "\n\t" + start_code + "return "+parsedstr+"\n"
    else:
        funcstr = start_code + "return "+parsedstr+"\n"
    if reuseterms != {}:
        illegalterms = ['globalindepvar', 'initcond', 'getindex', 'Jacobian',
                        'Jacobian_pars']
        reusestr, body_processed_dict = processReusedPy([exprname],
                                                {exprname: funcstr},
                                                copy.copy(reuseterms),
                                                dummyfuncspec,
                                                illegal=illegalterms)
        funcstr_processed = (len(reusestr)>0)*"# local definitions\n" \
            + reusestr + (len(reusestr)>0)*"\n" \
            + body_processed_dict[exprname]
    else:
        funcstr_processed = funcstr
    # find used variable names (assumes matching braces in supplied spec!)
    # so use original parsedstr (before being processed for reused terms)
    varnames_found = []
    currpos = 0
    done = False
    while not done:
        relfindpos = parsedstr[currpos:].find("v[")
        if relfindpos >= 0:
            findpos = relfindpos + currpos
            lbpos = findpos+1
            rbpos = findEndBrace(parsedstr[lbpos:], '[', ']')+lbpos
            varnames_found.append(parsedstr[lbpos+2:rbpos-1])
            currpos = rbpos
        else:
            done = True
    inputnames_found = []
    currpos = 0
    done = False
    while not done and inputnames != []:
        relfindpos = parsedstr[currpos:].find("p[")
        if relfindpos >= 0:
            findpos = relfindpos + currpos
            lbpos = findpos+1
            rbpos = findEndBrace(parsedstr[lbpos:], '[', ']')+lbpos
            in_name = parsedstr[lbpos+2:rbpos-1]
            if in_name in inputnames:
                inputnames_found.append(in_name)
            currpos = rbpos
        else:
            done = True
    newargs = ['dircode', 'funcspec', 'auxfnspec']
    if targetlang in ['c', 'C']:
        newargs.append('LLfuncspec')
        # convert any special C-specific functions
        expr = dummyfuncspec._processSpecialC(expr)
    elif targetlang == 'matlab':
        newargs.append('Matlabfuncspec')
        # convert any special C-specific functions
        expr = dummyfuncspec._processSpecialC(expr)
    for arg in newargs:
        if arg in argDict:
            print('Warning: `' + arg + '` already appears in argDict!')
            print('  This value will be overwritten.')
    argDict_out = copy.copy(argDict)
    funcname = "_f_"+exprname+"_ud"
    if parnames == []:
        pdefstr = "=None"
    else:
        pdefstr = ""
    # 'ds' plays role of 'self' (named for compatibility with FuncSpec's
    # automatic name resolution of auxiliary functions during parsing.
    if targetlang == 'matlab':
        funcstr_full = funcstr_processed
    else:
        funcstr_full = makeUniqueFn("def "+funcname+"(ds, v, p%s):\n\t"%pdefstr + \
                                    funcstr_processed, idstr="event")
    if varnames_found+inputnames_found != []:
        argDict_out['vars'] = {}.fromkeys(varnames_found+inputnames_found, None)
    argDict_out.update({'funcspec': funcstr_full,
                    'auxfnspec': dummyfuncspec.auxfns,
                    'expr': expr,
                    'dircode': dircode})
    if targetlang in ['c', 'C']:
        LLfuncstr = "return "+expr+";\n"
        if reuseterms != {}:
##          illegalterms = ['globalindepvar', 'initcond', 'getindex', 'Jacobian',
##                          'Jacobian_pars']
            LLreusestr, LLbody_processed_dict = processReusedC(['ev'],
                                                        {'ev': LLfuncstr},
                                                        copy.copy(reuseterms)) #,
##                                                       illegal=illegalterms)
            LLfuncstr_processed = (len(LLreusestr)>0)*"/* local definitions */\n" \
                + LLreusestr + (len(LLreusestr)>0)*"\n" + LLbody_processed_dict['ev']
        else:
            LLfuncstr_processed = LLfuncstr
        argDict_out['LLfuncspec'] = LLfuncstr_processed
        return LowLevelEvent(argDict_out)
    elif targetlang == 'matlab':
        LLfuncstr = "return "+expr+";\n"
        if reuseterms != {}:
            LLreusestr, LLbody_processed_dict = processReusedMatlab(['ev'],
                                                        {'ev': LLfuncstr},
                                                        copy.copy(reuseterms))

            LLfuncstr_processed = (len(LLreusestr)>0)*"% local definitions \n" \
                + LLreusestr + (len(LLreusestr)>0)*"\n" + LLbody_processed_dict['ev']
        else:
            LLfuncstr_processed = LLfuncstr
        argDict_out['Matlabfuncspec'] = LLfuncstr_processed
        return MatlabEvent(argDict_out)
    else:
        return HighLevelEvent(argDict_out)


def makePythonStateZeroCrossEvent(varname, targetvalue, dircode, argDict,
                                    var=None):
    """Python function-specified zero-crossing event in coordinate, or in
    time. Use 'var' argument to create a variable-linked event.

    varname may be a Quantity object. dircode is -1, 0, or 1.
    varname may be the reserved word 't', for the independent variable.
    """
    # set `var` only for non-var linked events
    newargs = ['dircode', 'funcspec', 'vars']
    for arg in newargs:
        if arg in argDict:
            print('Warning: `' + arg + '` already appears in argDict!')
            print('  This value will be overwritten.')
    if isinstance(varname, Var):
        # supporting a Variable object
        varname = varname.name
    elif not isinstance(varname, six.string_types):
        raise TypeError("Invalid type for event variable")
    funcname = "_f_"+varname+"_zc"
    if isinstance(targetvalue, six.string_types):
        # this is assumed to be a parameter name if not a numeric value,
        # but user preferred to add the p[' '] wrapping around a
        # parameter name
        if not (targetvalue[0:3] == "p['" and targetvalue[-2:] == "']"):
                targstr = "p['"+targetvalue+"']"
        else:
            targstr = targetvalue
        pdefstr = "=None"
    else:
        targstr = str(targetvalue)
        pdefstr = ""
    # 'ds' plays role of 'self' (named for compatibility with FuncSpec's
    # automatic name resolution of auxiliary functions during parsing.
    if isinstance(var, Variable):
        if not isinstance(targetvalue, six.string_types):
            assert targetvalue in var.depdomain, 'targetvalue not in var.depdomain'
        funcstr = "def "+funcname+"(ds, t, p%s):\n\treturn "%pdefstr + \
                    "ds.vars['"+varname+"'](t) - "+targstr+"\n"
    elif var is None:
        funcstr = "def "+funcname+"(ds, v, p%s):\n\treturn v['"%pdefstr + \
                    varname+"'] - " + targstr + "\n"
    else:
        raise ValueError('var must be a Variable object or None')
    argDict_out = copy.copy(argDict)
    argDict_out.update({'vars': {varname: var},
                    'funcspec': makeUniqueFn(funcstr, idstr="event"),
                    'expr': varname + ' - ' + targstr,
                    'dircode': dircode})
    return HighLevelEvent(argDict_out)


def collectReused(quant, allvars, allpars, allinputs, allfuns,
                  varnames, parnames, inputnames, auxfns, auxVarDefMap, flatspec):
    for symb in quant.freeSymbols:
        if symb in allvars:
            spectype = flatspec['spectypes'][symb]
            if spectype == 'RHSfuncSpec':
                if symb not in varnames:
                    varnames.append(symb)
            elif spectype == 'ExpFuncSpec':
                auxVarDef = allvars[symb]
                auxVarDefMap[symb] = auxVarDef
                # need to recurse in case auxvar depends on another
                symbQS = QuantSpec('__symb_expr__', auxVarDef)
                collectReused(symbQS, allvars, allpars, allinputs, allfuns,
                              varnames, parnames, inputnames, auxfns,
                              auxVarDefMap, flatspec)
        elif symb in allpars:
            if symb not in parnames:
                parnames.append(symb)
        elif symb in allinputs:
            if symb not in inputnames:
                inputnames.append(symb)
        elif symb in allfuns:
            if symb not in auxfns:
                auxfns[symb] = allfuns[symb]
                # Add any dependencies of the function
                symbQS = QuantSpec('__symb_expr__', allfuns[symb][1])
                collectReused(symbQS, allvars, allpars, allinputs, allfuns,
                              varnames, parnames, inputnames, auxfns,
                              auxVarDefMap, flatspec)


def processReusedPy(specnames, specdict, reuseterms, fspec, specials=[],
                        dovars=True, dopars=True, doinps=True, illegal=[]):
    """Process reused subexpression terms for Python code.
    (Similar to function of similar name in FuncSpec.py)
    """
    reused, specupdated, new_protected, order = FuncSpec._processReused(specnames,
                                                            specdict,
                                                            reuseterms,
                                                            _indentstr)
    fspec._protected_reusenames = new_protected
    fspec.varspecs.update(specupdated)
    # symbols to parse are at indices 2 and 4 of 'reused' dictionary
    reusedParsed = fspec._parseReusedTermsPy(reused, [2,4],
                                    specials=specials, dovars=dovars,
                                    dopars=dopars, doinps=doinps,
                                                illegal=illegal)
    reusedefs = {}.fromkeys(new_protected)
    for vname, deflist in reusedParsed.items():
        for d in deflist:
            reusedefs[d[2]] = d
    return (concatStrDict(reusedefs, intersect(order,reusedefs.keys())),
                    specupdated)


def processReusedC(specnames, specdict, reuseterms):
    """Process reused subexpression terms for C code.
    (Similar to function processReusedC in FuncSpec.py)
    """
    reused, specupdated, new_protected, order = FuncSpec._processReused(specnames,
                                                    specdict,
                                                    reuseterms,
                                                    '', 'double', ';')
    reusedefs = {}.fromkeys(new_protected)
    for vname, deflist in reused.items():
        for d in deflist:
            reusedefs[d[2]] = d
    return (concatStrDict(reusedefs, intersect(order, reusedefs.keys())),
                    specupdated)


def processReusedMatlab(specnames, specdict, reuseterms):
    """Process reused subexpression terms for matlab code.
    (Similar to function of similar name in FuncSpec.py)
    """
    reused, specupdated, new_protected, order = FuncSpec._processReused(specnames,
                                                    specdict,
                                                    reuseterms,
                                                    '', '', ';')
    reusedefs = {}.fromkeys(new_protected)
    for vname, deflist in reused.items():
        for d in deflist:
            reusedefs[d[2]] = d
    return (concatStrDict(reusedefs, intersect(order, reusedefs.keys())),
                    specupdated)


def findpreciseroot(ev, tlo, thi, parDict=None, vars=None, inputs=None,
                    globalt0=0, quadratic_interp=None):
    """Find root more accurately from a Variable object using bisection.

    (Adapted from scipy.optimize.minpack.bisection code to make use of
    quadratic interpolation, which assumes that tlo and thi are already known
    to be close enough together that the variable's curve is purely concave
    up or down in the neighbourhood, and so can be fitted accurately with a
    single quadratic).

    To use quadratic interpolation, pass a fit_quadratic instance as the
    quadratic_interp argument. Interpolation will also be done on any inputs
    provided (**not yet implemented**).
    """
    if tlo >= thi:
        raise ValueError('time limits are not finitely separated'
                            ' or are given in wrong order')
    if inputs is None:
        if quadratic_interp is not None:
            raise NotImplementedError
            #
            q = quadratic_interp
            dt = thi-tlo #??? # may not be variable's mesh dt
            # and so far Event has not assumed that variable is either defined
            # using a mesh or not !!!
            ts = linspace(tlo, thi, 8)
            res = smooth_pts(ts, [ev._fn(t) for t in ts])
            test_fn = res.results.f
        else:
            test_fn = ev._fn
    else:
        # only going to be calling this with individual t values
        if ev.varlinked:
            if quadratic_interp is not None:
                raise NotImplementedError
            else:
                def test_fn(x, p):
                    if p is None:
                        pidict = dict([(n,i(t+globalt0)) for n,i in inputs.items()])
                    else:
                        pidict = copy.copy(p)
                        pidict.update(dict([(n,i(t+globalt0)) for n,i in inputs.items()]))
                    return ev._fn(t, pidict)
        else:
            if quadratic_interp is not None:
                raise NotImplementedError
            else:
                def test_fn(x, p):
                    if p is None:
                        pidict = dict([(n,i(x['t']+globalt0)) for n,i in inputs.items()])
                    else:
                        pidict = copy.copy(p)
                        pidict.update(dict([(n,i(x['t']+globalt0)) for n,i in inputs.items()]))
                    return ev._fn(x, pidict)
    if ev.varlinked:
        assert vars is None, ("Only pass vars argument when event is not "
                                "linked to a variable")
        assert reduce(bool.__and__, [iscontinuous(var) \
                                    for var in ev.vars.values()]), \
                            'Only pass continously-defined Variables in event'
        elo = test_fn(tlo, parDict)
        ehi = test_fn(thi, parDict)
        if elo == 0:
            return (tlo, (tlo, thi))
        elif ehi == 0:
            return (thi, (tlo, thi))
        elif elo * ehi > 0:
            # event cannot be present
            return None
        a = tlo
        b = thi
        i = 1
        eva = test_fn(a, parDict)
        rootival = (a,b)
        while i <= ev.bisectlimit:
            d = (b-a)/2.0
            p = a + d
            evp = test_fn(p, parDict)
            if abs(evp-eva) < ev.eventtol or evp == 0:
                return (p, rootival)
            i += 1
            if evp*eva > 0:
                a = p
                eva = evp
            else:
                b = p
            # do this at end of while loop in case i > bisectlimit
            rootival = (a,b)
        # search failed after bisectlimit
        return (None, rootival)
    else:
        assert vars is not None, ("vars argument is required when event is not"
                                    " linked to a variable")
        assert reduce(bool.__and__, [iscontinuous(var) \
                                    for var in vars.values()]), \
                            'Only pass continously-defined Variables'
        varnames = sortedDictKeys(vars)+['t']
        dlo = dict(zip(varnames, [v(tlo) for v in sortedDictValues(vars)]+[tlo]))
        dhi = dict(zip(varnames, [v(thi) for v in sortedDictValues(vars)]+[thi]))
        elo = test_fn(dlo, parDict)
        ehi = test_fn(dhi, parDict)
        if elo == 0:
            return (tlo, (tlo, thi))
        elif ehi == 0:
            return (thi, (tlo, thi))
        elif elo * ehi > 0:
            # event cannot be present
            return None
        a = tlo
        b = thi
        i = 1
        da = dict(zip(varnames, [v(a) for v in sortedDictValues(vars)]+[a]))
        db = dict(zip(varnames, [v(b) for v in sortedDictValues(vars)]+[b]))
        eva = test_fn(da, parDict)
        rootival = (a,b)
        while i <= ev.bisectlimit:
            d = (b-a)/2.0
            p = a + d
            dp = dict(zip(varnames, [v(p) for v in sortedDictValues(vars)]+[p]))
            evp = test_fn(dp, parDict)
            if abs(evp-eva) < ev.eventtol or evp == 0:
                return (p, rootival)
            i += 1
            if evp*eva > 0:
                a = p
                eva = evp
            else:
                b = p
            # do this at end of while loop in case i > bisectlimit
            rootival = (a,b)
        # search failed after bisectlimit
        return (None, rootival)


