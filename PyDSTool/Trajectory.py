"""Trajectory classes.

   Robert Clewley, June 2005.
"""

# ----------------------------------------------------------------------------

# PyDSTool imports
from .Variable import *
from .Interval import *
from .Points import *
from .utils import *
from .common import *
from .common import _num_types
from .parseUtils import *
from .errors import *

# Other imports
from numpy import array, arange, float64, int32, concatenate, zeros, shape, \
     sometrue, alltrue, any, all, ndarray, asarray, unique
from scipy.optimize import bisect, newton
from numpy.linalg import norm
import numpy as np
import math
import copy
import sys

# -------------------------------------------------------------

## Exports

__all__ = ['Trajectory', 'HybridTrajectory', 'numeric_to_traj',
           'pointset_to_traj', 'convert_ptlabel_events', 'findApproxPeriod']

# -------------------------------------------------------------


def numeric_to_traj(vals, trajname, coordnames, indepvar=None, indepvarname='t',
                    indepdomain=None, all_types_float=True, discrete=True,
                    event_times=None, event_vals=None):
    """Utility function to convert one or more numeric type to a Trajectory.
    Option to make the trajectory parameterized or not, by pasing an indepvar
    value.

    To use integer types for trajectory coordinates unset the default
    all_types_float=True argument.

    To create interpolated (continuously defined) trajectories, set
    discrete=False, otherwise leave at its default value of True.

    If event_times and event_vals dictionaries (default None) are given, this
    will place them in the resulting Trajectory.
    """
    vars = numeric_to_vars(vals, coordnames, indepvar, indepvarname,
                           indepdomain, all_types_float, discrete)
    if event_times is not None:
        return Trajectory(trajname, list(vars.values()),
                      parameterized=indepvar is not None,
                      eventTimes=event_times, events=event_vals)
    else:
        return Trajectory(trajname, list(vars.values()),
                      parameterized=indepvar is not None)

def convert_ptlabel_events(pts, return_vals_dict=False):
    """Creates an eventTimes-like dictionary from a pointset's labels.
    (Event X time recorded in a label is recorded as "Event:X").

    With return_vals_dict=True (default False), the corresponding dictionary
    of pointlists for those events will be returned.
    """
    ev_labels = []
    for l in pts.labels.getLabels():
        if 'Event:' in l:
            ev_labels.append(l[6:])
    event_times = {}
    event_vals = {}
    for l in ev_labels:
        ts = []
        ixlist = []
        ix_dict = pts.labels.by_label['Event:'+l]
        # don't use tdict in labels in case indepvararray has been re-scaled
        for ix in ix_dict.keys():
            ts.append(pts.indepvararray[ix])
            ixlist.append(ix)
        ts.sort()
        if return_vals_dict:
            ixlist.sort()
            event_vals[l] = pointsToPointset([pts[ix] for ix in ixlist],
                                             't', ts)
        event_times[l] = ts
    if return_vals_dict:
        return event_times, event_vals
    else:
        return event_times


def pointset_to_traj(pts, events=None):
    """Convert a pointset into a trajectory using linear interpolation, retaining
    parameterization by an independent variable if present.

    If events (default None) is a corresponding event structure's dictionary
    from a Generator object (gen.eventstruct.events), this will preserve the
    events found in the pointset and place them in the resulting Trajectory.
    """
    if events is not None:
        all_ev_names = list(events.keys())
        ev_times, ev_vals = convert_ptlabel_events(pts, True)
        unused_keys = remain(all_ev_names, ev_times.keys())
        for k in unused_keys:
            ev_times[k] = []
            ev_vals[k] = None
    else:
        ev_times = ev_vals = None
    if isparameterized(pts):
        return numeric_to_traj(pts.coordarray, pts.name, pts.coordnames, pts.indepvararray,
                               pts.indepvarname, discrete=False,
                               event_times=ev_times, event_vals=ev_vals)
    else:
        return numeric_to_traj(pts.coordarray, pts.name, pts.coordnames, discrete=False,
                               event_times=ev_times, event_vals=ev_vals)


class Trajectory(object):
    """Parameterized and non-parameterized trajectory class.
    vals must be a sequence of variable objects.

    Non-parameterized objects are created by implicit function-defined
    generators, e.g. ImplicitFnGen.
    """

    def __init__(self, name, vals, coordnames=None, modelNames=None,
                 timeInterval=None, modelEventStructs=None,
                 eventTimes=None, events=None,
                 FScompatibleNames=None, FScompatibleNamesInv=None,
                 abseps=None, globalt0=0, checklevel=0, norm=2,
                 parameterized=True):
        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("name argument must be a string")
        varlist = vals
        if isinstance(varlist, Variable):
            # then a singleton was passed
            varlist = [varlist]
        elif isinstance(varlist, _seq_types):
            if not isinstance(varlist[0], Variable):
                raise TypeError("vals argument must contain Variable objects")
        else:
            raise TypeError("vals argument must contain Variable objects")
        if coordnames is None:
            self.coordnames = [v.name for v in varlist]
        else:
            # unlikely case: filter which variables are used
            # (because of common API with HybridTrajectory)
            self.coordnames = copy.copy(coordnames)
            varlist_new = [(v.name, v) for v in varlist]
            varlist_new.sort()  # sort by the names
            varlist = [v for (vname, v) in varlist_new if vname in self.coordnames]
        if not isUniqueSeq(self.coordnames):
            raise ValueError("Coordinate names must be unique")
        self.coordnames.sort()
        if abseps is not None:
            for v in varlist:
                try:
                    v.indepdomain._abseps = abseps
                except AttributeError:
                    # domain is just an array
                    pass
                try:
                    v.depdomain._abseps = abseps
                except AttributeError:
                    # domain is just an array
                    pass
                try:
                    v.trajirange._abseps = abseps
                except AttributeError:
                    # domain is just an array
                    pass
                try:
                    v.trajdrange._abseps = abseps
                except AttributeError:
                    # domain is just an array
                    pass
        # non-parameterized trajectory?
        indepvarname = varlist[0].indepvarname
        if indepvarname in self.coordnames or not parameterized:
            self._parameterized = False
        else:
            self._parameterized = True
        indepvartype = varlist[0].indepvartype
        if not all([compareNumTypes(v.coordtype, float) for v in varlist]):
            coordtype = float
        else:
            coordtype = varlist[0].coordtype
        self.depdomain = {}
        if varlist[0].trajirange is None:
            indepdomain = varlist[0].indepdomain
            self.depdomain[varlist[0].coordname] = varlist[0].depdomain
        else:
            indepdomain = varlist[0].trajirange
            self.depdomain[varlist[0].coordname] = varlist[0].trajdrange
        for v in varlist[1:]:
            if isinstance(indepdomain, ndarray):
                if v.trajirange is None:
                    if all(v.indepdomain != indepdomain):
                        raise ValueError("Some variables in varlist argument "
                                "have different independent variable domains")
                else:
                    if all(v.trajirange != indepdomain):
                        raise ValueError("Some variables in varlist argument "
                                "have different independent variable domains")
            else:
                if v.trajirange is None:
                    if v.indepdomain != indepdomain:
                        raise ValueError("Some variables in varlist argument "
                                "have different independent variable domains")
                else:
                    if v.trajirange != indepdomain:
                        raise ValueError("Some variables in varlist argument "
                                "have different independent variable domains")
            if v.indepvarname != indepvarname:
                raise ValueError("Some variables in varlist "
                  "argument have different independent variable names")
            if v.indepvartype != indepvartype:
                raise ValueError("Some variables in varlist "
                  "argument have different independent "
                  "variable types")
            if v.trajdrange is None:
                self.depdomain[v.coordname] = v.depdomain
            else:
                self.depdomain[v.coordname] = v.trajdrange
        self.indepvarname = indepvarname
        self.indepvartype = indepvartype
        # self.indepdomain is not checked when traj called, it's just for info
        self.indepdomain = indepdomain
        self.coordtype = coordtype
        self.variables = {}
        for v in varlist:
            assert isinstance(v, Variable)
            assert v.defined, ("Variables must be defined before building "
                               "a Trajectory object")
            self.variables[v.name] = v
        self.dimension = len(varlist)
        self._name_ix_map = invertMap(self.coordnames)
        self.globalt0 = globalt0
        self.checklevel = checklevel   # unused
        # order for norm of points within trajectory
        self._normord = norm
        # funcspec compatible names especially used when part of a large
        # model created using hierarchical object names
        if FScompatibleNames is None:
            self._FScompatibleNames = symbolMapClass()
        else:
            self._FScompatibleNames = FScompatibleNames
        if FScompatibleNamesInv is None:
            if FScompatibleNames is not None:
                self._FScompatibleNamesInv = FScompatibleNames.inverse()
            else:
                self._FScompatibleNamesInv = symbolMapClass()
        else:
            self._FScompatibleNamesInv = FScompatibleNamesInv
        # used by Generators that leave their mark here (for reference by
        # models in which these are embedded, and to be consistent with
        # HybridTraj API)
        self.modelNames = modelNames
        if modelEventStructs is None:
            self.modelEventStructs = None
        else:
            self.modelEventStructs = copy.copy(modelEventStructs)
        if events is None:
            self.events = {}
        else:
            self.events = copy.copy(events)
        if eventTimes is None:
            self._createEventTimes()
        else:
            self.eventTimes = eventTimes
        self.timePartitions = [(self.indepdomain, self.globalt0, checklevel)]

    def _createEventTimes(self):
        eventTimes = {}
        # if events was provided then fill in with that -- most
        # commonly used by Generators that only pass events object
        for evname, evpts in self.events.items():
            if evpts is None:
                eventTimes[evname] = []
            else:
                val = mapNames(self._FScompatibleNamesInv, evpts)
                if evname in eventTimes:
                    eventTimes[evname].extend(val.indepvararray.tolist())
                else:
                    eventTimes[evname] = val.indepvararray.tolist()
        self.eventTimes = eventTimes

    def delete_variables(self, coords):
        """coords is a list of coordinate names to remove"""
        assert alltrue([c in self.variables for c in coords]), \
               "Variable name %s doesn't exist"%c
        assert len(coords) < self.dimension, "Cannot delete every variable!"
        self.coordnames = remain(self.coordnames, coords)
        self._name_ix_map = invertMap(self.coordnames)
        self.dimension -= len(coords)
        for c in coords:
            del self.variables[c]
            del self.depdomain[c]

    def mapNames(self, themap):
        """themap is a symbolMapClass mapping object for remapping coordinate names
        """
        new_coordnames = array(themap(self.coordnames)).tolist()
        assert isUniqueSeq(new_coordnames), 'Coordinate names must be unique'
        new_coordnames.sort()
        if self._parameterized:
            assert self.indepvarname not in new_coordnames, \
                   'Coordinate names must not overlap with independent variable'
        for c in self.coordnames:
            self.variables[c].name = themap(c)
        self.variables = themap(self.variables)
        self.depdomain = themap(self.depdomain)
        self._FScompatibleNames = themap(self._FScompatibleNames)
        self._FScompatibleNamesInv = self._FScompatibleNames.inverse()
        self.coordnames = new_coordnames
        self._name_ix_map = invertMap(self.coordnames)


    def truncate_to_idx(self, idx):
        """Truncate trajectory according to a last coordinate specified by idx
        argument, provided trajectory is defined by an underlying mesh."""
        t_vals = None
        for vname, v in self.variables.items():
            # vmesh is an array with two rows, the first = indep var, the second = dep var
            vmesh = v.underlyingMesh()
            if vmesh is not None:
                # adjust mesh for this variable
                if t_vals is None:
                    t_vals = vmesh[0]
                    if idx > len(t_vals):
                        raise ValueError("Index out of range for truncation of variable"
                                 " %s in trajectory %s"%(vname, self.name))
                v.truncate_to_idx(idx)
                # adjust self.depdomain if it has changed
                if v.trajdrange is None:
                    self.depdomain[v.coordname] = v.depdomain
                else:
                    self.depdomain[v.coordname] = v.trajdrange
        if self._parameterized:
            new_t_end = t_vals[idx]
            # adjust own independent variable
            if self.globalt0 > new_t_end:
                self.globalt0 = new_t_end
            self.indepdomain.set([self.indepdomain[0], new_t_end])

    def truncate_to_indepvar(self, t):
        """Truncate trajectory according to an independent variable value given
        by t argument, provided trajectory is defined as parameterized."""
        #assert self._parameterized, "Only call parameterized trajectories"
        #if self.globalt0 > t:
        #    self.globalt0 = t
        raise NotImplementedError("This feature is not yet implemented")


    def __call__(self, t, coords=None, checklevel=None,
                 asGlobalTime=False, asmap=False):
        """Evaluate a parameterized trajectory at given independent
        variable value(s), interpreted as local times by default, unless
        asGlobalTime==True (default is False, unlike with sample method).

        asmap option allows continuous trajectory to be called as a map, with
        t = 0 or 1 to return the two endpoints. Result includes actual
        time values (in global time if asGlobalTime==True).
        """
        assert self._parameterized, "Only call parameterized trajectories"
        if checklevel is None:
            checklevel = self.checklevel
        if coords is None:
            coordlist = self.coordnames
        elif isinstance(coords, int):
            coordlist = [self.coordnames[coords]]
        elif isinstance(coords, str):
            coordlist = [coords]
        elif isinstance(coords, _seq_types):
            if all([isinstance(c, str) for c in coords]):
                coordlist = [v for v in self.coordnames if v in coords]
                if asmap:
                    # allow independent variable name to be listed now
                    test_names = self.coordnames + [self.indepvarname]
                else:
                    test_names = copy.copy(self.coordnames)
                if remain(coords, test_names) != []:
                    print("Valid coordinate names:%r" % self.coordnames)
                    raise ValueError("Invalid coordinate names passed")
            elif any([isinstance(c, str) for c in coords]):
                raise TypeError("Cannot mix string and numeric values in "
                                  "coords")
            else:
                # numeric list assumed
                coordlist = [v for v in self.coordnames if v in self._name_ix_map]
        else:
            raise TypeError("Invalid type for coords argument")
        if asmap:
            # assumes t will be 0, 1 or list of these only (self.indepdomain
            # will complain otherwise).
            # checklevel is not needed for this usage
            checklevel = 0
            if isinstance(t, _seq_types):
                for tval in t:
                    # convert to continuous times
                    t = [self.indepdomain[tval] for tval in t]
            else:
                # convert to continuous time
                t = self.indepdomain[t]
        if isinstance(t, _seq_types):
            if len(t) == 0:
                raise ValueError("No independent variable value")
            # ensure is an array so that we can subtract globalt0 from all
            if asGlobalTime and not asmap:
                indepvals = array(t) - self.globalt0
            else:
                # variable object will accept any sequence type
                indepvals = t
            try:
                vals = [v(indepvals, checklevel) \
                        for v in [self.variables[vn] for vn in coordlist]]
            except:
                if checklevel > 1:
                    print("\nProblem calling with coords: %r" % coordlist)
                    print("Independent variable values: %r" % indepvals)
                    print("Containment:" + self.indepdomain.contains(t))
                    try:
                        print(self.variables[coordlist[0]].indepdomain.get())
                        print(self.variables[coordlist[0]].depdomain.get())
                    except AttributeError:
                        # domains may be discrete
                        print("Discrete variable domain and/or range")
#                print self.variables[coordlist[0]].initialconditions
                raise
#                raise ValueError("Problem calling at these independent variable values")
            if asmap:
                if asGlobalTime:
                    vals.append(array(t))
                else:
                    # ensure is an array so that we can subtract globalt0 from all
                    # return t vals in global time coords
                    # (internally-generated local t vals)
                    vals.append(array(t) + self.globalt0)
                coordlist.append(self.indepvarname)
##            if len(coordlist) == 1:
##                return array(vals[0], self.coordtype)
##            else:
            #if asGlobalTime:
            #    offset = 0
            #else:
            offset = self.globalt0
            return self._FScompatibleNamesInv(
               Pointset({'coordarray': vals,
                         'coordnames': coordlist,
                         'coordtype': self.coordtype,
                         'indepvararray': array(indepvals)+offset,
                         'norm': self._normord,
                         'name': self.name + "_sample"}))
        else:
            if asGlobalTime and not asmap:
                t = t - self.globalt0
            try:
                varvals = [v(t, checklevel) for v in \
                                 [self.variables[vn] for vn in coordlist]]
            except:
                if checklevel > 1:
                    print("\nProblem calling with coords: %r" % coordlist)
                    print("Independent variable values:", t)
                    print("Containment: %s" % self.indepdomain.contains(t))
                    try:
                        print(self.variables[coordlist[0]].indepdomain.get())
                        print(self.variables[coordlist[0]].depdomain.get())
                    except AttributeError:
                        # domains may be discrete
                        print("Discrete variable domain and/or range")
                raise
            if asmap:
                coordlist.append(self.indepvarname)
                if asGlobalTime:
                    varvals.append(t)
                else:
                    # return t val in global time, so have to
                    # add globalt0 to the internally-generated local t val
                    varvals.append(t + self.globalt0)
##            if len(coordlist) == 1:
##                # covers asmap == False case
##                return self.variables[coordlist[0]](t, checklevel)
##            else:
            #if asGlobalTime:
            #    offset = 0
            #else:
            offset = self.globalt0
            return self._FScompatibleNamesInv(
                Point({'coordarray': varvals,
                       'coordnames': coordlist,
                       'coordtype': self.coordtype,
                       'indepvararray': array(t)+offset,
                       'norm': self._normord}))


    def underlyingMesh(self, coords=None, FScompat=True):
        """Return a dictionary of the underlying independent variables` meshes,
        where they exist. (Dictionary contains references to the meshes,
        not copies of the meshes.) Always returns data in local time frame.

        FScompat option selects whether to use FuncSpec compatible
        naming for variables (mainly for internal use)."""
        # contrast to code to pull out both dependent and independent
        # variables' data points in Variable class' getDataPoints
        if coords is None:
            coords = self.coordnames
        if not isinstance(coords, list):
            raise TypeError('coords argument must be a list')
        meshdict = {}
        if FScompat:
            cs = self._FScompatibleNames(coords)
        else:
            cs = coords
        for varname in cs:
            meshdict[varname] = self.variables[varname].underlyingMesh()
        if FScompat:
            return self._FScompatibleNamesInv(meshdict)
        else:
            return meshdict


    def sample(self, coords=None, dt=None, tlo=None, thi=None,
                   doEvents=True, precise=False, asGlobalTime=True):
        """Uniformly sample the named trajectory over range indicated.
        Returns a Pointset.

        If doEvents=True (default), the event points for the trajectory
          are included in the output, regardless of the dt sampling rate.

        The order of variable names given in the 'coords' argument is ignored.

        precise=True causes the trajectory position to be evaluated precisely
          at the t values specified, which will invoke slow interpolation
          (requires dt to be set otherwise an exception will be raised).
        precise=False (default) causes the nearest underlying mesh positions
          to be used, if available (otherwise the behaviour is the same as
          precise=True provided dt has been set, otherwise an exception will be
          raised).

        If dt is not given, the underlying time mesh is used, if available.
        """

        if coords is None:
            coords_sorted = self.coordnames
        else:
            if not isinstance(coords, list):
                assert isinstance(coords, str), \
                       "coords must be a list of strings or a singleton string"
                coords_sorted = [self._FScompatibleNames(coords)]
            else:
                coords_sorted = self._FScompatibleNames(coords)
                coords_sorted.sort()
        # keep these as separate calls in case of singleton domain (won't unpack properly)
        tlo_base = self.indepdomain.get(0)
        thi_base = self.indepdomain.get(1)
        if tlo is None:
            tlo = tlo_base
        elif asGlobalTime:
            tlo = tlo - self.globalt0
            if tlo < tlo_base:
                tlo = tlo_base
            if tlo >= thi_base:
                # message used by HybridTrajectory.sample
                raise ValueError("tlo too large")
        elif tlo < tlo_base:
            tlo = tlo_base
        elif tlo >= thi_base:
            # message used by HybridTrajectory.sample
            raise ValueError("tlo too large")
        if thi is None:
            thi = thi_base
        elif asGlobalTime:
            thi = thi - self.globalt0
            if thi > thi_base:
                thi = thi_base
            if thi <= tlo_base:
                # message used by HybridTrajectory.sample
                raise ValueError("thi too small")
        elif thi > thi_base:
            thi = thi_base
        elif thi <= tlo_base:
            # message used by HybridTrajectory.sample
            raise ValueError("thi too small")
        if self.indepdomain.issingleton:
            # res will be an array if coords_sorted is a singleton,
            # otherwise a pointset
            res = self.__call__([self.indepdomain.get()], coords_sorted)
            try:
                res.name = self.name + "_sample"
            except AttributeError:
                # res is array
                pass
            return res
        else:
            assert tlo < thi, 't start point must be less than t endpoint'
            if dt is not None and dt >= abs(thi-tlo):
                if precise:
                    print("dt = %f for interval [%f,%f]"%(dt,tlo,thi))
                    raise ValueError('dt must be smaller than time interval')
                else:
                    dt = (thi-tlo)/10.
        if precise:
            if dt is None:
                raise ValueError("Must specify an explicit dt when precise flag"
                                 " is set")
            else:
                tmesh = concatenate((arange(tlo, thi, dt),[thi]))
                loix = 0
                hiix = len(tmesh)
                meshes_ok = True
        else:
            # not precise, so attempt to use underlying mesh for each variable,
            # if available (always in local timeframe)
            meshdict = self.underlyingMesh(FScompat=False)
            meshes_ok = False
            if meshdict is not None:
                if not any([meshdict[v] is None for v in coords_sorted]):
                    # ensure all vars' meshes are identical!
                    firstvar_tmesh = asarray(meshdict[coords_sorted[0]][0].copy())
                    meshes_ok = True
                    # explicitly checking that the arrays are the same
                    # is very slow -- just catch a resulting error later
            if meshes_ok:
                loix_a = ( firstvar_tmesh >= tlo ).tolist()
                try:
                    loix = loix_a.index(1)
                except ValueError:
                    loix = 0
                hiix_a = ( firstvar_tmesh > thi ).tolist()
                try:
                    hiix = hiix_a.index(1)
                except ValueError:
                    hiix = len(firstvar_tmesh)
                firstvar_tmesh = firstvar_tmesh[loix:hiix]
                if dt is None:
                    tmesh = firstvar_tmesh
                else:
                    # this is just the goal mesh, but with precise=False
                    # only the closest actual mesh points will be used
                    tmesh = concatenate((arange(tlo, thi, dt), [thi]))
                    # get closest mesh indices corresponding to tmesh
                    # times for first var
                    closest = makeSeqUnique(findClosestArray(firstvar_tmesh,
                                                       tmesh, dt/2), True)
                    # filter any out-of-range tvals picked
                    tmesh = array([t for t in firstvar_tmesh[closest] \
                                   if t >= tlo and t <= thi])

        if meshes_ok:
            ix_range = arange(loix, hiix)
        else:
            if dt is None:
                raise PyDSTool_ValueError("Underlying mesh of trajectory is not"
                             " the same for all variables: dt must "
                             "be specified in this case")
            else:
                # tmesh not created
                precise = True
                tmesh = concatenate((arange(tlo, thi, dt), [thi]))

        # check for loss of precision when restoring to global time,
        # and if present, just filter out "equal" times
        if asGlobalTime:
            tmesh_glob = tmesh + self.globalt0
            if not isincreasing(tmesh_glob):
                # don't need tmesh_glob_filtered here
                tmesh_glob, good_ixs = unique(tmesh_glob, True)
                tmesh = tmesh[good_ixs]
                ix_range = ix_range[good_ixs]
        else:
            tmesh_glob = tmesh


        # Add Pointset labels for events
        # Work in global time in case of loss of precision in small
        # time differences between time points and event points
        evlabels = PointInfo()
        if doEvents:
            eventdata = self.getEventTimes(asGlobalTime=True)
            eventpts = self.getEvents(asGlobalTime=True)
            evtlist = [(t, evname) for (t, evname) in \
                       orderEventData(eventdata, bytime=True) \
                       if t >= tmesh_glob[0] - self.indepdomain._abseps and \
                          t <= tmesh_glob[-1] + self.indepdomain._abseps]
            ev_pts_list = []
            if evtlist != []:
                ev_ts = [t for (t, evname) in evtlist]
                for t, evname in evtlist:
                    # get each state point
                    tix = eventpts[evname].find(t)
                    ev_pts_list.append(eventpts[evname][tix])
                tmesh_glob, ins_ixs, close_ix_dict = insertInOrder( \
                                               tmesh_glob,
                                               ev_ts, return_ixs=True,
                                               abseps=self.indepdomain._abseps)
            else:
                ins_ixs = []
            if len(ins_ixs) != len(evtlist):
                # check to see if events need to be inserted at start
                # or end of tmesh (insertInOrder doesn't do that)
                # -- it may just be that the event times are already in tmesh
                start_ins = []
                for t in ev_ts:
                    if t < tmesh_glob[0] - self.indepdomain._abseps:
                        start_ins.append(t)
                    else:
                        break
                end_ins = []
                for t in ev_ts[::-1]:
                    if t - tmesh_glob[-1] > self.indepdomain._abseps:
                        end_ins.append(t)
                    else:
                        break
                end_ins.reverse()
                if start_ins != [] or end_ins != []:
                    lastins = len(tmesh_glob)
                    lenstart = len(start_ins)
                    tmesh_glob = concatenate((start_ins, tmesh_glob, end_ins))
                    ins_ixs = list(range(lenstart)) + [i+lenstart for i in ins_ixs] + \
                            list(range(lastins+lenstart, lastins+lenstart+len(end_ins)))
            tmesh_list = tmesh_glob.tolist()
            for i, (t, evname) in enumerate(evtlist):
                try:
                    ix = ins_ixs[i]
                except IndexError:
                    # already in tmesh
                    try:
                        # t may only match up to rounding error
                        ix = close_ix_dict[t]
                    except KeyError:
                        # t will match exactly
                        ix = tmesh_list.index(t)
                if asGlobalTime:
                    evlabels.update(ix, 'Event:'+evname, {'t': t})
                else:
                    evlabels.update(ix, 'Event:'+evname, {'t': t-self.globalt0})
        else:
            eventdata = None

        if asGlobalTime:
            tmesh = tmesh_glob - self.globalt0
        else:
            tmesh = tmesh_glob

        if len(tmesh) > 0:
            if dt is None:
                coorddict = dict(zip(coords_sorted, [m[1][ix_range] for m in \
                                 sortedDictValues(meshdict,
                                                onlykeys=coords_sorted)]))
                if eventdata:
                    # insert var values at events (SLOW!)
                    # can only insert to lists, so have to convert coorddict
                    # arrays to lists first.
                    for tpos in range(len(ins_ixs)):
                        tix = ins_ixs[tpos]
                        t = ev_ts[tpos]
                        x = self._FScompatibleNames(ev_pts_list[tpos])
                        for v in coords_sorted:
                            try:
                                coorddict[v].insert(tix, x[v])
                            except AttributeError:
                                coorddict[v] = coorddict[v].tolist()
                                coorddict[v].insert(tix, x[v])
                # if asGlobalTime, tmesh already includes
                # + self.globalt0 offset to tlo, thi, but must
                # return independent var values in global timeframe
                if asGlobalTime:
                    # have to watch out for loss of precision here, especially
                    # due to events very close together
                    tmesh += self.globalt0
                return self._FScompatibleNamesInv(
                        Pointset({'coorddict': coorddict,
                                  'coordtype': float64,
                                  'coordnames': coords_sorted,
                                  'indepvararray': tmesh,
                                  'indepvarname': self.indepvarname,
                                  'indepvartype': float64,
                                  'norm': self._normord,
                                  'name': self.name + "_sample",
                                  'labels': evlabels}))
            elif not precise:
                try:
                    coorddict = {}
                    for v in coords_sorted:
                        dat = self.variables[v].output.datapoints[1]
                        try:
                            coorddict[v] = dat[closest+loix]
                        except TypeError:
                            dat = array(dat)
                            coorddict[v] = dat[closest+loix]
                except (AttributeError, IndexError):
                    # meshes of variables did not match up, so
                    # continue to 'else' case, as if precise=True
                    pass
                else:
                    if eventdata:
                        # insert var values at events (SLOW!)
                        # can only insert to lists, so have to convert coorddict
                        # arrays to lists first.
                        for tpos in range(len(ins_ixs)):
                            tix = ins_ixs[tpos]
                            t = ev_ts[tpos]
                            x = self._FScompatibleNames(ev_pts_list[tpos])
                            for v in coords_sorted:
                                try:
                                    coorddict[v].insert(tix, x[v])
                                except AttributeError:
                                    coorddict[v] = coorddict[v].tolist()
                                    coorddict[v].insert(tix, x[v])
                    # if asGlobalTime, tmesh already includes
                    # self.globalt0 offset to tlo, thi, but must
                    # return independent var values in global timeframe
                    if asGlobalTime:
                        tmesh += self.globalt0
                    return self._FScompatibleNamesInv(
                        Pointset({'coorddict': coorddict,
                                   'coordtype': float64,
                                   'coordnames': coords_sorted,
                                   'indepvararray': tmesh,
                                   'indepvarname': self.indepvarname,
                                   'indepvartype': float64,
                                   'norm': self._normord,
                                   'name': self.name + "_sample",
                                   'labels': evlabels}))

            # else mesh not available for some variables so we have
            # a mixed-source trajectory, and simple solution is to
            # treat as if precise==True
            # Cases: either precise == True or could not extract
            # underlying meshes for some variables, or the meshes were not
            # identical.
            # call self with asGlobalTime=False b/c tmesh already in "local"
            # time frame, global offset already adjusted for.
            vals = self(tmesh, coords_sorted, asGlobalTime=False)
            try:
                # returned pointset
                vals = vals.toarray()
            except AttributeError:
                # was array already
                pass
            pset = Pointset({'coordarray': vals,
                     'coordnames': coords_sorted,
                     'indepvararray': array(tmesh),   # globalt0 added later
                     'indepvarname': self.indepvarname,
                     'name': self.name + "_sample",
                     'labels': evlabels})
            # if coords was not sorted, call to self could potentially return
            # an array in a different order to that of coords.
            if asGlobalTime and self.globalt0 != 0:
                pset.indepvararray += self.globalt0
                pset.makeIxMaps()
            return self._FScompatibleNamesInv(pset)
        else:
            return None


    def __copy__(self):
        pickledself = pickle.dumps(self)
        c = pickle.loads(pickledself)
##        c.indepdomain = copy.copy(self.indepdomain)
##        c.depdomain = copy.copy(self.depdomain)
##        for vname, v in self.variables.iteritems():
##            c.variables[vname] = copy.copy(v)
        return c


    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        # remove reference to Cfunc types by converting them to strings
        d['indepvartype'] = _num_type2name[self.indepvartype]
        d['coordtype'] = _num_type2name[self.coordtype]
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc types
        self.indepvartype = _num_name2type[self.indepvartype]
        self.coordtype = _num_name2type[self.coordtype]


    def __del__(self):
        # necessary for __del__ methods of variables to be accessed
        for v in self.variables.values():
            v.__del__()


    def __repr__(self):
        return self._infostr(verbose=0)


    __str__ = __repr__


    def _infostr(self, verbose=1):
        if verbose <= 0:
            outputStr = "Trajectory " + self.name
        elif verbose >= 1:
            outputStr = "Trajectory " + self.name + "\n  of variables: " + \
                    str(self._FScompatibleNamesInv(list(self.variables.keys())))
            outputStr += "\n  over domains: " + str(self.depdomain)
            if verbose == 2:
                outputStr += joinStrs(["\n"+v._infostr(1) for v in \
                                            self.variables.values()])
        return outputStr

    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))


    def getEvents(self, evnames=None, asGlobalTime=True):
        """Returns a pointset of all named events occuring in global time,
        unless asGlobalTime option set to False (default is True).
        If no events are named, all are used.
        """
        # self.events is a dict of pointsets keyed by event name
        if evnames is None:
            evnames = list(self.events.keys())
        if isinstance(evnames, str):
            # singleton
            assert evnames in self.events, "Invalid event name provided: %s"%evnames
            result = copy.copy(self.events[evnames])
            if asGlobalTime:
                try:
                    result.indepvararray += self.globalt0
                except AttributeError:
                    # empty pointset
                    pass
        else:
            result = {}
            # assume iterable
            assert all([ev in self.events for ev in evnames]), \
                   "Invalid event name(s) provided: %s"%str(evnames)
            for evname in evnames:
                evptset = self.events[evname]
                result[evname] = copy.copy(evptset)
                if asGlobalTime and self.globalt0 != 0:
                    try:
                        result[evname].indepvararray += self.globalt0
                    except AttributeError:
                        # empty pointset
                        pass
        return result


    def getEventTimes(self, evnames=None, asGlobalTime=True):
        """Returns a list of times at which the named events occurred in global
        time, unless asGlobalTime option set to False (default is True).
        If no events are named, all are used.
        """
        result = {}
        if evnames is None:
            evnames = list(self.events.keys())
        # self.eventTimes is a dict of lists keyed by event name
        if isinstance(evnames, str):
            # singleton
            assert evnames in self.events, "Invalid event name provided: %s"%evnames
            if asGlobalTime:
                result = list(array(self.eventTimes[evnames]) + self.globalt0)
            else:
                result = self.eventTimes[evnames]
        else:
            if asGlobalTime:
                t_offset = self.globalt0
            else:
                t_offset = 0
            # assume iterable
            assert all([ev in self.events for ev in evnames]), \
                   "Invalid event name(s) provided %s"%str(evnames)
            for evname in evnames:
                try:
                    evtimes = self.eventTimes[evname]
                except KeyError:
                    pass
                else:
                    result[evname] = list(array(evtimes) + t_offset)
        return result




class HybridTrajectory(Trajectory):
    """Hybrid, parameterized, trajectory class.
    Mimics API of a non-hybrid Trajectory.

    vals must be a sequence of trajectory segments.
    coords is optional (restricts variables to store).
    """
    def __init__(self, name, vals, coordnames=None, modelNames=None,
                 timeInterval=None, modelEventStructs=None,
                 eventTimes=None, events=None,
                 FScompatibleNames=None, FScompatibleNamesInv=None,
                 abseps=None, globalt0=0, checklevel=0, norm=2,
                 timePartitions=None):
        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("name argument must be a string")
        trajSeq = vals
        self.indepvarname = trajSeq[0].indepvarname
        if coordnames is None:
            # initial value for coordnames only
            self.coordnames = trajSeq[0].coordnames
            # choose coordnames to be all common variable names
            for traj in trajSeq:
                self.coordnames = intersect(self.coordnames, traj.coordnames)
                assert traj.indepvarname == self.indepvarname, \
                         "Inconsistent independent variable names"
        else:
            self.coordnames = copy.copy(coordnames)
            for traj in trajSeq:
                assert traj.indepvarname == self.indepvarname, \
                         "Inconsistent independent variable names"
            if not isUniqueSeq(self.coordnames):
                raise ValueError("Coordinate names must be unique")
        self.coordnames.sort()
        # don't enforce self.indepvartype == trajSeq[i].indepvartype
        # for when hybrid trajectories can consist of map and non-map parts
        # (some time points will simply not exist in those cases)
        self.indepvartype = trajSeq[0].indepdomain.type
        self.coordtype = float    # may have to change this
        if timeInterval is None:
            # extract from trajSeq
            tlo = trajSeq[0].indepdomain[0]
            thi = trajSeq[-1].indepdomain[1]
            timeInterval = Interval(self.indepvarname, self.indepvartype,
                                         [tlo, thi], abseps=self._abseps)
        # partitions are (interval, globalt0, checklevel) triples
        self.timePartitions = timePartitions
        # sequence (list) of other Trajectory or HybridTrajectory objects
        self.trajSeq = trajSeq
        # list of models generating each traj in trajSeq
        self.modelNames = modelNames
        self.modelEventStructs = modelEventStructs
        if events is None:
            self.events = {}
        else:
            self.events = events
        if eventTimes is None:
            self.eventTimes = {}
        else:
            self.eventTimes = eventTimes
        if FScompatibleNames is None:
            self._FScompatibleNames = symbolMapClass()
        else:
            self._FScompatibleNames = FScompatibleNames
        if FScompatibleNamesInv is None:
            self._FScompatibleNamesInv = symbolMapClass()
        else:
            self._FScompatibleNamesInv = FScompatibleNamesInv
        # mimic regular variables attribute of non-hybrid Trajectory
        self.variables = {}
        self.depdomain = {}
        for ov in self.coordnames:
            self.variables[ov] = HybridVariable(self, ov,
                                            timeInterval, abseps=abseps)
            self.depdomain[ov] = Interval(ov, float, [-np.Inf, np.Inf],
                                            abseps=abseps)
        self.globalt0 = globalt0
        self.indepdomain = timeInterval
        self._parameterized = True
        self.dimension = len(self.coordnames)
        self._name_ix_map = invertMap(self.coordnames)
        self.checklevel = checklevel    # unused
        # order for norm of points within trajectory
        self._normord = norm

    def underlyingMesh(self, coords=None, FScompat=True):
        """Return a dictionary of the underlying independent variables` meshes,
        where they exist. (Dictionary contains references to the meshes,
        not copies of the meshes.) Always returns data in local time frame.

        FScompat option selects whether to use FuncSpec compatible
        naming for variables (mainly for internal use)."""
        # contrast to code to pull out both dependent and independent
        # variables' data points in Variable class' getDataPoints
        if coords is None:
            coords = self.coordnames
        if not isinstance(coords, list):
            raise TypeError('coords argument must be a list')
        meshdict = {}
        if FScompat:
            cs = self._FScompatibleNames(coords)
        else:
            cs = coords
        for varname in cs:
            try:
                meshdict[varname] = self.variables[varname].underlyingMesh()
            except AttributeError:
                meshdict[varname] = None
        if FScompat:
            return self._FScompatibleNamesInv(meshdict)
        else:
            return meshdict

    def showRegimes(self):
        parts = [p[1] for p in self.timePartitions]
        for g, p in zip(self.modelNames, parts):
            print("Regime: %s, from t = %.5f"%(g,p))

    def info(self):
        return info(self, "Hybrid Trajectory")


    def __call__(self, t, coords=None, checklevel=None,
                 asGlobalTime=False, asmap=False):
        """Evaluate a parameterized hybrid trajectory at given independent
        variable value(s), interpreted as global times if optional
        asGlobalTime==True (default is False).

        asmap option allows the trajectory to be called as a map, with
        t = integer values to return the values at successive hybrid
        event times. Result includes actual time values
        (in global time if asGlobalTime==True).
        """
        if checklevel is None:
            checklevel = self.checklevel
        if coords is None:
            coords = copy.copy(self.coordnames)
        else:
            if isinstance(coords, str):
                coords = [self._FScompatibleNames(coords)]
            elif isinstance(coords, list):
                coords = self._FScompatibleNames(coords)
            if not [varname in self.coordnames for varname in coords]:
                raise ValueError('one or more variable names not in observable'
                             ' variable list')
        if asGlobalTime and not asmap:
            if isinstance(t, _seq_types):
                # ensure is an array so that we can subtract globalt0 from all
                t = array(t) - self.globalt0
            else:
                t = t - self.globalt0
        time_interval = self.indepdomain
        # timePartitions are (interval, globalt0, checklevel) triples
        time_partitions = self.timePartitions
        trajseq = self.trajSeq
        if asmap:
            # hybrid system treated as mapping -- include real time value
            # in returned point / pointset
            coords.append(self.indepvarname)
            num_parts = len(time_partitions)
            if isinstance(t, _seq_types):
                dim = len(coords)
                result = []
                for i in range(dim):
                    result.append([])
                for tval in t:
                    if not isinstance(tval, _int_types):
                        raise TypeError("time values must be of integer type "
                         "when treating system as a mapping")
                    if tval < num_parts:
                        # return first point of partition t
                        part_interval = time_partitions[tval][0]
                        callvals = trajseq[tval](0, coords,
                                        asGlobalTime=False, asmap=True)
                        if dim == 1:
                            # callvals is just a float
                            result[0].append(callvals)
                        else:
                            for i in range(dim):
                                result[i].append(callvals(coords[i]))
                        # adjust time
                        result[dim-1][-1] += part_interval[0]
                    elif tval == num_parts:
                        # return final point of final partition
                        part_interval = time_partitions[num_parts-1][0]
                        callvals = trajseq[num_parts-1](1, coords,
                                        asGlobalTime=False, asmap=True)
                        if dim == 1:
                            # callvals is just a float
                            result[0].append(callvals)
                        else:
                            for i in range(dim):
                                result[i].append(callvals(coords[i]))
                        # adjust time
                        result[dim-1][-1] += part_interval[1]
                    else:
                        raise ValueError("time values not in valid "
                                           "partition range")
                return Pointset({'coordarray': array(result),
                     'coordnames': self._FScompatibleNamesInv(coords),
                     'indepvartype': int32,
                     'indepvararray': t,
                     'indepvarname': 'event',
                     'norm': self._normord
                     })
            else:
                if not isinstance(t, int):
                    raise TypeError("time value must be an integer"
                                    " when treating system as a mapping")
                if t not in range(num_parts+1):
                    raise PyDSTool_BoundsError("time value not in valid"
                                                 " partition range")
                if t < num_parts:
                    # return first point of partition t
                    part_interval = time_partitions[t][0]
                    result = trajseq[t](0, coords,
                            asGlobalTime=False, asmap=True).toarray().tolist()
                    # adjust time (final entry of result)
                    result[-1] += part_interval[0]
                else:
                    # return final point of final partition
                    part_interval = time_partitions[num_parts-1][0]
                    result = trajseq[num_parts-1](1, coords,
                            asGlobalTime=False, asmap=True).toarray().tolist()
                    # adjust time (final entry of result)
                    result[-1] += part_interval[1]
                return Point({'coordarray': array(result),
                              'coordnames': \
                                  self._FScompatibleNamesInv(coords)+[self.indepvarname],
                              'norm': self._normord
                             })
        else:
            # hybrid system treated as curve
            if not isinstance(t, _seq_types):
                t = [t]
            retvals = []
            # lastpart_pos optimizes the re-searching of the time
            # partitions for long ranges of t
            ## CURRENTLY UNUSED
            ##lastpart_pos = 0
            tix = 0
            while tix < len(t):
                # list of t vals in a partition, to allow a
                # group-call to the trajectory for that partition
                trel_part = []
                tval = t[tix]
                if time_interval.contains(tval) is notcontained:
                    print("\n** Debugging info for Hybrid Traj %s: t value, interval, tolerance ="%self.name)
                    print("%f %r %f" % (tval, time_interval.get(), time_interval._abseps))
                    raise PyDSTool_BoundsError('time value outside of '
                        'trajectory`s time interval '
                        'of validity (if checklevel was >=2 then endpoints '
                        'were not included in trajectory)')
                trajseq_pos = 0
                tfound = False
                for (part_interval, gt0, cl) in time_partitions:
                    # trelative
                    trelative = tval - gt0   # a source of rounding error
                    contresult = part_interval.contains(trelative)
                    if contresult is contained:
                        # find more t vals in this partition, if any
                        trel_part.append(trelative)
                        part_done = False
                        while not part_done:
                            tix += 1
                            if tix >= len(t):
                                part_done = True
                                continue  # while loop
                            trel = t[tix] - gt0   # a source of rounding error
                            contresult_sub = part_interval.contains(trel)
                            if contresult_sub is contained:
                                trel_part.append(trel)
                            elif contresult_sub is uncertain:
                                try:
                                    dummy = trajseq[trajseq_pos](trel,
                                                        asGlobalTime=False,
                                                        checklevel=checklevel)
                                except (ValueError, PyDSTool_BoundsError):
                                    # traj segment didn't accept this t value
                                    # exit if statement and increment trajseq_pos
                                    call_ok = False
                                else:
                                    call_ok = True
                                if call_ok:
                                    trel_part.append(trel)
                                else:
                                    part_done = True
                            else:
                                part_done = True
                        tfound = True
                        break  # for loop
                    elif contresult is uncertain:
                        # first verify that this partition's trajectory
                        # segment will accept this value ... then,
                        # find more t vals in this partition, if any
                        try:
                            dummy = trajseq[trajseq_pos](trelative,
                                                         asGlobalTime=False,
                                                         checklevel=checklevel)
                        except (ValueError, PyDSTool_BoundsError):
                            # traj segment didn't accept this t value
                            # exit if statement and increment trajseq_pos
                            call_ok = False
                        else:
                            call_ok = True
                        if call_ok:
                            trel_part.append(trelative)
                            part_done = False
                            while not part_done:
                                tix += 1
                                if tix >= len(t):
                                    part_done = True
                                    continue  # while loop
                                trel = t[tix] - gt0   # a source of rounding error
                                contresult_sub = part_interval.contains(trel)
                                if contresult_sub is contained:
                                    trel_part.append(trel)
                                elif contresult_sub is uncertain:
                                    try:
                                        dummy = trajseq[trajseq_pos](trel,
                                                            asGlobalTime=False,
                                                            checklevel=checklevel)
                                    except (ValueError, PyDSTool_BoundsError):
                                        # traj segment didn't accept this t value
                                        # exit if statement and increment trajseq_pos
                                        call_ok = False
                                    else:
                                        call_ok = True
                                    if call_ok:
                                        trel_part.append(trel)
                                    else:
                                        part_done = True
                                else:
                                    part_done = True
                            tfound = True
                            break  # for loop
                    trajseq_pos += 1
                if tfound:
                    # append Pointset of varname values at the t vals
                    val = trajseq[trajseq_pos](trel_part, coords,
                                               asGlobalTime=False,
                                               checklevel=1)
                    if len(t) == 1:
                        retvals = val.toarray().ravel()
##                        try:
##                            retvals = val.ravel()
##                        except AttributeError:
##                            print "Trajectory.py WARNING line 1129 - wrong type found"
##                            retvals = val.toarray().ravel()
                    else:
                        retvals.append(val)
                else:
                    tinterval = self.indepdomain
                    print("valid t interval:" + tinterval.get())
                    print("t in interval? -->" + tinterval.contains(tval))
                    print("interval abs eps = " + tinterval._abseps)
                    raise ValueError('t = '+str(tval)+' is either not in any '
                                       'time interval defined for this '
                                       'trajectory, or there was an out-of-range'
                                       ' value computed (see above warnings). '
                                       'Decrease step-size / event tolerance or'
                                       ' increasing abseps tolerance.')
                # No need to increment tix here. It was incremented inside
                # the inner while loop the last time a tval was checked.
                # A failed containment there left tix at the next new value
            # Only return coords contained in all trajectory segments, but after
            # switching back to user-domain hierarchical names (if used) the
            # coords list needs to be re-sorted to correspond to the already
            # sorted values in retvals
            coordnames = self._FScompatibleNamesInv(intersect( \
                                      trajseq[trajseq_pos].coordnames,
                                      coords))
            coordnames.sort()
            if len(t) == 1:
                if len(coords) == 1:
                    return retvals[0]
                else:
                    return Point({'coordarray': retvals,
                          'coordnames': coordnames,
                           'norm': self._normord})
            else:
                return Pointset({'coordarray': concatenate(retvals).T,
                         'coordnames': coordnames,
                         'indepvartype': self.indepvartype,
                         'indepvararray': t,
                         'indepvarname': self.indepvarname,
                         'norm': self._normord})


    def sample(self, coords=None, dt=None, tlo=None, thi=None,
               doEvents=True, precise=False, asGlobalTime=True):
        """Uniformly sample the trajectory over range indicated,
          including any event points specified by optional 'doEvents'
          argument (default True).
        Returns a pointset.

        precise=False attempts to use the underlying mesh of the trajectory
          to return a Pointset more quickly. Currently, this can only be used
          for trajectories that have a single segment.

        If dt is not given, the underlying time mesh is used, if available.

        asGlobalTime is ignored for this class (time is always global for
          this class).
        """

        if coords is None:
            coords = self.coordnames
        else:
            if not isinstance(coords, list):
                assert isinstance(coords, str), \
                       "coords must be a list of strings or a singleton string"
                coords = [coords]
            coords = self._FScompatibleNames(coords)
        t_interval = self.indepdomain
        if tlo is None:
            tlo = t_interval[0]
        if thi is None:
            thi = t_interval[1]
        if t_interval.issingleton:
            t = t_interval.get()
            # [t] ensures return of a singleton Pointset
            pset = self.__call__([t], coords)
            pset.name = self.name + "_sample"
            if doEvents:
                for evname, ev_ts in list(self.eventTimes.items()):
                    if t in ev_ts:
                        pset.addlabel(0, 'Event:'+evname, {'t': t})
            return pset
        else:
            assert tlo < thi, 't start point must be less than t endpoint'
            if doEvents:
                evs_gen = self.eventTimes
            else:
                evs_gen = None
            if dt is not None:
                assert dt < abs(thi-tlo), 'dt must be smaller than time interval'
##        evs_mod_struct = self.eventStruct
        # evs_mod = ?
##        if evs_mod or evs_gen:
        if precise:
            if dt is None:
                raise ValueError("Must specify an explicit dt when precise flag"
                                 " is set")
            tmesh = concatenate([arange(tlo, thi, dt), [thi]])
            if evs_gen:
##                # combine event data dictionaries
##                evs = copy.copy(evs_gen)
##                if evs_mod != {}:
##                    evs.update(dict(evs_mod))
                # sort into ordered list of times
                evtlist = orderEventData(evs_gen, nonames=True)
                if evtlist != []:
                    tmesh = insertInOrder(tmesh, evtlist)
            if len(tmesh) > 0:
                pset = self(tmesh,coords)
            else:
                return None
        else:
            pset = None
            for traj in self.trajSeq:
                # use fact that Trajectory.sample automatically truncates
                # tlo and thi if they are beyond that segment's defined range.
                if pset is None:
                    try:
                        pset = traj.sample(coords, dt, tlo, thi, doEvents,
                                           False)
                    except ValueError as e:
                        if e.message[:7] in ('tlo too', 'thi too'):
                            # tlo or thi out of range
                            pass
                        else:
                            raise
                else:
                    try:
                        pset_new = traj.sample(coords, dt, tlo, thi,
                                                doEvents, False)
                    except ValueError as e:
                        if e.message[:7] in ('tlo too', 'thi too'):
                            # tlo or thi out of range
                            pass
                        else:
                            raise
                    else:
                        pset.append(pset_new, skipMatchingIndepvar=True)
        pset.indepvarname = self._FScompatibleNamesInv(pset.indepvarname)
        pset.mapNames(self._FScompatibleNamesInv)
        return pset


# ---------------------------------------------------------------------

def findApproxPeriod(traj, t0, t1_guess=None, T_guess=None, coordname=None,
               ttol=1e-5, rtol=1e-2, guess_tol=0.1):
    """findPeriod does not ensure minimum period (it is up to the user to
    select a good initial guess, but guess_tol fraction of T_guess will be
    checked as a bracketing interval for bisection."""
    if t1_guess is None and T_guess is None:
        raise ValueError("Must supply guess for either t1 or period, T")
    if t1_guess is None:
        if T_guess <= 0:
            raise ValueError("Guess for period T must be positive")
        t1_guess = t0+T_guess
    if T_guess is None:
        T_guess = t0+t1_guess
    if t1_guess <= t0:
        raise ValueError("t1 guess must be greater than t0")
    if coordname is None:
        coordname = traj._FScompatibleNamesInv(traj.coordnames[0])
    p0v = traj(t0, coordname)[0]
    def f(t):
        try:
            return traj(t, coordname)[0]-p0v
        #except (PyDSTool_BoundsError, ValueError):
        #    r = 1000.*(t-t0)
        except:
            print("Error at t=%f: "%t + sys.exc_info()[0] + sys.exc_info()[1])
            raise
    try:
        result = bisect(f, t1_guess-guess_tol*(t1_guess-t0),
                        t1_guess+guess_tol*(t1_guess-t0),
                        xtol=ttol)
    except RuntimeError:
        raise ValueError("Did not converge for this initial guess")

    if traj.dimension == 1:
        max_rval = abs(traj(result) - p0v)
        rval = max_rval/abs(p0v)  # used for error info
    else:
        xt0 = traj(t0)
        val = traj(result)-xt0
        rval = [abs(val[vn]/xt0[vn]) for vn in traj._FScompatibleNamesInv(traj.coordnames)]
        max_rval = max(rval)
    if max_rval<rtol:
        return abs(result-t0)
    else:
        print("Did not converge. The endpoint difference at t=%f was:\n"%result +
          repr(val) +
          "\nwith infinity-norm %f > %f tolerance.\n"%(max_rval,rtol) +
          "Try a different starting point," +
          "a different test variable, or increase relative tolerance.")
        raise ValueError("Did not converge")
