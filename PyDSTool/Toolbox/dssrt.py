"""Implementation of dominant scale analysis techniques for python.

Loosely based on Matlab tool DSSRT. This module is incomplete and is
still a work in progress! See my publications on this subject for further
details.

Robert Clewley, 2009

"""

from PyDSTool import common
from PyDSTool import utils
from PyDSTool import HybridModel, NonHybridModel, Interval
from PyDSTool import parseUtils
from PyDSTool import Symbolic, Vode_ODEsystem
from PyDSTool.Trajectory import numeric_to_traj
from PyDSTool.Points import Point, Pointset, pointsToPointset
from PyDSTool.errors import *

import numpy as np
import scipy as sp
import copy
import operator
from itertools import zip_longest, chain
from functools import reduce
from io import StringIO

#############

_classes = ['epoch', 'regime', 'dssrt_assistant', 'domscales',
            'Scorer', 'EpochSeqScorer', 'VarAlphabet']

_predicates = ['is_active', 'is_inactive', 'is_modulatory',
               'is_most_dominant', 'is_fast', 'is_slow', 'is_order1',
               'join_actives', 'leave_actives', 'become_most_dominant',
               'join_fast', 'join_slow', 'leave_fast', 'leave_slow']

_functions = ['find_epoch_period', 'transition_psi', 'transition_tau',
              'get_infs', 'get_taus', 'split_pts', 'define_psi_events',
              'normalized_psis', 'define_tau_events', 'find_regime_transition',
              'find_ep_ix', 'show_epochs', 'plot_psis', 'indent',
              'swdist', 'jaro', 'comp_seqs', 'editdist_edits',
              'MATCH', 'MISMATCH', 'APPROX', 'CLOSE', 'VCLOSE',
              'FAR', 'VFAR', 'GAP', 'get_symbol_sequence', 'tabulate_epoch_seqs']

__all__ = _classes + _functions + _predicates

#############


class join_actives(common.predicate):
    name = 'join_active'

    def precondition(self, epochs):
        res = self.subject not in epochs[0].actives
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, epoch):
        return self.subject in epoch.actives


class leave_actives(common.predicate):
    name = 'leave_actives'

    def precondition(self, epochs):
        res = self.subject in epochs[0].actives
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, epoch):
        return self.subject not in epoch.actives


class become_most_dominant(common.predicate):
    name = 'become_most_dominant'

    def precondition(self, epochs):
        res = self.subject != epochs[0].actives[0]
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, epoch):
        return self.subject == epoch.actives[0]


class join_fast(common.predicate):
    name = 'join_fast'

    def precondition(self, epochs):
        res = self.subject not in epochs[0].fast
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, epoch):
        return self.subject in epoch.fast


class join_slow(common.predicate):
    name = 'join_slow'

    def precondition(self, epochs):
        res = self.subject not in epochs[0].slow
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, epoch):
        return self.subject in epoch.slow


class leave_fast(common.predicate):
    name = 'leave_fast'

    def precondition(self, epochs):
        res = self.subject in epochs[0].fast
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, epoch):
        return self.subject not in epoch.fast


class leave_slow(common.predicate):
    name = 'leave_slow'

    def precondition(self, epochs):
        res = self.subject in epochs[0].slow
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, epoch):
        return self.subject not in epoch.slow


class is_active(common.predicate):
    name = 'is_active'

    def evaluate(self, epoch):
        return self.subject in epoch.actives


class is_inactive(common.predicate):
    name = 'is_inactive'

    def evaluate(self, epoch):
        return self.subject in epoch.inactives


class is_modulatory(common.predicate):
    name = 'is_modulatory'

    def evaluate(self, epoch):
        return self.subject in epoch.modulatory


class is_most_dominant(common.predicate):
    name = 'is_most_dominant'

    def evaluate(self, epoch):
        return self.subject == epoch.actives[0]


class is_fast(common.predicate):
    name = 'is_fast'

    def evaluate(self, epoch):
        return self.subject in epoch.fast


class is_slow(common.predicate):
    name = 'is_slow'

    def evaluate(self, epoch):
        return self.subject in epoch.slow


class is_order1(common.predicate):
    name = 'is_order1'

    def evaluate(self, epoch):
        return self.subject in epoch.order1


# -------------------------------------------------


def find_ep_ix(eps, t):
    """Finds epoch during time t"""
    for i, ep in enumerate(eps):
        if t >= ep.t0:
            try:
                test = t < eps[i+1].t0
            except IndexError:
                test = t <= eps[i].t1
            if test:
                return i
        else:
            raise ValueError("t must be in the epochs' time intervals")


### Currently unused
class regime(object):
    def __init__(self, epochs, global_criteria, transition_criteria):
        self.epochs = epochs
        # all actives assumed to be the same
        self.actives = epochs[0].actives
        self.t0 = epochs[0].t0
        self.t1 = epochs[-1].t1
        # criteria are simple predicates or compound predicates built
        # using and_op, or_op, not_op operator constructs
        self.global_criteria = global_criteria
        self.transition_criteria = transition_criteria


### Currently unused
#class criterion_record(object):
#    def __init__(self):
#        self.time


def test_criterion(epochs, glob, trans, min_t, force_precond=False):
    """Use force_precond (optional, default=False) to force any preconditions
    to return True. This is useful when recalculating epochs for small segments
    for which the preconditions are previously known to be OK but would not
    be met on the small segments.
    """
    records = []
    global_interval = None
    transition_interval = None

    check_global = glob is not None
    check_trans = trans is not None
    # global criteria: record OK interval
    # transition criteria: record last available ep.t1 in which post-
    #  transition_criteria continue to hold using transition_criteria(ep)
    #  and find earliest available overlap between all of these intervals
    glt0 = np.Inf
    glt1 = np.Inf
    trt0 = np.Inf
    trt1 = np.Inf
    trans_found = False
    glob_found = False
    #print "***"
    if not check_global and not check_trans:
        # nothing to check for this criterion!
        return records, global_interval, transition_interval

    # Check preconditions
    if check_trans:
        if not trans.precondition(epochs) and not force_precond:
            records.append(common.args(time=np.Inf,
                         epoch=None,
                         kind='transitional_precondition',
                         interval=(None,None),
                         criterion_record=trans.record))
            return records, global_interval, transition_interval

    for epix, ep in enumerate(epochs):
        # Check global criteria if present
        if check_global and not glob(ep) and ep.t1 >= min_t:
            if ep.t0 < min_t:
                # failed on first epoch
                records.append(common.args(time = np.Inf,
                                     epoch = epix,
                                     kind = 'global',
                                     interval = (None, None),
                                     criterion_record = glob.record))
                return records, global_interval, transition_interval
            else:
                glt0 = min_t
                glt1 = ep.t0
                # record time was glt0 prior to Feb 2013, but this was inconsistent with
                # return values at end of function
                records.append(common.args(time = glt1,
                                           epoch = epix,
                                           kind = 'global',
                                           interval = (glt0, glt1),
                                           criterion_record = glob.record))
                glob_found = True
                #print "B", glt0, glt1
                break

        # Check transitional criteria if present
        if check_trans:
            if trans(ep):
                if not trans_found:
                    if ep.t0 < min_t:
                        if ep.t1 > min_t:
                            trans_found = True
                            trt0 = min_t
                            trt1 = ep.t1
                        # else ignore: transition not found anywhere within this epoch
                    else:
                        trt0 = ep.t0
                        trt1 = ep.t1
                        trans_found = True
                else:
                    trt0 = max([min([trt0, ep.t0]), min_t])
                trt1 = max([ep.t1, min_t])
                #print "C", trt0, trt1
            elif trans_found:
                # may never be reached if end of epochs reached while
                # transition_criteria(ep) holds
                break  # for epochs loop

    if trans_found:
        records.append(common.args(time = trt0,
                                   epoch = epix,
                                   kind = 'transitional',
                                   interval = (trt0, trt1),
                                   criterion_record = trans.record))
    # the last nands might be redundant
    if check_global and glob_found and not (np.isinf(glt0) and np.isinf(glt1)):
        # glt1 can be infinite provided glt0 is finite
        # glt0 can be -inf provided glt1 is finite
        global_interval = Interval('glob', float, [glt0, glt1])
    if check_trans and trans_found and not (np.isinf(trt0) and np.isinf(trt1)):
        # trt1 can be infinite provided trt0 is finite
        # trt0 can be -inf provided trt1 is finite
        transition_interval = Interval('tran', float, [trt0, trt1])
    return records, global_interval, transition_interval



def find_regime_transition(criteria_list, min_tstart=-np.Inf,
                           force_precond=False):
    """Use force_precond (optional, default=False) to force any preconditions
    to return True during test_criterion call.

    This is useful when recalculating epochs for small segments
    for which the preconditions are previously known to be OK but would not
    be met on the small segments.
    """

    done = False
    global_intervals = {}  # by criterion
    transition_intervals = {} # by criterion
    all_records = {}   # by criterion: records are just for user-readable audit trail
    trans_earliest_t0_ixs = [None]*len(criteria_list)
    trans_earliest_t0s = [min_tstart]*len(criteria_list)
    glob_earliest_t0_ixs = [None]*len(criteria_list)
    glob_earliest_t0s = [min_tstart]*len(criteria_list)
    crit_ixs = list(range(len(criteria_list)))
    re_run = crit_ixs[:]  # initial copy
    re_run_glob = []
    re_run_trans = []
    tstart_trans = tstart_glob = min_tstart

    while not done:
        # look at different sets of criteria in turn, e.g. based on Psis vs Omegas
        for criterion_ix, (epochs, global_criteria, transition_criteria) in \
                                                      enumerate(criteria_list):

            if criterion_ix not in re_run:
                continue

            # for re-runs, restart at next epoch after previous interval found
            #
            if criterion_ix in re_run_trans:
                try:
                    tstart_trans = transition_intervals[criterion_ix][trans_earliest_t0_ixs[criterion_ix]][1]
                except (IndexError, KeyError):
                    tstart_trans = np.Inf
            elif len(re_run_trans+re_run_glob)>0:
                # not the first time through this for loop
                tstart_trans = np.Inf

            if criterion_ix in re_run_glob:
                try:
                    tstart_glob = global_intervals[criterion_ix][glob_earliest_t0_ixs[criterion_ix]][1]
                except (IndexError, KeyError):
                    tstart_glob = np.Inf
            elif len(re_run_trans+re_run_glob)>0:
                # not the first time through this for loop
                tstart_glob = np.Inf

            tstart = min(tstart_trans, tstart_glob)
            if tstart >= epochs[-1].t1:
                # no epochs left to search. matching failed
                done = True
                break

            records, global_interval, transition_interval = \
                test_criterion(epochs, global_criteria, transition_criteria, tstart,
                               force_precond=force_precond)

            if criterion_ix in all_records:
                if records[-1] == all_records[criterion_ix][-1]:
                    # identical new record, we'll be in an infinite loop
                    done = True
                    break
                else:
                    all_records[criterion_ix].extend(records)
            else:
                all_records[criterion_ix] = records

            if transition_interval is not None:
                if criterion_ix in transition_intervals:
                    transition_intervals[criterion_ix].append(transition_interval)
                    # find max interval t0
                    trans_earliest_t0_ixs[criterion_ix] = np.argmax([interval[0] for interval in \
                                                                 transition_intervals[criterion_ix]])
                    trans_earliest_t0s[criterion_ix] = transition_intervals[criterion_ix][trans_earliest_t0_ixs[criterion_ix]][0]
                else:
                    transition_intervals[criterion_ix] = [transition_interval]
                    trans_earliest_t0s[criterion_ix] = transition_interval[0]
                    trans_earliest_t0_ixs[criterion_ix] = 0


            if global_interval is not None:
                if criterion_ix in global_intervals:
                    global_intervals[criterion_ix].append(global_interval)
                    # find max interval t0
                    glob_earliest_t0_ixs[criterion_ix] = np.argmax([interval[0] for interval in \
                                                                 global_intervals[criterion_ix]])
                    glob_earliest_t0s[criterion_ix] = global_intervals[criterion_ix][glob_earliest_t0_ixs[criterion_ix]][0]
                else:
                    global_intervals[criterion_ix] = [global_interval]
                    glob_earliest_t0s[criterion_ix] = global_interval[0]
                    glob_earliest_t0_ixs[criterion_ix] = 0

        match_global = global_criteria is None
        # check across all criteria to see if there's a common global interval
        match_trans = transition_criteria is None
        # check across all criteria to see if there's a common transition interval
        done = match_trans and match_global
        if done:
            # nothing to do
            continue

        # find earliest possible match time for trans and global kinds
        # i.e. the latest interval t0 across criteria for each kind

        if match_trans:
            re_run_trans = []
            tri0 = Interval('tran_result', float, [-np.Inf, np.Inf])
            for crit_ix in range(len(criteria_list)):
                # for each criterion, find overlap of their intervals
                try:
                    tri0 = tri0.intersect(transition_intervals[crit_ix][trans_earliest_t0_ixs[crit_ix]])
                except:
                    # no elements, pass
                    break
        else:
            trans_earliest_crit_ix = np.argmax(trans_earliest_t0s)
            trans_earliest_t0 = trans_earliest_t0s[trans_earliest_crit_ix]  # may be Inf
            if np.isfinite(trans_earliest_t0):
                tri0 = Interval('tran_result', float, [-np.Inf, np.Inf])
                for crit_ix in range(len(criteria_list)):
                    # for each criterion, find overlap of their intervals
                    try:
                        tri0 = tri0.intersect(transition_intervals[crit_ix][trans_earliest_t0_ixs[crit_ix]])
                    except:
                        pass
                        #if tri0 is None:
                            #print "No intersection for transition intervals"
                            #return np.Inf, record
                        #else:
                        #    raise
                if tri0 is None:
                    # for all criteria that returned earlier intervals, re-run to see if
                    # they have later intervals that might match using min_t = earliest_t0
                    re_run_trans = utils.remain(crit_ixs, [trans_earliest_crit_ix])
                else:
                    match_trans = True
            else:
                # no match of common intervals
                pass

        if match_global:
            re_run_glob = []
            gli0 = Interval('glob_result', float, [-np.Inf, np.Inf])
            for crit_ix in range(len(criteria_list)):
                # for each criterion, find overlap of their intervals
                try:
                    gli0 = gli0.intersect(global_intervals[crit_ix][glob_earliest_t0_ixs[crit_ix]])
                except:
                    # no elements, pass
                    pass
        else:
            glob_earliest_crit_ix = np.argmax(glob_earliest_t0s)
            glob_earliest_t0 = glob_earliest_t0s[glob_earliest_crit_ix]  # may be Inf
            if np.isfinite(glob_earliest_t0):
                gli0 = Interval('glob_result', float, [-np.Inf, np.Inf])
                for crit_ix in range(len(criteria_list)):
                    # for each criterion, find overlap of their intervals
                    try:
                        gli0 = gli0.intersect(global_intervals[crit_ix][glob_earliest_t0_ixs[crit_ix]])
                    except:
                        raise
                        #if gli0 is None:
                            #print "No intersection for global intervals"
                            #return np.Inf, record
                        #else:
                        #    raise
                if gli0 is None:
                    # for all criteria that returned earlier intervals, re-run to see if
                    # they have later intervals that might match using min_t = earliest_t0
                    re_run_glob = utils.remain(crit_ixs, [glob_earliest_crit_ix])
                else:
                    match_global = True
            else:
                # no match of common intervals
                pass

        re_run = re_run_glob[:]
        re_run.extend(utils.remain(re_run_trans, re_run_glob))

        done = (match_trans and match_global) or re_run == []

    # commented code below is redundant with the if statement lower down
##    if done and not (match_trans or match_global):
##        # if matching failed and break statement reached
##        return np.Inf, all_records

    if match_trans:
        if match_global:
            if gli0[1] >= tri0[0]:
                return tri0[0], all_records
            else:
                return gli0[1], all_records
        else:
            return tri0[0], all_records
    else:
        if match_global:
            return gli0[1], all_records
        else:
            return np.Inf, all_records




class epoch(object):
    def __init__(self, focus_var, sigma, gamma, actives, modulatory, inactives,
                 order1, slow, fast, ever_slow, ever_fast, ranks, index0, index1,
                 influences, rel_taus, traj_pts, inv_psi_namemap, inv_tau_namemap,
                 relative_ratios=False, opts=None):
        # defined by a seq of active variables (a domscales object) for a given
        # focused variable, according to some influence threshold sigma.
        # original influence values are influences pointset.
        self.focus_var = focus_var
        self.actives = actives
        #self.actives.sort()
        self.modulatory = modulatory
        #self.modulatory.sort()
        self.inactives = inactives
        self.inactives.sort()
        self.ranks = ranks
        self.order1 = order1
        self.slow = slow
        self.fast = fast
        self.ever_slow = ever_slow
        self.ever_fast = ever_fast
        self.traj_pts = traj_pts
        self.sigma = sigma
        self.gamma = gamma
        # these maps take active names and restore prefixes to use in
        # referencing Generator-created trajectory auxiliary variables
        self.inv_psi_namemap = inv_psi_namemap
        self.inv_tau_namemap = inv_tau_namemap
        self.relative_ratios = relative_ratios
        self.length = len(influences)
        self.index0 = index0
        self.index1 = index1
        assert self.length == index1-index0+1
        self.opts = opts
        if traj_pts is not None:
            self.t0 = traj_pts.indepvararray[0]
            self.t1 = traj_pts.indepvararray[-1]
        else:
            self.t0 = None
            self.t1 = None
        self.influences = influences
        # timescales relative to focus variable
        self.rel_taus = rel_taus
        self.index_interval = Interval('indices', int, [index0, index1])
        self.time_interval = Interval('time', float, [self.t0, self.t1])


    def __cmp__(self, other):
        if isinstance(other, epoch):
            if self.focus_var == other.focus_var and self.sigma == other.sigma:
                return self.actives == other.actives and \
                       self.modulatory == other.modulatory and \
                       self.order1 == other.order1 and \
                       self.fast == other.fast and \
                       self.slow == other.slow
        else:
            raise TypeError("Only compare epochs to other epochs")

    __eq__ = __cmp__

    def seq_str(self):
        actives_w_speed = []
        for a in self.actives:
            if a in self.fast:
                actives_w_speed.append(a+'[F]')
            elif a in self.slow:
                actives_w_speed.append(a+'[S]')
            else:
                actives_w_speed.append(a)
        mod_w_speed = []
        for m in self.modulatory:
            if m in self.fast:
                mod_w_speed.append(m+'[F]')
            elif m in self.slow:
                mod_w_speed.append(m+'[S]')
            else:
                mod_w_speed.append(m)
        return actives_w_speed, mod_w_speed

    def _infostr(self, verboselevel=0):
        if self.opts.use_order:
            uo_str = "(ord.)"
        else:
            uo_str = "(~ord.)"
        str1 = "Epoch for %s at sigma %.3f %s: [%.4f, %.4f]" % \
             (self.focus_var, self.sigma, uo_str, self.t0, self.t1)
        if verboselevel == 0:
            return str1
        else:
            actives_w_speed, mod_w_speed = self.seq_str()
            str2 = " Actives: " + ",".join(actives_w_speed)
            str3 = "   Modulatory: " + ",".join(mod_w_speed)
            return str1 + str2 + str3

    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))

    def __str__(self):
        return self._infostr(0)

    __repr__ = __str__


def _comp(a, b, eps, atol):
    try:
        rat = a/b
    except ZeroDivisionError:
        rat = np.Inf
    return np.allclose(rat, eps, atol=atol)


def transition_psi(epoch, pt, atol):
    """Assume that, generically, there will only be one transition at a time.
    """
    acts = pt[epoch.inv_psi_namemap(epoch.actives)]
    # ensures order is the same
    acts_plain = epoch.inv_psi_namemap.inverse()(acts.coordnames)
    if epoch.modulatory == []:
        mods = []
        mods_plain = []
    else:
        mods = pt[epoch.inv_psi_namemap(epoch.modulatory)]
        mods_plain = epoch.inv_psi_namemap.inverse()(mods.coordnames)
    relrat = epoch.relative_ratios
    leavers = []
    joiners = []
    if relrat:
        # leavers
        if len(acts) > 1:
            ptixs = np.argsort(acts)[::-1]
            for i in range(len(ptixs)-1):
                if _comp(acts[ptixs[i]], acts[ptixs[i+1]], epoch.sigma, atol):
                    leavers.append(acts_plain[i+1])
        # else: can't be any leavers (case caught by joiners)
        # joiners
        min_act = min(acts)
        for i, m in enumerate(mods):
            if _comp(min_act, m, epoch.sigma, atol):
                joiners.append(mods_plain[i])
    else:
        max_ix = np.argmax(acts)
        min_ix = np.argmin(acts)
        # leavers
        if _comp(acts[max_ix], acts[min_ix], epoch.sigma, atol):
            leavers.append(acts_plain[min_ix])
        # joiners
        for i, m in enumerate(mods):
            if _comp(acts[max_ix], m, epoch.sigma, atol):
                joiners.append(mods_plain[i])
    if len(leavers) + len(joiners) == 0:
        raise PyDSTool_ValueError("No transition found: Tolerance too small?")
    elif len(leavers) + len(joiners) > 1:
        raise PyDSTool_ValueError("Too many transitions found: Tolerance too large?")
    else:
        if len(leavers) == 1:
            return ('leave', leavers[0])
        else:
            return ('join', joiners[0])


def transition_tau(epoch, pt, atol):
    """Assume that, generically, there will only be one transition at a time.
    """
    if epoch.slow == []:
        slow = []
        slow_plain = []
    else:
        slow = pt[epoch.inv_tau_namemap(epoch.slow)]
        slow_plain = epoch.inv_tau_namemap.inverse()(slow.coordnames)
    if epoch.fast == []:
        fast = []
        fast_plain = []
    else:
        fast = pt[epoch.inv_tau_namemap(epoch.fast)]
        fast_plain = epoch.inv_tau_namemap.inverse()(fast.coordnames)
    if epoch.order1 == []:
        order1 = []
        order1_plain = []
    else:
        order1 = pt[epoch.inv_tau_namemap(epoch.order1)]
        order1_plain = epoch.inv_tau_namemap.inverse()(order1.coordnames)
    slow_leavers = []
    slow_joiners = []
    max_ix = np.argmax(slow)
    min_ix = np.argmin(slow)
    # slow leavers -> order1
    if _comp(slow[min_ix], 1, epoch.gamma, atol):
        slow_leavers.append(slow_plain[min_ix])
    # slow joiners <- order1
    for i, m in enumerate(order1):
        if _comp(m, 1, epoch.gamma, atol):
            slow_joiners.append(order1_plain[i])
    #
    max_ix = np.argmax(fast)
    min_ix = np.argmin(fast)
    # fast leavers -> order1
    fast_leavers = []
    fast_joiners = []
    if _comp(1, fast[min_ix], epoch.gamma, atol):
        fast_leavers.append(fast_plain[min_ix])
    # fast joiners <- order1
    for i, m in enumerate(order1):
        if _comp(1, m, epoch.gamma, atol):
            fast_joiners.append(order1_plain[i])
    len_slow_l = len(slow_leavers)
    len_slow_j = len(slow_joiners)
    len_fast_l = len(fast_leavers)
    len_fast_j = len(fast_joiners)
    if len_slow_l + len_slow_j + \
       len_fast_l + len_fast_j == 0:
        raise PyDSTool_ValueError("No transition found: Tolerance too small?")
    elif len_slow_l + len_slow_j + \
         len_fast_l + len_fast_j > 1:
        raise PyDSTool_ValueError("Too many transitions found: Tolerance too large?")
    else:
        if len_slow_l == 1:
            return ('slow_leave', slow_leavers[0])
        elif len_slow_j == 1:
            return ('slow_join', slow_joiners[0])
        elif len_fast_l == 1:
            return ('fast_leave', fast_leavers[0])
        else:
            return ('fast_join', fast_joiners[0])

def check_opts(opts):
    def_vals = {'use_order': True, 'speed_fussy': True}
    ok_keys = def_vals.keys()
    if opts is None:
        return common.args(**def_vals)
    else:
        for k in ok_keys:
            if k not in list(opts.keys()):
                opts[k] = def_vals[k]
        rem_keys = utils.remain(list(opts.keys()), ok_keys)
        if rem_keys != []:
            raise ValueError("Invalid options passed in opts argument: %s" % rem_keys)
        else:
            return opts


class domscales(object):
    def __init__(self, focus_var, traj_pts, influence_pts,
                 influence_type, tau_refs, psi_refs, opts=None):
        ## !!! Assumes Psi-type influences only
        self.focus_var = focus_var
        self.traj_pts = traj_pts
        self.tau_refs = tau_refs
        base = 'tau_'
        avoid_prefix = len(base)
        tau_namedict = {}
        for c in traj_pts.coordnames:
            if c[:avoid_prefix] == base:
                tau_namedict[c] = c[avoid_prefix:]
        self.tau_namemap = parseUtils.symbolMapClass(tau_namedict)
        self.psi_refs = psi_refs
        base = 'psi_'+focus_var+'_'
        avoid_prefix = len(base)
        psi_namedict = {}
        for c in influence_pts.coordnames:
            psi_namedict[c] = c[avoid_prefix:]
        self.psi_namemap = parseUtils.symbolMapClass(psi_namedict)
        influence_pts.mapNames(self.psi_namemap)
        self.influence_pts = influence_pts
        # rank largest first
        ranks = np.argsort(influence_pts.coordarray.T)[:,::-1]
        greatest_vals = np.array([p[ranks[i,0]] for i, p in \
                                  enumerate(influence_pts.coordarray.T)])
        self.ranks = ranks
        self.greatest_vals = greatest_vals
        # normalized at each time point (largest = 1)
        self.normed_coordarray = influence_pts.coordarray/greatest_vals
        self.influence_type = influence_type
        self.coordnames = influence_pts.coordnames
        self.num_pts = len(influence_pts)
        self.opts = check_opts(opts)


    def calc_epochs(self, sigma, gamma, relative_ratios=False, ignore_singleton=True):
        """Calculate epochs without cycle checking"""
        assert sigma > 1, "sigma parameter must be > 1"
        assert gamma > 1, "gamma parameter must be > 1"
        epochs = []
        old_actives = []
        old_fast = []
        old_slow = []
        # we just use this dictionary to store unique mod names
        # the values are irrelevant
        all_modulatory = {}
        ep_start = 0
        complete_epoch = False
        for i in range(self.num_pts):
            # ignore_change is used to avoid single time-point epochs:
            # it ensures 'old' actives and time scale sets don't get updated for
            # the single time-point change
            ignore_change = False
            ds_pt = self.get_domscales_point(i, relative_ratios)
            groups = split(ds_pt, sigma)
            actives = groups[0].coordnames
            if not self.opts.use_order:
                actives.sort()
            # gather all modulatory variables
            if len(groups) > 1:
                modulatory = groups[1].coordnames
                if not self.opts.use_order:
                    modulatory.sort()
            else:
                modulatory = []
            if self.opts.speed_fussy:
                tau = get_taus(self.traj_pts[i], self.tau_refs)
                tau_ref = tau[self.focus_var]
                rel_tau = tau/tau_ref
                slow_test = rel_tau/gamma > 1
                fast_test = rel_tau*gamma < 1
                cs = tau.coordnames
                slow = common.intersect([cs[ci] for ci in range(len(cs)) if slow_test[ci]], actives)
                fast = common.intersect([cs[ci] for ci in range(len(cs)) if fast_test[ci]], actives)
                if i == 0:
                    if self.num_pts == 1:
                        complete_epoch = True
                        act_vars = actives
                        mod_vars = modulatory
                        i_stop = 1
                elif not (len(old_fast) == len(fast) and utils.remain(old_fast, fast) == [] and \
                          len(old_slow) == len(slow) and utils.remain(old_slow, slow) == []):
                    # non-identical sets
                    if i - ep_start > 1:
                        complete_epoch = True
                    else:
                        # ignore epoch made up of single time point
                        ignore_change = True
                    act_vars = old_actives
                    mod_vars = list(all_modulatory.keys())
                    i_stop = i
                if ignore_change:
                    old_fast = fast
                    old_slow = slow
                else:
                    old_fast = common.makeSeqUnique(utils.union(old_fast,fast))
                    old_slow = common.makeSeqUnique(utils.union(old_slow,slow))
            if old_actives == actives and i < self.num_pts-1:
                all_modulatory.update(dict(zip(modulatory,enumerate(modulatory))))
            else:
                # epoch changed or we reached the last point
                if old_actives == [] and self.num_pts > 1:
                    # first epoch: don't make an epoch object until it ends
                    all_modulatory.update(dict(zip(modulatory,enumerate(modulatory))))
                elif ignore_singleton and i - ep_start == 1 and i < self.num_pts-1:
                    # ignore epoch made up of single time point
                    if common.intersect(modulatory, old_actives) != []:
                        # clash -- cannot be in two places at once: a value was transitional
                        # so put it in with the current point, assuming that this trend will
                        # continue
                        #raise ValueError("Reduce step size! Single time-point epochs are not allowed")
                        old_actives = utils.remain(old_actives, common.intersect(modulatory, old_actives))
                    all_modulatory.update(dict(zip(modulatory,enumerate(modulatory))))
                    ignore_change = True
                else:
                    if self.num_pts == 1:
                        act_vars = actives
                        mod_vars = modulatory
                        i_stop = 1
                    else:
                        act_vars = old_actives
                        mod_vars = list(all_modulatory.keys())
                        i_stop = i
                    complete_epoch = True
                if ignore_change:
                    old_actives = common.makeSeqUnique(utils.union(old_actives, actives))
                else:
                    # update old_actives now that it has changes
                    old_actives = actives
            if complete_epoch:
                # reset
                complete_epoch = False
                influences = self.influence_pts[ep_start:i_stop]
                avg_influence = {}
                for a in act_vars:
                    avg_influence[a] = np.mean(influences[a])
                # get approximate order
                act_vars = common.sortedDictLists(avg_influence, reverse=True)[0]
                avg_influence = {}
                not_mod = []
                for m in mod_vars:
                    avg = np.mean(influences[m])
                    if avg > 0:
                        avg_influence[m] = avg
                    else:
                        # influence actually zero so actually should be in
                        # inactive set
                        not_mod.append(m)
                mod_vars = utils.remain(mod_vars, not_mod)
                # get approximate order
                mod_vars = common.sortedDictLists(avg_influence,
                                                  reverse=True)[0]
                inact_vars = utils.remain(self.coordnames,
                                           act_vars+mod_vars)
                pts = self.traj_pts[ep_start:i_stop]
                order1, slow, fast, ever_slow, ever_fast, rel_taus = \
                      self.calc_fast_slow(get_taus(pts, self.tau_refs),
                                          gamma)
                rel_taus.mapNames(self.tau_namemap)
                epochs.append(epoch(self.focus_var, sigma, gamma,
                                    act_vars, mod_vars, inact_vars,
                                    order1, slow, fast, ever_slow, ever_fast,
                                    self.ranks[ep_start:i_stop,:],
                                    ep_start, i_stop-1,
                                    self.influence_pts[ep_start:i_stop], rel_taus,
                                    pts, relative_ratios=ds_pt.relative_ratios,
                                    inv_psi_namemap=self.psi_namemap.inverse(),
                                    inv_tau_namemap=self.tau_namemap.inverse(),
                                    opts=self.opts))
                ep_start = i  # doesn't need to be updated for num_pts == 1
                all_modulatory = {}
        self.epochs = epochs


    def calc_fast_slow(self, taus, gamma):
        """Determines variables which are always fast and slow, and those
        which are ever fast and slow, over a given set of time scale points.
        """
        focus_taus = taus[self.focus_var]
        rel_taus = taus/focus_taus
        cs = taus.coordnames
        ixs = list(range(len(cs)))
        n = list(range(len(taus)))
        # these tests can never select focus_var which has
        # been normalized to tau = 1
        slow_test = rel_taus/gamma > 1
        fast_test = rel_taus*gamma < 1
        slow_array = [[i for i in ixs if slow_test[i,j]] for j in n]
        fast_array = [[i for i in ixs if fast_test[i,j]] for j in n]
        slow_ixs = ever_slow_ixs = slow_array[0]
        for row in slow_array[1:]:
            slow_ixs = utils.intersect(slow_ixs, row)
            ever_slow_ixs = np.unique(utils.union(ever_slow_ixs, row))
        slow_vars = [cs[i] for i in slow_ixs]
        ever_slow_vars = [cs[i] for i in ever_slow_ixs]
        fast_ixs = ever_fast_ixs = fast_array[0]
        for row in fast_array[1:]:
            fast_ixs = utils.intersect(fast_ixs, row)
            ever_fast_ixs = np.unique(utils.union(ever_fast_ixs, row))
        fast_vars = [cs[i] for i in fast_ixs]
        ever_fast_vars = [cs[i] for i in ever_fast_ixs]
        order1_vars = utils.remain(cs, slow_vars+fast_vars)
        return order1_vars, slow_vars, fast_vars, \
               list(ever_slow_vars), list(ever_fast_vars), rel_taus


    def lookup_index(self, ix):
        """Locates index ix in the epochs, returning the epoch index,
        epoch object, and the index within the epoch's pointset data."""
        assert self.epochs
        for epix, ep in enumerate(self.epochs):
            if ix >= ep.index0 and ix <= ep.index1:
                return epix, ep, ix - ep.index0

    def lookup_time(self, t):
        """Locates time t in the epochs, returning the epoch index,
        epoch object, and the approximate corresponding index within
        the epoch's pointset data.
        Note, because of the time discretization, time is compared up to
        the next epoch's start time."""
        assert self.epochs
        for epix, ep in enumerate(self.epochs):
            try:
                next_time = self.epochs[epix+1].t0
                comp = operator.lt
            except IndexError:
                next_time = ep.t1
                comp = operator.le
            if t >= ep.t0 and comp(t, next_time):
                ix = ep.influences.find(t, end=0)
                return epix, ep, ix


    def calc_regimes(self, options=None):
        assert self.epochs
        raise NotImplementedError


    def get_domscales_point(self, i, relative_ratios):
        ds_pt = common.args(relative_ratios=relative_ratios)
        ds_pt.coordnames = list(np.array(self.coordnames)[self.ranks[i]])
        # influences are in rank order (largest first)
        ds_pt.influences = list(self.influence_pts.coordarray[self.ranks[i],i])
        # diameter measures ratio of largest to smallest influence
        try:
            ds_pt.diameter = ds_pt.influences[0]/ds_pt.influences[-1]
        except ZeroDivisionError:
            ds_pt.diameter = np.inf
        ds_pt.relative_influences = list(self.normed_coordarray[self.ranks[i],i])
        if relative_ratios:
            as_ratios = as_relative_ratios
        else:
            as_ratios = as_absolute_ratios
        ds_pt.as_ratios = as_ratios(ds_pt.influences)
        return ds_pt


def define_psi_events(acts_all, mods_all, focus_var, ignore_transitions=None):
    if ignore_transitions is None:
        acts_leave = acts_all
        mods_join = mods_all
    else:
        remove_acts = []
        remove_mods = []
        for trans, var in ignore_transitions:
            if trans == 'leave':
                remove_acts.append(var)
            elif trans == 'join':
                remove_mods.append(var)
            else:
                raise ValueError("Invalid transition type: %s" % trans)
        acts_leave = [a for a in acts_all if a not in remove_acts]
        mods_join = [m for m in mods_all if m not in remove_mods]
    act_coordlist = ['psi_'+focus_var+'_'+c for c in acts_all]
    act_leave_coordlist = ['psi_'+focus_var+'_'+c for c in acts_leave]
    mod_coordlist = ['psi_'+focus_var+'_'+c for c in mods_join]
    evdefs = {}
    if len(act_leave_coordlist) > 0 and len(act_coordlist) > 0:
        if len(act_coordlist) == 1:
            term1 = '(' + str(act_coordlist[0]) + ')'
        else:
            term1 = "max( [ %s ] )" % ",".join(act_coordlist)
        if len(act_leave_coordlist) == 1:
            term2 = '(' + str(act_leave_coordlist[0]) + ')'
        else:
            term2 = "min( [ %s ] )" % ",".join(act_leave_coordlist)
        act_leave_def = "%s / %s - dssrt_sigma" % (term1, term2)
        evdefs['act_leave_ev'] = common.args(defn=act_leave_def, dirn=1, pars=['dssrt_sigma'])
    if len(act_coordlist) > 0 and len(mod_coordlist) > 0:
        if len(act_coordlist) == 1:
            term1 = '(' + str(act_coordlist[0]) + ')'
        else:
            term1 = "max( [ %s ] )" % ",".join(act_coordlist)
        if len(mod_coordlist) == 1:
            term2 = '(' + str(mod_coordlist[0]) + ')'
        else:
            term2 = "max( [ %s ] )" % ",".join(mod_coordlist)
        mod_join_def = "%s / %s - dssrt_sigma" % (term1, term2)
        evdefs['mod_join_ev'] = common.args(defn=mod_join_def, dirn=-1, pars=['dssrt_sigma'])
    return evdefs


def define_tau_events(slow, fast, order1, ref):
    ref_name = 'tau_'+ref
    assert ref in order1, "reference time scale must be in order1 list"
    evdefs = {}
    if slow != []:
        slow_list = ", ".join(['tau_'+c for c in slow])
        if len(slow) == 1:
            slow_leave_def = "(%s)/(%s) - dssrt_gamma" % (slow_list[0], ref_name)
        else:
            slow_leave_def = "min([ %s ])/(%s) - dssrt_gamma" % (slow_list, ref_name)
        evdefs['slow_leave_ev'] = common.args(defn=slow_leave_def, dirn=-1, pars=['dssrt_gamma'])
    if fast != []:
        fast_list = ", ".join(['tau_'+c for c in fast])
        if len(fast) == 1:
            fast_leave_def = "(%s)/(%s) - 1./dssrt_gamma" % (fast_list[0], ref_name)
        else:
            fast_leave_def = "max([ %s ])/(%s) - 1./dssrt_gamma" % (fast_list, ref_name)
        evdefs['fast_leave_ev'] = common.args(defn=fast_leave_def, dirn=1, pars=['dssrt_gamma'])
    other_o1 = utils.remain(order1, ref)
    if other_o1 != []:
        o1_list = ", ".join(['tau_'+c for c in other_o1])
        if len(other_o1) == 1:
            slow_join_def = "(%s)/(%s) - dssrt_gamma" % (o1_list[0], ref_name)
            fast_join_def = "(%s)/(%s) - 1./dssrt_gamma" % (o1_list[0], ref_name)
        else:
            slow_join_def = "max([ %s ])/(%s) - dssrt_gamma" % (o1_list, ref_name)
            fast_join_def = "min([ %s ])/(%s) - 1./dssrt_gamma" % (o1_list, ref_name)
        evdefs['slow_join_ev'] = common.args(defn=slow_join_def, dirn=1, pars=['dssrt_gamma'])
        evdefs['fast_join_ev'] = common.args(defn=fast_join_def, dirn=-1, pars=['dssrt_gamma'])
    return evdefs


class dssrt_assistant(object):
    def __init__(self, kw):
        model = kw['model']
        if isinstance(model, HybridModel):
            # check that there is a single sub-model (not a true hybrid system)
            assert len(model.registry) == 1
            self.model = list(model.registry.values())[0]
            self.gen = list(self.model.registry.values())[0]
        elif isinstance(model, NonHybridModel):
            assert len(model.registry) == 1
            self.model = model
            self.gen = list(self.model.registry.values())[0]
        else:
            # assume generator
            self.model = None
            self.gen = model
        FScompatMap = self.gen._FScompatibleNames
        self.reset()
        self.gamma1 = {}
        self.gamma2 = {}
        self.pars = self.gen.pars #FScompatMap(self.gen.pars)
        if self.model is None or self.model._mspecdict is None:
            # dict of each variable's inputs (FScompatMap will copy)
            inputs = kw['inputs'] #FScompatMap(kw['inputs'])
            # dummy input names such as Lk for leak channel, which
            # has no activation variable, will not be present in
            # the FScompatMap, so use this opportunity to update
            # the mapping
            for k, gam_inps in inputs.items():
                gamma1 = []
                try:
                    for i in gam_inps.gamma1:
                        if '.' in i and i not in FScompatMap:
                            i_FSc = i.replace('.','_')
                            FScompatMap[i] = i_FSc
                        #    gamma1.append(i_FSc)
                        #else:
                        gamma1.append(i)
                except AttributeError:
                    pass
                self.gamma1[k] = gamma1
                gamma2 = []
                try:
                    for i in gam_inps.gamma2:
                        if '.' in i and i not in FScompatMap:
                            i_FSc = i.replace('.','_')
                            FScompatMap[i] = i_FSc
                        #    gamma2.append(i_FSc)
                        #else:
                        gamma2.append(i)
                except AttributeError:
                    pass
                self.gamma2[k] = gamma2
            # all_vars includes 'mock' variables
            # given by auxiliary functions
            self.all_vars = list(inputs.keys())
        else:
            # deduce them from the dependencies in the
            # ModelSpec dict
            #self.inputs = ?
            # acquire auxiliary fn names
            #self.aux_fns = ?
            # acquire types of inputs (gamma sets)
            #self.gamma1 = ?
            #self.gamma2 = ?
            # Can expect only mspec
            self._init_from_MSpec(list(self.model._mspecdict.values())[0]['modelspec'])
        all_inputs = []
        for ins in chain(self.gamma1.values(), self.gamma2.values()):
            try:
                all_inputs.extend(ins)
            except TypeError:
                # ins is None
                pass
        self._FScompatMap = FScompatMap
        self._FScompatMapInv = FScompatMap.inverse()
        self.all_inputs = np.unique(all_inputs).tolist()
        self.tau_refs = kw['taus']
        self.inf_refs = kw['infs']
        self.psi_refs = kw['psis']
        opts = common.args()
        try:
            opts.use_order = kw['use_order']
        except KeyError:
            pass
        try:
            opts.speed_fussy = kw['speed_fussy']
        except KeyError:
            pass
        self.opts = opts
        self._setup_taus_psis()

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['taus']
        del d['taus_direct']
        del d['infs']
        del d['infs_direct']
        del d['psis']
        del d['psis_direct']
        return d

    def __setstate__(self):
        self.__dict__.update(state)
        self._setup_taus_psis()

    def _setup_taus_psis(self):
        # process these to extract whether these are parameters, expressions
        # or references to auxiliary functions and convert them to callable
        # objects
        #
        # also, every reference to an auxiliary variable must be mappable to
        # a dssrt_fn_X aux function to allow direct calculating of a tau or
        # Psi from only state variables in a Point.
        # ... this needs a parallel set of definitions to be created
        self._direct_fail = False
        d = {}
        fspecs = self.gen.funcspec._auxfnspecs
        fn_base = 'dssrt_fn_'
        for var, tau in self.tau_refs.items():
            if tau is None:
                continue
            # signature for the aux fn is in the Generator's FuncSpec, e.g.
            # 'Na_dssrt_fn_cond': (['m', 'h'], 'Na_g*m*m*m*h')
            # 'Na_dssrt_fn_tauh': (['V'], '1/(0.128*exp(-(50.0+V)/18)+4/(1+exp(-(V+27.0)/5)))')
            # for which prefix is 'Na'
            v_hier_list = var.split('.')
            prefix = '_'.join(v_hier_list[:-1])  # empty if no hierarchical name
            var_postfix = v_hier_list[-1]
            if len(prefix) > 0:
                fname = prefix+'_'+fn_base+'tau'+var_postfix
            else:
                fname = fn_base+'tau'+var_postfix
            # convert fn sig names to resolvable hierarchical names using knowledge
            # of inputs
            inps_to_var = self.gamma1[var] + self.gamma2[var]
            try:
                sig = fspecs[fname][0]
            except KeyError:
                # dssrt_ functions were not defined
                self._direct_fail = True
                break
            # match sig entries to final elements of inps_to_var
            for inp in inps_to_var:
                inp_hier_list = inp.split('.')
                try:
                    ix = sig.index(inp_hier_list[-1])
                except ValueError:
                    # postfix not in sig
                    pass
                else:
                    sig[ix] = inp
            d[tau.replace('.','_')] = fname+'(' + ",".join(sig) + ')'

        for var, inf in self.inf_refs.items():
            if inf is None:
                continue
            # signature for the aux fn is in the Generator's FuncSpec, e.g.
            # 'Na_dssrt_fn_cond': (['m', 'h'], 'Na_g*m*m*m*h')
            # 'Na_dssrt_fn_tauh': (['V'], '1/(0.128*exp(-(50.0+V)/18)+4/(1+exp(-(V+27.0)/5)))')
            # for which prefix is 'Na'
            v_hier_list = var.split('.')
            prefix = '_'.join(v_hier_list[:-1])  # empty if no hierarchical name
            var_postfix = v_hier_list[-1]
            if len(prefix) > 0:
                fname = prefix+'_'+fn_base+var_postfix+'inf'
            else:
                fname = fn_base+var_postfix+'inf'
            # convert fn sig names to resolvable hierarchical names using knowledge
            # of inputs
            inps_to_var = self.gamma1[var] + self.gamma2[var]
            try:
                sig = fspecs[fname][0]
            except KeyError:
                # dssrt_ functions were not defined
                self._direct_fail = True
                break
            # match sig entries to final elements of inps_to_var
            for inp in inps_to_var:
                inp_hier_list = inp.split('.')
                try:
                    ix = sig.index(inp_hier_list[-1])
                except ValueError:
                    # postfix not in sig
                    pass
                else:
                    sig[ix] = inp
            d[inf.replace('.','_')] = fname+'(' + ",".join(sig) + ')'

        self._direct_ref_mapping = parseUtils.symbolMapClass(d)
        FScMap = self._FScompatMap
        self.taus = {}
        self.taus_direct = {}
        for k, v in self.tau_refs.items():
            self.taus[k], self.taus_direct[k] = self._process_expr(FScMap(v))

        self.infs = {}
        self.infs_direct = {}
        for k, v in self.inf_refs.items():
            self.infs[k], self.infs_direct[k] = self._process_expr(FScMap(v))

        self.psis = {}
        self.psis_direct = {}
        for k, vdict in self.psi_refs.items():
            if vdict is None:
                self.psis[k], self.psis_direct[k] = (None, None)
            else:
                self.psis[k] = {}
                self.psis_direct[k] = {}
                for inp, psiv in vdict.items():
                    self.psis[k][inp], self.psis_direct[k][inp] = \
                                            self._process_expr(FScMap(psiv))


    def _init_from_MSpec(self, mspec):
        ## !!!
        # get Psi formulae by differentiating x_inf directly
        #deriv_list = [Symbolic.Diff(,i) for i in all_var_inputs]
        raise NotImplementedError


    def _process_expr(self, expr):
        """Create two Symbolic.Fun(ction) versions of the given expression:

        (1) The first is intended for processing post-integration
        pointsets to yield the dominant scale Psi quantities,
        when the pointsets contain already-computed auxiliary variables
        for taus, infs, etc.

        (2) The second is intended for on-demand calculation of Psi's
        when only the system's state variables are given, thus
        taus and infs etc. must be computed this class. If user did not
        provide the dssrt_fn functions this will be ignored and not created.

        The returned pair of pairs is
         ((arg_list_signature (1), symbolic function (1)),
          (arg_list_signature (2), symbolic function (2)))
        unless None was passed (in which case None is returned).
        """
        if expr is None:
            # None
            return (None, None)
        elif parseUtils.isNumericToken(expr):
            # numeric literal
            f = Symbolic.expr2fun(expr)
            return ( ([], f), ([], f) )
        elif expr in self.pars:
            # parameter
            val = 'refpars["%s"]' % expr
            defs = {'refpars': self.pars}
            f = Symbolic.expr2fun(val, **defs)
            return ( ([], f), ([], f) )
        elif expr in self.gen.inputs:
            # input literal - turn into a function of t
            f = Symbolic.expr2fun(expr+'(t)')
            return ( (['t'], f), (['t'], f) )
        else:
            # test for valid math expression
            # - may include auxiliary function call (with arguments)
            fnames = list(self.gen.auxfns.keys())
            themap = dict(self.gen.auxfns.items())
            themap.update(dict(self.gen.inputs.items()))
            defs = {'refpars': self.pars}
            pnames = list(self.pars.keys())
            pvals = ['refpars["%s"]' % pn for pn in pnames]
#            parmap = dict(zip(pnames,
#                              [Symbolic.expr2fun(v, **defs) for v in pvals]))
            #val, dummies = parseUtils.replaceCallsWithDummies(expr, fnames)
            val = expr   # retained in case revert to replaceCallsWithDummies(...)
            #dummies = {}
            val_q = Symbolic.QuantSpec('__expr__', val)
            mapping = dict(zip(pnames, pvals))
            used_inputs = utils.intersect(val_q.parser.tokenized,
                               list(self.gen.inputs.keys()))
            for inp in used_inputs:
                # turn into a function of t
                mapping[inp] = inp + '(t)'
            val_q.mapNames(mapping)
            #map.update(parmap)
            # Don't update with self.pars otherwise param values are hard-wired
#            map.update(self.pars)
#            for num, dummy in dummies.items():
#                dstr = '__dummy%i__' % num
#                dspec = Symbolic.QuantSpec('__temp__', dummy)
#                defs[dstr] = Symbolic.expr2fun(dspec, **map)
            #defs.update(parmap)
            #defs.update(self.pars)

            # Every reference to an auxiliary variable must be mappable to
            # a dssrt_fn_X aux function to allow direct calculating of a tau or
            # Psi from only state variables in a Point.
            if not self._direct_fail:
                val_q_direct = copy.copy(val_q)
                val_q_direct.mapNames(self._direct_ref_mapping)

            defs.update(themap)
            if val in defs:
                # then outermost level is just a single function call
                # which has been replaced unnecessarily by a dummy term
                fn = copy.copy(defs[val])
                del defs[val]
                return ( (fn._args, fn), (fn._args, fn) )
            else:
                fn = Symbolic.expr2fun(str(val_q), **defs)
                if not self._direct_fail:
                    fn_direct = Symbolic.expr2fun(str(val_q_direct), **defs)
                    return ( (fn._args, fn), (fn_direct._args, fn_direct) )
                else:
                    return ( (fn._args, fn), (None, None) )


    def reset(self):
        """Delete current state based on given trajectory"""
        self.traj = None
        self.pts = None
        self.times = None
        self.focus_var = None
        self.tau_inf_vals = {}
        self.psi_vals = {}
        self.Is = {}
        self.pars = None


    def ensure(self):
        assert self.gen is not None, "Must have a valid generator attribute"
        assert self.traj is not None, "Must provide trajectory attribute"
        assert self.focus_var is not None, "Must provide focus_var attribute"
        if self.pts is None:
            self.pts = self.traj.sample()
        if self.times is None:
            self.times = self.pts.indepvararray
        # refresh reference from generator
        self.pars = self.gen.pars


    def calc_tau(self, target, pt):
        """Calculate tau of target variable directly from a given single point
        containing only the state variables.
        """
        assert not self._direct_fail, "Option not available"
        ptFS = self._FScompatMap(pt)
        try:
            args, f = self.taus_direct[target]
        except TypeError:
            # entry in dict was None (can't broadcast)
            return np.NaN
        else:
            try:
                return f(**common.filteredDict(ptFS, args))
            except:
                print(f._args)
                print(f._call_spec)
                raise


    def calc_inf(self, target, pt):
        """Calculate inf of target variable directly from a given single point
        containing only the state variables.
        """
        assert not self._direct_fail, "Option not available"
        ptFS = self._FScompatMap(pt)
        try:
            args, f = self.infs_direct[target]
        except TypeError:
            # entry in dict was None (can't broadcast)
            return np.NaN
        else:
            try:
                return f(**common.filteredDict(ptFS, args))
            except:
                print(f._args)
                print(f._call_spec)
                raise


    def calc_psi(self, target, pt, focus=None):
        """Calculate Psi of focus variable w.r.t. the target variable directly
        from a given single point containing only the state variables.

        If focus is None (default) then focus_var attribute is assumed.
        """
        assert not self._direct_fail, "Option not available"
        ptFS = self._FScompatMap(pt)
        if focus is None:
            focus = self.focus_var
        try:
            args, f = self.psis_direct[focus][target]
        except TypeError:
            # entry in dict was None (can't broadcast)
            return np.NaN
        except KeyError:
            raise ValueError("Invalid focus or target variable")
        else:
            try:
                return f(**common.filteredDict(ptFS, args))
            except:
                print(f._args)
                print(f._call_spec)
                raise


    def calc_taus_infs(self, focus=False):
        """Calculate taus and infs from given trajectory.

        If focus=True (default False), results are only returned for the
        variable of focus given by focus_var attribute"""
        self.ensure()
        vals = {}
        if focus:
            vars = [self.focus_var]
        else:
            vars = list(self.taus.keys())
        pts = self._FScompatMap(self.pts)
#        ptvals = dict(self.pts)
#        # permit access from external inputs or other autonomous functions
#        ptvals['_time_'] = pts['t']
#        pts = Pointset(coorddict=ptvals, indepvararray=pts['t'])
        for k in vars:
            tau_info = self.taus[k]
            if tau_info is None:
                vals['tau_'+k] = np.array([np.NaN for pt in pts])
            else:
                args, f = tau_info
                try:
                    vals['tau_'+k] = np.array([f(**common.filteredDict(pt, args)) \
                                               for pt in pts])
                except:
                    print(f._args)
                    print(f._call_spec)
                    raise
            inf_info = self.infs[k]
            if inf_info is None:
                vals['inf_'+k] = np.array([np.NaN for pt in pts])
            else:
                args, f = inf_info
                try:
                    vals['inf_'+k] = np.array([f(**common.filteredDict(pt, args)) \
                                               for pt in pts])
                except:
                    print(f._args)
                    print(f._call_spec)
                    raise
        # we know that taus and infs have same set of keys -- the variable names
        self.tau_inf_vals.update(vals)


    def calc_Is(self):
        """Calculate currents from given trajectory"""
        raise NotImplementedError


    def calc_psis(self):
        """Calculate Psis from given trajectory"""
        self.ensure()
        FScMap = self._FScompatMap
        fv = self.focus_var
        if self.tau_inf_vals == {}:
            self.calc_taus_infs(focus=True)
        psi_defs = self.psis[fv]
        if psi_defs is None:
            self.psi_vals[fv] = None
            return
        psis = {}
        all_var_inputs = self.gamma1[fv] + self.gamma2[fv]
        pts = FScMap(self.pts)
        ts = pts.indepvararray.copy()
        ts.shape = (ts.shape[0],1)
        c = pts.coordarray.T
        all_coords = np.concatenate((c, ts), axis=1)
        # how to reuse tau and inf values instead of calling functions repeatedly
        #print "Use tau_v instead of tau_v(<args>) etc. for definition of psi function to take"
        #print "advantage of pre-computed taus and infs"
        if utils.remain(list(psi_defs.keys()), all_var_inputs) != []:
            print(("Warning: some influence definitions have labels that are" +\
                   " not matched by the recognized inputs to variable '%s':" % fv))
            print(("  %s" % str(utils.remain(list(psi_defs.keys()), all_var_inputs))))
        for inp in all_var_inputs:
            if inp not in psi_defs:
                continue
            psi_info = psi_defs[inp]
            if psi_info is None:
                psis[inp] = np.array([NaN]*len(pts))
            else:
                args, f = psi_info
                # make copy in case remove an argument
                temp_args = copy.copy(args)
                if 't' in args:
                    tix = args.index('t')
                    temp_args.remove('t')
                    arg_ixs = pts._map_names_to_ixs(temp_args)
                    arg_ixs.insert(tix, all_coords.shape[1]-1)
                else:
                    arg_ixs = pts._map_names_to_ixs(temp_args)
                try:
                    psis[inp] = np.array([f(*val_array.flatten()) \
                                      for val_array in all_coords[:,[arg_ixs]]])
                except:
                    print(f._args)
                    print(f._call_spec)
                    raise
        self.psi_vals[fv] = psis


    def make_pointsets(self):
        self.ensure()
        fv = self.focus_var
        assert self.psi_vals[fv] != {}
        assert self.tau_inf_vals != {}
        all_vals = {}
        all_vals.update(self.tau_inf_vals)
        base='psi_'+fv
        for inp, psis in self.psi_vals[fv].items():
            all_vals[base+'_'+inp] = psis
        self.psi_pts = Pointset(indepvararray=self.times, coorddict=all_vals)


    def calc_rankings(self, influence_type='psi'):
        self.ensure()
        if not hasattr(self, influence_type+'_pts'):
            raise AttributeError("Call make_pointsets first")
        else:
            influence_pts = getattr(self, influence_type+'_pts')
        varnames = influence_pts.coordnames
        # filter out any non-influence_type coordinates
        # (e.g. taus and infs that are included)
        influence_pts = influence_pts[[v for v in varnames if v[:3]==influence_type]]
        self.domscales = {influence_type: domscales(self.focus_var, self.pts,
                                                    influence_pts, influence_type,
                                                    self.tau_refs, self.psi_refs,
                                                    self.opts)}


    def reduced_Vode_system(self, vars):
        """Doesn't allow fast vars to be algebraic - better solution is to use
        ModelTransforms"""
        gen = self.gen
        name = "reduced_"+gen.funcspec._initargs['name']+"_"+"_".join(vars)
        varspecs = {}
        for xname in vars + gen.funcspec._initargs['auxvars']:
            varspecs[xname] = gen.funcspec._initargs['varspecs'][xname]
        sysargs = common.args(name=name, pars=gen.pars,
                              ics=common.filteredDict(gen.initialconditions,vars),
                              varspecs=varspecs, auxvars=gen.funcspec._initargs['auxvars'],
                              fnspecs=gen.funcspec._initargs['fnspecs'])
        nan_auxs = [xname for xname, val in gen.initialconditions.items() if not np.isfinite(val)]
        sysargs.pars.update(common.filteredDict(gen.initialconditions, vars+nan_auxs, neg=True))
        return Vode_ODEsystem(sysargs)


#### HELPER FUNCTIONS

def get_taus(pts, tau_names):
    """pts can be a Point or Pointset"""
    # filter this way in case there are None entries in tau_names.values()
    clist = [cname for cname in pts.coordnames if cname in list(tau_names.values())]
    t = pts[clist]
    t.mapNames(parseUtils.symbolMapClass(common.invertMap(tau_names)))
    return t

def get_infs(pts, inf_names):
    """pts can be a Point or Pointset"""
    # filter this way in case there are None entries in inf_names.values()
    clist = [cname for cname in pts.coordnames if cname in list(inf_names.values())]
    i = pts[clist]
    i.mapNames(parseUtils.symbolMapClass(common.invertMap(inf_names)))
    return i


def split_pts(pts, interval, reset_t0=None):
    """Return trajectory made up from pts argument restricted to the given
    interval.

    Use reset_t0 = T (float) to reset output trajectory t0 to T.
    Default is to use native time frame of pts argument."""
    t0, t1 = interval
    ix0 = pts.find(t0)
    if not isinstance(ix0, int):
        ix0 = ix0[0]
    ix1 = pts.find(t1)
    if not isinstance(ix1, int):
        ix1 = ix1[1]
    try:
        vals = pts[ix0:ix1].coordarray.copy()
    except ValueError:
        print("%d %d" % (ix0, ix1))
        raise
    ts = pts.indepvararray[ix0:ix1].copy()
    if reset_t0 is not None:
        ts += (reset_t0-ts[0])
    return numeric_to_traj(vals, 'split_%i_%i' % (ix0,ix1),
                           pts.coordnames,
                           indepvar=ts, discrete=False)


def find_epoch_period(epochs, verbose=False):
    cycle_len = None
    ixs = []
    # don't start i at 0 b/c will be in middle of an epoch
    for i, ep0 in enumerate(epochs[1:-1]):
        if verbose:
            print("start epoch = %d" % (i+1))
        for j, ep in enumerate(epochs[i+1:]):
            if verbose:
                print("end epoch = %d" % (i+j+1))
            try:
                res = [epochs[k]==epochs[k+j+1] for k in range(i,j)]
            except IndexError:
                # too close to end, start next i
                continue
            else:
                if verbose:
                    print("%d %d %s" % (i, j, res))
            if np.all(res) and j > i+1:
                if verbose:
                    print("Found period between epochs %d %d len=%d" % (i+1, j+2+i, j+1))
                cycle_len = j+1
                ixs = [i+1, i+j+2]
                return cycle_len, ixs
    return cycle_len, ixs


def split(ds_pt, thresh):
    """Split this grouping of values into sub-groups according to spectral
    gaps at threshold thresh"""
    sub_groups = []
    if ds_pt.relative_ratios:
        spectral_gaps = spectral_gaps_relative
    else:
        spectral_gaps = spectral_gaps_absolute
    sg = spectral_gaps(ds_pt.as_ratios, thresh)
    for ix_range in partition_range((0, len(ds_pt.influences)), sg):
        if ix_range[0] == ix_range[1]:
            continue
        sub_slice = slice(ix_range[0],ix_range[1])
        data = common.args()
        data.influences = ds_pt.influences[sub_slice]
        data.coordnames = ds_pt.coordnames[sub_slice]
        sub_groups.append(data)
    return sub_groups


def as_absolute_ratios(vals):
    """Express a single time point's values as ratios of the largest (>1).
    first entry is ratio of largest to itself == 1.
    """
    try:
        return [1.0]+[vals[0]/v for v in vals[1:]]
    except ZeroDivisionError:
        rats = [1.0]
        for i in range(1,len(vals)):
            if vals[i] > 0:
                rats.append(vals[0]/vals[i])
            else:
                # don't include any more (0 is smallest possible influence)
                rats.append(numpy.inf)
        return rats


def as_relative_ratios(vals):
    """Express a single time point's relative values as successive ratios (>1).
    first entry is ratio of largest to itself == 1.
    """
    try:
        return [1.0]+[vals[i-1]/vals[i] \
                      for i in range(1,len(vals))]
    except ZeroDivisionError:
        rats = [1.0]
        for i in range(1,len(vals)):
            if vals[i] > 0:
                rats.append(vals[i-1]/vals[i])
            else:
                # don't include any more (0 is smallest possible influence)
                rats.append(numpy.inf)
        return rats


def spectral_gaps_relative(as_rel_ratios, thresh):
    """List of indices of spectral gaps larger than thresh in size"""
    # enumerating an array of Bools
    return [i for i, is_gap in enumerate(np.asarray(as_rel_ratios) >= thresh) if is_gap]


def spectral_gaps_absolute(as_abs_ratios, thresh):
    """List of indices of spectral gaps larger than thresh in size"""
    as_abs_ratios = np.asarray(as_abs_ratios)
    gaps = []
    while True:
        test = np.array(as_abs_ratios >= thresh, int)
        if 1 in test:
            # only makes sense to find gaps if there's a value > thresh
            cutoff_ix = np.argmax(test)
        else:
            # only 0s present so no gaps remain
            break
        if gaps == []:
            gaps.append(cutoff_ix)
        else:
            gaps.append(cutoff_ix+gaps[-1])
        as_abs_ratios = as_abs_ratios[cutoff_ix:]
        if len(utils.remain(as_abs_ratios, [np.inf])) <= 1:
            break
        else:
            as_abs_ratios /= min(as_abs_ratios)
    return gaps


def partition_range(range_tuple, split_ixs):
    # assumes everything is non-negative integer, and
    # split_ixs is a list
    if not range_tuple[0] < range_tuple[1]:
        raise ValueError("Range tuple not in increasing order")
    if not np.alltrue(ix in range(range_tuple[0],range_tuple[1]) \
                      for ix in split_ixs):
        raise ValueError("Split indices out of range")
    if range_tuple[0] in split_ixs or range_tuple[1] in split_ixs:
        raise ValueError("Split indices out of range")
    if split_ixs == []:
        partition = [range_tuple]
    else:
        # make smallest index last
        split_ixs.sort()
        split_ixs.reverse()
        end_ix = split_ixs.pop()
        # end_ix not included in partition
        partition = [(range_tuple[0],end_ix)]
        while split_ixs != []:
            # pop comes off end (smallest first)
            next_split = (end_ix,split_ixs.pop())
            end_ix = next_split[1]
            partition.append(next_split)
        partition.append((end_ix,range_tuple[1]))
    return partition


def normalized_psis(epochs, root, midpoint_only=True, small=1e-16):
    """Returns Pointset of normalized* psis based on contents of 'epochs' argument,
    which would be returned from running da.domscales['psi'].calc_epochs()
    where da is a domscales assistant class.

      * Largest influence always normalized to 1 for every time point.
    """
    def zero_to_small(v):
        if v == 0:
            return small
        else:
            return v

    inpnames = epochs[0].influences[0].coordnames
    res = []
    ts = []
    if midpoint_only:
        for ep in epochs:
            numpts = len(ep.influences)
            midpt = int(numpts/2)
            ts.append(ep.influences['t'][midpt])
            repinf = ep.influences[midpt]
            ranks = ep.ranks[midpt]
            inv_ranks = common.invertMap(ranks)
            vals_ord = np.take(repinf.coordarray, ranks)
            ratios = np.array([1.0] + [zero_to_small(v)/vals_ord[0] for v in vals_ord[1:]])
            res.append(np.take(ratios, common.sortedDictValues(inv_ranks)))
    else:
        for ep in epochs:
            numpts = len(ep.influences)
            for nix in range(numpts):
                ts.append(ep.influences['t'][nix])
                repinf = ep.influences[nix]
                ranks = ep.ranks[nix]
                inv_ranks = common.invertMap(ranks)
                vals_ord = np.take(repinf.coordarray, ranks)
                ratios = np.array([1.0] + [zero_to_small(v)/vals_ord[0] for v in vals_ord[1:]])
                res.append(np.take(ratios, common.sortedDictValues(inv_ranks)))
    coordnames = ['psi_%s_%s'%(root, i) for i in inpnames]
    return Pointset(coordarray=np.array(res).T, indepvararray=ts, coordnames=coordnames)



def show_epochs(eps):
    "Small utility to print more detailed information about epochs"
    for i, ep in enumerate(eps):
        print(i, end=' ')
        ep.info()


def plot_psis(da, cols=None, do_vars=None, do_log=True, use_prefix=True):
    """da is a dssrt_assistant object.
    cols is an optional dictionary mapping names of Psi entries to specific color/style character codes.
    Pass do_vars list of Psi names to restrict, otherwise all will be plotted.
    Option to plot on vertical log scale.

    Option to switch off 'psi_' + da's focus_var as prefix for coordnames.

    Requires matplotlib.
    """
    from PyDSTool.matplotlib_import import plot
    pts=da.psi_pts
    if use_prefix:
        root = 'psi_'+da.focus_var+'_'
    else:
        root = ''
    ts = pts['t']
    if do_vars is None:
        do_vars = [c[len(root):] for c in pts.coordnames if root in c]
    if cols is None:
        colvals = ['g', 'r', 'k', 'b', 'c', 'y', 'm']
        styles = ['-', ':', '--']
        cols = []
        for s in styles:
            for c in colvals:
                cols.append(c+s)
    if len(do_vars) > len(cols):
        raise ValueError("Max number of permitted variables for these colors/styles is %i"%len(cols))
    print("Color scheme:")
    if do_log:
        for i, v in enumerate(do_vars):
            if root+v in pts.coordnames:
                print(" %s %s" % (v, cols[i]))
                plot(ts, np.log(pts[root+v]), cols[i])
    else:
        for i, v in enumerate(do_vars):
            if root+v in pts.coordnames:
                print(" %s %s" % (v, cols[i]))
                plot(ts, pts[root+v], cols[i])

# ---------------------------------------------------------------------------
# Sequence comparison code

MATCH = 'match'
MISMATCH = 'mismatch'
CLOSE = 'close'
FAR = 'far'
VCLOSE = 'vclose'
VFAR = 'vfar'
GAP = 'gap'
APPROX = 'approx'

class Scorer(object):
    def __init__(self, alphabet):
        self.match = None
        self.approx = None
        self.mismatch = None
        self.gap = None
        self.extension = None
        self.alphabet = alphabet
        pairs = utils.cartesianProduct(alphabet.all_vars, alphabet.all_vars)
        self.score_dict = {}.fromkeys(pairs, None)

    def __setitem__(self, k, v):
        if k in self.score_dict:
            self.score_dict[k] = v
            # also symmetric value
            a, b = k
            self.score_dict[(b, a)] = v
        else:
            raise KeyError("Invalid key %s"%k)

    def __getitem__(self, k):
        return getattr(self, self.score_dict[k])


class EpochSeqScorer(Scorer):
    def __init__(self, alphabet):
        Scorer.__init__(self, alphabet)
        self.match = 5
        self.gap = -2  # unused?
        self.mismatch = -4
        self.extension = -1

        fast = alphabet.fast
        slow = alphabet.slow

        s1 = Scorer(alphabet)
        s1.match = 5
        s1.vclose = 4
        s1.close = 2
        s1.far = -1
        s1.vfar = -3
        s1.mismatch = -5
        s1.gap = -2
        s1.extension = -1

        s2 = Scorer(alphabet)
        s2.gap = 1
        s2.mismatch = 0
        s2.far = 2
        s2.vfar = 1
        s2.close = 3
        s2.vclose = 4
        s2.match = 6
        s2.extension = 0

        for a in alphabet.dynamic_vars:
            s1[(a,a)] = MATCH
            s1[(fast(a),fast(a))] = MATCH
            s1[(slow(a),slow(a))] = MATCH
            s1[(fast(a),a)] = VCLOSE
            s1[(slow(a),a)] = VCLOSE
            s1[(slow(a),fast(a))] = CLOSE

            # same
            s2[(a,a)] = MATCH
            s2[(fast(a),fast(a))] = MATCH
            s2[(slow(a),slow(a))] = MATCH
            s2[(fast(a),a)] = VCLOSE
            s2[(slow(a),a)] = VCLOSE
            s2[(slow(a),fast(a))] = CLOSE

        s1[('_','_')] = CLOSE
        s2[('_','_')] = MATCH

        for a in alphabet.non_dynamic_vars:
            s1[(a,a)] = MATCH
            s1[(a,'_')] = CLOSE
            s2[(a,a)] = MATCH #VCLOSE
            s2[(a,'_')] = CLOSE

        for k, v in s1.score_dict.items():
            if v is None:
                d_common = len(common.intersect(alphabet.dynamic_vars, strip_speed(k)))
        #        s.score_dict[k] = MISMATCH
                if (k[0] == '_' or k[1] == '_') and k[0] != k[1]:
                    if d_common > 0:
                        s1.score_dict[k] = MISMATCH
                        s2.score_dict[k] = VFAR
                    else:
                        s1.score_dict[k] = MISMATCH
                        s2.score_dict[k] = VFAR
                elif d_common > 0:
                    s1.score_dict[k] = MISMATCH
                    s2.score_dict[k] = VFAR #MISMATCH
                else:
                    s1.score_dict[k] = MISMATCH
                    s2.score_dict[k] = CLOSE

        self.s1 = s1
        self.s2 = s2

        self.weight_unordered = 0.35
        self.weight_s1 = 0.2
        self.weight_s2 = 0.45

    def _validate_weights(self):
        assert self.weight_s1 + self.weight_s2 + self.weight_unordered == 1

    def __getitem__(self, k):
        return self.scorer(k[0], k[1])*self.match


    def scorer(self, a, b):
        self._validate_weights()
        s1 = self.s1
        s2 = self.s2
        minlen = min(len(a),len(b))
        maxlen = max(len(a),len(b))
        a_un = []
        b_un = []
        a_strip = strip_speed(a)
        b_strip = strip_speed(b)
        seen_cs = []
        for c in utils.union(a, b):
            cs = strip_speed([c])[0]
            if cs in seen_cs:
                continue
            else:
                seen_cs.append(cs)
            try:
                ixb = b_strip.index(cs)
            except ValueError:
                pass
            else:
                try:
                    ixa = a_strip.index(cs)
                except ValueError:
                    pass
                else:
                    a_un.append(a[ixa])
                    b_un.append(b[ixb])
        a_un = a_un + utils.remain(a, a_un)
        b_un = b_un + utils.remain(b, b_un)
        a_dyn = only_dynamic(pad(a, maxlen), s1.alphabet.dynamic_vars)
        b_dyn = only_dynamic(pad(b, maxlen), s1.alphabet.dynamic_vars)
        #unordered_sim = swdist(pad(a_un, maxlen), pad(b_un, maxlen), s)
        unordered_sim = sum([s2[(pad(a_un, maxlen)[i], pad(b_un, maxlen)[i])] for i in range(maxlen)])/(maxlen*s2.match)
        #print pad(a_un, maxlen), pad(b_un, maxlen), [s2[(pad(a_un, maxlen)[i], pad(b_un, maxlen)[i])] for i in range(maxlen)]
        #print "\n ", unordered_sim, swdist(a_dyn, b_dyn, s1), swdist(a, b, s2)
        return (self.weight_unordered*unordered_sim + \
                self.weight_s1*swdist(a_dyn, b_dyn, s1) + \
                self.weight_s2*swdist(a, b, s2))
        #return (unordered_sim*swdist(a_dyn, b_dyn, s1)*swdist(a,b,s2))**0.33333333333


def get_symbol_sequence(epoch_list, get_actives=True, get_modulatory=False):
    seq = []
    for ep in epoch_list:
        a, m = ep.seq_str()
        if get_actives:
            if get_modulatory:
                l = (a, m)
            else:
                l = a
        else:
            if get_modulatory:
                l = m
            else:
                raise ValueError("Must specify at least one type of output")
        seq.append(l)
    return seq

class VarAlphabet(object):
    def __init__(self, dynamic, non_dynamic):
        self.dynamic_vars = dynamic
        self.non_dynamic_vars = non_dynamic
        alphabet = dynamic + non_dynamic
        for speed in ['F', 'S']:
            alphabet.extend([v+'[%s]'%speed for v in dynamic])
        self.all_vars = alphabet + ['_']

    def fast(self, a):
        if a in self.dynamic_vars:
            return a+'[F]'
        else:
            raise ValueError("%s is not a dynamic variable" % a)

    def slow(self, a):
        if a in self.dynamic_vars:
            return a+'[S]'
        else:
            raise ValueError("%s is not a dynamic variable" % a)

# INTERNAL DSSRT SEQUENCE UTILITIES
def strip_speed(seq):
    res = []
    for a in seq:
        if a[-1] == ']':
            res.append(a[:-3])
        else:
            res.append(a)
    return res

def pad(seq, n):
    m = len(seq)
    if m > n:
        raise ValueError
    elif m < n:
        return seq + ['_']*(n-m)
    else:
        return seq

def only_dynamic(seq, dynamic_vars):
    res = []
    for a in seq:
        if strip_speed([a])[0] in dynamic_vars:
            res.append(a)
        else:
            res.append('_')
    return res


def comp_seqs(seq1, seq2, scorer):
    """Based on code by Gergely Szollosi"""
    rows=len(seq1)+1
    cols=len(seq2)+1
    a=np.zeros((rows,cols),float)
    if isinstance(seq1, list):
        blank = [['_']]
    else:
        blank = ['_']
    for i in range(1,rows):
        for j in range(1,cols):
            # Dynamic programing -- aka. divide and conquer:
            # Since gap penalties are linear in gap size
            # the score of an alignmet of length l only depends on the
            # the l-th characters in the alignment (match - mismatch - gap)
            # and the score of the one shorter (l-1) alignment,
            # i.e. we can calculate how to extend an arbritary alignment
            # soley based on the previous score value.
            choice1 = a[i-1, j-1] + scorer[(seq1[i-1], seq2[j-1])]
            choice2 = a[i-1, j] + scorer.gap
            choice3 = a[i, j-1] + scorer.gap
            a[i, j] = max(choice1, choice2, choice3)

    aseq1 = []
    aseq2 = []
    #We reconstruct the alignment into aseq1 and aseq2,
    i = len(seq1)
    j = len(seq2)
    while i>0 and j>0:
        #by performing a traceback of how the matrix was filled out above,
        #i.e. we find a shortest path from a[n,m] to a[0,0]
        score = a[i, j]
        score_diag = a[i-1, j-1]
        score_up = a[i, j-1]
        score_left = a[i-1, j]
        if score == score_diag + scorer[(seq1[i-1], seq2[j-1])]:
            aseq1 = [seq1[i-1]] + aseq1
            aseq2 = [seq2[j-1]] + aseq2
            i -= 1
            j -= 1
        elif score == score_left + scorer.gap:
            aseq1 = [seq1[i-1]] + aseq1
            aseq2 = blank + aseq2
            i -= 1
        elif score == score_up + scorer.gap:
            aseq1 = blank + aseq1
            aseq2 = [seq2[j-1]] + aseq2
            j -= 1
        else:
            #should never get here..
            raise RuntimeError
    while i>0:
        #If we hit j==0 before i==0 we keep going in i.
        aseq1 = [seq1[i-1]] + aseq1
        aseq2 = blank + aseq2
        i -= 1

    while j>0:
        #If we hit i==0 before i==0 we keep going in j.
        aseq1 = blank + aseq1
        aseq2 = [seq2[j-1]] + aseq2
        j -= 1

    final_score = swdist(aseq1, aseq2, scorer)
    #g = make_graph(seq1, seq2, a, rows, cols, scorer)
    return (aseq1, aseq2), final_score


##def make_graph(seq1, seq2, a, rows, cols, scorer):
##    #the simplest way is to make a graph of the possible constructions of the values in a
##    graph={}
##    for i in range(1,cols)[::-1]:
##        graph[(i,0)] = [(i-1,0)]
##        graph[(0,i)] = [(0,i-1)]
##        for j in range(1,cols)[::-1]:
##            graph[(i,j)]=[]
##            score = a[i, j]
##            score_diag = a[i-1, j-1]
##            score_up = a[i, j-1]
##            score_left = a[i-1, j]
##            if score == score_diag + scorer[(seq1[i-1], seq2[j-1])]:
##                graph[(i,j)] += [(i-1,j-1)]
##            if score == score_left + scorer.gap:
##                graph[(i,j)] += [(i-1,j)]
##            if score == score_up + scorer.gap:
##                graph[(i,j)] += [(i,j-1)]
##    return graph

def swdist(str1, str2, scorer, common_divisor='longest', min_threshold=None):
    """Return approximate string comparator measure (between 0.0 and 1.0)
       using the Smith-Waterman distance.

    USAGE:
      score = swdist(str1, str2, scorer, common_divisor, min_threshold)

    ARGUMENTS:
      str1            The first sequence
      str2            The second sequence
      scorer          scorer object instance to compare two symbols
      common_divisor  Method of how to calculate the divisor, it can be set to
                      'average','shortest', or 'longest' , and is calculated
                      according to the lengths of the two input strings
      min_threshold   Minimum threshold between 0 and 1

    DESCRIPTION:
      Smith-Waterman distance is commonly used in biological sequence alignment.

      Scores for matches, misses, gap and extension penalties are set to values
      described in the scorer
    """

    # Quick check if the strings are empty or the same - - - - - - - - - - - - -
    #

    n = len(str1)
    m = len(str2)

    if n*m == 0:
        return 0.0
    elif str1 == str2:
        return 1.0

    # Scores used for Smith-Waterman algorithm - - - - - - - - - - - - - - - - -
    #
#    match_score =       5
#    approx_score =      2
#    mismatch_score =   -5
#    gap_penalty =       5
#    extension_penalty = 1

    # Calculate the divisor - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #
    if (common_divisor not in ['average','shortest','longest']):
        raise ValueError('Illegal value for common divisor: %s' % \
                         (common_divisor))

    if common_divisor == 'average':
        divisor = 0.5*(n+m)*scorer.match  # Average maximum score
    elif common_divisor == 'shortest':
        divisor = min(n,m)*scorer.match
    else:  # Longest
        divisor = max(n,m)*scorer.match

    best_score = 0  # Keep the best score while calculating table

    d = np.zeros((n+1,m+1),float)  # distance matrix

    for i in range(1,n+1):
        for j in range(1,m+1):

            match = d[i-1, j-1] + scorer[(str1[i-1], str2[j-1])]

            insert = 0
            for k in range(1,i):
                score = d[i-k, j] + scorer.gap + k*scorer.extension
                insert = max(insert, score)

            delete = 0
            for l in range(1,j):
                score = d[i, j-l] + scorer.gap + l*scorer.extension
                delete = max(delete, score)

            d[i, j] = max(match, insert, delete, 0)
            best_score = max(d[i, j], best_score)

    # best_score can be min(len(str1),len(str2))*match_score (if one string is
    # a sub-string of the other string).
    #
    # The lower best_score the less similar the sequences are.
    #
    w = float(best_score) / float(divisor)
    assert (w >= 0.0) and (w <= 1.0), 'Similarity weight outside 0-1: %f' % (w)
    return w


def editdist_edits(str1, str2):
    """Return approximate string comparator measure (between 0.0 and 1.0)
     using the edit (or Levenshtein) distance as well as a triplet with the
     counts of the actual edits (inserts, deletes and substitutions).

  USAGE:
    score, edit_counts = editdist_edits(str1, str2)

  ARGUMENTS:
    str1           The first seq
    str2           The second seq

  DESCRIPTION:
    The edit distance is the minimal number of insertions, deletions and
    substitutions needed to make two strings equal.

    edit_counts  is a list with three elements that contain the number of
                 inserts, deletes and substitutions that were needed to convert
                 str1 into str2.

    For more information on the modified Soundex see:
    - http://www.nist.gov/dads/HTML/editdistance.html
  """

    # Check if the strings are empty or the same - - - - - - - - - - - - - - - -
    #
    n = len(str1)
    m = len(str2)

    if n + m == 0:
        return 0.0, [0,0,0]

    elif n*m == 0:
        if n == 0:
            return 0.0, [m,0,0]    # Inserts needed to get from empty to str1
        else:
            return 0.0, [0,n,0,0]  # Deletes nedded to get from str2 to empty

    elif str1 == str2:
        return 1.0, [0,0,0]

    d = []  # Table with the full distance matrix

    current = list(range(n+1))
    d.append(current)

    for i in range(1,m+1):

        previous = current
        current =  [i]+n*[0]
        str2char = str2[i-1]

        for j in range(1,n+1):
            substitute = previous[j-1]
            if (str1[j-1] != str2char):
                substitute += 1

            # Get minimum of insert, delete and substitute
            #
            current[j] = min(previous[j]+1, current[j-1]+1, substitute)

        d.append(current)

    # Count the number of edits that were needed - - - - - - - - - - - - - - - -
    #
    num_edits = [0,0,0]  # Number of Inserts, deletes and substitutions

    d_curr = d[m][n]  # Start with final position in table
    j = n
    i = m

    while (d_curr > 0):
        if (d[i-1][j-1]+1 == d_curr):  # Substitution
            i -= 1
            j -= 1
            num_edits[2] += 1
        elif (d[i-1][j]+1 == d_curr):  # Delete
            i -= 1
            num_edits[1] += 1
        elif (d[i][j-1]+1 == d_curr):  # Insert
            j -= 1
            num_edits[0] += 1

        else:  # Current position not larger than any of the previous positions
            if (d[i-1][j-1] == d_curr):
                i -= 1
                j -= 1
            elif (d[i-1][j] == d_curr):
                i -= 1
            elif (d[i][j-1] == d_curr):
                j -= 1
        d_curr = d[i][j]  # Update current position in table

    w = 1.0 - float(d[m][n]) / float(max(n,m))
    assert (w >= 0.0) and (w <= 1.0), 'Similarity weight outside 0-1: %f' % (w)

    return w, num_edits



JARO_MARKER_CHAR = '*'

def jaro(str1, str2, min_threshold=None):
    """Return approximate string comparator measure (between 0.0 and 1.0)

  USAGE:
    score = jaro(str1, str2, min_threshold)

  ARGUMENTS:
    str1           The first string
    str2           The second string
    min_threshold  Minimum threshold between 0 and 1 (currently not used)

  DESCRIPTION:
    As desribed in 'An Application of the Fellegi-Sunter Model of
    Record Linkage to the 1990 U.S. Decennial Census' by William E. Winkler
    and Yves Thibaudeau.
  """

    # Quick check if the strings are empty or the same - - - - - - - - - - - - -
    #
    if (str1 == '') or (str2 == ''):
        return 0.0
    elif (str1 == str2):
        return 1.0

    len1 = len(str1)
    len2 = len(str2)

    halflen = max(len1,len2) / 2 + 1

    ass1 = ''  # Characters assigned in str1
    ass2 = ''  # Characters assigned in str2

    workstr1 = str1  # Copy of original string
    workstr2 = str2

    common1 = 0  # Number of common characters
    common2 = 0

    # Analyse the first string  - - - - - - - - - - - - - - - - - - - - - - - - -
    #
    for i in range(len1):
        start = max(0,i-halflen)
        end   = min(i+halflen+1,len2)
        index = workstr2.find(str1[i],start,end)
        if (index > -1):  # Found common character
            common1 += 1
            ass1 = ass1 + str1[i]
            workstr2 = workstr2[:index]+JARO_MARKER_CHAR+workstr2[index+1:]

    # Analyse the second string - - - - - - - - - - - - - - - - - - - - - - - - -
    #
    for i in range(len2):
        start = max(0,i-halflen)
        end   = min(i+halflen+1,len1)
        index = workstr1.find(str2[i],start,end)
        if (index > -1):  # Found common character
            common2 += 1
            ass2 = ass2 + str2[i]
            workstr1 = workstr1[:index]+JARO_MARKER_CHAR+workstr1[index+1:]

    if (common1 != common2):
        print('Jaro: Wrong common values for strings "%s" and "%s"' % (str1, str2)
              + ', common1: %i, common2: %i' % (common1, common2)
              + ', common should be the same.')
        common1 = float(common1+common2) / 2.0  ##### This is just a fix #####

    if (common1 == 0):
        return 0.0

    # Compute number of transpositions  - - - - - - - - - - - - - - - - - - - - -
    #
    transposition = 0
    for i in range(len(ass1)):
        if (ass1[i] != ass2[i]):
            transposition += 1
    transposition = transposition / 2.0

    common1 = float(common1)
    w = 1./3.*(common1 / float(len1) + common1 / float(len2) + \
               (common1-transposition) / common1)

    assert (w >= 0.0) and (w <= 1.0), 'Similarity weight outside 0-1: %f' % (w)

    return w


def tabulate_epoch_seqs(epseq1, epseq2):
    """Create a text table of epochs"""
    str_table = []
    assert len(epseq1) == len(epseq2)
    for i in range(len(epseq1)):
        eps1 = epseq1[i]
        eps2 = epseq2[i]
        str_table.append([",".join(eps1), ",".join(eps2)])
    print(indent(str_table))

def indent(rows, hasHeader=False, headerChar='-', delim=' | ', justify='left',
           separateRows=False, prefix='', postfix='', wrapfunc=lambda x:x):
    """Indents a table by column.
       - rows: A sequence of sequences of items, one sequence per row.
       - hasHeader: True if the first row consists of the columns' names.
       - headerChar: Character to be used for the row separator line
         (if hasHeader==True or separateRows==True).
       - delim: The column delimiter.
       - justify: Determines how are data justified in their column.
         Valid values are 'left','right' and 'center'.
       - separateRows: True if rows are to be separated by a line
         of 'headerChar's.
       - prefix: A string prepended to each printed row.
       - postfix: A string appended to each printed row.
       - wrapfunc: A function f(text) for wrapping text; each element in
         the table is first wrapped by this function."""
    # closure for breaking logical rows to physical, using wrapfunc
    def rowWrapper(row):
        newRows = [wrapfunc(item).split('\n') for item in row]
        return list(zip_longest(*newRows, fillvalue=''))
    # break each logical row into one or more physical ones
    logicalRows = [rowWrapper(row) for row in rows]
    # columns of physical rows
    columns = zip_longest(*reduce(operator.add, logicalRows), fillvalue='')
    # get the maximum of each column by the string length of its items
    maxWidths = [max([len(str(item)) for item in column]) for column in columns]
    rowSeparator = headerChar * (len(prefix) + len(postfix) + sum(maxWidths) + \
                                 len(delim)*(len(maxWidths)-1))
    # select the appropriate justify method
    justify = {'center':str.center, 'right':str.rjust, 'left':str.ljust}[justify.lower()]
    output = StringIO()
    if separateRows:
        print(rowSeparator, file=output)
    for physicalRows in logicalRows:
        for row in physicalRows:
            print(prefix
                  + delim.join([justify(str(item),width) for (item,width) in zip(row,maxWidths)])
                  + postfix, file=output)
        if separateRows or hasHeader:
            print(rowSeparator, file=output)
            hasHeader=False
    return output.getvalue()
