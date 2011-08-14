"""
    Phase plane utilities.

    Some 2011 functionality has not yet been updated to use the plotter
    phase plane plotting manager.


IMPORTANT NOTE DURING DEVELOPMENT:
For now, many operations with nullclines assume that they are NOT multi-valued
as a function of their variables, and that they are monotonic only as the x
variable increases.

R. Clewley, 2006 - 2011
"""

from __future__ import division

from PyDSTool import *
from PyDSTool.MProject import *
from PyDSTool.common import args, metric, metric_L2, metric_weighted_L2, \
     metric_float, remain, fit_quadratic, fit_exponential, fit_diff_of_exp, \
     smooth_pts, nearest_2n_indices, make_poly_interpolated_curve, simple_bisection
from PyDSTool.common import _seq_types, _num_types
import PyDSTool.Redirector as redirc

import numpy as np
try:
    from numpy import unique
except ImportError:
    # older version of numpy
    from numpy import unique1d as unique
import matplotlib.pyplot as pp

from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import fsolve, minpack
from scipy.optimize import minpack, zeros
try:
    newton_meth = minpack.newton
except AttributeError:
    # newer version of scipy
    newton_meth = zeros.newton
from scipy import linspace, isfinite, sign, alltrue, sometrue, arctan

from random import uniform
import copy
import sys

norm = np.linalg.norm

# ----------------------------------------------------------------------------

_functions = ['find_nullclines', 'make_distance_to_line_auxfn',
              'make_distance_to_known_line_auxfn', 'crop_2D',
              'find_period', 'make_flow_normal_event', 'filter_NaN',
              'find_fixedpoints', 'find_steadystates', 'find_equilibria',
              'get_perp', 'get_orthonormal', 'get_rotated', 'angle_to_vertical',
              'is_min_bracket', 'find_nearest_sample_points_by_angle',
              'closest_perp_distance_between_splines',
              'closest_perp_distance_between_sample_points',
              'closest_perp_distance_on_spline',
              'line_intersection',
              'find_saddle_manifolds', 'show_PPs', 'get_PP']

_classes = ['distance_to_pointset', 'mesh_patch_2D', 'dx_scaled_2D',
            'phaseplane', 'fixedpoint_nD', 'fixedpoint_2D', 'nullcline',
            'Point2D', 'plotter_2D']

_features = ['inflection_zone_leaf', 'inflection_zone_node',
             'max_curvature_zone_leaf', 'max_curvature_zone_node']

__all__ = _functions + _classes + _features + ['plotter']

# ----------------------------------------------------------------------------


class distance_to_pointset(object):
    """First and second maximum and/or minimum distances of a point q
    to a set of points, returning a dictionary keyed by 'min' and
    'max' to dictionaries keyed by integers 1 and 2 (respectively).
    The values of this dictionary are dictionaries of
      'd' -> distance
      'pos' -> index into pts

    To restrict the search to a lower-dimensional subspace, specify a sub-set
    of the variables in q. Defaults to Euclidean distance (normord=2),otherwise
    specify metric-defining norm order.

    To speed up the search, use n > 1 to create n segments of the pointset,
    from which representative points will first be assessed, before the search
    narrows to a particular segment. Be sure to use a large enough n given
    the variation in each segment (which will otherwise cause spurious results
    for certain test points). Default n=30.
    [NB This option currently only returns the first max/min distances]

    The radius option provides an opportunity to use information from
    a previous call. This works for minimum distances only, and forces
    the same segment to be searched, provided that the new point is
    within the radius of the previous point (using 2-norm unless a 2D
    Point is specified). CURRENTLY, THIS DOES NOT WORK WELL. KEEP IT
    SWITCHED OFF USING (default) radius=0.

    If gen is not None, it should contain a generator for use with isochron-related
    events, for more accurate computations. The remaining arguments are associated
    with this usage.
    """

    _keys = ['d', 'pos']

    def __init__(self, pts, n=30, radius=0, gen=None, iso_ev=None, other_evnames=None,
                 pars_to_vars=None):
        self.all_pts = pts
        self.radius = radius
        num_pts = len(pts)
        assert n < num_pts, "Use a larger pointset or fewer segments"
        assert n > 0, "n must be a positive integer"
        assert type(n) == int, "n must be a positive integer"
        self.gen = gen
        self.iso_ev = iso_ev
        self.other_evnames = other_evnames
        self.pars_to_vars = pars_to_vars
        if n == 1:
            # always search whole pointset
            self.byseg = False
            self.segments = None
            self.rep_pts = None
            self.seg_start_ixs = None
        else:
            # use segments to speed up search
            self.byseg = True
            step, m = divmod(num_pts, n)
            if step < 2:
                # Not enough points to use segments
                self.byseg = False
                self.segments = None
                self.rep_pts = None
                self.seg_start_ixs = None
            else:
                half_step = int(floor(step/2.))
                # space the segment representatives in middle of segments
                self.seg_start_ixs = [step*i for i in range(n)]
                rep_pts = [pts[step*i+half_step] for i in range(n)]
                # treat any remainder segment specially
                # if remainder is too small then combine it with previous
                # segment (choose m>=6 fairly arbitrarily)
                self.segments = []
                base_ix = 0
                for i in range(n):
                    self.segments.append(pts[base_ix:base_ix+step])
                    base_ix += step
                if m >= 6:
                    half_step_last = int(floor(m/2.))
                    self.segments.append(pts[base_ix:])
                    rep_pts.append(pts[n*half_step+half_step_last])
                    self.seg_start_ixs.append(step*n)
                else:
                    # replace last segment with all the rest of the points,
                    # making it longer
                    self.segments[-1] = pts[base_ix-step:]
                    # leave rep_pts[-1] alone
                self.rep_pts = pointsToPointset(rep_pts)
        # record of [last segment identified for 1st and 2nd MINIMUM distance,
        #                                                           last point]
        # max distances are not currently recorded
        self.clear_history()

    def clear_history(self):
        self.history = [None, None, None]

    def __call__(self, q, use_norm=False, normord=2, minmax=['min', 'max']):
        dic_info = self._distance(q, use_norm, normord, minmax)
        if self.gen is not None:
            # refine distance using gen_test_fn
            perp_ev, t_ev = _find_min_pt(self.gen, q,
                                         dic_info['min'][1]['pos'], self.all_pts,
                                         self.pars_to_vars, self.iso_ev,
                                         self.other_evnames)
            ev_pt = perp_ev[0][q.coordnames]
            if use_norm:
                dic_info['min'][0] = norm(q - ev_pt, normord)
            else:
                m = q - ev_pt
                try:
                    vars = q.coordnames
                except AttributeError:
                    raise TypeError("Input point must be a Point object")
                m[vars[0]] = abs(m[vars[0]])
                m[vars[1]] = abs(m[vars[1]])
                dic_info['min'][0] = m
            # leave pos information alone -- it says what the nearest point is in
            # self.all_pts
        return dic_info

    def _distance(self, q, use_norm, normord, minmax):
        try:
            vars = q.coordnames
        except AttributeError:
            raise TypeError("Input point must be a Point object")
        # HACK: cannot use Inf in dmin for these comparisons
        # have to use a big number
        if use_norm:
            if 'min' in minmax:
                dmin = (_num_inf, NaN)
                dmin_old = (_num_inf, NaN)
                def le(a,b):
                    return a < b
            if 'max' in minmax:
                dmax = (0, NaN)
                dmax_old = (0, NaN)
                def ge(a,b):
                    return a > b
            def fn(p,q):
                return norm(p-q, normord)
        else:
            if 'min' in minmax:
                dmin = (array([_num_inf for v in vars]), NaN)
                dmin_old = (array([_num_inf for v in vars]), NaN)
                def le(a,b):
                    return sometrue(a-b<0)
            if 'max' in minmax:
                dmax = (array([0. for v in vars]), NaN)
                dmax_old = (array([0. for v in vars]), NaN)
                def ge(a,b):
                    return sometrue(a-b>0)
            def fn(p,q):
                m = p-q
                m[vars[0]] = abs(m[vars[0]])
                m[vars[1]] = abs(m[vars[1]])
                return m
#        if remain(self.all_pts.coordnames, vars) != []:
#            raise TypeError("Input points must be a Pointset object sharing "
#                            "common coordinate names")
        if self.byseg:
            if self.history[0] is None or self.radius == 0:
                use_history = False
            else:
                # use last known segment only if q is within radius of
                # last q
                try:
                    test = q-self.history[2]   # a Point
                except:
                    # problem with self.history[2] (externally tampered with?)
                    print "History is: ", self.history
                    raise RuntimeError("Invalid history object in "
                                       "distance_to_pointset class instance")
                try:
                    use_history = abs(test[0]) < self.radius[0] and \
                       abs(test[1]) < self.radius[1]
                except:
                    # radius is a scalar
                    use_history = norm(test) < self.radius
            if use_history and minmax == ['min']:
                    res_min1 = self.search_min(q, self.segments[self.history[0]][vars],
                                    fn, le, dmin, dmin_old)
                    res_min2 = self.search_min(q, self.segments[self.history[1]][vars],
                                    fn, le, dmin, dmin_old)
                    rm1 = res_min1['min'][1]
                    rm2 = res_min2['min'][1]
                    if rm1['d'] < rm2['d']:
                        seg_min_pos1 = rm1['pos']
                        seg_min_d1 = rm1['d']
                        seg_min1 = self.history[0]
                    else:
                        seg_min_pos1 = rm2['pos']
                        seg_min_d1 = rm2['d']
                        seg_min1 = self.history[1]
                    global_min_ix1 = self.seg_start_ixs[seg_min1] + seg_min_pos1
                    return {'min': {1: {'d': seg_min_d1, 'pos': global_min_ix1},
                                    2: {'d': None, 'pos': None}}}
            else:
                if 'min' in minmax and 'max' in minmax:
                    first_pass = self.search_both(q, self.rep_pts[vars],
                                 fn, le, ge, dmin, dmin_old, dmax, dmax_old)
                    # search 1st and 2nd segments for min and max
                    # the 'pos' index will be into the self.segments list
                    seg_ixs_min = [first_pass['min'][1]['pos']]
                    seg_ixs_min.append(first_pass['min'][2]['pos'])
                    seg_ixs_max = [first_pass['max'][1]['pos']]
                    seg_ixs_max.append(first_pass['max'][2]['pos'])
                    # min
                    res_min1 = self.search_min(q, self.segments[seg_ixs_min[0]][vars],
                                    fn, le, dmin, dmin_old)
                    res_min2 = self.search_min(q, self.segments[seg_ixs_min[1]][vars],
                                    fn, le, dmin, dmin_old)
                    rm1 = res_min1['min'][1]
                    rm2 = res_min2['min'][1]
                    if rm1['d'] < rm2['d']:
                        seg_min_pos1 = rm1['pos']
                        seg_min_d1 = rm1['d']
                        seg_min1 = seg_ixs_min[0]
                        seg_min2 = seg_ixs_min[1]
                    else:
                        seg_min_pos1 = rm2['pos']
                        seg_min_d1 = rm2['d']
                        seg_min1 = seg_ixs_min[1]
                        seg_min2 = seg_ixs_min[0]
                    global_min_ix1 = self.seg_start_ixs[seg_min1] + seg_min_pos1
                    # do second minimum here (NOT IMPLEMENTED)
                    # max
                    res_max1 = self.search_max(q, self.segments[seg_ixs_max[0]][vars],
                                    fn, ge, dmax, dmax_old)
                    res_max2 = self.search_max(q, self.segments[seg_ixs_max[1]][vars],
                                    fn, ge, dmax, dmax_old)
                    rm1 = res_max1['max'][1]
                    rm2 = res_max2['max'][1]
                    if rm1['d'] < rm2['d']:
                        seg_max_pos1 = rm1['pos']
                        seg_max_d1 = rm1['d']
                        seg_max1 = seg_ixs_max[0]
                    else:
                        seg_max_pos1 = rm2['pos']
                        seg_max_d1 = rm2['d']
                        seg_max1 = seg_ixs_max[1]
                    global_max_ix1 = self.seg_start_ixs[seg_max1] + seg_max_pos1
                    # do second maximum here (NOT IMPLEMENTED)
                    # array(q) is a cheap way to take a copy of q without
                    # making a new Point object
                    self.history = [seg_min1, seg_min2, array(q)]
                    return {'min': {1: {'d': seg_min_d1, 'pos': global_min_ix1},
                                    2: {'d': None, 'pos': None}},
                            'max': {1: {'d': seg_max_d1, 'pos': global_max_ix1},
                                    2: {'d': None, 'pos': None}}}
                elif 'min' in minmax:
                    first_pass = self.search_min(q, self.rep_pts[vars],
                                 fn, le, dmin, dmin_old)
                    seg_ixs_min = [first_pass['min'][1]['pos']]
                    seg_ixs_min.append(first_pass['min'][2]['pos'])
                    res_min1 = self.search_min(q, self.segments[seg_ixs_min[0]][vars],
                                    fn, le, dmin, dmin_old)
                    res_min2 = self.search_min(q, self.segments[seg_ixs_min[1]][vars],
                                    fn, le, dmin, dmin_old)
                    rm1 = res_min1['min'][1]
                    rm2 = res_min2['min'][1]
                    if rm1['d'] < rm2['d']:
                        seg_min_pos1 = rm1['pos']
                        seg_min_d1 = rm1['d']
                        seg_min1 = seg_ixs_min[0]
                        seg_min2 = seg_ixs_min[1]
                    else:
                        seg_min_pos1 = rm2['pos']
                        seg_min_d1 = rm2['d']
                        seg_min1 = seg_ixs_min[1]
                        seg_min2 = seg_ixs_min[0]
                    global_min_ix1 = self.seg_start_ixs[seg_min1] + seg_min_pos1
                    # do second minimum here (NOT IMPLEMENTED)
                    # array(q) is a cheap way to take a copy of q without
                    # making a new Point object
                    self.history = [seg_min1, seg_min2, array(q)]
                    return {'min': {1: {'d': seg_min_d1, 'pos': global_min_ix1},
                                    2: {'d': None, 'pos': None}}}
                elif 'max' in minmax:
                    first_pass = self.search_max(q, self.rep_pts[vars],
                                 fn, ge, dmax, dmax_old)
                    seg_ixs_max = [first_pass['max'][1]['pos']]
                    seg_ixs_max.append(first_pass['max'][2]['pos'])
                    res_max1 = self.search_max(q, self.segments[seg_ixs_max[0]][vars],
                                    fn, ge, dmax, dmax_old)
                    res_max2 = self.search_max(q, self.segments[seg_ixs_max[1]][vars],
                                    fn, ge, dmax, dmax_old)
                    rm1 = res_max1['max'][1]
                    rm2 = res_max2['max'][1]
                    if rm1['d'] < rm2['d']:
                        seg_max_pos1 = rm1['pos']
                        seg_max_d1 = rm1['d']
                        seg_max1 = seg_ixs_max[0]
                    else:
                        seg_max_pos1 = rm2['pos']
                        seg_max_d1 = rm2['d']
                        seg_max1 = seg_ixs_max[1]
                    global_max_ix1 = self.seg_start_ixs[seg_max1] + seg_max_pos1
                    # do second maximum here (NOT IMPLEMENTED)
                    # don't update history because it's only for minimum distance
                    return {'max': {1: {'d': seg_max_d1, 'pos': global_max_ix1},
                                    2: {'d': None, 'pos': None}}}
                else:
                    raise RuntimeError("Invalid min/max option")
        else:
            if 'min' in minmax and 'max' in minmax:
                return self.search_both(q, self.all_pts[vars],
                             fn, le, ge, dmin, dmin_old, dmax, dmax_old)
            elif 'min' in minmax:
                return self.search_min(q, self.all_pts[vars],
                             fn, le, dmin, dmin_old)
            elif 'max' in minmax:
                return self.search_max(q, self.all_pts[vars],
                             fn, ge, dmax, dmax_old)
            else:
                raise RuntimeError("Invalid min/max option")


    def search_both(self, q, pts,
               fn, le, ge, dmin, dmin_old, dmax, dmax_old):
        for i, p in enumerate(pts):
            d = fn(p,q)
            if le(d, dmin[0]):
                dmin_old = dmin
                dmin = (d, i)
            elif le(d, dmin_old[0]):
                dmin_old = (d, i)
            if ge(d, dmax[0]):
                dmax_old = dmax
                dmax = (d, i)
            elif ge(d, dmax_old[0]):
                dmax_old = (d,i)
        return {'min': {1: dict(zip(self._keys,dmin)),
                        2: dict(zip(self._keys,dmin_old))},
                'max': {1: dict(zip(self._keys,dmax)),
                        2: dict(zip(self._keys,dmax_old))}}

    def search_min(self, q, pts, fn, le, dmin, dmin_old):
        for i, p in enumerate(pts):
            d = fn(p,q)
            if le(d, dmin[0]):
                dmin_old = dmin
                dmin = (d, i)
            elif le(d, dmin_old[0]):
                dmin_old = (d, i)
        return {'min': {1: dict(zip(self._keys,dmin)),
                        2: dict(zip(self._keys,dmin_old))}}

    def search_max(self, q, pts, fn, ge, dmax, dmax_old):
        for i, p in enumerate(pts):
            d = fn(p,q)
            if ge(d, dmax[0]):
                dmax_old = dmax
                dmax = (d, i)
            elif ge(d, dmax_old[0]):
                dmax_old = (d,i)
        return {'max': {1: dict(zip(self._keys,dmax)),
                        2: dict(zip(self._keys,dmax_old))}}


def find_nullclines(gen, xname, yname, x_dom=None, y_dom=None, fps=None, n=10,
                    t=0, xtol=None, fixed_vars=None, jac=None, max_step=0,
                    max_num_points=1000, only_var=None):
    """Find nullclines of a two-dimensional sub-system of the given
    Generator object gen, specified by xname and yname.

    Inputs:

    gen is a Generator object
    xname, yname are state variable names of gen

    Optional inputs:

    Restriction of x and y to sub-domains can be made using x_dom and y_dom
    lists of [min, max] values (default to domains given in generator).

    Setting of unused variables that are fixed for 2D nullclines can be given
      by the dict or Point fixed_vars, otherwise the Generator's initial
      conditions will be used.

    n = initial number of meshpoints for fsolve. Don't set this large if using
      PyCont, e.g. use n=3. Default is 10.

    Set t value for non-autonomous systems (default 0). Support for Jacobians
      with non-autonomous systems is not yet provided.

    jac is a Jacobian function that accepts keyword arguments including t for
      time (even if Jacobian is time-independent).

    max_step (dictionary) tells PyCont the largest step size to use for each
      variable. Integer 0 (default) switches off PyCont use. Use None
      to tell PyCont to use default max step size (5e-1).

    fps can be a list of points previously calculated to be fixed points,
      which this function will use as additional starting points for
      computation.

    xtol sets the accuracy to which the nullclines are calculated. Default is
      approx. 1e-8.

    only_var (variable name) requests that only the nullcline for that variable
      will be computed. The variable name must correspond to xname or yname.


    Output:
        (x_null, y_null) arrays of (x,y) pairs. Fixed points will be included if
        they are provided (they are expected to be calculated to high enough
        accuracy to smoothly fit with the nullcline sample points).

    Note that the points returned are not guaranteed to be in any particular
    order when PyCont is not used.
    """
    vardict = filteredDict(copy.copy(gen.initialconditions), gen.funcspec.vars)
    if fixed_vars is not None:
        vardict.update(filteredDict(gen._FScompatibleNames(dict(fixed_vars)),
                        remain(gen.funcspec.vars, [xname, yname])))
    var_ix_map = invertMap(gen.funcspec.vars)
    max_step = gen._FScompatibleNames(max_step)
    xname_orig = gen._FScompatibleNamesInv(xname)
    yname_orig = gen._FScompatibleNamesInv(yname)
    xname = gen._FScompatibleNames(xname)
    yname = gen._FScompatibleNames(yname)
    if only_var is None:
        do_vars = [xname, yname]
    else:
        assert only_var in [xname, yname], "only_var must be one of xname or yname"
        do_vars = [only_var]
    x_ix = var_ix_map[xname]
    y_ix = var_ix_map[yname]
    pp_vars = [xname, yname]
    pp_vars.sort()
    pp_var_ix_map = invertMap(pp_vars)
    def xdot_x(x, y, t):
        vardict[yname] = y
        vardict[xname] = x
        return gen.Rhs(t,vardict,gen.pars)[x_ix]
    def ydot_y(y, x, t):
        vardict[yname] = y
        vardict[xname] = x
        return gen.Rhs(t,vardict,gen.pars)[y_ix]
    def xdot_y(x, y, t):
        vardict[yname] = y
        vardict[xname] = x
        return gen.Rhs(t,vardict,gen.pars)[y_ix]
    def ydot_x(y, x, t):
        vardict[yname] = y
        vardict[xname] = x
        return gen.Rhs(t,vardict,gen.pars)[x_ix]
    if x_dom is None and xname in do_vars:
        x_dom = gen.xdomain[xname]
        if not (isfinite(x_dom[0]) and isfinite(x_dom[1])):
            raise PyDSTool_ExistError("Must specify finite range for %s"%xname)
    if y_dom is None and yname in do_vars:
        y_dom = gen.xdomain[yname]
        if not (isfinite(y_dom[0]) and isfinite(y_dom[1])):
            raise PyDSTool_ExistError("Must specify finite range for %s"%yname)
    if gen.haveJacobian():
        # user-supplied Jacobian if present
        def xfprime_x(x, y, t):
            vardict[xname] = x
            vardict[yname] = y
            return gen.Jacobian(t, vardict, gen.pars)[x_ix]
        def xfprime_y(x, y, t):
            vardict[xname] = x
            vardict[yname] = y
            return gen.Jacobian(t, vardict, gen.pars)[y_ix]
        def yfprime_x(y, x, t):
            vardict[xname] = x
            vardict[yname] = y
            return gen.Jacobian(t, vardict, gen.pars)[x_ix]
        def yfprime_y(y, x, t):
            vardict[xname] = x
            vardict[yname] = y
            return gen.Jacobian(t, vardict, gen.pars)[y_ix]
    elif jac is not None:
        # assumes **args-compatible signature!
        xix = pp_var_ix_map[xname]
        yix = pp_var_ix_map[yname]
        # x will be first argument to these
        xfprime_x = lambda x, y, t: array(jac(**{'t': t, xname: x,
                                      yname: y})[xix])
        xfprime_y = lambda x, y, t: array(jac(**{'t': t, xname: x,
                                      yname: y})[yix])
        # y will be first argument to these
        yfprime_x = lambda y, x, t: array(jac(**{'t': t, xname: x,
                                      yname: y})[xix])
        yfprime_y = lambda y, x, t: array(jac(**{'t': t, xname: x,
                                      yname: y})[yix])
    else:
        xfprime_x = None
        xfprime_y = None
        yfprime_x = None
        yfprime_y = None
    if xtol is None:
        xtol = 1.49012e-8
    x_range = list(linspace(x_dom[0],x_dom[1],n))
    y_range = list(linspace(y_dom[0],y_dom[1],n))
    x_null_pts = []
    y_null_pts = []
    if fps is not None:
        fps_FS = [gen._FScompatibleNames(fp) for fp in fps]
        add_pts_x = [fp[xname] for fp in fps_FS]
        add_pts_y = [fp[yname] for fp in fps_FS]
    else:
        add_pts_x = []
        add_pts_y = []
    rout = redirc.Redirector(redirc.STDOUT)
    rout.start()
    if yname in do_vars:
        for x0 in add_pts_x + x_range[::int(ceil(n/10.))]:
            try:
                y_null_pts.extend([(_xinf_ND(xdot_y,x0,args=(y,t),
                               xddot=xfprime_y,xtol=xtol),y) for y in y_range])
                y_null_pts.extend([(x0,_xinf_ND(ydot_y,y,args=(x0,t),
                               xddot=yfprime_y,xtol=xtol)) for y in y_range])
            except (OverflowError, ZeroDivisionError):
                rout.stop()
            except:
                rout.stop()
                raise
    if xname in do_vars:
        for y0 in add_pts_y + y_range[::int(ceil(n/10.))]:
            try:
                x_null_pts.extend([(x,_xinf_ND(ydot_x,y0,args=(x,t),
                               xddot=yfprime_x,xtol=xtol)) for x in x_range])
                x_null_pts.extend([(_xinf_ND(xdot_x,x,args=(y0,t),
                               xddot=xfprime_x,xtol=xtol),y0) for x in x_range])
            except (OverflowError, ZeroDivisionError):
                rout.stop()
            except:
                rout.stop()
                raise
    rout.stop()
    # 5% tolerance outside domain for keeping points in nullclines
    # final points will be cropped to domain with this tolerance
    tol = 0.02
    xwidth = abs(x_dom[1]-x_dom[0])
    ywidth = abs(y_dom[1]-y_dom[0])
    xinterval=Interval('xdom', float, [x_dom[0]-tol*xwidth, x_dom[1]+tol*xwidth])
    yinterval=Interval('ydom', float, [y_dom[0]-tol*ywidth, y_dom[1]+tol*ywidth])
    x_null = filter_close_points(crop_2D(filter_NaN(x_null_pts),
                                xinterval, yinterval), xtol*10)
    y_null = filter_close_points(crop_2D(filter_NaN(y_null_pts),
                                xinterval, yinterval), xtol*10)
    # max_step = 0 means do not use PyCont to improve accuracy
    if max_step != 0:
        add_fp_pts = array([add_pts_x, add_pts_y]).T
        if not isinstance(max_step, dict):
            max_step = {xname: max_step, yname: max_step}
        # use PyCont to improve accuracy
        # PyCont will order the points so can draw lines
        loop_step = 5
        # current limitation!
        # pycont won't find disconnected branches of nullclines so we have to seed
        # with multiple starting points
        # - also, the following assumes more than one point was already added to
        # these lists
        # Y
        if yname in do_vars:
            if len(y_null) > 1:
                x_init = y_null[1,0]
                y_init = y_null[1,1]
            elif len(y_null) == 1:
                x_init = y_null[0,0]
                y_init = y_null[0,1]
            elif fps is not None and len(fps) > 0:
                x_init = fps[0][xname_orig] + 1e-4*(x_dom[1]-x_dom[0])
                y_init = fps[0][yname_orig] + 1e-4*(y_dom[1]-y_dom[0])
            else:
                x_init = (x_dom[0]+x_dom[1])/2.
                y_init = (y_dom[0]+y_dom[1])/2.
            if x_init < x_dom[0]:
                x_init = x_dom[0]
            elif x_init > x_dom[1]:
                x_init = x_dom[1]
            if y_init < y_dom[0]:
                y_init = y_dom[0]
            elif y_init > y_dom[1]:
                y_init = y_dom[1]
            sysargs_y = args(name='nulls_y', pars={xname: x_init},
                           ics={yname: y_init},
                           varspecs={yname: gen.funcspec._initargs['varspecs'][yname]},
                           xdomain={yname: y_dom}, pdomain={xname: x_dom})
            if 'inputs' in gen.funcspec._initargs: # and isinstance(gen.funcspec._initargs['inputs'], dict):
                if len(gen.funcspec._initargs['inputs']) > 0:
                    sysargs_y['inputs'] = gen.inputs.copy()
            if 'fnspecs' in gen.funcspec._initargs:
                sysargs_y.update({'fnspecs':renameClashingAuxFnPars(gen.funcspec._initargs['fnspecs'], [xname])})
            sysargs_y.pars.update(gen.pars)
            sysargs_y.pars.update(filteredDict(vardict, [xname, yname], neg=True))
            sysargs_y.pars['time'] = t
            sys_y = Vode_ODEsystem(sysargs_y)
            P = ContClass(sys_y)
            PCargs = args(name='null_curve_y', type='EP-C')
            PCargs.freepars = [xname]
            PCargs.MinStepSize = 1e-4
            PCargs.VarTol = PCargs.FuncTol = xtol
            PCargs.TestTol = 1e-6
            if max_step is None:
                PCargs.MaxStepSize = 5e-1
                PCargs.StepSize = 1e-2
            else:
                PCargs.MaxStepSize = max_step[xname]
                PCargs.StepSize = max_step[xname]/2
            PCargs.MaxNumPoints = loop_step
            P.newCurve(PCargs)
            # check every loop_step points until go out of bounds
            done = False
            num_points = 0
            while not done:
                try:
                    P['null_curve_y'].forward()
                except PyDSTool_ExistError:
                    print 'null_curve_y failed in forward direction'
                    raise
                else:
                    num_points += loop_step
                    done = not check_bounds(array([P['null_curve_y'].new_sol_segment[xname],
                                                  P['null_curve_y'].new_sol_segment[yname]]).T,
                                            xinterval, yinterval) \
                          or num_points > max_num_points
            done = False
            num_points = 0
            while not done:
                try:
                    P['null_curve_y'].backward()
                except PyDSTool_ExistError:
                    print 'null_curve_y failed in backward direction'
                    raise
                else:
                    num_points += loop_step
                    done = not check_bounds(array([P['null_curve_y'].new_sol_segment[xname],
                                                  P['null_curve_y'].new_sol_segment[yname]]).T,
                                            xinterval, yinterval) \
                         or num_points > max_num_points
            # can get repetition of some points
            y_null = crop_2D(array([P['null_curve_y'].sol[xname],
                                    P['null_curve_y'].sol[yname]]).T,
                             xinterval, yinterval)
            x_vals = y_null[:,0]
            x_vals_unique, indices = unique(x_vals, return_index=True)
            y_null = y_null[indices]
            # add fixed points to nullcline, assuming sufficient accuracy
            # also searchsorted assumes monotonicity in x values, i.e.
            # that the nullcline is a function of x
            fp_ixs = np.searchsorted(x_vals_unique, add_pts_x)
            for n, ix in enumerate(fp_ixs):
                # +n offsets fact that n entries were already added
                y_null = np.insert(y_null, ix+n, add_fp_pts[n], axis=0)

        # X
        if xname in do_vars:
            if len(x_null) > 1:
                x_init = x_null[1,0]
                y_init = x_null[1,1]
            elif len(x_null) == 1:
                x_init = x_null[0,0]
                y_init = x_null[0,1]
            elif fps is not None and len(fps) > 0:
                x_init = fps[0][xname_orig] + 1e-4*(x_dom[1]-x_dom[0])
                y_init = fps[0][yname_orig] + 1e-4*(y_dom[1]-y_dom[0])
            else:
                x_init = (x_dom[0]+x_dom[1])/2.
                y_init = (y_dom[0]+y_dom[1])/2.
            if x_init < x_dom[0]:
                x_init = x_dom[0]
            elif x_init > x_dom[1]:
                x_init = x_dom[1]
            if y_init < y_dom[0]:
                y_init = y_dom[0]
            elif y_init > y_dom[1]:
                y_init = y_dom[1]
            sysargs_x = args(name='nulls_x', pars={yname: y_init},
                           ics={xname: x_init}, tdata=[t,t+1],
                           varspecs={xname: gen.funcspec._initargs['varspecs'][xname]},
                           xdomain={xname: x_dom}, pdomain={yname: y_dom})
            if 'inputs' in gen.funcspec._initargs: # and isinstance(gen.funcspec._initargs['inputs'], dict):
                if len(gen.funcspec._initargs['inputs']) > 0:
                    sysargs_x['inputs'] = gen.inputs.copy()
            if 'fnspecs' in gen.funcspec._initargs:
                sysargs_x.update({'fnspecs': renameClashingAuxFnPars(gen.funcspec._initargs['fnspecs'], [yname])})
            sysargs_x.pars.update(gen.pars)
            sysargs_x.pars.update(filteredDict(vardict, [xname, yname], neg=True))
            sysargs_x.pars['time'] = t
            sys_x = Vode_ODEsystem(sysargs_x)
            P = ContClass(sys_x)
            PCargs = args(name='null_curve_x', type='EP-C')
            PCargs.freepars = [yname]
            PCargs.MinStepSize = 1e-4
            PCargs.VarTol = PCargs.FuncTol = xtol
            PCargs.TestTol = 1e-6
            if max_step is None:
                PCargs.MaxStepSize = 5e-1
                PCargs.StepSize = 1e-2
            else:
                PCargs.MaxStepSize = max_step[yname]
                PCargs.StepSize = max_step[yname]/2
            PCargs.MaxNumPoints = loop_step
            P.newCurve(PCargs)
            done = False
            num_points = 0
            while not done:
                try:
                    P['null_curve_x'].forward()
                except PyDSTool_ExistError:
                    print 'null_curve_x failed in forward direction'
                    raise
                else:
                    num_points += loop_step
                    done = not check_bounds(array([P['null_curve_x'].new_sol_segment[xname],
                                                  P['null_curve_x'].new_sol_segment[yname]]).T,
                                            xinterval, yinterval) \
                         or num_points > max_num_points
            done = False
            num_points = 0
            while not done:
                try:
                    P['null_curve_x'].backward()
                except PyDSTool_ExistError:
                    print 'null_curve_x failed in backward direction'
                    raise
                else:
                    num_points += loop_step
                    done = not check_bounds(array([P['null_curve_x'].new_sol_segment[xname],
                                                  P['null_curve_x'].new_sol_segment[yname]]).T,
                                            xinterval, yinterval) \
                         or num_points > max_num_points
            # can get repetition of some points
            x_null = crop_2D(array([P['null_curve_x'].sol[xname],
                                    P['null_curve_x'].sol[yname]]).T,
                             xinterval, yinterval)
            x_vals = x_null[:,0]
            x_vals_unique, indices = unique(x_vals, return_index=True)
            x_null = x_null[indices]
            # add fixed points to nullcline, assuming sufficient accuracy
            # also searchsorted assumes monotonicity in x values, i.e.
            # that the nullcline is a function of x
            fp_ixs = np.searchsorted(x_vals_unique, add_pts_x)
            for n, ix in enumerate(fp_ixs):
                # +n offsets fact that n entries were already added
                x_null = np.insert(x_null, ix+n, add_fp_pts[n], axis=0)

    return (gen._FScompatibleNamesInv(x_null), gen._FScompatibleNamesInv(y_null))



def find_fixedpoints(gen, subdomain=None, n=5, maxsearch=1e3, eps=1e-8,
                     t=0, jac=None):
    """Find fixed points of a system in a given domain,
    on the assumption that they are isolated points.

    Returns list of dictionaries mapping the variable names to the values.

    Set t value for non-autonomous systems (default 0).
    """
    # get state variable domains if subdomain dictionary not given
    if subdomain is None:
        subdomain = filteredDict(gen.xdomain, gen.funcspec.vars)
    else:
        subdomain = gen._FScompatibleNames(subdomain)
        assert remain(subdomain.keys(),gen.funcspec.vars) == [] and \
               remain(gen.funcspec.vars,subdomain.keys()) == []
    # only vary over domains that are given as ranges: fixed values
    # are not counted
    D = 0
    xdict = {}.fromkeys(gen.funcspec.vars)
    fixed_vars = {}
    for xname, dom in subdomain.iteritems():
        if isinstance(dom, (tuple,list)):
            if not (isfinite(dom[0]) and isfinite(dom[1])):
                raise RuntimeError("Must specify a finite range for %s"%xname)
            D += 1
        else:
            xdict[xname] = dom
            # put fixed vars back into return value
            fixed_vars[xname] = dom
    var_ix_map = invertMap(gen.funcspec.vars)
    # pick n uniformly distributed starting points in domains,
    # ensuring n isn't so large as to require more than maxsearch
    # number of searches
    while True:
        if n**D > maxsearch:
            n = n-1
        else:
            break
        if n < 3:
            raise "maxsearch too small"
    x0_coords = np.zeros((D,n),'f')
    x0_ixs = []
    x0_names = []
    ix = 0
    # sort names by key to ensure same ordering as generator variables
    for xname, xdom in sortedDictItems(subdomain,byvalue=False):
        if isinstance(xdom, (tuple, list)):
            x0_ixs.append(var_ix_map[xname])
            x0_names.append(xname)
            x0_coords[ix,:] = linspace(xdom[0], xdom[1], n)
            ix += 1
    # NOTE: def Rhs(self, t, xdict, pdict) and Jacobian signature
    # has same form, so need to use a wrapper function to convert order
    # of arguments to suit solver.
    #
    Rhs_wrap = make_RHS_wrap(gen, xdict, x0_names)
##    def Rhs_wrap(x, t, pdict):
##        xdict.update(dict(zip(x0_names, x)))
##        try:
##            return take(gen.Rhs(t, xdict, pdict), x0_ixs)
##        except (OverflowError, ValueError):
##            return array([1e4]*D)
    if gen.haveJacobian():
##        def Jac_wrap(x, t, pdict):
##            xdict.update(dict(zip(x0_names, x)))
##            try:
##                return take(take(gen.Jacobian(t, xdict, pdict), x0_ixs,0), x0_ixs,1)
##            except (OverflowError, ValueError):
##                # penalty
##                return array([[1e4]*D]*D)
        fprime = make_Jac_wrap(gen, xdict, x0_names)
    elif jac is not None:
        def Jac_wrap(x, t, pdict):
            xdict.update(dict(zip(x0_names, x)))
            argdict = filteredDict(xdict, jac._args)
            argdict['t'] = t
            try:
                return array(jac(**argdict))
            except (OverflowError, ValueError):
                # penalty
                return array([[1e4]*D]*D)
        fprime = Jac_wrap
    else:
        fprime = None
    # solve xdot on each starting point
    fps = []
    fp_listdict = []
    d_posns = base_n_counter(n,D)
    xtol = eps/10.
    for dummy_ix in xrange(n**D):
        x0 = array([x0_coords[i][d_posns[i]] for i in xrange(D)])
        res = fsolve(Rhs_wrap,x0,(t,gen.pars),xtol=xtol,
                          fprime=fprime,full_output=True)
        xinf_val = res[0]
        # treat f.p.s within epsilon (2-norm) of each other as identical
        if alltrue(isfinite(xinf_val)):
            if len(fps) == 0 or not sometrue([norm(fp-xinf_val)<eps for fp in fps]):
                ok = res[2]==1
                # check that bounds were met
                if ok:
                    for ix, xname in enumerate(x0_names):
                        ok = ok and \
                          gen.variables[xname].depdomain.contains(xinf_val[ix]) is not notcontained
                if ok:
                    fps.append(xinf_val)
                    fp_pt = dict(zip(gen._FScompatibleNamesInv(x0_names),
                                                xinf_val))
                    fp_pt.update(fixed_vars)
                    fp_listdict.append(fp_pt)
        d_posns.inc()
    return tuple(fp_listdict)


# convenient aliases
find_steadystates = find_equilibria = find_fixedpoints


class Point2D(Point):
    """Convenience sub-class of PyDSTool.Point for 2D Euclidean points.
    If initialized using a dictionary, xname and yname optional arguments
    must specify which coordinate is associated with 'x' and 'y' axes.

    Main advantage is lower overhead of initialization and the more convenient
    notation point.x and point.y for the two coordinates, that may
    nonetheless be given optional actual names (xname and yname). Note that
    Python's "dot-based" attribute lookup is only valid for 'x' and 'y' fields.

    Optional norm order and labels argument still permitted, as for original
    Point. For 2D Pointsets, use original PyDSTool.Pointset class.
    """
    def __init__(self, x, y=None, xname='x', yname='y', norm=2, labels=None):
        """If y is None, expects 2D array or dictionary in x"""
        if y is None:
            try:
                # array x?
                self.x = float(x[0])
                self.y = float(x[1])
            except (TypeError, KeyError):
                # dictionary x?
                self.x = float(x[xname])
                self.y = float(x[yname])
        else:
            self.x = float(x)
            self.y = float(y)
        self.xname = xname
        self.yname = yname
        self._parameterized = False
        if labels is None:
            self.labels = {}
        else:
            self.addlabel(labels)
        self._normord = norm
        self.coordtype = float
        self.dimension = 2
        self.coordarray = None  # Not implemented this way

    def mapNames(self, themap):
        self.xname = themap(self.xname)
        self.yname = themap(self.yname)
        self.labels = mapNames(themap, self.labels)

    def __contains__(self, coord):
        return coord in (self.xname, self.yname)

    def todict(self, aslist=False):
        """Convert Point2D to a dictionary of array values (or of list with aslist=True)."""
        if aslist:
            return {self.xname: [self.x],
                    self.yname: [self.y]}
        else:
            return {self.xname: self.x,
                    self.yname: self.y}

    def toarray(self):
        return np.array( (self.x, self.y) )

    def toPoint(self):
        """Coerce to regular PyDSTool.Point object"""
        return Point(coorddict={xname: self.x, yname: self.y},
                     coordtype=float,
                     norm=self._normord,
                     labels=self.labels)

    def get(self, coord, d=None):
        if coord == self.xname:
            return self.x
        elif coord == self.yname:
            return self.y
        else:
            return d

    def __len__(self):
        return 2

    def __abs__(self):
        return np.linalg.norm( (self.x, self.y), self._normord)

    def update(self, d):
        self.x = d[self.xname]
        self.y = d[self.yname]

    def items(self):
        return ((self.xname, self.x), (self.yname, self.y))

    def iteritems(self):
        return iter(((self.xname, self.x), (self.yname, self.y)))

    def values(self):
        return [self.x, self.y]

    def itervalues(self):
        return iter([self.x, self.y])

    def keys(self):
        return [self.xname, self.yname]

    def iterkeys(self):
        return iter([self.xname, self.yname])

    def has_key(self, k):
        return k in (self.xname, self.yname)

    def __getitem__(self, ix):
        if ix == 0:
            return self.x
        elif ix == 1:
            return self.y
        elif isinstance(ix, slice):
            return (self.x, self.y)
        else:
            raise StopIteration #IndexError("Index out of range")

    __call__ = __getitem__

    def __setitem__(self, ix, val):
        if ix == 0:
            self.x = val
        elif ix == 1:
            self.y = val
        else:
            raise IndexError("Index out of range")

    def __delitem__(self, k):
        raise NotImplementedError

    def __str__(self):
        return "Point2D( %f, %f )" % (self.x, self.y)

    def __repr__(self):
        if self.labels == {}:
            labstr = 'no labels'
        else:
            labstr = 'labels'
        return "Point2D( %f, %f, %s, %s, %i, %s )" % (self.x, self.y,
                            self.xname, self.yname, self._normord, labstr)

    def info(self, verboselevel=1):
        if verboselevel == 1:
            print self.__str__()
        elif verboselvel > 1:
            print self.__repr__()

    def __copy__(self):
        return Point2D(self.x, self.y, self.xname, self.yname, self._normord,
                       self.labels)

    def __add__(self, other):
        try:
            return Point2D( self.x + other[0], self.y + other[1],
                            self.xname, self.yname, self._normord, self.labels)
        except TypeError:
            return Point2D( self.x + other, self.y + other,
                            self.xname, self.yname, self._normord, self.labels )

    def __sub__(self, other):
        try:
            return Point2D( self.x - other[0], self.y - other[1],
                            self.xname, self.yname, self._normord, self.labels )
        except TypeError:
            return Point2D( self.x - other, self.y - other,
                            self.xname, self.yname, self._normord, self.labels )

    __radd__ = __add__

    def __rsub__(self, other):
        try:
            return Point2D( other[0] - self.x, other[1] - self.y,
                            self.xname, self.yname, self._normord, self.labels )
        except TypeError:
            return Point2D( other - self.x, other - self.y,
                            self.xname, self.yname, self._normord, self.labels )

    def __mul__(self, c):
        # scalar
        return Point2D( self.x * c, self.y * c,
                            self.xname, self.yname, self._normord, self.labels)

    def __div__(self, c):
        # scalar
        return Point2D( self.x / float(c), self.y / float(c),
                            self.xname, self.yname, self._normord, self.labels )

    def __rmul__(self, c):
        # scalar
        return Point2D( self.x * c, self.y * c,
                            self.xname, self.yname, self._normord, self.labels)

    def __rdiv__(self, c):
        # scalar
        return Point2D( float(c) / self.x, float(c) / self.y,
                            self.xname, self.yname, self._normord, self.labels )

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __neg__(self):
        return Point2D( -self.x, -self.y,
                            self.xname, self.yname, self._normord, self.labels )

    def __pow__(self, other):
        return Point2D( self.x ** other, self.y ** other,
                            self.xname, self.yname, self._normord, self.labels )

    def __eq__(self, other):
        return self.x == other[0] and self.y == other[1]

    def __lt__(self, other):
        try:
            return np.linalg.norm( (self.x, self.y), self._normord ) < \
                   np.linalg.norm( (other.x, other.y), self._normord )
        except AttributeError:
            # scalar or array
            return np.linalg.norm( (self.x, self.y), self._normord ) < \
                   np.linalg.norm( other, self._normord )

    def __gt__(self, other):
        try:
            return np.linalg.norm( (self.x, self.y), self._normord ) > \
                   np.linalg.norm( (other.x, other.y), self._normord )
        except AttributeError:
            # scalar or array
            return np.linalg.norm( (self.x, self.y), self._normord ) > \
                   np.linalg.norm( other, self._normord )

    def __le__(self, other):
        try:
            return np.linalg.norm( (self.x, self.y), self._normord ) <= \
                   np.linalg.norm( (other.x, other.y), self._normord )
        except AttributeError:
            # scalar or array
            return np.linalg.norm( (self.x, self.y), self._normord ) <= \
                   np.linalg.norm( other, self._normord )

    def __ge__(self, other):
        try:
            return np.linalg.norm( (self.x, self.y), self._normord ) >= \
                   np.linalg.norm( (other.x, other.y), self._normord )
        except AttributeError:
            # scalar or array
            return np.linalg.norm( (self.x, self.y), self._normord ) >= \
                   np.linalg.norm( other, self._normord )


class nullcline(object):
    """Nullcline representation class in 2D (x, y) plane parameterizable
    by x variable only. A third-order univariate spline will be fitted to
    the given sample point array using all as knots, and available
    through the 'spline' attribute.

    Input:
      Names are given to the x and y axis directions at initialization.
      Third argument is the N-by-2 array of N two-dimensional data points.

    N.B.: Only works for nullclines that can be written y = f(x). Future
    versions may support non-functions.

    Some functions acting on nullclines currently also rely on monotonicity
    of y=f(x), and will verify this using the is_monotonic method before
    proceeding. Future versions will attempt to resolve that.
    """
    def __init__(self, xname, yname, nullc_array):
        self.xname = xname
        self.yname = yname
        self.array = nullc_array
        # ensure monotonicity
        if not isincreasing(nullc_array[:,0]):
            raise AssertionError("x axis '%s' values must be monotonically increasing" % xname)
        self.spline = InterpolatedUnivariateSpline(nullc_array[:,0],
                                                   nullc_array[:,1])

    def __call__(self, x):
        """Returns a scalar if x is scalar."""
        if isinstance(x, _num_types):
            return self.spline(x)[0]
        else:
            return self.spline(x)

    def createParameterization(self):
        """Creates a (non-arc-length) parameterization"""
        raise NotImplementedError

    def toPointset(self, with_param=False):
        if with_param:
            self.createParameterization()
            #return arrayToPointset(x_vals, y_vals, self.xname, self.yname, params, 's')
        else:
            return Pointset({self.xname: self.array[:,0],
                             self.yname: self.array[:,1]})

    def tgt_vec(self, x):
        """Return tangent vector to an interior point of nullcline,
        normalized to length 1."""
        try:
            tgt = np.array((1,self.spline.derivatives(x)[1]))
        except AssertionError:
            raise ValueError("Derivative not available for endpoints or outside of spline domain")
        return tgt/np.linalg.norm(tgt)

    def curvature(self, x):
        """Signed curvature at x is returned (scalar)"""
        ders = self.spline.derivatives(x)
        return ders[2]/pow(1+ders[1]*ders[1],1.5)

    def curvature_at_sample_points(self):
        """Returns array of signed scalars for each
        x data point (knot) of the nullcline spline.

        Positive values mean concave "up" in the plane.
        Assumes nullcline is monotonic."""
        xdata = self.array[:,0]
        curvature = []
        curve_fn = self.curvature
        # interior points only in loop --
        # end points are checked just inside boundaries
        # separately
        x0 = xdata[0] + 0.01*(xdata[1] - xdata[0])
        curvature.append( curve_fn(x0) )
        for x in xdata[1:-1]:
            curvature.append( curve_fn(x) )
        x1 = xdata[-1] - 0.01*(xdata[-1] - xdata[-2])
        curvature.append( curve_fn(x1) )
        return np.array(curvature)

    def grad_curvature(self, x):
        """Derivative of the curvature at x is returned (scalar)"""
        ders = self.spline.derivatives(x)
        denom1 = pow(1+ders[1]*ders[1],1.5)
        denom2 = pow(1+ders[1]*ders[1],2.5)
        return ders[3]/denom1 - 3*(ders[2]*ders[2]*ders[1])/denom2

    def grad_curvature_at_sample_points(self):
        """Returns array of signed scalars for each
        x data point (knot) of the nullcline spline.

        Assumes nullcline is monotonic."""
        xdata = self.array[:,0]
        gcurvature = []
        gcurve_fn = self.grad_curvature
        # interior points only in loop --
        # end points are checked just inside boundaries
        # separately
        x0 = xdata[0] + 0.01*(xdata[1] - xdata[0])
        gcurvature.append( gcurve_fn(x0) )
        for x in xdata[1:-1]:
            gcurvature.append( gcurve_fn(x) )
        x1 = xdata[-1] - 0.01*(xdata[-1] - xdata[-2])
        gcurvature.append( gcurve_fn(x1) )
        return np.array(gcurvature)

    def concavity(self, x):
        """Concavity scalar +/- 1 or 0 at x is returned.

        Positive values mean concave "up" in the plane."""
        ders = self.spline.derivatives(x)
        return np.sign( ders[2]/pow(1+ders[1]*ders[1],1.5) )

    def concavity_at_sample_points(self):
        """Returns array of +/- 1 or 0 scalars for each
        x data point (knot) of the nullcline spline.

        Positive values mean concave "up" in the plane.
        Assumes nullcline is monotonic."""
        xdata = self.array[:,0]
        concavity = []
        # interior points only in loop --
        # end points are checked just inside boundaries
        # separately
        conc_fn = self.concavity
        x0 = xdata[0] + 0.01*(xdata[1] - xdata[0])
        concavity.append( conc_fn(x0) )
        for x in xdata[1:-1]:
            concavity.append( conc_fn(x) )
        x1 = xdata[-1] - 0.01*(xdata[-1] - xdata[-2])
        concavity.append( conc_fn(x1) )
        return np.array(concavity)

    def crop(self, xdom, ydom, include_resamples=True):
        """Returns a new Nullcline object cropped to the (x, y) domain
        given by pairs xdom and ydom.

        If include_resamples = True (default True), points at the domain
        ends and additional sample points will be included in the new nullcline
        object regardless of whether they were in the original sample point set
        for the cropped nullcline.

        Given the assumption that nullclines are montonic functions of x,
        this guarantees a unique, contiguous piece of the nullcline is returned.
        """
        xinterval = Interval('xdom', float, xdom)
        yinterval = Interval('ydom', float, ydom)
        sample_vals = list(crop_2D(self.array, xinterval, yinterval))
        if include_resamples:
            width = xdom[1] - xdom[0]
            new_xs = xinterval.uniformSample(dt=width*0.1)
            tol = width*0.05
            for x in new_xs:
                # exclude any that are within 5% of domain extent of
                # existing sample points
                if np.all(abs(x - self.array[:,0])>tol):
                    sample_vals.append((x,self(x)))
            sample_vals = np.array(sample_vals)
            ixs = argsort(sample_vals[:,0])  # sort by x
            sample_vals = sample_vals[ixs]
        try:
            return nullcline(self.xname, self.yname, np.array(sample_vals))
        except:
            print "Error cropping nullcline at sample points", sample_vals
            print "MAYBE TOO FEW VALUES SAMPLED: number was", len(sample_vals)
            raise

    def is_monotonic(self):
        is_y_inc = isincreasing(self.array[:,1])
        is_y_dec = isincreasing(self.array[::-1,1])
        return (is_y_inc or is_y_dec)

    def __copy__(self):
        return nullcline(self.xname, self.yname, self.array)


def get_perp(v):
    """Find perpendicular vector in 2D (assumes 2D input)"""
    vperp=v.copy()  # ensures correct return type
    vperp[0] = v[1]
    vperp[1] = -v[0]
    return vperp

def get_orthonormal(v):
    """Returns orthonormal vector in 2D (assumes 2D input)"""
    vperp=v.copy()  # ensures correct return type
    vperp[0] = v[1]
    vperp[1] = -v[0]
    return vperp/np.linalg.norm(vperp)

def get_rotated(x, theta):
    res = copy.copy(x) # ensure same type of result as input
    z=(x[0]+x[1]*1j)*(cos(theta)+sin(theta)*1j)
    res[0] = z.real
    res[1] = z.imag
    return res

def filter_close_points(pts, eps, normord=2):
    """Remove points from iterable pts (e.g. array or Pointset)
    according to whether they are closer than epsilon to each other
    in the given norm.
    """
    # start with all indices, and remove those that are unwanted
    remaining = range(len(pts))
    for i, p in enumerate(pts):
        for j in range(i+1, len(pts)):
            if norm(p-pts[j],normord) < eps:
                try:
                    remaining.remove(j)
                except ValueError:
                    # already removed
                    pass
    return pts[remaining]


def filter_NaN(pts):
    """Filter out any points containing NaNs.
    Expects array input.
    """
    return array([p for p in pts if all(isfinite(p))])

def check_bounds(sol_array, xinterval, yinterval):
    # x is assumed to be the dynamic variable
    return alltrue([xinterval.contains(p[0]) is not notcontained and \
                    yinterval.contains(p[1]) is not notcontained for p in sol_array])

def crop_2D(sol_array, xinterval, yinterval):
    """Filter out points that are outside the domains given by xdom, ydom.
    Expects array input
    """
    return array([p for p in sol_array if xinterval.contains(p[0]) is not notcontained \
              and yinterval.contains(p[1]) is not notcontained])

def make_vec_at_A_face_B(vec_at_a, Ay, By):
    """Assumes monotonicity of functions on which points A and B lie,
    using their y components as arguments."""
    up = np.array( (0.0,1.0) )
    orient = 1   # default value
    if Ay > By:
        # A is above B so vec must orient "down"
        if np.dot( up, vec_at_a ) > 0:
            # vec is up
            orient = -1
    else:
        # B is above A so vec must orient "up"
        if np.dot( up, vec_at_a ) < 0:
            # vec is down
            orient = -1
    return orient * vec_at_a


def angle_to_vertical(v):
    """Return an angle between 0 and 2*pi measured clockwise from vertical."""
    up = np.array((0.,1.))
    theta = np.arccos(np.dot(up, v)/np.linalg.norm(v))
    if np.cross(up, v) > 0:
        # angle will be greater than pi
        return 2*np.pi - theta
    else:
        return theta


# not currently used, but helpful if lines are not arranged correctly for intersection test
def is_counter_clockwise(A,B,C):
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)


def determine_if_intersect(A,B,C,D):
    ccw = is_counter_clockwise
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def line_intersection(A, B, C, D):
    """Assumes line segments AB vs. CD actually cross (must determine this
    separately), where points A - D have fields 'x' and 'y' e.g. a Point2D
    object.

    Uses algorithm from:
    http://www.bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    and
    http://paulbourke.net/geometry/lineline2d/

    In particular, this code assumes lines are not parallel or collinear.
    """
    ua =  ( (D.x-C.x)*(A.y-C.y)-(D.y-C.y)*(A.x-C.x) ) / \
               ( (D.y-C.y)*(B.x-A.x) - (D.x-C.x)*(B.y-A.y) )
    return Point2D( A.x+(ua*(B.x-A.x)), A.y+(ua*(B.y-A.y)) )


class fixedpoint_nD(object):
    _classifcations = ('spiral', 'node', 'saddle')
    _stability = ('s','c','u')

    def __init__(self, gen, pt, coords=None, jac=None, description='',
                 normord=2, eps=1e-12):
        """pt must have same dimension as generator, but if a sub-system
        is being analyzed then specify the sub-system's variables using
        the coords argument.
        """
        if not isinstance(pt, Point):
            raise PyDSTool_TypeError("Fixed point must be a Point type")
        if coords is None:
            self.dimension = gen.dimension
            self.fp_coords = gen.funcspec.vars
        else:
            self.dimension = len(coords)
            coords.sort()
            self.fp_coords = coords
        self.eps = eps
        self.gen = gen
        if jac is None:
            try:
                gen.Jacobian(0, gen.initialconditions)
            except PyDSTool_ExistError:
                # no full Jac available
                raise NotImplementedError()
            else:
                # in the future, make a jac function that extracts
                # the n components from Jacobian (if n < gen.dimension)
                raise NotImplementedError()
        self.jac = jac
        jac_test_arg = filteredDict(pt, self.fp_coords)
        jac_test_arg['t'] = 0
        try:
            self.D = asarray(jac(**jac_test_arg))
        except:
            # placeholder
            raise
        else:
            assert self.D.shape == (self.dimension, self.dimension)
        assert normord > 0
        self.normord = normord
        assert pt._normord == normord, "Mismatching norm order for point"
        self._get_eigen()
        evals = self.evals
        evecs = self.evecs
        if self.dimension < gen.dimension:
            subs_str = "sub-"
        else:
            subs_str = "full "
        if self.dimension != len(evals):
            raise ValueError("Dimension of %ssystem must equal number of eigenvalues" % subs_str)
        if self.dimension != len(evecs):
            raise ValueError("Dimension of %ssystem must equal number of eigenvectors" % subs_str)
        if gen.dimension != len(pt):
            raise ValueError("Dimension of full system must equal dimension of fixed point")
        self.point = pt
        # assume autonomous system
        var_ixs = []
        for v in self.fp_coords:
            var_ixs.append(gen.funcspec.vars.index(v))
        fp_evaluated = array(gen.Rhs(0, pt, gen.pars))[var_ixs]
        if sometrue([abs(fp_i) > eps for fp_i in fp_evaluated]):
            raise ValueError("Given point is not a fixed point of the system at given tolerance")
        self.coordnames = pt.coordnames
        self._classify()
        self.description = description

    def _get_eigen(self):
        evals, evecs_array = np.linalg.eig(self.D)
        evecs_pt = []
        for i, v in enumerate(self.fp_coords):
            d = {}
            for j, w in enumerate(self.fp_coords):
                # funky indexing is to recover the float value
                # from a 0-dimensional array (e.g. array(0.5))
                # that has no length
                d[w] = real(evecs_array.T[i][j])[()]
            evecs_pt.append(Point(d))
        self.evals = evals
        evecs_normed = []
        for ev in evecs_pt:
            assert ev._normord == self.normord, "Mismatching norm order for eigenvector"
            n = norm(ev, self.normord)
            if n != 1:
                ev = ev/n
            evecs_normed.append(ev)
        self.evecs = tuple(evecs_normed)

    def __getitem__(self, k):
        return self.point[k]

    def __setitem__(self, k, v):
        self.point[k] = v

    def _classify(self):
        if self.dimension == 2:
            print "Use fixedpoint_2D class"
        raise NotImplementedError("Not implemented for n dimensions where n != 2")


class fixedpoint_2D(fixedpoint_nD):
    def __init__(self, gen, pt, coords=None, jac=None, description='', normord=2, eps=1e-12):
        fixedpoint_nD.__init__(self, gen, pt, coords, jac, description=description,
                               normord=normord, eps=eps)
        if self.dimension != 2:
            raise TypeError("This class is for 2D systems only")

    def _classify(self):
        real_evals = (isreal(self.evals[0]), isreal(self.evals[1]))
        equal_evals = abs(self.evals[0] - self.evals[1]) < self.eps
        zero_evals = (abs(self.evals[0]) < self.eps,
                      abs(self.evals[1]) < self.eps)
        if alltrue(real_evals):
            sign_evals = (sign(self.evals[0]), sign(self.evals[1]))
            if sign_evals[0] == sign_evals[1]:
                self.classification = 'node'
            else:
                self.classification = 'saddle'
        else:
            self.classification = 'spiral'
        real_parts = real(self.evals)
        if alltrue(real_parts<0):
            self.stability = 's'
        elif alltrue(real_parts==0):
            self.stability = 'c'
        else:
            self.stability = 'u'
        self.degenerate = sometrue(zero_evals) or equal_evals


def make_distance_to_known_line_auxfn(fname, p, dp=None, q=None):
    """Builds definition of an auxiliary function of (x,y) measuring
    (signed) distance from a 2D line given by the point p and either
    (a) the vector dp, or (b) the point q. (All inputs must be Points.)
    """
    if dp is None:
        assert q is not None, "Only specify dp or q, not both"
        assert len(p)==len(q)==2
        other = q
    else:
        assert q is None, "Only specify dp or q, not both"
        assert len(p)==len(dp)==2
        other = dp
    try:
        assert p.coordnames == other.coordnames
    except AttributeError:
        raise TypeError("Only pass Point objects as inputs")
    if dp is None:
        q0_m_p0 = q[0]-p[0]
        q1_m_p1 = q[1]-p[1]
    else:
        # based on idea that q - p = dp
        # to prevent loss of precision
        q0_m_p0 = dp[0]
        q1_m_p1 = dp[1]
    denom = sqrt(q0_m_p0*q0_m_p0 + q1_m_p1*q1_m_p1)
    term = q0_m_p0*p[1]-q1_m_p1*p[0]
    if denom == 1.0:
        d = '%s-%s*y+%s*x'%(repr(term), repr(q0_m_p0), repr(q1_m_p1))
    else:
        d = '(%s-%s*y+%s*x)/%s'%(repr(term), repr(q0_m_p0),
                                 repr(q1_m_p1), repr(denom))
    return {fname: (['x','y'], d)}


def make_distance_to_line_auxfn(linename, fname, p, by_vector_dp=True):
    """Builds definition of an auxiliary function of (x,y) measuring
    (signed) distance from a 2D line given at run-time by coordinate
    names in the tuple of strings p and either a vector dp,
    or a point q, depending on the second input argument.
    Also returns list of parameter names used.
    """
    assert len(p)==2 and isinstance(p[0], str) and isinstance(p[1], str)
    p0 = linename+'_p_'+p[0]
    p1 = linename+'_p_'+p[1]
    pars = [p0, p1]
    if by_vector_dp:
        # based on idea that q - p = dp
        # to prevent loss of precision
        dp0 = linename+'_dp_'+p[0]
        dp1 = linename+'_dp_'+p[1]
        pars.extend([dp0, dp1])
        q0_m_p0 = dp0
        q1_m_p1 = dp1
    else:
        q0 = linename+'_q_'+p[0]
        q1 = linename+'_q_'+p[1]
        pars.extend([q0, q1])
        q0_m_p0 = '(%s-%s)'%(q0,p0)
        q1_m_p1 = '(%s-%s)'%(q1,p1)
    denom = 'sqrt(%s*%s+%s*%s)'%(q0_m_p0,q0_m_p0,q1_m_p1,q1_m_p1)
    p1_m_y = '(%s-y)'%(p1)
    p0_m_x = '(%s-x)'%(p0)
    d = '(%s*%s-%s*%s)/%s'%(q0_m_p0, p1_m_y, p0_m_x, q1_m_p1, denom)
    return {'pars': pars, 'auxfn': {fname: (['x','y'], d)}}


def bisection(func, a, b, args=(), xtol=1e-10, maxiter=400, normord=2):
    """Bisection root-finding method.  Given a function returning +/- 1
    and an interval with func(a) * func(b) < 0, find the root between a and b.

    Variant of scipy.optimize.minpack.bisection with exception for too many iterations
    """
    eva = func(a,*args)
    evb = func(b,*args)
    assert (eva*evb < 0), "Must start with interval with func(a) * func(b) <0"
    i = 1
    while i<=maxiter:
        dist = (b-a)/2.0
        p = a + dist
        # in case dist is a scalar
        if norm(np.asarray(dist).flatten(),normord) < xtol:
            return p
        try:
            ev = func(p,*args)
        except RuntimeError:
            return p
        if ev == 0:
            return p
        i += 1
        if ev*eva > 0:
            a = p
            eva = ev
        else:
            b = p
    raise RuntimeError("Method failed after %d iterations." % maxiter)




# ------


class plotter_2D(object):
    """Plotting manager for phase plane analysis. A global instance of
    this class called 'plotter' is already created and exported by this
    module. Operate on that instance directly.

    All plotting will take place in figure numbered self.curr_fig, which
    will be resized to the given shape and limits.

    All added methods of form plot_X take an optional named figure which would
    correspond to whatever figure number is allocated in the fig_directory
    dictionary attribute. New plot methods of this kind must first check
    that do_display is True, and then call the 'setup' method before, and
    'teardown' after doing the necessary plotting.

    """
    def __init__(self, x_dom=None, y_dom=None, figsize=(6,5),
                 curr_fig=1, do_display=False):
        self.x_dom = x_dom
        self.y_dom = y_dom
        self.do_display = do_display
        self.figsize = figsize
        self.curr_fig = 1
        self.fig_directory = {}
        # layers will permit types of plot data to be removed or restored
        # to a given plot axes without recomputation
        self.layers = {}

    def set_curr_fig(self, name):
        try:
            self.curr_fig = self.fig_directory[name]
        except KeyError:
            raise ValueError("Create entry in fig_directory attribute for this figure name")

    def setup(self, figname):
        if figname is not None:
            self.set_curr_fig(figname)
        pp.figure(self.curr_fig, figsize=self.figsize)

    def teardown(self):
        pp.figure(self.curr_fig)
        if self.x_dom is not None:
            pp.xlim( self.x_dom )
        if self.y_dom is not None:
            pp.ylim( self.y_dom )
        pp.draw()

    def plot_nullcline(self, nullc, style, lw=1, N=100, figname=None):
        if not self.do_display:
            return
        self.setup(figname)
        x_data = nullc.array[:,0]
        y_data = nullc.array[:,1]
        xs = np.sort( np.concatenate( (np.linspace(x_data[0], x_data[-1], N), x_data) ) )
        ys = nullc.spline(xs)
        pp.plot(xs, ys, style, linewidth=lw)
        pp.plot(x_data, y_data, style[0]+'o')
        #pars = np.sort( np.concatenate( (np.linspace(0, 1, N), nullc.param) ) )
        #pp.plot(nullc.splinex(pars), nullc.spliney(pars), 'b')
        self.teardown()

    def plot_line_from_points(self, A, B, style, lw=1, figname=None):
        if not self.do_display:
            return
        self.setup(figname)
        pp.plot([A[0], B[0]], [A[1], B[1]], style, linewidth=lw)
        self.teardown()

    def plot_point(self, A, style, figname=None):
        if not self.do_display:
            return
        self.setup(figname)
        pp.plot(A[0], A[1], style)
        self.teardown()

# global instance
plotter = plotter_2D()

def _newton_step(Q0, nullc, A, B, phi, tol):
    """Internal function for line intersection involving splines"""
    global plotter
    # Still need to choose the 0.2 better to ensure correct scaling with A and B

    err = 1000*tol
    Q = Q0
    while err > tol:
        # Q0 prime is a distant point on the spline's tangent line at Q0
        Q_prime = Q + nullc.tgt_vec(Q.x)
        plotter.plot_line_from_points(Q, Q_prime, 'b')
        P1 = line_intersection(Q, Q_prime, A, B)
        Q1 = Point2D(P1.x, nullc.spline(P1.x))  # project onto spline
        plotter.plot_point(Q1, 'ko')
        err = abs(phi-angle_to_vertical(Q1-A))
        Q = Q1
    return Q


def find_nearest_sample_points_by_angle(thetas, phi, return_NaN=False):
    """Returns two successive indices of the angles in the array argument
    thetas, for which theta values straddle angle phi.

    This function is useful if phi represents the angle from the vertical of
    the normal to a curve's tangent line, and thetas are angles from the
    vertical of another curve in the plane. This function can therefore help
    find closest or furthest perpendicular distances between curves.

    Assumes thetas are in increasing order, and that phi is properly contained
      between min(thetas) and max(thetas) (otherwise raises a ValueError).

    For the corner case where one difference between angles is exactly 0 (not
      generic), one of the thetas values given by the returned indices is
      exactly phi."""
    global plotter
    pos_diffs = (thetas - phi) > 0
    if np.all(pos_diffs) or not np.any(pos_diffs):
        if return_NaN:
            return np.NaN
        else:
            raise ValueError("Outside domain")
    else:
        if pos_diffs[0]:
            # Trues then Falses in pos_diffs
            # first_neg_ix guaranteed not to be 0
            first_neg_ix = np.argmin(np.asarray(pos_diffs, int))
            straddling_ixs = (first_neg_ix-1, first_neg_ix)
        else:
            # Falses then Trues in pos_diffs
            first_pos_ix = np.argmax(np.asarray(pos_diffs, int))
            straddling_ixs = (first_pos_ix-1, first_pos_ix)
        return straddling_ixs


def closest_perp_distance_between_sample_points(NullcA, NullcB, xa, x0B, x1B,
                                                newt_tol=1e-6):
    """Closest perpendicular distance from (xa, ya) on Nullcline A to Nullcline B,
    given that it's known to be between sample points x0B and x1B on Nullcline B.

    Uses a Newton-like step, solving up to an angular tolerance given by newt_tol.
    """
    global plotter
    ya = NullcA(xa)
    a = np.array([xa,ya])
    tgt_vec_at_a = NullcA.tgt_vec(xa)
    # normal has length 1
    normal_at_a = make_vec_at_A_face_B(get_orthonormal(tgt_vec_at_a),
                                ya, NullcB(xa))
    plotter.plot_line_from_points(a, a+tgt_vec_at_a, 'k:')
    phi = angle_to_vertical(normal_at_a)

    A = Point2D(a)
    B = Point2D(a + normal_at_a)
    C = Point2D(x0B, NullcB(x0B))
    D = Point2D(x1B, NullcB(x1B))

    P0 = line_intersection(A, B, C, D)
    Q0 = Point2D(P0.x, NullcB(P0.x))  # project onto spline
    theta0 = angle_to_vertical(Q0-a)

    plotter.plot_line_from_points(A, B, 'k-')
    plotter.plot_line_from_points(C, D, 'r--')
    plotter.plot_point(P0, 'gs')
    plotter.plot_point(Q0, 'ys')

    try:
        Q = _newton_step(Q0, NullcB, A, B, phi, newt_tol)
    except ValueError:
        return np.Inf
    else:
        vec = Q-A
        #vec[0] = vec[0] / 100.  # hack to investigate differing x and y scales
        return np.linalg.norm(vec)


def closest_perp_distance_on_spline(NullcA, NullcB, xa):
    """at point given by xa on Nullcline A"""
    global plotter
    A = Point2D(xa, NullcA(xa))
    a_to_NullcB = NullcB.array - NullcA.array[1]
    thetas_B = np.apply_along_axis(angle_to_vertical, -1, a_to_NullcB)
    # monotonicity assumptions ensures that y coord of other nullcline
    # indicates which direction normal vector must face
    # ... also assumes that nullcline B is defined for all x values that
    # nullcline A is
    norm_vec_at_a = make_vec_at_A_face_B(get_orthonormal(NullcA.tgt_vec(xa)),
                                         A.y, NullcB(xa))
    phi_a = angle_to_vertical(norm_vec_at_a)
    plotter.plot_line_from_points(A, A + norm_vec_at_a, 'k-.')

    # determine closest sample points
    try:
        ix_B_lo, ix_B_hi = find_nearest_sample_points_by_angle(thetas_B, phi_a)
    except ValueError:
        # Outside domain of other nullcline
        return np.Inf
    else:
        if ix_B_lo == ix_B_hi:
            ix_B_lo = np.max( [0, ix_B_lo-1] )
            ix_B_hi = np.min( [len(NullcB.array)-1, ix_B_hi+1] )
        x0B = NullcB.array[ix_B_lo,0]
        x1B = NullcB.array[ix_B_hi,0]

    # refine to closest point on spline of Nullcline B
    return closest_perp_distance_between_sample_points(NullcA, NullcB, xa, x0B, x1B)


def is_min_bracket(triple):
    """For a triple of positive numbers (a,b,c), returns a triple
    (boolean, index, error)

    where:
    boolean is True if a > b < c (this brackets a minimum value);

    index is 1 (middle) if input brackets a minimum value, otherwise
    index indicates which end of the bracket has the smallest value;

    error is the smallest absolute difference between distances in the
    triple (a, b, c)
    """
    a,b,c = triple
    if b < a and b < c:
        return True, 1, min( a-b, c-b )
    else:
        if a < c:
            # then monotonic a < b < c
            return False, 0, min( b-a, c-b )
        else:
            # a > c, monotonic a > b > c
            return False, 2, min( a-b, b-c )


def closest_perp_distance_between_splines(NullcA, NullcB, dist_delta_tol=1e-5):
    """Measure closest perpendicular distance between two nullcline objects
    (via their spline representations).
    Tolerance argument measures that of Euclidean distance between the splines
    representing the two nullclines.

    Assumes monotonicity of both nullclines, which may be relaxed in future
    versions.
    """
    assert NullcA.is_monotonic()
    assert NullcB.is_monotonic()
    # search interior sample points of spline A first
    dists = []
    xa_search_vals = NullcA.array[1:-1,0]
    for xa in xa_search_vals:
        dists.append( closest_perp_distance_on_spline(NullcA, NullcB, xa) )

    xa_ix = np.argmin(dists)  # used later too to recover associated xa value used
    xa_start_ix1 = xa_ix + 1 # ignored spline endpoint
    other_dists = dists[:]
    other_dists[xa_ix] = np.Inf
    if np.any( np.isfinite( np.array(other_dists) )):
        xa_start_ix2 = np.argmin(other_dists) + 1
        if xa_start_ix2 < xa_start_ix1:
            # switch
            xa_start_ix1, xa_start_ix2 = (xa_start_ix2, xa_start_ix1)
    else:
        # both ix1 and ix2 would be the same, so extend search over next
        # neighboring indices, if available
        xa_start_ix2 = xa_start_ix1 + 1
        xa_start_ix1 = xa_start_ix1 - 1

    # Minimization method from xa over given indices in NullcA.array[ix1:ix2,0]
    # except if either is an endpoint, in which case move in until distance to
    # NullcB becomes well-defined
    if xa_start_ix1 > 0:
        xa_lo = NullcA.array[xa_start_ix1,0]
    else:
        raise NotImplementedError

    if xa_start_ix2 < len(NullcA.array):
        xa_hi = NullcA.array[xa_start_ix2,0]
    else:
        raise NotImplementedError

    assert xa_hi > xa_lo

    # given sufficient number of sample points for nullclines...
    #   concavity test for NullcA: shortest dist either at endpoints (monotonic assumed),
    #   or in interior (quadratic-like minimization).

    xa_mid = xa_search_vals[xa_ix] # may be the same as one of the endpoints
    if xa_mid in (xa_lo, xa_hi):
        # xa_mid is the same as an endpoint
        # split the difference midway
        xa_mid = 0.5*(xa_lo+xa_hi)
        d_mid = closest_perp_distance_on_spline(NullcA, NullcB, xa_mid)
    else:
        d_mid = dists[xa_ix]

    # Begin bisection-like linearly-convergent minimization (with convergence
    # guarantee due to bracketing)
    x_bracket = (xa_lo, xa_mid, xa_hi)
    d_bracket = (closest_perp_distance_on_spline(NullcA, NullcB, xa_lo),
                 d_mid,
                 closest_perp_distance_on_spline(NullcA, NullcB, xa_hi)
                 )

    is_brack, ix_min, error = is_min_bracket(d_bracket)
    #plotter.do_display = True

    while error > dist_delta_tol:
        a, b, c = x_bracket
        if is_brack:
            b_prime = 0.5*(a + b) # pick side to the "left" (w.l.o.g.)
            d_new = closest_perp_distance_on_spline(NullcA, NullcB, b_prime)
            # test bracket 1 is xa_lo, xa_new, xa_mid
            test_bracket1 = (d_bracket[0], d_new, d_bracket[1])
            is_brack_test, ix_min, d_error = is_min_bracket(test_bracket1)
            if is_brack_test:
                # done
                x_bracket = (a, b_prime, b)
                d_bracket = test_bracket1
                is_brack = True
                error = d_error
                continue # while
            # test bracket 2 is xa_new, xa_mid, xa_hi
            test_bracket2 = (d_new, d_bracket[1], d_bracket[2])
            is_brack_test, ix_min, d_error = is_min_bracket(test_bracket2)
            if is_brack_test:
                # done
                x_bracket = (b_prime, b, c)
                d_bracket = test_bracket2
                is_brack = True
                error = d_error
                # continue while
            else:
                raise AssertionError("huh?")
        else:
            # true d function may be monotonic, but must try to find a better
            # intermediate point that creates a bracket.

            # assuming minimum exists for d in this range,
            # pick a point between the two smallest d's and continue search for bracket.
            if ix_min == 0:
                b_prime = 0.5*(a + b)
                d_new = closest_perp_distance_on_spline(NullcA, NullcB, b_prime)
                # does the new selection create a bracket?
                # test bracket 1 is xa_lo, xa_new, xa_mid
                test_bracket = (d_bracket[0], d_new, d_bracket[1])
            else:
                # ix_min == 2
                b_prime = 0.5*(b + c)
                d_new = closest_perp_distance_on_spline(NullcA, NullcB, b_prime)
                # does the new selection create a bracket?
                # test bracket 1 is xa_mid, xa_new, xa_hi
                test_bracket = (d_bracket[1], d_new, d_bracket[2])
            is_brack_test, ix_min, error = is_min_bracket(test_bracket)
    return x_bracket[ix_min], d_bracket[ix_min]


# EXPERIMENTAL CLASS
class dx_scaled_2D(object):
    """Supports a delta x vector that automatically re-scales
    according to the known scalings of each of the vector's component
    directions.

    Highly experimental class!
    """

    def __init__(self, dx, scale=(1,1)):
        assert isreal(dx) # and dx>0
        self.dx = dx
        assert len(scale)==2 and scale[0] > 0 and scale[1] > 0
        m = max(scale)
        self.x_scale = scale[0]/m
        self.y_scale = scale[1]*1./scale[0]/m

    def __call__(self, dir):
        # angle is between 0 and 1: 0 = entirely in x-axis, 1 = entirely in y-axis
        # angle gives relative y-axis contribution of dir vector
        try:
            angle = 2*atan2(dir[1],dir[0])/pi
        except TypeError:
            # e.g. unsubscriptable object, from a __mul__ call
            return self.dx*dir
        else:
            return self.dx*dir*(angle*self.y_scale+(1-angle))

    __mul__ = __call__

    def __rmul__(self, dir):
        # angle is between 0 and 1: 0 = entirely in x-axis, 1 = entirely in y-axis
        # angle gives relative y-axis contribution of dir vector
        try:
            angle = 2*atan2(dir[1],dir[0])/pi
        except TypeError:
            # e.g. unsubscriptable object
            return dx_scaled_2D(self.dx*dir, (1,self.y_scale))
        else:
            return self.dx*dir*(angle*self.y_scale+(1-angle))


def find_saddle_manifolds(fp, dx=None, dx_gamma=None, dx_perp=None, tmax=None,
                          max_len=None, ic=None,
                          ic_dx=None, max_pts=1000, directions=(1,-1),
                          which=('s', 'u'), other_pts=None, rel_scale=None,
                          dx_perp_fac=0.75, verboselevel=0, fignum=None):
    """Compute any branch of the stable or unstable sub-manifolds of a saddle.
    Accepts fixed point instances of class fixedpoint_2D.

    Required inputs:
      fp:       fixed point object
      dx:       arc-length step size (**fixed**)
      dx_gamma: determines the positions of the Gamma_plus and Gamma_minus
                event surfaces (can be a real scalar or a pair if not symmetric)
      dx_perp:  initial perturbation from the local linear sub-manifolds to
                find starting points.
      tmax:     maximum time to compute a trajectory before 'failing'
      max_len:  maximum arc length to compute
      max_pts:  maximum number of points to compute on each sub-manifold branch
      Specify either ic or ic_dx for initial point (e.g. to restart the calc
        after a previous failure) or a certain distance from the saddle point.

    Optional inputs:
      rel_scale:  a pair giving relative scalings of x and y coordinates in
        the plane, to improve stepping in the different directions.
        e.g. (1,10) would make dx steps in the y-direction 10 times larger than
        in the x-direction.

      which:  which sub-manifold to compute 's', 'u' or ('s', 'u').
        Default is both.

      directions:  which directions along chosen sub-manifolds? (1,), (-1,)
        or (1,-1). Default is both.

      other_pts can be a list of points whose proximity will be checked,
    and the computation halted if they get within dx of the manifold.

      dx_perp_fac:  For advanced use only. If you get failures saying dx_perp
         too small and that initial displacement did not straddle manifold, try
         increasing this factor towards 1 (default 0.75). Especially for
         unstable manifolds, initial values for dx_perp may diverge, but if
         dx_perp is shrunk too quickly with this factor the sweet spot may be
         missed.

      verboselevel
      fignum
    """

    assert fp.classification == 'saddle' and not fp.degenerate
    if fp.evals[0] < 0:
        eval_s = fp.evals[0]
        eval_u = fp.evals[1]
        evec_s = fp.evecs[0]
        evec_u = fp.evecs[1]
    else:
        eval_s = fp.evals[1]
        eval_u = fp.evals[0]
        evec_s = fp.evecs[1]
        evec_u = fp.evecs[0]
    gen = fp.gen
    assert 'Gamma_out_plus' in gen.eventstruct, "Detection event surface(s) not present"
    assert 'Gamma_out_minus' in gen.eventstruct, "Detection event surface(s) not present"
    # Dividing fixed point's inherited epsilon tolerance by 100
    eps = fp.eps / 100
    dx_perp_eps = 1e-12
    if dx_perp_fac >= 1 or dx_perp_fac <= 0:
        raise ValueError("dx_perp_fac must be between 0 and 1")
    normord = fp.normord
    if rel_scale is None:
        rel_scale = (1,1)
    dxscaled = dx_scaled_2D(dx, rel_scale)
    if isinstance(dx_gamma, dict):
        assert len(dx_gamma) == 2, "Invalid value for dx_gamma"
        assert remain(dx_gamma.keys(), [1,-1]) == [], \
            "Invalid value for dx_gamma"
    else:
        try:
            dx_gamma = {1: dx_gamma, -1: dx_gamma}
        except:
            raise TypeError("Invalid type for dx_gamma")

    def test_fn(x, dircode):
        if verboselevel>1:
            print "Test point", x[x.coordnames[0]], x[x.coordnames[1]], "in direction", dircode, "\n"
        gen.set(ics=x)
        try:
            test = gen.compute('test', dirn=dircode)
        except:
            raise RuntimeError("Integration failed")
        events = gen.getEvents()
        if verboselevel>1:
            pts=test.sample(coords=x.coordnames)
            plot(pts[x.coordnames[0]][:25],pts[x.coordnames[1]][:25],'b-')
        if events['Gamma_out_plus'] is None:
            if events['Gamma_out_minus'] is None:
                if verboselevel>1:
                    pts=test.sample(coords=x.coordnames)
                    print "Last computed point was\n", pts[-1]
                    print "...after time", pts['t'][-1]
                    plot(pts[x.coordnames[0]],pts[x.coordnames[1]],'b-')
                raise RuntimeError("Did not reach Gamma surfaces")
            else:
                # hit Gamma_out_minus
                if verboselevel>1:
                    print "Reached Gamma minus at t=", events['Gamma_out_minus']['t'][0]
                sgn = -1
        else:
            if events['Gamma_out_minus'] is None:
                # hit Gamma_out_plus
                if verboselevel>1:
                    print "Reached Gamma plus at t=", events['Gamma_out_plus']['t'][0]
                sgn = 1
            else:
                if verboselevel>1:
                    pts=test.sample(coords=x.coordnames)
                    print "Last computed point was\n", pts[-1]
                    print "...after time", pts['t'][-1]
                    plot(pts[x.coordnames[0]],pts[x.coordnames[1]],'b-')
                raise RuntimeError("Did not reach Gamma surfaces")
        return sgn

    def onto_manifold(x_ic, dn, normal_dir, dircode='f'):
        try:
            return bisection(test_fn, x_ic+dn*normal_dir, x_ic-dn*normal_dir,
                            args=(dircode,), xtol=eps, maxiter=100,
                            normord=normord)
        except AssertionError:
            if verboselevel>1:
                xp = x_ic+dn*normal_dir
                xm = x_ic-dn*normal_dir
                plot(xp[var_x], xp[var_y], 'gx')
                plot(xm[var_x], xm[var_y], 'gx')
            raise RuntimeError("dx_perp too small? +/- initial displacement did not straddle manifold")
        except RuntimeError:
            if verboselevel>1:
                xp = x_ic+dn*normal_dir
                xm = x_ic-dn*normal_dir
                plot(xp[var_x], xp[var_y], 'gx')
                plot(xm[var_x], xm[var_y], 'gx')
            raise

    gen.eventstruct['Gamma_out_plus'].activeFlag=True  # terminal
    gen.eventstruct['Gamma_out_minus'].activeFlag=True  # terminal
    var_x = fp.point.coordnames[0]
    var_y = fp.point.coordnames[1]
    assert tmax > 0
    manifold = {}
    man_names = {'s': 'stable', 'u': 'unstable'}

    for w in which:
        ### w = 's' => stable branch
        ### w = 'u' => unstable branch
        if verboselevel>0:
            print "Starting %s branch" % man_names[w]
        if w == 's':
            col = 'g'
            w_sgn = -1
            integ_dircode = 'f'
            evec = evec_u
            evec_other = evec_s
        elif w == 'u':
            col = 'r'
            w_sgn = 1
            integ_dircode = 'b'
            evec = evec_s
            evec_other = evec_u
        # set Gamma_out surfaces on "outgoing" branch
        # (polarity is arbitrary)
        p0_plus = fp.point + dx_gamma[1]*evec
        p0_minus = fp.point - dx_gamma[-1]*evec
        evec_perp = get_perp(evec)
        # !! Must choose the event directions correctly
        # Algorithm should be based on: event dircode = 1
        # if ev(traj(ev_t - delta)) < 0 and ev(traj(ev_t + delta)) > 0
        # where the evec along the flow towards the event surfaces
        # determines which is "before" and which is "after" the
        # event surface (time may be reversed depending on which
        # manifold is being computed)
        print "Set these event directions according to your problem..."
        gen.eventstruct.setEventDir('Gamma_out_plus', -1)
        gen.eventstruct.setEventDir('Gamma_out_minus', 1)
        gen.set(pars={'Gamma_out_plus_p_'+var_x: p0_plus[var_x],
                      'Gamma_out_plus_p_'+var_y: p0_plus[var_y],
                      'Gamma_out_plus_dp_'+var_x: evec_perp[var_x],
                      'Gamma_out_plus_dp_'+var_y: evec_perp[var_y],
                      'Gamma_out_minus_p_'+var_x: p0_minus[var_x],
                      'Gamma_out_minus_p_'+var_y: p0_minus[var_y],
                      'Gamma_out_minus_dp_'+var_x: evec_perp[var_x],
                      'Gamma_out_minus_dp_'+var_y: evec_perp[var_y],
    ##                  'fp_'+var_x: fp.point[var_x], 'fp_'+var_y: fp.point[var_y]
                      },
                tdata = [0,tmax])
        if verboselevel>1:
            if fignum is None:
                fignum=figure()
            else:
                figure(fignum)
            plot([p0_plus[var_x]-dxscaled*evec_perp[var_x],p0_plus[var_x]+dxscaled*evec_perp[var_x]],
                 [p0_plus[var_y]-dxscaled*evec_perp[var_y],p0_plus[var_y]+dxscaled*evec_perp[var_y]], 'k-', linewidth=2)
            plot([p0_minus[var_x]-dxscaled*evec_perp[var_x],p0_minus[var_x]+dxscaled*evec_perp[var_x]],
                 [p0_minus[var_y]-dxscaled*evec_perp[var_y],p0_minus[var_y]+dxscaled*evec_perp[var_y]], 'k-', linewidth=2)
            draw()
        check_other_pts = other_pts is not None
        piece = {}
        if ic_dx is None:
            ic_dx = dxscaled
        else:
            ic_dx = dx_scaled_2D(ic_dx, rel_scale)
        if ic is None:
            ic = fp.point
            f_ic = -w_sgn * evec_other
            curve_len = 0
            # initial estimate x0 = a point close to f.p. along manifold with
            # opposite stability
        else:
            # initial curve length from previous independent variable, if present
            # otherwise, assume zero
            if isinstance(ic, Pointset):
                assert len(ic) == 1, "Only pass a length-1 pointset"
                # guarantee curve_len > 0
                curve_len = abs(ic['arc_len'][0])
                ic = ic[0]
            else:
                curve_len = 0
            f_ic = -w_sgn * gen.Rhs(0, ic, gen.pars)  # array
        for sgn in directions:
            if verboselevel>0:
                print "Starting direction", sgn
            # PREDICTION
            x0_ic = ic+w_sgn*sgn*ic_dx*f_ic/norm(f_ic, normord)
            if verboselevel>1:
                figure(fignum)
                plot(x0_ic[var_x], x0_ic[var_y], 'go', linewidth=1)
            # put x0 initial estimate onto stable manifold
            f = -w_sgn * gen.Rhs(0, x0_ic, gen.pars)  # array
            norm_to_flow = get_perp(f/norm(f, normord))
            if verboselevel>1:
                plot([x0_ic[var_x], x0_ic[var_x]+dxscaled*f[0]/norm(f,normord)],
                     [x0_ic[var_y], x0_ic[var_y]+dxscaled*f[1]/norm(f,normord)],
                     'r-')
                plot([x0_ic[var_x], x0_ic[var_x]+dxscaled*norm_to_flow[0]],
                     [x0_ic[var_y], x0_ic[var_y]+dxscaled*norm_to_flow[1]],
                     'r:')
            dx_perp_default = dx_perp
            # CORRECTION
            while dx_perp > dx_perp_eps:
                try:
                    x = onto_manifold(x0_ic, dx_perp, norm_to_flow,
                                      dircode=integ_dircode)
                except RuntimeError, e:
                    dx_perp *= dx_perp_fac
                else:
                    break
            if dx_perp <= dx_perp_eps:
                # RuntimeError was raised and could not continue reducing dx_perp
                print "dx_perp reached lower tolerance =", dx_perp_eps
                print e
                raise RuntimeError("Initial point did not converge")
            else:
                curve_len += norm(x-ic, normord)
                piece[sgn*curve_len] = x
                num_pts = 1
                last_x = x
                if verboselevel>0:
                    print "Initial point converged to (%.6f, %.6f)\n" % \
                          (x[var_x], x[var_y])
            dx_perp = dx_perp_default
            last_f = f_ic
            # step backwards along flow
            while curve_len < max_len and num_pts < max_pts:
                if verboselevel>0:
                    figure(fignum)
                    plot(last_x[var_x], last_x[var_y], col+'.', linewidth=1)
                if check_other_pts and sometrue([norm(last_x - pt, normord) < dx \
                                 for pt in other_pts]):
                    # we've hit a different fixed point (or other feature), so stop
                    break
                f = -w_sgn * gen.Rhs(0, last_x, gen.pars)
                if all(sign(f) != sign(last_f)):
                    f = -f
                    # on other side of manifold so must keep stepping in the
                    # same direction, therefore switch signs!
                # PREDICTION
                x_ic = last_x + w_sgn*sgn*dxscaled*f/norm(f,normord)
                last_f = f
                if verboselevel>1:
                    print "\nStarting from point ", last_x
                    delta = w_sgn*sgn*dxscaled*f/norm(f,normord)
                    print "Trying point ", x_ic, "in direction (%.6f, %.6f)\n" % (delta[0], delta[1])
                dx_perp = dx_perp_default
                # CORRECTION
                while dx_perp > dx_perp_eps:
                    try:
                        x = onto_manifold(x_ic, dx_perp, get_perp(f/norm(f,normord)),
                                          dircode=integ_dircode)
                    except RuntimeError, e:
                        dx_perp *= 0.75
                    else:
                        break
                if dx_perp <= dx_perp_eps:
                    # RuntimeError was raised and could not continue reducing dx_perp
                    print "dx_perp reached lower tolerance =", dx_perp_eps
                    print e
                    break  # end while search
                else:
                    curve_len += norm(x-last_x, normord)
                    piece[sgn*curve_len] = x
                    last_x = x
                    num_pts += 1
                    if verboselevel>1:
                        print "\nManifold has %i points" % num_pts
                    elif verboselevel>0:
                        print ".",
                        sys.stdout.flush()
        if verboselevel>0:
            # finish the line
            print " "
        indepvar, piece_sorted = sortedDictLists(piece, byvalue=False)
        manifold[w] = pointsToPointset(piece_sorted, indepvarname='arc_len',
                                       indepvararray=indepvar, norm=normord)
    gen.eventstruct['Gamma_out_plus'].activeFlag=False
    gen.eventstruct['Gamma_out_minus'].activeFlag=False
##    gen.eventstruct['fp_closest'].activeFlag=False

    return manifold


def make_flow_normal_event(x, y, dxdt, dydt, targetlang, flatspec=None,
                           fnspec=None, evtArgs=None):
    """For 2D vector fields only.

    Supply flatspec if built vector field using ModelConstructor tools,
    otherwise specify funcspec argument.
    """
    x_par = x+'_normflow_p'
    y_par = y+'_normflow_p'

    if fnspec is None:
        if flatspec is None:
            raise ValueError("Must supply one of funcspec or flatspec")
        varnames=[]
        parnames=[]
        inputnames=[]
        fnspecs={}
    else:
        if flatspec is not None:
            raise ValueError("Must supply only one of funcspec or flatspec")
        try:
            varnames = fnspec['vars']
        except KeyError:
            varnames = []
        try:
            parnames = fnspec['pars']
        except KeyError:
            parnames = []
        try:
            inputnames = fnspec['inputs']
        except KeyError:
            inputnames = []
        try:
            fnspecs = fnspec['auxfns']
        except KeyError:
            fnspecs = {}
    if evtArgs is None:
        evtArgs = {'name': 'flow_normal_2D_evt',
                   'eventtol': 1e-5,
                   'eventdelay': 1e-4,
                   'starttime': 0,
                   'precise': True,
                   'term': False}
    else:
        evtArgs['term'] = False
        evtArgs['name'] = 'flow_normal_2D_evt'
    v_dot_f = '(' + x_par + ' - ' + x + ') * (' + dxdt + ') + ' + \
              '(' + y_par + ' - ' + y + ') * (' + dydt + ')'
    norm_v = 'sqrt(' + x_par + '*' + x_par + '+' + y_par + '*' + y_par + ')'
    norm_f = 'sqrt(pow((' + dydt + '),2) + pow((' + dxdt + '),2))'
    flow_n_str = v_dot_f + '/(' + norm_v + '+' + norm_f + ')'
    ev = Events.makeZeroCrossEvent(expr=flow_n_str,
                                 dircode=0,
                                 argDict=evtArgs,
                                 targetlang=targetlang,
                                 varnames=varnames,
                                 parnames=parnames+[x_par,y_par],
                                 inputnames=inputnames, fnspecs=fnspecs,
                                 flatspec=flatspec)
    ev_helper = {'event_names': {2:'flow_normal_2D_evt'}, # 2 for 2D
                 'pars_to_vars': {x_par:x, y_par:y}}
    return (ev, ev_helper)


class phaseplane(object):
    """Working environment for 2D phase-plane analysis.

    Effectively a thin wrapper around a Generator class
    (will later be around a Model class, in order to support hybrid
    2D vector fields)
    """

    def __init__(self, vf_info, x_axis='',
                 name='', normord=2, eps=1e-12):
        """Use x_axis to specify which variable should appear on the x-axis"""

        assert isreal(eps) and eps > 0, "eps tolerance must be a positive real number"
        self.eps = eps
        assert isinstance(normord, int) and normord > 0, "normord must be a positive integer"
        self.normord = normord
        self.name = name
        try:
            assert isinstance(vf_info.ODEclass, ODEsystem), \
               "Can only use this class with ODE system Generators"
            # Should make this compatible with ModelSpec construction too
            assert len(vf_info.ODEargs.vars)==2, "Only 2D systems permitted for phase planes"
        except AttributeError:
            raise TypeError("Invalid form given for vf_info")
        self.vf_info = vf_info
        # store copy of the actual vector definition to make sure this doesn't
        # change when things are added or removed and the vf is rebuilt --
        # otherwise all self.objects will be invalidated
        self._vf_core_copy = copy.copy(vf_info.ODEargs)
        if x_axis != '':
            assert x_axis in vf_info.ODEargs.vars
            self.x_axis = x_axis
            self.y_axis = remain(vf_info.ODEargs.vars, x_axis)[0]
        else:
            self.x_axis, self.y_axis = vf_info.ODEargs.vars
        try:
            self.build_vf()
        except ValueError:
            self.vf = None
        self.objects = args(fixedpoints=None,nullclines=None,limitcycles=None,
                            submanifolds=None,trajectories=None,specialpoints=None)

    def build_vf(self):
        # Later, wrap generator into a Model object using embed()
        # Verify that self._vf_info.ODEargs has same core attributes as
        # self._vf_core_copy, otherwise raise an exception
        try:
            self.vf = self.vf_info.ODEclass.__new__(self.vf_info.ODEclass,
                                                    self.vf_info.ODEargs)
        except:
            raise ValueError("Invalid or incomplete parameters for building vector field Generator")

    def add(self, attribute, value):
        try:
            setattr(self.vf_info['args'], attribute, value)
        except:
            raise ValueError("Invalid vector field attribute %s or value %s"%(attribute,str(value)))

    def remove(self, attribute, value):
        try:
            del self.vf_info['args'][attribute][value]
        except:
            raise ValueError("Could not delete vector field value %s of attribute %s"%(str(value), attribute))



def find_period(pts, thresh, dir=1, with_indices=False):
    """pts is a Pointset.

    thresh is a 1D Point or a dictionary, whose coordinate name corresponds
    to a variable in the pts pointset.

    dir is either 1 or -1, indicating which direction the threshold must be
    crossed to be counted (default 1 = increasing).

    with_indices is a Boolean indicating whether to return the pair of indices
    indicating the beginning and end points of the last period (default False).
    """
    try:
        threshdict = dict(thresh)
    except:
        raise TypeError("thresh must be a 1D Point object or dictionary")
    assert len(threshdict)==1, "thresh must be a 1D Point object or dictionary"
    var = thresh.keys()[0]
    a = pts[var]
    t = pts['t']
    ts = []
    ixs = []
    th = thresh[var]
    # if don't start on appropriate side of thresh, wait
    def inc_fn(x):
        return x < th
    def dec_fn(x):
        return x > th
    if dir == 1:
        off_fn = inc_fn
    elif dir == -1:
        off_fn = dec_fn
    else:
        raise ValueError("dir must be 1 or -1")
    off = off_fn(a[0])
    for i,av in enumerate(a):
        if off and not off_fn(av):
            ts.append(t[i])
            ixs.append(i)
            off = False
##            print "Detected at i=%d, a=%f"%(i,av)
##            print ts
        elif not off and off_fn(av):
            off = True
    if len(ts) >= 1:
        if with_indices:
            return ts[-1]-ts[-2], ixs
        else:
            return ts[-1]-ts[-2]
    else:
        print len(ts), "is not enough periods",
        return NaN


# ---------------------------------------------------------------

# Feature sub-classes
class zone_node(ql_feature_node):
    """Phase plane 'zone' node of hierarchical qualitative feature abstract class.
    A zone is a bounding box in the phase plane containing a feature, grown to include
    a region that satisfies some local condition.

    Must define _leaf_class for each implementation of a zone_node class
    """
    def finish(self, target):
        self.subfeatures = dict.fromkeys(self.results.zone_center_ixs, None)
        for ix in self.results.zone_center_ixs:
            feat = self._leaf_class('zone_%i'%ix, pars=self.pars)
            feat.pars.all_zone_ixs = list(self.results.zone_center_ixs)
            feat.pars.index = ix
            self.subfeatures[ix] = feat
            feat(target)

    def _local_init(self):
        pass


class zone_leaf(ql_feature_leaf):
    """Phase plane 'zone' for leaf of hierarchical qualitative feature abstract class.
    A zone is a bounding box in the phase plane containing a feature, grown to include
    a region that satisfies some local condition.
    """
    def evaluate(self, target):
        self.results.zones = None

    def finish(self, target):
        """Override this function to be called if evaluate returns True"""
        pass


# Abstract classes
class nullcline_zone_node(zone_node):
    pass


class nullcline_zone_leaf(zone_leaf):
    """Parameters to apply:
    par: xtol
    optional par: find_exact_center (unused)
    """
    def _prepare(self, nullc):
        center_ix = self.pars.index
        x_center_approx, y_center_approx = nullc.array[center_ix]
        # Do we care what is the exact position of the zone's defining position?
        try:
            exact = self.pars.find_exact_center
        except AttributeError:
            # default False
            exact = False
        if exact:
            raise NotImplementedError
        else:
            x_center = x_center_approx
            y_center = y_center_approx
        return center_ix, x_center, y_center

    def _set_mutex_zone_by_width(self, center_ix, x_center, y_center, nullc):
        """Fix zone size by width or halfway to next center, if closer."""
        try:
            zone_width = self.pars.zone_width
        except AttributeError:
            raise ValueError("Must set a zone_width parameter for this feature")
        min_zone_x = x_center - zone_width
        max_zone_x = x_center + zone_width
        x_range = [np.nan, np.nan]
        y_range = [np.nan, np.nan]
        zone_position = self.pars.all_zone_ixs.index(center_ix)
        if zone_position == 0:
            # everything to the left is already included
            ix = 0
        else:
            ix = self.pars.all_zone_ixs[zone_position-1]
        x_range[0] = max( 0.5*(nullc.array[ix,0]+x_center), min_zone_x )
        y_range[0] = nullc(x_range[0])

        if center_ix == self.pars.all_zone_ixs[-1]:
            # everything to the right is already included
            ix = -1
        else:
            ix = self.pars.all_zone_ixs[zone_position+1]
        x_range[1] = min( 0.5*(nullc.array[ix,0]+x_center), max_zone_x )
        y_range[1] = nullc(x_range[1])
        return [np.array(x_range), np.array(y_range)]


    def _grow_zone_by_radius(self, center_ix, x_center, y_center, nullc, property_func):
        raise NotImplementedError

    def _grow_zone_by_width(self, center_ix, x_center, y_center, nullc, property_func):
        """property_func must return +/- 1 or 0/1 as a function of x"""
        try:
            zone_width = self.pars.zone_width
        except AttributeError:
            zone_width = np.inf  # may fail unexpectedly
            print "Warning: zone_width parameter defaulted to infinity"
        min_zone_x = x_center - zone_width
        max_zone_x = x_center + zone_width
        # Grow domain of zone past known sample points with same property
        # until intermediate position between sample points found
        x_range = [np.nan, np.nan]
        y_range = [np.nan, np.nan]
        zone_position = self.pars.all_zone_ixs.index(center_ix)

        # Left search
        if zone_position == 0:
            # everything to the left is already included
            x_range[0] = max( nullc.array[0,0], min_zone_x )
            y_range[0] = nullc(x_range[0])
        else:
            # all sample points with indices ix_after:center_ix (inclusive)
            # have the same property.
            ix_before = self.pars.all_zone_ixs[zone_position-1]
            ix_after = ix_before + 1
            if nullc.array[ix_after,0] < min_zone_x:
                x_range[0] = min_zone_x
                y_range[0] = nullc(x_range[0])
            else:
                # search between these points on the nullcline
                xsol = bisection(property_func, nullc.array[ix_before,0],
                                  nullc.array[ix_after,0], xtol=self.pars.xtol)
                if abs(xsol - nullc.array[ix_before,0]) < self.pars.xtol:
                    # corner case where zero is right at a sample point
                    # try again with extended sample points, where available
                    if ix_before > 0:
                        xsol = bisection(property_func, nullc.array[ix_before-1,0],
                                  nullc.array[ix_after,0], xtol=self.pars.xtol)
                elif abs(xsol - nullc.array[ix_after,0]) < self.pars.xtol:
                    # corner case where zero is right at a sample point
                    # try again with extended sample points, where available
                    if ix_after < len(nullc.array) - 1:
                        xsol = bisection(property_func, nullc.array[ix_before,0],
                                  nullc.array[ix_after+1,0], xtol=self.pars.xtol)
                x_range[0] = xsol
                y_range[0] = nullc(xsol)

        # Right search
        if center_ix == self.pars.all_zone_ixs[-1]:
            # everything to the right is already included
            x_range[1] = nullc.array[-1,0]
            y_range[1] = nullc.array[-1,1]
        else:
            # all sample points with indices center_ix:ix_before (inclusive)
            # have the same property.
            ix_before = self.pars.all_zone_ixs[zone_position+1]
            ix_after = ix_before + 1
            if nullc.array[ix_before,0] > max_zone_x:
                x_range[1] = max_zone_x
                y_range[1] = nullc(x_range[1])
            else:
                # search between these points on the nullcline
                xsol = bisection(property_func, nullc.array[ix_before,0],
                                  nullc.array[ix_after,0], xtol=self.pars.xtol)
                if abs(xsol - nullc.array[ix_before,0]) < self.pars.xtol:
                    # corner case where zero is right at a sample point
                    # try again with extended sample points, where available
                    if ix_before > 0:
                        xsol = bisection(property_func, nullc.array[ix_before-1,0],
                                  nullc.array[ix_after,0], xtol=self.pars.xtol)
                elif abs(xsol - nullc.array[ix_after,0]) < self.pars.xtol:
                    # corner case where zero is right at a sample point
                    # try again with extended sample points, where available
                    if ix_after < len(nullc.array) - 1:
                        xsol = bisection(property_func, nullc.array[ix_before,0],
                                  nullc.array[ix_after+1,0], xtol=self.pars.xtol)
                x_range[1] = xsol
                y_range[1] = nullc(xsol)

        return [np.array(x_range), np.array(y_range)]


# Implementations
class fixedpoint_zone(nullcline_zone_leaf):
    # needs *two* nullcline objects and a f.p. object
    pass


# define prior to node so that can provide leaf class name to node class
class inflection_zone_leaf(nullcline_zone_leaf):
    """A single inflection point zone.

    pars:
      xtol --> precision of zone center finding
      find_exact -> exact x value of inflection (not implemented yet)
      zone_width -> positive scalar (default Inf grows to next inflection point)
    """
    #Only use one of zone_radius or zone_width, otherwise zone_width will be used.
    def evaluate(self, nullc):
        # Indices of inflection mark last known sample point having
        # same concavity to those locally on its left in the point order
        # from left to right.
        center_ix, x_center, y_center = self._prepare(nullc)
        self.results.x_center = x_center
        self.results.y_center = y_center
        self.results.zone = self._grow_zone_by_width(center_ix, x_center, y_center, nullc,
                                                     nullc.concavity)
        # if got this far, succeeded in finding a zone
        return True


class inflection_zone_node(nullcline_zone_node):
    """Find all inflection point zones
    """
    _leaf_class = inflection_zone_leaf

    def evaluate(self, nullc):
        """Assumes at least 2 sample points in nullcline.
        """
        # Find all approximate places (sample points) where
        # inflections occur.
        # Concavities are -1, 0, 1 but we don't care about the signs
        # so much as their changes. To this end, replace all zeros
        # with their previous neighbor's value
        concs = nullc.concavity_at_sample_points()
        conc_zeros = concs == 0
        while np.sometrue(conc_zeros):
            concs_list = list(concs)
            # find lowest index with a zero
            zero_ix = concs_list.index(0)
            if zero_ix > 0:
                concs[zero_ix] = concs[zero_ix-1]
            else:
                # replace with next non-zero value
                try:
                    pos_ix = concs_list.index(1)
                except ValueError:
                    # no +1's
                    pos_ix = np.inf
                try:
                    neg_ix = concs_list.index(-1)
                except ValueError:
                    # no -1's
                    neg_ix = np.inf
                min_ix = min( pos_ix, neg_ix )
                if isfinite(min_ix):
                    concs[0] = concs[min_ix]
                else:
                    concs[0] = 1
            conc_zeros = concs == 0
        # ignore changes within a single sample point of the endpoints
        # only -1 indicate. The extra -1 subtraction ensures all the +1's
        # become zero so that argwhere can discount those locations.
        self.results.zone_center_ixs = np.argwhere(concs[:-1] * concs[1:] - 1).flatten()
        return len(self.results.zone_center_ixs) > 0



class max_curvature_zone_leaf(nullcline_zone_leaf):
    """A single zone of locally maximal curvature

    pars:
      xtol --> precision of zone center finding
      direction (currently ignored -- looks for both)
      find_exact -> exact x value of max curvature (not implemented yet)
      zone_width -> positive scalar (default Inf grows to next local maximum)
    """
    #Only use one of zone_radius or zone_width, otherwise zone_width will be used.
    def evaluate(self, nullc):
        # Indices of max curvature mark last known sample point having
        # same curvature gradient to those locally on its left in the point order
        # from left to right.
        center_ix, x_center, y_center = self._prepare(nullc)
        self.results.x_center = x_center
        self.results.y_center = y_center
        self.results.zone = self._set_mutex_zone_by_width(center_ix, x_center,
                                                          y_center, nullc)
        # if got this far, succeeded in finding a zone
        return True


class max_curvature_zone_node(nullcline_zone_node):
    """Find all zones with locally maximal curvature.
    Optional selection of those facing up or down only:
    par: direction (-1, 0, 1, where 0 selects either (default))
    """
    _leaf_class = max_curvature_zone_leaf

    def evaluate(self, nullc):
        print "TEMP: curvature maximized in BOTH directions (par ignored)"
        try:
            dirn = self.pars.direction
        except AttributeError:
            dirn = 0  # both
        gcurvature = nullc.grad_curvature_at_sample_points()
        # find local maxima, including endpoint maxima
        # TEMP: don't look at endpoints for now
        # inside endpoints, look for where d/dx( curvature ) = 0
        # look for sign changes, just like concavity signs
        # for inflection feature
        gsigns = np.sign( gcurvature )
        gsign_zeros = gsigns == 0
        while np.sometrue(gsign_zeros):
            gsigns_list = list(gsigns)
            # find lowest index with a zero
            zero_ix = gsigns_list.index(0)
            if zero_ix > 0:
                gsigns[zero_ix] = gsigns[zero_ix-1]
            else:
                # replace with next non-zero value
                try:
                    pos_ix = gsigns_list.index(1)
                except ValueError:
                    # no +1's
                    pos_ix = np.inf
                try:
                    neg_ix = gsigns_list.index(-1)
                except ValueError:
                    # no -1's
                    neg_ix = np.inf
                min_ix = min( pos_ix, neg_ix )
                if isfinite(min_ix):
                    gsigns[0] = gsigns[min_ix]
                else:
                    gsigns[0] = 1
            gsign_zeros = gsigns == 0
        # ignore changes within a single sample point of the endpoints
        # only -1 indicate. The extra -1 subtraction ensures all the +1's
        # become zero so that argwhere can discount those locations.
        self.results.zone_center_ixs = np.argwhere(gsigns[:-1] * gsigns[1:] - 1).flatten()
        return len(self.results.zone_center_ixs) > 0


class min_curvature_zone(nullcline_zone_leaf):
    pass




# ---------------------------------------------------------------
# ANIMATION TOOLS



def get_PP(gen, pt, vars, doms=None, doplot=True,
           t=0, saveplot=None, format='svg', trail_pts=None,
           null_style='-', orbit_col='g', orbit_style='-'):
    """Get a phase plane for generator gen at point pt.
    Specify t if the system is non-autonomous.
    If trail_pts are given (e.g., via show_PPs) then these are added
    behind the point pt.

    Requires matplotlib
    """
    xFS, yFS = gen._FScompatibleNames(vars)
    x, y = gen._FScompatibleNamesInv(vars)
    pt = gen._FScompatibleNamesInv(pt)
    ptFS = gen._FScompatibleNames(pt)

    sub_dom = filteredDict(dict(ptFS),
                           remain(gen.funcspec.vars, [x,y]))
    if doms is None:
        doms = gen.xdomain
    else:
        doms = gen._FScompatibleNames(doms)
    sub_dom[xFS] = doms[xFS]
    sub_dom[yFS] = doms[yFS]
    x_dom = doms[xFS]
    y_dom = doms[yFS]
    x_interval = Interval(xFS, float, x_dom, abseps=0)
    y_interval = Interval(yFS, float, y_dom, abseps=0)
    fps = find_fixedpoints(gen, n=6, subdomain=sub_dom,
                           t=t, eps=1e-8)

    f = figure(1)
    nulls_x, nulls_y, handles = find_nullclines(gen, xFS, yFS,
                                    x_dom=x_dom, y_dom=y_dom,
                                    fixed_vars=ptFS, n=3, t=t,
                                    max_step={xFS: 0.1, yFS: 1},
                                    max_num_points=10000, fps=fps,
                                    doplot=doplot, plot_style=null_style,
                                    newfig=False)
    if doplot:
        tol = 0.01
        xwidth = abs(x_dom[1]-x_dom[0])
        ywidth = abs(y_dom[1]-y_dom[0])
        x_lims = [x_dom[0]-tol*xwidth, x_dom[1]+tol*xwidth]
        y_lims = [y_dom[0]-tol*ywidth, y_dom[1]+tol*ywidth]
        if trail_pts is not None:
            pp.plot(trail_pts[x], trail_pts[y], orbit_col)
        pp.plot(pt[x], pt[y], orbit_col+'o')
        for fp in fps:
            if fp[x] in x_interval and fp[y] in y_interval:
                pp.plot(fp[x], fp[y], 'ko')
        ## the following commands can introduce artifact lines!!
        ax = pp.gca()
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        draw()
        ##
        if saveplot is not None:
            if saveplot[-4:] != '.' + format:
                saveplot = saveplot + '.' + format
            f.savefig(saveplot, format=format)
            f.clf()
    return fps, nulls_x, nulls_y


def show_PPs(gen, x, y, traj, t_start, t_step, t_end, moviename=None, doms=None,
             show_trail=False, add_traj=None):
    """Show phase plane x vs y for Generator gen over trajectory traj,
    over times t_start:t_step:t_end.
    Option to create a movie, provided ffmpeg if installed.
    Optional domain dictionary for variables can be provided, otherwise their
    extents in the given trajectory will be used.
    Option to show trail behind green orbit (default False).
    Option to add a comparison trajectory / nullclines to the plot.

    Requires matplotlib and ffmpeg
    """
    t = t_start
    pt0 = traj(t_start)
    pt1 = traj(t_end)
    xs = [pt0[x], pt1[x]]
    ys = [pt0[y], pt1[y]]
    if doms is None:
        doms = {x: [min(xs),max(xs)], y: [min(ys),max(ys)]}
    if moviename is not None:
        os.system('mkdir %s' % moviename)
        os.chdir(moviename)
    i = 0
    while t <= t_end:
        pt = traj(t)
        i += 1
        print "Frame %i, t = %.4f: step %.4f until %.4f" % (i, t, t_step, t_end)
        if moviename is not None:
            saveplot = 'pp_fig_%03d' % i
            format = 'png'
        else:
            saveplot = None
            format = 'svg'
        if show_trail and t > t_start:
            trail_pts = traj.sample(tlo=t_start, thi=t)
            if add_traj is not None:
                trail_pts2 = add_traj.sample(tlo=t_start, thi=t)
        else:
            trail_pts = None
            trail_pts2 = None
        if add_traj is None:
            get_PP(gen, pt, [x, y], doms=doms,
               t=t, doplot=True, saveplot=saveplot, format=format,
               trail_pts=trail_pts)
        else:
            get_PP(gen, pt, [x, y], doms=doms,
               t=t, doplot=True, saveplot=None,
               trail_pts=trail_pts, null_style='-', orbit_col='g', orbit_style='-')
            get_PP(gen, add_traj(t), [x, y], doms=doms,
               t=t, doplot=True, saveplot=saveplot, format=format,
               trail_pts=trail_pts2, null_style=':', orbit_col='c', orbit_style='-')
        t += t_step
    if moviename is not None:
        # copy the final frame a few times to pad out end of video
        for j in range(i+1, i+51):
            os.system('cp pp_fig_%03d.png pp_fig_%03d.png' % (i, j))
        os.system('ffmpeg -f image2 -i pp_fig_%03d.png -r 14 pp_movie.avi')
        os.chdir('..')


# -----------------------------------------------------------

# EXPERIMENTAL CLASS
class mesh_patch_2D(object):
    """2D mesh patch generator (for 4 or 8 points of a fixed distance
    from a central point). i.e., total number of mesh points may be 5
    or 9.

    Once either 5 or 9 points have been selected, this is fixed for
    the mesh patch instance. The central point p0 and radii r of the
    mesh points are mutable using the 'update' method. If
    shape='circle' (default), mesh points are equidistant from p0. If
    shape='square', mesh points are distributed around the corners and
    mid-sides of a square of side length 2*r.

    The points are in the order p0 followed by points in 'clockwise'
    order in the phase plane defined by the two variable names in p0.

    Functions / callable objects can be applied to all points in the
    mesh patch by calling the mesh patch's 'eval' method with the
    function. It returns an a dictionary of mesh points -> function
    return values.

    The direction of minimum (interpolated) gradient of a functional
    over the mesh can be determined from the 'min_gradient_dir' method.
    A similar method 'max_gradient_dir' also exists.

    Highly experimental class!

    Rob Clewley, Jan 2007
    """
    _s2 = sqrt(2)/2
    _mesh5 = [(0,0), (0,1), (1,0), (0,-1), (-1,0)]
    _mesh9s = [(0,0), (0,1), (1,1), (1,0), (1,-1),
               (0,-1), (-1,-1), (-1,0), (-1, 1)]
    _mesh9c = [(0,0), (0,1), (_s2,_s2), (1,0), (_s2,-_s2),
               (0,-1), (-_s2,-_s2), (-1,0), (-_s2,_s2)]

    def __init__(self, p0, r, n=5, shape='circle'):
        """p0 should be of type Point, r must be either a scalar or a Point
        giving weighted radius in each of two directions, n must be 5 or 9.
        """
        self.p0 = p0
        try:
            if len(r) == 2:
                assert r.coordnames == p0.coordnames, "Coordinate names of "\
                       "r and p0 must match"
        except TypeError:
            # r is a scalar
            pass
        self.r = r
        self.n = n
        self.shape = shape
        self.make_mesh(makelocalmesh=True)
        self.derive_angles()

    def make_mesh(self, makelocalmesh=False):
        if self.n==5:
            mesh = [self.r*array(p, 'd') for p in self._mesh5]
            self.shape = 'circle'    # force it regardless of its value
        elif self.n==9:
            if self.shape=='circle':
                m = self._mesh9c
            elif self.shape=='square':
                m = self._mesh9s
            else:
                raise ValueError("Shape argument must be 'square' or 'circle'")
            mesh = [self.r*array(p, 'd') for p in m]
        else:
            raise ValueError("Only n=5 or 9 meshes are supported")
        # self.mesh is the absolute coords of the mesh
        try:
            self.mesh = pointsToPointset([self.p0+p for p in mesh])
        except AssertionError:
            raise TypeError("Central point must be given by a 2D Point object")
        if makelocalmesh:
            # Only needs to be done once
            # Make (0,0) as a Point with the correct coordnames, for convenience
            zp = self.p0*0
            # self.mesh_local is the mesh in local coordinates (relative to p0)
            self.mesh_local = pointsToPointset([zp+p for p in mesh])

    def update(self, p0=None, r=None):
        """Change the central point p0 or patch radius r
        """
        do_update = False
        if p0 is not None:
            self.p0 = p0
            do_update = True
        if r is not None:
            self.r = r
            do_update = True
        if do_update:
            self.make_mesh()

    def derive_angles(self):
        # angles are always measured in the positive direction here
        # angles don't include the centre point at (0,0), and are measured
        # from the "x" axis
        if self.n == 5:
            self._angles = [pi/2, 0, 3*pi/2, pi]
        else:
            # only mesh-points not aligned with axes need to be calculated
            # the others are constant regardless of re-scaling of axes
            vars = self.p0.coordnames
            self._angles = []
            for p in self.mesh_local[1:]:
                a = atan2(p[vars[1]],p[vars[0]])
                if a < 0:
                    a += 2*pi
                self._angles.append(a)


    def __getitem__(self, ix):
        return self.mesh[ix]

    def eval(self, f):
        """Evaluate the function or callable object f at the mesh patch points.
        Returns a dictionary of
            mesh point index -> function value
        Indices are always in the same order as the Pointset attribute 'mesh'
        """
        res = {}
        for i, p in enumerate(self.mesh):
            try:
                res[i] = f(p)
            except:
                print "Problem evaluating supplied function on mesh patch at " \
                    "point ", p
                raise
        return res

    def setup_grad(self, valdict, orient, dir):
        try:
            vals = [valdict[i] for i in range(self.n)]
        except KeyError:
            raise ValueError("vals must have same length as number "
                             "of mesh points")
        if orient is not None and dir is not None:
            ixs = []
            try:
                vars = orient.coordnames
            except AttributeError:
                raise TypeError("orient must be a vector (Point object)")
            if dir not in [-1,1]:
                raise ValueError("dir value must be 1 or -1")
            x = dir*orient[vars[0]]
            y = dir*orient[vars[1]]
            # angle of directed orientation
            theta = atan2(y,x)
            if theta < 0:
                theta += 2*pi
            # filter out mesh points whose defining angles are outside of
            # the half-plane specified by the normal vector
            twopi = 2*pi
            for i, a in enumerate(self._angles):
                test = abs(a - theta)
                if test <= pi/2 or twopi-test <= pi/2:
                    # append 1+i to correspond to 1..n mesh points
                    ixs.append(i+1)
##            print "Theta was ", theta
##            print "Angles: ", self._angles
##            print "Min grad using ixs", ixs
            use_vals = [vals[i] for i in ixs]
        else:
            ixs = range(1,self.n)
            use_vals = vals
        return vals, ixs, use_vals

    def min_gradient_dir(self, valdict, orient=None, dir=None):
        """Find the direction of any local minima of a functional over
        the mesh patch, whose dictionary of
            mesh point index -> function value
        is given by valdict. The magnitudes of these values are
        compared. The direction is indicated by a Point object, a
        relative position at the patch's characteristic radius from
        p0.

        If the functional is constant everywhere on the patch (degenerate)
        then a random direction is returned. If the functional is constant
        everywhere except at the centre, then the vector (0,0) is returned
        if the centre has lower value; a random vector is returned
        otherwise.

        If the functional is partially degenerate on some subset of
        mesh points then the optional orient and dir arguments can
        provide a means to select, provided the mesh resolution is
        sufficiently high.

        orient is a vector (Point object) relative to p0 orienting a
        "positive" direction, such that the integer dir = -1 or +1
        selects one half of the mesh points on which to conduct the
        search.
        """
        return self.__min_grad(valdict, orient, dir, 1)

    def min_gradient_mesh(self, valdict, orient=None, dir=None):
        """Find the direction of any local minima of a functional over
        the mesh patch, whose dictionary of
            mesh point index -> function value
        is given by valdict. The magnitudes of these values are
        compared. The direction is returned as a pair containing the two
        closest mesh points to the direction (in mesh local coords).

        If the functional is constant everywhere on the patch (degenerate)
        then a random direction is returned. If the functional is constant
        everywhere except at the centre, then the vector (0,0) is returned
        if the centre has lower value; a random vector is returned
        otherwise.

        If the functional is partially degenerate on some subset of
        mesh points then the optional orient and dir arguments can
        provide a means to select, provided the mesh resolution is
        sufficiently high.

        orient is a vector (Point object) relative to p0 orienting a
        "positive" direction, such that the integer dir = -1 or +1
        selects one half of the mesh points on which to conduct the
        search.
        """
        return self.__min_grad(valdict, orient, dir, 0)

    def __min_grad(self, valdict, orient, dir, return_type):
        vals, ixs, use_vals = self.setup_grad(valdict, orient, dir)
        degen_everywhere = alltrue([v==vals[0] for v in vals])
        degen_exterior = alltrue([v==vals[ixs[0]] for v in use_vals])
        degen_exterior_lower = degen_exterior and vals[0] >= vals[ixs[0]]
        degen_exterior_higher = degen_exterior and vals[0] < vals[ixs[0]]
        if degen_everywhere or degen_exterior_lower:
            # degenerate case -- return random vector
            x = uniform(-1,1)
            y = uniform(-1,1)
            vl = sqrt(x*x + y*y)
            x = x/vl
            y = y/vl
            if return_type == 0:
                raise RuntimeError("No valid directions to point in")
            else:
                return Point(dict(zip(self.p0.coordnames, [x,y])))
        elif degen_exterior_higher:
            if return_type == 0:
                raise RuntimeError("No valid directions to point in")
            else:
                return Point(dict(zip(self.p0.coordnames, [0,0])))
        d1 = (Inf, NaN)
        d2 = (Inf, NaN)
        for i, v in enumerate(use_vals):
            if ixs[i] == 0:
                # don't include centre point
                continue
            if abs(v) < d1[0]:
                d2 = d1
                d1 = (abs(v), ixs[i])
            elif abs(v) < d2[0]:
                d2 = (abs(v), ixs[i])
        # the two best mesh patch points (unless partially degenerate)
        pt1 = self.mesh_local[d1[1]]
        pt2 = self.mesh_local[d2[1]]
        if return_type == 0:
            return (pt1, pt2)
        else:
            score1 = d1[0]
            score2 = d2[0]
            if score1 + score2 == 0:
                interp_pt = pt1+pt2
                return self.r*interp_pt/norm(interp_pt)
            elif score1 == 0:
                return pt1
            elif score2 == 0:
                return pt2
            else:
                interp_pt = pt1/score1+pt2/score2
                return self.r*interp_pt/norm(interp_pt)


    def max_gradient_dir(self, valdict, orient=None, dir=None):
        """Find the direction of maximum gradient of a functional over
        the mesh patch, whose dictionary of
            mesh point index -> function value
        is given by valdict. The magnitudes of the values are
        compared. The direction is indicated by a Point object, a
        relative position at the patch's characteristic radius from
        p0.

        If the functional is constant everywhere on the patch (degenerate)
        then a random direction is returned. If the functional is constant
        everywhere except at the centre, then the vector (0,0) is returned
        if the centre has higher value; a random vector is returned
        otherwise.

        If the functional is partially degenerate on some subset of
        mesh points then the optional orient and dir arguments can
        provide a means to select, provided the mesh resolution is
        sufficiently high.

        orient is a vector (Point object) relative to p0 orienting a
        "positive" direction, such that the integer dir = -1 or +1
        selects one half of the mesh points on which to conduct the
        search.
        """
        return self.__max_grad(valdict, orient, dir, 1)

    def max_gradient_mesh(self, valdict, orient=None, dir=None):
        """Find the direction of maximum gradient of a functional over
        the mesh patch, whose dictionary of
            mesh point index -> function value
        is given by valdict. The magnitudes of the values are
        compared. The direction is returned as a pair containing the two
        closest mesh points to the direction (in mesh local coords).

        If the functional is constant everywhere on the patch (degenerate)
        then a random direction is returned. If the functional is constant
        everywhere except at the centre, then the vector (0,0) is returned
        if the centre has higher value; a random vector is returned
        otherwise.

        If the functional is partially degenerate on some subset of
        mesh points then the optional orient and dir arguments can
        provide a means to select, provided the mesh resolution is
        sufficiently high.

        orient is a vector (Point object) relative to p0 orienting a
        "positive" direction, such that the integer dir = -1 or +1
        selects one half of the mesh points on which to conduct the
        search.
        """
        return self.__max_grad(valdict, orient, dir, 0)

    def __max_grad(self, valdict, orient, dir, return_type):
        vals, ixs, use_vals = self.setup_grad(valdict, orient, dir)
        degen_everywhere = alltrue([v==vals[0] for v in vals])
        degen_exterior = alltrue([v==vals[1] for v in use_vals])
        degen_exterior_lower = degen_exterior and vals[0] >= vals[ixs[0]]
        degen_exterior_higher = degen_exterior and vals[0] < vals[ixs[0]]
        if degen_everywhere or degen_exterior_higher:
            # degenerate case -- return random vector
            x = uniform(-1,1)
            y = uniform(-1,1)
            vl = sqrt(x*x + y*y)
            x = x/vl
            y = y/vl
            if return_type == 0:
                raise RuntimeError("No valid directions to point in")
            else:
                return Point(dict(zip(self.p0.coordnames, [x,y])))
        elif degen_exterior_lower:
            if return_type == 0:
                raise RuntimeError("No valid directions to point in")
            else:
                return Point(dict(zip(self.p0.coordnames, [0,0])))
        d1 = (0, NaN)
        d2 = (0, NaN)
        for i, v in enumerate(use_vals):
            if ixs[i] == 0:
                # don't include centre point
                continue
            if abs(v) > d1[0]:
                d2 = d1
                d1 = (abs(v), ixs[i])
            elif abs(v) > d2[0]:
                d2 = (abs(v), ixs[i])
        # the two best mesh patch points (unless partially degenerate)
        pt1 = self.mesh_local[d1[1]]
        pt2 = self.mesh_local[d2[1]]
        if return_type == 0:
            return (pt1, pt2)
        else:
            score1 = d1[0]
            score2 = d2[0]
            interp_pt = pt1*score1+pt2*score2
            return self.r*interp_pt/norm(interp_pt)

    def __repr__(self):
        return "Mesh patch [" + ", ".join(["(" + \
                                str(m).replace(' ','').replace('\n',', ')+")" \
                                for m in self.mesh]) +"]"

    __str__ = __repr__



# -----------------------------------------------------------------------------

## PRIVATE CLASSES, FUNCTIONS, and CONSTANTS

# For use when operations don't allow symbol Inf
_num_inf = 1.0e100


def create_test_fn(gen, tmax, dist_pts):
    """Create integration test function using the supplied generator.

    The test function will return a dictionary of the two closest
    points (keys 1 and 2) mapping to their respective distances and
    pointset index positions.
    """

    gen.set(tdata=[0,tmax])

    def Tn_integ(ic):
        gen.set(ics=ic)
        try:
            test_traj = gen.compute('test')
        except:
            print "Problem integrating test trajectory at i.c. ", ic
            raise
        test_pts = test_traj.sample(coords=dist_pts.all_pts.coordnames)
        # distance of endpoint to pointset
        try:
            d_info = dist_pts(test_pts[-1], use_norm=True, minmax=['min'])
        except ValueError:
            # This error happens when fsolve tries initial conditions that
            # break the integrator
            return (_num_inf,NaN)
        pos = d_info['min'][1]['pos']
        return (test_pts[-1]-dist_pts.all_pts[pos], pos)
    return Tn_integ


def create_test_fn_with_events(gen, tmax, dist_pts, iso_ev, other_evnames, pars_to_vars):
    """Create integration test function using the supplied generator,
    assuming it contains isochron-related events.

    The test function will return a dictionary of the two closest
    points (keys 1 and 2) mapping to their respective distances and
    pointset index positions.
    """

    def Tn_integ(ic):
        gen.set(ics=ic, tdata=[0,tmax])
        try:
            test_traj = gen.compute('test')
        except:
            print "Problem integrating test trajectory at i.c. ", ic
            raise
        test_pts = test_traj.sample(coords=dist_pts.all_pts.coordnames)
        # distance of endpoint to pointset
        try:
            d_info = dist_pts(test_pts[-1], use_norm=True, minmax=['min'])
        except ValueError:
            # This error happens when fsolve tries initial conditions that
            # break the integrator
            return (_num_inf,NaN)
        # refine min position using isochron-related events
        q=test_pts[-1]
        perp_ev, t_ev = _find_min_pt(gen, q,
                                     d_info['min'][1]['pos'], dist_pts.all_pts,
                                     pars_to_vars, iso_ev, other_evnames)
        ev_pt = perp_ev[0][q.coordnames]
        # different return format to version w/o events
        return (test_pts[-1]-ev_pt, t_ev, ev_pt)
    return Tn_integ



def _xinf_ND(xdot,x0,args=(),xddot=None,xtol=1.49012e-8):
    """Private function for wrapping the fsolving for x_infinity
    for a variable x in N dimensions"""
    try:
        result = float(fsolve(xdot,x0,args,fprime=xddot,xtol=xtol,full_output=1))
    except (ValueError, TypeError, OverflowError):
        xinf_val = NaN
    except:
        print "Error in fsolve:", sys.exc_info()[0], sys.exc_info()[1]
        xinf_val = NaN
    else:
        if result[2] == 1:
            xinf_val = result[0]
        else:
            xinf_val = NaN
    return xinf_val


def _xinf_1D(xdot,x0,args=(),xddot=None,xtol=1.49012e-8):
    """Private function for wrapping the solving for x_infinity
    for a variable x in 1 dimension"""
    try:
        if xddot is None:
            xinf_val = float(fsolve(xdot,x0,args,xtol=xtol))
        else:
            xinf_val = float(newton_meth(xdot,x0,fprime=xddot,args=args))
    except RuntimeError:
        xinf_val = NaN
    return xinf_val


def _find_min_pt(gen, q, pos1, pts, pars_to_vars, iso_ev, other_evnames,
                 offset=4):
    ts = pts['t']
    if pos1 > offset:
        t_pos1n = ts[pos1-offset]
        p = pts[pos1-offset]
    else:
        dt = ts[pos1+1] - ts[pos1]
        t_pos1n = ts[pos1] - offset*dt
        p = pts[pos1-offset]  # OK b/c assume periodic orbit
    if pos1 < len(pts)-offset:
        t_pos1p = ts[pos1+offset]
    else:
        dt = ts[pos1] - ts[pos1-1]
        t_pos1p = ts[pos1] + offset*dt
    tdata_local = [0, t_pos1p-t_pos1n + 1e-5]
    gen.eventstruct.setActiveFlag(iso_ev, True)
    for evn in other_evnames:
        gen.eventstruct.setActiveFlag(evn, False)
    qpars = {}
    for parname, varname in pars_to_vars.items():
        qpars[parname] = q[varname]
    gen.set(ics=p, tdata=tdata_local, pars=qpars)
##    # temp
##    vs=dict(p)
##    vs['t']=tdata_local[0]
##    print "\n"
##    print "dist 1 = ", norm(p-q)
##    print "ev fn 1 = ", gen.eventstruct.events[iso_ev]._fn(vs, gen.pars)
##    print "ptset min dist = ", norm(pts[pos1]-q)
##    vs=dict(pts[pos1+2])
##    vs['t']=tdata_local[1]
##    print "dist 2 = ", norm(pts[pos1+2]-q)
##    print "ev fn 2 = ", gen.eventstruct.events[iso_ev]._fn(vs, gen.pars)
##    # end temp
    test_traj2 = gen.compute('test2')
    gen.eventstruct.setActiveFlag(iso_ev, False)
    # assert presence of terminal event
    perp_ev = gen.getEvents()[iso_ev]
    try:
        assert len(perp_ev) == 1
        t_ev = perp_ev['t'][0] + tdata_local[0]
    except:
        print "Event name:", iso_ev
        print perp_ev
        raise ValueError("No nearest point found with event")
    return (perp_ev, t_ev)


class base_n_counter(object):
    """Simple counter in base-n, using d digits.
    Resets to zeros when state (n-1, n-1, ..., n-1) is
    incremented.

    base_n_counter(n, d)

    Public attribute:
        counter

    Implements methods:
        inc            -- increment counter (returns nothing)
        __getitem__    -- return ith component of counter
        reset          -- set counter to (0, 0, ..., 0)
    """
    def __init__(self, n, d):
        self._maxval = n-1
        self._d = d
        self.counter = np.zeros((d,))

    def inc(self):
        ix = 0
        while True:
            if ix == self._d:
                self.counter = np.zeros((self._d,))
                break
            if self.counter[ix] < self._maxval:
                self.counter[ix] += 1
                break
            else:
                self.counter[ix] = 0
                ix += 1

    def __getitem__(self, i):
        try:
            return self.counter[i]
        except IndexError:
            raise IndexError("Invalid index for counter")

    def reset(self):
        self.counter = np.zeros((self._d,))

    def __str__(self):
        return str(self.counter.tolist())

    __repr__ = __str__

