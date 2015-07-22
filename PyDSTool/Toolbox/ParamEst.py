"""Parameter estimation classes for ODEs.

   Robert Clewley.
"""

from __future__ import division, absolute_import, print_function

# PyDSTool imports
from PyDSTool.Points import Point, Pointset
from PyDSTool.Model import Model
from PyDSTool.common import Utility, _seq_types, metric, args, sortedDictValues, \
     remain, metric_L2, metric_L2_1D, metric_float, metric_float_1D
from PyDSTool.utils import intersect, filteredDict
from PyDSTool.errors import *
from PyDSTool import common
from PyDSTool.core.context_managers import RedirectNoOp, RedirectStderr, \
    RedirectStdout
from PyDSTool.ModelContext import qt_feature_leaf, process_raw_residual
from PyDSTool.Toolbox.optimizers import *

try:
    from constraint import Problem, FunctionConstraint, RecursiveBacktrackingSolver, \
         BacktrackingSolver, MinConflictsSolver
except ImportError:
    # constraint package must be installed to use some parameter estimation features:
    # http://labix.org/python-constraint
    Problem = None
    FunctionConstraint = None
    RecursiveBacktrackingSolver = None
    BacktrackingSolver = None
    MinConflictsSolver = None

from scipy.optimize import minpack, optimize
from numpy.linalg import norm, eig, eigvals, svd
from scipy.linalg import svdvals
from scipy.io import *
import sys, traceback
import operator

from numpy import linspace, array, arange, zeros, sum, power, \
     swapaxes, asarray, ones, alltrue, concatenate, ravel, argmax, \
     argmin, argsort, float, sign
import numpy as np

import math, types
from copy import copy, deepcopy
# ------------------------------------------------------------------------

_pest_classes = ['ParamEst', 'LMpest', 'BoundMin', 'residual_fn_context',
                 'residual_fn_context_1D', 'L2_feature', 'L2_feature_1D']

_deprecated_functions = ['get_slope_info', 'get_extrema',
           'compare_data_from_events', 'get_extrema_from_events']

_ctn_functions = ['do_2Dstep', 'do_2Ddirn', 'ctn_residual_info']

_generic_opt = ['make_opt', 'restrict_opt']

_utils = ['sweep1D', 'filter_feats', 'filter_pars', 'select_pars_for_features',
          'grad_from_psens', 'norm_D_sum', 'filter_iface', 'organize_feature_sens']

_errors = ['Converged', 'ConstraintFail']

__all__ = _pest_classes + _deprecated_functions + _ctn_functions + \
        _generic_opt + _utils + _errors


class Converged(PyDSTool_Error):
    pass

class ConstraintFail(PyDSTool_Error):
    pass


solver_lookup = {'RecursiveBacktrackingSolver': RecursiveBacktrackingSolver,
           'BacktrackingSolver': BacktrackingSolver,
           'MinConflictsSolver': MinConflictsSolver}

# ----
# Used to suppress output from legacy codes


# ----------------------------------------------------------------------------

# Simple parameter continuation of residual vector (maintain residual constant
# while varying free parameters) -- assumes locally 'regular' landscape, esp.
# no folds, low curvature

def do_2Dstep(fun, p, dirn, maxsteps, stepsize, atol, i0, orig_res, orig_dirn,
              all_records):
    """
    Residual vector continuation step in 2D parameter space.

    orig_dirn corresponds to direction of positive dirn, in case when
    re-calculating gradient the sign flips"""
    record = {}
    print("Recalculating gradient")
    grad = fun.gradient(p)
    neut = np.array([grad[1], -grad[0]])
    neut = neut/norm(neut)
    if np.sign(dot(neut, orig_dirn)) != 1:
        print("(neut was flipped for consistency with direction)")
        neut = -neut
    print("Neutral direction:", neut)
    record['grad'] = grad
    record['neut'] = neut
    residuals = []
    # inner loop - assumes curvature will be low (no adaptive step size)
    print("\n****** INNER LOOP")
    new_pars = copy(p)
    i = 0
    while True:
        if i > maxsteps:
            break
        new_pars += dirn*stepsize*neut
        res = fun(new_pars)
        d = abs(res-orig_res)
        if res > 100 or d > atol:
            # fail
            break
        step_ok = d < atol/2.
        if step_ok:
            r = (copy(new_pars), res)
            residuals.append(r)
            all_records[i0+dirn*i] = r
            num_dirn_steps = len([k for k in all_records.keys() if \
                                  k*dirn >= abs(i0)])
            i += 1
            print(len(all_records), "total steps taken, ", num_dirn_steps, \
                    "in since grad re-calc: pars =", new_pars, " res=", res)
        else:
            # re-calc gradient
            break
    if len(residuals) > 0:
        record['p_new'] = residuals[-1][0]
        record['n'] = i
        record['i0_new'] = i0+dirn*i
    else:
        record['p_new'] = p
        record['n'] = 0
        record['i0_new'] = i0
    return record


def do_2Ddirn(fun, p0, dirn, maxsteps, stepsize, atol, orig_res, orig_dirn,
              all_records):
    """
    Residual vector continuation in a single neutral direction in 2D parameter
    space, given by dirn = +1/-1 from point p0.

    maxsteps is *per* direction"""
    print("\nStarting direction:", dirn)
    dirn_rec = []
    p = p0
    if dirn == 1:
        # will count 0, 1, ...
        i = 0
    else:
        # will count -1, -2, ...
        i = -1
    done = False
    while not done:
        print("Steps from i =", i)
        r = do_step(fun, p, dirn, maxsteps, stepsize, atol, i, orig_res,
                    orig_dirn, all_records)
        if r['n'] == 0:
            # no steps taken successfully
            done = True
        else:
            dirn_rec.append(r)
            p = r['p_new']
            i = r['i0_new']
        if abs(i) > maxsteps:
            done = True
    return dirn_rec


def ctn_residual_info(recs, do_plot=False):
    """Temporary helper function for use with continuation functions."""
    ixlims = [min(recs.keys()), max(recs.keys())]
    if do_plot:
        from PyDSTool.matplotlib_import import plt
        plt.figure()
    r = []
    dr = []
    old_res = None
    for i in range(ixlims[0], ixlims[1]+1):
        pars, res = recs[i]
        r.append(res)
        if old_res is not None:
            dr.append(-np.log(abs(res-old_res)/old_res))
        old_res = res
        if do_plot:
            plt.plot(pars[0],pars[1],'ko')
    return r, dr



def sweep1D(fun, interval, resolution):
    numpoints = (interval[1]-interval[0])/resolution + 1
    ps = linspace(interval[0], interval[1], numpoints)
    res = []
    for p in ps:
        res.append(fun(array([p])))
    return ps, array(res)


class residual_fn_context(helpers.ForwardFiniteDifferencesCache):
    def _res_fn(self, p, extra_args=None):
        # p comes in as an array
        pest = self.pest
        for i, parname in enumerate(pest.freeParNames):
            pest.modelArgs[pest.parTypeStr[i]][parname] = p[i]
        pest.testModel.set(**pest.modelArgs)
        try:
            return pest.evaluate()
        except KeyboardInterrupt:
            raise
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("******************************************")
            print("Problem evaluating residual function")
            print("  ", exceptionType, exceptionValue)
            for line in traceback.format_exc().splitlines()[-4:-1]:
                print("   " + line)
            print("  originally on line:", traceback.tb_lineno(exceptionTraceback))
            if self.pest.verbose_level > 1:
                raise
            else:
                print("(Proceeding with penalty values)\n")
            return 10*ones(pest.context.res_len)

class residual_fn_context_1D(helpers.ForwardFiniteDifferencesCache):
    def _res_fn(self, p, extra_args=None):
        # p comes in as an array
        pest = self.pest
        pest.modelArgs[pest.parTypeStr][pest.freeParNames[0]] = p
        pest.testModel.set(**pest.modelArgs)
        try:
            return pest.evaluate()[0]
        except KeyboardInterrupt:
            raise
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("******************************************")
            print("Problem evaluating residual function")
            print("  ", exceptionType, exceptionValue)
            for line in traceback.format_exc().splitlines()[-4:-1]:
                print("   " + line)
            print("  originally on line:", traceback.tb_lineno(exceptionTraceback))
            if self.pest.verbose_level > 1:
                raise
            else:
                print("(Proceeding with penalty value)\n")
            return 100


# ----------------------------------------------------------------------------

## EXPERIMENTAL FUNCTIONS (in development)


def grad_from_psens(psens, pest):
    pd2a = pest.pars_dict_to_array
    pd = {}
    for pname, fsens_dict in psens.items():
        res_list = []
        for feat, sens_array in fsens_dict.items():
            res_list.extend(list(sens_array))
        pd[pname] = sum()
    return pd2a(pd)


def filter_feats(parname, feat_sens):
    """Filter features whose residual vectors show a *net* increase (dirn=1)
    or decrease (dirn=-1) as one parameter is varied. Provided the
    sensitivities were measured appropriately, dirn=0 will select any
    non-smoothly changing features (e.g. discrete-valued).

    feat_sens is a dictionary of feature sensitivities keyed by parameter
    name, e.g. as returned by the ParamEst.par_sensitivity method.

    Returns a list of ((model interface, feature), sensitivity) pairs, where
    the feature belongs to the model interface (in case of duplication in
    multiple interfaces), and the sensitivity is the absolute value of the
    net increase/decrease. The lists are ordered by decreasing
    magnitude of sensitivity.

    Definition of net increase:
    e.g. if sensitivity for a given feature with a 3-vector residual is
    [-0.1 0.4 1.5] then the sum is +1.8 and will be selected for the
    'increasing' direction.
    """
    incr = []
    decr = []
    neut = []
    for mi, fdict in feat_sens[parname].items():
        for f, sens in fdict.items():
            sum_sens = sum(sens)
            sign_ss = np.sign(sum_sens)
            abs_ss = abs(sum_sens)
            if alltrue(sign_ss == 1):
                incr.append(((mi,f), abs_ss, parname))
            elif alltrue(sign_ss == -1):
                decr.append(((mi, f), abs_ss, parname))
            else:
                neut.append(((mi, f), 0.))
    return sorted(incr, reverse=True, key=operator.itemgetter(1)), \
           sorted(decr, reverse=True, key=operator.itemgetter(1)), neut


def filter_pars(mi_feat, feat_sens):
    """
    For a given (model interface, feature) pair, find all parameters
    that change the feature's net residual in the same direction, or not at all.
    (Provided the sensitivities were measured appropriately, this will select
    any non-smoothly changing features (e.g. discrete-valued)).

    feat_sens is a dictionary of feature sensitivities keyed by parameter
    name, e.g. as returned by the ParamEst.par_sensitivity method.

    Returns a triple of lists of (parameter names, sensitivity) pairs:
    increasing, decreasing, and neutral. The lists are ordered by decreasing
    magnitude of sensitivity.
    """
    mi, feat = mi_feat
    incr = []
    decr = []
    neut = []
    for pname, fdict in feat_sens.items():
        sens = fdict[mi][feat]
        sum_sens = sum(sens)
        sign_ss = np.sign(sum_sens)
        abs_ss = abs(sum_sens)
        if alltrue(sign_ss == 1):
            incr.append((pname, abs_ss, (mi, feat)))
        elif alltrue(sign_ss == -1):
            decr.append((pname, abs_ss, (mi, feat)))
        else:
            neut.append((pname, 0., (mi, feat)))
    return sorted(incr, reverse=True, key=operator.itemgetter(1)), \
           sorted(decr, reverse=True, key=operator.itemgetter(1)), neut


def _present_and_sensitive(xsf, L, thresh, rejected, neutral):
    """Helper function to return whether x is present in list L with an
    associated sensitivity s larger than thresh, where L is made up of
    (y, ysens) pairs, provided the sensitivities are larger than threshold.

    xsf is the triple (x, xsens, feature). rejected and neutral arguments
    should be lists to store the new rejected and neutral items.
    """
    x, xs, xf = xsf
    flag = None
    p_and_s = False
    for i, (y, ys, yf) in enumerate(L):
        if x == y:
            # present
            x_is_sens = xs >= thresh
            y_is_sens = ys >= thresh
            if x_is_sens and y_is_sens:
                # clash
                p_and_s = True
                flag = i
                if xs > ys:
                    rejected.append((x, xs, xf))
                else:
                    rejected.append((y, ys, yf))
            elif x_is_sens and not y_is_sens:
                # effecively not present in L, so delete
                p_and_s = False
                flag = i
            elif not x_is_sens and y_is_sens:
                # effectively present in L
                p_and_s = True
            else:
                # not x_is_sens and not y_is_sens
                # ... effectively not present in L but x too small to be
                # considered present in result anyway, so treat as p_and_s = True
                # (will be considered neutral)
                p_and_s = True
                flag = i
                if xs > ys:
                    neutral.append((x, xs, xf))
                else:
                    neutral.append((y, ys, yf))
            break
    if flag is not None:
        # filter out item so that it's not kept in L
        del L[i]
    return p_and_s


def _makeUnique(L):
    seen = {}
    for x, s, f in L:
        if x in seen:
            if s > seen[x][0]:
                seen[x] = (s, f)
        else:
            seen[x] = (s, f)
    Lu = [(x,pair[0], pair[1]) for x, pair in seen.items()]
    return sorted(Lu, reverse=True, key=operator.itemgetter(1))

def pp(l):
    """List pretty printer"""
    print("[", end=' ')
    for x in l:
        print(x, ",")
    print("]")


# ---------------------------------


def select_pars_for_features(desired_feats, feat_sens, deltas, neg_tol=0, pos_tol=0.001,
                             method='RecursiveBacktrackingSolver',
                             forwardCheck=True, verbose=False):
    """Default tol > 0 in case there are undifferentiable features that will
    never lead to satisfaction of constraints with tol=0.

    Returns a problem object and the by-parameter sensitivity dictionary of
    derivatives, D."""
    try:
        solver = solver_lookup[method]
    except KeyError:
        print("Specified solver not found. Will try recursive backtracking solver")
        solver = RecursiveBacktrackingSolver
    try:
        problem = Problem(solver(forwardCheck))
    except TypeError:
        print("Install constraint package from http://labix.org/python-constraint")
        raise ImportError("Must have constraint package installed to use this feature")
    global D, feat_name_lookup, pars, par_deltas
    par_deltas = deltas
    pars = list(feat_sens.keys())
    pars.sort()
    num_pars = len(pars)
    assert len(par_deltas) == num_pars

    D = {}

    max_val = 0
    for par in pars:
        for mi, fdict in feat_sens[par].items():
            for f, ra in fdict.items():
                val = sum(ra)
                if norm(val) > max_val:
                    max_val = val
                if f not in D:
                    D[f] = {}
                D[f][par] = val


    values = [-1, 0, 1]
    problem.addVariables(pars, values)

    feat_name_lookup = {}
    for f in D.keys():
        feat_name_lookup[f.name] = f

    constraint_funcs = []
    for f in D.keys():
        if f in desired_feats:
            op = 'operator.le'
            this_tol = neg_tol
        else:
            op = 'operator.lt'
            this_tol = pos_tol
        code =  "def D%s(*p):\n"%f.name
        if verbose:
            code += "  print '%s', p\n"%f.name
        code += "  pd = D[feat_name_lookup['%s']]\n"%f.name
        code += "  prods = [pd[par]*par_deltas[pi]*p[pi] for pi, par in enumerate(pars)]\n"
        if verbose:
            code += "  print sum(prods)\n"
        code += "  return %s(sum(prods), %f)\n"%(op,this_tol)
        exec(code)
        constraint_funcs.append(locals()['D'+f.name])

    for func in constraint_funcs:
        problem.addConstraint(FunctionConstraint(func), pars)

    print("Use problem.getSolution() to find a solution")
    return problem, D


def norm_D_sum(D_sum):
    """Normalize a D_sum by the elements for each feature by
    smallest absolute size (that element becomes 1 in norm).
    For unweighted feature sensitivities, or else it
    unweights weighted ones."""
    D_n = {}
    for feat, pD in D_sum.items():
        pD_min_abs = min(abs(array(pD.values())))
        D_n[feat] = {}
        for pname, pD_val in pD.items():
            D_n[feat][pname] = pD_val/pD_min_abs
    return D_n

def filter_iface(psens, iface):
    ws = {}
    for parname, wdict in psens.items():
        ws[parname] = {iface: wdict[iface]}
    return ws

def organize_feature_sens(feat_sens, discrete_feats=None):
    # Model interface is currently ignored -- assumes no clashing feature names
    # between related MIs
    if discrete_feats is None:
        discrete_feats = []
    pars = list(feat_sens.keys())
    pars.sort()

    D_sum = {}
    D_vec = {}

    max_val = 0
    for par in pars:
        for mi, fdict in feat_sens[par].items():
            for f, ra in fdict.items():
                val = sum(ra)
                if norm(val) > max_val:
                    max_val = val
                if f not in D_sum and not(f in discrete_feats):
                    D_sum[f] = {}
                    D_vec[f] = {}
                if f in discrete_feats:
                    continue
                else:
                    D_sum[f][par] = val
                    D_vec[f][par] = ra
    return D_sum, D_vec


def make_opt(pnames, resfnclass, model, context, parscales=None,
             parseps=None, parstep=None, parlinesearch=None,
             stopcriterion=None, grad_ratio_tol=10,
             use_filter=False, verbose_level=2):
    """Create a ParamEst manager object and an instance of an optimizer from the
    Toolbox.optimize sub-package, returned as a pair.

    Inputs:

    pnames:     list of free parameters in the model
    resfnclass: residual function class (e.g. residual_fn_context_1D exported
                from this module)
    model:      the model to optimize, of type Model (not a Generator)
    context:    the context object that defines the objective function
                criteria via "model interfaces" and their features, etc.
    parscales:  for models that do not have parameters varying over similar
                scales, this dictionary defines what "O(1)" change in dynamics
                refers to for each parameter. E.g. a parameter that must change by
                several thousand in order to make an O(1) change in model output
                can have its scale set to 1000. This will also be the maximum
                step size in that direction for the Scaled Line Search method, if used.
                Defaults to 10*parseps for each parameter.
    parseps:    dictionary to indicate what change in parameter value to use for
                forward finite differencing, for reasons similar to those given in
                description of the parscales argument. Default is 1e-7 for each parameter.
    parstep:    choice of optimization algorithm stepper, defaults to
                conjugate gradient step.CWConjugateGradientStep.
    parlinesearch:  choice of line search method, defaults to scaled
                    line search method line_search.ScaledLineSearch.
    stopcriterion:  choice of stop criteria for the optimization iterations. Defaults to
                    ftol=1e-7, gtol=1e-7, iterations_max=200.
    grad_ratio_tol: For residual functions with poor smoothness in some directions,
                    this parameter (default = 10) prevents those directions being used
                    for gradient information if the ratio of residual values found during
                    finite differencing is greater in magnitude than this tolerance value.
                    (Experimental option only -- set very large, e.g. 1e6 to switch off).
    use_filter:     activate use of filtering out largest directions of gradients that may
                    be unreliable. Default is False. (Experimental option only).
    verbose_level:  Default to 2 (high verbosity).
    """
    parnames = copy(pnames)
    parnames.sort()
    if parscales is None:
        freepars = parnames
    else:
        freepars = filteredDict(parscales, parnames)
    if parseps is None:
        parseps = {}.fromkeys(parnames, 1e-7)
    if parscales is None:
        parscales = parseps.copy()
        for k, v in parscales.items():
            parscales[k] = 10*v
    pest = ParamEst(freeParams=freepars,
                 testModel=model,
                 context=context,
                 residual_fn=resfnclass(eps=[parseps[p] for p in parnames],
                                        grad_ratio_tol=grad_ratio_tol),
                 verbose_level=verbose_level
                )
    if parstep is None:
        parstep = step.CWConjugateGradientStep()
    if parlinesearch is None:
        parlinesearch = line_search.ScaledLineSearch(max_step=[parscales[p] for \
                                                p in parnames], filter=use_filter)
    if stopcriterion is None:
        stopcriterion = criterion.criterion(ftol=1e-7, gtol=1e-7,
                              iterations_max=200)
    return pest, optimizer.StandardOptimizer(function=pest.fn,
                    step=parstep,
                    line_search=parlinesearch,
                    criterion=stopcriterion,
                    x0=pest.pars_dict_to_array(pest.testModel.pars))


def restrict_opt(pest, feat_list, opt, pars=None):
    """Restrict parameter estimation to certain features and parameters.

    If pars is None (default) then all free parameters of pest are used.
    """
    if pars is None:
        pars = pest.freeParNames
    if parseps is None:
        parseps = {}.fromkeys(pest.freeParNames, 1e-7)
    if parstep is None:
        parstep = step.CWConjugateGradientStep()
    if parlinesearch is None:
        parlinesearch = line_search.ScaledLineSearch(max_step = \
                                [pest.parScales[p] for p in pars])
    if stopcriterion is None:
        stopcriterion = criterion.criterion(ftol=1e-7, gtol=1e-7,
                              iterations_max=100)
    new_pest = ParamEst(context=pest.context,
                        freeParams=filteredDict(pest.parScales, pars),
                        testModel=pest.testModel,
                        verbose_level=pest.verbose_level)
    new_fn = pest.fn.__class__(eps=[parseps[p] for p in pars],
                               pest=new_pest)
    new_pest.setFn(new_fn)
#    full_feat_list = pest.context.res_feature_list
#    wdict = {}
#    for mi, feat in full_feat_list:
#        if (mi, feat) not in feat_list:
#            try:
#                wdict[mi][feat] = 0
#            except KeyError:
#                wdict[mi] = {feat: 0}
    # leave weights for selected features at their previous values from
    # pest.context
#    new_pest.context.set_weights(wdict)
    new_pest.fn.pest = new_pest   # otherwise logging goes to wrong place
    new_opt = optimizer.StandardOptimizer(function=new_pest.fn, step=parstep,
                            line_search=parlinesearch,
                            criterion=stopcriterion,
                            x0=new_pest.pars_dict_to_array(pest.testModel.pars))
    return new_pest, new_opt



class L2_feature_1D(qt_feature_leaf):
    """Use with scalar optimizers such as BoundMin"""
    def _local_init(self):
        self.metric = metric_L2_1D()
        if hasattr(self.pars, 'num_samples'):
            self.metric_len = self.pars.num_samples
        else:
            self.metric_len = len(self.pars.t_samples)

    def postprocess_ref_traj(self):
        if hasattr(self.pars, 'num_samples'):
            tvals = linspace(self.pars.trange[0], self.pars.trange[1],
                         self.metric_len)
        else:
            tvals = self.pars.t_samples
        self.pars.tvals = tvals
        self.pars.ref_samples = self.ref_traj(tvals, coords=[self.pars.coord])

    def evaluate(self, target):
        return self.metric(self.pars.ref_samples,
                           target.test_traj(self.pars.tvals,
                                            self.pars.coord)) < self.pars.tol

class L2_feature(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_L2()
        if hasattr(self.pars, 'num_samples'):
            self.metric_len = self.pars.num_samples
        else:
            self.metric_len = len(self.pars.t_samples)

    def postprocess_ref_traj(self):
        if hasattr(self.pars, 'num_samples'):
            tvals = linspace(self.pars.trange[0], self.pars.trange[1],
                         self.metric_len)
        else:
            tvals = self.pars.t_samples
        self.pars.tvals = tvals
        self.pars.ref_samples = self.ref_traj(tvals, coords=[self.pars.coord])

    def evaluate(self, target):
        return self.metric(self.pars.ref_samples,
                           target.test_traj(self.pars.tvals,
                                            self.pars.coord)) < self.pars.tol


class ParamEst(Utility):
    """General-purpose parameter estimation class.
    freeParams keyword initialization argument may be a list of
    names or a dictionary of scales for determining appropriate
    step sizes for O(1) changes in the residual function.

    In its absence, the scales will default to 1.
    """

    def __init__(self, **kw):
        self.needKeys = ['freeParams', 'testModel', 'context']
        self.optionalKeys = ['verbose_level', 'residual_fn', 'extra_pars']
        try:
            self.context = kw['context']
            if isinstance(kw['freeParams'], list):
                self.freeParNames = kw['freeParams']
                self.numFreePars = len(self.freeParNames)
                self.parScales = dict.fromkeys(self.freeParNames, 1)
            else:
                self.parScales = kw['freeParams']
                self.freeParNames = list(self.parScales.keys())
                self.numFreePars = len(self.freeParNames)
            self.freeParNames.sort()
            self.testModel = kw['testModel']
            assert isinstance(self.testModel, Model), \
                   "testModel argument must be a Model instance"
            self._algParamsSet = False
        except KeyError:
            raise PyDSTool_KeyError('Incorrect argument keys passed')
        self.foundKeys = len(self.needKeys)   # lazy way to achieve this!
        if 'residual_fn' in kw:
            self.setFn(kw['residual_fn'])
            self.foundKeys += 1
        else:
            try:
                res_fn = res_fn_lookup[self.__class__]
            except KeyError:
                raise ValueError("Must explicitly set residual function for this class")
            else:
                self.setFn(res_fn(pest=self))
        if 'extra_pars' in kw:
            self._extra_pars = kw['extra_pars']
            self.foundKeys += 1
        else:
            self._extra_pars = {}
        self.parsOrig = {}
        # in case explicit jacobian of residual fn is not present
        self._residual_fn_jac = None
        if 'verbose_level' in kw:
            self.verbose_level = kw['verbose_level']
            self.foundKeys += 1
        else:
            self.verbose_level = 0
        # Set up model arguments (parameter value will be set before needed)
        self.modelArgs = {}
        self.resetParArgs()
        # used for Ridders' method output statistics if selected for
        # calculating gradient using gradient or Hessian methods
        self._grad_info = {}
        if self.foundKeys < len(kw):
            raise PyDSTool_KeyError('Incorrect argument keys passed')
        self.reset_log()


    def resetParArgs(self):
        self.parTypeStr = []
        for i in range(self.numFreePars):
            if self.freeParNames[i] in self.testModel.obsvars:
                # for varying initial conditions
                self.parTypeStr.append('ics')
                if 'ics' not in self.modelArgs:
                    self.modelArgs['ics'] = {}
            elif self.freeParNames[i] in self.testModel.pars:
                # for varying regular pars
                self.parTypeStr.append('pars')
                if 'pars' not in self.modelArgs:
                    self.modelArgs['pars'] = {}
            else:
                raise ValueError("free parameter '"+self.freeParNames[i]+"'"\
                                   " not found in test model")
            # initialize model argument to None
            self.modelArgs[self.parTypeStr[i]][self.freeParNames[i]] = None


    def setAlgParams(self, *args):
        """Set algorithmic parameters."""
        raise NotImplementedError("This is only an abstract function "
                                  "definition")

    def setFn(self, fn):
        self.fn = fn
        # reciprocal reference
        self.fn.pest = self


    def evaluate(self, extra_record_info=None):
        """Evaluate residual vector, record result, and display step
        information (if verbose).
        """
        res, raw_res = self.context.residual(self.testModel, include_raw=True)
        log_entry = args(pars=filteredDict(self.testModel.query('pars'),
                                                 self.freeParNames),
                             ics=filteredDict(self.testModel.query('ics'),
                                              self.freeParNames),
                             weights=self.context.weights,
                             residual_vec=res,
                             raw_residual_vec=raw_res,
                             residual_norm=norm(res),
                             trajectories=[copy(ref_mi.get_test_traj()) for \
                              ref_mi in self.context.ref_interface_instances])
        self.log.append(log_entry)
        key = {}
        key.update(log_entry.pars)
        key.update(log_entry.ics)
        self._keylogged[tuple(sortedDictValues(key))] = log_entry
        if extra_record_info is not None:
            self.log[-1].update({'extra_info': args(**extra_record_info)})
        if norm(res) < self.log[self._lowest_res_log_ix].residual_norm:
            # this is now the lowest recorded, so make note of this index into self.log
            self._lowest_res_log_ix = len(self.log)-1
        if self.verbose_level > 0:
            self.show_log_record(self.iteration)
        self.iteration += 1
        return res


    def reset_log(self):
        self.iteration = 0
        self.log = []
        self._lowest_res_log_ix = 0
        # key logged is used for faster cache lookup using pars + ics
        self._keylogged = {}

    def key_logged_residual(self, pars_ics, weights):
        """pars_ics must be a sequence type"""
        try:
            log_entry = self._keylogged[tuple(pars_ics)]
        except KeyError:
            raise KeyError("Pars and ICs not found in log record")
        if all(log_entry.weights == weights):
            return log_entry.residual_vec
        else:
            return process_raw_residual(log_entry.raw_residual_vec, weights)


    def find_logs(self, res_val=None, condition='<'):
        """Find log entries matching given condition on their residual norm
        values. Returns a list of log indices.

        if res_val is not given, the residual norm of the first entry in the
        current log is used.

        Use '<' and '>' for the condition argument (default is <).
        """
        if res_val is None:
            res_val = self.log[0].residual_norm
        res_data = array([e.residual_norm for e in self.log])
        sort_ixs = argsort(res_data)
        if condition == '<':
            ix = argmin(res_data[sort_ixs] < res_val)
            return list(sort_ixs[:ix])
        elif condition == '>':
            ix = argmax(res_data[sort_ixs] > res_val)
            return list(sort_ixs[ix:])

    def show_log_record(self, i, full=False):
        """Use full option to show residuals mapped to their feature names,
        including information about weights."""
        try:
            entry = self.log[i]
        except IndexError:
            raise ValueError("No such call %i recorded"%i)
        print("\n  **** Call %i"%i, "Residual norm: %f"%entry.residual_norm)
        if entry.ics != {}:
            print("Ics:", entry.ics)
        if entry.pars != {}:
            print("Pars:", entry.pars)
        if full:
            print("Res:\n")
            self.context.show_res_info(entry.residual_vec)
        else:
            print("Res:", entry.residual_vec)

    def pars_to_ixs(self):
        all_pars = sortedDictKeys(self.testModel.pars)
        inv_ixs = [all_pars.index(p) for p in self.freeParNames]
        inv_ixs.sort()
        return inv_ixs

    def pars_array_to_dict(self, parray):
        return dict(zip(self.freeParNames, parray))

    def pars_dict_to_array(self, pdict):
        return array(sortedDictValues(filteredDict(pdict, self.freeParNames)))

    def par_sensitivity(self, pdict=None, non_diff_feats=None, extra_info=False):
        """Parameter sensitivity of the context's features at the free parameter
        values given as a dictionary or args. If none provided, the current
        test model parameter values will be used. A dictionary mapping parameter names to
          {interface_instance: {feat1: sensitivity_array, ..., featn: sensitivity_array}}
        is returned.

        Specify any non-differentiable features in the non_diff_feats list
        as pairs (interface instance, feature instance).

        Sensitivity entry > 0 means that increasing the parameter will
        increase the absolute value of that residual, i.e. worsen the "fit".

        extra_info optional argument makes this method return both the feature sensitivity
        dictionary and a dictionary containing additional information to reconstruct
        the gradient of the residual norm, to save re-calculation of it at this point.
        This gradient will also respect the non_diff_feats argument, if provided.
        """
        old_weights = self.context.weights
        self.context.reset_weights()
        wdict={}
        if non_diff_feats is not None:
            for mi, f in non_diff_feats:
                if mi in wdict:
                    wdict[mi][f] = 0
                else:
                    wdict[mi] = {f:0}
        self.context.set_weights(wdict)
        if pdict is None:
            pdict = filteredDict(self.testModel.pars, self.freeParNames)
        p = self.pars_dict_to_array(pdict)
        f = self.fn.residual
        res = f(p)
        if extra_info:
            info_dict = {'res': res}
            res_dict = {}
            grad_dict = {}
        feat_sens = {}
        for pi, pn in enumerate(self.freeParNames):
            p_copy = p.copy()
            try:
                h = self.fn.eps[pi]
            except TypeError:
                # scalar
                h = self.fn.eps
            p_copy[pi] += h
            res_eps = f(p_copy)
            # multiple and check inclusive inequality in case sign_res has
            # components at exactly 0
            assert alltrue(res * res_eps >= 0), "step for %s too large"%pn
            D_res = (res_eps-res)/h
            if extra_info:
                res_dict[pn] = (res_eps, h)
                grad_dict[pn] = (norm(old_weights*res_eps)-norm(old_weights*res))/h
            feat_sens[pn] = self.context._map_to_features(D_res)
        self.context.reset_weights(old_weights)
        if extra_info:
            info_dict['res_dict'] = res_dict
            info_dict['weights'] = old_weights
            info_dict['grad'] = grad_dict
            return feat_sens, info_dict
        else:
            return feat_sens

    def weighted_par_sensitivity(self, feat_sens):
        """Return parameter sensitivities weighted according to current feature
        weights, based on a previous output from par_sensitivity method.
        """
        ws = self.context.feat_weights
        wfeat_sens = {}
        for pn, sensdict in feat_sens.items():
            pd = wfeat_sens[pn] = {}
            for mi, fdict in sensdict.items():
                md = pd[mi] = {}
                for f, sens in fdict.items():
                    md[f] = sens*ws[(mi,f)]
        return wfeat_sens

    def run(self):
        """Run parameter estimation. Returns a dictionary:

            'success' -> boolean
            'pars_sol' -> fitted values of pars
            'pars_orig' -> original values of optimized pars
            'sys_sol' -> trajectory of best fit Model trajectory
            'alg_results' -> all other algorithm information (list)
        """
        raise NotImplementedError("This is only an abstract method definition")

    def iterate(self):
        raise NotImplementedError("This is only an abstract method definition")


class LMpest(ParamEst):
    """Unconstrained least-squares parameter and initial condition optimizer
    for n-dimensional DS trajectories. Fits N-dimensional parameter spaces.

    Uses MINPACK Levenberg-Marquardt algorithm wrapper from SciPy.minimize.
    """

    def setAlgParams(self, changed_parDict=None):
        # defaults
        parDict = {
                 'residuals' : None,
                 'p_start'   : None,
                 'args'      : None,
                 'Dfun'      : None,
                 'full_output' : 1,
                 'col_deriv'   : 0,
                 'ftol'        : 5e-5,
                 'xtol'        : 5e-5,
                 'gtol'        : 0.0,
                 'maxfev'      : 100,
                 'epsfcn'      : 0.0,
                 'factor'      : 100,
                 'diag'        : None
                 }

        if changed_parDict is None:
            changed_parDict = {}
        parDict.update(copy(changed_parDict))
        assert len(parDict) == 13, "Incorrect param dictionary keys used"

        self._residuals   = parDict['residuals']
        self._p_start     = parDict['p_start']
        self._args        = parDict['args']
        self._Dfun        = parDict['Dfun']
        self._full_output = parDict['full_output']
        self._col_deriv   = parDict['col_deriv']
        self._ftol        = parDict['ftol']
        self._xtol        = parDict['xtol']
        self._gtol        = parDict['gtol']
        self._maxfev      = parDict['maxfev']
        self._epsfcn      = parDict['epsfcn']
        self._factor      = parDict['factor']
        self._diag        = parDict['diag']
        # flag for run() to start
        self._algParamsSet = True


    def run(self, parDict=None, extra_pars=None, verbose=False):
        """Begin parameter estimation run.

        parDict can include arbitrary additional runtime arguments to
        the residual function.

        If tmesh is not supplied an attempt will be made to create one
        from the goal trajectory's independent domain limits, if the
        trajectory has been provided. Default mesh resolution is 20 points.
        """
        if parDict is None:
            parDict_new = {}
        else:
            parDict_new = copy(parDict)
        self._extra_pars = extra_pars
        parsOrig = []
        self.numFreePars = len(self.freeParNames)
        self.resetParArgs()
        for i in range(self.numFreePars):
            val = self.testModel.query(self.parTypeStr[i])\
                            [self.freeParNames[i]]
            self.parsOrig[self.freeParNames[i]] = val
            parsOrig.append(val)
        parsOrig = array(parsOrig)
        parDict_new['p_start'] = copy(parsOrig)
        if 'residuals' not in parDict_new:
            parDict_new['residuals'] = self.fn.residual
        if 'Dfun' not in parDict_new:
            # may be None
            if not isinstance(self.fn, helpers.FiniteDifferencesFunction):
                parDict_new['Dfun'] = self.fn.jacobian

        # Setting default minimizer pars
##        if not self._algParamsSet:
        self.setAlgParams(parDict_new)

        self.reset_log()
        # perform least-squares fitting
        rout = RedirectNoOp() if verbose else RedirectStdout(os.devnull)
        rerr = RedirectNoOp() if verbose else RedirectStderr(os.devnull)
        try:
            with rout, rerr:
                results = minpack.leastsq(self._residuals,
                                          self._p_start,
                                          args   = self._args,
                                          Dfun   = self._Dfun,
                                          full_output = self._full_output,
                                          col_deriv   = self._col_deriv,
                                          ftol   = self._ftol,
                                          xtol   = self._xtol,
                                          gtol   = self._gtol,
                                          maxfev = self._maxfev,
                                          epsfcn = self._epsfcn,
                                          factor = self._factor,
                                          diag   = self._diag)
        except:
            print("Calculating residual failed for pars:", \
                  parsOrig)
            raise

        # build return information
        success = results[4] == 1
        if isinstance(results[0], float):
            res_par_list = [results[0]]
            orig_par_list = [parsOrig[0]]
        else:
            res_par_list = results[0].tolist()
            orig_par_list = parsOrig.tolist()
        alg_results = results[2]
        alg_results['message'] = results[3]
        self.pestResult = {'success': success,
                      'cov': results[1],
                      'pars_sol': dict(zip(self.freeParNames,
                                           res_par_list)),
                      'pars_orig': dict(zip(self.freeParNames,
                                            orig_par_list)),
                      'alg_results': alg_results,
                      'sys_sol': self.testModel
                      }

        if verbose:
            # This is a very output-sensitive hack for finding instances where
            # the algorithm stopped because it reached tolerances, not
            # because it converged.
            if success or results[3].find('at most') != -1:
                if success:
                    print('Solution of ', self.freeParNames, ' = ', results[0])
                else:
##                    parvals = [self.testModel.pars[p] for p in \
##                               self.freeParNames]
                    print('Closest values of ', self.freeParNames, ' = ', \
                          results[0])
##                          parvals
                print('Original values = ', parsOrig)
                print('Number of fn evals = ', results[2]["nfev"], \
                             '(# iterations)')
                if not success:
                    print('Solution not found: '+results[3])
            else:
                print('Solution not found: '+results[3])
        return copy(self.pestResult)


    def _make_res_float(self, pars):
        """Returns a function that converts residual vector to its norm
        (a single floating point total residual).

        (Helper method for gradient and Hessian)
        """
        def _residual_float(x):
            return Point({'r': \
                      self.fn(array([x[n] for n in pars], 'd'))})
        return _residual_float


    def gradient_total_residual(self, x, eps=None, pars=None, use_ridder=False):
        """Compute gradient of total residual (norm of the residual function)
        at x as a function of parameter names specified (defaults to all
        free parameters).
        """
        if pars is None:
            pars = self.freeParNames
        if eps is None:
            eps = self.fn.eps
        if use_ridder:
            # Ridders' method (more accurate, slower)
            return common.diff(self._make_res_float(pars),
                        Point(filteredDict(x, pars)),
                        vars=pars, eps=eps, output=self._grad_info)
        else:
            # regular finite differences
            return common.diff2(self._make_res_float(pars),
                        Point(filteredDict(x, pars)),
                        vars=pars, eps=eps)


    def Hessian_total_residual(self, x, eps_inner=None, eps_outer=None,
                pars=None, use_ridder_inner=False, use_ridder_outer=False):
        """Compute Hessian of total residual (norm of the residual function)
        at x as a function of parameter names specified (defaults to all
        free parameters), USING FINITE DIFFERENCES.

        Option to use different eps scalings for the inner gradient
        calculations versus the outer gradient of those values.

        It might be more accurate to calculate the Hessian using a QR
        decomposition of the Jacobian.
        """
        if pars is None:
            pars = self.freeParNames
        res_fn = self._make_res_float(pars)
        if use_ridder_inner:
            diff_inner = common.diff
        else:
            diff_inner = common.diff2

        def Dfun(x):
            diffx=array(diff_inner(res_fn, Point(filteredDict(x, pars)),
                                 vars=pars, eps=eps_inner))
            diffx.shape=(len(pars),)
            return Point(coordarray=diffx, coordnames=pars)

        if use_ridder_outer:
            # Ridders' method (more accurate, slower)
            return common.diff(Dfun,
                        Point(filteredDict(x, pars)),
                        vars=pars, eps=eps_outer, output=self._grad_info)
        else:
            # regular finite differences
            return common.diff2(Dfun,
                        Point(filteredDict(x, pars)),
                        vars=pars, eps=eps_outer)




class BoundMin(ParamEst):
    """Bounded minimization parameter and initial condition optimizer
    for one-dimensional DS trajectories. Fits 1 parameter only.

    Uses SciPy.optimize fminbound algorithm.
    """

    def __init__(self, **kw):
        assert len(kw['freeParams']) == 1, ("Only one free parameter can "
                                            "be specified for this class")
        ParamEst.__init__(self, **kw)
        if self.freeParNames[0] in self.testModel.obsvars:
            # for varying initial conditions
            self.parTypeStr = 'ics'
        elif self.freeParNames[0] in self.testModel.pars:
            # for varying regular pars
            self.parTypeStr = 'pars'
        else:
            raise ValueError('free parameter name not found in test model')
        # Set up model arguments (parameter value will be set before needed)
        self.modelArgs = {self.parTypeStr: {self.freeParNames[0]: None}}


    def run(self, parConstraints, xtol=5e-5, maxiter=500,
            extra_args=(), verbose=False):
        val = self.testModel.query(self.parTypeStr)[self.freeParNames[0]]
        self.parsOrig = {self.freeParNames[0]: val}
        parsOrig = val

        self.reset_log()
        full_output = 1
        rout = RedirectNoOp() if verbose else RedirectStdout(os.devnull)
        rerr = RedirectNoOp() if verbose else RedirectStderr(os.devnull)
        with rout, rerr:
            results = optimize.fminbound(self.fn.residual, parConstraints[0],
                        parConstraints[1], extra_args, xtol, maxiter,
                                    full_output,
                                    int(verbose))

        # build return information
        success = results[2] == 0
        self.pestResult = {'success': success,
                      'pars_sol': {self.freeParNames[0]: results[0]},
                      'pars_orig': {self.freeParNames[0]: parsOrig},
                      'alg_results': results[3],
                      'sys_sol': self.testModel
                      }

        if verbose:
            if success:
                print('Solution of ', self.freeParNames[0], ' = ', results[0])
                print('Original value = ', parsOrig)
                print('Number of fn evals = ', results[3], "(# iterations)")
                print('Error tolerance = ', xtol)
            else:
                print('No convergence of BoundMin')
                print(results)
        return copy(self.pestResult)



res_fn_lookup = {LMpest: residual_fn_context,
                 BoundMin: residual_fn_context_1D}

# ----------------------------------------------------------------------------

## DEPRECATED FUNCTIONS
## Utility functions for estimation using objective functions measuring
# extrema locations. These functions assume the presence of events to
# detect extrema during generation of test trajectories.

def get_slope_info(x, lookahead=1, prec=1e-3, default=1):
    """DEPRECATED. Use features - they work more efficiently and robustly.
    e.g. see Toolbox/neuro_data.py

    Helper function for qualitative fitting.

    Local slope information about data array x. Values of 1 in the
    return array indicate increasing slopes over a local extent given
    by the the lookahead argument, whereas 0 indicates non-increasing
    slopes.

    The default value specifies the value taken by the returned
    array in the indices from len(x)-lookahead to len(x).
    """
    if default==1:
        s = ones(shape(x), 'float')
    elif default==0:
        s = zeros(shape(x), 'float')
    else:
        raise ValueError("Use default = 0 or 1 only")
    for i in range(len(x)-lookahead):
        s[i,:] = [max(prec,val) for val in \
                  ravel((x[i+lookahead,:]-x[i,:]).toarray())]
    return s


def get_extrema(x, t, tmin, tmax, coords, per, pertol_frac,
                lookahead, lookahead_tol, fit_fn_class=None,
                verbose=False):
    """DEPRECATED. Use features - they work more efficiently and robustly.
    e.g. see Toolbox/neuro_data.py

    Helper function for qualitative fitting.

    per is an estimate of the period, per <= tmax.
    pertol_frac is fraction of period used as tolerance for finding extrema.
    fit_fn_class switches on interpolation of extremum by the fitting of a
          local function (uses least squares criterion) - specify a sub-class
          of fit_function (default None).
    """
    # want slope lookahead to be smaller than noise-avoidance lookahead
    slope_lookahead = max([2, lookahead/2])
    slopes = Pointset(coordarray=transpose(get_slope_info(x[coords],
                                     slope_lookahead, prec=0, default=0))>0,
                      coordnames=coords,
                      indepvararray=t)
    maxs_t = {}.fromkeys(coords)
    mins_t = {}.fromkeys(coords)
    maxs_v = {}.fromkeys(coords)
    mins_v = {}.fromkeys(coords)
    last_t = {}.fromkeys(coords)
    last_type = {}.fromkeys(coords)
    detect_on = []
    for c in coords:
        maxs_t[c] = []
        mins_t[c] = []
        maxs_v[c] = []
        mins_v[c] = []
        last_t[c] = [-1,-1]
        last_type[c] = -1
        detect_on.append([True,True])   # False when outside of pertol_frac tolerance for each max and min
    assert pertol_frac < 1 and pertol_frac > 0
    assert per > 0 and per <= tmax
    assert tmin > t[0] and tmax < t[-1]
    detect_tol = pertol_frac*per
    halfper=per/2.
    max_ix = len(t)-1
    ms={0:'min', 1:'max'}
    res = x.find(tmin)
    if isinstance(res, tuple):
        tix_lo = res[0]
    else:
        tix_lo = res
    res = x.find(tmax)
    if isinstance(res, tuple):
        tix_hi = res[0]
    else:
        tix_hi = res
    if fit_fn_class is None:
        do_fit = False
    else:
        do_fit = True
        fit_fn = fit_fn_class()
    # provide some initial history for slope detection info at t = tmin - dt
    last_inc = slopes.coordarray[:,tix_lo-1]
    for local_ix, tval in enumerate(t[tix_lo:tix_hi]):
        ival = local_ix + tix_lo
        ival_ml = max([0, ival - lookahead])
        ival_pl = min([max_ix, ival + lookahead])
        if verbose:
            print("*********** t =", tval, " , ival =", ival)
        v = slopes.coordarray[:,ival]
        for ci, c in enumerate(coords):
            for m in [0,1]:
                if detect_on[ci][m]:
                    pass
##                    if last_t[c][m]>0:
##                        if tval-last_t[c][m]+per+detect_tol > detect_tol:
##                            detect_on[ci][m] = False
##                            print " - %s detect for %s now False:"%(ms,c), last_t[c][m], detect_tol
                else:
                    if tval>(last_t[c][m]+per-detect_tol):
                        if not detect_on[ci][1-m] and 1-m == last_type[ci]:
                            if tval>(last_t[c][1-m]+halfper-detect_tol):
                                detect_on[ci][m] = True
                                if verbose:
                                    print(" + %s detect (>half per of %s) for %s now True:"%(ms[m],ms[1-m],c), last_t[c][1-m], last_t[c][1-m]+halfper-detect_tol)
                        # The next segment allows consecutive local extrema of the
                        # same type without an intermediate of the other type.
                        # This is generally not desirable!
##                        else:
##                        detect_on[ci][m] = True
##                        if verbose:
##                            print " + %s detect (>per) for %s now True:"%(ms[m],c), last_t[c][m], last_t[c][m]+per-detect_tol
                    # The next segment allows consecutive local extrema of the
                    # same type without an intermediate of the other type.
                    # This is generally not desirable!
##                    elif tval>(last_t[c][m]+halfper-detect_tol) and m==last_type[ci] and not detect_on[ci][1-m]:
##                        detect_on[ci][1-m] = True
##                        if verbose:
##                            print " + %s detect (>half per) for %s now True:"%(ms[1-m],c), last_t[c][1-m], last_t[c][m]+halfper-detect_tol
            do_anything = detect_on[ci][0] or detect_on[ci][1]
            if verbose:
                print("Detecting for %s? (min=%i) (max=%i)"%(c, int(detect_on[ci][0]), int(detect_on[ci][1])))
                print("   v[ci] = %.4f, last_inc[ci] = %.4f"%(v[ci], last_inc[ci]))
            if do_anything and v[ci] != last_inc[ci]:
                # extremum if changed sign
                if v[ci]>0:
                    # - +  => min
                    if detect_on[ci][0]:
                        min_ival = argmin(x[c][ival:ival_pl])+ival
                        if verbose:
                            print("Possible min:")
                            print(x[c][max([0,min_ival-lookahead])], x[c][min_ival], x[c][min([max_ix,min_ival+lookahead])])
                        if x[c][min([max_ix,min_ival+lookahead])] - x[c][min_ival] > lookahead_tol and \
                           x[c][max([0,min_ival-lookahead])] - x[c][min_ival] > lookahead_tol:
                            if do_fit:
                                ixs_lo = max([0,min_ival-int(lookahead/4.)])
                                ixs_hi = min([max_ix,min_ival+int(lookahead/4.)])
                                res = fit_fn.fit(x['t'][ixs_lo:ixs_hi],
                                                      x[c][ixs_lo:ixs_hi])
                                xs_fit = res.ys_fit
                                p = res.pars_fit
                                min_tval, min_xval = res.results.peak
                            else:
                                min_tval = x['t'][min_ival]
                                min_xval = x[c][min_ival]
                            if verbose:
                                print("found min for %s at "%c, min_tval)
                            mins_t[c].append(min_tval)
                            mins_v[c].append(min_xval)
                            last_t[c][0]=min_tval
                            detect_on[ci][0] = False
                            detect_on[ci][1] = False
                            last_type[ci]=0
##                    else:
##                        print "    ... ignoring b/c not detecting"
                else:
                    # + - => max
                    if detect_on[ci][1]:
                        max_ival = argmax(x[c][ival:ival_pl])+ival
                        if verbose:
                            print("Possible max:")
                            print(x[c][max([0,max_ival-lookahead])], x[c][max_ival], x[c][min([max_ix,max_ival+lookahead])])
                        if x[c][max_ival] - x[c][min([max_ix,max_ival+lookahead])] > lookahead_tol and \
                           x[c][max_ival] - x[c][max([0,max_ival-lookahead])] > lookahead_tol:
                            if do_fit:
                                ixs_lo = max([0,max_ival-int(lookahead/4.)])
                                ixs_hi = min([max_ix,max_ival+int(lookahead/4.)])
                                res = fit_fn.fit(x['t'][ixs_lo:ixs_hi],
                                                      x[c][ixs_lo:ixs_hi])
                                xs_fit = res.ys_fit
                                p = res.pars_fit
                                max_tval, max_xval = res.results.peak
                            else:
                                max_tval = x['t'][max_ival]
                                max_xval = x[c][max_ival]
                            if verbose:
                                print("found max for %s at "%c, max_tval)
                            maxs_t[c].append(max_tval)
                            maxs_v[c].append(max_xval)
                            last_t[c][1]=max_tval
                            detect_on[ci][0] = False
                            detect_on[ci][1] = False
                            last_type[ci]=1
##                    else:
##                        print "    ... ignoring b/c not detecting"
        last_inc = v
    return (mins_t, maxs_t, mins_v, maxs_v)


def get_extrema_from_events(gen, coords, tmin, tmax, per, pertol_frac,
                            verbose=False):
    """Helper function for qualitative fitting of extrema using least squares.
    This function returns the variable values at the extrema, unlike the
    related function get_extrema (for data).

    per is an estimate of the period, per <= tmax.
    pertol_frac is fraction of period used as tolerance for finding extrema.
    """
    evdict = gen.getEvents()
    assert pertol_frac < 1 and pertol_frac > 0
    assert per > 0 and per <= tmax
    detect_tol = pertol_frac*per
    halfper=per/2.
    ms={0:'min', 1:'max'}
    maxs_t = {}.fromkeys(coords)
    mins_t = {}.fromkeys(coords)
    maxs_v = {}.fromkeys(coords)
    mins_v = {}.fromkeys(coords)
    last_t = {}.fromkeys(coords)
    last_type = {}.fromkeys(coords)
    detect_on = []
    for c in coords:
        maxs_t[c] = []
        maxs_v[c] = []
        mins_t[c] = []
        mins_v[c] = []
        last_t[c] = [-1,-1]
        last_type[c] = -1
        detect_on.append([True,True])   # False when outside of pertol_frac tolerance for each max and min

    ev_list = []
    for ci, c in enumerate(coords):
        for ei, evt in enumerate(evdict['min_ev_'+c]['t']):
            ev_list.append((evt, 'min', ei, c, ci))
        for ei, evt in enumerate(evdict['max_ev_'+c]['t']):
            ev_list.append((evt, 'max', ei, c, ci))
    # sort on time
    ev_list.sort()

    for (tval, ex_type, ei, coord, coord_ix) in ev_list:
        if tval < tmin or tval > tmax:
            continue
        if verbose:
            print("******* t = ", tval)
        for ci, c in enumerate(coords):
            for m in [0,1]:
                if not detect_on[ci][m]:
                    if tval>(last_t[c][m]+per-detect_tol):
                        if not detect_on[ci][1-m] and 1-m == last_type[ci]:
                            if tval>(last_t[c][1-m]+halfper-detect_tol):
                                detect_on[ci][m] = True
                                if verbose:
                                    print(" + %s detect (>half per of %s) for %s now True:"%(ms[m],ms[1-m],c), last_t[c][1-m], last_t[c][1-m]+halfper-detect_tol)
                        # The next segment allows consecutive local extrema of the
                        # same type without an intermediate of the other type.
                        # This is generally not desirable!
##                        else:
##                            detect_on[ci][m] = True
##                            if verbose:
##                                print " + %s detect (>per) for %s now True:"%(ms[m],c), last_t[c][m], last_t[c][m]+per-detect_tol
##                    elif tval>(last_t[c][m]+halfper-detect_tol) and m==last_type[ci] and not detect_on[ci][1-m]:
##                        detect_on[ci][1-m] = True
##                        if verbose:
##                            print " + %s detect (>half per) for %s now True:"%(ms[1-m],c), last_t[c][m], last_t[c][m]+halfper-detect_tol
            do_anything = detect_on[coord_ix][0] or detect_on[coord_ix][1]
            if verbose:
                print("Detecting for %s? (min=%i) (max=%i)"%(c, int(detect_on[ci][0]), int(detect_on[ci][1])))
            if do_anything:
                if ex_type == 'min':
                    if detect_on[coord_ix][0]:
                        min_tval = tval
                        if verbose:
                            print("... found min for %s at "%coord, min_tval)
                        mins_t[coord].append(min_tval)
                        mins_v[coord].append(evdict['min_ev_'+coord][ei][coord])
                        last_t[coord][0]=min_tval
                        detect_on[coord_ix][0] = False
                        detect_on[coord_ix][1] = False
                        last_type[coord_ix]=0
                else:
                    if detect_on[coord_ix][1]:
                        max_tval = tval
                        if verbose:
                            print("... found max for %s at "%coord, max_tval)
                        maxs_t[coord].append(max_tval)
                        maxs_v[coord].append(evdict['max_ev_'+coord][ei][coord])
                        last_t[coord][1]=max_tval
                        detect_on[coord_ix][0] = False
                        detect_on[coord_ix][1] = False
                        last_type[coord_ix]=1
    return (mins_t, maxs_t, mins_v, maxs_v)


def compare_data_from_events(gen, coords, traj, tmesh, data_mins_t, data_maxs_t,
                  data_mins_v, data_maxs_v, num_expected_mins, num_expected_maxs,
                  tdetect, per, pertolfrac, verbose=False):
    test_mins_t, test_maxs_t, test_mins_v, test_maxs_v = \
                 get_extrema_from_events(gen, coords,
                                       tdetect[0], tdetect[1],
                                       per, pertolfrac, verbose=verbose)
    try:
        num_mins = [min([len(test_mins_t[c]), len(data_mins_t[c])]) for c in coords]
        num_maxs = [min([len(test_maxs_t[c]), len(data_maxs_t[c])]) for c in coords]
    except TypeError:
        print("Problem with mins and maxs in coords %s in generator %s"%(str(coords),gen.name))
        print("Number of events found =", len(gen.getEvents()))
        print(per, pertolfrac, tdetect)
        print(type(test_mins_t), type(test_maxs_t))
        print(type(data_mins_t), type(data_maxs_t))
        raise
    res_mins_t = []
    res_maxs_t = []
    res_mins_v = []
    res_maxs_v = []
    for ci,c in enumerate(coords):
        nmin = num_mins[ci]
        if len(data_mins_t[c]) != num_expected_mins or len(test_mins_t[c]) < num_expected_mins:
            print("Wrong number of minima for %s (expected %i)"%(c, num_expected_mins))
            print(data_mins_t[c], test_mins_t[c])
            raise RuntimeError("Wrong number of minima")
        # assume 0:num_expected is good
        t1 = array(data_mins_t[c])-array(test_mins_t[c])[:num_expected_mins]
        v1 = data_mins_v[c]-array(test_mins_v[c])[:num_expected_mins]
        res_mins_t.extend(list(t1))
        res_mins_v.extend(list(ravel(v1)))
        # max
        nmax = num_maxs[ci]
        if len(data_maxs_t[c]) != num_expected_maxs or len(test_maxs_t[c]) < num_expected_maxs:
            print("Wrong number of maxima for %s (expected %i)"%(c, num_expected_maxs))
            print(data_maxs_t[c], test_maxs_t[c])
            raise RuntimeError("Wrong number of maxima")
        # assume 0:num_expected is good
        t1 = array(data_maxs_t[c])-array(test_maxs_t[c])[:num_expected_maxs]
        v1 = data_maxs_v[c]-array(test_maxs_v[c])[:num_expected_maxs]
        res_maxs_t.extend(list(t1))
        res_maxs_v.extend(list(ravel(v1)))
    return (res_mins_t, res_mins_v, res_maxs_t, res_maxs_v)
