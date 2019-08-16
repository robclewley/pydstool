"""
    Internal utilities.

    Robert Clewley, September 2005.
"""

from __future__ import absolute_import, print_function

from .errors import *

import sys, types
import numpy as npy
import scipy as spy
from scipy.optimize import minpack
# In future, will convert these specific imports to be referred as npy.X
from numpy import Inf, NaN, atleast_1d, clip, less, greater, logical_or, \
     searchsorted, isfinite, shape, mat, sign, any, all, sometrue, alltrue, \
     array, swapaxes, zeros, ones, finfo, double, exp, log, \
     take, less_equal, putmask, ndarray, asarray, \
     int, float, complex, complexfloating, integer, floating, \
     int_, int0, int8, int16, int32, int64, float_, float32, float64, \
     complex_, complex64, complex128, argmin, argmax
from numpy.linalg import norm
from math import sqrt

try:
    from numpy import float96
except ImportError:
    _all_numpy_float = (float_, float32, float64)
else:
    _all_numpy_float = (float_, float32, float64, float96)


try:
    from numpy import complex192
except ImportError:
    _all_numpy_complex = (complex_, complex64, complex128)
else:
    _all_numpy_complex = (complex_, complex64, complex128, complex192)

try:
    from scipy.special import factorial
except ImportError:
    try:
        # Retain backward compatibility with older scipy versions
        from scipy.misc import factorial
    except ImportError:
        # Retain backward compatibility with even older scipy versions
        from scipy import factorial


import time
from copy import copy, deepcopy
import os
from six.moves import cPickle as pickle
import six

# ----------------------------------------------------------------------------
### EXPORTS

_classes = ['interpclass', 'interp0d', 'interp1d', 'Utility',
            'args', 'pickle', 'Diagnostics',
            'metric', 'metric_float', 'metric_float_1D', 'metric_L2',
            'metric_L2_1D', 'metric_weighted_L2', 'metric_weighted_deadzone_L2',
            'predicate', 'null_predicate', 'and_op', 'or_op', 'not_op']

_mappings = ['_num_type2name', '_num_name2type',
             '_num_equivtype', '_num_name2equivtypes',
             '_pytypefromtype', '_num_maxmin'
             ]

_functions = ['isUniqueSeq', 'makeArrayIxMap', 'className',
              'compareBaseClass', 'compareClassAndBases', 'timestamp',
              'makeUniqueFn', 'copyVarDict', 'concatStrDict',
              'invertMap', 'makeSeqUnique', 'insertInOrder', 'uniquePoints',
              'sortedDictKeys', 'sortedDictValues', 'sortedDictItems',
              'sortedDictLists', 'compareNumTypes', 'diff', 'diff2',
              'listid', 'idfn', 'noneFn', 'isincreasing', 'ismonotonic',
              'extent', 'n_sigdigs_str',
              'linearInterp', 'object2str', 'getSuperClasses',
              'filteredDict', 'simplifyMatrixRepr',
              'makeMultilinearRegrFn', 'fit_quadratic', 'fit_quadratic_at_vertex',
              'fit_exponential', 'fit_diff_of_exp', 'fit_linear', 'fit_cubic',
              'smooth_pts', 'nearest_2n_indices',
              'KroghInterpolator', 'BarycentricInterpolator',
              'PiecewisePolynomial', 'make_poly_interpolated_curve',
              'simple_bisection', 'get_opt', 'array_bounds_check',
              'verify_intbool', 'verify_nonneg', 'verify_pos',
              'verify_values', 'ensurefloat']

_constants = ['Continuous', 'Discrete', 'targetLangs', '_seq_types',
              '_num_types', '_int_types', '_float_types', '_complex_types',
              '_real_types', '_all_numpy_int', '_all_numpy_float',
              '_all_numpy_complex', '_all_int', '_all_float', '_all_complex',
              'LargestInt32']

__all__ = _functions + _mappings + _classes + _constants

# ----------------------------------------------------------------------------

# global reference for supported target languages
targetLangs = ['c', 'python', 'matlab'] #, 'xpp', 'dstool'


# type mappings and groupings

_num_types = (float, int, floating, integer) # complex, complexfloating

_int_types = six.integer_types + (integer, )
_float_types = (float, floating)
_complex_types = (complex, complexfloating)
_real_types = (int, integer, float, floating)

_seq_types = (list, tuple, ndarray)

_all_numpy_int = (int_, int0, int8, int16, int32, int64)

_all_int = _int_types+_all_numpy_int
_all_float = _float_types+_all_numpy_float
_all_complex = _complex_types+_all_numpy_complex

LargestInt32 = 2147483647
Macheps = finfo(double).eps

# bind common names
_num_type2name = {float: 'float', int: 'int'} #, complex: 'complex'}
_num_equivtype = {float: float64, int: int32} #, complex: complex128}
for f in _all_float:
    _num_type2name[f] = 'float'
    _num_equivtype[f] = float64
for i in _all_int:
    _num_type2name[i] = 'int'
    _num_equivtype[i] = int32
# Don't yet support complex numbers
##for c in _all_complex:
##    _num_type2name[c] = 'complex'
##    _num_equivtype[c] = complex128

# equivalent types for comparison
_num_name2equivtypes = {'float': _all_float,
                'int': _all_int}
##                'complex': _all_complex}

# default types used by PyDSTool when named
_num_name2type = {'float': float64, 'int': int32} #, 'complex': complex128}

_num_maxmin = {float64: [-Inf, Inf],
             int32: [-LargestInt32-1, LargestInt32],
##             complex128: [-Inf-Inf*1.0j, Inf+Inf*1.0j]
             }

_typefrompytype = {float: float64, int: int32} #, complex: complex128}
_pytypefromtype = {float64: float, int32: int} #, complex128: complex}


#-------------------------------------------------------------------------
### PREDICATES ETC

class predicate_op(object):
    def __init__(self, predicates):
        self.predicates = predicates
        self.record = []

    def precondition(self, objlist):
        res = npy.all([p.precondition(objlist) for p in self.predicates])
        self.record = [(self.name, [p.record for p in self.predicates])]
        return res

    def __call__(self, obj):
        res = self.evaluate(obj)
        self.record = [(self.name, [p.record for p in self.predicates])]
        return res

    def evaluate(self, obj):
        raise NotImplementedError

    def __repr__(self):
        return self.name + '(' + \
               ', '.join([repr(p) for p in self.predicates]) + ')'


class and_op(predicate_op):
    name = 'AND'

    def evaluate(self, obj):
        return npy.all([p(obj) for p in self.predicates])


class or_op(predicate_op):
    name = 'OR'

    def evaluate(self, obj):
        return npy.any([p(obj) for p in self.predicates])


class not_op(predicate_op):
    name = 'NOT'

    def __init__(self, predicate):
        # make a singleton so that inherited repr works
        self.predicates = [predicate]
        self.record = []

    def precondition(self, objlist):
        res = self.predicates[0].precondition(objlist)
        self.record = [self.name, self.predicates[0].record]
        return res

    def __call__(self, obj):
        res = self.evaluate(obj)
        self.record = [self.name, self.predicates[0].record]
        return res

    def evaluate(self, obj):
        return not self.predicates[0](obj)


class predicate(object):
    # override name in subclass if needed
    name = ''

    def __init__(self, subject):
        self.subject = subject
        self.record = []

    def precondition(self, objlist):
        """Override if needed"""
        return True

    def __call__(self, obj):
        res = self.evaluate(obj)
        self.record = (self.name, self.subject, res)
        return res

    def evaluate(self, obj):
        raise NotImplementedError

    def __repr__(self):
        if self.subject is None:
            s = '<no subject>'
        else:
            s = self.subject
        return self.name + '(' + s + ')'

    __str__ = __repr__


class null_predicate_class(predicate):
    name = 'null'

    def evaluate(self, obj):
        return True

null_predicate = null_predicate_class(None)

# ------------------------------------------------------


class metric(object):
    """Abstract metric class for quantitatively comparing scalar or vector
    quantities.
    Can include optional explicit Jacobian function.

    Create concrete sub-classes for specific applications.
    Store the measured (*1D array only*) value in self.results for use as part
    of a parameter estimation residual value. Residual norm will be taken
    by optimizer routines.
    """
    def __init__(self):
        self.results = None

    def __call__(self, x, y):
        raise NotImplementedError("Override with a concrete sub-class")

    def Jac(self, x, y):
        raise NotImplementedError("Override with a concrete sub-class")


class metric_float(metric):
    """Simple metric between two real-valued floats.
    """
    def __call__(self, x, y):
        self.results = asarray([x - y]).flatten()
        return norm(self.results)

class metric_float_1D(metric):
    """Simple metric between two real-valued floats. Version that is suitable for
    scalar optimizers such as BoundMin.
    """
    def __call__(self, x, y):
        self.results = abs(asarray([x - y]).flatten())
        return self.results[0]

class metric_L2(metric):
    """Measures the standard "distance" between two 1D pointsets or arrays
    using the L-2 norm."""
    def __call__(self, pts1, pts2):
        self.results = asarray(pts1-pts2).flatten()
        return norm(self.results)

class metric_L2_1D(metric):
    """Measures the standard "distance" between two 1D pointsets or arrays
    using the L-2 norm."""
    def __call__(self, pts1, pts2):
        norm_val = norm(asarray(pts1-pts2).flatten())
        self.results = array([norm_val])
        return norm_val

class metric_weighted_L2(metric):
    """Measures the standard "distance" between two 1D pointsets or arrays
    using the L-2 norm, after weighting by weights attribute
    (must set weights after creation, e.g. in a feature's _local_init
    method)."""
    def __call__(self, pts1, pts2):
        self.results = array(pts1-pts2).flatten()*self.weights
        return norm(self.results)

class metric_weighted_deadzone_L2(metric):
    """Measures the standard "distance" between two 1D pointsets or arrays
    using the L-2 norm, after weighting by weights attribute.
    Then, sets distance vector entries to zero if they fall
    below corresponding entries in the deadzone vector/scalar.
    (Must set weights and deadzone vectors/scalars after creation, e.g.
    in a feature's _local_init method).
    """
    def __call__(self, pts1, pts2):
        v = array(pts1-pts2).flatten()*self.weights
        v = (abs(v) > self.deadzone).astype(int) * v
        self.results = v
        return norm(v)


def n_sigdigs_str(x, n):
    """Return a string representation of float x with n significant digits,
    where n > 0 is an integer.
    """
    format = "%." + str(int(n)) + "g"
    s = '%s' % float(format % x)
    if '.' in s:
        # handle trailing ".0" when not one of the sig. digits
        pt_idx = s.index('.')
        if s[0] == '-':
            # pt_idx is one too large
            if pt_idx-1 >= n:
                return s[:pt_idx]
        else:
            if pt_idx >= n:
                return s[:pt_idx]
    return s


class args(object):
    """Mapping object class for building arguments for class initialization
    calls. Treat as a dictionary.
    """

    def __init__(self, **kw):
        self.__dict__ = kw

    def _infostr(self, verbose=1, attributeTitle='args',
                 ignore_underscored=False):
        # removed offset=0 from arg list
        if len(self.__dict__) > 0:
            res = "%s ("%attributeTitle
            for k, v in self.__dict__.items():
                if k[0] == '_' and ignore_underscored:
                    continue
                if verbose == 0:
                    # don't resolve any deeper
                    if hasattr(v, 'name'):
                        name = ' ' + v.name
                    else:
                        name = ''
                    istr = str(type(v)) + name
                else:
                    try:
                        istr = v._infostr(verbose-1) #, offset+2)
                    except AttributeError:
                        istr = str(v)
                res += "\n%s%s = %s,"%(" ",k,istr)
                # was " "*offset
            # skip last comma
            res = res[:-1] + "\n)"
            return res
        else:
            return "No %s defined"%attributeTitle

    def __repr__(self):
        return self._infostr()

    def info(self):
        print(self._infostr())

    __str__ = __repr__

    def values(self):
        return list(self.__dict__.values())

    def keys(self):
        return list(self.__dict__.keys())

    def items(self):
        return list(self.__dict__.items())

    def itervalues(self):
        return iter(self.__dict__.values())

    def iterkeys(self):
        return iter(self.__dict__.keys())

    def iteritems(self):
        return iter(self.__dict__.items())

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__.__setitem__(k, v)

    def update(self, d):
        self.__dict__.update(d)

    def copy(self):
        return copy(self)

    def clear(self):
        self.__dict__.clear()

    def get(self, k, d=None):
        return self.__dict__.get(k, d)

    def has_key(self, k):
        return k in self.__dict__

    def pop(self, k, d=None):
        return self.__dict__.pop(k, d)

    def popitem(self):
        raise NotImplementedError

    def __contains__(self, v):
        return self.__dict__.__contains__(v)

    def fromkeys(self, S, v=None):
        raise NotImplementedError

    def setdefault(self, d):
        raise NotImplementedError

    def __delitem__(self, k):
        del self.__dict__[k]

    __hash__ = None

    def __cmp__(self, other):
        return self.__dict__ == other

    def __eq__(self, other):
        return self.__dict__ == other

    def __ne__(self, other):
        return self.__dict__ != other

    def __gt__(self, other):
        return self.__dict__ > other

    def __ge__(self, other):
        return self.__dict__ >= other

    def __lt__(self, other):
        return self.__dict__ < other

    def __le__(self, other):
        return self.__dict__ <= other

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __add__(self, other):
        d = self.__dict__.copy()
        d.update(other.__dict__)
        return args(**d)


def get_opt(argopt, attr, default=None):
    """Get option from args object otherwise default to the given value. Can
    also specify that an AttributeError is raised by passing default=Exception.
    """
    try:
        return getattr(argopt, attr)
    except AttributeError:
        if default is Exception:
            raise PyDSTool_AttributeError("Missing option: "+attr)
        else:
            return default


class Diagnostics(object):
    """General purpose diagnostics manager."""

    def __init__(self, errmessages=None, errorfields=None, warnmessages=None,
                 warnfields=None, errorcodes=None, warncodes=None,
                 outputinfo=None, propagate_dict=None):
        if warnfields is None:
            warnfields = {}
        if warnmessages is None:
            warnmessages = {}
        if warncodes is None:
            warncodes = {}
        if errorfields is None:
            errorfields = {}
        if errmessages is None:
            errmessages = {}
        if errorcodes is None:
            errorcodes = {}
        self._warnfields = warnfields
        self._warnmessages = warnmessages
        self._warncodes = warncodes
        self._errorfields = errorfields
        self._errmessages = errmessages
        self._errorcodes = errorcodes
        self.errors = []
        self.warnings = []
        # traceback may store information about variable state, pars, etc.
        # at time of an error that breaks the solver
        self.traceback = {}
        self.outputStatsInfo = outputinfo
        self.outputStats = {}
        if propagate_dict is None:
            # use dict so that un-initialized inputs attribute
            # of generator etc. can be passed-by-reference
            self.propagate_dict = {}
        else:
            self.propagate_dict = propagate_dict

    def update(self, d):
        """Update warnings and errors from another diagnostics object"""
        self.traceback.update(d.traceback)
        self.warnings.extend(d.warnings)
        self.errors.extend(d.errors)
        self.outputStats.update(d.outputStats)
        self._warnfields.update(d._warnfields)
        self._warnmessages.update(d._warnmessages)
        self._warncodes.update(d._warncodes)
        self._errorfields.update(d._errorfields)
        self._errmessages.update(d._errmessages)
        self._errorcodes.update(d._errorcodes)

    def clearAll(self):
        self.clearErrors()
        self.clearWarnings()
        self.outputStats = {}
        self.traceback = {}

    def clearWarnings(self):
        self.warnings = []
        for obj in self.propagate_dict.values():
            try:
                obj.diagnostics.clearWarnings()
            except AttributeError:
                if hasattr(obj, 'name'):
                    name = obj.name
                else:
                    name = str(obj)
                raise TypeError("Object %s has no diagnostics manager"%name)

    def showWarnings(self):
        if len(self.warnings)>0:
            print(self.getWarnings())

    def getWarnings(self):
        if len(self.warnings)>0:
            output = 'Warnings: '
            for (w, d) in self.warnings:
                dstr = ''
                for i in range(len(d)):
                    dentry = d[i]
                    dstr += self._warnfields[w][i] + ' = ' + str(dentry) + ", "
                dstr = dstr[:-2]  # drop trailing comma
                output += ' Warning code %s:  %s\n Info:  %s ' %(w, \
                                                self._warnmessages[w], dstr)
        else:
            output = ''
        return output

    def findWarnings(self, code):
        """Return time-ordered list of warnings of kind specified using a
        single Generator warning code"""
        res = []
        for wcode, wdata in self.warnings:
            if wcode == code:
                res.append(wdata)
        res.sort()  # increasing order
        return res

    def hasWarnings(self):
        return self.warnings != []

    def hasErrors(self):
        return self.errors != []

    def clearErrors(self):
        self.errors = []
        for obj in self.propagate_dict.values():
            try:
                obj.diagnostics.clearErrors()
            except AttributeError:
                if hasattr(obj, 'name'):
                    name = obj.name
                else:
                    name = str(obj)
                raise TypeError("Object %s has no diagnostics manager"%name)

    def showErrors(self):
        if len(self.errors)>0:
            print(self.getErrors())

    def getErrors(self):
        if len(self.errors)>0:
            output = 'Errors: '
            for (e, d) in self.errors:
                dstr = ''
                for i in range(len(d)):
                    dentry = d[i]
                    dstr += self._errorfields[e][i] + ' = ' + str(dentry) + ", "
                dstr = dstr[:-2]  # drop trailing comma
                output += ' Error code %s:  %s\n Info:\n  %s ' %(e, \
                                                    self._errmessages[e], dstr)
        else:
            output = ''
        return output

    def info(self, verboselevel=0):
        self.showErrors()
        self.showWarnings()


## ------------------------------------------------------------------

## Internally used functions

def compareNumTypes(t1, t2):
    try:
        return sometrue([_num_type2name[t1] == _num_type2name[t] for t in t2])
    except TypeError:
        # t2 not iterable, assume singleton
        try:
            return _num_type2name[t1] == _num_type2name[t2]
        except KeyError:
            return False
    except KeyError:
        return False


def filteredDict(d, keys, neg=False):
    """returns filtered dictionary containing specified keys,
    or *not* containing the specified keys if option neg=True."""

    guard = (lambda k: k not in keys) if neg else (lambda k: k in keys)
    return {k: d[k] for k in d.keys() if guard(k)}


def concatStrDict(d, order=None):
    """Concatenates all entries of a dictionary (assumed to be
    lists of strings), in optionally specified order."""

    return ''.join([''.join(d[k]) for k in (order or d.keys())])


def copyVarDict(vardict, only_cts=False):
    """Copy dictionary of Variable objects.
    Use the only_cts Boolean optional argument (default False) to select only
    continuous-valued variables (mainly for internal use).
    """
    if only_cts:
        out_vars = []
        out_varnames = []
        sorted_varnames = sortedDictKeys(vardict)
        for varname in sorted_varnames:
            var = vardict[varname]
            if var.is_continuous_valued():
                out_varnames.append(varname)
                out_vars.append(var)
        return dict(zip(out_varnames, out_vars))
    else:
        return dict(zip(sortedDictKeys(vardict), [copy(v) for v in \
                                              sortedDictValues(vardict)]))


def insertInOrder(sourcelist, inslist, return_ixs=False, abseps=0):
    """Insert elements of inslist into sourcelist, sorting these
      lists in case they are not already in increasing order. The new
      list is returned.

    The function will not create duplicate entries in the list, and will
      change neither the first or last entries of the list.

    If sourcelist is an array, an array is returned.
    If optional return_ixs=True, the indices of the inserted elements
      in the returned list is returned as an additional return argument.
    If abseps=0 (default) the comparison of elements is done exactly. For
      abseps > 0 elements are compared up to an absolute difference no
      greater than abseps for determining "equality".
    """
    try:
        sorted_inslist = inslist.tolist()
    except AttributeError:
        sorted_inslist = copy(inslist)
    sorted_inslist.sort()
    try:
        sorted_sourcelist = sourcelist.tolist()
        was_array = True
    except AttributeError:
        sorted_sourcelist = copy(sourcelist)
        was_array = False
    sorted_sourcelist.sort()
    close_ixs = []
    tix = 0
    # optimize by having separate versions of loop
    if return_ixs:
        ins_ixs = []
        for t in sorted_inslist:
            tcond = less_equal(sorted_sourcelist[tix:], t).tolist()
            try:
                tix = tcond.index(0) + tix  # lowest index for elt > t
            except ValueError:
                # no 0 value in tcond, so t might be equal to the final value
                if abs(sorted_sourcelist[-1] - t) < abseps:
                    close_ixs.append((t,len(sorted_sourcelist)-1))
            else:
                if abs(sorted_sourcelist[tix-1] - t) >= abseps:
                    if tix >= 0:
                        sorted_sourcelist.insert(tix, t)
                        ins_ixs.append(tix)
                else:
                    close_ixs.append((t,tix-1))
        if was_array:
            if abseps > 0:
                return array(sorted_sourcelist), ins_ixs, dict(close_ixs)
            else:
                return array(sorted_sourcelist), ins_ixs
        else:
            if abseps > 0:
                return sorted_sourcelist, ins_ixs, dict(close_ixs)
            else:
                return sorted_sourcelist, ins_ixs
    else:
        for t in sorted_inslist:
            tcond = less_equal(sorted_sourcelist[tix:], t).tolist()
            try:
                tix = tcond.index(0) + tix  # lowest index for elt > t
            except ValueError:
                # no 0 value in tcond, so t might be equal to the final value
                if abs(sorted_sourcelist[-1] - t) < abseps:
                    close_ixs.append((t,len(sorted_sourcelist)-1))
            else:
                if abs(sorted_sourcelist[tix-1] - t) >= abseps:
                    if tix >= 0:
                        sorted_sourcelist.insert(tix, t)
                else:
                    close_ixs.append((t,tix-1))
        if was_array:
            if abseps > 0:
                return array(sorted_sourcelist), dict(close_ixs)
            else:
                return array(sorted_sourcelist)
        else:
            if abseps > 0:
                return sorted_sourcelist, dict(close_ixs)
            else:
                return sorted_sourcelist


def simplifyMatrixRepr(m):
    """Convert matrix object to a compact array
    representation or numeric value."""
    ma=array(m)
    l = len(shape(ma))
    if l == 0:
        return m
    elif l>0 and shape(ma)[0] == 1:
        return simplifyMatrixRepr(ma[0])
    elif l>1 and shape(ma)[1] == 1:
        return simplifyMatrixRepr(ma[:,0])
    else:
        return ma


def makeMultilinearRegrFn(arg, xs, ys):
    """Convert two lists or arrays mapping x intervals to y intervals
    into a string function definition of a multilinear regression
    scalar function that these define. A.k.a. makes a "piecewise
    linear" scalar function from the input data. The two input data
    sequences can each be either all numeric values or all
    strings/symbolic objects, but not a mixture. """
    assert len(xs)==len(ys), \
           "You must give x and y lists that are the same length"
    assert not isinstance(arg, _num_types), \
           "arg must be a string or symbolic object"
    argname = str(arg)

    def sub_str(a,b):
        return '(' + str(a) + '-' + str(b) + ')'
    def sub_val(a,b):
        return repr(a-b)

    def interp(n):
        return rep_y(ys[n-1]) +'+(' + argname + '-(' + rep_x(xs[n-1]) \
          + '))*' + sub_y(ys[n],ys[n-1]) +'/'+ sub_x(xs[n],xs[n-1])

    x_test = [isinstance(xs[n], _num_types) for n in range(len(xs))]
    if all(x_test):
        rep_x = lambda x: repr(x)
        sub_x = sub_val
    elif any(x_test):
        raise TypeError("xlist must contain either all string/symbolic types "
                        "or all numeric values")
    else:
        rep_x = lambda x: str(x)
        sub_x = sub_str
    y_test = [isinstance(ys[n], _num_types) for n in range(len(ys))]
    if all(y_test):
        rep_y = lambda y: repr(y)
        sub_y = sub_val
    elif any(y_test):
        raise TypeError("ylist must contain either all string/symbolic types "
                        "or all numeric values")
    else:
        rep_y = lambda y: str(y)
        sub_y = sub_str
    mLR = '+'.join(['heav(%s-%s)*(1-heav(%s-%s))*(%s)'%(argname, \
                     rep_x(xs[n-1]),argname,rep_x(xs[n]),interp(n)) \
                   for n in range(1,len(xs))])
    return ([argname], mLR)


def _scalar_diff(func, x0, dx):
    """Numerical differentiation of scalar function by central differences.
    Returns tuple containing derivative evaluated at x0 and error estimate,
    using Ridders' method and Neville's algorithm.
    """
    max_order = 10
    BIG = 1e50
    CON = 1.4
    CON2 = CON*CON
    SAFE = 2
    a=zeros((max_order,max_order),'f')
    a[0,0] = (func(x0+dx)-func(x0-dx))/(2.*dx)
    err=BIG
    ans = NaN
    for i in range(1,max_order):
        dx /= CON
        # try a smaller stepsize
        a[0,i] = (func(x0+dx)-func(x0-dx))/(2.*dx)
        fac = CON2
        for j in range(1,i):
            # compute extrapolations of various orders, using Neville's
            # algorithm
            a[j,i] = (a[j-1,i]*fac-a[j-1,i-1])/(fac-1.)
            fac *= CON2
            errt = max([abs(a[j,i]-a[j-1,i]),abs(a[j,i]-a[j-1][i-1])])
            # error strategy:
            # compare each new extrapolation to one order lower, both at the
            # present stepsize and the previous one
            if errt <= err:
                err = errt
                ans  = a[j,i]
        if abs(a[i,i] - a[i-1,i-1]) >= SAFE*err:
            # if higher order is worse by a significant factor SAFE, then
            # quit early
            break
    return (ans, err, dx)


def diff(func, x0, vars=None, axes=None, eps=None, output=None):
    """Numerical 1st derivative of R^N -> R^M scalar or array function
    about x0 by central finite differences. Uses Ridders' method of
    polynomial extrapolation, based on an implementation in the book
    "Numerical Recipes". Returns a matrix.

    vars argument specifies which elements of x0 are to be treated as
      variables for the purposes of taking the Jacobian.
    If axes argument is unused or set to be all axes, the Jacobian of the
      function evaluated at x0 with respect to the variables is returned,
      otherwise a sub-matrix of it is returned.
    eps is assumed to be the scale in x for which the function varies by O(1).
      If eps is not given an appropriate step size is chosen.
    output = True returns an optional dictionary which will be updated
      with error and derivative information.
    """

    if isinstance(x0, ndarray):
        x0type = 'array'
        if not compareNumTypes(x0.dtype.type, _all_float):
            raise TypeError("Only real-valued arrays valid")
    elif isinstance(x0, _real_types):
        x0type = 'num'
        x0 = float(x0)
    else:
        # Point type
        try:
            assert compareNumTypes(x0.coordtype, _all_float)
            x0.coordnames
            x0.dimension
        except (AssertionError, AttributeError):
            raise TypeError("Function and x0 must use real-valued scalar,"
                            "array, or Point types only")
        x0type = 'point'
    output_info = {}
    if vars is None:
        if x0type == 'array':
            dim = len(x0)
            vars = list(range(dim))
        elif x0type == 'num':
            dim = 1
            vars = [0]
        else:
            # Point type
            dim = x0.dimension
            vars = x0.coordnames
    else:
        assert isinstance(vars, _seq_types), \
               "vars argument must be a sequence type"
        if x0type in ['array', 'num']:
            assert all(v >= 0 for v in vars), \
                    "vars argument must hold non-negative integers"
        else:
            assert all([isinstance(vars[i], str) \
                 for i in range(len(vars))]), "vars argument must hold strings"
        dim = len(vars)
    fx0 = func(x0)
    sfx0 = shape(fx0)
    try:
        # ensure fx0 is a vector or at least only a D x 1 matrix
        assert sfx0[1] == 1
    except IndexError:
        # if shape is of form (D,) then that's fine
        if len(sfx0) > 0:
            if sfx0[0] == 0:
                raise TypeError("Invalid function return type")
    except AssertionError:
        print("fx0 shape is %d" % sfx0)
        print(fx0)
        raise ValueError("Function should return an N-vector or N x 1 matrix,"
                 " but it returned a matrix with shape %s" % str(sfx0))
    if isinstance(fx0, _float_types):
        dimf = 1
    elif isinstance(fx0, ndarray):
        if not compareNumTypes(fx0.dtype.type, _all_float):
            raise TypeError("Only real-valued functions valid")
        try:
            dimf = sfx0[0]
        except IndexError:
            dimf = 1
    else:
        try:
            assert compareNumTypes(fx0.coordtype, _all_float)
        except (AssertionError, AttributeError):
            raise TypeError("Only real-valued functions valid")
        dimf = sfx0[0]
    if axes is None:
        if x0type in ['array', 'num']:
            try:
                axes = list(range(sfx0[0]))
            except IndexError:
                # then singleton (scalar) was returned
                axes = [0]
        else:
            axes = fx0.coordnames
    else:
        assert isinstance(axes, _seq_types), \
               "axes argument must be a sequence type"
        if x0type in ['array', 'num']:
            assert all(a>= 0 for a in axes), \
                   "axes argument must hold non-negative integers"
        else:
            assert all([isinstance(axes[i], str) \
                 for i in range(len(axes))]), "axes argument must hold strings"
    if eps is None:
        eps = sqrt(Macheps)
    else:
        assert all(eps > 0), "eps scaling array must be strictly positive"
        if isinstance(eps, _num_types):
            eps = ones(dim)*eps
        else:
            assert len(eps) == len(vars), \
                   "eps scaling array has length mismatch with vars"

    if x0type == 'array':
        dx = eps*(abs(x0[vars]) + array(x0[vars]==zeros(dim),'float64'))
    elif x0type == 'num':
        dx = eps*(abs(x0) + int(x0==0))
    else:
        # Point
        x0a = x0[vars].toarray()
        dx = dict(zip(vars,
                      eps*(abs(x0a) + array(x0a==zeros(dim),'float64'))))
    try:
        dim_mat = len(axes)
    except TypeError:
        raise TypeError("axes argument must be a sequence type")
    assert dim_mat <= dimf, "Number of axes greater than dimension of function"
    df = zeros([dim_mat,dim], 'float64')
    if x0type == 'array':
        output_info['error'] = zeros([dim_mat,dim], 'float64')
        output_info['dx'] = zeros([dim_mat,dim], 'float64')
        def update(xa, i, x):
            xa[i] = x
            return xa
        for i, vix in enumerate(vars):
            try:
                # for numpy arrays (otherwise copy returns a regular 'array'!)
                x0_d = x0.copy()
            except AttributeError:
                x0_d = copy(x0)
            if dimf > 1:
                for j in range(dim_mat):
                    f_d = lambda x: func(update(x0_d, vix, x))[axes[j]]
                    df_d, errval, dx_d = _scalar_diff(f_d, x0_d[vix], dx[i])
                    df[j,i] = df_d
                    output_info['error'][j,i] = errval
                    output_info['dx'][j,i] = dx_d
            else:
                for j in range(dim_mat):
                    f_d = lambda x: func(update(x0_d, vix, x))
                    df_d, errval, dx_d = _scalar_diff(f_d, x0_d[vix], dx[i])
                    df[j,i] = df_d
                    output_info['error'][j,i] = errval
                    output_info['dx'][j,i] = dx_d
        df = mat(df)
        output_info['df'] = df
        if output is not None:
            try:
                output.update(output_info)
            except:
                raise TypeError("Invalid type for 'output' argument")
        return df
    elif x0type == 'num':
        df, errval, dx_d = _scalar_diff(func, x0, dx)
        output_info['df'] = df
        output_info['error'] = errval
        output_info['dx'] = dx_d
        if output is not None:
            try:
                output.update(output_info)
            except:
                raise TypeError("Invalid type for 'output' argument")
        return df
    else:
        # Point type
        output_info['error'] = zeros([dim_mat,dim], 'float64')
        output_info['dx'] = zeros([dim_mat,dim], 'float64')
        def update(xa, vn, x):
            xa[vn] = x
            return xa
        for i in range(dim):
            vname = vars[i]
            x0_d = copy(x0)
            for j in range(dim_mat):
                f_d = lambda x: func(update(x0_d, vname, x))[axes[j]]
                df_d, errval, dx_d = _scalar_diff(f_d, x0_d[vname], dx[vname])
                df[j,i] = df_d
                output_info['error'][j,i] = errval
                output_info['dx'][j,i] = dx_d
        df = mat(df)
        output_info['df'] = df
        if output is not None:
            try:
                output.update(output_info)
            except:
                raise TypeError("Invalid type for 'output' argument")
        return df


def diff2(func, x0, vars=None, axes=None, dir=1, eps=None):
    """Numerical 1st derivative of R^N -> R^M scalar or array function
    about x0 by forward or backward finite differences. Returns a matrix.

    dir=1 uses finite forward difference.
    dir=-1 uses finite backward difference.
    List-valued eps rescales finite differencing in each axis separately.
    vars argument specifies which elements of x0 are to be treated as
      variables for the purposes of taking the Jacobian.
    If axes argument is unused or set to be all axes, the Jacobian of the
      function evaluated at x0 with respect to the variables is returned,
      otherwise a sub-matrix of it is returned.
    eps is assumed to be the scale in x for which the function varies by O(1).
      If eps is not given an appropriate step size is chosen
      (proportional to sqrt(machine precision)).
    """

    if isinstance(x0, ndarray):
        x0type = 'array'
        if not compareNumTypes(x0.dtype.type, _all_float):
            try:
                x0 = x0.astype(float)
            except:
                print("Found type: %s" % x0.dtype.type)
                raise TypeError("Only real-valued arrays valid")
    elif isinstance(x0, _real_types):
        x0type = 'num'
        x0 = float(x0)
    else:
        # Point type
        try:
            assert compareNumTypes(x0.coordtype, _all_float)
            x0.coordnames
            x0.dimension
        except (AssertionError, AttributeError):
            raise TypeError("Function and x0 must use real-valued scalar,"
                            "array, or Point types only")
        x0type = 'point'
    if vars is None:
        if x0type == 'array':
            dim = len(x0)
            vars = list(range(dim))
        elif x0type == 'num':
            dim = 1
            vars = [0]
        else:
            # Point type
            dim = x0.dimension
            vars = x0.coordnames
    else:
        assert isinstance(vars, _seq_types), \
               "vars argument must be a sequence type"
        if x0type in ['array', 'num']:
            assert all(vars>=0), \
                    "vars argument must hold non-negative integers"
        else:
            assert all([isinstance(vars[i], str) \
                 for i in range(len(vars))]), "vars argument must hold strings"
        dim = len(vars)
    fx0 = func(x0)
    sfx0 = shape(fx0)
    if isinstance(fx0, _float_types):
        dimf = 1
    elif isinstance(fx0, ndarray):
        if not compareNumTypes(fx0.dtype.type, _all_float):
            raise TypeError("Only real-valued functions valid")
        try:
            dimf = sfx0[0]
        except IndexError:
            dimf = 1
        try:
            # ensure fx0 is a vector or at least only a D x 1 matrix
            assert sfx0[1] == 1
        except IndexError:
            # if shape is of form (D,) then that's fine
            if len(sfx0) > 0:
                if sfx0[0] == 0:
                    raise TypeError("Invalid function return type")
            else:
                raise TypeError("Invalid function return type")
        except AssertionError:
            print("fx0 shape is %d" % sfx0)
            print(fx0)
            raise ValueError("Function should return an N-vector or N x 1 matrix,"
                     " but it returned a matrix with shape %s" % str(sfx0))
    else:
        try:
            assert compareNumTypes(fx0.coordtype, _all_float)
        except (AssertionError, AttributeError):
            raise TypeError("Only real-valued functions valid")
        dimf = sfx0[0]
    if axes is None:
        if x0type in ['array', 'num']:
            try:
                axes = list(range(sfx0[0]))
            except IndexError:
                # then singleton (scalar) was returned
                axes = [0]
        else:
            axes = fx0.coordnames
    else:
        assert isinstance(axes, _seq_types), \
               "axes argument must be a sequence type"
        if x0type in ['array', 'num']:
            assert all(axes>=0), \
                   "axes argument must hold non-negative integers"
        else:
            assert all([isinstance(axes[i], str) \
                 for i in range(len(axes))]), "axes argument must hold strings"
    if eps is None:
        eps = sqrt(Macheps)
    else:
        assert all(eps > 0), "eps scaling array must be strictly positive"
        if isinstance(eps, float):
            if x0type in ['array', 'num']:
                eps = ones(dim)*eps
        else:
            assert len(eps) == len(vars), \
                   "eps scaling array has length mismatch with vars"
            eps = asarray(eps, 'float64')
    # ensure dx is not 0, and make into an appropriate length vector
    if x0type == 'array':
        dx = eps*(abs(x0[vars]) + array(x0[vars]==zeros(dim),'float64'))
    elif x0type == 'num':
        dx = eps*(abs(x0)+int(x0==0))
    else:
        # Point
        x0a = x0[vars].toarray()
        dx = dict(zip(vars,
                      eps*(abs(x0a) + array(x0a==zeros(dim),'float64'))))

    assert dir==1 or dir==-1, "Direction code must be -1 or 1"
    try:
        dim_mat = len(axes)
    except TypeError:
        raise TypeError("axes argument must be a sequence type")
    assert dim_mat <= dimf, "Number of axes greater than dimension of function"
    df = zeros([dim_mat,dim], 'float64')
    if x0type == 'array':
        for i in range(dim):
            vix = vars[i]
            try:
                # for numpy arrays (otherwise copy returns a regular 'array'!)
                x0_d = x0.copy()
            except AttributeError:
                x0_d = copy(x0)
            x0_d[vix] += dir * dx[i]
            fx0_d = func(x0_d)
            if dim_mat > 1:
                fx0_d_v = array([fx0_d[n] for n in axes])
                fx0_v = array([fx0[n] for n in axes])
            else:
                if dimf > 1:
                    fx0_d_v = fx0_d[axes[0]]
                    fx0_v = fx0[axes[0]]
                else:
                    fx0_d_v = fx0_d
                    fx0_v = fx0
            df[:,i] = dir*(fx0_d_v - fx0_v)/dx[i]
        return mat(df)
    elif x0type == 'num':
        x0_d = x0 + dir*dx
        fx0_d = func(x0_d)
        df = dir*(fx0_d - fx0)/dx
        return df
    else:
        # Point type
        for i in range(dim):
            vname = vars[i]
            x0_d = copy(x0)
            x0_d[vname] = x0_d(vname) + dir * dx[vname]
            fx0_d = func(x0_d)[axes]
            fx0_v = fx0[axes]
            df[:,i] = dir*(fx0_d - fx0_v).toarray()/dx[vname]
        return mat(df)


def ensurefloat(v):
    try:
        # singleton Point will return scalar here
        v = v.toarray()
    except AttributeError:
        pass
    try:
        # numeric literal as Quantity will return scalar here
        v = v.tonumeric()
    except AttributeError:
        pass
    return float(v)

_verify_type_names = {_all_int: 'an integer',
                      _all_float: 'a float',
                      _real_types: 'a real number',
                      _all_complex: 'a complex number'}

# Only support lists because the primary use of these functions is for
# checking input to SWIG-interfaced data structures passed down to C
# and Fortran, which must be basic types only.

def verify_values(name, value, values, list_ok=False, list_len=None):
    """Use list_ok if a list of values of these types is acceptable.
    list_len can be used to specify that a list must be of a certain length,
    either a fixed integer or a variable integer value given as the first
    value of a pair, the second being the name of the variable (for use in
    error messages)
    """
    if list_ok:
        if isinstance(value, list):
            if list_len is not None:
                if isinstance(list_len, _all_int):
                    ok = (len(value) == list_len)
                    len_name = '%d' % list_len
                else:
                    ok = (len(value) == list_len[0])
                    len_name = list_len[1]
                if not ok:
                    raise ValueError("list "+name+" length must equal "+len_name)
            for v in value:
                try:
                    # make sure v is not a list too
                    verify_values(name, v, values)
                except ValueError:
                    raise ValueError(name+" must be in " + str(values) + \
                                     " or a list of these")
                except TypeError:
                    raise TypeError(name+" must be in " + str(values) + \
                                     " or a list of these")
        else:
            raise TypeError(name+" must be in " + str(values) + \
                            " or a list of these")
    else:
        if value not in values:
            raise ValueError(name+" must be in " + str(values))


def verify_intbool(name, value, list_ok=False, list_len=None):
    """Use list_ok if a list of values of these types is acceptable.
    list_len can be used to specify that a list must be of a certain length,
    either a fixed integer or a variable integer value given as the first
    value of a pair, the second being the name of the variable (for use in
    error messages)
    """
    if list_ok:
        if isinstance(value, list):
            if list_len is not None:
                if isinstance(list_len, _all_int):
                    ok = (len(value) == list_len)
                    len_name = '%d' % list_len
                else:
                    ok = (len(value) == list_len[0])
                    len_name = list_len[1]
                if not ok:
                    raise ValueError("list "+name+" length must equal "+len_name)
            for v in value:
                try:
                    # make sure v is not a list too
                    verify_intbool(name, v)
                except ValueError:
                    raise ValueError(name+" must be 0, 1, or a boolean," + \
                                     " or a list of these")
                except TypeError:
                    raise TypeError(name+" must be 0, 1, or a boolean," + \
                                     " or a list of these")
        else:
            raise TypeError(name+" must be 0, 1, or a boolean," + \
                            " or a list of these")
    elif isinstance(value, _all_int):
        if value not in [0, 1]:
            raise ValueError("integer "+name+" must be 0 or 1")
    elif not isinstance(value, bool):
        raise TypeError(name+" must be 0, 1 or a boolean")


def verify_nonneg(name, value, types, list_ok=False, list_len=None):
    """Use list_ok if a list of values of these types is acceptable.
    list_len can be used to specify that a list must be of a certain length,
    either a fixed integer or a variable integer value given as the first
    value of a pair, the second being the name of the variable (for use in
    error messages)
    """
    if isinstance(value, types):
        if value < 0:
            raise ValueError(name+" must be non-negative")
    elif list_ok:
        if isinstance(value, list):
            if list_len is not None:
                if isinstance(list_len, _all_int):
                    ok = (len(value) == list_len)
                    len_name = '%d' % list_len
                else:
                    ok = (len(value) == list_len[0])
                    len_name = list_len[1]
                if not ok:
                    raise ValueError("list "+name+" length must equal "+len_name)
            for v in value:
                try:
                    # make sure v is not a list too
                    verify_nonneg(name, v, types)
                except ValueError:
                    raise ValueError(name+" must be "+_verify_type_names[types]+ \
                                     " and non-negative, or a list of these")
                except TypeError:
                    raise TypeError(name+" must be "+_verify_type_names[types]+ \
                                     " and non-negative, or a list of these")
        else:
            raise TypeError(name+" must be "+_verify_type_names[types]+ \
                            " and non-negative, or a list of these")
    else:
        raise TypeError(name+" must be "+_verify_type_names[types]+ \
                            " and non-negative")


def verify_pos(name, value, types, list_ok=False, list_len=None):
    """Use list_ok if a list of values of these types is acceptable.
    list_len can be used to specify that a list must be of a certain length,
    either a fixed integer or a variable integer value given as the first
    value of a pair, the second being the name of the variable (for use in
    error messages)
    """
    if isinstance(value, types):
        if value <= 0:
            raise ValueError(name+" must be positive")
    elif list_ok:
        if isinstance(value, list):
            if list_len is not None:
                if isinstance(list_len, _all_int):
                    ok = (len(value) == list_len)
                    len_name = '%d' % list_len
                else:
                    ok = (len(value) == list_len[0])
                    len_name = list_len[1]
                if not ok:
                    raise ValueError("list "+name+" length must equal "+len_name)
            for v in value:
                try:
                    # make sure v is not a list too
                    verify_nonneg(name, v, types)
                except ValueError:
                    raise ValueError(name+" must be "+_verify_type_names[types]+ \
                                     " and positive, or a list of these")
                except TypeError:
                    raise TypeError(name+" must be "+_verify_type_names[types]+ \
                                     " and positive, or a list of these")
        else:
            raise TypeError(name+" must be "+_verify_type_names[types]+ \
                            " and positive, or a list of these")
    else:
        raise TypeError(name+" must be "+_verify_type_names[types]+ \
                            " and positive")


def array_bounds_check(a, bounds, dirn=1):
    """Internal utility function to test a 1D array for staying within given
    bounds (min val, max val).

    Returns the largest index +1 if the array is within bounds, otherwise the
    first offending index, where 'first' is the earliest in a if direction
    dirn=1, or the latest if dirn=-1."""
    if dirn == 1:
        OK_ix = len(a)
        alo = asarray(a<bounds, int)
        ahi = asarray(a>bounds, int)
        alo_first = alo.argmax()
        ahi_first = ahi.argmax()
        test_val = 0
        compare = min
    elif dirn == -1:
        OK_ix = -1
        alo = 1 - asarray(a<bounds, int)
        ahi = 1 - asarray(a>bounds, int)
        alo_first = alo.argmin()
        ahi_first = ahi.argmin()
        test_val = 1
        compare = max
    else:
        raise ValueError("Invalid direction")
    first_fail_ix = OK_ix
    if alo[alo_first] != test_val:
        # an element was below lower bound
        first_fail_ix = alo_first
    if ahi[ahi_first] != test_val:
        # an element was above upper bound
        if first_fail_ix == OK_ix:
            first_fail_ix = ahi_first
        else:
            first_fail_ix = compare(first_fail_ix, ahi_first)
    return first_fail_ix


def linearInterp(y0, ygoal, y1, x0, x1):
    """Internal utility function to linearly interpolate between two
    data points."""
    return ( x1 * (ygoal - y0) + x0 * ( y1 - ygoal) ) / (y1 - y0)


def makeUniqueFn(fstr, tdigits=0, idstr=None):
    """Add unique ID to function names.

    Used when functions are executed in global namespace to avoid name
    clashes, and need to be distinguished when DS objects are copied."""
    # check for syntax errors
    try:
        code = compile(fstr, 'test', 'exec')
    except:
        print(" Cannot make unique function because of a syntax (or other) error " \
              "in supplied code:\n")
        print(fstr)
        raise
    bracepos = fstr.index("(")
    if idstr is None:
        idstr_insert = ""
    else:
        idstr_insert = "_" + idstr
    if tdigits > 0:
        fname = fstr[4:bracepos] + idstr_insert + "_" + timestamp(tdigits)
    else:
        fname = fstr[4:bracepos]
    fstr_new = "def " + fname + fstr[bracepos:]
    return (fstr_new, fname)


def timestamp(tdigits=8):
    """Return a unique timestamp string for the session. useful for ensuring
    unique function identifiers, etc.
    """
    try:
        t = time.process_time()
    except AttributeError:
        # Python 2.7 compatibility
        t = time.clock()
    return str(t).replace(".", "").replace("-", "")[:tdigits + 1]


def isUniqueSeq(objlist):
    """Check that list contains items only once"""
    return len(set(objlist)) == len(objlist)


def makeSeqUnique(seq, asarray=False):
    """Return a 1D sequence that only contains the unique values in seq"""
    seen = set()
    seen_add = seen.add
    list_ = [it for it in seq if it not in seen and not seen_add(it)]
    return array(list_) if asarray else list_


def object2str(x, digits=5):
    """Convert occurrences of types / classes,
    to pretty-printable strings."""
    try:
        if type(x) in six.class_types + (type, ):
            return className(x, True)
        elif isinstance(x, list):
            # search through any iterable parts (that aren't strings)
            rx = "["
            if len(x)>0:
                for o in x:
                    rx += object2str(o, digits) + ", "
                return rx[:-2]+"]"
            else:
                return rx+"]"
        elif isinstance(x, tuple):
            rx = "("
            if len(x)>0:
                for o in x:
                    rx += object2str(o, digits) + ", "
                return rx[:-2]+")"
            else:
                return rx+")"
        elif isinstance(x, dict):
            rx = "{"
            if len(x)>0:
                for k, o in x.items():
                    rx += object2str(k, digits) + ": " + object2str(o, digits) + ", "
                return rx[:-2]+"}"
            else:
                return rx+"}"
        elif isinstance(x, str):
            # this removes extraneous single quotes around dict keys, for instance
            return x
        elif isinstance(x, float):
            format_str = '"%%.%if"'%digits
            return eval(format_str + '%x')
        else:
            return repr(x)
    except:
        raise TypeError("object2str cannot format this object type")


#  The class types can show different roots when they originate from
#  different parts of the PyDSTool package -- it might be a bug.
#  e.g. baseClass might be Generator.Generator, but here this type will be
#  <class 'PyDSTool.Generator.baseclasses.Generator'>
#  and input.__class__ will boil down to
#  <class 'Generator.baseclasses.Generator'>
#  even though these classes are identical (constructed from the same class
#  in the same module!)
def compareBaseClass(input, baseClass):
    """input may be a class or a class instance representing that class.
    baseClass may be a class or a string name of a class.

    Comparison is made using class names only."""
    if isinstance(baseClass, type):
        base_str = baseClass.__name__
    elif isinstance(baseClass, str):
        base_str = baseClass
    else:
        raise TypeError("Must pass either a class or a class name (string)")
    if isinstance(input, type):
        bases = input.__bases__
    else:
        try:
            bases = input.__class__.__bases__
        except AttributeError:
            # not the kind of baseClass PyDSTool is interested in
            # e.g. an exception type
            return False
    return sometrue([base_str == c.__name__ for c in bases])


def compareClassAndBases(input, arg):
    """arg can be a single or sequence of classes"""
    try:
        # if arg is iterable
        return sometrue([compareClassAndBases(input, a) for a in arg])
    except TypeError:
        try:
            if isinstance(input, type):
                # input is a class
                return issubclass(input, arg)
            else:
                # input is an instance
                return isinstance(input, arg)
        except TypeError:
            raise TypeError("Invalid class(es) provided: input %s vs. %s" \
                            %(str(input)+" of type "+className(input),className(arg,True)))


def getSuperClasses(obj, limitClasses=None):
    """Return string names of all super classes of a given object"""
    if limitClasses == None:
        limitClassNames = ['object']
    elif isinstance(limitClasses, list):
        limitClassNames = [className(lc) for lc in limitClasses]
    else:
        # singleton class
        limitClassNames = [className(limitClasses)]
    # ensure "object" safety net is present
    if 'object' not in limitClassNames:
        limitClassNames.append('object')
    search_obj = [obj.__class__]
    sclasses = [className(search_obj[0])]
    # don't start while loop if obj is already of a type in limitClasses
    done = (sclasses[0] in limitClassNames)
    c = 0
    while not done and c < 10:
        c += 1
        search_temp = []
        for so in search_obj:
            search_temp.extend(list(so.__bases__))
        search_obj = search_temp
        for b in search_obj:
            sclass = className(b)
            done = sclass in limitClassNames
            if done:
                break
            else:
                sclasses.append(sclass)
    return sclasses


def className(obj, addPrefix=False):
    """Return human-readable string of class name."""
    if isinstance(obj, str):
        class_str = obj
        # don't add prefix -- it's unknown
        prefix = ""
    elif isinstance(obj, type):
        class_str = obj.__name__
        if addPrefix:
            prefix = "Class "
        else:
            prefix = ""
    elif isinstance(obj, types.ModuleType):
        class_str = obj.__name__
        if addPrefix:
            prefix = "Module "
        else:
            prefix = ""
    else:
        try:
            class_str = obj.__class__.__name__
        except AttributeError:
            class_str = str(type(obj))
        prefix = ""
    return prefix + class_str


# little utility function to wrap value as a singleton list
def listid(val):
    return [val]


# the identity function
def idfn(val):
    return copy(val)


# utility function representing a "none" function
def noneFn(x):
    return None


# returns the mapping from the entries in an array or list to their indices
def makeArrayIxMap(a):
    return dict(zip(a, range(len(a))))


# invert an index mapping or other form of mapping
def invertMap(themap):
    """invert an index mapping or other form of mapping.

    If argument is a dict or sequence type, returns a dictionary,
    but if argument is a parseUtils.symbolMapClass then that type is
    returned."""
    if isinstance(themap, dict):
        try:
            return dict([(v, k) for k, v in themap.items()])
        except TypeError:
            # e.g., list objects are unhashable
            # try it the slow way for this case
            result = {}
            for k, v in themap.items():
                if isinstance(v, (list,tuple)):
                    for val in v:
                        result[val] = k
                else:
                    result[v] = k
            return result
    elif isinstance(themap, (list,tuple)):
        # input domain is the position index
        return dict(zip(themap, range(len(themap))))
    elif isinstance(themap, ndarray):
        # input domain is the position index
        return dict(zip(themap.tolist(), range(len(themap))))
    elif hasattr(themap, 'inverse'):
        # symbolMapClass type
        return themap.inverse()
    else:
        raise TypeError("Unsupported type for map")


def isincreasing(theseq, withVal=False):
    """
    Check whether a sequence is in increasing order. The withVal
    option (default False) causes the function to return the first
    two offending values that are not repeated.
    """
    # Note: This version of the function has better speed on the
    # 'usual' case where this function is used internally by PyDSTool
    # -- which is where the sequence *is* increasing and the input is
    # already an array
    try:
        v_old = theseq[0]
    except IndexError:
        raise ValueError("Problem with sequence passed to "
                         "function `isincreasing` -- is it empty?")
    v = array(theseq)
    res = v[1:] > v[:-1]
    if withVal:
        if all(res):
            return True, None, None
        else:
            pos = res.tolist().index(False)
            return False, theseq[pos], theseq[pos+1]
    else:
        return all(res)


def ismonotonic(theseq, withVal=False):
    """
    Check whether a sequence is in strictly increasing or decreasing
    order. The withVal option (default False) causes the function to
    return the first two offending values that are not repeated.
    """
    if withVal:
        res_incr, pos1, pos2 = isincreasing(theseq, True)
        res_decr = isincreasing(theseq[::-1], False)
        if res_incr or res_decr:
            return True, None, None
        else:
            return False, pos1, pos2
    else:
        res_incr = isincreasing(theseq)
        res_decr = isincreasing(theseq[::-1])
        return res_incr or res_decr


def extent(data):
    """Returns a pair of the min and max values of a dataset, or just a numeric type if these are equal.
    (Ignores NaNs.)
    """
    minval = npy.nanmin(data)
    maxval = npy.nanmax(data)
    if minval == maxval:
        return minval
    else:
        return [minval, maxval]

def uniquePoints(ar):
    """For an n by m array input, return only points that are unique"""
    result = []
    seq = set()
    for a in ar:
        a = tuple(a)
        if a not in seq:
            result.append(a)
            seq.add(a)
    return array(result)


def sortedDictValues(d, onlykeys=None, reverse=False):
    """Return list of values from a dictionary in order of sorted key list.

    Adapted from original function by Alex Martelli:
     added filtering of keys.
    """
    if onlykeys is None:
        keys = list(d.keys())
    else:
        keys = intersect(d.keys(), onlykeys)
    keys.sort()
    if reverse:
        keys.reverse()
    return list(map(d.get, keys))

def sortedDictKeys(d, onlykeys=None, reverse=False):
    """Return sorted list of keys from a dictionary.

    Adapted from original function by Alex Martelli:
     added filtering of keys."""
    if onlykeys is None:
        keys = list(d.keys())
    else:
        keys = intersect(d.keys(), onlykeys)
    keys.sort()
    if reverse:
        keys.reverse()
    return keys

def sortedDictLists(d, byvalue=True, onlykeys=None, reverse=False):
    """Return (key list, value list) pair from a dictionary,
    sorted by value (default) or key.
    Adapted from an original function by Duncan Booth.
    """
    if onlykeys is None:
        onlykeys = list(d.keys())
    if byvalue:
        i = [(val, key) for (key, val) in d.items() if key in onlykeys]
        i.sort()
        if reverse:
            i.reverse()
        rvals = [val for (val, key) in i]
        rkeys = [key for (val, key) in i]
    else:
        # by key
        i = [(key, val) for (key, val) in d.items() if key in onlykeys]
        i.sort()
        if reverse:
            i.reverse()
        rvals = [val for (key, val) in i]
        rkeys = [key for (key, val) in i]
    return (rkeys, rvals)

def sortedDictItems(d, byvalue=True, onlykeys=None, reverse=False):
    """Return list of (key, value) pairs of a dictionary,
    sorted by value (default) or key.
    Adapted from an original function by Duncan Booth.
    """
    return list(zip(*sortedDictLists(d, byvalue, onlykeys, reverse)))

# ----------------------------------------------------------------------

## private versions of these utils (cannot import them from utils!)

# find intersection of two lists, sequences, etc.
def intersect(a, b):
    return [e for e in a if e in b]


# find remainder of two lists, sequences, etc., after intersection
def remain(a, b):
    return [e for e in a if e not in b]


# ----------------------------------------------------------------------

# The Utility class may be abandoned in future versions.
class Utility(object):
    """
    Utility abstract class for manipulating and analyzing dynamical systems.

    Robert Clewley, March 2005.

Subclasses of Utility could include such things as continuation tools,
dimension reduction tools, parameter estimation tools.
"""
    pass


# --------------------------------------------------------------------
# This section adapted from scipy.interpolate
# --------------------------------------------------------------------


class interpclass(object):
    """Abstract class for interpolators."""
    interp_axis = -1    # used to set which is default interpolation
                        # axis.  DO NOT CHANGE OR CODE WILL BREAK.


class interp0d(interpclass):
    """Design of this class based on SciPy's interp1d"""

    def __init__(self, x, y, axis=-1, makecopy=0, bounds_error=1,
                 fill_value=None):
        """Initialize a piecewise-constant interpolation class

        Description:
          x and y are arrays of values used to approximate some function f:
            y = f(x)
          This class returns a function whose call method uses piecewise-
          constant interpolation to find the value of new points.

        Inputs:
            x -- a 1d array of monotonically increasing real values.
                 x cannot include duplicate values. (otherwise f is
                 overspecified)
            y -- an nd array of real values.  y's length along the
                 interpolation axis must be equal to the length
                 of x.
            axis -- specifies the axis of y along which to
                    interpolate. Interpolation defaults to the last
                    axis of y.  (default: -1)
            makecopy -- If 1, the class makes internal copies of x and y.
                    If 0, references to x and y are used. The default
                    is NOT to copy. (default: 0)
            bounds_error -- If 1, an error is thrown any time interpolation
                            is attempted on a value outside of the range
                            of x (where extrapolation is necessary).
                            If 0, out of bounds values are assigned the
                            NaN (#INF) value.  By default, an error is
                            raised, although this is prone to change.
                            (default: 1)
        """
        self.datapoints = (array(x, float), array(y, float))   # RHC -- for access from PyDSTool
        self.type = float   # RHC -- for access from PyDSTool
        self.axis = axis
        self.makecopy = makecopy   # RHC -- renamed from copy to avoid nameclash
        self.bounds_error = bounds_error
        if fill_value is None:
            self.fill_value = NaN   # RHC -- was:   array(0.0) / array(0.0)
        else:
            self.fill_value = fill_value

        # Check that both x and y are at least 1 dimensional.
        if len(shape(x)) == 0 or len(shape(y)) == 0:
            raise ValueError("x and y arrays must have at least one dimension.")
        # make a "view" of the y array that is rotated to the
        # interpolation axis.
        oriented_x = x
        oriented_y = swapaxes(y,self.interp_axis,axis)
        interp_axis = self.interp_axis
        len_x,len_y = shape(oriented_x)[interp_axis], \
                            shape(oriented_y)[interp_axis]
        if len_x != len_y:
            raise ValueError("x and y arrays must be equal in length along "
                              "interpolation axis.")
        if len_x < 2 or len_y < 2:
            raise ValueError("x and y arrays must have more than 1 entry")
        self.x = array(oriented_x,copy=self.makecopy)
        self.y = array(oriented_y,copy=self.makecopy)


    def __call__(self,x_new):
        """Find piecewise-constant interpolated y_new = <name>(x_new).

        Inputs:
          x_new -- New independent variables.

        Outputs:
          y_new -- Piecewise-constant interpolated values corresponding to x_new.
        """
        # 1. Handle values in x_new that are outside of x.  Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        ## RHC -- was   x_new = atleast_1d(x_new)
        x_new_1d = atleast_1d(x_new)
        out_of_bounds = self._check_bounds(x_new_1d)
        # 2. Find where in the orignal data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] = x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x,x_new_1d)
        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1.  Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = clip(x_new_indices,1,len(self.x)-1).astype(int)
        # 4. Calculate the region that each x_new value falls in.
        lo = x_new_indices - 1; hi = x_new_indices

        # !! take() should default to the last axis (IMHO) and remove
        # !! the extra argument.
        # 5. Calculate the actual value for each entry in x_new.
        y_lo = take(self.y,lo,axis=self.interp_axis)
        y_hi = take(self.y,hi,axis=self.interp_axis)
        y_new = (y_lo+y_hi)/2.
        # 6. Fill any values that were out of bounds with NaN
        # !! Need to think about how to do this efficiently for
        # !! mutli-dimensional Cases.
        yshape = y_new.shape
        y_new = y_new.ravel()
        new_shape = list(yshape)
        new_shape[self.interp_axis] = 1
        sec_shape = [1]*len(new_shape)
        sec_shape[self.interp_axis] = len(out_of_bounds)
        out_of_bounds.shape = sec_shape
        new_out = ones(new_shape)*out_of_bounds
        putmask(y_new, new_out.ravel(), self.fill_value)
        y_new.shape = yshape
        # Rotate the values of y_new back so that they correspond to the
        # correct x_new values.
        result = swapaxes(y_new,self.interp_axis,self.axis)
        try:
            len(x_new)
            return result
        except TypeError:
            return result[0]
        return result


    def _check_bounds(self,x_new):
        # If self.bounds_error = 1, we raise an error if any x_new values
        # fall outside the range of x.  Otherwise, we return an array indicating
        # which values are outside the boundary region.
        # !! Needs some work for multi-dimensional x !!
        below_bounds = less(x_new,self.x[0])
        above_bounds = greater(x_new,self.x[-1])
        #  Note: sometrue has been redefined to handle length 0 arrays
        # !! Could provide more information about which values are out of bounds
        # RHC -- Changed these ValueErrors to PyDSTool_BoundsErrors
        if self.bounds_error and any(sometrue(below_bounds)):
##            print "Input:", x_new
##            print "Bound:", self.x[0]
##            print "Difference input - bound:", x_new-self.x[0]
            raise PyDSTool_BoundsError(" A value in x_new is below the"
                              " interpolation range.")
        if self.bounds_error and any(sometrue(above_bounds)):
##            print "Input:", x_new
##            print "Bound:", self.x[-1]
##            print "Difference input - bound:", x_new-self.x[-1]
            raise PyDSTool_BoundsError(" A value in x_new is above the"
                              " interpolation range.")
        # !! Should we emit a warning if some values are out of bounds.
        # !! matlab does not.
        out_of_bounds = logical_or(below_bounds,above_bounds)
        return out_of_bounds


    # RHC added
    def __getstate__(self):
        d = copy(self.__dict__)
        # remove reference to Cfunc self.type
        d['type'] = _num_type2name[self.type]
        return d

    # RHC added
    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc self.type
        self.type = _num_name2type[self.type]



class interp1d(interpclass):    # RHC -- made this a new-style Python class
    def __init__(self, x, y, kind='linear', axis=-1,
                 makecopy = 0, bounds_error=1, fill_value=None):
        """Initialize a 1d piecewise-linear interpolation class

        Description:
          x and y are arrays of values used to approximate some function f:
            y = f(x)
          This class returns a function whose call method uses linear
          interpolation to find the value of new points.

        Inputs:
            x -- a 1d array of monotonically increasing real values.
                 x cannot include duplicate values. (otherwise f is
                 overspecified)
            y -- an nd array of real values.  y's length along the
                 interpolation axis must be equal to the length
                 of x.
            kind -- specify the kind of interpolation: 'nearest', 'linear',
                    'cubic', or 'spline'
            axis -- specifies the axis of y along which to
                    interpolate. Interpolation defaults to the last
                    axis of y.  (default: -1)
            makecopy -- If 1, the class makes internal copies of x and y.
                    If 0, references to x and y are used. The default
                    is NOT to copy. (default: 0)
            bounds_error -- If 1, an error is thrown any time interpolation
                            is attempted on a value outside of the range
                            of x (where extrapolation is necessary).
                            If 0, out of bounds values are assigned the
                            NaN (#INF) value.  By default, an error is
                            raised, although this is prone to change.
                            (default: 1)
        """
        self.datapoints = (array(x, float), array(y, float))   # RHC -- for access from PyDSTool
        self.type = float   # RHC -- for access from PyDSTool
        self.axis = axis
        self.makecopy = makecopy   # RHC -- renamed from copy to avoid nameclash
        self.bounds_error = bounds_error
        if fill_value is None:
            self.fill_value = NaN   # RHC -- was:   array(0.0) / array(0.0)
        else:
            self.fill_value = fill_value

        if kind != 'linear':
            raise NotImplementedError("Only linear supported for now. "
                                      "Use fitpack routines for other types.")

        # Check that both x and y are at least 1 dimensional.
        if len(shape(x)) == 0 or len(shape(y)) == 0:
            raise ValueError("x and y arrays must have at least one dimension.")
        # make a "view" of the y array that is rotated to the
        # interpolation axis.
        oriented_x = x
        oriented_y = swapaxes(y,self.interp_axis,axis)
        interp_axis = self.interp_axis
        len_x,len_y = shape(oriented_x)[interp_axis], \
                            shape(oriented_y)[interp_axis]
        if len_x != len_y:
            raise ValueError("x and y arrays must be equal in length along "
                              "interpolation axis.")
        if len_x < 2 or len_y < 2:
            raise ValueError("x and y arrays must have more than 1 entry")
        self.x = array(oriented_x,copy=self.makecopy)
        self.y = array(oriented_y,copy=self.makecopy)


    def __call__(self,x_new):
        """Find linearly interpolated y_new = <name>(x_new).

        Inputs:
          x_new -- New independent variables.

        Outputs:
          y_new -- Linearly interpolated values corresponding to x_new.
        """
        # 1. Handle values in x_new that are outside of x.  Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        ## RHC -- was   x_new = atleast_1d(x_new)
        x_new_1d = atleast_1d(x_new)
        out_of_bounds = self._check_bounds(x_new_1d)
        # 2. Find where in the orignal data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] = x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x,x_new_1d)
        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1.  Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = clip(x_new_indices,1,len(self.x)-1).astype(int)
        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1; hi = x_new_indices

        # !! take() should default to the last axis (IMHO) and remove
        # !! the extra argument.
        x_lo = take(self.x,lo,axis=self.interp_axis)
        x_hi = take(self.x,hi,axis=self.interp_axis)
        y_lo = take(self.y,lo,axis=self.interp_axis)
        y_hi = take(self.y,hi,axis=self.interp_axis)
        slope = (y_hi-y_lo)/(x_hi-x_lo)
        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope*(x_new_1d-x_lo) + y_lo
        # 6. Fill any values that were out of bounds with NaN
        # !! Need to think about how to do this efficiently for
        # !! mutli-dimensional Cases.
        yshape = y_new.shape
        y_new = y_new.ravel()
        new_shape = list(yshape)
        new_shape[self.interp_axis] = 1
        sec_shape = [1]*len(new_shape)
        sec_shape[self.interp_axis] = len(out_of_bounds)
        out_of_bounds.shape = sec_shape
        new_out = ones(new_shape)*out_of_bounds
        putmask(y_new, new_out.ravel(), self.fill_value)
        y_new.shape = yshape
        # Rotate the values of y_new back so that they correspond to the
        # correct x_new values.
        result = swapaxes(y_new,self.interp_axis,self.axis)
        try:
            len(x_new)
            return result
        except TypeError:
            return result[0]
        return result


    def _check_bounds(self,x_new):
        # If self.bounds_error = 1, we raise an error if any x_new values
        # fall outside the range of x.  Otherwise, we return an array indicating
        # which values are outside the boundary region.
        # !! Needs some work for multi-dimensional x !!
        below_bounds = less(x_new,self.x[0])
        above_bounds = greater(x_new,self.x[-1])
        #  Note: sometrue has been redefined to handle length 0 arrays
        # !! Could provide more information about which values are out of bounds
        # RHC -- Changed these ValueErrors to PyDSTool_BoundsErrors
        if self.bounds_error and any(sometrue(below_bounds)):
##            print "Input:", x_new
##            print "Bound:", self.x[0]
##            print "Difference input - bound:", x_new-self.x[0]
            raise PyDSTool_BoundsError("A value in x_new is below the"
                              " interpolation range.")
        if self.bounds_error and any(sometrue(above_bounds)):
##            print "Input:", x_new
##            print "Bound:", self.x[-1]
##            print "Difference input - bound:", x_new-self.x[-1]
            raise PyDSTool_BoundsError("A value in x_new is above the"
                              " interpolation range.")
        # !! Should we emit a warning if some values are out of bounds.
        # !! matlab does not.
        out_of_bounds = logical_or(below_bounds,above_bounds)
        return out_of_bounds


    # RHC added
    def __getstate__(self):
        d = copy(self.__dict__)
        # remove reference to Cfunc self.type
        d['type'] = _num_type2name[self.type]
        return d

    # RHC added
    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc self.type
        self.type = _num_name2type[self.type]


# The following interpolation functions were written and (c) Anne
# Archibald.

class KroghInterpolator(object):
    """The interpolating polynomial for a set of points

    Constructs a polynomial that passes through a given set of points,
    optionally with specified derivatives at those points.
    Allows evaluation of the polynomial and all its derivatives.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial, although they can be obtained
    by evaluating all the derivatives.

    Be aware that the algorithms implemented here are not necessarily
    the most numerically stable known. Moreover, even in a world of
    exact computation, unless the x coordinates are chosen very
    carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon. In general, even with well-chosen
    x values, degrees higher than about thirty cause problems with
    numerical instability in this code.

    Based on Krogh 1970, "Efficient Algorithms for Polynomial Interpolation
    and Numerical Differentiation"
    """
    def __init__(self, xi, yi):
        """Construct an interpolator passing through the specified points

        The polynomial passes through all the pairs (xi,yi). One may additionally
        specify a number of derivatives at each point xi; this is done by
        repeating the value xi and specifying the derivatives as successive
        yi values.

        Parameters
        ----------
        xi : array-like, length N
            known x-coordinates
        yi : array-like, N by R
            known y-coordinates, interpreted as vectors of length R,
            or scalars if R=1

        Example
        -------
        To produce a polynomial that is zero at 0 and 1 and has
        derivative 2 at 0, call

        >>> KroghInterpolator([0,0,1],[0,2,0])
        """
        self.xi = npy.asarray(xi)
        self.yi = npy.asarray(yi)
        if len(self.yi.shape)==1:
            self.vector_valued = False
            self.yi = self.yi[:,npy.newaxis]
        elif len(self.yi.shape)>2:
            raise ValueError("y coordinates must be either scalars or vectors")
        else:
            self.vector_valued = True

        n = len(xi)
        self.n = n
        nn, r = self.yi.shape
        if nn!=n:
            raise ValueError("%d x values provided and %d y values; must be equal" % (n, nn))
        self.r = r

        c = npy.zeros((n+1,r))
        c[0] = yi[0]
        Vk = npy.zeros((n,r))
        for k in range(1,n):
            s = 0
            while s<=k and xi[k-s]==xi[k]:
                s += 1
            s -= 1
            Vk[0] = yi[k]/float(factorial(s))
            for i in range(k-s):
                assert xi[i]!=xi[k]
                if s==0:
                    Vk[i+1] = (c[i]-Vk[i])/(xi[i]-xi[k])
                else:
                    Vk[i+1] = (Vk[i+1]-Vk[i])/(xi[i]-xi[k])
            c[k] = Vk[k-s]
        self.c = c

    def __call__(self,x):
        """Evaluate the polynomial at the point x

        Parameters
        ----------
        x : scalar or array-like of length N

        Returns
        -------
        y : scalar, array of length R, array of length N, or array of length N by R
            If x is a scalar, returns either a vector or a scalar depending on
            whether the interpolator is vector-valued or scalar-valued.
            If x is a vector, returns a vector of values.
        """
        if npy.isscalar(x):
            scalar = True
            m = 1
        else:
            scalar = False
            m = len(x)
        x = npy.asarray(x)

        n = self.n
        pi = 1
        p = npy.zeros((m,self.r))
        p += self.c[0,npy.newaxis,:]
        for k in range(1,n):
            w = x - self.xi[k-1]
            pi = w*pi
            p = p + npy.multiply.outer(pi,self.c[k])
        if not self.vector_valued:
            if scalar:
                return p[0,0]
            else:
                return p[:,0]
        else:
            if scalar:
                return p[0]
            else:
                return p

    def derivatives(self,x,der=None):
        """Evaluate many derivatives of the polynomial at the point x

        Produce an array of all derivative values at the point x.

        Parameters
        ----------
        x : scalar or array-like of length N
            Point or points at which to evaluate the derivatives
        der : None or integer
            How many derivatives to extract; None for all potentially
            nonzero derivatives (that is a number equal to the number
            of points). This number includes the function value as 0th
            derivative.
        Returns
        -------
        d : array
            If the interpolator's values are R-dimensional then the
            returned array will be der by N by R. If x is a scalar,
            the middle dimension will be dropped; if R is 1 then the
            last dimension will be dropped.

        Example
        -------
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives(0)
        array([1.0,2.0,3.0])
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives([0,0])
        array([[1.0,1.0],
               [2.0,2.0],
               [3.0,3.0]])
        """
        if npy.isscalar(x):
            scalar = True
            m = 1
        else:
            scalar = False
            m = len(x)
        x = npy.asarray(x)

        n = self.n
        r = self.r

        if der is None:
            der = self.n
        dern = min(self.n,der)
        pi = npy.zeros((n,m))
        w = npy.zeros((n,m))
        pi[0] = 1
        p = npy.zeros((m,self.r))
        p += self.c[0,npy.newaxis,:]

        for k in range(1,n):
            w[k-1] = x - self.xi[k-1]
            pi[k] = w[k-1]*pi[k-1]
            p += npy.multiply.outer(pi[k],self.c[k])

        cn = npy.zeros((max(der,n+1),m,r))
        cn[:n+1,...] += self.c[:n+1,npy.newaxis,:]
        cn[0] = p
        for k in range(1,n):
            for i in range(1,n-k+1):
                pi[i] = w[k+i-1]*pi[i-1]+pi[i]
                cn[k] = cn[k]+pi[i,:,npy.newaxis]*cn[k+i]
            cn[k]*=factorial(k)

        cn[n,...] = 0
        if not self.vector_valued:
            if scalar:
                return cn[:der,0,0]
            else:
                return cn[:der,:,0]
        else:
            if scalar:
                return cn[:der,0]
            else:
                return cn[:der]
    def derivative(self,x,der):
        """Evaluate one derivative of the polynomial at the point x

        Parameters
        ----------
        x : scalar or array-like of length N
            Point or points at which to evaluate the derivatives
        der : None or integer
            Which derivative to extract. This number includes the
            function value as 0th derivative.
        Returns
        -------
        d : array
            If the interpolator's values are R-dimensional then the
            returned array will be N by R. If x is a scalar,
            the middle dimension will be dropped; if R is 1 then the
            last dimension will be dropped.

        Notes
        -----
        This is computed by evaluating all derivatives up to the desired
        one and then discarding the rest.
        """
        return self.derivatives(x,der=der+1)[der]


class BarycentricInterpolator(object):
    """The interpolating polynomial for a set of points

    Constructs a polynomial that passes through a given set of points.
    Allows evaluation of the polynomial, efficient changing of the y
    values to be interpolated, and updating by adding more x values.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial.

    This class uses a "barycentric interpolation" method that treats
    the problem as a special case of rational function interpolation.
    This algorithm is quite stable, numerically, but even in a world of
    exact computation, unless the x coordinates are chosen very
    carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon.

    Based on Berrut and Trefethen 2004, "Barycentric Lagrange Interpolation".
    """
    def __init__(self, xi, yi=None):
        """Construct an object capable of interpolating functions sampled at xi

        The values yi need to be provided before the function is evaluated,
        but none of the preprocessing depends on them, so rapid updates
        are possible.

        Parameters
        ----------
        xi : array-like of length N
            The x coordinates of the points the polynomial should pass through
        yi : array-like N by R or None
            The y coordinates of the points the polynomial should pass through;
            if R>1 the polynomial is vector-valued. If None the y values
            will be supplied later.
        """
        self.n = len(xi)
        self.xi = npy.asarray(xi)
        if yi is not None and len(yi)!=len(self.xi):
            raise ValueError("yi dimensions do not match xi dimensions")
        self.set_yi(yi)
        self.wi = npy.zeros(self.n)
        self.wi[0] = 1
        for j in range(1,self.n):
            self.wi[:j]*=(self.xi[j]-self.xi[:j])
            self.wi[j] = npy.multiply.reduce(self.xi[:j]-self.xi[j])
        self.wi**=-1

    def set_yi(self, yi):
        """Update the y values to be interpolated

        The barycentric interpolation algorithm requires the calculation
        of weights, but these depend only on the xi. The yi can be changed
        at any time.

        Parameters
        ----------
        yi : array-like N by R
            The y coordinates of the points the polynomial should pass through;
            if R>1 the polynomial is vector-valued. If None the y values
            will be supplied later.
        """
        if yi is None:
            self.yi = None
            return
        yi = npy.asarray(yi)
        if len(yi.shape)==1:
            self.vector_valued = False
            yi = yi[:,npy.newaxis]
        elif len(yi.shape)>2:
            raise ValueError("y coordinates must be either scalars or vectors")
        else:
            self.vector_valued = True

        n, r = yi.shape
        if n!=len(self.xi):
            raise ValueError("yi dimensions do not match xi dimensions")
        self.yi = yi
        self.r = r


    def add_xi(self, xi, yi=None):
        """Add more x values to the set to be interpolated

        The barycentric interpolation algorithm allows easy updating by
        adding more points for the polynomial to pass through.

        Parameters
        ----------
        xi : array-like of length N1
            The x coordinates of the points the polynomial should pass through
        yi : array-like N1 by R or None
            The y coordinates of the points the polynomial should pass through;
            if R>1 the polynomial is vector-valued. If None the y values
            will be supplied later. The yi should be specified if and only if
            the interpolator has y values specified.
        """
        if yi is not None:
            if self.yi is None:
                raise ValueError("No previous yi value to update!")
            yi = npy.asarray(yi)
            if len(yi.shape)==1:
                if self.vector_valued:
                    raise ValueError("Cannot extend dimension %d y vectors with scalars" % self.r)
                yi = yi[:,npy.newaxis]
            elif len(yi.shape)>2:
                raise ValueError("y coordinates must be either scalars or vectors")
            else:
                n, r = yi.shape
                if r!=self.r:
                    raise ValueError("Cannot extend dimension %d y vectors with dimension %d y vectors" % (self.r, r))

            self.yi = npy.vstack((self.yi,yi))
        else:
            if self.yi is not None:
                raise ValueError("No update to yi provided!")
        old_n = self.n
        self.xi = npy.concatenate((self.xi,xi))
        self.n = len(self.xi)
        self.wi**=-1
        old_wi = self.wi
        self.wi = npy.zeros(self.n)
        self.wi[:old_n] = old_wi
        for j in range(old_n,self.n):
            self.wi[:j]*=(self.xi[j]-self.xi[:j])
            self.wi[j] = npy.multiply.reduce(self.xi[:j]-self.xi[j])
        self.wi**=-1

    def __call__(self, x):
        """Evaluate the interpolating polynomial at the points x

        Parameters
        ----------
        x : scalar or array-like of length M

        Returns
        -------
        y : scalar or array-like of length R or length M or M by R
            The shape of y depends on the shape of x and whether the
            interpolator is vector-valued or scalar-valued.

        Notes
        -----
        Currently the code computes an outer product between x and the
        weights, that is, it constructs an intermediate array of size
        N by M, where N is the degree of the polynomial.
        """
        scalar = npy.isscalar(x)
        x = npy.atleast_1d(x)
        c = npy.subtract.outer(x,self.xi)
        z = c==0
        c[z] = 1
        c = self.wi/c
        p = npy.dot(c,self.yi)/npy.sum(c,axis=-1)[:,npy.newaxis]
        i, j = npy.nonzero(z)
        p[i] = self.yi[j]
        if not self.vector_valued:
            if scalar:
                return p[0,0]
            else:
                return p[:,0]
        else:
            if scalar:
                return p[0]
            else:
                return p

# RHC - made a sub-class of interpclass
class PiecewisePolynomial(interpclass):
    """Piecewise polynomial curve specified by points and derivatives.

    This class represents a curve that is a piecewise polynomial. It
    passes through a list of points and has specified derivatives at
    each point. The degree of the polynomial may very from segment to
    segment, as may the number of derivatives available. The degree
    should not exceed about thirty.

    Appending points to the end of the curve is efficient.
    """
    def __init__(self, xi, yi, orders=None, direction=None):
        """Construct a piecewise polynomial

        Parameters
        ----------
        xi : array-like of length N
            a sorted list of x-coordinates
        yi : list of lists of length N
            yi[i] is the list of derivatives known at xi[i]
        orders : list of integers, or integer
            a list of polynomial orders, or a single universal order
        direction : {None, 1, -1}
            indicates whether the xi are increasing or decreasing
            +1 indicates increasing
            -1 indicates decreasing
            None indicates that it should be deduced from the first two xi

        Notes
        -----
        If orders is None, or orders[i] is None, then the degree of the
        polynomial segment is exactly the degree required to match all i
        available derivatives at both endpoints. If orders[i] is not None,
        then some derivatives will be ignored. The code will try to use an
        equal number of derivatives from each end; if the total number of
        derivatives needed is odd, it will prefer the rightmost endpoint. If
        not enough derivatives are available, an exception is raised.
        """
        # RHC added datapoints for use by PyDSTool
        # don't store any derivative information in datapoints
        self.datapoints = (array(xi, float), array(yi[:,0], float))
        self.type = float    # RHC -- for access from PyDSTool
        yi0 = npy.asarray(yi[0])
        if len(yi0.shape)==2:
            self.vector_valued = True
            self.r = yi0.shape[1]
        elif len(yi0.shape)==1:
            self.vector_valued = False
            self.r = 1
        else:
            raise ValueError("Each derivative must be a vector, not a higher-rank array")

        self.xi = [xi[0]]
        self.yi = [yi0]
        self.n = 1

        self.direction = direction
        self.orders = []
        self.polynomials = []
        self.extend(xi[1:],yi[1:],orders)

    def _make_polynomial(self,x1,y1,x2,y2,order,direction):
        """Construct the interpolating polynomial object

        Deduces the number of derivatives to match at each end
        from order and the number of derivatives available. If
        possible it uses the same number of derivatives from
        each end; if the number is odd it tries to take the
        extra one from y2. In any case if not enough derivatives
        are available at one end or another it draws enough to
        make up the total from the other end.
        """
        n = order+1
        n1 = min(n//2,len(y1))
        n2 = min(n-n1,len(y2))
        n1 = min(n-n2,len(y1))
        if n1+n2!=n:
            raise ValueError("Point %g has %d derivatives, point %g has %d derivatives, but order %d requested" % (x1, len(y1), x2, len(y2), order))
        assert n1<=len(y1)
        assert n2<=len(y2)

        xi = npy.zeros(n)
        if self.vector_valued:
            yi = npy.zeros((n,self.r))
        else:
            yi = npy.zeros((n,))

        xi[:n1] = x1
        yi[:n1] = y1[:n1]
        xi[n1:] = x2
        yi[n1:] = y2[:n2]

        return KroghInterpolator(xi,yi)

    def append(self, xi, yi, order=None):
        """Append a single point with derivatives to the PiecewisePolynomial

        Parameters
        ----------
        xi : float
        yi : array-like
            yi is the list of derivatives known at xi
        order : integer or None
            a polynomial order, or instructions to use the highest
            possible order
        """

        yi = npy.asarray(yi)
        if self.vector_valued:
            if (len(yi.shape)!=2 or yi.shape[1]!=self.r):
                raise ValueError("Each derivative must be a vector of length %d" % self.r)
        else:
            if len(yi.shape)!=1:
                raise ValueError("Each derivative must be a scalar")

        if self.direction is None:
            self.direction = npy.sign(xi-self.xi[-1])
        elif (xi-self.xi[-1])*self.direction < 0:
            raise ValueError("x coordinates must be in the %d direction: %s" % (self.direction, self.xi))

        self.xi.append(xi)
        self.yi.append(yi)


        if order is None:
            n1 = len(self.yi[-2])
            n2 = len(self.yi[-1])
            n = n1+n2
            order = n-1

        self.orders.append(order)
        self.polynomials.append(self._make_polynomial(
            self.xi[-2], self.yi[-2],
            self.xi[-1], self.yi[-1],
            order, self.direction))
        self.n += 1


    def extend(self, xi, yi, orders=None):
        """Extend the PiecewisePolynomial by a list of points

        Parameters
        ----------
        xi : array-like of length N1
            a sorted list of x-coordinates
        yi : list of lists of length N1
            yi[i] is the list of derivatives known at xi[i]
        orders : list of integers, or integer
            a list of polynomial orders, or a single universal order
        direction : {None, 1, -1}
            indicates whether the xi are increasing or decreasing
            +1 indicates increasing
            -1 indicates decreasing
            None indicates that it should be deduced from the first two xi
        """

        for i in range(len(xi)):
            if orders is None or npy.isscalar(orders):
                self.append(xi[i],yi[i],orders)
            else:
                self.append(xi[i],yi[i],orders[i])

    def __call__(self, x):
        """Evaluate the piecewise polynomial

        Parameters
        ----------
        x : scalar or array-like of length N

        Returns
        -------
        y : scalar or array-like of length R or length N or N by R
        """
        if npy.isscalar(x):
            pos = npy.clip(npy.searchsorted(self.xi, x) - 1, 0, self.n-2)
            y = self.polynomials[pos](x)
        else:
            x = npy.asarray(x)
            m = len(x)
            pos = npy.clip(npy.searchsorted(self.xi, x) - 1, 0, self.n-2)
            if self.vector_valued:
                y = npy.zeros((m,self.r))
            else:
                y = npy.zeros(m)
            for i in range(self.n-1):
                c = pos==i
                y[c] = self.polynomials[i](x[c])
        return y

    def derivative(self, x, der):
        """Evaluate a derivative of the piecewise polynomial

        Parameters
        ----------
        x : scalar or array-like of length N
        der : integer
            which single derivative to extract

        Returns
        -------
        y : scalar or array-like of length R or length N or N by R

        Notes
        -----
        This currently computes all derivatives of the curve segment
        containing each x but returns only one. This is because the
        number of nonzero derivatives that a segment can have depends
        on the degree of the segment, which may vary.
        """
        return self.derivatives(x,der=der+1)[der]

    def derivatives(self, x, der):
        """Evaluate a derivative of the piecewise polynomial

        Parameters
        ----------
        x : scalar or array-like of length N
        der : integer
            how many derivatives (including the function value as
            0th derivative) to extract

        Returns
        -------
        y : array-like of shape der by R or der by N or der by N by R

        """
        if npy.isscalar(x):
            pos = npy.clip(npy.searchsorted(self.xi, x) - 1, 0, self.n-2)
            y = self.polynomials[pos].derivatives(x,der=der)
        else:
            x = npy.asarray(x)
            m = len(x)
            pos = npy.clip(npy.searchsorted(self.xi, x) - 1, 0, self.n-2)
            if self.vector_valued:
                y = npy.zeros((der,m,self.r))
            else:
                y = npy.zeros((der,m))
            for i in range(self.n-1):
                c = pos==i
                y[:,c] = self.polynomials[i].derivatives(x[c],der=der)
        return y
    # FIXME: provide multiderivative finder

    # RHC added
    def __getstate__(self):
        d = copy(self.__dict__)
        # remove reference to Cfunc self.type
        d['type'] = _num_type2name[self.type]
        return d

    # RHC added
    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc self.type
        self.type = _num_name2type[self.type]

# --------------------------------------------------------------------

def simple_bisection(tlo, thi, f, tol, imax=100):
    sol = None
    flo = f(tlo)
    fhi = f(thi)
    i = 1
    while i <= imax:
        d = (thi - tlo)/2.
        p = tlo + d
        if d < tol:
            sol = p
            break
        fp = f(p)
        if fp == 0:
            sol = p
            break
        i += 1
        if fp*flo > 0:
            tlo = p
            flo = fp
        else:
            thi = p
    if i == imax:
        sol = p
    return sol

# Function fitting tools

class fit_function(object):
    """Abstract super-class for fitting explicit functions to 1D arrays of data
    using least squares.

    xs -- independent variable data
    ys -- dependent variable data
    pars_ic -- initial values defining the function

    Optional algorithmic parameters to minpack.leastsq can be passed in the
    algpars argument: e.g.,
    ftol -- Relative error desired in the sum of squares (default 1e-6).
    xtol -- Relative error desired in the approximate solution (default 1e-6).
    gtol -- Orthogonality desired between the function vector
            and the columns of the Jacobian (default 1e-8).

    Other parameters may be used for concrete sub-classes. Pass these as a dict
    or args object in the opts argument.

    Returns an args object with attributes:

    ys_fit --   the fitted y values corresponding to the given x data,
    pars_fit -- the function parameters at the fit
    info --     diagnostic feedback from the leastsq algorithm
    results --  dictionary of other function specific information (such as peak
                 position)
    """

    def __init__(self, pars_ic=None, algpars=None, opts=None,
                 verbose=False):
        # defaults
        self.algpars = args(ftol=1e-8, xtol=1e-6, gtol=1e-8, maxfev=100)
        if algpars is not None:
            self.algpars.update(algpars)
        self.verbose = verbose
        self.pars_ic = pars_ic
        if hasattr(opts, 'weight'):
            self.weight = opts.weight
        else:
            self.weight = 1

    def fn(self, x, *pars):
        raise NotImplementedError("Override in a concrete sub-class")

    def _do_fit(self, constraint, xs, ys, pars_ic):
        xs = asarray(xs)
        ys = asarray(ys)
        weight = self.weight

        if constraint is None:
            if self.verbose:
                def res_fn(p):
                    print("\n%r" % p)
                    r = self.fn(xs, *p) - ys
                    print("Residual = %f"%norm(r*weight))
                    return r*weight
            else:
                def res_fn(p):
                    r = self.fn(xs, *p) - ys
                    return r*weight
        else:
            if self.verbose:
                def res_fn(p):
                    print("\n%r" % (p,))
                    r = npy.concatenate((constraint(*p), (self.fn(xs, *p) - ys)*weight))
                    print("Residual = %f"%norm(r))
                    return r
            else:
                def res_fn(p):
                    return npy.concatenate((constraint(*p), (self.fn(xs, *p) - ys)*weight))

        try:
            res = minpack.leastsq(res_fn, pars_ic,
                              full_output = True,
                              ftol = self.algpars.ftol,
                              xtol = self.algpars.xtol,
                              gtol = self.algpars.gtol,
                              maxfev = self.algpars.maxfev)
        except:
            print("Error at parameters %r" % pars_ic)
            raise
        if self.verbose:
            print("Result: %r" % (res,))
        return res

    def fit(self, xs, ys, pars_ic=None, opts=None):
        raise NotImplementedError("Override in a concrete sub-class")


class fit_quadratic(fit_function):
    """Fit a quadratic function y=a*x^2+b*x+c to the (x,y) array data.
    If initial parameter values = (a,b,c) are not given, the values
    (1,1,0) will be used.

    If peak_constraint is a tuple of values (x_index, y_value, weight_x,
    weight_y) for the approximate position of a turning point in the data,
    then this will be used as a soft constraint in the fit.

    result.peak is a (xpeak, ypeak) pair.
    result.f is the fitted function (accepts x values).
    """

    def fn(self, x, a, b, c):
        return a*x**2+b*x+c

    def fit(self, xs, ys, pars_ic=None, opts=None):
        try:
            peak_constraint = opts.peak_constraint
        except AttributeError:
            peak_constraint = None

        if pars_ic is None:
            if self.pars_ic is None:
                pars_ic = array([1.,1.,0.])
            else:
                pars_ic = self.pars_ic

        if peak_constraint is None:
            constraint = None
        else:
            x_index, y_value, weight_x, weight_y = peak_constraint
            def constraint(a,b,c):
                return array([weight_y*(self.fn(xs[x_index],a,b,c)-y_value),
                              weight_x*(xs[x_index]+b/(2*a))])
        res = self._do_fit(constraint, xs, ys, pars_ic)
        sol = res[0]
        a,b,c = sol
        def f(x):
            return a*x**2+b*x+c
        ys_fit = f(xs)
        xpeak = -b/(2*a)
        ypeak = f(xpeak)
        return args(ys_fit=ys_fit, pars_fit=(a,b,c), info=res,
                          results=args(peak=(xpeak, ypeak),
                                       f=f))

class fit_quadratic_at_vertex(fit_function):
    """Fit a quadratic function y=a*(x+h)**2+k to the (x,y) array data,
    constrained to have a vertex at (h, k), leaving only the free parameter
    a for the curvature. (h, k) is specified through the peak_constraint
    option in the initialization argument 'opts'.

    If initial parameter value = a is not given, the value 1 will be used.

    result.peak is a (xpeak, ypeak) pair, but corresponds to (h,k).
    result.f is the fitted function (accepts x values).
    """

    def fn(self, x, a):
        return a*(x+self.h)**2+self.k

    def fit(self, xs, ys, pars_ic=None, opts=None):
        self.h, self.k = opts.peak_constraint
        if pars_ic is None:
            if self.pars_ic is None:
                pars_ic = (1,)
            else:
                pars_ic = (self.pars_ic,)

        res = self._do_fit(None, xs, ys, pars_ic)
        sol = res[0]
        a = sol
        def f(x):
            return a*(x+self.h)**2+self.k
        ys_fit = f(xs)
        return args(ys_fit=ys_fit, pars_fit=a, info=res,
                          results=args(peak=(self.h, self.k),
                                       f=f))

class fit_cubic(fit_function):
    """Fit a cubic function y=a*x^3+b*x^2+c*x+d to the (x,y) array data.
    If initial parameter values = (a,b,c,d) are not given, the values
    (1,1,1,0) will be used.

    result.f is the fitted function (accepts x values).
    """

    def fn(self, x, a, b, c,d):
        return a*x**3+b*x*x+c*x+d

    def fit(self, xs, ys, pars_ic=None, opts=None):
        if pars_ic is None:
            if self.pars_ic is None:
                pars_ic = array([1.,1.,1.,0.])
            else:
                pars_ic = self.pars_ic

        res = self._do_fit(None, xs, ys, pars_ic)
        sol = res[0]
        a,b,c,d = sol
        def f(x):
            return a*x**3+b*x*x+c*x+d
        ys_fit = f(xs)
        return args(ys_fit=ys_fit, pars_fit=(a,b,c,d), info=res,
                          results=args(f=f))



class fit_exponential(fit_function):
    """Fit an exponential function y=a*exp(b*x) to the (x,y) array data.
    If initial parameter values = (a,b) are not given, the values
    (1,-1) will be used.

    result.f is the fitted function (accepts x values).
    """

    def fn(self, x, a, b):
        return a*exp(b*x)

    def fit(self, xs, ys, pars_ic=None, opts=None):
        if pars_ic is None:
            if self.pars_ic is None:
                pars_ic = array([1.,-1.])
            else:
                pars_ic = self.pars_ic

        res = self._do_fit(None, xs, ys, pars_ic)
        sol = res[0]
        a,b = sol
        def f(x):
            return a*exp(b*x)
        ys_fit = f(xs)
        return args(ys_fit=ys_fit, pars_fit=(a,b), info=res,
                          results=args(f=f))


class fit_diff_of_exp(fit_function):
    """Fit a 'difference of two exponentials' function
    y = k*a*b*(exp(-a*x)-exp(-b*x))/(b-a) to the (x,y) array data.
    If initial parameter values = (k,a,b) are not given, the values
    (1,1,1) will be used (where the function degenerates to
    y = k*a*a*x*exp(-a*x).

    Optional use_xoff feature adds offset to x, so that
    y = k*a*a*(x+xoff)*exp(-a*(x+xoff))     (yes, "+ xoff")
    etc., in case fitting data that starts at larger values than its tail.
    Then initial parameter values will be (1,1,1,0) unless given otherwise.

    If peak_constraint option is used, it is a tuple of values (x_index,
    y_value, weight_x, weight_y) for the approximate position of a turning point
    in the data, then this will be used as a soft constraint in the fit.

    result.peak_pos is a (xpeak, ypeak) pair.
    result.f is the fitted function (accepts x values).
    """

    def fn(self, x, k, a, b, xoff=0):
        if a==b:
            # classic "alpha" function
            return k*a*a*((x+xoff)*exp(-a*(x+xoff)) - xoff*exp(-a*xoff))
        else:
            return k*a*b*(exp(-a*(x+xoff))+exp(-a*xoff)-exp(-b*(x+xoff))-exp(-b*xoff))/(b-a)

    def fit(self, xs, ys, pars_ic=None, opts=None):
        try:
            peak_constraint = opts.peak_constraint
        except AttributeError:
            peak_constraint = None
        try:
            use_xoff = opts.use_xoff
        except AttributeError:
            use_xoff = False

        def peak_pos(k, a, b, xoff=0):
            if a==b:
                return 1./a - xoff
            else:
                return ((b-a)*xoff+log(a/b))/(a-b)

        if pars_ic is None:
            if self.pars_ic is None:
                if use_xoff:
                    pars_ic = array([1.,1.,1.,0.])
                else:
                    pars_ic = array([1.,1.,1.])
            else:
                pars_ic = self.pars_ic
                if (len(self.pars_ic) == 4 and not use_xoff) or \
                   (len(self.pars_ic) == 3 and use_xoff):
                    raise ValueError("Inconsistent use_xoff setting with pars_ic")

        if peak_constraint is None:
            constraint = None
        else:
            x_index, y_value, weight_x, weight_y = peak_constraint
            def constraint(k,a,b,xoff=0):
                return array([weight_y*(self.fn(xs[x_index],k,a,b,xoff)-y_value),
                              weight_x*(xs[x_index]-peak_pos(k,a,b,xoff))])
        res = self._do_fit(constraint, xs, ys, pars_ic)
        sol = res[0]
        if use_xoff:
            k,a,b,xoff = sol
        else:
            k,a,b = sol
            xoff = 0
        if xoff == 0:
            if a == b:
                # exceptional case
                def f(x):
                    return k*a*a*x*exp(-a*x)
            else:
                def f(x):
                    return k*a*b*(exp(-a*x)-exp(-b*x))/(b-a)
        else:
            if a == b:
                # exceptional case
                def f(x):
                    return k*a*a*((x+xoff)*exp(-a*(x+xoff)) - xoff*exp(-a*xoff))
            else:
                def f(x):
                    return k*a*b*(exp(-a*(x+xoff))+exp(-a*xoff)-exp(-b*(x+xoff))-exp(-b*xoff))/(b-a)
        ys_fit = f(xs)
        xpeak = peak_pos(k,a,b,xoff)
        ypeak = f(xpeak)
        if use_xoff:
            pars_fit = (k, a, b, xoff)
        else:
            pars_fit = (k, a, b)
        return args(ys_fit=ys_fit, pars_fit=pars_fit, info=res,
                          results=args(peak=(xpeak, ypeak),
                                       f=f))

class fit_linear(fit_function):
    """Fit a linear function y=a*x+b to the (x,y) array data.
    If initial parameter values = (a,b) are not given, the values
    (1,0) will be used.

    result.f is the fitted function (accepts x values).
    """

    def fn(self, x, a, b):
        return a*x+b

    def fit(self, xs, ys, pars_ic=None, opts=None):
        if pars_ic is None:
            if self.pars_ic is None:
                pars_ic = array([1.,0.])
            else:
                pars_ic = self.pars_ic

        res = self._do_fit(None, xs, ys, pars_ic)
        sol = res[0]
        a,b = sol
        def f(x):
            return a*x+b
        ys_fit = f(xs)
        return args(ys_fit=ys_fit, pars_fit=(a,b), info=res,
                          results=args(f=f))

def make_poly_interpolated_curve(pts, coord, model):
    """Only for a 1D curve from a Model object (that has an associated
    vector field for defining 1st derivative of curve).
    """
    coord_ix = pts.coordnames.index(coord)
    x = pts[coord]
    t = pts.indepvararray
    p = model.query('pars')
    dx = array([model.Rhs(tval, pts[tix], p, asarray=True)[coord_ix] for \
                tix, tval in enumerate(t)])
    return PiecewisePolynomial(t, array([x, dx]).T, 2)

def smooth_pts(t, x, q=None):
    """Use a local quadratic fit on a set of nearby 1D points and obtain
    a function that represents that fit in that neighbourhood. Returns a
    structure (args object) with attributes ys_fit, pars_fit, info, and
    results. The function can be referenced as results.f

    Assumed that pts is small enough that it is either purely concave up or
    down but that at it contains at least five points.

    If this function is used repeatedly, pass a fit_quadratic instance
    as the argument q
    """
    ## Uncomment verbose-related statements for debugging
#    verbose = True
    if q is None:
        q = fit_quadratic(verbose=False)  # verbose=verbose
    ixlo = 0
    ixhi = len(t)-1
    assert ixhi >= 4, "Provide at least five points"
    # concavity assumed to be simple: whether midpoint of x
    # is above or below the chord between the endpoints
    midpoint_ix = int(ixhi/2.)
    midpoint_chord = x[0]+(t[midpoint_ix]-t[0])*(x[-1]-x[0])/(t[-1]-t[0])
    midpoint_x = x[midpoint_ix]
    # a_sign is -1 if concave down
    a_sign = sign(midpoint_chord - midpoint_x)
    ixmax = argmax(x)
    ixmin = argmin(x)
    # to estimate |a| need to know where best to put centre for
    # central second difference formula:
    # if extremum not at endpoints then use one endpoint
    # else use central point
    if (ixmax in (ixhi, ixlo) and a_sign == -1) or \
       (ixmin in (ixhi, ixlo) and a_sign == 1):
        # use central point, guaranteed to be at least 2 indices away from
        # ends
        ix_cent = midpoint_ix
    else:
        # use an endpoint + 1
        ix_cent = ixlo+2
    # use mean of right and left t steps as h (should be safe for
    # smooth enough data)
    h = 0.25*(t[ix_cent+2]-t[ix_cent-2])
    second_diff = (-x[ix_cent-2]+16*x[ix_cent-1]-30*x[ix_cent]+\
                   +16*x[ix_cent+1]-x[ix_cent+2])/(12*h**2)
    assert sign(second_diff) == a_sign, "Data insufficiently smooth"
    # a_est based on second deriv of quadratic formula = 2a
    a_est = second_diff/2.
    if a_sign == -1:
        extreme_x = x[ixmin]
        extreme_t = t[ixmin]
    else:
        extreme_x = x[ixmax]
        extreme_t = t[ixmax]
    # using vertex form of quadratic, x = a*( t-extreme_t )^2 + extreme_x
    # then in regular formula x = at^2 + bt + c used by quadratic fit class,
    # b = -2*extreme_t, and c = a*extreme_t^2 + extreme_x
    b_est = -2*extreme_t
    c_est = a_est*extreme_t**2 + extreme_x
    return q.fit(t, x, pars_ic=(a_est,b_est,c_est))
    # for debugging, set res = q.fit() and then return it after the following...
#    if verbose:
#        print "h =", h
#        print "a_est =", a_est, "b_est =", b_est, "c_est =", c_est
#        print "extremum estimate at (%f,%f)"%(extreme_t,extreme_x)
#        plot(t, x, 'go-')
#        tval, xval = res.results.peak
#        plot(tval, xval, 'rx')
#        xs_fit = res.ys_fit
#        plot(t, xs_fit, 'k:')

def nearest_2n_indices(x, i, n):
    """Calculates the nearest 2n indices centred at i in an array x, or as close
    as possible to i, taking into account that i might be within n indices of
    an endpoint of x.

    The function returns the limiting indices as a pair, and always returns
    an interval that contains 2n+1 indices, assuming x is long enough.

    I.e., away from endpoints, the function returns (i-n, i+n).
    If i is within n of index 0, the function returns (0, 2n).
    If i is within n of last index L, the function returns (L-2n, L).

    Remember to add one to the upper limit if using it in a slice.
    """
    assert len(x) > 2*n, "x is not long enough"
    # ixlo = 0
    ixhi = len(x)-1
    if i < n:
        # too close to low end
        return (0, 2*n)
    elif i > ixhi - n:
        # too close to high end
        return (ixhi-2*n, ixhi)
    else:
        return (i-n, i+n)


# --------------------------------------------------------------------

class DomainType(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        try:
            return self.name == other.name
        except:
            return False

    def __ne__(self, other):
        try:
            return self.name != other.name
        except:
            return False

    def __repr__(self):
        return self.name

    __str__ = __repr__

# treat these as "constants" as they are empty
global Continuous, Discrete

Continuous = DomainType("Continuous Domain")
Discrete = DomainType("Discrete Domain")
