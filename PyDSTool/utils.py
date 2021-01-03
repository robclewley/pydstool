"""
    User utilities.
"""

from distutils.util import get_platform
from numpy.distutils import misc_util

from .errors import *
from .common import *
from .parseUtils import joinStrs
from PyDSTool.core.context_managers import RedirectStdout

# !! Replace use of these named imports with np.<X>
from numpy import isfinite, less, greater, sometrue, alltrue, \
     searchsorted, take, argsort, array, swapaxes, asarray, zeros, transpose, \
     float64, int32, argmin, ndarray, concatenate
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minpack, zeros
try:
    newton_meth = minpack.newton
except AttributeError:
    # newer version of scipy
    newton_meth = zeros.newton
import time, sys, os, platform
import copy


# --------------------------------------------------------------------

# EXPORTS

_classes = []

_functions = ['intersect', 'remain', 'union', 'cartesianProduct',
              'makeImplicitFunc', 'orderEventData',
              'saveObjects', 'loadObjects', 'info', 'compareList',
              'findClosestArray', 'findClosestPointIndex', 'find',
              'makeMfileFunction', 'make_RHS_wrap', 'make_Jac_wrap',
              'progressBar', 'distutil_destination', 'architecture',
              'extra_arch_arg', 'arclength']

_mappings = ['_implicitSolveMethods', '_1DimplicitSolveMethods']

__all__ = _classes + _functions + _mappings



## ------------------------------------------------------------------
# File for stdout redirecting
_logfile = os.devnull

## Utility functions

def makeMfileFunction(name, argname, defs):
    """defs is a dictionary of left-hand side -> right-hand side definitions"""
    # writeout file <name>.m
    mfile = open(name+".m", 'w')
    mfile.write("function %s = %s(%s)\n"%(name,name,argname))
    for k, v in defs.items():
        if k != name:
            mfile.write("%s = %s;\n"%(k,v))
    # now the final definition of tau_recip or inf
    mfile.write("%s = %s;\n"%(name,defs[name]))
    mfile.write("return\n")
    mfile.close()


def info(x, specName="Contents", offset=1, recurseDepth=1,
                 recurseDepthLimit=2, _repeatFirstTime=False):
    """Pretty printer for showing argument lists and dictionary
    specifications."""

    if recurseDepth == 1:
        if not _repeatFirstTime:
            # first time through
            print("Information for " + specName + "\n")
    else:
        print(specName + ":", end=' ')
    if x.__class__ is type:
        return
    if hasattr(x, 'items'):
        x_keys = sortedDictKeys(x)
        if len(x_keys) == 0:
            print("< empty >")
        elif recurseDepth != 1:
            print("")
        for k in x_keys:
            v = x[k]
            kstr = object2str(k)
            basestr = " "*(offset-1) + kstr
            if hasattr(v, 'items'):
                info(v, basestr, offset+4, recurseDepth+1,
                             recurseDepthLimit)
            else:
                vStrList = object2str(v).split(', ')
                if len(vStrList)==0:
                    vStrList = ['< no information >']
                elif len(vStrList)==1 and vStrList[0] == '':
                    vStrList = ['< empty >']
                outStrList = [basestr+": "]
                for i in range(len(vStrList)):
                    if len(vStrList[i] + outStrList[-1]) < 78:
                        outStrList[-1] += ", "*(i>0) + vStrList[i]
                    else:
                        if i>0:
                            if i != len(vStrList):
                                # add trailing comma to previous line
                                outStrList[-1] += ","
                            # start on new line
                            outStrList.append(" "*(len(kstr)+3) + vStrList[i])
                        else:
                            # too long for line and string has no commas
                            # could work harder here, but for now, just include
                            # the long line
                            outStrList[-1] += vStrList[i]
                if recurseDepth==1 and len(outStrList)>1:
                    # print an extra space between topmost level entries
                    # provided those entries occupy more than one line.
                    print("\n")
                for s in outStrList:
                    print(s)
    elif hasattr(x, '__dict__') and recurseDepth <= recurseDepthLimit:
        info(x.__dict__, specName, offset, recurseDepth,
                     recurseDepthLimit, True)
    else:
        xstr = repr(x)
        if xstr == '':
            xstr = '< no information >'
        print(xstr)


_implicitSolveMethods = ['newton', 'bisect', 'steffe', 'fsolve']
_1DimplicitSolveMethods = ['newton', 'bisect', 'steffe']


def makeImplicitFunc(f, x0, fprime=None, extrafargs=(), xtolval=1e-8,
                        maxnumiter=100, solmethod='newton', standalone=True):
    """Builds an implicit function representation of an N-dimensional curve
    specified by (N-1) equations. Thus argument f is a function of 1 variable.
    In the case of the 'fsolve' method, f may have dimension up to N-1.

    Available solution methods are: newton, bisect, steffensen, fsolve.
    All methods utilize SciPy's Minpack wrappers to Fortran codes.

    Steffenson uses Aitken's Delta-squared convergence acceleration.
    fsolve uses Minpack's hybrd and hybrj algorithms.

    Standalone option (True by default) returns regular function. If False,
    an additional argument is added, so as to be compatible as a method
    definition."""

    if solmethod == 'bisect':
        assert isinstance(x0, _seq_types), \
               "Invalid type '"+str(type(x0))+"' for x0 = "+str(x0)
        assert len(x0) == 2
    elif solmethod == 'fsolve':
        assert isinstance(x0, (_seq_types, _num_types)), \
               "Invalid type '"+str(type(x0))+"' for x0 = "+str(x0)
    else:
        assert isinstance(x0, _num_types), \
               "Invalid type '"+str(type(x0))+"' for x0 = "+str(x0)

    # define the functions that could be used
    # scipy signatures use y instead of t, but this naming is consistent
    # with that in the Generator module
    try:
        if standalone:
            def newton_fn(t):
                with RedirectStdout(_logfile):
                    res = float(newton_meth(f, x0, args=(t,)+extrafargs, tol=xtolval,
                                        maxiter=maxnumiter, fprime=fprime))
                    return res

            def bisect_fn(t):
                with RedirectStdout(_logfile):
                    res = minpack.bisection(f, x0[0], x0[1], args=(t,)+extrafargs,
                                        xtol=xtolval, maxiter=maxnumiter)
                    return res

            def steffe_fn(t):
                with RedirectStdout(_logfile):
                    res = minpack.fixed_point(f, x0, args=(t,)+extrafargs,
                                            xtol=xtolval, maxiter=maxnumiter)
                    return res

            def fsolve_fn(t):
                with RedirectStdout(_logfile):
                    res = minpack.fsolve(f, x0, args=(t,)+extrafargs,
                                        xtol=xtolval, maxfev=maxnumiter,
                                        fprime=fprime)
                    return res
        else:
            def newton_fn(s, t):
                with RedirectStdout(_logfile):
                    res = float(newton_meth(f, x0, args=(t,)+extrafargs, tol=xtolval,
                                        maxiter=maxnumiter, fprime=fprime))
                    return res

            def bisect_fn(s, t):
                with RedirectStdout(_logfile):
                    res = minpack.bisection(f, x0[0], x0[1], args=(t,)+extrafargs,
                                        xtol=xtolval, maxiter=maxnumiter)
                    return res

            def steffe_fn(s, t):
                with RedirectStdout(_logfile):
                    res = minpack.fixed_point(f, x0, args=(t,)+extrafargs,
                                            xtol=xtolval, maxiter=maxnumiter)
                    return res

            def fsolve_fn(s, t):
                with RedirectStdout(_logfile):
                    res = minpack.fsolve(f, x0, args=(t,)+extrafargs,
                                        xtol=xtolval, maxfev=maxnumiter,
                                        fprime=fprime)
                    return res

    except TypeError as e:
        if solmethod == 'bisect':
            infostr = " (did you specify a pair for x0?)"
        else:
            infostr = ""
        raise TypeError("Could not create function" +infostr + ": "+str(e))

    if solmethod == 'newton':
        return newton_fn
    elif solmethod == 'bisect':
        if fprime is not None:
            print("Warning: fprime argument unused for bisection method")
        return bisect_fn
    elif solmethod == 'steffe':
        if fprime is not None:
            print("Warning: fprime argument unused for aitken method")
        return steffe_fn
    elif solmethod == 'fsolve':
        return fsolve_fn
    else:
        raise ValueError("Unrecognized type of implicit function solver")


def findClosestPointIndex(pt, target, tol=np.Inf, in_order=True):
    """
    Find index of the closest N-dimensional Point in the target N by M array
    or Pointset. Uses norm of order given by the Point
    or Pointset, unless they are inconsistent, in which case an exception is
    raised, or unless they are both arrays, in which case 2-norm is assumed.

    With the in_order boolean option (default True), the function will
    attempt to determine the local "direction" of the values and return an
    insertion index that will preserve this ordering. This option is
    incompatible with the tol option (see below).

    If the optional tolerance, tol, is given, then an index is returned only
    if the closest distance is within the tolerance. Otherwise, a ValueError
    is raised. This option is incompatible with the in_order option.
    """
    try:
        normord = pt._normord
    except AttributeError:
        normord = 2
    try:
        if target._normord != normord:
            raise ValueError("Incompatible order of norm defined for inputs")
    except AttributeError:
        pass

    dists = [norm(pt-x, normord) for x in target]
    index = argmin(dists)

    if in_order:
        if index > 0:
            lo_off = 1
            # insertion offset index
            ins_off = 1
            if index < len(target):
                hi_off = 1
            else:
                hi_off = 0
        else:
            lo_off = 0
            hi_off = 2
            # insertion offset index
            ins_off = 0

        pta = array([pt]) # extra [] to get compatible shape for concat
        dim_range = list(range(target.shape[1]))
        # neighborhood
        nhood = target[index-lo_off:index+hi_off]
        if all(ismonotonic(nhood[:,d]) for d in dim_range):
            # try inserting at index, otherwise at index+1
            new_nhood = concatenate((nhood[:ins_off], pta, nhood[ins_off:]))
            if not all(ismonotonic(new_nhood[:,d]) for d in dim_range):
                ins_off += 1
                index += 1
                new_nhood = concatenate((nhood[:ins_off], pta, nhood[ins_off:]))
                if not all(ismonotonic(new_nhood[:,d]) for d in dim_range):
                    raise ValueError("Cannot add point in order, try deactivating the in_order option")

    if in_order:
        return index
    else:
        if dists[index] < tol:
            return index
        else:
            raise ValueError("No index found within distance tolerance")


def findClosestArray(input_array, target_array, tol):
    """
    Find the set of elements in (1D) input_array that are closest to
    elements in target_array.  Record the indices of the elements in
    target_array that are within tolerance, tol, of their closest
    match. Also record the indices of the elements in target_array
    that are outside tolerance, tol, of their match.

    For example, given an array of observations with irregular
    observation times along with an array of times of interest, this
    routine can be used to find those observations that are closest to
    the times of interest that are within a given time tolerance.

    NOTE: input_array must be sorted! The array, target_array, does not have to be sorted.

    Inputs:
      input_array:  a sorted float64 array
      target_array: a float64 array
      tol:          a tolerance

    Returns:
      closest_indices:  the array of indices of elements in input_array that are closest to elements in target_array

    Author: Gerry Wiener, 2004
    Version 1.0
    """
    # NOT RETURNED IN THIS VERSION:
#       accept_indices:  the indices of elements in target_array that have a match in input_array within tolerance
#      reject_indices:  the indices of elements in target_array that do not have a match in input_array within tolerance

    input_array_len = len(input_array)
    closest_indices = searchsorted(input_array, target_array) # determine the locations of target_array in input_array
#    acc_rej_indices = [-1] * len(target_array)
    curr_tol = [tol] * len(target_array)

    est_tol = 0.0
    for i in range(len(target_array)):
        best_off = 0          # used to adjust closest_indices[i] for best approximating element in input_array

        if closest_indices[i] >= input_array_len:
            # the value target_array[i] is >= all elements in input_array so check whether it is within tolerance of the last element
            closest_indices[i] = input_array_len - 1
            est_tol = target_array[i] - input_array[closest_indices[i]]
            if est_tol < curr_tol[i]:
                curr_tol[i] = est_tol
#                acc_rej_indices[i] = i
        elif target_array[i] == input_array[closest_indices[i]]:
            # target_array[i] is in input_array
            est_tol = 0.0
            curr_tol[i] = 0.0
#            acc_rej_indices[i] = i
        elif closest_indices[i] == 0:
            # target_array[i] is <= all elements in input_array
            est_tol = input_array[0] - target_array[i]
            if est_tol < curr_tol[i]:
                curr_tol[i] = est_tol
#                acc_rej_indices[i] = i
        else:
            # target_array[i] is between input_array[closest_indices[i]-1] and input_array[closest_indices[i]]
            # and closest_indices[i] must be > 0
            top_tol = input_array[closest_indices[i]] - target_array[i]
            bot_tol = target_array[i] - input_array[closest_indices[i]-1]
            if bot_tol <= top_tol:
                est_tol = bot_tol
                best_off = -1           # this is the only place where best_off != 0
            else:
                est_tol = top_tol

            if est_tol < curr_tol[i]:
                curr_tol[i] = est_tol
#                acc_rej_indices[i] = i

        if est_tol <= tol:
            closest_indices[i] += best_off

#    accept_indices = compress(greater(acc_rej_indices, -1),
#                                       acc_rej_indices)
#    reject_indices = compress(equal(acc_rej_indices, -1),
#                                       arange(len(acc_rej_indices)))
    return closest_indices #, accept_indices, reject_indices)


def find(x, v, next_largest=1, indices=None):
    """Returns the index into the 1D array x corresponding to the
    element of x that is either equal to v or the nearest to
    v. x is assumed to contain unique elements.

    if v is outside the range of values in x then the index of the
    smallest or largest element of x is returned.

    If next_largest == 1 then the nearest element taken is the next
    largest, otherwise if next_largest == 0 then the next smallest
    is taken.

    The optional argument indices speeds up multiple calls to this
    function if you have pre-calculated indices=argsort(x).
    """
    if indices is None:
        indices=argsort(x)
    xs=take(x, indices, axis=0)
    assert next_largest in [0,1], "next_largest must be 0 or 1"
    eqmask=(xs==v).tolist()
    try:
        ix = eqmask.index(1)
    except ValueError:
        if next_largest:
            mask=(xs<v).tolist()
        else:
            mask=(xs>v).tolist()
        try:
            ix=min([max([0,mask.index(1-next_largest)+next_largest-1]),len(mask)-1])
        except ValueError:
            ix = 0+next_largest-1
    return indices[ix]


def orderEventData(edict, evnames=None, nonames=False, bytime=False):
    """Time-order event data dictionary items.

    Returns time-ordered list of (eventname, time) tuples.

    If 'evnames' argument included, this restricts output to only the named
      events.
    The 'nonames' flag (default False) forces routine to return only the event
      times, with no associated event names.
    The 'bytime' flag (default False) only works with nonames=False and returns
      the list in (time, eventname) order.
    """

    if evnames is None:
        evnames = list(edict.keys())
    else:
        assert remain(evnames, edict.keys()) == [], "Invalid event names passed"
    # put times as first tuple entry of etuplelist
    if nonames:
        alltlist = []
        for (evname,tlist) in edict.items():
            if evname in evnames:
                alltlist.extend(tlist)
        alltlist.sort()
        return alltlist
    else:
        etuplelist = []
        for (evname,tlist) in edict.items():
            if evname in evnames:
                etuplelist.extend([(t,evname) for t in tlist])
        # sort by times
        etuplelist.sort()
        if bytime:
            return etuplelist
        else:
            # swap back to get event names as first tuple entry
            return [(evname,t) for (t,evname) in etuplelist]

## ------------------------------------------------------------
## Generator wrapping utilities

def make_RHS_wrap(gen, xdict_base, x0_names, use_gen_params=False, overflow_penalty=1e4):
    """Return function wrapping Generator argument gen's RHS function,
    but restricting input and output dimensions to those specified by
    x0_names. All other variable values will be given by those in xdict_base.
    In case of overflow or ValueError during a call to the wrapped function,
    an overflow penalty will be used for the returned values (default 1e4).

    if use_gen_params flag is set (default False)
    then:
      Return function has signature Rhs_wrap(x,t)
      and takes an array or list of x state variable values and scalar t,
      returning an array type of length len(x). The Generator's current param
      values (at call time) will be used.
    else:
      Return function has signature Rhs_wrap(x,t,pdict)
      and takes an array or list of x state variable values, scalar t, and a
      dictionary of parameters for the Generator, returning an array type of
      length len(x).

    NB: xdict_base will be copied as it will be updated in the wrapped
    function."""
    var_ix_map = invertMap(gen.funcspec.vars)
    x0_names.sort()   # ensures sorted
    x0_ixs = [var_ix_map[xname] for xname in x0_names]
    dim = len(x0_names)
    xdict = xdict_base.copy()
    if use_gen_params:
        def Rhs_wrap(x, t):
            xdict.update(dict(zip(x0_names, x)))
            try:
                return take(gen.Rhs(t, xdict, gen.pars), x0_ixs)
            except (OverflowError, ValueError):
                return array([overflow_penalty]*dim)

    else:
        def Rhs_wrap(x, t, pdict):
            xdict.update(dict(zip(x0_names, x)))
            try:
                return take(gen.Rhs(t, xdict, pdict), x0_ixs)
            except (OverflowError, ValueError):
                return array([overflow_penalty]*dim)

    return Rhs_wrap


def make_Jac_wrap(gen, xdict_base, x0_names, use_gen_params=False, overflow_penalty=1e4):
    """Return function wrapping Generator argument gen's Jacobian function,
    but restricting input and output dimensions to those specified by
    x0_names. All other variable values will be given by those in xdict_base.
    In case of overflow or ValueError during a call to the wrapped function,
    an overflow penalty will be used for the returned values (default 1e4).

    if use_gen_params flag is set (default False)
    then:
      Return function Jac_wrap(x,t) takes an array or list of x variable
      values and scalar t, returning a 2D array type of size len(x) by len(x).
      The Generator's current param values (at call time) will be used.
    else:
      Return function Jac_wrap(x,t,pdict) takes an array or list of x variable
      values, scalar t, and a dictionary of parameters for the Generator,
      returning a 2D array type of size len(x) by len(x).

    NB: xdict_base will be copied as it will be updated in the wrapped
    function."""
    if not gen.haveJacobian():
        raise ValueError("Jacobian not defined")
    var_ix_map = invertMap(gen.funcspec.vars)
    x0_names.sort()   # ensures sorted
    x0_ixs = [var_ix_map[xname] for xname in x0_names]
    dim = len(x0_names)
    xdict = xdict_base.copy()
    if use_gen_params:
        def Jac_wrap(x, t):
            xdict.update(dict(zip(x0_names, x)))
            try:
                return take(take(gen.Jacobian(t, xdict, gen.pars), x0_ixs,0), x0_ixs,1)
            except (OverflowError, ValueError):
                return array([overflow_penalty]*dim)
    else:
        def Jac_wrap(x, t, pdict):
            xdict.update(dict(zip(x0_names, x)))
            try:
                return take(take(gen.Jacobian(t, xdict, pdict), x0_ixs,0), x0_ixs,1)
            except (OverflowError, ValueError):
                return array([overflow_penalty]*dim)

    return Jac_wrap


## ------------------------------------------------------------

# User-interaction utilities

def progressBar(i, total, width=50):
    """Print an increasing number of dashes up to given width, reflecting
    i / total fraction of progress. Prints and refreshes on one line.
    """
    percent = float(i)/total
    dots = int(percent*width)
    progress = str('[').ljust(dots+1, '-')
    sys.stdout.write('\r'+progress.ljust(width, ' ')+str('] %.2f%%' % (percent*100.)))
    sys.stdout.flush()


## ------------------------------------------------------------

def saveObjects(objlist, filename, force=False):
    """Store PyDSTool objects to file. Argument should be a tuple or list,
    but if a singleton non-sequence object X is given then it will be
    saved as a list [ X ].

    Some PyDSTool objects will not save using this function, and will complain
    about attributes that do not have definitions in __main__.
    """

    # passing protocol = -1 to pickle means it uses highest available
    # protocol (e.g. binary format)
    if not force:
        if os.path.isfile(filename):
            raise ValueError("File '" + filename + "' already exists")
    pklfile = open(filename, 'wb')
    opt = 0
    if not isinstance(objlist, list):
        objlist=[objlist]
    for obj in objlist:
        try:
            pickle.dump(obj, pklfile, opt)
        except:
            if hasattr(obj, 'name'):
                print("Failed to save '%s'"%obj.name)
            else:
                print("Failed to save object '%s'"%str(obj))
            raise
    pklfile.close()



def loadObjects(filename, namelist=None):
    """Retrieve PyDSTool objects from file. Returns list of objects
    unless namelist option is given as a singleton string name.
    Also, if only one object X was stored, it will be returned as [X],
    and thus you will have to index the returned list with 0 to get X itself.

    Optional namelist argument selects objects to return by name,
    provided that the objects have name fields (otherwise they are ignored).
    If namelist is a single string name then a single object is returned.
    """

    # Since names are not intended to be unique in PyDSTool, the while
    # loop always goes to the end of the file, and pulls out *all*
    # occurrences of the names.
    if not os.path.isfile(filename):
        raise ValueError("File '" + filename + "' not found")
    if namelist is None:
        namelist = []
    was_singleton_name = isinstance(namelist, str)
    if not isinstance(namelist, list):
        if was_singleton_name:
            namelist = [copy.copy(namelist)]
        else:
            raise TypeError("namelist must be list of strings or singleton string")
    if not isUniqueSeq(namelist):
        raise ValueError("Names must only appear once in namelist argument")
    pklfile = open(filename, 'rb')
    if namelist == []:
        getall = True
    else:
        getall = False
    objlist = []
    notDone = True
    while notDone:
        try:
            if getall:
                objlist.append(pickle.load(pklfile))
            else:
                tempobj = pickle.load(pklfile)
                if hasattr(tempobj, 'name'):
                    if tempobj.name in namelist:
                        objlist.append(tempobj)
        except EOFError:
            notDone = False
        except:
            print("Error in un-pickling %s:"%filename)
            print("Was the object created with an old version of PyDSTool?")
            pklfile.close()
            raise
    pklfile.close()
    if objlist == []:
        if getall:
            print("No objects found in file")
        else:
            print("No named objects found in file")
    if was_singleton_name:
        return objlist[0]
    else:
        return objlist


def intersect(a, b):
    """Find intersection of two lists, sequences, etc.
    Returns a list that includes repetitions if they occur in the inputs."""
    return [e for e in a if e in b]

def union(a, b):
    """Find union of two lists, sequences, etc.
    Returns a list that includes repetitions if they occur in the input lists.
    """
    return list(a)+list(b)

def remain(a, b):
    """Find remainder of two lists, sequences, etc., after intersection.
    Returns a list that includes repetitions if they occur in the inputs."""
    return [e for e in a if e not in b]

def compareList(a, b):
    """Compare elements of lists, ignoring order (like sets)."""
    return len(intersect(a,b))==len(a)==len(b)

def cartesianProduct(a, b):
    """Returns the cartesian product of the sequences."""
    ret = []
    for i in a:
        ret.extend([(i, j) for j in b])
    return ret

def arclength(pts):
    """
    Return array of L2 arclength progress along parameterized pointset
    in all the dimensions of the pointset
    """
    x0 = pts[0]
    arclength = np.zeros(len(pts))
    for i, x in enumerate(pts[1:]):
        arclength[i+1] = np.linalg.norm(x - pts[i]) + arclength[i]
    return arclength


# ------------------------

def distutil_destination():
    """Internal utility that makes the goofy destination directory string so that PyDSTool
    can find where the distutils fortran/gcc compilers put things.

    If your temp directory turns out to be different to the one created here, contact us
    on sourceforge.net, but in the meantime you can override destdir with whatever directory
    name you find that is being used.
    """
    import scipy
    osname = str.lower(platform.system())
    pyname = platform.python_version_tuple()
    machinename = platform.machine()
    if osname == 'linux':
        destdir = 'src.'+osname+'-'+machinename+'-'+pyname[0] + '.' + pyname[1]
    elif osname in ['darwin', 'freebsd']:
        # use the same version string as numpy.distutils.core.setup used by ContClass.CompileAutoLib
        osver = get_platform()
        destdir = 'src.' + osver + '-' +pyname[0] + '.' + pyname[1]
    elif osname == 'windows':
        destdir = 'src.win32-'+pyname[0]+'.'+pyname[1]
    else:
        destdir = ''
    # TEMP for debugging
    #import os
    #os.system('echo %s > temp_dist.txt' % (os.path.abspath('.') + " : " + destdir))
    return destdir


def architecture():
    """
    Platform- and version-independent function to determine 32- or 64-bit architecture.
    Used primarily to determine need for "-m32" option to C compilers for external library
    compilation, e.g. by AUTO, Dopri, Radau.

    Returns integer 32 or 64.
    """
    import struct
    return struct.calcsize("P") * 8

def extra_arch_arg(arglist):
    """
    Adds '-m32' flag to existing list of extra compiler/linker flags passed
    as argument, based on whether architecture is detected as 32 bit. Otherwise,
    it performs the identity function.
    """
    if architecture() == 32:
        return arglist + ['-m32']
    else:
        return arglist


def get_lib_extension():
    return misc_util.get_shared_lib_extension()
