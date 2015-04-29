"""Symbolic expression support, and associated utilities.

Robert Clewley, October 2005.

   Includes code for symbolic differentiation by Pearu Peterson
   and Ryan Gutenkunst.


 Symbolic Differentiation code:
    Copyright 1999 Pearu Peterson, <pearu@ioc.ee>
    March 11-12, 1999
    Pearu Peterson

    Modified by Ryan Gutenkunst, <rng7@cornell.edu>
    April 28th, 2005
    General clean-up of code.
    Also changed saving mechanism to explicitly save the results,
    rather than doing it each time a derivative is taken.

    Modified by Robert Clewley, <rhc28@cornell.edu>
    September 15th, 2005
    As this branch of Peterson's development of symbolic features in Python
      seems to be stopped, I have increased the version number to 0.3 from 0.2.
    Added division by zero check in function dodiv().
    Added decimal point to integer constants in division to avoid Python
      integer division rules being applied to the expressions. This involves
      an added optional argument to trysimple(), and a new function
      ensuredecimalconst(). This option is conrolled using the global DO_DEC
      switch.
    Added functionality in dofun() for using "pow(x,y)" (with two arguments only)
      in input expressions.
    Also, added DO_POW option to make all output involving powers to use this
      "pow(x,y)" syntax, rather than "x**y". (Set the global DO_POW switch).
    dopower(), domul(), etc. also accept decimals 1.0 and 0.0 to perform
      simplifications if the DO_DEC switch is set.
    Adapted Diff function to accept PyDSTool Var and QuantSpec types, and
      always return those types, and also to accept an array (acts like Jacobian).
      Code is entirely backwards compatible if PyDSTool is not used (except
      Jacobians won't work).
    Renamed original, pure-string Diff function to DiffStr.
    test3() augmented with test for "pow(x,y)" syntax.
    Added test4() for Diff as Jacobian and using Var and QuantSpec objects,
      if available.
    Note that with DO_DEC option off you get "incorrect" results for the derivative
      of f[1] in test4().

    25 February 2006
    Moved package into Symbolic.py
    Added scopes to eval statements to prevent argument names clashing with
      names appearing in the eval'd string.
    Made title case symbolic versions of math functions and constants
      from ModelSpec compatible with Diff.
    Improved support for pow(x,y) syntax under differentiation.
    Added support for symbolic vectors.
"""

# ----------------------------------------------------------------------------

from __future__ import division, absolute_import, print_function
import os, sys, types
import six
from six.moves import cPickle
from .Interval import Interval
from .common import *
from .utils import info as utils_info
from .parseUtils import *
from .errors import *

#from math import *
from .utils import *
from numpy import array, Inf, NaN, isfinite, mod, sum, float64, int32
from numpy import sometrue, alltrue
# replacements of math functions so that expr2fun etc. produce vectorizable math functions
from numpy import arccos, arcsin, arctan, arctan2, arccosh, arcsinh, arctanh, \
     ceil, cos, cosh, exp, fabs, floor, fmod, frexp, hypot, ldexp, log, log10, \
     modf, power, sin, sinh, sqrt, tan, tanh
# for compatibility with numpy 1.0.X
from math import degrees, radians
# constants
from math import pi, e

from copy import copy, deepcopy
import math, random, numpy, scipy, scipy.special

# alternative names for inverse trig functions, supported by expr2fun, symbolic eval, etc.
asin = arcsin
acos = arccos
atan = arctan
atan2 = arctan2
acosh = arccosh
asinh = arcsinh
atanh = arctanh
pow = power

# ---------------------------------------------------------------------------
### Constants
# determine what "globals" will be for math evaluations in _eval method
math_dir = dir(math)
math_globals = dict(zip(math_dir,[getattr(math, m) for m in math_dir]))
math_globals['Inf'] = Inf
math_globals['NaN'] = NaN

# protected (function) names come from parseUtils.py
# (constants are all-caps and are filtered out)
# Adds scipy special function names *without* 'special_' prefix
# but does not include logical infix operators from builtins
allmathnames = [a for a in protected_mathnames + protected_randomnames + \
                protected_scipynames + protected_numpynames + \
                scipy_specialfns if not a.isupper()] + \
                ['abs', 'pow', 'min', 'max', 'sum']
allmathnames_symbolic = [a.title() for a in allmathnames]

# the definitions for the math names are made lower down in this file
mathNameMap = dict(zip(allmathnames_symbolic, allmathnames))
inverseMathNameMap = dict(zip(allmathnames, allmathnames_symbolic))

specTypes = ('RHSfuncSpec', 'ImpFuncSpec', 'ExpFuncSpec')


# --------------------------------------------------------------------------
### Exports
_functions = ['isMultiDef', 'isMultiRef', 'isMultiDefClash', 'checkbraces',
              'findMultiRefs', 'processMultiRef', 'evalMultiRefToken',
              'expr2fun', 'subs', 'ensureStrArgDict', 'ensureQlist',
##              'loadDiffs', 'saveDiffs',
              'Diff', 'DiffStr', 'prepJacobian']

_classes = ['QuantSpec', 'Quantity', 'Var', 'Par', 'Input', 'Fun']

_constants = ['specTypes', 'allmathnames_symbolic', 'allmathnames'] \
                + allmathnames_symbolic + ['mathNameMap', 'inverseMathNameMap']

__all__ = _functions + _classes + _constants


# -----------------------------------------------------------------------------
# create namemap for title-case to lower case for evaluation
# to float attempt
feval_map_const = {'e': repr(e), 'pi': repr(pi)}
feval_map_symb = {}
for symb in allmathnames_symbolic:
    feval_map_symb[symb] = symb.lower()


# -----------------------------------------------------------------------------
## Enable mathematical functions to be used like QuantSpecs when building
## model specs, by substituting overloaded objects with the same names in
## their place.

mathlookup = {}.fromkeys(protected_mathnames, 'math.')
randomlookup = {}.fromkeys(protected_randomnames, 'random.')
builtinlookup = {'abs': '', 'pow': '', 'max': '', 'min': '', 'sum': ''}
numpylookup = {}.fromkeys(protected_numpynames, 'numpy.')
scipylookup = {}.fromkeys(protected_scipynames, 'scipy.')
scipylookup.update( {}.fromkeys(scipy_specialfns, 'scipy.special.') )
modlookup = {}
modlookup.update(mathlookup)
modlookup.update(randomlookup)
modlookup.update(builtinlookup)
modlookup.update(scipylookup)
modlookup.update(numpylookup)

# only rename actual functions (or callable objects)

# DEBUG VERSION
#funcnames = []
#for n in allmathnames:
#    try:
#        f = eval(modlookup[n]+n)
#    except:
#        print("Failed on", n)
#        raise
#    if hasattr(f, "__call__"):
#        print(n)
#        funcnames.append(n)

# REGULAR VERSION
funcnames = [n for n in allmathnames if hasattr(eval(modlookup[n]+n),
                                                "__call__")]

# wrapper and overloader class
class _mathobj(object):
    def __init__(self, name):
        self.name = name.title()
        if name not in funcnames:
            raise ValueError("This class is intended for mathematical "
                             "functions only")

    def __call__(self, *argtuple):
        args = list(argtuple)
        isQuantity = [compareClassAndBases(a,Quantity) for a in args]
        isQuantSpec = [compareClassAndBases(a,QuantSpec) for a in args]
        if len(args) == 0:
            return QuantSpec("__result__", self.name)
        elif sometrue(isQuantity+isQuantSpec):
            if not sometrue(isQuantity):
                # can only make this evaluate to a number if args is only
                # made up of numbers and QuantSpecs that are in allmathnames.
                convert = True   # initial value
                for i in range(len(args)):
                    if isQuantSpec[i]:
                        if args[i].subjectToken in allmathnames:
                            # convert into original math name
                            args[i] = eval(args[i].subjectToken.lower())
                        else:
                            # if any one argument or used symbol is a
                            # non-mathname QS then no conversion -- return a QS
                            symbs = args[i].usedSymbols
                            allnonnumbers = [n for n in symbs if isNameToken(n)]
                            if len(allnonnumbers) > 0 \
                               and alltrue([n in allmathnames for n in allnonnumbers]):
                                namemap = dict(zip(allnonnumbers,
                                               [a.lower() for a in allnonnumbers]))
                                args[i].mapNames(namemap)
                                args[i] = eval(args[i]())
                            else:
                                convert = False
                                break
            else:
                convert = False
            argstr = ", ".join(map(str,args))
            if convert:
                try:
                    return eval(self.name.lower())(*args)
                except TypeError:
                    print("Failed to evaluate function %s on args:"%self.name.lower() + argstr)
                    raise
            else:
                return QuantSpec("__result__", self.name + "(" + argstr +")")
        else:
            arglist = []
            for a in args:
                if isinstance(a, str):
                    arglist.append(float(a))
                else:
                    arglist.append(a)
            try:
                return float(eval(self.name.lower())(*arglist))
            except NameError:
                argstr = ", ".join(map(str,args))
                return QuantSpec("__result__", self.name + "(" + argstr +")")
            except TypeError:
                print("Error evaluating %s on args:"%self.name.lower() + args)
                raise

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name + " (ModelSpec wrapper)"


# now assign all the function names to _mathobj versions
for f in funcnames:
    if f == 'random':
        # skip this because it's also a module name and it messes around
        # with the namespace too much
        continue
    six.exec_(str(f).title() + " = _mathobj('"+f+"')")
    math_globals[str(f).title()] = eval(str(f).title())
# assignment of constants' names continues after QuantSpec is defined

# -------------------------------------------------------------------------

def resolveSpecTypeCombos(t1, t2):
    """Resolve Quantity and QuantSpec types when combined.
     QuantSpec <op> QuantSpec -> specType rules:
     same types -> that type
     ExpFuncSpec o RHSfuncSpec -> RHSfuncSpec
     ImpFuncSpec o RHSfuncSpec -> RHSfuncSpec (should this be an error?)
     ImpFuncSpec o ExpFuncSpec -> ImpFuncSpec"""
    if t1=='RHSfuncSpec' or t2=='RHSfuncSpec':
        return 'RHSfuncSpec'
    elif t1=='ImpFuncSpec' or t2=='ImpFuncSpec':
        return 'ImpFuncSpec'
    else:
        return 'ExpFuncSpec'


def getdim(s):
    """If s is a representation of a vector, return the dimension."""
    if len(s) > 4 and s[0]=='[' and s[-1]==']':
        return len(splitargs(s[1:-1], ['(','['], [')',']']))
    else:
        return 0


def strIfSeq(x):
    """Convert sequence of QuantSpecs or strings to a string representation of
    that sequence, otherwise return the argument untouched."""
    if isinstance(x, _seq_types):
        if len(x) > 0:
            xstrlist = list(map(str,x))
            return '['+",".join(xstrlist)+']'
        else:
            return x
    else:
        return x


def ensureQlist(thelist):
    o = []
    for s in thelist:
        if isinstance(s, str):
            o.append(QuantSpec('__dummy__', s))
        elif isinstance(s, QuantSpec) or isinstance(s, Quantity):
            o.append(s)
        else:
            raise TypeError("Invalid type for symbolic quantities list")
    return o


def ensureStrArgDict(d, renderForCode=True):
    """Optional renderForCode argument converts any uppercase built-in
    symbolic function names to lowercase numeric function names
    (default True)"""
    o = {}
    try:
        # d is a Point or dictionary
        for k, v in d.items():
            if isinstance(v, tuple):
                if alltrue([isinstance(x, str) for x in v[0]]):
                    o[str(k)] = (v[0], str(v[1]))
                else:
                    o[str(k)] = ([str(x) for x in v[0]], str(v[1]))
            else:
                o[str(k)] = str(v)
    except AttributeError:
        # d is not a Point or Dictionary
        if isinstance(d, list):
            for q in d:
                o.update(ensureStrArgDict(q))
        else:
            try:
                if hasattr(d, 'signature'):
                    # q is a Fun Quantity
                    if renderForCode:
                        o[d.name] = (d.signature, str(d.spec.renderForCode()))
                    else:
                        o[d.name] = (d.signature, d.spec.specStr)
                elif hasattr(d, 'name'):
                    # d is any other type of Quantity
                    if renderForCode:
                        o[d.name] = str(d.spec.renderForCode())
                    else:
                        o[d.name] = d.spec.specStr
                elif hasattr(d, 'subjectToken'):
                    # d is a QuantSpec
                    if renderForCode:
                        o[d.subjectToken] = str(d.renderForCode())
                    else:
                        o[d.subjectToken] = d.specStr
                else:
                    raise TypeError("Invalid type %s"%str(type(d)))
            except TypeError as err:
                print("Argument was:%s\n" % d)
                raise TypeError("Invalid argument type: %s "%str(err) + \
                   "or argument's values cannot be converted to key:value" + \
                                "strings")
    return o


def _eval(q, math_globals, local_free, eval_at_runtime):
    if isinstance(q, str):
        qstr = q
    else:
        qstr = str(q)
    try:
        val = eval(qstr, math_globals, local_free)
    except NameError as err:
        raise ValueError("Invalid input.\nProblem was: %s"%str(err))
    except TypeError as err:
        # trap getitem or call error for string version of object
        # try without runtime components, if any
        if eval_at_runtime != []:
            try:
                val = eval(qstr, math_globals,
                           filteredDict(local_free, eval_at_runtime, neg=True))
            except (NameError, TypeError):
                val = q
        else:
            val = q
    except KeyboardInterrupt:
        raise
    except:
        print("Internal error evaluating function spec")
        info(local_free, " ** Local free dict was")
        info(math_globals, " ** Math globals dict was")
        raise
    return val


def _propagate_str_eval(s):
    if isinstance(s, list):
        return "["+",".join([_propagate_str_eval(s_el) for s_el in s])+"]"
    elif isinstance(s, tuple):
        return "("+",".join([_propagate_str_eval(s_el) for s_el in s])+")"
    elif isinstance(s, str):
        return s
    else:
        raise TypeError("Don't know how to evaluate to string: %s"%(str(s)))


def _join_sublist(qspec, math_globals, local_free, eval_at_runtime):
    if qspec.isvector():
        sl = qspec.fromvector()
        # sl is guaranteed to be seq of QuantSpecs
        return "["+",".join([str(_eval(q, math_globals, local_free,
                        eval_at_runtime)) for q in sl])+"]"
    else:
        return str(_eval(qspec, math_globals, local_free, eval_at_runtime))


def expr2fun(qexpr, ensure_args=None, ensure_dynamic=None, fn_name='',
             for_funcspec=True, **values):
    """qexpr is a string, Quantity, or QuantSpec object.
    values is a dictionary binding free names to numerical values
    or other objects defined at runtime that are in local scope.

    If values contains pairs (fnsig, fndef) from a FuncSpec function
    definition then the function will be incorporated into the result.
    The function definition can contain math names that are still
    in non-code-rendered (lowercase) form (i.e. "Exp" rather than "exp").

    If any names use the hierarchical naming convention (using the '.'
    separator) then those names will be converted to use the '_' separator
    instead. For end-use convenience, the necessary name map for arguments
    is stored in the '_namemap' attribute, which will otherwise be the ID map.
    Calling with keyed objects that use hierarchical names (especially Point
    objects) can then be done using the return object's alt_call method,
    which takes any keyed_object that will be filtered for the function
    arguments and name-mapped appropriately before evaluation.

    You can force the function to accept the named arguments in the
    ensure_args list of strings, e.g. when creating a Jacobian
    that is expected to always take 't' as an argument even if
    the system is autonomous.

    You can force the function to expect a dictionary of dynamically
    accessible numeric values that can be looked up during function
    evaluation, using the optional ensure_dynamic dictionary. Later,
    you can modify the values in this dictionary directly.

    You can name the function explicitly using fn_name option. This will
    become the __name__ attribute of the generated class.

    You can force the processing to be for non-FuncSpec output (e.g. to
    support full Python syntax) by setting for_funcspec = False. Default
    is True.

    The arguments to the returned callable function object are
    given by its attribute '_args', and its definition string by
    '_call_spec'.
    """
    # put in local namespace for sake of final class method's
    # alt_call use of filteredDict
    from .common import filteredDict
    # convert values into a dictionary of strings where possible
    valDict = {}
    # functions that will end up as methods in the wrapper
    embed_funcs = {}
    local_free = {}
    local_funcs = {}  # temp Fun for eval'ing strings
    eval_at_runtime = []
    free = []
    h_map = symbolMapClass()
    for k, v in values.items():
        if isinstance(v, str):
            vs = v
        elif isinstance(v, _num_types):
            # retain most accuracy in float
            vs = repr(v)
        elif isinstance(v, tuple):
            # ensure it's a valid function definition
            try:
                fnsig, fndef = v
            except ValueError:
                raise ValueError("Only pass pairs in 'values' argument that are function definitions")
            try:
                vs = Fun(fndef, fnsig, k)
            except:
                raise ValueError("Only pass pairs in 'values' argument that are valid function definitions")
            local_funcs[k] = Var(k)  # placeholder
            eval_at_runtime.append(k)
            for vfree in vs.freeSymbols:
                if vfree not in values and vfree not in math_globals and vfree not in ensure_dynamic:
                    raise ValueError("Free name %s inside function %s must be bound" % (vfree, k))
            embed_funcs[k] = (fnsig, fndef)
        else:
            vs = str(v)
        if isNumericToken(vs):
            if '.' in k and for_funcspec:
                kFS = k.replace('.','_')
                h_map.update({k: kFS})
            else:
                kFS = k
            valDict[kFS] = vs
        elif not isinstance(vs, Fun):
            if '.' in k and for_funcspec:
                kFS = k.replace('.','_')
                h_map.update({k: kFS})
            else:
                kFS = k
            local_free[kFS] = v
            eval_at_runtime.append(kFS)
    # ensure QuantSpec
    if isinstance(qexpr, Quantity):
        qexpr = qexpr.spec
    # now make a Var from the QuantSpec
    try:
        qvar = Var(qexpr, 'q')
    except TypeError as e:
        raise TypeError("Invalid qexpr argument type: " + str(e))
    # eval will fail if qvar contains python syntax outside of
    # math expressions, e.g. for auto-code gen involving dictionary
    # mapping lookups. So, at least only eval if valDict is non-empty
    if len(valDict) > 0:
        fspec = qvar.eval(**valDict)
    else:
        fspec = qvar.spec

    dyn_map = symbolMapClass()
    temp_dynamic = {}
    if ensure_dynamic is not None:
        dyn_keys = list(ensure_dynamic.keys())
        # _pardict will only be used if for FuncSpec output
        if for_funcspec:
            dyn_vals = ["self._pardict['%s']" %d for d in dyn_keys]
            dyn_map.update(dict(zip(dyn_keys, dyn_vals)))
        eval_at_runtime.extend(dyn_keys)
        for k in dyn_keys:
            temp_dynamic[k] = Var(k)
    else:
        dyn_keys = []

    for fname, (fnsig, fndef) in embed_funcs.items():
        qtemp = QuantSpec('q', fndef)
        for emb_fname in embed_funcs.keys():
            # check for embedded function names in these definitions that need
            # prefix 'self.' for later function creation
            if emb_fname in qtemp:
                # replace all occurrences
                qtemp.mapNames({emb_fname: 'self.'+emb_fname})
            qtemp.mapNames(dyn_map)
        new_fndef = str(qtemp.eval(**filteredDict(valDict, fnsig, neg=True)).renderForCode())
        embed_funcs[fname] = (fnsig, new_fndef)

    free.extend(remain(fspec.freeSymbols,
                  list(math_globals.keys())+list(local_free.keys())+list(embed_funcs.keys())+dyn_keys))
    # hack to exclude object references which will be taken care of at runtime
    # or string literals that aren't supposed to be free symbols
    free = [sym for sym in free if not \
            ("'%s'"%sym in qexpr or '"%s"'%sym in qexpr)]
    mapnames = list(local_free.keys()) + list(embed_funcs.keys())
    for k in local_free.keys():
        for sym in free:
            if '.' in sym and sym[:sym.index('.')] == k:
                mapnames.append(sym)
                free.remove(sym)
    # eval simplifies the arithmetic where possible, and defining
    # local QuantSpecs to correspond to the free names allows eval to return
    # a new QuantSpec.
    #
    # first, get rid of hierarchical names before running eval, which will think
    # they are python objects!
    if for_funcspec:
        # These statements assume that no underscore-leading names are used, e.g. 'a._b'
        assert '._' not in str(fspec)
        # enforce unique in case e.g. 'K.n' and 'K_n' were both separately in free symbols
        # otherwise end up with duplicates of 'K_n' in argument list later
        free_h = makeSeqUnique([symb.replace('.', '_') for symb in free])
        # Don't want to replace '_' in free names if not part of hierarchical name
        # e.g. if it's a module name for non-funcspec use
        free_h_nonFS = makeSeqUnique([symb.replace('_', '.') for symb in free])
        h_map.update(dict(zip(free_h_nonFS, free_h)))
        free = free_h
    defs = filteredDict(local_free, [k for k, v in local_free.items() if v is not None])
    local_free.update(dict(zip(free, [QuantSpec(symb) for symb in free])))
    scope = filteredDict(local_free, [k for k, v in local_free.items() if v is not None])
    scope.update(local_funcs)
    scope.update(temp_dynamic)
    fspec.mapNames(h_map)
    eval_globals = math_globals.copy()
    eval_globals['max'] = QuantSpec('max', '__temp_max__')
    eval_globals['min'] = QuantSpec('min', '__temp_min__')
    eval_globals['sum'] = QuantSpec('sum', '__temp_sum__')
    if fspec.isvector():
        # e.g. for a Jacobian or other matrix-valued function
        # expect a list of lists of Quantities (i.e., rank 2 only)
        temp_str = "[" + ",".join([_join_sublist(q, eval_globals,
                          scope, eval_at_runtime) \
                                   for q in fspec.fromvector()]) + "]"
        fspec_eval = QuantSpec('__temp__', temp_str).renderForCode()
        if eval_at_runtime != []:
            fspec_eval.mapNames(dict(zip(mapnames,
                                    ['self.'+k for k in mapnames])))
            fspec_eval.mapNames(dyn_map)
        fspec_str = str(fspec_eval)
    else:
        if for_funcspec:
            # BUG? Why is dyn_map not used here when eval_at_runtime != []?
            fspec_eval = _eval(fspec, eval_globals,
                           scope, eval_at_runtime)
        else:
            # no need to try to simplify
            fspec_eval = fspec

    def append_call_parens(k):
        """For defined references to future fn_wrapper attributes that need
        calling to evaluate, append the call parentheses for the text that
        will be embedded in the fn_wrapper definition."""
        try:
            if defs[k]._args == []:
                return '()'
            else:
                return ''
        except (AttributeError, KeyError):
            return ''
    if isinstance(fspec_eval, _num_types):
        if eval_at_runtime != []:
            fspec.mapNames(dict(zip(mapnames,
                        ['self.'+k+append_call_parens(k) for k in mapnames])))
            fspec_str = str(fspec)
        else:
            fspec_str = repr(fspec_eval)
    elif not fspec.isvector():
        # already found fspec_str using eval_at_runtime for isvector case
        if eval_at_runtime != []:
            fspec.mapNames(dict(zip(mapnames,
                        ['self.'+k+append_call_parens(k) for k in mapnames])))
            # replace any occurrences of eval-at-runtime parameters x
            # with 'self._pardict[x]'
            fspec.mapNames(dyn_map)
            fspec_str = str(fspec)
        else:
            if for_funcspec:
                fspec_str = str(fspec_eval.renderForCode())
            else:
                # fspec may be using indexing in special use cases (general python
                # syntax such as dictionary indexing for non-core PyDSTool parsing)
                fspec_str = str(fspec_eval)
    arglist = copy(free)
    if ensure_args is not None:
        arglist.extend(remain(ensure_args, free))
    if len(arglist) > 0:
        arglist_str = ', ' + ", ".join(arglist)
    else:
        arglist_str = ''
        # only include if the 'function' has no arguments and can
        # be treated as a dynamic-valued variable
    # now that exec has to be guarded with locals(), etc must add these explicit
    # imports as per header of Symbolic.py
    my_locals = locals()
    my_locals.update(math_globals)
    if fn_name == '' or fn_name is None:
        # make a random, unique name
        import time
        fn_name = 'fn_' + str(time.time()).replace('.', '_')
    # !!! PERFORM MAGIC
    def_str = """
from __future__ import division
class """+fn_name+"""_fn_wrapper(object):
    __name__ = '""" + fn_name + """'
    def __call__(self""" + arglist_str + """):
        return """ + fspec_str + """

    def alt_call(self, keyed_arg):
        return self.__call__(**filteredDict(self._namemap(keyed_arg), self._args))
    """
    if len(embed_funcs) > 0:
        for fname, (fnsig, fndef) in embed_funcs.items():
            if len(fnsig) > 0:
                embed_sig_str = ", " + ", ".join(fnsig)
            else:
                embed_sig_str = ""
            def_str += "\n    def " + fname + "(self" + embed_sig_str + \
                       "):\n        return " + fndef
    try:
        six.exec_(def_str)
    except:
        print("Problem defining function:")
        raise
    cls = locals()[fn_name+'_fn_wrapper']
    evalfunc = cls()
    evalfunc.__dict__.update(defs)
    if dyn_keys != [] and for_funcspec:
        # Don't make copy of the dict to allow automatic update.
        # We don't need _pardict for non-funcspec definitions.
        evalfunc._pardict = ensure_dynamic
    evalfunc._args = arglist
    evalfunc._call_spec = fspec_str
    evalfunc._namemap = h_map
    # register the class in the global namespace to assist in pickling
    g = globals()
    g[cls.__name__] = cls
    return evalfunc

def nonworking_call_to_metaclass():
    # test call to new metaclass for expr2fun
    kwargs = {'fn_name': fn_name,
              'arglist_str': arglist_str,
              'fspec_str': fspec_str}
    if len(embed_funcs) > 0:
        embed_func_strs = {}
        for fname, (fnsig, fndef) in embed_funcs.items():
            if len(fnsig) > 0:
                embed_sig_str = ", " + ", ".join(fnsig)
            else:
                embed_sig_str = ""
            embed_func_strs[fname] = (embed_sig_str, fndef)
        kwargs["embed_funcs"] = embed_func_strs
    kwargs['arglist'] = arglist
    kwargs['_call_spec'] = fspec_str
    kwargs['_namemap'] = h_map
    kwargs['defs'] = defs
    if dyn_keys != [] and for_funcspec:
        # Don't make copy of the dict to allow automatic update.
        # We don't need _pardict for non-funcspec definitions.
        kwargs["_pardict"] = ensure_dynamic

    #class Y(object):
    #    __metaclass__ = fn_wrapper
    return fn_wrapper(kwargs) #type(fn_name, (Y,), kwargs)

# Not working
class fn_wrapper(type):
    def __init__(cls, kwargs):
        call_str = """def _call(self%s):
            print "In call"
            return %s
            """ % (kwargs["arglist_str"], kwargs["fspec_str"])
        alt_call_str = """def _alt_call(self, keyed_arg):
            return self.__call__(**filteredDict(self._namemap(keyed_arg), self._args))

        """
        print("__init__ of fn_wrapper")
        six.exec_(call_str)
        six.exec_(alt_call_str)
        setattr(cls, "__call__", types.MethodType(locals()['_call'], cls))
        setattr(cls, "alt_call", types.MethodType(locals()['_alt_call'], cls))
        setattr(cls, "__name__", kwargs["fn_name"])
        setattr(cls, "_namemap", kwargs["_namemap"])
        setattr(cls, "_args", kwargs["arglist"])
        setattr(cls, "_call_spec", kwargs["fspec_str"])
        if "embed_funcs" in kwargs:
            for embed_fname, (embed_sig_str, embed_fdef_str) in kwargs["embed_funcs"].items():
                ef_str = """def %s(self%s):
                return %s
                """ % (embed_fname, embed_sig_str, embed_fdef_str)
                six.exec_(ef_str)
                ef = locals()[embed_fname]
                setattr(cls, embed_fname, types.MethodType(ef, cls))
        if "defs" in kwargs:
            for def_key, def_val in kwargs["defs"].items():
                setattr(cls, def_key, def_val)


def old_expr2fun_final():
    # !!! PERFORM MAGIC
    def_str = """
from __future__ import division
class """+fn_name+"""_fn_wrapper(object):
    __name__ = '""" + fn_name + """'
    def __call__(self""" + arglist_str + """):
        return """ + fspec_str + """

    def alt_call(self, keyed_arg):
        return self.__call__(**filteredDict(self._namemap(keyed_arg), self._args))
"""
    if len(embed_funcs) > 0:
        for fname, (fnsig, fndef) in embed_funcs.items():
            if len(fnsig) > 0:
                embed_sig_str = ", " + ", ".join(fnsig)
            else:
                embed_sig_str = ""
            def_str += "\n    def " + fname + "(self" + embed_sig_str + \
                       "):\n        return " + fndef
    try:
        six.exec_(def_str)
    except:
        print("Problem defining function:")
        raise
    evalfunc = locals()['fn_wrapper']()
    evalfunc.__dict__.update(defs)
    if dyn_keys != [] and for_funcspec:
        # Don't make copy of the dict to allow automatic update.
        # We don't need _pardict for non-funcspec definitions.
        evalfunc._pardict = ensure_dynamic
    evalfunc._args = arglist
    evalfunc._call_spec = fspec_str
    evalfunc._namemap = h_map
    return evalfunc



def subs(qexpr, *bindings):
    """Substitute pars or other name bindings (as a dictionary)
    into a Quantity, QuantSpec, or string containing a symbolic expression."""
    valDict = {}
    qstype = qexpr.specType
    isQuant = compareBaseClass(qexpr, Quantity)
    if isQuant:
        qtemp = qexpr()
        varname = qexpr.name
    else:
        qtemp = copy(qexpr)
        varname = ""
    for b in bindings:
        if isinstance(b, (list, tuple)):
            # in case called from a function when the bindings
            # cannot be unravelled in the call arguments
            for bentry in b:
                qtemp = subs(qtemp, bentry)
        elif isinstance(b, Par):
            valDict[str(b)] = str(b.eval())
        elif isinstance(b, dict):
            for k, v in b.items():
                if k in valDict:
                    raise ValueError("Value for %s already bound"%k)
                valDict[k] = str(v)
        else:
            raise TypeError("Invalid binding type '%s' for substitution"%type(b))
    tempvar = Var(qtemp, '__result__')
    tempresult = tempvar.eval(**valDict)
    if isQuant:
        result = Var(tempresult, varname)
    else:
        result = tempresult
    result.specType = qstype
    return result


def _parse_func_deriv(fname):
    """Internal utility to test whether a function name is generated
    from a symbolic differentiation call.
    """
    if '_' in fname:
        # find position from end (in case more than one)
        neg_ix = fname[::-1].index('_')
        # expect integer after _
        orig_name = fname[:-neg_ix-1]
        int_part = fname[-neg_ix:]
        if len(orig_name) > 0 and len(int_part) > 0 and \
           isIntegerToken(int_part):
            return (orig_name, int(int_part))
        else:
            return (fname, None)
    else:
        return (fname, None)

def _generate_subderivatives(symbols, fnspecs):
    """Internal utility for prepJacobian"""
    add_fnspecs = {}
    new_free = []
    math_globals_keys = list(math_globals.keys())
    for symb in symbols:
        # derivatives of an unknown function f(x,y,z) will be marked
        # f_0, f_1, or f_2 depending on which variable was selected
        orig_name, pos = _parse_func_deriv(symb)
        if pos is not None:
            # then it was a legitimate derivative of function
            try:
                fnsig, fndef = fnspecs[orig_name]
            except (KeyError, TypeError):
                raise KeyError("Function %s not present in fnspecs" % orig_name)
            try:
                var = fnsig[pos]
            except KeyError:
                # position was not legitimate for this function
                raise ValueError("Invalid position in signature of function %s" % orig_name)
            Df = Diff(fndef, var)
            add_fnspecs[symb] = (fnsig, str(Df))
            new_free.extend(remain(Df.freeSymbols, new_free+math_globals_keys))
    # if any auxiliary functions showed up as free names, add them to add_fnspecs
    for f in new_free:
        if f in fnspecs and f not in add_fnspecs:
            add_fnspecs[f] = fnspecs[f]
    return new_free, add_fnspecs


def prepJacobian(varspecs, coords, fnspecs=None, max_iter_depth=20):
    """Returns a symbolic Jacobian Var and updated function specs to support
    its definition from variable specifications. Only makes the Jacobian
    with respect to the named coordinates, which will be sorted into
    alphabetical order. The *varspecs* specifications can be a dictionary
    or a Symbolic Fun(ction) object of the coordinate variables given by
    *coords* (and additional variables that will be ignored). The function
    may have one additional and first argument for 't' if the system is
    non-autonomous.

    Coordinates will be rearranged into alphabetically sorted order.
    """
    coords = list(coords) # in case of tuple or other immutable sequence
    coords.sort()
    if isinstance(varspecs, Fun):
        # either dim == number of coords or one fewer if 't' is included in
        # the function signature, which should be the last argument anyway
        D = varspecs.dim
        siglen = len(varspecs.signature)
        if siglen in (D, D + 1):
            if D == len(coords):
                # ensure coords and relevant signature elements are the same
                if siglen == D+1:
                    offset = 1
                else:
                    offset = 0
                coords_sigorder = varspecs.signature[offset:]
                try:
                    ix_map = dict(zip(coords,[coords_sigorder.index(c) for c in coords]))
                except:
                    raise ValueError("Incompatible coords with function sig")
                coords_sigorder.sort()
                if coords_sigorder != coords:
                    raise ValueError("Incompatible coords with function sig")
                else:
                    new_varspecs = {}
                    for co in coords:
                        new_varspecs[co] = varspecs.fromvector(ix_map[co])
            elif D > len(coords):
                # only a subset were chosen
                # ensure each coord given is in signature
                new_varspecs = {}
                for co in coords:
                    try:
                        ix = varspecs.signature.index(co)
                    except ValueError:
                        raise ValueError("Incompatible coords with function sig")
                    else:
                        new_varspecs[co] = varspecs.fromvector(ix)
            else:
                raise ValueError("Mismatch of function dimension and coords")
        else:
            raise ValueError("Incompatible Function type for Jacobian")
        if siglen == D+1:
            # check that non-coord (usually 't') is first
            if varspecs.signature[0] in coords:
                raise ValueError("Cannot have coordinate variable first in Jac sig")
        # keep record of old Fun
        fun_var = varspecs
        # overwrite with actual dictionary
        varspecs = need_specs = new_varspecs
    else:
        need_specs = filteredDict(varspecs, coords)
    jac = Diff(sortedDictValues(need_specs), coords)
    free = jac.freeSymbols
    if fnspecs is None:
        new_fnspecs = {}
    else:
        new_fnspecs = copy(fnspecs)
    # iterate until all inter-function inter-dependencies resolved
    n = 0
    while n < max_iter_depth:
        free, add_fnspecs = _generate_subderivatives(free, new_fnspecs)
        if add_fnspecs == {}:
            break
        else:
            new_fnspecs.update(add_fnspecs)
            n += 1
    return jac, new_fnspecs



def checkbraces(s):
    """Internal utility to verify that braces match."""
    if s.count('(') != s.count(')') or \
       s.count('[') != s.count(']'):
        raise SyntaxError("Mismatch of braces in expression %s" % s)


# Local version of who function because it's defined in PyDSTool.__init__
# which cannot be imported here (circular import)
# This function only finds Quantity or QuantSpec objects
def whoQ(objdict=None, deepSearch=False, frameback=3):
    objdict_out = {}
    if objdict is None:
        # look a default of 3 stack frames further back to access user's frame
        assert int(frameback)==frameback, "frameback argument must be a positive integer"
        assert frameback > 0, "frameback argument must be a positive integer"
        frame = sys._getframe()
        for i in range(frameback):
            frame = frame.f_back
        objdict = frame.f_globals
    typelist = [Quantity, QuantSpec]
    for objname, obj in objdict.items():
        if not isinstance(obj, six.class_types + (types.ModuleType,)):
            if compareClassAndBases(obj, typelist):
                objdict_out[objname] = obj
            elif deepSearch:
                if isinstance(obj, list) or isinstance(obj, tuple):
                    if sometrue([compareClassAndBases(x, typelist) \
                                 for x in obj]):
                        objdict_out[objname] = obj
                elif isinstance(obj, dict):
                    if sometrue([compareClassAndBases(x, typelist) \
                                 for x in obj.values()]):
                        objdict_out[objname] = obj
    return objdict_out   # original 'returnlevel=2'

# ----------------------------------------------------------------------

_localfuncnames = protected_macronames + builtin_auxnames
_avoid_math_symbols = ['e', 'pi'] + allmathnames_symbolic

class QuantSpec(object):
    """Specification for a symbolic numerical quantity."""

    def __init__(self, subjectToken, specStr="", specType=None,
                 includeProtected=True, treatMultiRefs=True,
                 preserveSpace=False, ignoreSpecial=None):
        if isNameToken(subjectToken) or (isMultiRef(subjectToken) \
                   and treatMultiRefs and alltrue([char not in subjectToken \
                                       for char in ['+','-','/','*','(',')']])):
            token = subjectToken.strip()
##            if token in ('for','sum'):
##                raise ValueError("Cannot use reserved macro name '%s' as a "%token +\
##                                 "QuantSpec's name")
##            else:
            self.subjectToken = token
        else:
            print("Found:%s" % subjectToken)
            raise ValueError("Invalid subject token")
        if specStr != "" and specStr[0] == '+':
            if len(specStr) > 1:
                specStr = specStr[1:]
            else:
                raise ValueError("Invalid specification string")
        if specType is None:
            self.specType = 'ExpFuncSpec'
        elif specType in specTypes:
            self.specType = specType
        else:
            raise ValueError("Invalid specType provided: %s"%specType)
        ignoreTokens = ['[',']']
        if ignoreSpecial is not None:
            ignoreTokens.extend(remain(ignoreSpecial,ignoreTokens))
        self.parser = parserObject(specStr, includeProtected,
                                   treatMultiRefs=treatMultiRefs,
                                   ignoreTokens=ignoreTokens,
                                   preserveSpace=preserveSpace)
        self.specStr = "".join(self.parser.tokenized)
        checkbraces(self.specStr)
        # transfer these up to this level
        self.usedSymbols = self.parser.usedSymbols
        self.freeSymbols = self.parser.freeSymbols
        try:
            # assume max of rank 2
            self.dim = getdim(self.specStr)  # for vector
        except IndexError:
            # spec not long enough therefore self cannot be a vector
            self.dim = 0
        self._check_self_ref()

    def _check_self_ref(self):
        if self.specType == 'ExpFuncSpec':
            if self.subjectToken in self.parser.usedSymbols:
                locs = self.parser.find(self.subjectToken)
                for loc in locs:
                    if loc <= 1 or self.parser.tokenized[loc-2:loc] != ['initcond', '(']:
                        raise ValueError("Cannot define the symbol "+self.subjectToken+" in"
                                 " terms of itself with spec type ExpFuncSpec")


    def redefine(self, specStr):
        p = self.parser
        self.parser = parserObject(specStr, p.includeProtected,
                         p.treatMultiRefs, p.ignoreTokens, p.preserveSpace)
        self.specStr = "".join(self.parser.tokenized)
        checkbraces(self.specStr)
        self.usedSymbols = self.parser.usedSymbols
        self.freeSymbols = self.parser.freeSymbols
        try:
            # assume max of rank 2
            self.dim = getdim(self.specStr)  # for vector
        except IndexError:
            # spec not long enough therefore self cannot be a vector
            self.dim = 0


    def isspecdefined(self):
        return self.specStr.strip() != ""


    def isDefined(self, verbose=False):
        return self.specStr.strip() != ""


    def __call__(self, *args):
        # args is a trick to avoid an exception when Python eval() function
        # is called on an expression involving a function, and the function
        # name has only been defined as a QuantSpec: then this is where eval
        # will be calling, with whatever arguments appeared in the expression!
        # In that case just return the arguments as a string in a string form
        # of the function call (like the identity function)
        if self.isDefined():
            # get QuantSpec's definition, if it exists
            base_str = self.specStr
        else:
            # otherwise, it may be an empty QuantSpec, like the way
            # pi is defined
            base_str = self.subjectToken
        if args == ():
            return base_str
        else:
            # need to return a QuantSpec type if being called with arguments
            # and propagate str() calls to all sub-elements of any sequence types
            return QuantSpec("__result__", base_str + "(" + \
                       _propagate_str_eval(args) + ")", self.specType)


    def isCompound(self, ops = ['+','-','*','/']):
        return self.parser.isCompound(ops)


    def __contains__(self, tok):
        return tok in self.parser.tokenized

    def __getitem__(self, ix):
        return self.parser.tokenized[ix]

    def __setitem__(self, ix, s):
        raise NotImplementedError

    def __delitem__(self, ix):
        raise NotImplementedError

    # -------------- method group for spec string assembly
    def __combine(self, other, opstr, reverseorder=0):
        # includes dealing with 0 + s -> s, s*0 -> 0, 1 * s -> s, etc.
        # doesn't tackle issues with 0 appearing in a division
        # QuantSpec <op> QuantSpec -> specType rules used by
        # resolveSpecTypeCombos() function:
        #   same types -> that type
        #   ExpFuncSpec o RHSfuncSpec -> RHSfuncSpec
        #   ImpFuncSpec o RHSfuncSpec -> RHSfuncSpec (should this be an error?)
        #   ImpFuncSpec o ExpFuncSpec -> ImpFuncSpec
        specType = self.specType   # default initial value
        self_e = None
        try:
            self_e = eval(self.specStr, {}, {})
            check_self_e = True
        except:
            check_self_e = False
        include_self = True
        if check_self_e and (isinstance(self_e, _num_types)):
            if self_e == 0:
                # we don't try to deal with divisions involving 0 here!
                if opstr == '*':
                    if hasattr(other, 'specType'):
                        specType = resolveSpecTypeCombos(self.specType,
                                                         other.specType)
                    else:
                        specType = self.specType
                    return QuantSpec("__result__", "0", specType)
                elif opstr in ['+','-']:
                    include_self = False
            elif self_e == 1 and opstr == '*':
                include_self = False
        if self.isDefined():
            if reverseorder:
                if self.specStr[0] == '-' and opstr == '-':
                    opstr = '+'
                    self_specStr_temp = self.specStr[1:]
                else:
                    self_specStr_temp = self.specStr
            else:
                self_specStr_temp = self.specStr
        else:
            self_specStr_temp = self.subjectToken
        if isinstance(other, Quantity):
            otherstr = other.name
            otherbraces = 0
            specType = resolveSpecTypeCombos(self.specType, other.specType)
        elif isinstance(other, QuantSpec):
            specType = resolveSpecTypeCombos(self.specType, other.specType)
            e = None
            # get QuantSpec's definition, if it exists
            otherstr = other()
            try:
                e = eval(otherstr, {}, {})
                check_e = True
            except:
                check_e = False
            if check_e and isinstance(e, _num_types):
                if e == 0:
                    # we don't try to deal with divisions involving 0 here!
                    if opstr == '*':
                        return QuantSpec("__result__", "0", specType)
                    elif opstr in ['+','-']:
                        return QuantSpec("__result__", self_specStr_temp,
                                         specType)
                    elif opstr == '/' and reverseorder:
                        return QuantSpec("__result__", "0", self.specType)
                elif e == 1 and (opstr == '*' or \
                                 (opstr == '/' and not reverseorder)):
                    return QuantSpec("__result__", self_specStr_temp, specType)
                elif e == -1 and (opstr == '*' or \
                                 (opstr == '/' and not reverseorder)):
                    return QuantSpec("__result__", '-'+self_specStr_temp,
                                     specType)
            if otherstr[0] == '-' and opstr == '-' and not reverseorder:
                opstr = '+'
                otherstr = str(otherstr[1:])
            otherbraces = int((opstr == '*' or opstr == '-') and other.isCompound(['+','-']) \
                           or opstr == '/' and other.isCompound())
        elif isinstance(other, str):
            e = None
            try:
                e = eval(other, {}, {})
                check_e = True
            except:
                check_e = False
            if check_e and isinstance(e, _num_types):
                if e == 0:
                    # we don't try to deal with divisions involving 0 here!
                    if opstr == '*':
                        return QuantSpec("__result__", "0", self.specType)
                    elif opstr in ['+','-']:
                        return QuantSpec("__result__", self_specStr_temp,
                                         self.specType)
                    elif opstr == '/' and reverseorder:
                        return QuantSpec("__result__", "0", self.specType)
                elif e == 1 and (opstr == '*' or \
                                 (opstr == '/' and not reverseorder)):
                    return QuantSpec("__result__", self_specStr_temp,
                                     self.specType)
                elif e == -1 and (opstr == '*' or \
                                 (opstr == '/' and not reverseorder)):
                    return QuantSpec("__result__", '-'+self_specStr_temp,
                                     self.specType)
            if other[0] == '-' and opstr == '-' and not reverseorder:
                opstr = '+'
                otherstr = str(other[1:]).strip()
            else:
                otherstr = str(other).strip()
            temp = QuantSpec("__result__", otherstr, self.specType)
            otherbraces = int((opstr == '*' or opstr == '-') and temp.isCompound(['+','-']) \
                           or opstr == '/' and temp.isCompound())
        elif isinstance(other, _num_types):
            if other == 0:
                # we don't try to deal with divisions involving 0 here!
                if opstr == '*':
                    return QuantSpec("__result__", "0", self.specType)
                elif opstr in ['+','-']:
                    return QuantSpec("__result__", self_specStr_temp,
                                     self.specType)
                elif opstr == '/' and reverseorder:
                    return QuantSpec("__result__", "0", self.specType)
            elif other == 1 and (opstr == '*' or \
                                 (opstr == '/' and not reverseorder)):
                return QuantSpec("__result__", self_specStr_temp,
                                 self.specType)
            elif other == -1 and (opstr == '*' or \
                                 (opstr == '/' and not reverseorder)):
                return QuantSpec("__result__", '-'+self_specStr_temp,
                                 self.specType)
            if other < 0 and opstr == '-' and not reverseorder:
                opstr = '+'
                otherstr = str(-other)
            else:
                otherstr = str(other)
            otherbraces = 0
        else:
            raise TypeError("Invalid type to combine with QuantSpec: %s"%type(other))
        if otherstr == "":
            combinestr = ""
            selfbraces = 0
        else:
            selfbraces = int((opstr == '*' or opstr == '-') and self.isCompound(['+','-']) \
                          or opstr in ['-','/'] and self.isCompound())
            combinestr = " " + opstr + " "
        if not include_self:
            selfstr = ""
            combinestr = ""
            otherbraces = 0   # override
        else:
            selfstr = selfbraces*"(" + self_specStr_temp + selfbraces*")"
        if reverseorder:
            resultstr = otherbraces*"(" + otherstr + otherbraces*")" \
                   + combinestr + selfstr
        else:
            resultstr = selfstr + combinestr + otherbraces*"(" + otherstr \
                                                  + otherbraces*")"
        return QuantSpec("__result__", resultstr, specType)

    def __add__(self, other):
        return self.__combine(other, "+")

    def __mul__(self, other):
        return self.__combine(other, "*")

    def __sub__(self, other):
        return self.__combine(other, "-")

    def __div__(self, other):
        return self.__combine(other, "/")

    def __radd__(self, other):
        return self.__combine(other, "+", 1)

    def __rmul__(self, other):
        return self.__combine(other, "*", 1)

    def __rsub__(self, other):
        return self.__combine(other, "-", 1)

    def __rdiv__(self, other):
        return self.__combine(other, "/", 1)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __rpow__(self, other):
        return self.__pow__(other, 1)

    def __pow__(self, other, reverseorder=0):
        if self.isDefined():
            selfstr = self.specStr
        else:
            selfstr = self.subjectToken
        if reverseorder:
            specType = self.specType
            # other can only be non-Quant for rpow to be called
            if isinstance(other, str):
                try:
                    otherval = float(other)
                    if otherval == 0:
                        # result is 0
                        return QuantSpec("__result__", '0', specType)
                    elif otherval == 1:
                        # result is 1
                        return QuantSpec("__result__", '1', specType)
                except ValueError:
                    otherstr = other
            elif isinstance(other, _num_types):
                if other == 0:
                    # result is 0
                    return QuantSpec("__result__", '0', specType)
                elif other == 1:
                    # result is 1
                    return QuantSpec("__result__", '1', specType)
                else:
                    otherstr = str(other)
            else:
                raise TypeError("Invalid type to combine with QuantSpec: %s"%type(other))
            arg1str = otherstr
            arg2str = selfstr
        else:
            if isinstance(other, Quantity):
                otherstr = other.name
                specType = resolveSpecTypeCombos(self.specType, other.specType)
            elif isinstance(other, QuantSpec):
                specType = resolveSpecTypeCombos(self.specType, other.specType)
                # get QuantSpec's definition, if it exists
                otherstr = other()
                try:
                    otherval = float(otherstr)
                    if otherval == 0:
                        # result is 1
                        return QuantSpec("__result__", '1', specType)
                    elif otherval == 1:
                        # result is self
                        return QuantSpec("__result__", selfstr, specType)
                except ValueError:
                    # take no action
                    pass
            elif isinstance(other, str):
                try:
                    otherval = float(other)
                    if otherval == 0:
                        # result is 1
                        return QuantSpec("__result__", '1', specType)
                    elif otherval == 1:
                        # result is self
                        return QuantSpec("__result__", selfstr, specType)
                except ValueError:
                    otherstr = other
                specType = self.specType
            elif isinstance(other, _num_types):
                specType = self.specType
                if other == 0:
                    # result is 1
                    return QuantSpec("__result__", '1', specType)
                elif other == 1:
                    # result is self
                    return QuantSpec("__result__", selfstr, specType)
                else:
                    otherstr = str(other)
            else:
                raise TypeError("Invalid type to combine with QuantSpec: %s"%type(other))
            arg1str = selfstr
            arg2str = otherstr
        return QuantSpec("__result__", 'Pow('+arg1str+','+arg2str+')', specType)

    def __neg__(self):
        if self.isCompound():
            return QuantSpec("__result__", "-("+self.specStr+")", self.specType)
        elif self.specStr != "":
            if self.specStr[0]=='-':
                return QuantSpec("__result__", self.specStr[1:], self.specType)
            else:
                return QuantSpec("__result__", "-"+self.specStr, self.specType)
        else:
            return QuantSpec("__result__", "-"+self.subjectToken, self.specType)

    def __pos__(self):
        return copy(self)

    def __len__(self):
        return len(self.specStr)
    # -------------- end method group

    def __repr__(self):
        return "QuantSpec %s (%s)"%(self.subjectToken,self.specType)

    __str__ = __call__

    __hash__ = None

    def __eq__(self, other, diff=False):
        results = []
        try:
            results.append(type(self) == type(other))
            results.append(self.subjectToken == other.subjectToken)
            results.append(self.specStr.replace(" ","") == \
                           other.specStr.replace(" ",""))
            results.append(self.specType == other.specType)
            results.append(self.parser() == other.parser())
        except AttributeError as err:
            if diff:
                print("Type: %s %r" % (className(self),  results))
                print("  " + str(err))
            return False
        if diff:
            print("Type: %s %r" % (className(self),  results))
        return alltrue(results)


    def __ne__(self, other):
        return not self.__eq__(other)


    def difference(self, other):
        """Show differences in comparison to another QuantSpec."""
        self.__eq__(other, diff=True)


    def isvector(self):
        """Is the quantity spec of the form [a,b,c]"""
        try:
            return '['==self.specStr[0] and ']'==self.specStr[-1]
        except IndexError:
            return False


    def fromvector(self, ix=None):
        """Turn the QuantSpec's specification of a vector to a list of
        the element's specifications."""
        if self.isvector():
            if self.dim > 0:
                arglist = splitargs(self.specStr[1:-1], ['(','['], [')',']'])
                res = [QuantSpec(self.subjectToken+'_%i'%i,
                                 arglist[i], self.specType) \
                        for i in range(self.dim)]
            else:
                res = [QuantSpec(self.subjectToken+"_0", self.specStr[1:-1],
                                     self.specType)]
            if ix is None:
                return res
            else:
                try:
                    return res[ix]
                except IndexError:
                    raise IndexError("Index out of range in vector QuantSpec")
        else:
            raise TypeError("This QuantSpec does not specify a vector")


    def simplify(self):
        """Simplify expression *IN PLACE*, but do not substitute terms from
        other quantities or reduce constants such as Pi to a floating
        point number."""
        qtemp = self._eval(0, {})
        self.parser = qtemp.parser
        self.specStr = qtemp.specStr
        self.freeSymbols = qtemp.freeSymbols
        self.usedSymbols = qtemp.usedSymbols


    def renderForCode(self):
        """Return code-ready version of spec (e.g. uncapitalize built-in
        function names)"""
        return self._eval(-1, {})


    def eval(self, *scopearg, **defs):
        """Evaluate expression, in an optional scope dictionary (defaults to locals).
        Explicit definitions override scope dictionary definitions.

        Scope dictionary maps variable name -> defining object."""
        return self._eval(1, *scopearg, **defs)


    def _eval(self, eval_type, *scopearg, **defs):
        """Internal helper function for eval. The use of indirection using
        _eval actually allows the correct scope to be accessed by whoQ
        when this function automatically collects definitions."""
        # only one of defs or scopearg can be defined.
        # if both are undefined then caller's local scope used to find
        # possible definitions.
        scope_funs = {}
        if eval_type == -1:
            scopeobjs = {}
            scope = {}
        else:
            if defs == {}:
                # defs was not defined
                if scopearg == ():
                    scope = {}
                    # empty variable, parameter declarations pass through
                    if self.specStr == '':
                        scopeobjs = {}
                    else:
                        # try eval'ing to a number
                        try:
                            strval = repr(eval(self.specStr,{},{}))
                        except:
                            # use local names if eval_type==1
                            if eval_type==1:
                                scopeobjs = whoQ()
                            else:
                                scopeobjs = {}
                        else:
                            if self.subjectToken != "__result__":
                                scope[self.subjectToken] = strval
                            scopeobjs = {}
                    # process scope objects
                    for key, obj in scopeobjs.items():
                        if isinstance(obj, Quantity):
                            if obj.name != self.subjectToken:
                                strval = str(obj.spec._eval(eval_type, {}))
                                if obj.name in [strval, "__result__"]:
                                    # not a valid key
                                    continue
                                # try to evaluate to float if not one already
                                try:
                                    strval = repr(eval(strval,{},{}))
                                except:
                                    pass
                                valq = QuantSpec('__dummy__', strval)
                                if isinstance(obj, Fun):
                                    # functions dealt with in scope_funs
                                    scope_funs[obj.name] = obj
                                else:
                                    if valq.isCompound() and strval[0] != '(':
                                        scope[obj.name] = '('+strval+')'
                                    else:
                                        scope[obj.name] = strval
                        else:
                            if obj.subjectToken != self.subjectToken:
                                strval = str(obj._eval(eval_type, {}))
                                if obj.subjectToken in [strval, "__result__"]:
                                    continue
                                # try to evaluate to float if not one already
                                try:
                                    strval = repr(eval(strval,{},{}))
                                except:
                                    pass
                                valq = QuantSpec('__dummy__', strval)
                                if valq.isCompound() and strval[0] != '(':
                                    scope[obj.subjectToken] = '('+strval+')'
                                else:
                                    scope[obj.subjectToken] = strval
                else:
                    if not hasattr(scopearg[0], 'keys') and len(scopearg)==1:
                        raise TypeError("scope argument must be a dictionary or Point")
                    scopeobjs = scopearg[0]
                    # process scope objects, avoiding identity mappings of
                    # tokens that could lead to infinite loops
                    scope = {}
                    for key, obj in scopeobjs.items():
                        if key == '__result__':
                            # not a valid key
                            continue
                        if isinstance(obj, Quantity):
                            if obj.name != self.subjectToken and \
                                        (obj.name != key or obj.isspecdefined()):
                                strval = str(obj.spec._eval(eval_type,
                                                filteredDict(scopeobjs, [key], True)))
                                # try to evaluate to float if not one already
                                try:
                                    strval = repr(eval(strval,{},{}))
                                except:
                                    pass
                                valq = QuantSpec('__dummy__', strval)
                                if valq.isCompound() and strval[0] != '(':
                                    scope[key] = '('+strval+')'
                                else:
                                    scope[key] = strval
                        elif isinstance(obj, QuantSpec):
                            if obj.subjectToken != self.subjectToken \
                                       and (obj.subjectToken != key or obj.isspecdefined()):
                                strval = str(obj._eval(eval_type,
                                                filteredDict(scopeobjs, [key], True)))
                                # try to evaluate to float if not one already
                                try:
                                    strval = repr(eval(strval,{},{}))
                                except:
                                    pass
                                valq = QuantSpec('__dummy__', strval)
                                if valq.isCompound() and strval[0] != '(':
                                    scope[key] = '('+strval+')'
                                else:
                                    scope[key] = strval
                        else:
                            if isinstance(obj, _num_types):
                                if key != repr(obj):
                                    scope[key] = repr(obj)
                            elif isinstance(obj, str):
                                if key != obj:
                                    scope[key] = obj
                            else:
                                raise TypeError("Invalid scope value")
            else:
                # defs was defined
                if scopearg == ():
                    scopeobjs = copy(defs)
                    # process scope objects
                    scope = {}
                    for key, obj in scopeobjs.items():
                        if key == '__result__':
                            continue
                        if isinstance(obj, Quantity):
                            if obj.name != self.subjectToken and \
                                          obj.name != key:
                                strval = str(obj.spec._eval(eval_type,
                                            filteredDict(scopeobjs,
                                                         [self.subjectToken], True)))
                                # try to evaluate to float if not one already
                                try:
                                    strval = repr(eval(strval,{},{}))
                                except:
                                    pass
                                valq = QuantSpec('__dummy__', strval)
                                if valq.isCompound() and strval[0] != '(':
                                    scope[key] = '('+strval+')'
                                else:
                                    scope[key] = strval
                        elif isinstance(obj, QuantSpec):
                            if obj.subjectToken != self.subjectToken and \
                                       obj.subjectToken != key:
                                strval = str(obj._eval(eval_type,
                                            filteredDict(scopeobjs,
                                                         [self.subjectToken], True)))
                                # try to evaluate to float if not one already
                                try:
                                    strval = repr(eval(strval,{},{}))
                                except:
                                    pass
                                valq = QuantSpec('__dummy__', strval)
                                if valq.isCompound() and strval[0] != '(':
                                    scope[key] = '('+strval+')'
                                else:
                                    scope[key] = strval
                        else:
                            if isinstance(obj, _num_types):
                                if key != repr(obj):
                                    scope[key] = repr(obj)
                            elif isinstance(obj, str):
                                if key != obj:
                                    scope[key] = obj
                            else:
                                raise TypeError("Invalid scope value")
                else:
                    raise ValueError("Cannot provide both a scope dictionary and explicit "
                                     "definitions")
        if self.isvector():
            res_eval = [qt._eval(eval_type, scope) for qt in self.fromvector()]
            return QuantSpec(self.subjectToken,
                             '['+','.join([r() for r in res_eval])+']',
                             self.specType)
        # apply definitions in scope
        arglist = list(scope.keys())
        arglist.sort()
        # convert any numeric substitutions into strings
        for k, v in defs.items():
            if not isinstance(v, str) and k not in [str(v), "__result__"]:
                scope[k] = str(v)
            else:
                continue
            valq = QuantSpec('__dummy__', scope[k])  # not necessarily = v!
            # ensure brace around compound args after substitution
            if valq.isCompound() and scope[k][0] != '(':
                scope[k] = '(' + scope[k] + ')'
        # just ignore extra evaluation arguments for symbols that are
        # not used in the expression -- e.g. for when subs() is called
        # blindly, including unused pars. this saves preparing
        # lists of which pars go with which expressions.
        # i.e. keep following statement commented out!
##        assert remain(arglist, freesymbs) == [], "Invalid arguments to eval"
        argmap = symbolMapClass(scope)
        toks = self.parser.tokenized
        if toks == []:
            return self
        elif len(toks)==1:
            mappedval=argmap(argmap(toks[0]))
            if mappedval in ['E', 'Pi']:
                if eval_type==1:
                    # these two have numeric values
                    mappedval = repr(eval(mappedval.lower()))
                elif eval_type==-1:
                    mappedval = mappedval.lower()
            return QuantSpec(self.subjectToken, mappedval, self.specType)
        # deal with string literals that would otherwise lose their quote marks
        # during later evaluation.
        # ... first, convert all single quotes to double quotes
        for q_ix in range(len(toks)):
            if toks[q_ix] == "'":
                toks[q_ix] = '"'
        # ensure no string literals get converted
        num_quotes = toks.count('"')
        literals = []
        repliterals = []
        # Don't want to return math names to title case if eval_type==1 or -1
        # so don't add these to the inverse map in those cases
        if eval_type == 0:
            FScompat_namemap = {}
            FScompat_namemapInv = invertMap(mathNameMap)
        elif eval_type == 1:
            FScompat_namemap = mathNameMap
            FScompat_namemapInv = {}
        else:
            # eval_type == -1
            FScompat_namemap = mathNameMap
            FScompat_namemapInv = {}
        if num_quotes > 0:
            new_toks = []
            n_ix = -1
            if mod(num_quotes,2)!=0:
                raise PyDSTool_ValueError("Mismatch in number of quotes in spec")
            # ensure single quotes always end up as double quotes (makes it C-compatible
            #  and simplify_str() annoyingly converts doubles to singles)
            FScompat_namemapInv.update({"'": '"'})
            for qname in range(int(num_quotes/2)):
                next_ix = toks[n_ix+1:].index('"')
                new_toks.extend(toks[n_ix+1:next_ix])
                n_ix = next_ix
                end_ix = toks[n_ix+1:].index('"')+n_ix
                literal = "".join(toks[n_ix+1:end_ix+1])
                literals.append(literal)
                qlit = '"'+literal+'"'
                replit = '"__literal_'+literal+'"'
                repliterals.append(replit[1:-1])
                new_toks.append(replit[1:-1])
                # add to inverse name map to restore literals after eval()
                # (don't include quotes)
                FScompat_namemapInv[replit[1:-1]] = qlit
            # finish adding the remaining tokens
            new_toks.extend(toks[end_ix+2:])
            qtemp = QuantSpec(self.subjectToken, "".join(argmap(new_toks)),
                          self.specType)
        else:
            new_toks = toks
            qtemp = self.__copy__()
            qtemp.mapNames(argmap)
        # now apply scope mapping so that literals remain unaffected.
        # second pass for one level of indirect references
        # but only use keys that weren't present before, to avoid circular
        # mapping: e.g. s->t then t->s
        if scope != {}:
            qtemp.mapNames(filteredDict(scope,remain(toks,scope.keys())))
        # references to builtin aux functions or macros require temp
        # definitions of those as Fun objects
        qtoks = qtemp.parser.tokenized
        localfuncs = intersect(_localfuncnames, qtoks)
        for symb in remain(qtemp.freeSymbols, repliterals):
            if symb not in _avoid_math_symbols:
                FScompat_namemap[symb] = symb.replace('.', '_')
                FScompat_namemapInv[FScompat_namemap[symb]] = symb
        for symb in localfuncs:
            FScompat_namemap[symb] = symb+"_"
            FScompat_namemapInv[symb+"_"] = symb
        qtemp.mapNames(FScompat_namemap)
        qtemp_feval = copy(qtemp)
        # map these separately because need to keep the two dicts distinct
        qtemp_feval.mapNames(feval_map_const)
        qtemp_feval.mapNames(feval_map_symb)
        # once math names -> lower case, try mapping scope defs again
        qtemp_feval.mapNames(scope)
        # are qtemp or qtemp_feval already containing only numeric values
        # or built-in math functions?
        num_qtemp = qtemp.freeSymbols == []
        num_qtemp_feval = qtemp_feval.freeSymbols == []
        sq = str(qtemp)
        sq_feval = str(qtemp_feval)
        # try pre-simpilfying the expressions
        try:
            defstr = simplify_str(sq)
        except (SyntaxError, MemoryError):
            # Probably from 'for' syntax, which contains a clause with
            # a '+' or '-' symbol without being part of an arithmetical expr
            # Would like to make this cleaner...
            defstr = sq
        except TypeError:
            print(self.specStr + sq)
            raise
        else:
            # HACK: fix simplify_str()'s return of vector arg with round braces
            if self.isvector() and defstr[0]=='(' and defstr[-1]==')':
                defstr = '['+defstr[1:-1]+']'
        try:
            defstr_feval = simplify_str(sq_feval)
        except (SyntaxError, MemoryError):
            # Probably from 'for' syntax, which contains a clause with
            # a '+' or '-' symbol without being part of an arithmetical expr
            # Would like to make this cleaner...
            defstr_feval = sq_feval
        else:
            # HACK: fix simplify_str()'s return of vector arg with round braces
            if self.isvector() and defstr_feval[0]=='(' and defstr_feval[-1]==')':
                defstr_feval = '['+defstr_feval[1:-1]+']'
        # otherwise, continue pre-process simplified definition strings
        for clause_type in ('if', 'max', 'min', 'sum'):
            if clause_type not in localfuncs:
                continue
            # protect condition or max/min clauses from being Python-eval'd
            # because Python will turn it into a boolean value or mess up max/min!
            # ... so substitute the whole clause with a dummy name
            num_stmts = qtoks.count(clause_type)
            n_ix = 0
            for stmt_ix in range(num_stmts):
                n_ix = qtoks[n_ix:].index(clause_type) + n_ix
                if clause_type == 'if':
                    stop_ix = qtoks[n_ix:].index(',')+n_ix
                else:
                    stop_ix = findEndBrace(qtoks[n_ix+1:])+n_ix+1
                clause_original = "".join(qtoks[n_ix+2:stop_ix])
                clause = "".join(qtemp.parser.tokenized[n_ix+2:stop_ix])
                defstr = defstr.replace(clause,'__temp_clause__'+str(stmt_ix))
                defstr_feval = defstr_feval.replace(clause,'__temp_clause__'+str(stmt_ix))
                # add to inverse name map to restore clauses after eval()
                FScompat_namemapInv['__temp_clause__'+str(stmt_ix)] = clause_original
                qtemp.freeSymbols.append('__temp_clause__'+str(stmt_ix))
                n_ix = stop_ix
        else:
            num_stmts = 0
        # add underscored suffix to built in function names to avoid clashing
        # with actual Python names, e.g. "if", "for"
        if not num_qtemp:
            freenamelist = remain(qtemp.freeSymbols,
                                  allmathnames_symbolic + localfuncs)
            local_scope = dict(zip(freenamelist,[QuantSpec(symb) for symb in freenamelist]))
            local_scope.update(local_fndef)
            local_scope_feval = copy(local_scope)
            if eval_type == 1:
                # Allow constants like Pi and E and function calls like
                # Log(10) to become floating point numbers
                local_scope_feval.update(math_globals)
            # crucial to update with scope_funs last, so that these "true" definitions
            # can overwrite the "dummy" placeholder definitions that were there
            local_scope.update(scope_funs)
            local_scope_feval.update(scope_funs)
        else:
            if eval_type == 1:
                local_scope = {}
                local_scope_feval = math_globals
            else:
                local_scope = local_scope_feval = {}
        # DEBUG: info(local_scope, "Local scope")
        # because the free names have been defined locally as QuantSpecs,
        # eval will return another QuantSpec
        if eval_type == 1:
            # attempt to evaluate to a numeric type, but settle for the most
            # simplified numeric version otherwise (including lower case functions)
            try:
                new_qtemp = eval(defstr_feval, {}, local_scope_feval)
                assert isinstance(new_qtemp, (Quantity, QuantSpec, str, list, \
                                        _num_types))
            except:
                new_qtemp = QuantSpec(self.subjectToken, defstr_feval, self.specType)
        elif eval_type == 0:
            try:
                new_qtemp = eval(defstr, {}, local_scope)
                assert isinstance(new_qtemp, (Quantity, QuantSpec, str, list, \
                                    _num_types))
                # assume that number of braces correlates with complexity
                # only keep this if trying to simplify and expression becomes
                # simpler, or we're trying to reduce float sub-expressions
                assert str(new_qtemp).count('(') <= defstr.count('('), \
                       "Didn't get simpler -- going to except clause"
            except:
                new_qtemp = QuantSpec(self.subjectToken, defstr, self.specType)
            # if just simplifying then assume that number of braces
            # correlates with complexity
            if str(new_qtemp).count('(') > self.specStr.count('('):
                # didn't get simpler, so quit now
                return self
        else:
            new_qtemp = QuantSpec(self.subjectToken, defstr, self.specType)
        if isinstance(new_qtemp, _num_types):
            new_qtemp = QuantSpec(self.subjectToken, repr(new_qtemp), self.specType)
        elif not isinstance(new_qtemp, QuantSpec):
            # could be list of Quantities or QuantSpecs
            new_qtemp = QuantSpec(self.subjectToken, strIfSeq(new_qtemp), self.specType)
        else:
            # Only do the following when not rendering for code output
            # Turn math functions back to title case if float eval failed
            # but only those that weren't originally in lower case
            if eval_type != -1:
                new_qtemp.mapNames(filteredDict(inverseMathNameMap,
                                                qtoks, 1))
            # make sure specType is set correctly
            new_qtemp.specType = self.specType
        new_qtemp.mapNames(FScompat_namemapInv)
        return new_qtemp


    def tonumeric(self):
        if self.freeSymbols == []:
            if self.isvector():
                try:
                    return array(eval(self.specStr, {}, {}))
                except:
                    raise TypeError("Cannot convert this QuantSpec to a numeric type")
            else:
                try:
                    return eval(self.specStr, {}, {})
                except:
                    raise TypeError("Cannot convert this QuantSpec to a numeric type")
        else:
            raise TypeError("Cannot convert this QuantSpec to a numeric type")


    def mapNames(self, nameMap):
        """Can pass a dictionary or a symbolMapClass instance to this method,
        but it will convert it internally to a symbolMapClass."""
        namemap = symbolMapClass(nameMap)
        newsubj = namemap(self.subjectToken)
        if isinstance(newsubj, str) and \
              (isNameToken(newsubj, True) or \
               isHierarchicalName(newsubj, treatMultiRefs=True)):
            self.subjectToken = newsubj
        else:
            print("Attempted to rename %s with object of type %s, value:"%(self.subjectToken,str(type(newsubj))) + newsubj)
            raise TypeError("Cannot rename a QuantSpec with a non-name string")
        toks = [namemap(t) for t in self.parser.tokenized]
        # strip() in case spaces were preserved and there were trailing spaces
        # (these can confuse function definitions made from these into thinking
        # that there's a badly indented code block following the function body)
        self.specStr = "".join(toks).strip()
        self.parser.specStr = self.specStr
        self.specStr = \
                self.parser.parse(self.parser.ignoreTokens, None,
                                  includeProtected=self.parser.includeProtected,
                                  reset=True)
        self.usedSymbols = self.parser.usedSymbols
        self.freeSymbols = self.parser.freeSymbols


    # difficult to reconstruct original options at __init__, so just copy by pickle
    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


# --------------------------------------------------------------------------
### Multi-quantity reference and multi-quantity definition utilities

def processMultiDef(s, specstr, specType):
    # Assume attempt was made at using a multiple Quantity definition
    # (i.e. that isMultiDef() returned true).
    # so if further tests show the wrong syntax, then cause error,
    # not just return False (because square brackets are not otherwise
    # allowed in a QuantSpec subjectToken.
    # Correct format is "<symbol 1>[<symbol 2>,<int 1>,<int 2>]"
    # where s1 != s2 and i1 < i2, and s1 does not end in a number
    m = parserObject(s, ignoreTokens=['['])
    i1 = eval(m.tokenized[4])
    i2 = eval(m.tokenized[6])
    rootname = m.tokenized[0]
    ixname = m.tokenized[2]
    if len(rootname) > 1:
        test_ix = -1
    else:
        test_ix = 0
    root_test = isNameToken(rootname) and rootname[test_ix] not in num_chars
    if not alltrue([len(m.usedSymbols)==4, m.tokenized[1]=="[",
                root_test, isNameToken(ixname),
                m.tokenized[7]=="]", rootname != ixname,
                m.tokenized[3]==m.tokenized[5]==",",
                compareNumTypes(type(i1), type(i2)) and isinstance(i1, _int_types),
                 i1 < i2]):
        raise ValueError("Syntax error in multiple Quantity definition "
                         "spec: " + s)
    ## Check that all x[f(i)] references have f(i) integers
    # any references outside i1..i2 become freeSymbols, for x
    # either z or another symbol (known as 'root' below).
    # Also, any zXX refs present in the mrefs, where XX is not in
    # the range [i1,i2], are free symbols of this Quantity
    # because they aren't being defined here
    defrange = range(i1,i2+1)
    # Quantity subject token is 'z[<ixname>]'
    # and future 'name' attribute
    token = rootname + "[" + ixname + "]"
    all_mrefs = {}    # dict of rootname -> valsref'd list
    mrefs = findMultiRefs(specstr)
    actual_freeSymbols = []
    for tup in mrefs:
        mref = processMultiRef(tup[1], tup[0],
                              ixname, defrange)
        all_mrefs[tup[0]] = mref
        if tup[0] == rootname:
            # ignore ones being defined here
            out_of_range = remain(mref, defrange)
        else:
            out_of_range = mref
        # only add those names not yet present in freeSymbols
        actual_freeSymbols.extend(remain([tup[0] + str(i) for i in\
                             out_of_range], actual_freeSymbols))
    ## Find direct references to z1-z10 in spec.
    # avoid using square brackets when making temp_spec
    surrogateToken = "__"+rootname+"__"
    surrogateSpec = specstr
    for (r, targetExpr) in mrefs:
        if r==rootname:
            surrogateSpec = surrogateSpec.replace(targetExpr,
                                                  surrogateToken)
    # making temp a QuantSpec gives us a free check for the
    # legality of the specType
    try:
        temp_qspec = QuantSpec(surrogateToken, surrogateSpec,
                               specType, treatMultiRefs=True)
    except ValueError as e:
        # replace temp name __z__ with actual token name z[i]
        new_e = (e[0].replace(surrogateToken, token))
        raise ValueError(new_e)
    for f in temp_qspec.freeSymbols:
        try:
            is_int = f[-1] in num_chars
        except:
            is_int = False
        if is_int:
            # f ends in a number
            p = findNumTailPos(f)
            # find out if the root of the symbol is rootname
            # then find out if the tail number is in the
            # defining range.
            if f[:p] == rootname and int(f[p:]) \
                    not in defrange:
                # it is not in range, so is not being defined
                # by this multiple Quantity definition, thus
                # it is a true free Symbol of this Quantity
                if f not in actual_freeSymbols:
                    actual_freeSymbols.append(f)
        elif f == surrogateToken or f == ixname:
            pass
        else:
            # f is a free symbol unrelated to token
            if f not in actual_freeSymbols:
                actual_freeSymbols.append(f)
    # use multi-ref treatment option for QuantSpec
    # NOTE -- 'subjectToken' was just 'token'
    subjectToken = rootname+'['+ixname+','+str(i1)+','+str(i2)+']'
    actual_spec = QuantSpec(subjectToken, specstr, specType,
                            treatMultiRefs=True)
    mdef = (rootname, ixname, i1, i2, token)
    return mdef, actual_spec, actual_freeSymbols


def isMultiDef(s):
    """Determine whether string is a candidate multiple Quantity definition."""

    temp = parserObject(s, ignoreTokens=['['])
    try:
        return alltrue([not temp.isCompound(), len(temp.tokenized)==8,
                        isNameToken(temp.tokenized[0]),
                        isNameToken(temp.tokenized[2]),
                        temp.tokenized[1]=="[", temp.tokenized[-1]=="]"])
    except IndexError:
        # e.g. if string doesn't contain enough tokens
        return False


def findMultiRefs(s):
    temp = parserObject(s, ignoreTokens=['['])
    # reflist contains tuples of (rootname, expr)
    reflist = []
    searchSymbols = temp.freeSymbols
    if len(temp.freeSymbols) == 0:
        return reflist
    # for each free symbol, see if it's followed by [ ... ]
    toks_to_process = temp.tokenized
    while True:
        next_ixs = []
        for f in searchSymbols:
            try:
                next_ixs.append(toks_to_process.index(f))
            except ValueError:
                # not found
                pass
        if len(next_ixs) > 0:
            next_free = min(next_ixs)
        else:
            break
        # find contents of [ ... ]
        try:
            if toks_to_process[next_free+1] == "[":
                try:
                    rb_ix = toks_to_process[next_free+1:].index("]") \
                            + next_free + 1
                except ValueError:
                    raise ValueError("Mismatched square braces in multiple "
                                     "quantity reference specification: " + s)
                reflist.append((toks_to_process[next_free],
                           "".join(toks_to_process[next_free:rb_ix+1])))
                toks_to_process = toks_to_process[rb_ix+1:]
            else:
                toks_to_process = toks_to_process[next_free+1:]
        except IndexError:
            # symbol is last in tokenized
            break
    return reflist


def isMultiRef(s):
    """Simple litmus test for candidate multiple quantity reference
    (or definition).

    Requires further testing using one of the other associated multiple
    quantity reference/definition tests."""
    if len(s) > 1:
        sparts = s.split('[')
        # check that '[' occurs as '<name>['
        if len(sparts) > 1:
            try:
                reftest = alltrue([sp[-1].isalnum() for sp in sparts[:-1]])
            except IndexError:
                # e.g. sp == ''
                reftest = False
        else:
            reftest = False
        return reftest
    else:
        return False


def processMultiRef(s, rootname, ixname, ok_range):
    # Is valid multiple quantity ref? ... correct format is
    # "rootname[<expr in ixname>]" for a valid rootname (does not end in
    # a number), where the expression evaluates to an integer when ixname
    # is substituted by an integer in the range ok_range = [i1,i2].
    temp = parserObject(s, ignoreTokens=['['])
    try:
        if len(rootname) > 1:
            test_ix = -1
        else:
            test_ix = 0
        roottest = rootname[test_ix] not in num_chars
        if not alltrue([roottest, len(temp.tokenized) >= 4,
                    temp.tokenized[0] == rootname,
                    rootname in temp.freeSymbols,
                    ixname in temp.freeSymbols,
                    rootname not in temp.tokenized[1:],
                    temp.tokenized[1]=="[", temp.tokenized[-1]=="]"]):
            raise ValueError("Error in multiple Quantity reference "
                             "spec: " + s)
        # then we have a fully syntax-checked multiple quantity ref
    except IndexError:
        # e.g. temp.tokenized[1] --> index out of bounds
        raise ValueError("Error in multiple Quantity reference "
                         "spec: " + s)
    # test evaluation of every expression for ixname in range [i1, i2] to be
    # an integer, where expr is everything between the square brackets
    original_tokenlist = temp.tokenized[2:-1]
    ixname_occs = []
    for i in range(len(original_tokenlist)):
        if original_tokenlist[i] == ixname:
            ixname_occs.append(i)
    # make a list of all referenced integers by this expression, over this
    # range ok_range = [i1,i2]
    vals_refd = []
    try:
        for ixval in ok_range:
            tokenlist = original_tokenlist
            for i in ixname_occs:
                tokenlist[i] = str(ixval)
            expr = "".join(tokenlist)
            try:
                val = eval(expr)
            except:
                val = None
            if not isinstance(val, _int_types):
                if int(val) == val:
                    # functions like scipy.mod return floats even
                    # though their values are equal to integers!
                    val = int(val)
                else:
                    raise AssertionError
            if val not in vals_refd:
                vals_refd.append(val)
    except AssertionError:
        raise ValueError("Non-integer evaluation of the index-generating "
                         "expression of multiple Quantity reference spec:"
                         " " + s)
    return vals_refd


def isMultiDefClash(obj, multiDefRefs, parentname=None):
    result = False    # default, initial value
    if isMultiDef(obj.name):
        objinfo = obj.multiDefInfo
        # objinfo == (True, rootname, ixname, i1, i2)
        if objinfo[0]:
            if parentname is None:
                rootname = objinfo[1]
            else:
                rootname = parentname + NAMESEP + objinfo[1]
            try:
                infotuple = multiDefRefs[rootname]
                # infotuple == (ixname, i1, i2)
                is_present = True
            except KeyError:
                is_present = False
            if is_present:
                # have more work to do, to establish which numbers are
                # already defined.
                obj_range = Interval('obj', int,
                                     [objinfo[3], objinfo[4]])
                self_range = Interval('self', int,
                                      [infotuple[1], infotuple[2]])
                if obj_range.intersect(self_range) is not None:
                    result = True
        else:
            # obj.name should not have returned as isMultiDef!
            raise ValueError("Object %s contains invalid multiple quantity"
                             " definitions"%obj.name)
    else:
        # is the single variable obj.name one of the multiDefRefs
        try:
            p = findNumTailPos(obj.name)
        except ValueError:
            # obj.name does not end in a number, so cannot clash
            return False
        obj_rootname = obj.name[:p]
        obj_ix = int(obj.name[p:])    # known to be ok integer
        for rootname, infotuple in multiDefRefs.items():
            if obj_rootname == rootname:
                # root of the name is present
                # so check if the number falls in the subject range
                if obj_ix in range(infotuple[3], infotuple[4]+1):
                    result = True
                    break
    return result


def evalMultiRefToken(mref, ixname, val):
    """Helper function for evaluating multi-reference tokens for
    given index values."""
    return eval(mref.replace(ixname,str(val)), {}, {})

# ------------------------------------------------------------------------

class Quantity(object):
    """Abstract class for symbolic numerical quantities."""

    typestr = None
    compatibleGens = ()
    targetLangs = ()

    def __init__(self, spec, name="", domain=None, specType=None):
        self.multiDefInfo = (False, "", "", 0, 0)   # initial, default value
        if isinstance(strIfSeq(spec), str):
            spec = strIfSeq(spec)
            if isNameToken(spec):
                # then we're declaring a single symbol
                if isNameToken(name):
                    # spec symbol is being renamed to name
                    token = name.strip()
                    actual_spec = QuantSpec(token, spec, specType)
                elif name=="":
                    # spec and name are to be the same
                    token = spec.strip()
                    actual_spec = QuantSpec(token, "", specType)
                else:
                    raise ValueError("name argument contains illegal symbols")
            elif isNameToken(name):
                # declaring a symbol and its non-trivial definition in spec
                token = name.strip()
                actual_spec = QuantSpec(token, spec, specType)
            elif isMultiDef(name):
                # declaring multiple symbols at once, with their definition
                # in spec
                if self.typestr == 'auxfn':
                    raise TypeError("Multiple quantity definition is invalid "
                                    "for function definitions")
                mdef, actual_spec, actual_freeSymbols = processMultiDef(name,
                                                                        spec,
                                                                    specType)
                # z[i,1,10] --> rootname = 'z', ixname = 'i', i1=1, i2=10
                rootname = mdef[0]
                ixname = mdef[1]
                i1 = mdef[2]
                i2 = mdef[3]
                token = name
                self.multiDefInfo = (True, rootname, ixname, i1, i2)
            else:
                print("Token name argument was:%s" % name)
                raise ValueError("Invalid Quantity token name")
        elif isinstance(spec, QuantSpec):
            actual_spec = copy(spec)
            if spec.subjectToken == "__result__" or name != "":
                if isNameToken(name):
                    token = name.strip()
                    actual_spec.subjectToken = token
                elif isMultiDef(name):
                    # declaring multiple symbols at once, with their definition
                    # in spec
                    if self.typestr == 'auxfn':
                        raise TypeError("Multiple quantity definition is invalid "
                                        "for function definitions")
                    if specType is not None:
                        actual_specType = specType
                    else:
                        actual_specType = spec.specType
                    mdef, actual_spec, actual_freeSymbols = \
                           processMultiDef(name, spec.specStr, actual_specType)
                    # z[i,1,10] --> rootname = 'z', ixname = 'i', i1=1, i2=10
                    rootname = mdef[0]
                    ixname = mdef[1]
                    i1 = mdef[2]
                    i2 = mdef[3]
                    token = name
                    self.multiDefInfo = (True, rootname, ixname, i1, i2)
                else:
                    print("Token name argument was:%s" % name)
                    raise TypeError("Invalid type for Quantity token name")
            else:
                token = spec.subjectToken
            if specType is not None:
                # override default specType from QuantSpec in case
                # e.g. it was made from connecting RHSfuncSpec with
                # something --> RHSfuncSpec, but user wants this Quantity
                # to be a different type
                actual_spec.specType = specType
        elif isinstance(spec, (Par, Var)):
            # referring to a parameter or variable name as a new variable
            if isNameToken(name):
                token = name.strip()
            else:
                raise ValueError("Invalid name '%s' for this Quantity"%name)
            actual_spec = QuantSpec(token, spec.name)
            if specType is not None:
                # override default specType from QuantSpec in case
                # e.g. it was made from connecting RHSfuncSpec with
                # something --> RHSfuncSpec, but user wants this Quantity
                # to be a different type
                actual_spec.specType = specType
        elif isinstance(spec, _num_types):
            # convert numeric literals
            token = name.strip()
            actual_spec = QuantSpec(token, repr(spec), specType)
        else:
            raise TypeError("Invalid spec argument")
        if token == 'for':
            raise ValueError("Cannot use reserved macro name 'for' as a "
                             "Quantity's name")
        else:
            self.name = token
        self.spec = actual_spec
        self.specType = self.spec.specType
        if self.multiDefInfo[0]:
            # must override spec's free symbol list with doctored one
            self.freeSymbols = remain(actual_freeSymbols, self.name)
#            self.usedSymbols = actual_usedSymbols
        else:
            self.freeSymbols = remain(self.spec.freeSymbols, self.name)
        self.usedSymbols = self.spec.usedSymbols
        # various ways to specify domain ...
        if domain is None:
            # assume continuous-valued from -Inf to Inf
            self.domain = (float, Continuous, [-Inf, Inf])
        else:
            self.setDomain(domain)
        try:
            # assume max of rank 2
            self.dim = getdim(self.spec.specStr)  # for vector
        except IndexError:
            # spec not long enough therefore self cannot be a vector
            self.dim = 0
        self.funcSpecDict = {}
        self.validate()


    def setDomain(self, domain):
        if isinstance(domain, list):
            # assume continuous-valued float
            assert len(domain)==2, "Domain-as-list must have length 2"
            d = (float, Continuous, domain)
        elif isinstance(domain, tuple):
            # full spec is a 3-tuple
            assert len(domain)==3, "Domain-as-tuple must have length 3"
            assert compareNumTypes(domain[0], _num_types), \
                   "Invalid domain numeric type"
            assert domain[1] in [Continuous, Discrete], \
                            "Invalid domain type"
            if compareNumTypes(domain[0], _int_types):
                assert domain[1] is Discrete, "Domain type mismatch"
            d = domain
        else:
            raise TypeError("Invalid domain argument: "+type(domain))
        self.domain = d

    def redefine(self, specStr):
        self.spec.redefine(specStr)

    def isspecdefined(self):
        return self.spec.specStr.strip() != ''


    def __contains__(self, tok):
        return tok in self.spec.parser.tokenized


    def __call__(self):
        return copy(self.spec)

    def __len__(self):
        return self.dim

    def isvector(self):
        """Is the quantity spec of the form [a,b,c]"""
        try:
            # minimum check necessary at this point to determine this
            return '['==self.spec.specStr[0] and ']'==self.spec.specStr[-1]
        except IndexError:
            return False


    def fromvector(self, ix=None):
        """Turn the Quantity's specification of a vector to a list of
        the element's specifications."""
        if self.isvector():
            if self.dim > 0:
                arglist = splitargs(self.spec.specStr[1:-1], ['(','['], [')',']'])
                res = [QuantSpec(self.name+'_%i'%i,
                                 arglist[i], self.specType) \
                        for i in range(self.dim)]
            else:
                res = [QuantSpec(self.name+"_0", self.spec.specStr[1:-1],
                                     self.specType)]
            if ix is None:
                return res
            else:
                try:
                    return res[ix]
                except IndexError:
                    raise IndexError("Index out of range in vector Quantity")
        else:
            raise TypeError("This Quantity does not specify a vector")


    def __setitem__(self, x, v):
        raise NotImplementedError

    def __delitem__(self, x):
        raise NotImplementedError

    def __getitem__(self, x):
        # kludge! This is overloaded.
        # Self may be either a multiRef or for a vector
        if self.multiDefInfo[0]:
            m = self.multiDefInfo
            rootname = m[1]
            ixname = m[2]
            i1 = m[3]
            i2 = m[4]
            if x in range(i1,i2+1):
                tokenized = []
                for t in self.spec.parser.tokenized:
                    # can trust that t starting with '[' is a valid mref
                    # because it was checked at initialization
                    if t[0] == '[':
                        # append evaluated integer without brackets
                        tokenized.append(str(evalMultiRefToken(t[1:-1],
                                                               ixname, x)))
                    elif t == ixname:
                        tokenized.append(str(x))
                    else:
                        tokenized.append(t)
                # make the appropriate sub-type of Quantity to return
                q = Quantity.__new__(self.__class__)
                q.__init__("".join(tokenized), rootname + str(x),
                           specType=self.specType)
                return q
            else:
                raise IndexError("Index to multiple Quantity definition out "
                                 "of the valid range [%i,%i]"%(i1,i2))
        elif self.isvector():
            return self.fromvector(x)
        else:
            raise IndexError("Quantity %s does not contain multiple "
                     "definitions and is not a vector of symbols"%self.name)

    def tonumeric(self):
        if self.spec.freeSymbols == []:
            if self.isvector():
                try:
                    return array(eval(self.spec.specStr, {}, {}))
                except:
                    raise TypeError("Cannot convert this Quantity to a numeric type")
            else:
                try:
                    return eval(self.spec.specStr, {}, {})
                except:
                    raise TypeError("Cannot convert this Quantity to a numeric type")
        else:
            raise TypeError("Cannot convert this Quantity to a numeric type")


    # -------------- method group for spec string assembly
    def __combine(self, other, opstr, reverseorder=0):
        if isinstance(other, QuantSpec):
            other_specType = other.specType
            otherstr = other()
        elif isinstance(other, Quantity):
            other_specType = other.specType
            otherstr = other.name
        elif isinstance(other, str):
            other_specType = 'ExpFuncSpec'
            otherstr = other
        elif isinstance(other, _num_types):
            otherstr = str(other)
            other_specType = 'ExpFuncSpec'
        else:
            raise TypeError("Invalid types to combine")
        try:
            r1 = QuantSpec("__result__", otherstr, other_specType)
        except TypeError:
            print("Failed to compile: '" + otherstr + "'")
            raise
        r2 = QuantSpec("__result__", self.name, self.specType)
        if reverseorder:
            return eval('r1' + opstr + 'r2')
        else:
            return eval('r2' + opstr + 'r1')

    def __add__(self, other):
        return self.__combine(other, "+")

    def __mul__(self, other):
        return self.__combine(other, "*")

    def __sub__(self, other):
        return self.__combine(other, "-")

    def __div__(self, other):
        return self.__combine(other, "/")

    def __radd__(self, other):
        return self.__combine(other, "+", 1)

    def __rmul__(self, other):
        return self.__combine(other, "*", 1)

    def __rsub__(self, other):
        return self.__combine(other, "-", 1)

    def __rdiv__(self, other):
        return self.__combine(other, "/", 1)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __pow__(self, other):
        dummy = QuantSpec('__result__', self.name)
        return dummy.__pow__(other)

    def __rpow__(self, other):
        dummy = QuantSpec('__result__', self.name)
        return dummy.__pow__(other, 1)

    def __neg__(self):
        return QuantSpec("__result__", "-" + self.name, self.specType)

    def __pos__(self):
        return QuantSpec("__result__", self.name, self.specType)

    # -------------- end method group

    def mapNames(self, nameMap):
        if self.name in nameMap:
            new_name = nameMap[self.name]
            if self.multiDefInfo[0]:
                # then add non-definition version of name to dictionary,
                # i.e. add z[i] to dictionary containing z[i,0,4]
                rootname = self.name.split('[')[0]
                ix = self.multiDefInfo[2]
                parentname = new_name.split(NAMESEP)[0]
                nameMap[rootname+'['+ix+']'] = \
                                 parentname+NAMESEP+rootname+'['+ix+']'
            self.name = new_name
        self.spec.mapNames(nameMap)
        self.spec._check_self_ref()
        self.freeSymbols = remain(self.spec.freeSymbols, self.name)
        self.usedSymbols = self.spec.usedSymbols


    def compileFuncSpec(self):
        raise NotImplementedError("Only call this method on a concrete sub-class")


    def simplify(self):
        qtemp = self.spec._eval(0)
        self.spec = qtemp
        self.freeSymbols = remain(self.spec.freeSymbols, self.name)
        self.usedSymbols = self.spec.usedSymbols


    def renderForCode(self):
        """Return code-ready version of spec (e.g. uncapitalize built-in
        function names)"""
        quant = copy(self)
        quant.spec = self.spec.renderForCode() #._eval(0, {})
        quant.freeSymbols = remain(quant.spec.freeSymbols, quant.name)
        quant.usedSymbols = quant.spec.usedSymbols
        return quant

    def rename(self, new_name):
        self.mapNames({self.name: new_name})

    def eval(self, *scopearg, **defs):
        """eval does not require all free symbols to be resolved, but any
        signature arguments (for a function) must be resolved."""
        arglist = list(defs.keys())
        if hasattr(self, 'signature'):
            if not alltrue([s in arglist for s in self.signature]):
                raise ValueError("All function signature arguments are "
                                 "necessary for evaluation")
        return self.spec._eval(1, *scopearg, **defs)


    def validate(self):
        if not isinstance(self.domain, tuple):
            raise TypeError("Expected 3-tuple for domain of token '" \
                                + self.name + "'")
        if len(self.domain) != 3:
            raise TypeError("Expected 3-tuple for domain of token '" \
                                + self.name + "'")
        if not isinstance(self.spec, QuantSpec):
            raise TypeError("spec field not of type QuantSpec")
        if self.spec.subjectToken != self.name:
            # only raise error if it's not a multiple quantity definition,
            # for which these aren't going to match
            if not isMultiDef(self.name):
                raise ValueError("spec target mismatch for token '" \
                             + self.name + "' vs. '" + self.spec.subjectToken \
                             +"'")
        assert isinstance(self.spec.specStr, str), "specStr invalid"
        assert self.spec.specType in specTypes, "specType invalid"
        bad_tls = remain(self.targetLangs, targetLangs)
        if len(bad_tls)>0:
            print("Invalid target languages:%s" % bad_tls)
            raise ValueError("Invalid target language specified")
        if self.typestr != 'var' and self.specType != 'ExpFuncSpec':
            raise TypeError("Invalid spec type specified for %s"%className(self))
        # !! check that compatibleGens point to known Generator types !!
        # if get this far without raising exception, then valid


    def isDefined(self, verbose=False):
        self.validate()
        if not self.spec.isDefined(verbose) and self.typestr != 'par':
            if verbose:
                print("'%s' ill defined: bound spec string is required"%self.name)
            return False
        if len(self.compatibleGens) == 0:
            if verbose:
                print("'%s' ill defined: compatibleGens is empty"%self.name)
            return False
        return True

# bindSpec is broken until it knows how to deal with multi-quantity defs/refs
#    def bindSpec(self, spec):
#        if isinstance(spec, QuantSpec):
#            if spec.subjectToken != self.name:
#                raise ValueError(self.typestr + " name '" + self.name + "'" \
#                             + " does not match '" + spec.subjectToken + "'")
#            self.spec = spec
#        else:
#            raise TypeError("Invalid spec argument")
#        self.freeSymbols = self.findFreeSymbols()??
#        self.specType = spec.specType
#        self.validate()

    def __repr__(self):
        return "%s %s (%s)"%(className(self),self.name,self.specType)

    def __str__(self):
        return self.name

    def info(self, verboselevel=1):
        if verboselevel > 0:
            # info is defined in utils.py
            utils_info(self.__dict__, "ModelSpec " + self.name,
                 recurseDepthLimit=1+verboselevel)
        else:
            print(self.__repr__())

    def __hash__(self):
        h = (hash(type(self)), hash(self.name), hash(str(self.spec)),
               hash(str(self.domain[0])), hash(str(self.domain[1])),
               hash(str(self.domain[2])))
        return int(sum(h))

    def __eq__(self, other, diff=False):
        results = []
        try:
            results.append(type(self) == type(other))
            results.append(self.name == other.name)
            results.extend([str(self.domain[i]) == str(other.domain[i]) \
                           for i in [0,1,2]])
            results.append(self.spec == other.spec)
        except AttributeError as e:
            if diff:
                print("Type:%s" % className(self) + results)
                print("  " + e)
            return False
        if diff:
            print("Type:%s" % className(self), results)
        return alltrue(results)

    def __ne__(self, other):
        return not self.__eq__(other)

    def difference(self, other):
        """Show differences in comparison to another Quantity."""
        self.__eq__(other, diff=True)

#    def compileFuncSpec(self):
#        self.funcSpecDict = {self.typestr+'s': [deepcopy(self)]}

    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

# ------------------------------------------------------------------------

class Par(Quantity):
    typestr = 'par'


class Var(Quantity):
    typestr = 'var'


class Input(Quantity):
    typestr = 'input'


class Fun(Quantity):
    typestr = 'auxfn'

    def __init__(self, spec, signature, name="", domain=None, specType=None):
        if not isinstance(signature, _seq_types):
            raise TypeError("signature must be a list/tuple of strings")
        sigstrs = []  # allows signature to be made up of Var names, etc.
        for s in signature:
            # signature may be empty, for function call fn()
            sigstrs.append(str(s))
            assert isNameToken(str(s)), "signature symbol %s is invalid"%s
        self.signature = sigstrs
        Quantity.__init__(self, spec, name, domain, specType)
        self.freeSymbols = remain(self.freeSymbols, self.signature)
##        if remain(self.signature, self.spec.parser.tokenized) != []:
##            raise ValueError("All signature arguments were not used in function definition")
        if self.spec.isDefined():
            protected_auxnamesDB.addAuxFn(self.name, self.spec.parser)

#    def bindSpec(self, spec):
#        Quantity.bindSpec(self, spec)
#        protected_auxnamesDB.addAuxFn(self.name, self.spec.parser,
#                                      overwrite=True)

    def __call__(self, *args):
        lensig = len(self.signature)
        if lensig != len(args):
            raise ValueError("Invalid number of arguments for auxiliary function")
        argstr = ", ".join(map(str,args))
        if self.spec.specStr == '':
            return QuantSpec("__result__", self.name+'('+argstr+')', specType=self.specType)
        else:
            # avoid name clashes for free names and bound arg names during substitution
            argNameMapInv = {}
            args = [copy(a) for a in args]
            for i in range(len(args)):
                if isinstance(args[i], (Quantity, QuantSpec)):
                    if args[i].freeSymbols == []:
                        try:
                            check_names = [args[i].name]
                        except AttributeError:
                            check_names = [args[i].subjectToken]
                    else:
                        check_names = args[i].freeSymbols
                    #check_names = args[i].freeSymbols
                    isection = intersect(check_names, self.signature)
                    for name in isection:
                        # dummy arg name
                        newname = "__"+name+"__"
                        while newname in list(argNameMapInv.keys()) + \
                                self.signature+self.freeSymbols:
                            # ensure unique name
                            newname += "_"
                        argNameMapInv[newname] = name
                        args[i].mapNames({name:newname})
                elif isinstance(args[i], str):
                    qtemp = QuantSpec("__arg__", args[i])
                    isection = intersect(qtemp.freeSymbols, self.signature)
                    for name in isection:
                        # dummy arg name
                        newname = "__"+name+"__"
                        while newname in list(argNameMapInv.keys()) + \
                                self.signature+self.freeSymbols:
                            # ensure unique name
                            newname += "_"
                        argNameMapInv[newname] = name
                        qtemp.mapNames({name:newname})
                    args[i] = str(qtemp)
            res = self.spec._eval(1, **dict(zip(self.signature,args)))
            res.mapNames(argNameMapInv)
            return res


# -------------------------------------------------------------------------
### Finish up overrides for inbuilt math names -> TitleCase symbolic versions

# assign math constants to QuantSpecs that hold their names
for c in remain(allmathnames,funcnames):
    six.exec_(str(c).title() + " = QuantSpec('"+str(c)+"', '', includeProtected=False)")
    math_globals[str(c).title()] = eval(str(c).title())

# references to builtin aux functions or macros require temp defs of those
# as Fun objects
local_fndef = {}
for fname in builtinFnSigInfo.keys():
    # function will not have to return anything in order to be eval'd
    # but need a return value. we use '0'.
    # final underscore is to avoid name clash with actual Python names
    # when the strings are "eval"'d locally, e.g. "if", "for".
    argstr = ""
    # add signature
    for i in range(builtinFnSigInfo[fname]):
        argstr += "'arg"+str(i)+"',"
    funcstr = "Fun('', [%s], '%s')"%(argstr,fname+"_")
    local_fndef[fname+"_"] = eval(funcstr)


# --------------------------------------------------------------------------
### Symbolic differentiation

##__Diff_saved = {}
##
##def loadDiffs(filename):
##    global __Diff_saved
##    try:
##        __Diff_saved = cPickle.load(file(filename, 'rb'))
##    except IOError:
##        pass
##
##def saveDiffs(filename):
##    f = file(filename, 'wb')
##    cPickle.dump(__Diff_saved, f, 2)
##    f.close()


def getlftrtD(t,a):
    lft=t[1]
    rt=t[3:]
    if len(rt)>1:
        rt=[t[0]]+rt
        drt=DiffStr(rt,a)
        if drt[0] not in ['NUMBER']:
            drt=['atom',['LPAR','('],drt,['RPAR',')']]
    else:
        rt=rt[0]
        drt=DiffStr(rt,a)
    dlft=DiffStr(lft,a)
    return lft,dlft,rt,drt


def dofun(l,r):
    retr=ensurebare(r)
    if l=='log':
        if r in ONES: return '0'
        if r=='e': return '1'
    if l=='log10':
        if r in ONES: return '0'
        if r in TENS: return '1'
    if l=='log_0': return dodiv('1',r)
    if l=='log10_0': return dodiv(dofun('log','10'),retr)
    if l=='ln_0': return dodiv('1',r)
    if l=='sin_0': return dofun('cos',retr)
    if l=='sinh_0': return dofun('cosh',retr)
    if l=='cosh_0': return dofun('sinh',retr)
    if l=='tanh_0': return dodiv('1',dopower(dofun('cosh',retr),'2'))
    if l=='coth_0': return dodiv('-1',dopower(dofun('sinh',retr),'2'))
    if l=='asin_0': return dodiv('1',dofun('sqrt',dosub('1',dopower(r,'2'))))
    if l=='acos_0': return dodiv('-1',dofun('sqrt',dosub('1',dopower(r,'2'))))
    if l=='atan_0': return dodiv('1',doadd('1',dopower(r,'2')))
    if l=='acot_0': return dodiv('-1',doadd('1',dopower(r,'2')))
    if l=='tan_0': return dodiv('1',dopower(dofun('cos',retr),'2'))
    if l=='cot_0': return doneg(dodiv('1',dopower(dofun('sin',retr),'2')))
    if l=='cos_0': return doneg(dofun('sin',retr))
    if l=='exp_0': return dofun('exp',retr)
    if l=='sqrt_0': return dodiv('1',domul('2',dofun('sqrt',retr)))
    return '%s(%s)'%(l,retr)




# --------------------------------------------------------------------------

qtypes = (Quantity, QuantSpec)

def Diff(t, a):
    """Symbolic differentiation of argument t with respect to variables in a.

    Diff expects these arguments to be strings, Quantity, or QuantSpec types,
    a mixture of these, or a list of strings.

    The return type is a QuantSpec. To get the string version, either use DiffStr or
    call the method q.renderForCode().specStr on the returned q object.
    """

    # deal with t -----------------------------------------
    if compareClassAndBases(t, qtypes):
        if t.isvector():
            return Diff(t.fromvector(), a)
        qt = copy(t)
        qt.mapNames(feval_map_symb)
        if isinstance(t, Fun):
            # shorthand so that function f can be diff'd without having to
            # specify f(x,y,z) (i.e. without signature)
            t_arg = str(qt(*([str(qarg) for qarg in qt.signature])))
        else:
            try:
                # extract symbol definition, if any
                t_arg = qt.spec.specStr
            except AttributeError:
                t_arg = str(qt)
            else:
                if t_arg == '':
                    # use symbol name
                    t_arg = str(qt)
    elif isinstance(t, str):
        t_strip = t.strip()
        if t_strip[0] == "[" and t_strip[-1] == "]":
            return Diff(splitargs(t_strip[1:-1]), a)
        else:
            qt = QuantSpec('qt', t, treatMultiRefs=False)
            qt.mapNames(feval_map_symb)
            t_arg = str(qt)
    elif compareClassAndBases(t, (list, tuple)):
        ostr = "[" + ",".join([str(Diff(o,a)) for o in t]) + "]"
        return QuantSpec("__result__", ostr)
    else:
        print("Found:%s type (%s)" % (t, type(t)))
        raise TypeError("Invalid type for t")
    # deal with a -----------------------------------------
    if compareClassAndBases(a, qtypes):
        a_arg = str(a)
    elif isinstance(a, list):
        ostr = "[" + ",".join([str(Diff(t,o)) for o in a]) + "]"
        return QuantSpec("__result__", ostr)
    elif isinstance(a, str):
        a_arg = a
    else:
        print("Found: type (%s)" % (a, type(a)))
        raise TypeError("Invalid type for a")
    # if didn't already recurse then fully reduced to strings
    # ... attempt to simplify string using eval() method call
    try:
        res = QuantSpec("__result__", DiffStr(t_arg, a_arg))
        res.mapNames(inverseMathNameMap)
        res.simplify()
        return res
    except:
        print("Tried to diff '%s' w.r.t. '%s'"%(t_arg,a_arg))
        raise


def DiffStr(t, a='x'):
    """Diff for strings only.

    This is Pearu's original Diff function renamed to DiffStr."""

    if isinstance(a, list):
##        s= ''.join([t, '__derivWRT__', str(a)])
##        r=__Diff_saved.get(s, None)
##        if r is not None:
##            return r

        o = string2ast(t)
        for aa in a:
            o = DiffStr(o,aa)
##        res=ast2string(o)
##        __Diff_saved[s] = res
##        return res
        return ast2string(o)
    elif isinstance(t, str):
        return ast2string(DiffStr(string2ast(t),a))
##        s= ''.join([t, '__derivWRT__', str(a)])
##        r=__Diff_saved.get(s, None)
##        if r is not None:
##            return r
##
##        res=ast2string(DiffStr(string2ast(t),a))
##        __Diff_saved[s] = res
##        return res

    if isinstance(t, list) and len(t) == 1:
        return DiffStr(t[0],a)
    if t[0]=='NUMBER': return [t[0],'0']
    if t[0]=='NAME':
        if t[1]==a: return ['NUMBER','1']
        return ['NUMBER','0']
    if t[0]=='factor':
        st=ast2string(t)
        return simplify(string2ast(doneg(ast2string(DiffStr(string2ast(st[1:]),a)))))
    if t[0]=='arith_expr':
        o='0'
        for i in range(1,len(t),2):
            if t[i-1][1]=='-':
                o=doadd(o,doneg(ast2string(DiffStr(t[i],a))))
            else:
                o=doadd(o,ast2string(DiffStr(t[i],a)))
        return simplify(string2ast(o))
    if t[0]=='term':
        ttemp = ensureparen_div(t)
        lft,dlft,rt,drt=[ast2string(simplify(x)) for x in getlftrtD(ttemp,a)]
        if ttemp[2][0]=='STAR':
            return simplify(string2ast(doadd(domul(dlft,rt),domul(lft,drt))))
        if ttemp[2][0]=='SLASH':
            return simplify(string2ast(dosub(dodiv(dlft,rt),
                                    domul(domul(lft,drt),dopower(rt,'-2')))))
    if t[0]=='xor_expr' and t[2]=='CIRCUMFLEX': # covers alternative power syntax
        alt = copy(t)
        alt[0] = 'power'; alt[2]=='DOUBLESTAR'
        return simplify(toCircumflexSyntax(string2ast(DiffStr(alt,a))))
    if t[0]=='power':
        # covers math functions like sin, cos, and log10 as well!
        if t[2][0]=='trailer':
            if len(t)>3:
                term1 = simplify(t[:3])
                if term1[0] in ['power', 'NUMBER', 'NAME']:
                    formatstr='%s'
                else:
                    formatstr='(%s)'
                return simplify(string2ast(DiffStr([t[0],
                                    string2ast(formatstr%ast2string(term1)),t[3:]],a)))
            elif len(t)==3 and t[1][1]=='pow':
                # 'pow' syntax case
                return simplify(toPowSyntax(string2ast(DiffStr(mapPowStr(t),a))))
            # else ignore and carry on
        ts=[];o=[];dts=[]
        for i in t[1:]:
            if i[0]=='DOUBLESTAR':
                if len(o)==1: o=o[0]
                ts.append(o);
                dts.append(DiffStr(o,a))
                o=[]
            else: o.append(i)
        if len(o)==1: o=o[0]
        ts.append(o)
        dts.append(DiffStr(o,a))
        if t[2][0]=='DOUBLESTAR':
            st,lft,dlft,rt,drt=[ast2string(simplify(x)) for x in [t,ts[0],dts[0],ts[1],dts[1]]]
            rt1=trysimple('%s-1'%rt)
            if drt=='0':
                return toPowSyntax(string2ast(domul(domul(rt,dlft),dopower(lft,rt1))))
            return toPowSyntax(string2ast(domul(doadd(domul(dofun('log',lft),drt),
                                          dodiv(domul(dlft,rt),lft)),st)))
        if t[2][0]=='trailer':
            return simplify(toPowSyntax(dts[0]))
    if t[0] in ['arglist','testlist']:
        o=[]
        for i in t[1::2]:
            o.append(DiffStr(ast2string(i),a))
        return string2ast(','.join(o))
    if t[0]=='atom':
##        if t[1][0]=='LPAR' and t[-1][0]=='RPAR':
##            return simplify(DiffStr(t[2:-1],a))
##        else:
        return string2ast(ensureparen(ast2string(DiffStr(t[2:-1],a))))
    if t[1][0]=='trailer': # t=[[NAME,f],[trailer,[(],[ll],[)]]]
        aa=t[1][2]
        ll=splitargs(ast2string(DiffStr(aa,a)))
        ii=0;o='0'
        for i in ll:
            o=doadd(o,domul(i,dofun(t[0][1]+'_'+repr(ii),ast2string(aa))))
            ii=ii+1
        return simplify(string2ast(o))
    return t
