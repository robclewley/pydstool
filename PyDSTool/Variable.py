"""Variable is a one-dimensional discrete and continuous real variable class.

   Robert Clewley, July 2005
"""

# ----------------------------------------------------------------------------

# PyDSTool imports
from .utils import *
from .common import *
from .errors import *
from .Points import *
from .Interval import *
from .FuncSpec import ImpFuncSpec

from numpy import isfinite, sometrue, alltrue, any, all, \
     array, float64, int32, ndarray, asarray
import numpy as np
import copy
import types, math, random

__all__ = ['Variable', 'HybridVariable',
           'OutputFn', 'isinputcts', 'isinputdiscrete',
           'isoutputcts', 'isoutputdiscrete',
           'iscontinuous', 'isdiscrete',
           'numeric_to_vars', 'pointset_to_vars']

# ------------------------------------------------------------------


class VarDiagnostics(Diagnostics):
    def getWarnings(self):
        if self.warnings != []:
            output = "Warnings:"
            for (i,d) in self.warnings:
                if d is None:
                    output += "Independent variable value %s was out of "% i + \
                          "bounds"
                else:
                    output += "Dependent variable value %s was out of " % s + \
                          "bounds at independent variable value %s" % i
        else:
            output = ''
        return output


def pointset_to_vars(pts, discrete=True):
    """Utility to convert Pointset to a dictionary of Variables.
    If discrete option set to False (default is True) then the
    Variables will be linearly interpolated within their domain.

    Any labels in the pointset will be preserved in the Variables
    in case of their re-extraction using the getDataPoints method.
    """
    coordnames = pts.coordnames
    vals = pts.coordarray
    all_types_float = pts.coordtype == float
    if isparameterized(pts):
        indepvar = pts.indepvararray
        indepvarname = pts.indepvarname
        if discrete:
            indepvartype = int
        else:
            indepvartype = float
        indepdomain = Interval(pts.indepvarname, indepvartype,
                               extent(pts.indepvararray),
                               abseps=pts._abseps)
    else:
        indepvar = None
        indepvarname = None
        indepdomain = None
    return numeric_to_vars(vals, coordnames, indepvar, indepvarname,
                           indepdomain, all_types_float, discrete,
                           pts._abseps, pts.labels)


def numeric_to_vars(vals, coordnames, indepvar=None, indepvarname='t',
                    indepdomain=None, all_types_float=True, discrete=True,
                    abseps=None, labels=None):
    """Utility to convert numeric types to a dictionary of Variables.
    If discrete option set to True (default is False) then the
    Variables will be linearly interpolated within their domain.
    """
    if isinstance(coordnames, str):
        coordnames = [coordnames]
    if isinstance(vals, _num_types):
        vals = [[vals]]
    vars = {}
    if indepvar is None:
        for i, c in enumerate(coordnames):
            if all_types_float:
                vartype = float
            else:
                vartype = array(vals[i]).dtype.type
            if discrete:
                vars[c] = Variable(outputdata=Pointset({'coordnames': c,
                                                    'coordarray': vals[i],
                                                    'coordtype': vartype}),
                                                    name=c, abseps=abseps,
                                                    labels=labels)
            else:
                raise AssertionError("Cannot use continuously defined "
                                     "option without an independent variable")
        return vars
    else:
        if isinstance(indepvar, _num_types):
            indepvartype = type(indepvar)
            indepvar = [indepvar]
        else:
            indepvartype = asarray(indepvar).dtype.type
        if indepdomain is None:
            indepdomain = indepvarname
        else:
            if isinstance(indepdomain, Interval):
                assert indepvarname == indepdomain.name, "Indep varname mismatch"
            else:
                if discrete:
                    var_type = int
                else:
                    var_type = float
                indepdomain = Interval(indepvarname, var_type, indepdomain)
        for i, c in enumerate(coordnames):
            if all_types_float:
                vartype = float
            else:
                vartype = array(vals[i]).dtype.type
            if discrete:
                vars[c] = Variable(outputdata=Pointset({'coordnames': c,
                                                'coordarray': vals[i],
                                                'coordtype': vartype,
                                                'indepvarname': indepvarname,
                                                'indepvararray': indepvar,
                                                'indepvartype': indepvartype}),
                                    indepdomain=indepdomain, name=c,
                                    abseps=abseps, labels=labels)
            else:
                dom_int = Interval(c, vartype, extent(vals[i]),
                                   abseps=abseps)
                vars[c] = Variable(outputdata=interp1d(indepvar, vals[i]),
                                   indepdomain=indepdomain,
                                   depdomain=dom_int, name=c,
                                   abseps=abseps, labels=labels)
    return vars


class Variable(object):
    """One-dimensional discrete and continuous real variable class.
    """

    def __init__(self, outputdata=None, indepdomain=None, depdomain=None,
                 name='noname', abseps=None, labels=None):
        # funcreg stores function data for dynamically created methods
        # to allow a Variable to be copied using pickling
        self._funcreg = {}
        if isinstance(name, str):
            # !!! name is probably redundant
            self.name = name
        else:
            raise TypeError("name argument must be a string")
        # defaults for empty 'placeholder' Variables used by ODEsystem
        if outputdata is None or isinstance(outputdata, (Pointset, interp1d)):
            if indepdomain is None:
                indepdomain = 't'
            if depdomain is None:
                depdomain = 'x'
        # set some initial values so that can test if what changed
        # after calling setOutput()
        self._vectorizable = True
        self.defined = False
        self.indepdomain = None
        self.indepvartype = None
        self.indepvarname = None
        self.depdomain = None
        self.coordtype = None
        self.coordname = None
        self._refvars = None   # for use with ExplicitFnGen
        # Ranges covered by the current trajectory held (if known)
        self.trajirange = None
        self.trajdrange = None
        # independent variable domain
        self.setIndepdomain(indepdomain, abseps)
        # used internally, especially for "input" variables
        self._internal_t_offset = 0
        # output function
        self.setOutput(outputdata, abseps)
        # dependent variable domain
        self.setDepdomain(depdomain, abseps)
        assert self.coordname != self.indepvarname, ("Independent variable "
                                "name and coordinate name must be different")
        self.diagnostics = VarDiagnostics()
        # labels is for internal use in case Variable data is from a Pointset
        # that uses labels. This preserves them for getDataPoints method to
        # restore them.
        self.labels = labels


    def is_continuous_valued(self):
        return isoutputcts(self)

    def is_discrete_valued(self):
        return not isoutputcts(self)


    # Auxiliary functions for user-defined code to call
    def _auxfn_globalindepvar(self, parsinps, t):
        return self.globalt0 + t


    def _auxfn_initcond(self, parsinps, varname):
        return self.initialconditions[varname]


    def _auxfn_heav(self, parsinps, x):
        if x>0:
            return 1
        else:
            return 0


    def _auxfn_if(self, parsinps, c, e1, e2):
        if c:
            return e1
        else:
            return e2


    def _auxfn_getindex(self, parsinps, varname):
        return self._var_namemap[varname]


    def addMethods(self, funcspec):
        """Add dynamically-created methods to Veriable object"""

        # Add the auxiliary function specs to this Variable's namespace
        for auxfnname in funcspec.auxfns:
            fninfo = funcspec.auxfns[auxfnname]
            if not hasattr(Variable, fninfo[1]):
                # user-defined auxiliary functions
                # (built-ins are provided explicitly)
                try:
                    exec(fninfo[0], globals())
                except:
                    print('Error in supplied auxiliary function code')
                    raise
                self._funcreg[fninfo[1]] = ('Variable', fninfo[0])
                setattr(Variable, fninfo[1], eval(fninfo[1]))
        # Add the spec function to this Variable's namespace
        fninfo_spec = funcspec.spec
        if not hasattr(Variable, fninfo_spec[1]):
            try:
                exec(fninfo_spec[0], globals())
            except:
                print('Error in supplied functional specification code')
                raise
            self._funcreg[fninfo_spec[1]] = ('Variable', fninfo_spec[0])
            setattr(Variable, fninfo_spec[1], eval(fninfo_spec[1]))
        # Add the auxiliary spec function (if present) to this Var's namespace
        if funcspec.auxspec:
            fninfo_auxspec = funcspec.auxspec
            if not hasattr(Variable, fninfo_auxspec[1]):
                try:
                    exec(fninfo_auxspec[0], globals())
                except:
                    print('Error in supplied auxiliary variable code')
                    raise
                self._funcreg[fninfo_auxspec[1]] = ('Variable', fninfo_auxspec[0])
                setattr(Variable, fninfo_auxspec[1], eval(fninfo_auxspec[1]))
        # For implicit functions
        if isinstance(funcspec, ImpFuncSpec):
            impfn_name = funcspec.algparams['impfn_name']
            if funcspec.algparams['jac']:
                jac_str = "fprime=funcspec.algparams['jac'],"
            else:
                jac_str = ""
            # Wrap spec fn like this as it has been set up as a
            # method, but want to call as regular function
            # *** ALSO *** spec fn has signature (ds, t, x, p)
            # but implicit function solvers expect
            # (x, t, p), so have to switch 1st and 2nd args here
            # after 'ds' filled with None
            if len(funcspec.vars) == 1:
                # dimension == 1, so have to make list output from spec
                # into a scalar
                # Also, scalar a1 needs to be put into list form for
                # acceptance as x in spec fn
                specfn_str = "lambda a1, a2, a3: " \
                  + fninfo_spec[1] \
                  + "(None, a2, [a1], a3)[0]"
            else:
                # for dimension > 1 a1 will already be an array / list
                specfn_str = "lambda a1, a2, a3: " \
                  + fninfo_spec[1] \
                  + "(None, a2, a1, a3)"
            this_scope = globals()   # WE CHANGE GLOBALS()
            this_scope.update({'funcspec': locals()['funcspec'],
                               'fninfo_spec': locals()['fninfo_spec']})
            impfn_str = impfn_name + \
                " = makeImplicitFunc(" + specfn_str + "," \
                + jac_str + """x0=funcspec.algparams['x0'],
                               extrafargs=(funcspec.algparams['pars'],),
                               xtolval=funcspec.algparams['atol'],
                               maxnumiter=funcspec.algparams['maxnumiter'],
                               solmethod=funcspec.algparams['solvemethod'],
                               standalone=False)"""
            try:
                exec(impfn_str, this_scope)
            except:
                print('Error in supplied implicit function code')
                raise
            # record special reference to the implicit fn,
            # as its a method of Variable (for delete method).
            self._funcreg['_impfn'] = (impfn_name, impfn_str)
            # In previous versions setattr was to self, not the Variable class
            setattr(Variable, impfn_name, eval(impfn_name))
            # clean up globals() afterwards
            del this_scope['funcspec']
            del this_scope['fninfo_spec']

    def getDataPoints(self):
        """Reveal underlying mesh and values at mesh points, provided
        Variable is based on a mesh (otherwise None is returned).
        The returned Pointset will be time-shifted according to the
        Variable's current _internal_t_offset attribute.

        Any pointset labels present when the variable was created will
        be restored.
        """
        if isinstance(self.output, VarCaller):
            return Pointset(indepvarname=self.indepvarname,
                            indepvararray=self.output.pts.indepvararray + self._internal_t_offset,
                            coordnames=[self.coordname],
                            coordarray=self.output.pts.coordarray[0],
                            labels=self.labels)
        elif hasattr(self.output, 'datapoints'):
            datapoints = self.output.datapoints
            return Pointset(indepvarname=self.indepvarname,
                            indepvararray=datapoints[0] + self._internal_t_offset,
                            coordnames=[self.coordname],
                            coordarray=datapoints[1],
                            labels=self.labels)
        else:
            return None

    def underlyingMesh(self):
        """Reveal underlying mesh as arrays, rather than Pointset
        as returned by getDataPoints method. If no underlying mesh is
        present, None is returned."""
        try:
            # works if .output is an interpclass instance
            mesh = self.output.datapoints
        except AttributeError:
            try:
                # works if .output is a VarCaller instance (with underlying Pointset)
                pts = self.output.pts
                mesh = array([pts.indepvararray, pts.coordarray[0]])
            except AttributeError:
                mesh = None
        return mesh

    def truncate_to_idx(self, idx):
        mesh = self.underlyingMesh()
        if mesh is None:
            raise RuntimeError("Cannot truncate a Variable without an underlying mesh by index")
        try:
            new_t_end = mesh[0][idx]
        except IndexError:
            raise ValueError("Truncation index %d out of range"%idx)
        except TypeError:
            raise TypeError("Index must be an integer")
        if isinstance(self.indepdomain, Interval):
            self.indepdomain.set([self.indepdomain[0], new_t_end])
        else:
            # ndarray type
            self.indepdomain = self.indepdomain[0:idx]
        # adjust depdomain for array type of dep domain
        # (nothing to change for Interval type)
        if isinstance(self.depdomain, ndarray):
            self.depdomain = self.depdomain[0:idx]
        # adjust trajirange and trajdrange
        self._setRanges(self.depdomain._abseps)

    def _setRanges(self, abseps=None):
        # set trajirange and trajdrange for the two types of Variable output method
        # that these are associated with (see method setOutput)
        try:
            output = self.output
        except AttributeError:
            # output not set or not a compatible type for trajirange and trajdrange
            return
        if isinstance(output, VarCaller):
            self.trajirange = Interval('traj_indep_bd',
                                       self.indepvartype,
                                       extent(output.pts.indepvararray),
                                   abseps=abseps)
            self.trajdrange = Interval('traj_dep_bd',
                                       self.coordtype,
                                       extent(output.pts.coordarray[0]),
                                   abseps=abseps)
        elif isinstance(output, (OutputFn, interpclass, type)):
            if hasattr(output, 'types'):
                deptype = output.types[0]
                indeptype = output.types[1]
            else:
                # default
                deptype = indeptype = float
            if isinstance(output.datapoints[0], Interval):
                assert compareNumTypes(output.types[0], \
                       output.datapoints[0].type), \
                       "Inconsistent type with Interval bounds"
                self.trajirange = output.datapoints[0]
            else:
                self.trajirange = Interval('traj_indep_bd', indeptype,
                                          extent(output.datapoints[0]),
                                      abseps=abseps)
            if isinstance(output.datapoints[1], Interval):
                assert compareNumTypes(output.types[1], \
                       output.datapoints[1].type), \
                       "Inconsistent type with Interval bounds"
                self.trajdrange = output.datapoints[1]
            else:
                self.trajdrange = Interval('traj_dep_bd', deptype,
                                          extent(output.datapoints[1]),
                                      abseps=abseps)

    def setOutput(self, outputdata, funcspec=None, globalt0=0,
                  var_namemap=None, ics=None, refvars=None, abseps=None):
        """Dynamically create 'output' method of Variable"""

        self.globalt0 = globalt0
        if type(outputdata) in [types.FunctionType,
                                types.BuiltinFunctionType,
                                types.MethodType]:
            # Variable generated from function, given in closed form
            self.output = outputdata
            assert ics is None, "Invalid option for this type of output"
            if outputdata != noneFn:
                self.defined = True
        elif isinstance(outputdata, tuple):
            # For ExplicitFnGen or ImplicitFnGen types, whose functional forms
            # may need to access these at call time.
            assert len(outputdata) == 2, "Incorrect size of outputdata tuple"
            if funcspec is not None:
                self.addMethods(funcspec)
                self._var_namemap = var_namemap
                self._funcreg['funcspec'] = (None, funcspec)
            else:
                raise ValueError('funcspec missing in setOutput')
            # Add the specific mapping functions for Ex/ImplicitFnGen objects
            try:
                exec(outputdata[1], globals())
            except:
                print('Internal Error in _mapspecfn code')
                raise
            has_op = hasattr(self, 'output')
            # have to define this function in here because use of lambda
            # won't allow me to pickle the Variable object
            if not has_op or (has_op and self.output is noneFn):
                def wrap_output(arg):
                    return eval(outputdata[0])(self, arg)
                setattr(self, 'output', wrap_output)
            self._funcreg['outputdata'] = (None, outputdata)
            t0 = self.indepdomain[0]
            if ics is None and not isinstance(funcspec, ImpFuncSpec):
                try:
                    self.initialconditions = {self.coordname: self.output(t0)}
                except ValueError:
                    self.initialconditions = {self.coordname: np.NaN}
                except TypeError:
                    print("Debugging info: self.output = %s" % self.output)
                    raise
            else:
                self.initialconditions = ics
            self._vectorizable = False
            self._refvars = refvars
            self.defined = True
        elif isinstance(outputdata, (OutputFn, interpclass, type)):
            # Variable generated by callable object that generates values over
            # mesh points that it holds, e.g. by interpolation
            # (InstanceType and TypeType are for backwards compatibility, e.g.
            # for old SciPy interpolate code that uses Classic Classes)
            assert ics is None, "Invalid option for this type of output"
            assert '__call__' in dir(outputdata), "Must provide callable object"
            self.output = outputdata
            if hasattr(outputdata, 'datapoints'):
                self._setRanges(abseps)
            self.defined = True
        elif isinstance(outputdata, Pointset):
            # Variable generated from a pointset (without interpolation)
            assert ics is None, "Invalid option for this type of output"
            assert isparameterized(outputdata), ("Must only pass parameterized"
                                                 " pointsets")
            if outputdata.dimension == 1:
                self.coordname = copy.copy(outputdata.coordnames[0])
                self.indepvarname = outputdata.indepvarname
                self.output = VarCaller(outputdata)
                self.coordtype = outputdata.coordtype
                self.indepvartype = outputdata.indepvartype
                if self.indepdomain is not None:
                    for v in outputdata[self.indepvarname]:
                        if not v in self.indepdomain:
                            raise ValueError("New Pointset data violates "
                               "independent variable domain already specified")
                if self.depdomain is not None:
                    for v in outputdata[self.coordname]:
                        if not v in self.depdomain:
                            raise ValueError("New Pointset data violates "
                               "dependent variable domain already specified")
                self._setRanges(abseps)
                self.defined = True
            else:
                raise ValueError("Pointset data must be 1D to create a "
                                 "Variable")
        elif outputdata is None:
            # placeholder for an unknown output type
            assert ics is None, "Invalid option when outputdata argument is None"
            self.output = noneFn
            self.defined = False
        else:
            raise TypeError("Invalid type for data argument: " \
                              +str(type(outputdata)))


    def setIndepdomain(self, indepdomain, abseps=None):
        if isinstance(indepdomain, str):
            self.indepvarname = indepdomain
            if self.indepdomain is not None:
                # If indepdomain already set and indepvarname is none then
                # name won't get put in place unless we force it here
                self.indepvarname = indepdomain
                self.indepdomain.name = indepdomain
            else:
                self.indepdomain = Interval(self.indepvarname, float,
                                           [-np.Inf, np.Inf], abseps=abseps)
            self.indepvartype = float
        else:
            if isinstance(indepdomain, Interval):
                if self.trajirange:
                    if indepdomain.contains(self.trajirange) is notcontained:
                        raise ValueError("Cannot set independent variable"
                                         " domain inside current trajectory's"
                                         " range")
                self.indepdomain = indepdomain
                self.indepvarname = indepdomain.name
                self.indepvartype = _num_name2type[indepdomain.typestr]
            elif isinstance(indepdomain, dict):
                # enumerated discrete domains
                assert len(indepdomain) == 1, "Independent variable " \
                                         "dictionary must have only 1 entry"
                d = list(indepdomain.values())[0]
                assert all(isfinite(d)), "Independent variable values must be" \
                                         " finite"
                if self.trajirange:
                    assert self.trajirange[0] in d
                    assert self.trajirange[1] in d
                self.indepvarname = list(indepdomain.keys())[0]
                if isinstance(d, (list, tuple)):
                    if self.coordtype is not None:
                        self.indepdomain = array(d, self.coordtype)
                    else:
                        self.indepdomain = array(d)
                elif isinstance(d, ndarray):
                    da = array(d)
                    if self.indepvartype is not None and \
                       self.indepvartype != da.dtype.type:
                        raise TypeError("Mismatch between type of indepdomain "
                                          "argument and Pointset data")
                    else:
                        self.indepdomain = da
                else:
                    raise TypeError("Invalid type for independent "
                                      "variable domain")
                # assert this after self.indepdomain has been made an array
                # because isincreasing is most efficient on already-created
                # arrays
                assert isincreasing(self.indepdomain), \
                       "Independent variable values must be increasing"
                self.indepvartype = self.indepdomain.dtype.type
            else:
                print("Independent variable argument domain was: %r" % indepdomain)
                raise TypeError("Invalid type for independent variable "
                                  "domain")


    def setDepdomain(self, depdomain, abseps=None):
        if isinstance(depdomain, str):
            self.coordname = depdomain
            if self.depdomain is None:
                if self.coordtype is None:
                    self.depdomain = Interval(self.coordname, float,
                                                 [-np.Inf, np.Inf], abseps=abseps)
                    self.coordtype = float
                else:
                    self.depdomain = Interval(self.coordname,
                                                 self.coordtype,
                                                 _num_maxmin[self.coordtype],
                                             abseps=abseps)
            else:
                # If interp functions supplied then don't have a name for
                # Interval yet, so update it.
                if isinstance(self.output, interpclass) and \
                   isinstance(self.depdomain, Interval):
                    self.depdomain.name = depdomain
                else:
                    assert isinstance(self.output, Pointset)
                    self.diagnostics.warnings.append((self.depdomain.name,
                                          "Dependent variable already named. "
                                          "Ignoring user-supplied name."))
        else:
            if isinstance(depdomain, Interval):
                if self.trajdrange:
                    if depdomain.contains(self.trajdrange) is notcontained:
                        raise ValueError("Cannot set dependent variable "
                                          "domain inside current trajectory's "
                                          "range")
                self.depdomain = depdomain
                self.coordname = depdomain.name
                if self.coordtype is None:
                    self.coordtype = depdomain.type
                elif self.coordtype == depdomain.type:
                    pass
                else:
                    raise TypeError("Mismatch between type of depdomain "
                                      "argument and Pointset coord data")
            elif isinstance(depdomain, dict):
                assert len(depdomain) == 1, \
                       "Depend variables dictionary must have only 1 entry"
                d = list(depdomain.values())[0]
                if self.trajdrange:
                    assert self.trajdrange[0] in d
                    assert self.trajdrange[1] in d
                ## Assume d is in increasing order
                assert all(isfinite(d)), "Values must be finite"
                self.coordname = list(depdomain.keys())[0]
                if isinstance(d, (list, tuple)):
                    if self.coordtype is not None:
                        self.depdomain = array(d, self.coordtype)
                    else:
                        self.depdomain = array(d)
                elif isinstance(d, ndarray):
                    da = array(d)
                    if self.coordtype is not None and \
                       self.coordtype != da.dtype.type:
                        raise TypeError("Mismatch between type of depdomain "
                                          "argument and Pointset coord data")
                    else:
                        self.depdomain = da
                else:
                    raise TypeError("Invalid type for dependent variable "
                                      "domain")
                self.coordtype = self.depdomain.dtype.type
            else:
                print("Dependent variable domain argument was: %r" % depdomain)
                raise TypeError("Invalid type for dependent variable domain")
            if isinstance(self.output, Pointset):
                assert self.coordname == self.output.coordnames[0], \
                       "Mismatch between Pointset coord name and declared name"
                assert self.indepvarname == self.output.indepvarname, \
                       ("Mismatch between Pointset independent variable name "
                        "and declared name")


    def __call__(self, indepvar, checklevel=0):
        # Set actual time by subtracting internal offset. Especially for use by
        # "input" variables that are based on inherently time-shifted
        # arrays of values, with nothing to do with the globalt0 of hybrid
        # trajectories.
        indepvar = asarray(indepvar) - self._internal_t_offset
        if checklevel == 0:
            # level 0 -- no depvar bounds checking at all
            # (no need to check for indepvar as list case, which output
            # should know how to handle)
            try:
                if not self._vectorizable and isinstance(indepvar, _seq_types):
                    return [self.output(ival) for ival in indepvar]
                else:
                    return self.output(indepvar)
            except (OverflowError, ValueError):
                self.diagnostics.errors.append((indepvar, self.name + ": Overflow error in output"))
                raise
            except PyDSTool_BoundsError:
                self.diagnostics.errors.append((indepvar, self.name + ": Bounds error in output"))
                raise
        elif checklevel in [1,2]:
            if self.trajirange is None:
                idep = self.indepdomain
            else:
                # use known bounds on indep variable imposed by self.output
                idep = self.trajirange
            indepvar_ok = True
            # level 1 -- ignore uncertain cases (treat as contained)
            # level 2 -- warn on uncertain (treat as contained)
            if isinstance(indepvar, _seq_types):
                vectorizable = self._vectorizable
                for d in indepvar:
                    # use 'in' so that this is compatible with
                    # interval, array and index indeps
                    try:
                        contresult = d in idep
                    except PyDSTool_UncertainValueError:
                        contresult = True
                        # adjust for rounding error so that interpolator
                        # does not barf on out-of-range values
                        if d < idep[0]:
                            try:
                                # list
                                dix = indepvar.index(d)
                            except AttributeError:
                                # array
                                dix = indepvar.tolist().index(d)
                            indepvar[dix] = idep[0]
                        elif d > idep[1]:
                            try:
                                # list
                                dix = indepvar.index(d)
                            except AttributeError:
                                # array
                                dix = indepvar.tolist().index(d)
                            indepvar[dix] = idep[1]
                        if checklevel == 2:
                            self.diagnostics.warnings.append((d, None))
                    if not contresult:
                        indepvar_ok = False
                        break
            else:
                vectorizable = True
                try:
                    indepvar_ok = indepvar in idep
                except PyDSTool_UncertainValueError as errinfo:
                    # adjust for rounding error so that interpolator
                    # does not barf on out-of-range values
                    if indepvar < idep[0]:
                        indepvar = idep[0]
                    elif indepvar > idep[1]:
                        indepvar = idep[1]
                    if checklevel == 2:
                        self.diagnostics.warnings.append((indepvar, None))
            # continue to get dependent variable value, unless indep
            # value was not OK
            if not indepvar_ok:
##                print "*** Debug info for variable: ", self.name
##                print "Interval rounding tolerance was", idep._abseps
                if checklevel == 2:
                    self.diagnostics.errors.append((indepvar,
                                self.name + " : " + self.indepdomain._infostr(1)))
                if vectorizable:
                    raise ValueError('Independent variable value(s) '
                                       'out of range in Variable call')
                else:
                    raise ValueError('Independent variable value '+\
                                   str(indepvar) + ' out of '
                                   'range in Variable call')
            try:
                if vectorizable:
                    depvar = self.output(indepvar)
                else:
                    depvar = [self.output(ival) for ival in indepvar]
                depvar_ok = True
            except PyDSTool_BoundsError as errinfo:
                depvar_ok = False
            # Now check that all computed values were in depdomain
            if depvar_ok:
                # no need to use self.trajdrange instead of
                # self.depdomain because we trust that self.output
                # generated the output within its own bounds!
                if isinstance(depvar, (_seq_types, Pointset)):
                    if isinstance(depvar, Pointset):
                        dv = depvar.toarray()
                    else:
                        dv = depvar
                    for d in dv:
                        # use 'in' so that this is compatible with
                        # interval, array and index indeps
                        try:
                            contresult = d in self.depdomain
                        except PyDSTool_UncertainValueError as errinfo:
                            contresult = True
                            if checklevel == 2:
                                # find which indepvar was the cause of
                                # the uncertain value
                                try:
                                    # list
                                    depix = dv.index(d)
                                except AttributeError:
                                    # array
                                    depix = dv.tolist().index(d)
                                self.diagnostics.warnings.append((indepvar[depix], errinfo.value))
                        if not isfinite(d):
                            # DEBUG
                            #print dv
                            #print self.output, "\n"
                            raise PyDSTool_BoundsError("Return value was not finite/defined (%s)"%str(d))
                        if not contresult:
                            depvar_ok = False
                            break
                elif depvar is None:
                    # DEBUG
                    #print "*** Debug info for variable: ", self.name
                    #print "Independent variable domain: ", self.indepdomain._infostr(1)
                    #print "Dependent variable domain: ", self.depdomain._infostr(1)
                    raise ValueError("Cannot compute a return value for "
                                          "independent variable value "
                                          + str(indepvar))
                else:
                    if isinstance(depvar, Point):
                        dv = depvar[0]
                    else:
                        dv = depvar
                    try:
                        depvar_ok = dv in self.depdomain
                    except PyDSTool_UncertainValueError as errinfo:
                        if checklevel == 2:
                            self.diagnostics.warnings.append((indepvar, errinfo.varval))
                    if not isfinite(dv):
                        # DEBUG
                        #print dv
                        #print self.output, "\n"
                        raise PyDSTool_BoundsError("Return value was not finite/defined (%s)"%str(dv))
            # return value if depvar in bounds
            if depvar_ok:
                return dv
            else:
                # DEBUG
                #print "Variable '%s' -"%self.name, "dependent var domain: ", \
                #      self.depdomain._infostr(1)
                #self.diagnostics.showWarnings()
                if vectorizable:
                    # DEBUG
                    #print self.output(indepvar), "\n"
                    raise PyDSTool_BoundsError('Computed value(s) %f outside'%dv + \
                                   ' validity range in Variable call')
                else:
                    raise PyDSTool_BoundsError('Computed value %f outside'%dv + \
                                   ' validity range in Variable call')
        else:
            # level 3 -- exception will be raised for uncertain case
            indepvar_ok = False
            try:
                # don't trap uncertain case exception from
                # Interval.__contains__
                if isinstance(indepvar, _seq_types):
                    vectorizable = self._vectorizable
                    indepvar_ok = all([i in self.indepdomain for i in \
                                           indepvar])
                else:
                    vectorizable = True
                    indepvar_ok = indepvar in self.indepdomain
            except TypeError as e:
                raise TypeError('Something messed up with the Variable '
                                  'initialization: '+str(e))
            else:
                if not indepvar_ok:
                    raise ValueError('Independent variable '+str(indepvar)+\
                                       ' out of range in Variable call')
            # Don't need 'if indepvar_ok' because exception would have
            # been raised.
            # For this checklevel, don't trap uncertain case exception from
            # Interval.__contains__
            try:
                if vectorizable:
                    depvar = self.output(indepvar)
                    depvar_ok = depvar in self.depdomain
                else:
                    depvar = [self.output(ival) for ival in indepvar]
                    depvar_ok = all([d in self.depdomain for d in \
                                         depvar])
            except PyDSTool_BoundsError as e:
                raise ValueError("Cannot compute a return value for "
                                      "this independent variable value: "
                                      + str(e))
            except PyDSTool_TypeError:
                if not self.defined:
                    print("Variable '%s' not fully defined."%self.name)
                    return None
                else:
                    raise
            else:
                if depvar_ok:
                    return depvar
                else:
                    if vectorizable:
                        raise PyDSTool_BoundsError('Computed value(s) '
                                    'outside validity range in Variable call')
                    else:
                        raise PyDSTool_BoundsError('Computed value '+str(depvar)+\
                                    'outside validity range in Variable call')


    def __repr__(self):
        return self._infostr(verbose=0)


    __str__ = __repr__


    def _infostr(self, verbose=1):
        if verbose == 0:
            return "Variable "+self.coordname+"("+self.indepvarname+")"
        else:
            try:
                if isinputcts(self):
                    ipstr = "continuous"
                else:
                    ipstr = "discrete"
            except ValueError:
                ipstr = "not defined"
            outputStr = "Variable:\n  Independent variable '" \
                        + self.indepvarname + "' [" + ipstr + "]\n"
            try:
                if isoutputcts(self):
                    opstr = "continuous"
                else:
                    opstr = "discrete"
            except ValueError:
                opstr = "not defined"
            outputStr += "    defined in domain  " + str(self.indepdomain)
            if verbose == 2:
                if self.trajirange is None:
                    outputStr += "\n    ranges not known for this trajectory"
                else:
                    outputStr += "\n    trajectory ranges  "+str(self.trajirange)
            outputStr += "\nDependent variable '" + self.coordname + \
                        "' [" + opstr + "]\n    defined in domain  "
            if not isinstance(self.depdomain, Interval):
                outputStr += _num_type2name[self.coordtype]+": "
            outputStr += str(self.depdomain)
            if verbose == 2:
                if self.trajdrange is None:
                    outputStr += "\n    ranges not known for this trajectory"
                else:
                    outputStr += "\n    trajectory ranges  "+str(self.trajdrange)
            return outputStr


    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))


    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        # remove reference to Cfunc types by converting to strings
        d['indepvartype'] = _num_type2name[self.indepvartype]
        d['coordtype'] = _num_type2name[self.coordtype]
        if 'funcspec' in self._funcreg:
            # then self is Imp/ExplicitFnGen and 'output' could not
            # be put in _funcreg because it relies on wrap_output
            # function that's not in the global namespace (so pickle fails
            # to find it)
            del d['output']
        for fname, finfo in self._funcreg.items():
            if finfo[0] == 'self':
                try:
                    del d[fname]
                except KeyError:
                    pass
            # else it's a Variable class method which won't get pickled
            # anyway, and will be restored to any class not in possession
            # of it if this object is unpickled
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        #print self.name, "- setstate: self.depdomain = ", self.depdomain.get()
        # reinstate Cfunc types
        self.indepvartype = _num_name2type[self.indepvartype]
        self.coordtype = _num_name2type[self.coordtype]
        # reinstate dynamic methods / functions
        for fname, finfo in self._funcreg.items():
            if finfo[0] == 'self' and not hasattr(eval(finfo[0]), fname):
                # avoids special entry for 'outputdata'
                setattr(eval(finfo[0]), fname, finfo[1])
        if 'funcspec' in self._funcreg:
            # Add the specific mapping functions for Ex/ImplicitFnGen objects
            funcspec = self._funcreg['funcspec'][1]
            outputdata = self._funcreg['outputdata'][1]
            if hasattr(self, '_var_namemap'):
                var_namemap = self._var_namemap
            else:
                var_namemap = None
            if hasattr(self, 'initialconditions'):
                ics = copy.copy(self.initialconditions)
            else:
                ics = None
            if hasattr(self, '_refvars'):
                if self._refvars is not None and self._refvars != []:
                    refvars = [copy.copy(v) for v in self._refvars]
                else:
                    refvars = None
            else:
                refvars = None
            # if refvars in dictionary then just leave them there!
            self.setOutput(outputdata, funcspec,
              self.globalt0, var_namemap, ics, refvars)


    def __del__(self):
        # delete object-specific class methods etc. before deleting
        # to avoid crowding namespace
##        if hasattr(self, 'output'):
##            del self.output
        for fname, finfo in self._funcreg.items():
            # Treat special cases first
            if finfo[0] is None:
                # don't want to eval(None) below
                continue
            elif fname == '_impfn':
                exec_str = 'del Variable.' + finfo[0]
                try:
                    exec(exec_str)
                except AttributeError:
                    # Uncertain why the name appears multiple times for their
                    # to be multiple attempts to delete it (which of course
                    # fail after the first successful attempt)
                    pass
            elif fname == 'funcspec':
                # doesn't refer to any dynamically-created methods
                # so ignore
                pass
            elif fname == 'outputdata':
                # doesn't refer to any dynamically-created methods
                # so ignore
                pass
            elif hasattr(eval(finfo[0]), fname):
                exec_str = 'del '+ finfo[0] + '.' + fname
                try:
                    exec(exec_str)
                except RuntimeError:
                    # sometimes get these when objects improperly delted
                    # and new objects with the same name created
                    pass
        if hasattr(self, '_refvars'):
                if self._refvars is not None and self._refvars != []:
                    for v in self._refvars:
                        v.__del__()


class HybridVariable(Variable):
    """Mimics part of the API of a non-hybrid variable.

    This is a somewhat ugly hack as it's implemented by using a whole
    HybridTrajectory object to extract individual variable values,
    rather than having extracted a sequence of Variable objects from
    a HT and stitching them back together as a single entity."""
    def __init__(self, hybridtraj, coordname, indepdomain, abseps=None):
        # store reference to the hybrid trajectory
        self._ht = hybridtraj
        self.name = 'Hybrid variable '+coordname
        self.outputdata = None    # not used
        self.defined = True
        self.indepvarname = 't'
        self.indepdomain = indepdomain
        self.indepvartype = float
        self.coordname = coordname
        self.depdomain = Interval(self.coordname, float,
                                    [-np.Inf, np.Inf], abseps=abseps)
        self.coordtype = float
        self.trajirange = None
        self.trajdrange = None
        self.diagnostics = Diagnostics()
        # important that this isn't a Pointset for Variable.py's
        # isinputcts, isoutputcts, etc.
        self.output = None

    def __call__(self, indepvar, checklevel=0):
        return self._ht(indepvar, self.coordname, checklevel=checklevel)

    def getDataPoints(self):
        """Returns a Pointset of independent and dependent variable values,
        provided variable is based on a mesh (otherwise None is returned).
        """
        return self._ht.sample([self.coordname])

    def underlyingMesh(self):
        """Reveal underlying mesh as arrays, rather than Pointset as returned
        by getDataPoints method."""
        vs = self._ht.sample([self.coordname])
        return array([vs.indepvararray, vs.coordarray[0]])

    def __repr__(self):
        return "Hybrid variable "+self.coordname

    __str__ = __repr__

    def info(self, verboselevel=1):
        return "Hybrid variable "+self.coordname

    # overrides from Variable class

    def __getstate__(self):
        return copy.copy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        # must override Variable.__del__
        pass



class OutputFn(object):
    """One-dimensional function wrapper."""

    def __init__(self, fn, datapoints=None, numtypes=(float64,float64),
                 abseps=None):
        assert isinstance(fn, types.FunctionType) or \
               isinstance(fn, types.BuiltinFunctionType), \
               ("fn argument must be a regular Python function")
        self.fn = fn
        # datapoints can be exhaustive list of known values for fn or
        # a Interval range for continuous-valued functions
        if datapoints is None:
            datapoints = (Interval('indepvardom', numtypes[0], [-np.Inf, np.Inf],
                                   abseps=abseps),
                          Interval('depvardom', numtypes[1], [-np.Inf, np.Inf],
                               abseps=abseps))
        try:
            self.datapoints = (datapoints[0], datapoints[1])
        except TypeError:
            raise TypeError("datapoints argument must be a 2-tuple or list "
                              "of 2-tuples or lists")
        try:
            self.types = (numtypes[0], numtypes[1])
        except TypeError:
            raise TypeError("numtypes argument must be a 2-tuple or list "
                              "of 2-tuples or lists")


    def __call__(self, arg):
        if isinstance(arg, _seq_types):
            try:
                return self.fn(arg)
            except:
                return array([self.fn(v) for v in arg])
        else:
            return self.fn(arg)


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        # remove reference to Cfunc types by converting to strings
        d['types'] = (_num_type2name[self.types[0]],
                      _num_type2name[self.types[1]])
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc types
        self.types = (_num_name2type[self.types[0]],
                      _num_name2type[self.types[1]])


# ---------------------------------------------------------------------


def isinputcts(obj):
    if isinstance(obj, Variable):
        if obj.defined:
            if compareNumTypes(obj.indepvartype, float64):
                return isinstance(obj.indepdomain, Interval) and not \
                       isinstance(obj.output, Pointset)
            elif compareNumTypes(obj.indepvartype, int32):
                return False
            else:
                raise TypeError("Unsupported independent variable type for Variable")
        else:
            raise ValueError("Variable is not fully defined")
    else:
        # provide support for e.g. Trajectories. Cannot use Trajectory class
        # name explicitly here because will run into an infinite import loop
        # between Variable and Trajectory!
        if compareNumTypes(obj.indepvartype, float64):
            return isinstance(obj.indepdomain, Interval)


def isinputdiscrete(var):
    return not isinputcts(var)

##def isinputdiscrete(var):
##    if compareNumTypes(var.indepvartype, float64):
##        return type(var.indepdomain) == ndarray or \
##               isinstance(var.output, Pointset)
##    elif compareNumTypes(var.indepvartype, int32):
##        return True
##    else:
##        raise TypeError("Unsupported independent variable type for Variable")


def isoutputcts(var):
    assert isinstance(var, Variable), "Argument must be a Variable"
    if var.defined:
        if compareNumTypes(var.coordtype, float64):
            return isinstance(var.depdomain, Interval) and not \
                   isinstance(var.output, Pointset)
        elif compareNumTypes(var.coordtype, int32):
            return False
        else:
            raise TypeError("Unsupported dependent variable type for Variable")
    else:
        raise ValueError("Variable is not fully defined")


def isoutputdiscrete(obj):
    return not isoutputcts(obj)


def iscontinuous(var):
    """Determine if variable is continuously defined on its input and
    output domains."""
    assert isinstance(var, Variable), "Argument must be a Variable"
    return isinputcts(var) and isoutputcts(var)


def isdiscrete(var):
    """Determine if variable is discretely defined on its input and
    output domains."""
    return not (isinputcts(var) and isoutputcts(var))

