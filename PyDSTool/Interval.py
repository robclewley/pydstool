"""Interval class

  Robert Clewley, June 2005

Interval objects have attributes:
   name: str  (variable label, which DOES NOT have to be unique)
   typestr: 'int' or 'float' (immutable) - doesn't handle 'complex' yet
   type: type
   _loval (low endpoint val): numeric
   _hival (hi endpoint val): numeric
   _abseps: numeric
   issingleton: boolean
   _intervalstr: str
"""
from __future__ import division, absolute_import, print_function

# Note: The integer intervals will later be used as the basis for
# supporting finitely-sampled real ranges.

## PyDSTool imports
from .utils import *
from .common import *
from .errors import *

## Other imports
from numpy import Inf, NaN, isfinite, isinf, isnan, array, sign, linspace, arange
import re, math
import copy

MIN_EXP = -15

# type identifiers
re_number = r'([-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)'

c=1  # contained const
n=0  # notcontained const
u=-1  # uncertain const


__all__ = ['notcontained', 'contained', 'uncertain', 'Interval',
           'isinterval', 'issingleton']


class IntervalMembership(int):
    """Numeric Interval membership type."""

    def __init__(self, *arg):
        val = arg[0]
        if val == -1:
            self.valstr = 'uncertain'
        if val == 0:
            self.valstr = 'notcontained'
        if val == 1:
            self.valstr = 'contained'
        if self.__int__() not in [-1, 0, 1]:
            raise ValueError('Invalid value for numeric Interval membership type')

    def __repr__(self):
        return self.valstr

    __str__ = __repr__

    def __and__(self, v):
        """Interval `logical AND` for checking interval containment.
        e.g. if both endpoints are contained then the whole interval is."""
        sv = self.__int__()
        ov = v.__int__()
        if sv == n or ov == n:
            return notcontained
        elif sv == u or ov == u:
            return uncertain
        else: # sv == ov == c is only remaining case
            return contained

    def __rand__(self, v):
        return self.__and__(v)

    def __or__(self, v):
        """Interval `logical OR` for checking interval intersection.
        e.g. if at least one endpoint is contained then there is intersection."""
        sv = self.__int__()
        ov = v.__int__()
        if sv == c or ov == c:
            return contained
        elif sv == u and ov == u:
            return uncertain
        else:
            return notcontained

    def __ror__(self, v):
        return self.__or__(v)


global notcontained, contained, uncertain
notcontained = IntervalMembership(False)
contained = IntervalMembership(True)
uncertain = IntervalMembership(-1)



class Interval(object):
    """Numeric Interval class.

    Numeric Interval implementation for integer and float types.

    If the interval is not specified fully on initialisation then operations
        on the object are limited to set().
    """

    def __init__(self, name, intervaltype, intervalspec=None, abseps=None):
#        if not isinstance(name, six.string_types):
#            raise PyDSTool_TypeError('Name must be a string')
        self.name = name
        try:
            self.type = _num_equivtype[intervaltype]
        except KeyError:
            raise PyDSTool_TypeError('Incorrect type specified for interval')
        self.typestr = _num_type2name[self.type]
        if compareNumTypes(self.type, _int_types):
            abseps = 0
            self.isdiscrete = True
        else:
            if abseps is None:
                abseps = 1e-13
            else:
                assert abseps >= 0, "abseps argument must be non-negative"
            self.isdiscrete = False
        self._abseps = abseps
        self.defined = False  # default value
        self._maxexp = None  # default value, unused for singletons
        if intervalspec is not None:
            self.set(intervalspec)
        else:
            # just declare the name and type
            self._loval = None
            self._hival = None


    def info(self, verboselevel=0):
        if verboselevel > 0:
            # info is defined in utils.py
            info(self.__dict__, "Interval " + self.name,
                 recurseDepthLimit=1+verboselevel)
        else:
            print(self.__repr__())


    # Return all the interval's relevant data in a tuple
    def __call__(self):
        if self.defined:
            return (self.name, self.type, self.typestr,\
                    (self._loval, self._hival))
        else:
            raise PyDSTool_ExistError('Interval undefined')

    __hash__ = None

    def __eq__(self, other):
        assert isinstance(other, Interval)
        return self.defined == other.defined and \
               self.type == other.type and \
               self._loval == other._loval and \
               self._hival == other._hival and \
               self._abseps == other._abseps


    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        if isinstance(other, Interval):
            raise NotImplementedError
        elif isinstance(other, _seq_types):
            return [o <= self._loval + self._abseps for o in other]
        else:
            return other <= self._loval + self._abseps

    def __ge__(self, other):
        if isinstance(other, Interval):
            raise NotImplementedError
        elif isinstance(other, _seq_types):
            return [o >= self._hival - self._abseps for o in other]
        else:
            return other >= self._hival - self._abseps

    def __lt__(self, other):
        if isinstance(other, Interval):
            return other._loval - other._abseps > self._hival + self._abseps
        elif isinstance(other, _seq_types):
            return [o > self._hival + self._abseps for o in other]
        else:
            return other > self._hival + self._abseps

    def __gt__(self, other):
        if isinstance(other, Interval):
            return self._loval - self._abseps > other._hival + other._abseps
        elif isinstance(other, _seq_types):
            return [o < self._loval - self._abseps for o in other]
        else:
            return other < self._loval - self._abseps

    def __add__(self, val):
        c = copy.copy(self)
        c.set((self._loval+val,self._hival+val))
        return c

    def __radd__(self, val):
        c = copy.copy(self)
        c.set((self._loval+val,self._hival+val))
        return c

    def __sub__(self, val):
        c = copy.copy(self)
        c.set((self._loval-val,self._hival-val))
        return c

    def __rsub__(self,val):
        c = copy.copy(self)
        # switch endpoint order
        c.set((val-self._hival,val-self._loval))
        return c

    def __mul__(self, val):
        c = copy.copy(self)
        c.set((self._loval*val,self._hival*val))
        return c

    def __rmul__(self, val):
        c = copy.copy(self)
        c.set((self._loval*val,self._hival*val))
        return c

    def __truediv__(self, val):
        c = copy.copy(self)
        c.set((self._loval/val,self._hival/val))
        return c

    def __rtruediv__(self,val):
        c = copy.copy(self)
        # switch endpoint order
        if isfinite(self._hival):
            if self._hival==0:
                new_lo = sign(val)*Inf
            else:
                new_lo = self.type(val/self._hival)
        else:
            new_lo = val/self._hival
        if isfinite(self._loval):
            if self._loval==0:
                new_hi = sign(val)*Inf
            else:
                new_hi = self.type(val/self._loval)
        else:
            new_hi = val/self._loval
        if new_hi < new_lo:
            # negative and division changes order
            c.set((new_hi,new_lo))
        else:
            c.set((new_lo,new_hi))
        return c

    def __neg__(self):
        c = copy.copy(self)
        # switch endpoint order
        c.set((-self._hival,-self._loval))
        return c

    def __contains__(self, val):
        testresult = self.contains(val)
        if testresult == notcontained:
            return False
        elif testresult == contained:
            return True
        else:
            raise PyDSTool_UncertainValueError('cannot determine membership '
                                         '(uncertain)', val)

    def contains(self, val):
        """Report membership of val in the interval,
        returning type IntervalMembership."""
        if isinstance(val, _seq_types):
            return [self.contains(v) for v in val]
        try:
            if not self.defined:
                raise PyDSTool_ExistError('Interval undefined')
            if self._maxexp is None and not self.issingleton:
                try:
                    loexp = math.log(abs(self._loval), 10)
                except (OverflowError, ValueError):
                    loexp = 0
                try:
                    hiexp = math.log(abs(self._hival), 10)
                except (OverflowError, ValueError):
                    hiexp = 0
                self._maxexp = max(loexp, hiexp)
            if isinstance(val, _num_name2equivtypes[self.typestr]):
                compval = val
                if compareNumTypes(self.type, _int_types):
                    eps = 0
                else:
                    eps = self._abseps
            else:
                if isinstance(val, _all_int):
                    compval = float(val)
                    eps = self._abseps
                elif isinstance(val, _all_float):
                    # cannot ever compare a float in an integer interval,
                    # unless int(val) == val or val is not finite
                    if isinf(val):
                        compval = val
                        eps = 0
                    elif int(val) == val:
                        compval = int(val)
                        eps = 0
                    else:
                        raise PyDSTool_TypeError('Incorrect type of query value')
                elif not val.issingleton:
                    # catches non-intervals or non-singleton intervals
                    if not val.defined:
                        raise PyDSTool_ExistError('Input interval undefined')
                    if not compareNumTypes(val.type, self.type) and \
                       compareNumTypes(val.type, _all_float):
                        # meaningless to ask if float interval is contained in an
                        # integer interval!
                        raise PyDSTool_TypeError('Interval type mismatch')
                    if compareNumTypes(val.type, self.type) and \
                       compareNumTypes(self.type, _all_int):
                        eps = 0
                    else:
                        eps = max(self._abseps, val._abseps)
                        try:
                            minexpallowed = math.ceil(-MIN_EXP - self._maxexp)
                        except (TypeError, OverflowError):
                            # _maxexp is None
                            minexpallowed = Inf
                        if eps > 0 and -math.log(eps,10) > minexpallowed:
                            eps = math.pow(10,-minexpallowed)
                    if isfinite(val._loval) or isfinite(self._loval):
                        tempIlo = val._loval >= (self._loval + eps)
                    else:
                        tempIlo = False
                    if isfinite(val._hival) or isfinite(self._hival):
                        tempIhi = val._hival <= (self._hival - eps)
                    else:
                        tempIhi = False
                    if tempIlo and tempIhi:
                        return contained
                    elif eps == 0:
                        return notcontained
                    else:
                        # having already tested for being contained, this is
                        # sufficient for uncertainty
                        if isfinite(val._loval) or isfinite(self._loval):
                            tempUlo = val._loval > (self._loval - eps)
                            tempElo = val._loval <= (self._loval - eps)
                        else:
                            tempUlo = val._loval == self._loval
                            tempElo = False
                        if isfinite(val._hival) or isfinite(self._hival):
                            tempUhi = val._hival < (self._hival + eps)
                            tempEhi = val._hival >= (self._hival + eps)
                        else:
                            tempUhi = val._hival == self._hival
                            tempEhi = False
                        if ((tempUlo and not tempEhi) or (tempUhi and \
                                                         not tempElo)) \
                                         and not self.isdiscrete:
                            return uncertain
                        else:
                            # else must be notcontained
                            return notcontained
                else:
                    # val is a singleton interval type
                    # issingleton == True implies interval is defined
                    # Now go through same sequence of comparisons
                    if compareNumTypes(val.type, self.type):
                        compval = val.get()
                        if compareNumTypes(self.type, _all_int):
                            eps = 0
                        else:
                            eps = max(self._abseps, val._abseps)
                            try:
                                loexp = math.log(abs(self._loval), 10)
                            except (OverflowError, ValueError):
                                loexp = 0
                            try:
                                hiexp = math.log(abs(self._hival), 10)
                            except (OverflowError, ValueError):
                                hiexp = 0
                            minexpallowed = math.ceil(-MIN_EXP - max(loexp,
                                                                     hiexp))
                            if eps > 0 and -math.log(eps,10) > minexpallowed:
                                eps = math.pow(10,-minexpallowed)
                    else:
                        if compareNumTypes(val.type, _all_int):
                            compval = val.get()
                            eps = self._abseps
                        elif compareNumTypes(val.type, _all_float):
                            # cannot ever compare a float in an integer interval
                            # unless bd values are equal to their int() versions
                            if int(val._loval) == val._loval and \
                               int(val._hival) == val._hival:
                                compval = (int(val[0]), int(val[1]))
                                eps = 0
                            else:
                                raise PyDSTool_TypeError('Invalid numeric type '
                                                         'of query value')
                        else:  # unexpected (internal) error
                            raise PyDSTool_TypeError('Invalid numeric type of '
                                                     'query value')
        except AttributeError:
            raise PyDSTool_TypeError('Expected a numeric type or a singleton '
                                 'interval. Got type '+str(type(val)))
        else:
            tempIlo = compval >= (self._loval + eps)
            tempIhi = compval <= (self._hival - eps)
            if tempIlo and tempIhi:
                return contained
            elif eps == 0: # only other possibility (no uncertainty)
                return notcontained
            else:
                # having already tested for being contained, this is
                # sufficient for uncertainty
                tempUlo = compval > (self._loval - eps)
                tempUhi = compval < (self._hival + eps)
                tempElo = compval <= (self._loval - eps)
                tempEhi = compval >= (self._hival + eps)
                if ((tempUlo and not tempEhi) or (tempUhi and not tempElo)) and \
                             not self.isdiscrete:
                    return uncertain
                # else must be notcontained
                else:
                    return notcontained


    def intersect(self, other):
        if not isinstance(other, Interval):
            raise PyDSTool_TypeError("Can only intersect with other Interval "
                                     "types")
        result = None    # default, initial value if no intersection
        if self.type != other.type:
            raise PyDSTool_TypeError("Can only intersect with other Intervals "
                                     "having same numeric type")
        if compareNumTypes(self.type, _all_complex) or \
           compareNumTypes(other.type, _all_complex):
            raise TypeError("Complex intervals not supported")
        if self.contains(other):
            result = other
        elif other.contains(self):
            result = self
        else:
            # no total containment possible
            if other.contains(self._hival) is contained:
                # then also self.contains(other._loval)
                result = Interval('__result__', self.type, [other._loval,
                                                    self._hival])
            elif other.contains(self._loval) is contained:
                # then also self.contains(other._hival)
                result = Interval('__result__', self.type, [self._loval,
                                                    other._hival])
        return result


    def atEndPoint(self, val, bdcode):
        """val, bdcode -> Bool

        Determines whether val is at the endpoint specified by bdcode,
        to the precision of the interval's _abseps tolerance.
        bdcode can be one of 'lo', 'low', 0, 'hi', 'high', 1"""

        assert self.defined, 'Interval undefined'
        assert isinstance(val, (_int_types, _float_types)), \
               'Invalid value type'
        assert isfinite(val), "Can only test finite argument values"
        if bdcode in ['lo', 'low', 0]:
            if self.isdiscrete:
                return val == self._loval
            else:
                return abs(val - self._loval) < self._abseps
        elif bdcode in ['hi', 'high', 1]:
            if self.isdiscrete:
                return val == self._hival
            else:
                return abs(val - self._hival) < self._abseps
        else:
            raise ValueError('Invalid boundary spec code')

    def sample(self, dt, strict=False, avoidendpoints=False):
        """Sample the interval, returning a list.

    Arguments:

    dt : sample step

    strict : (Boolean) This option forces dt to be used throughout the interval,
      with a final step of < dt if not commensurate. Default of False
      is used for auto-selection of sample rate to fit interval
      (choice based on dt argument).

    avoidendpoints : (Boolean, default False). When True, ensures that the first and
      last independent variable ("t") values are not included, offset by
      an amount given by self._abseps (the endpoint tolerance).
    """
        assert self.defined
        intervalsize = self._hival - self._loval
        assert isfinite(intervalsize), "Interval must be finite"
        if dt > intervalsize:
            print("Interval size = %f, dt = %f"%(intervalsize, dt))
            raise ValueError('dt must be smaller than size of interval')
        if dt <= 0:
            raise ValueError('Must pass dt >= 0')
        if compareNumTypes(self.type, _all_float):
            if strict:
                # moved int() to samplist's xrange
                samplelist = list(arange(self._loval, self._hival, dt,
                                         dtype=float))
                if self._hival not in samplelist:
                    samplelist.append(self._hival)
            else: # choose automatically
                n = max(int(round(intervalsize / dt)), 2)
                dt = intervalsize/n
                samplelist = list(linspace(self._loval, self._hival, n))
            if avoidendpoints:
                samplelist[-1] = self._hival - 1.1*self._abseps
                samplelist[0] = self._loval + 1.1*self._abseps
        elif compareNumTypes(self.type, _all_int):
            if not isinstance(dt, _int_types):
                raise ValueError("dt must be an integer for integer "
                                        "intervals")
            if strict:
                print("Warning: 'strict' option is invalid for integer " + \
                      "interval types")
            if avoidendpoints:
                loval = self._loval+1
                hival = self._hival-1
                assert loval <= hival, ('There are no points to return with '
                                        'these options!')
            else:
                loval = self._loval
                hival = self._hival
            samplelist = list(range(loval, hival+1, dt))
            # extra +1 on hival because of python range() policy!
        else:
            raise TypeError("Unsupported value type")
        # return a list (not an array) so that pop method is available to VODE Generator, etc.
        return samplelist

    # deprecated syntax
    uniformSample = sample


    def set(self, arg):
        """Define interval in an Interval object"""
        if isinstance(arg, _seq_types) and len(arg)==2:
            if arg[0] == arg[1]:
                # attempt to treat as singleton
                self.set(arg[0])
            else:
                self.issingleton = False
                loval = arg[0]
                hival = arg[1]
                #assert not isnan(loval) and not isnan(hival), \
                #       "Cannot specify NaN as interval endpoint"
                try:
                    if not loval < hival:
                        print("set() was passed loval = ", loval, \
                            " and hival = ", hival)
                        raise PyDSTool_ValueError('Interval endpoints must be '
                                        'given in order of increasing size')
                except TypeError:
                    # unorderable types
                    pass
                self._intervalstr = '['+str(loval)+',' \
                                    +str(hival)+']'
                if compareNumTypes(type(loval), self.type):
                    self._loval = loval
                elif compareNumTypes(self.type, _float_types):
                    self._loval = float(str(loval))
                elif isinf(loval):
                    # allow Inf to be used for integer types
                    self._loval = loval
                else:
                    raise TypeError("Invalid interval endpoint type")
                if compareNumTypes(type(hival), self.type):
                    self._hival = hival
                elif compareNumTypes(self.type, _float_types):
                    self._hival = float(str(hival))
                elif isinf(hival):
                    # allow Inf to be used for integer types
                    self._hival = hival
                else:
                    raise TypeError("Invalid interval endpoint type")
                self.defined = True
        elif isinstance(arg, (_int_types, _float_types)):
            assert isfinite(arg), \
                   "Singleton interval domain value must be finite"
            if self.isdiscrete:
                # int types or floats=ints only
                if not int(arg)==arg:
                    raise TypeError("Invalid interval singleton type")
            else:
                arg = float(arg)
            self.issingleton = True
            self._intervalstr = str(arg)
            self._loval = arg
            self._hival = arg
            self.defined = True
        else:
            print("Error in argument: %r of type %s" % (arg, type(arg)))
            raise PyDSTool_TypeError('Interval spec must be a numeric or '
                                     'a length-2 sequence type')

    def __setitem__(self, ix, val):
        if ix == 0:
            self.set((val, self._hival))
        elif ix == 1:
            self.set((self._loval, val))
        else:
            raise PyDSTool_TypeError('Invalid endpoint')

    def __getitem__(self, ix):
        if self.defined:
            if ix == 0:
                return self._loval
            elif ix == 1:
                return self._hival
            else:
                raise PyDSTool_ValueError("Invalid endpoint")
        else:
            raise PyDSTool_ExistError('Interval undefined')

    def isfinite(self):
        if self.defined:
            if self.issingleton:
                return isfinite(self._loval)
            else:
                return (isfinite(self._loval), isfinite(self._hival))
        else:
            raise PyDSTool_ExistError('Interval undefined')

    def get(self, ix=None):
        """Get the interval as a tuple or a number (for singletons),
        or an endpoint if ix is not None"""
        if self.defined:
            if ix == 0:
                return self._loval
            elif ix == 1:
                return self._hival
            elif ix is None:
                if self.issingleton:
                    return self._loval
                else:
                    return [self._loval, self._hival]
            else:
                raise PyDSTool_TypeError('Invalid return form specified')
        else:
            raise PyDSTool_ExistError('Interval undefined')


    def _infostr(self, verbose=1):
        """Get info on a known interval definition."""

        if verbose > 0:
            infostr = "Interval "+self.name+"\n"
            if self.defined:
                infostr += '  ' + self.typestr+': '+\
                          self.name+' = '+self._intervalstr+ \
                          ' @ eps = '+str(self._abseps)
            else:
                infostr += '  ' + self.typestr+': '+self.name+' @ eps = '+ \
                          str(self._abseps) + " (not fully defined)"
        else:
            infostr = "Interval "+self.name
        return infostr


    def __repr__(self):
        return self._infostr(verbose=0)


    __str__ = __repr__


    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))


    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        # remove reference to Cfunc self.type
        d['type'] = None
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc self.type
        self.type = _num_name2type[self.typestr]


#--------------------------------------------------------------------

# Exported utility functions

def isinterval(obj):
    """Determines whether the given obj is a Interval object."""
    return isinstance(obj, Interval)


# Check whether an interval is a singleton
def issingleton(ni):
    if ni.defined:
        return ni.issingleton
    else:
        raise PyDSTool_ExistError('Interval undefined')
