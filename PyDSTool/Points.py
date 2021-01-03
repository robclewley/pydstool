"""Point and Pointset enhanced array classes.

(Objects of both classes are mutable.)

    Robert Clewley, February 2006
"""

# ----------------------------------------------------------------------------


from collections import defaultdict

## PyDSTool imports
from .utils import *
from .common import *
from .errors import *
from .parseUtils import symbolMapClass, mapNames

## Other imports
from numpy import isfinite, array2string, r_, c_, \
    less, greater, linalg, shape, array, argsort, savetxt, \
    take, zeros, transpose, resize, indices, concatenate, isscalar

from numpy import complex, complexfloating, int, integer, \
     float, floating, float64, complex128, int32
from numpy import any, all, alltrue, sometrue, ndarray
import numpy as np

import sys
from copy import copy, deepcopy
import warnings


__all__ = ['Point', 'Pointset', 'isparameterized', 'pointsToPointset',
           'PointInfo', 'makeNonParameterized', 'arrayToPointset',
           'VarCaller', 'comparePointCoords', 'importPointset',
           'exportPointset', 'mergePointsets', 'padPointset',
           'export_pointset_to_CSV']

#----------------------------------------------------------------------------


class VarCaller(object):
    """Wrapper for Variable type to call Pointset and return array type."""

    def __init__(self, pts):
        if isinstance(pts, (Point, Pointset)):
            self.pts = pts
        else:
            raise TypeError("Invalid type for pts argument")

    def __call__(self, x):
        return self.pts(x).toarray()


#----------------------------------------------------------------------------


# for internal use
point_keys = ['coorddict', 'coordarray', 'coordnames', 'coordtype', 'norm', 'labels']


def _cvt_type(t):
    """Convert Python type to NumPy type"""
    try:
        return _num_equivtype[t]
    except KeyError:
        raise TypeError('Coordinate type %s not valid for Point' % str(t))


def _check_type(value, ctype):
    """Smart type checking for scalar values"""

    if not isscalar(value):
        raise ValueError('Type checking works only for scalar values')

    if isinstance(value, _float_types):
        assert compareNumTypes(ctype, _float_types), 'type mismatch'
    elif isinstance(value, _int_types):
        assert compareNumTypes(ctype, _real_types), 'type mismatch'
    #elif isinstance(value, _complex_types):
        #assert compareNumTypes(ctype, complex), 'type mismatch'
    else:
        raise TypeError("Must pass numeric type")


class Point(object):
    """N-dimensional point class."""

    # Contains an ordered list of names for the coordinates (to
    # suggest how the points belong to a particular space specified
    # using a particular basis)
    def __init__(self, kwd=None, **kw):
        if kwd is not None:
            if kw != {}:
                raise ValueError("Cannot mix keyword dictionary and keywords")
            kw = kwd
        self._parameterized = False
        self.labels = {}
        if intersect(kw.keys(), point_keys) == []:
            temp_kw = {}
            temp_kw['coorddict'] = copy(kw)
            kw = copy(temp_kw)

        # Setting Point type
        # By default creating 'float' Point.  To make 'int' Point explicitly
        # set 'coordtype' argument or use 'int' array in 'coordarray' argument.
        # Explicit type setting has higher priority
        ctype = _num_equivtype[float]
        if 'coordarray' in kw and isinstance(kw['coordarray'], ndarray):
                ctype = _cvt_type(kw['coordarray'].dtype.type)
        if 'coordtype' in kw:
                ctype = _cvt_type(kw['coordtype'])

        self.coordtype = ctype

        # Preparing coord data (names and values)
        coordnames = []
        coordvalues = []
        if 'coorddict' in kw:
            vals = []
            for n, v in kw['coorddict'].items():
                # Add coord name with type checking
                coordnames.append(n if isinstance(n, str) \
                                  else repr(n))

                # Add coord value with type checking
                ## Extract value from sequence if needed
                if isinstance(v, (list, tuple, ndarray)):
                    if len(v) == 0:
                        raise ValueError('Values sequence must be nonempty')
                    if len(v) > 1:
                        warnings.warn(
                            'Sequence value %s truncated to first element' % v,
                            UserWarning,
                            stacklevel=2
                        )
                    val = v[0]
                else:
                    val = v
                _check_type(val, ctype)
                vals.append(val)

            coordvalues = array(vals, ctype)
        elif 'coordarray' in kw:
            vals = kw['coordarray']
            if np.ndim(vals) > 1:
                raise ValueError("Invalid rank for coordinate array: %i" % np.ndim(vals))
            if len(vals) == 0:
                raise ValueError('Values sequence must be nonempty')

            if isinstance(vals, ndarray):
                _check_type(vals[0], ctype)
                coordvalues = vals.astype(ctype)
            elif isinstance(vals, (list, tuple)):
                for v in vals:
                    _check_type(v, ctype)
                coordvalues = array(vals, ctype)
            else:
                raise TypeError('Coordinate type %s not valid for Point' % str(type(vals)))

            if 'coordnames' in kw:
                if isinstance(kw['coordnames'], str):
                    coordnames = list(kw['coordnames'])
                else:
                    coordnames = kw['coordnames']
            else:
                coordnames = [str(cix) for cix in range(coordvalues.shape[0])]
        else:
            raise ValueError("Missing coord info in keywords")

        # Sanity checks for prepared data
        assert isUniqueSeq(coordnames), 'Coordinate names must be unique'
        if len(coordnames) != coordvalues.shape[0]:
            msg = "Point initialization error:\n" \
                  "Found coord names: {} (dimension = {}) " \
                  "vs. data dimension = {}".format(
                      coordnames, len(coordnames), coordvalues.shape[0]
                  )
            print(msg)
            raise ValueError("Mismatch between number of coordnames and "
                             "dimension of data")
        r = np.ndim(coordvalues)
        if r == 1:
            self.coordarray = coordvalues
        elif r == 0:
            self.coordarray = coordvalues.ravel()

        # Sorting
        cnames = array(coordnames)
        order = cnames.argsort()
        self.coordnames = cnames[order].tolist()
        self.coordarray = self.coordarray[order]
        self.dimension = len(self.coordnames)
        self.makeIxMaps()

        if 'norm' in kw:
            if kw['norm'] == 0:
                raise ValueError("Norm order for point cannot be zero")
            self._normord = kw['norm']
        else:
            self._normord = 2
        # extra information (for special bifurcation point data)
        if 'labels' in kw:
            self.addlabel(kw['labels'])

    def mapNames(self, themap):
        """Map coordinate names and label(s), using a symbol
        map of class symbolMapClass."""
        new_coordnames = array(themap(self.coordnames))
        assert isUniqueSeq(new_coordnames.tolist()), 'Coordinate names must be unique'
        order = argsort(new_coordnames)
        self.coordarray = self.coordarray[order]
        self.coordnames = new_coordnames[order].tolist()
        self.makeIxMaps()
        # the following call will be inherited by Pointset, and
        # works on Point labels-as-dict and Pointset labels-as-
        # PointInfo objects, as the latter have their own
        # mapNames method which will get called.
        self.labels = mapNames(themap, self.labels)


    def addlabel(self, label):
        if label is None:
            pass
        elif isinstance(label, str):
            self.labels = {label: {}}
        elif isinstance(label, tuple) and len(label)==2:
            if isinstance(label[0], str) and \
               isinstance(label[1], dict):
                self.labels[label[0]] = label[1]
        elif isinstance(label, dict):
            self.labels = label
        else:
            raise TypeError("Point label must be a string, a pair, or a dict")


    def removelabel(self):
        self.labels = {}


    def makeIxMaps(self):
        self._name_ix_map = dict(zip(self.coordnames, range(self.dimension)))
        self._ix_name_map = copy(self.coordnames)


    def todict(self, aslist=False):
        """Convert Point to a dictionary of array values (or of list with aslist=True)."""
        if aslist:
            return dict(zip(self._ix_name_map, self.coordarray.tolist()))
        else:
            return dict(zip(self._ix_name_map, self.coordarray))

    def __contains__(self, coord):
        return coord in self.coordnames

    def __delitem__(self, k):
        raise NotImplementedError

    def get(self, coord, d=None):
        if coord in self.coordnames:
            return self.__call__(coord)
        else:
            return d

    def update(self, d):
        for k, v in d.items():
            self.coordarray[self._map_names_to_ixs(k)] = v

    def items(self):
        return list(zip(self._ix_name_map, self.coordarray))

    def iteritems(self):
        return iter(zip(self._ix_name_map, self.coordarray))

    def values(self):
        return self.coordarray.tolist()

    def itervalues(self):
        return iter(self.coordarray.tolist())

    def keys(self):
        return self._ix_name_map

    def iterkeys(self):
        return iter(self._ix_name_map)

    def has_key(self, k):
        return k in self.coordnames


    def _map_names_to_ixs(self, namelist):
        try:
            try:
                # single string
                return self._name_ix_map[namelist]
            except TypeError:
                # list of strings
                return [self._name_ix_map[n] for n in namelist]
        except KeyError as e:
            raise PyDSTool_KeyError("Name not found: "+str(e))


    def __len__(self):
        return self.dimension


    def _force_coords_to_ixlist(self, x):
        if x is None:
            return list(range(self.dimension))
        elif x in range(self.dimension):
            # only used for recursive calls
            return [x]
        elif x in self.coordnames:
            # only used for recursive calls
            return [self._name_ix_map[x]]
        elif isinstance(x, _seq_types):
            if len(x) == 0:
                return list(range(self.dimension))
            else:
                return [self._force_coords_to_ixlist(el)[0] for el in x]
        elif isinstance(x, slice):
            stop = x.stop or self.dimension
            s1, s2, s3 = x.indices(stop)
            if s1 < 0 or s2 > self.dimension or s1 >= self.dimension:
                raise ValueError("Slice index out of range")
            return list(range(s1, s2, s3))
        else:
            raise ValueError("Invalid coordinate / index: %s"%str(x) + \
                             " -- coord names are: %s"%str(self.coordnames))


    def __call__(self, coords):
        if coords in range(self.dimension+1):
            if coords == self.dimension:
                # trap for when Point is used as an iterator, i.e. as
                # for x in pt -- avoids writing an __iter__ method that
                # will be inherited by Pointset, which already iterates fine
                raise StopIteration
            else:
                return self.coordarray[coords]
        elif coords in self.coordnames:
            ix = self._name_ix_map[coords]
            return self.coordarray[ix]
        else:
            ixlist = self._force_coords_to_ixlist(coords)
            return Point({'coordarray': self.coordarray[ixlist],
                      'coordnames': [self.coordnames[i] for i in ixlist],
                      'coordtype': self.coordtype,
                      'norm': self._normord,
                      'labels': self.labels})

    __getitem__ = __call__

#    def __iter__(self):
#        return self.coordarray.__iter__()


    def __setitem__(self, ixarg, val):
        """Change coordinate array values."""
        ixs = self._force_coords_to_ixlist(ixarg)
        if len(ixs) == 1:
            val = [val]
        try:
            for i, v in zip(ixs,val):
                self.coordarray[i] = v
        except TypeError:
            raise TypeError("Bad value type for Point")


    def toarray(self):
        if self.dimension == 1:
            return self.coordarray[0]
        else:
            return self.coordarray


    def __add__(self, other):
        res = self.copy()
        try:
            res.coordarray += other.coordarray
        except AttributeError:
            res.coordarray += other
        return res

    __radd__ = __add__

    def __sub__(self, other):
        res = self.copy()
        try:
            res.coordarray -= other.coordarray
        except AttributeError:
            res.coordarray -= other
        return res

    def __rsub__(self, other):
        res = self.copy()
        try:
            res.coordarray = other.coordarray - res.coordarray
        except AttributeError:
            res.coordarray = other - res.coordarray
        return res

    def __mul__(self, other):
        res = self.copy()
        try:
            res.coordarray *= other.coordarray
        except AttributeError:
            res.coordarray *= other
        return res

    __rmul__ = __mul__

    def __div__(self, other):
        res = self.copy()
        try:
            res.coordarray /= other.coordarray
        except AttributeError:
            res.coordarray /= other
        return res

    __truediv__ = __div__

    def __rdiv__(self, other):
        res = self.copy()
        try:
            res.coordarray = other.coordarray / res.coordarray
        except AttributeError:
            res.coordarray = other / res.coordarray
        return res

    __rtruediv__ = __rdiv__

    def __pow__(self, other):
        res = self.copy()
        res.coordarray **= other
        return res

    def __neg__(self):
        res = self.copy()
        res.coordarray = - res.coordarray
        return res

    def __pos__(self):
        return self.copy()

    def __lt__(self, other):
        try:
            assert shape(self) == shape(other)
            if hasattr(other, 'coordnames'):
                if self.coordnames != other.coordnames:
                    raise ValueError("Coordinate mismatch")
            return linalg.norm(self.coordarray, self._normord) < \
                   linalg.norm(other.coordarray, self._normord)
        except (AttributeError, TypeError, AssertionError):
            return self.coordarray < other
        except ZeroDivisionError:
            raise ValueError("Norm order for point cannot be zero")

    def __gt__(self, other):
        try:
            assert shape(self) == shape(other)
            if hasattr(other, 'coordnames'):
                if self.coordnames != other.coordnames:
                    raise ValueError("Coordinate mismatch")
            return linalg.norm(self.coordarray, self._normord) > \
                   linalg.norm(other.coordarray, self._normord)
        except (AttributeError, TypeError, AssertionError):
            return self.coordarray > other
        except ZeroDivisionError:
            raise ValueError("Norm order for point cannot be zero")

    def __le__(self, other):
        try:
            assert shape(self) == shape(other)
            if hasattr(other, 'coordnames'):
                if self.coordnames != other.coordnames:
                    raise ValueError("Coordinate mismatch")
            return linalg.norm(self.coordarray, self._normord) <= \
                   linalg.norm(other.coordarray, self._normord)
        except (AttributeError, TypeError, AssertionError):
            return self.coordarray <= other
        except ZeroDivisionError:
            raise ValueError("Norm order for point cannot be zero")

    def __ge__(self, other):
        try:
            assert shape(self) == shape(other)
            if hasattr(other, 'coordnames'):
                if self.coordnames != other.coordnames:
                    raise ValueError("Coordinate mismatch")
            return linalg.norm(self.coordarray, self._normord) >= \
                   linalg.norm(other.coordarray, self._normord)
        except (AttributeError, TypeError, AssertionError):
            return self.coordarray >= other
        except ZeroDivisionError:
            raise ValueError("Norm order for point cannot be zero")

    def __eq__(self, other):
        try:
            assert shape(self) == shape(other)
            if hasattr(other, 'coordnames'):
                if self.coordnames != other.coordnames:
                    raise ValueError("Coordinate mismatch")
            return linalg.norm(self.coordarray, self._normord) == \
                   linalg.norm(other.coordarray, self._normord)
        except (AttributeError, TypeError, AssertionError):
            return self.coordarray == other
        except ZeroDivisionError:
            raise ValueError("Norm order for point cannot be zero")

    def __ne__(self, other):
        try:
            assert shape(self) == shape(other)
            if hasattr(other, 'coordnames'):
                if self.coordnames != other.coordnames:
                    raise ValueError("Coordinate mismatch")
            return linalg.norm(self.coordarray, self._normord) != \
                   linalg.norm(other.coordarray, self._normord)
        except (AttributeError, TypeError, AssertionError):
            return self.coordarray != other
        except ZeroDivisionError:
            raise ValueError("Norm order for point cannot be zero")

    __hash__ = None

    def _infostr(self, verbose=0):
        precision = 8
        if verbose < 0:
            outputStr = "Point"
        elif verbose == 0:
            outputStr = "Point with coords:\n"
            for c in self.coordnames:
                outputStr += c
                if c != self.coordnames[-1]:
                    outputStr += "\n"
        elif verbose > 0:
            outputStr = ''
            for c in self.coordnames:
                v = self.coordarray[self._map_names_to_ixs(c)]
                if isinstance(v, ndarray):
                    dvstr = str(v[0])
                else:
                    # only alternative is a singleton numeric value (not list)
                    dvstr = str(v)
                outputStr += c+':  '+dvstr
                if c != self.coordnames[-1]:
                    outputStr += "\n"
            for label, infodict in self.labels.items():
                outputStr += "\nLabels: %s (%s)"%(label, str(infodict))
        return outputStr


    def __repr__(self):
        return self._infostr(verbose=1)


    __str__ = __repr__


    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))


    def __abs__(self):
        return linalg.norm(self.coordarray, self._normord)


    def __copy__(self):
        return Point({'coordarray': copy(self.coordarray),
                      'coordnames': copy(self.coordnames),
                      'coordtype': self.coordtype,
                      'norm': self._normord,
                      'labels': self.labels})

    copy = __copy__


    def __getstate__(self):
        d = copy(self.__dict__)
        # remove reference to Cfunc type
        d['coordtype'] = _num_type2name[self.coordtype]
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc type
        self.coordtype = _num_name2type[self.coordtype]


#----------------------------------------------------------------------------


class Pointset(Point):
    """1D parameterized or non-parameterized set of discrete points.
    (If present, the independent variable must be a float64 or an int32)"""

    def __init__(self, kwd=None, **kw):
        if kwd is not None:
            if kw != {}:
                raise ValueError("Cannot mix keyword dictionary and keywords")
            kw = kwd
            if intersect(kw.keys(), point_keys) == []:
                # creating Pointset from dictionary
                temp_kw = {}
                temp_kw['coorddict'] = copy(kw)
                kw = copy(temp_kw)
        # Deal with independent variable, if present
        if 'indepvardict' in kw:
            assert len(kw['indepvardict']) == 1
            try:
                it = kw['indepvartype']
            except KeyError:
                self.indepvartype = float64
            else:
                try:
                    self.indepvartype = _num_equivtype[it]
                except KeyError:
                    raise TypeError('Independent variable type %s not valid'%str(it))
            vals = list(kw['indepvardict'].values())[0]
            self.indepvarname = list(kw['indepvardict'].keys())[0]
            if isinstance(vals, _seq_types):
                self.indepvararray = array(vals, self.indepvartype)
            else:
                try:
                    assert self.indepvartype == _num_equivtype[type(vals)]
                except (AssertionError, KeyError):
                    raise TypeError("Invalid type for independent variable value")
                else:
                    self.indepvararray = array([vals], self.indepvartype)
        elif 'indepvararray' in kw:
            if 'indepvarname' in kw:
                self.indepvarname = kw['indepvarname']
            else:
                self.indepvarname = 't'
            vals = kw['indepvararray']
            if isinstance(vals, list):
                try:
                    it = kw['indepvartype']
                except:
                    self.indepvartype = float64
                else:
                    try:
                        self.indepvartype = _num_equivtype[it]
                    except KeyError:
                        raise TypeError('Independent variable type %s not valid'%str(it))
                self.indepvararray = array(vals, self.indepvartype)
            elif isinstance(vals, ndarray):
                # call 'array' constructor to ensure copy is made in case
                # either array is independently changed.
                if np.ndim(vals) in [0,2]:
                    self.indepvararray = array(vals.ravel())
                else:
                    self.indepvararray = array(vals)
                try:
                    self.indepvartype = _num_equivtype[self.indepvararray.dtype.type]
                except KeyError:
                    raise TypeError('Independent variable type '
                                    '%s not valid'%self.indepvararray.dtype)
            else:
                raise TypeError("Invalid type for independent variable "
                                "array: "+str(type(vals)))

        else:
            # non-parameterized case
            self.indepvarname = None
            self.indepvartype = None
            self.indepvararray = None
            self._parameterized = False
        if self.indepvarname:
            # do validation checks
            assert isinstance(self.indepvarname, str), \
                   'independent variable name must be a string'
            try:
                self.indepvartype = _num_equivtype[self.indepvararray.dtype.type]
            except KeyError:
                raise TypeError('Independent variable type '
                                    '%s not valid'%self.indepvararray.dtype)
            r = np.ndim(self.indepvararray)
            if r == 1:
                pass
            elif r == 0:
                self.indepvararray = self.indepvararray.ravel()
            else:
                raise ValueError("Invalid rank for "
                                "independent variable array %i"%r)
            # if user gave independent variable array in reverse order,
            # then we'll reverse this and the coord arrays and the labels
            # at the end of initialization
            do_reverse = not isincreasing(self.indepvararray)
            self._parameterized = True
        # Deal with coordinate data
        if 'coorddict' in kw:
            coorddict = {}
            try:
                ct = kw['coordtype']
            except KeyError:
                self.coordtype = float64
            else:
                try:
                    self.coordtype = _num_equivtype[ct]
                except KeyError:
                    raise TypeError('Coordinate type %s not valid for Point'%str(ct))
            for c, v in kw['coorddict'].items():
                if isinstance(c, str):
                    c_key = c
                else:
                    c_key = repr(c)
                if isinstance(v, list):
                    coorddict[c_key] = array(v, self.coordtype)
                elif isinstance(v, ndarray):
                    # call 'array' constructor on array to ensure it is a copy
                    # if either array is independently changed.
                    coorddict[c_key] = array(v, self.coordtype)
                elif isinstance(v, Pointset):
                    coorddict[c_key] = v.toarray()
                else:
                    try:
                        assert self.coordtype == _num_equivtype[type(v)]
                    except (AssertionError, KeyError):
                        raise TypeError("Must pass arrays, lists, or numeric types")
                    else:
                        coorddict[c_key] = array([v], self.coordtype)
            self.coordnames = list(coorddict.keys())
            # only way to order dictionary keys for array is to sort
            self.coordnames.sort()
            self.dimension = len(self.coordnames)
            datalist = []
            # loop over coordnames to ensure correct ordering of coordarray
            if self._parameterized:
                my_len = len(self.indepvararray)
            else:
                my_len = len(coorddict[self.coordnames[0]])
            for c in self.coordnames:
                xs = coorddict[c]
                if my_len != len(xs):
                    if self._parameterized:
                        raise ValueError('Independent variable array length must match '
                           'that of each coordinate array')
                    else:
                        raise ValueError('All coordinate arrays must have same length')
                datalist.append(xs)
            self.coordarray = array(datalist, self.coordtype)
            r = np.ndim(self.coordarray)
            if r == 2:
                pass
            elif r == 1:
                self.coordarray = array([self.coordarray], self.coordtype)
            elif r == 0:
                self.coordarray = array([self.coordarray.ravel()], self.coordtype)
            else:
                raise ValueError("Invalid rank for coordinate array: %i"%r)
            assert self.dimension == self.coordarray.shape[0], "Invalid coord array"
        elif 'coordarray' in kw:
            if not isinstance(kw['coordarray'], _seq_types):
                raise TypeError('Coordinate type %s not valid for Pointset'%str(type(kw['coordarray'])))
            try:
                ct = kw['coordtype']
            except KeyError:
                self.coordtype = float64
            else:
                try:
                    self.coordtype = _num_equivtype[ct]
                except KeyError:
                    raise TypeError('Coordinate type %s not valid'%str(ct))
            # calling 'array' constructor creates a copy if original or new
            # array is altered
            array_temp = array(kw['coordarray'], self.coordtype)
            r = np.ndim(array_temp)
            if r == 2:
                self.coordarray = array_temp
            elif r == 1:
                self.coordarray = array([kw['coordarray']], self.coordtype)
            elif r == 0:
                self.coordarray = array([array_temp.ravel()], self.coordtype)
            else:
                raise ValueError("Invalid rank for coordinate array %i"%r)
            self.dimension = self.coordarray.shape[0]
            if 'coordnames' in kw:
                if isinstance(kw['coordnames'], str):
                    coordnames = [kw['coordnames']]
                else:
                    coordnames = kw['coordnames']
            else:
                coordnames = [str(cix) for cix in range(self.dimension)]
            if len(coordnames) != self.dimension:
                print("Pointset initialization error:")
                print("Found Coordnames: %r(dimension = %s)" % (coordnames, len(coordnames)))
                print("vs. data dimension =%d" % self.dimension)
                raise ValueError("Mismatch between number of coordnames and "
                                 "dimension of data")
            cs = array(coordnames)
            order = cs.argsort()
            self.coordnames = cs[order].tolist()
            self.coordarray = take(self.coordarray,order,axis=0)
            self.coordtype = self.coordarray.dtype.type
        else:
            raise ValueError("Missing coord info in keywords")
        assert isUniqueSeq(self.coordnames), 'Coordinate names must be unique'
        self.makeIxMaps()
        if self._parameterized:
            assert self.indepvarname not in self.coordnames, \
                   "Independent variable name appeared in coordinate names"
            #            if len(self.coordarray.shape) > 1:
            assert self.coordarray.shape[1] == len(self.indepvararray), \
                   ("Coord array length mismatch with independent variable"
                    " array length")
            #else:
            #    assert self.coordarray.shape[0] == len(self.indepvararray)
            # process choice of indep var tolerance
            if 'checklevel' in kw:
                checklevel = kw['checklevel']
                if checklevel in [0,1]:
                    self.checklevel = checklevel
                else:
                    raise ValueError("Invalid check level")
            else:
                # default to use tolerance in indep val resolution
                self.checklevel = 1
            if 'tolerance' in kw:
                tol = kw['tolerance']
                if tol > 0:
                    self._abseps = tol
                else:
                    raise ValueError("Tolerance must be a positive real number")
            else:
                self._abseps = 1e-13
        if 'name' in kw:
            if isinstance(kw['name'], str):
                self.name = kw['name']
            else:
                raise TypeError("name argument must be a string")
        else:
            self.name = ""
        if 'norm' in kw:
            if kw['norm'] == 0:
                raise ValueError("Norm order for point cannot be zero")
            self._normord = kw['norm']
        else:
            self._normord = 2
        if 'labels' in kw:
            try:
                self.labels = PointInfo(kw['labels'].by_index)
            except AttributeError:
                self.labels = PointInfo(kw['labels'])
        else:
            self.labels = PointInfo()
        if 'tags' in kw:
            self.tags = kw['tags']
        else:
            self.tags = {}
        if self._parameterized:
            if do_reverse:
                # finish the operation of reversing the reverse-order
                # input arrays
                self.indepvararray = self.indepvararray[::-1]
                self.reverse()
            if not isincreasing(self.indepvararray):
                raise ValueError("Independent variable values must be in "
                                       "increasing order")


    def __delitem__(self, k):
        """Remove point by index or by coordinate."""
        if k in self.coordnames:
            cs = remain(self.coordnames, k)
            p_result = copy(self[cs])
            self.coordnames = cs
            self.coordarray = p_result.coordarray
            self.labels = p_result.labels
            self.indepvararray = p_result.indepvararray
            self.makeIxMaps()
        else:
            # assume integer
            self.remove(k)


    def remove(self, ix):
        """Remove individual Point by its index."""
        if ix == 0:
            try:
                p_result = copy(self[1:])
            except ValueError:
                # slice index out of range => only 1 point left!
                raise ValueError("Cannot remove only point in pointset!")
        else:
            ix = ix % len(self)
            p_result = copy(self[:ix])
            try:
                p_result.append(self[ix+1:])
            except ValueError:
                # ix was at end, so nothing left to append
                pass
        self.coordarray = p_result.coordarray
        self.labels = p_result.labels
        self.indepvararray = p_result.indepvararray
        self.makeIxMaps()


    def reverse(self):
        """Reverse order of points *IN PLACE*."""
        self.coordarray = self.coordarray[:,::-1]
        self.labels.mapIndices(dict(zip(range(0,len(self)),range(len(self)-1,-1,-1))))

    def rename(self, coord, newcoord):
        """Rename a coordinate."""
        try:
            ix = self.coordnames.index(coord)
        except ValueError:
            raise ValueError("No such coordinate: %s"%coord)
        self.coordnames[ix] = newcoord
        self.makeIxMaps()

    def makeIxMaps(self):
        self._name_ix_map = dict(zip(self.coordnames, range(self.dimension)))
        self._ix_name_map = copy(self.coordnames)
        if self._parameterized:
            self._indepvar_ix_map = makeArrayIxMap(self.indepvararray)
        else:
            self._indepvar_ix_map = None


    def addlabel(self, ix, label, info=None):
        """Add string label to indexed point. info dictionary is optional"""
        if ix < 0:
            ix = len(self)+ix
        if ix in range(len(self)):
            self.labels.update(ix, label, info)
        else:
            raise ValueError("Index out of range")


    def removelabel(self, ix):
        """Remove all labels at indexed point."""
        del self.labels[ix]


    def bylabel(self, s):
        """Return pointset containing points labelled with the supplied
        labels. Argument s can be a string or a list of strings."""
        if isinstance(s, str):
            if s == '':
                raise ValueError("Label must be non-empty")
            else:
                ixlist = sortedDictKeys(self.labels[s])
                if ixlist != []:
                    return self[ixlist]
                else:
                    return None
        elif isinstance(s, list):
            ixlist = []
            for ss in s:
                if isinstance(ss, str):
                    if ss == '':
                        raise ValueError("Label must be non-empty")
                    ixlist = sortedDictKeys(self.labels[ss])
                else:
                    raise TypeError("Invalid label type")
            if ixlist != []:
                return self[ixlist]
            else:
                return None
        else:
            raise TypeError("Invalid label type")



    def __setitem__(self, ix, p):
        """Change individual points, accessed by index (no slicing supported).
        Individual coordinate values of a point can be changed by adding a
        cross-reference coordinate name or index.
        If ix is a variable name then the entire row can be changed (again,
        no slicing supported)."""
        if isinstance(ix, _int_types):
            if isinstance(p, Point):
                if compareNumTypes(self.coordtype, int32) and \
                   compareNumTypes(p.coordtype, float64):
                    raise ValueError("Cannot update integer pointset with a float")
                self.coordarray[:,ix] = p.toarray()
                if len(p.labels) > 0:
                    self.labels.update({ix: p.labels})
            elif isinstance(p, dict):
                vlist = []
                for k in self.coordnames:
                    vlist.append(p[k])
                self.coordarray[:,ix] = array(vlist, self.coordtype)
            elif isinstance(p, _seq_types):
                self.coordarray[:,ix] = array(p, self.coordtype)
            else:
                raise TypeError("Invalid index reference")
        elif isinstance(ix, tuple) and len(ix) == 2:
            # note that index order must be reversed
            try:
                c = self._name_ix_map[ix[1]]
            except KeyError:
                c = ix[1]
            if isinstance(p, _int_types):
                self.coordarray[c,ix[0]] = p
            elif isinstance(p, _float_types):
                if self.coordtype == float64:
                    self.coordarray[c,ix[0]] = p
                else:
                    raise TypeError("Cannot update an integer pointset with a float")
            elif isinstance(p, ndarray) and p.shape==(1,):
                self.coordarray[c,ix[0]] = p[0]
            elif isinstance(p, list) and len(list) == 1:
                self.coordarray[c,ix[0]] = p[0]
            elif isinstance(p, Point) and p.dimension == 1:
                self.coordarray[c,ix[0]] = p[0]
                if len(p.labels) > 0:
                    self.labels.update({ix: p.labels})
            else:
                raise TypeError("New value is not a singleton numeric type")
        elif isinstance(ix, str):
            if ix == self.indepvarname:
                if isinstance(p, Pointset):
                    if compareNumTypes(self.indepvartype, int32) and \
                       compareNumTypes(p.indepvartype, float64):
                        raise ValueError("Cannot update integer independent variable with a float")
                    if len(self) == len(p):
                        self.indepvararray = p.toarray()
                    else:
                        raise ValueError("Size mismatch for new independent variable array")
                    # labels ignored
                elif isinstance(p, dict):
                    if len(self) == len(p[c]):
                        self.indepvararray = array(p[c], self.indepvartype)
                    else:
                        raise ValueError("Size mismatch for new independent variable array")
                elif isinstance(p, _seq_types):
                    if len(self) == len(p):
                        self.indepvararray = array(p, self.indepvartype)
                    else:
                        raise ValueError("Size mismatch for new independent variable array")
                else:
                    raise TypeError("Invalid data")
            elif ix in self.coordnames:
                c = self._name_ix_map[ix]
                if isinstance(p, Pointset):
                    if compareNumTypes(self.coordtype, int32) and \
                       compareNumTypes(p.coordtype, float64):
                        raise ValueError("Cannot update integer pointset with a float")
                    self.coordarray[c,:] = p.toarray()
                    # labels ignored
                elif isinstance(p, dict):
                    self.coordarray[c,:] = array(p[c], self.coordtype)
                elif isinstance(p, _seq_types):
                    self.coordarray[c,:] = array(p, self.coordtype)
                elif isinstance(p, _real_types):
                    self.coordarray[c,:] = float(p)
                else:
                    raise TypeError("Invalid data")
            else:
                raise TypeError("Invalid variable reference")
        else:
            raise TypeError("Invalid Pointset reference")


    def __getitem__(self, ix):
        # select points
        if isinstance(ix, _int_types):
            # The labels (PointInfo) object doesn't understand -ve indices,
            # but don't take modulo length otherwise iteration will break
            if ix < 0:
                ix = ix + self.coordarray.shape[1]
            if ix in self.labels:
                label = self.labels[ix]
            else:
                label = {}
            return Point({'coordarray': self.coordarray[:,ix],
                          'coordnames': self.coordnames,
                          'norm': self._normord,
                          'labels': label})
        elif isinstance(ix, tuple):
            if len(ix) != 2:
                raise ValueError("Only use 2-tuples in referencing pointset")
            ref1 = ix[0]
            ref2 = ix[1]
        elif isinstance(ix, str):
            # reference by coord name
            if self._parameterized:
                if ix == self.indepvarname:
                    return self.indepvararray
                else:
                    return self.coordarray[self._map_names_to_ixs(ix),:]
            else:
                return self.coordarray[self._map_names_to_ixs(ix),:]
        elif isinstance(ix, list):
            if all([x in self.coordnames for x in ix]):
                ref1 = slice(len(self))
                ref2 = ix
            else:
                ref1 = ix
                ref2 = None
        elif isinstance(ix, (ndarray, slice)):
            ref1 = ix
            ref2 = None
        else:
            raise IndexError("Illegal index %s"%str(ix))
        if isinstance(ref1, (list, ndarray, _int_types)):
            if isinstance(ref1, _int_types):
                ref1 = [ref1]
            try:
                ca = take(self.coordarray, ref1, axis=1)
            except ValueError:
                raise ValueError("Invalid variable names given: "%(str(ref1)))
            try:
                ci = take(self.indepvararray, ref1, axis=0)
            except (IndexError, AttributeError):
                # non-parameterized pointset
                pass
            cl = self.labels[ref1]
            cl_ixs = cl.getIndices()
            ixmap = invertMap(ref1)
            new_cl_ixs = [ixmap[i] for i in cl_ixs]
        elif isinstance(ref1, slice):
            ls = len(self)
            if ref1.stop is None:
                stop = ls
            else:
                if ref1.stop < 0:
                    stop = ref1.stop + self.coordarray.shape[1] + 1
                else:
                    stop = ref1.stop
            s1, s2, s3 = ref1.indices(stop)
            if s1 < 0 or s2 > ls or s1 >= ls:
                raise ValueError("Slice index out of range")
            ca = take(self.coordarray, range(s1, s2, s3), axis=1)
            try:
                ci = take(self.indepvararray, range(s1, s2, s3),axis=0)
            except (IndexError, AttributeError):
                # non-parameterized pointset
                pass
            cl = self.labels[ref1]
            cl_ixs = cl.getIndices()
            lowest_ix = ref1.start or 0
            if lowest_ix < 0:
                lowest_ix = len(self)+lowest_ix
            new_cl_ixs = [i-lowest_ix for i in cl_ixs]
        else:
            print("ref1 argument =%r" % ref1)
            raise TypeError("Type %s is invalid for Pointset indexing"%str(type(ref1)))
        ixlist = self._force_coords_to_ixlist(ref2)
        ca = take(ca, ixlist, axis=0)
        try:
            cl.mapIndices(dict(zip(cl_ixs, new_cl_ixs)))
        except AttributeError:
            pass
        if self._parameterized:
            return Pointset({'coordarray': ca,
                             'coordnames': [self.coordnames[i] for i in ixlist],
                             'indepvararray': ci,
                             'indepvarname': self.indepvarname,
                             'norm': self._normord,
                             'labels': cl})
        else:
            return Pointset({'coordarray': ca,
                            'coordnames': [self.coordnames[i] for i in ixlist],
                            'norm': self._normord,
                            'labels': cl})


    def _resolve_indepvar(self, p):
        if self.checklevel == 0:
            return self._indepvar_ix_map[p]
        else:
            try:
                return self._indepvar_ix_map[p]
            except:
                ixs = self.findIndex(p)
                lval = self.indepvararray[ixs[0]]
                rval = self.indepvararray[ixs[1]]
                if p - lval < self._abseps:
                    return ixs[0]
                elif rval - p <= self._abseps:
                    return ixs[1]
                else:
                    lerr = p - lval
                    rerr = rval - p
                    raise KeyError( \
                  "%f not found in (%f, %f) @tol=%.16f: mismatches=(%.16f, %.16f)"%(p,lval,rval,self._abseps,lerr,rerr))


    def setTol(self, tol):
        if tol > 0:
            self._abseps = tol
        else:
            raise ValueError("tolerance must be a positive real number")


    def __call__(self, p, coords=None):
        if not self._parameterized:
            raise TypeError("Cannot call a non-parameterized Pointset")
        if isinstance(p, _seq_types):
            # assume p is an all-numeric list, so it should be treated as
            # an independent variable.
            try:
                ix = [self._resolve_indepvar(i) for i in p]
            except KeyError:
                raise ValueError("Independent variable value not valid: %s"%str(p))
        else:
            # assume p is an integer or float, appropriate to independent var
            try:
                ix = self._resolve_indepvar(p)
            except KeyError:
                raise ValueError("Independent variable value not valid: " \
                                 + str(p))
        if coords is None:
            if isinstance(ix, _int_types):
                label = self.labels[ix]
                try:
                    label.mapIndices({ix: 0})
                except AttributeError:
                    # empty
                    pass
                return Point({'coordarray': self.coordarray[:,ix],
                              'coordnames': self.coordnames,
                              'norm': self._normord,
                              'labels': label})
            else:
                labels = self.labels[ix]
                cl_ixs = labels.getIndices()
                ixmap = invertMap(ix)
                new_cl_ixs = [ixmap[i] for i in cl_ixs]
                if isinstance(ix, slice):
                    lowest_ix = ix.start or 0
                    new_cl_ixs = [i-lowest_ix for i in cl_ics]
                elif isinstance(ix, (list, ndarray)):
                    new_cl_ixs = [ixmap[i] for i in cl_ixs]
                try:
                    labels.mapIndices(dict(zip(cl_ixs, new_cl_ixs)))
                except AttributeError:
                    # empty
                    pass
                return Pointset({'coordarray': take(self.coordarray, ix, axis=1),
                         'coordnames': self.coordnames,
                         'indepvarname': self.indepvarname,
                         'indepvararray': take(self.indepvararray, ix, axis=0),
                         'norm': self._normord,
                         'labels': labels})
        else:
            clist = self._force_coords_to_ixlist(coords)
            if isinstance(ix, _int_types):
                label = self.labels[ix]
                try:
                    label.mapIndices({ix: 0})
                except AttributeError:
                    # empty
                    pass
                return Point({'coordarray': self.coordarray[clist, ix],
                          'coordnames': [self.coordnames[i] for i in clist],
                          'norm': self._normord,
                          'labels': label})
            else:
                labels = self.labels[ix]
                try:
                    labels.mapIndices(dict(zip(labels, [i-ix[0] for i in labels.getIndices()])))
                except AttributeError:
                    # empty
                    pass
                return Pointset({'coordarray': take(self.coordarray[clist], ix, axis=1),
                                 'coordnames': [self.coordnames[i] for i in clist],
                                 'indepvarname': self.indepvarname,
                                 'indepvararray': take(self.indepvararray, ix, axis=0),
                                 'norm': self._normord,
                                 'labels': labels})


    def __len__(self):
        return self.coordarray.shape[1]


    def __contains__(self, other):
        for i in range(len(self)):
            if comparePointCoords(self.__getitem__(i), other):
                return True
        return False


    def __lt__(self, other):
        if isinstance(other, Pointset):
            if not all(self.indepvararray == other.indepvararray):
                raise ValueError("Independent variable arrays are not the same")
            return array([self[i] < other[i] for i in range(len(self))], dtype=np.bool_)
        elif isinstance(other, Point):
            return array([p < other for p in self], dtype=np.bool_)
        else:
            try:
                return self.coordarray < other
            except:
                raise TypeError("Invalid type for comparison with Pointset")

    def __gt__(self, other):
        if isinstance(other, Pointset):
            if not all(self.indepvararray == other.indepvararray):
                raise ValueError("Independent variable arrays are not the same")
            return array([self[i] > other[i] for i in range(len(self))], dtype=np.bool_)
        elif isinstance(other, Point):
            return array([p > other for p in self], dtype=np.bool_)
        else:
            try:
                return self.coordarray > other
            except:
                raise TypeError("Invalid type for comparison with Pointset")

    def __le__(self, other):
        if isinstance(other, Pointset):
            if not all(self.indepvararray == other.indepvararray):
                raise ValueError("Independent variable arrays are not the same")
            return array([self[i] <= other[i] for i in range(len(self))], dtype=np.bool_)
        elif isinstance(other, Point):
            return array([p <= other for p in self], dtype=np.bool_)
        else:
            try:
                return self.coordarray <= other
            except:
                raise TypeError("Invalid type for comparison with Pointset")

    def __ge__(self, other):
        if isinstance(other, Pointset):
            if not all(self.indepvararray == other.indepvararray):
                raise ValueError("Independent variable arrays are not the same")
            return array([self[i] >= other[i] for i in range(len(self))], dtype=np.bool_)
        elif isinstance(other, Point):
            return array([p >= other for p in self], dtype=np.bool_)
        else:
            try:
                return self.coordarray >= other
            except:
                raise TypeError("Invalid type for comparison with Pointset")

    __hash__ = None

    def __eq__(self, other):
        if isinstance(other, Pointset):
            if not all(self.indepvararray == other.indepvararray):
                raise ValueError("Independent variable arrays are not the same")
            return array([self[i] == other[i] for i in range(len(self))], dtype=np.bool_)
        elif isinstance(other, Point):
            return array([p == other for p in self], dtype=np.bool_)
        else:
            try:
                return self.coordarray == other
            except:
                raise TypeError("Invalid type for comparison with Pointset")

    def __ne__(self, other):
        if isinstance(other, Pointset):
            if not all(self.indepvararray == other.indepvararray):
                raise ValueError("Independent variable arrays are not the same")
            return array([self[i] != other[i] for i in range(len(self))], dtype=np.bool_)
        elif isinstance(other, Point):
            return array([p != other for p in self], dtype=np.bool_)
        else:
            try:
                return self.coordarray != other
            except:
                raise TypeError("Invalid type for comparison with Pointset")


    def insert(self, parg, ix=None):
        """Insert individual Point or Pointset before the given index.

        If ix is not given then the source and target Pointsets must
        be parameterized. In this case the Point or Pointset will be
        inserted according to the ordering of independent variable
        values."""
        p=copy(parg)
        if ix is None:
            if self._parameterized:
                if isinstance(p, Point) and self.indepvarname in p.coordnames:
                    t = p[self.indepvarname]
                    tix = self.find(t)
                    if isinstance(tix, tuple):
                        self.insert(p, tix[1])
                    else:
                        # tix was an integer, meaning that t is
                        # already present in Pointset
                        raise ValueError("Point at independent variable"
                                         "value %f already present"%t)
                elif isinstance(p, Pointset) and p._parameterized and \
                       p.indepvarname == self.indepvarname:
                    # Don't do a straight self.insert call in case the
                    # new indep var values need to be interleaved with
                    # the present ones.
                    #
                    # convert self.indepvararray and self.coordarray into lists (by self.todict())
                    iva = self.indepvararray.tolist()
                    vd = self.todict(aslist=True)
                    # get list of findIndex results for each of p indepvar vals
                    # add i for each one because each previous one will have been inserted,
                    # increasing the length of self.
                    if len(intersect(self._ix_name_map, p._ix_name_map)) != self.dimension:
                        raise ValueError("Dimension mismatch with inserted Pointset")
                    iva_p = p.indepvararray
                    lenp = len(p)
                    vd_p = p.todict()
                    try:
                        s_ixs = [self.findIndex(iva_p[i])[1]+i for i in range(lenp)]
                    except TypeError:
                        raise ValueError("Independent variable "
                                         "values in Pointset already present")
                    p_label_ixs = p.labels.getIndices()
                    s_label_ixs = self.labels.getIndices()
                    sLabelMap = {}
                    pLabelMap = {}
                    for i in range(lenp):
                        s_ix = s_ixs[i]
                        if i in p_label_ixs:
                            pLabelMap[i] = s_ix
                        for s_label_ix in s_label_ixs:
                            if s_label_ix >= s_ix-i:
                                sLabelMap[s_label_ix] = s_label_ix+i+1
                    # for each one, list-insert new point data
                    for p_ix in range(lenp):
                        s_ix = s_ixs[p_ix]
                        iva.insert(s_ix, iva_p[p_ix])
                        for k in self._ix_name_map:
                            vd[k].insert(s_ix, vd_p[k][p_ix])
                    # restore self's arrays
                    self.indepvararray = array(iva)
                    datalist = []
                    for c in p._ix_name_map:
                        datalist.append(vd[c])
                    self.coordarray = array(datalist, self.coordtype)
                    # update labels
                    self.labels.mapIndices(sLabelMap)
                    p_labels = copy(p.labels)
                    p_labels.mapIndices(pLabelMap)
                    self.labels.update(p_labels)
                else:
                    raise TypeError("Inserted Point/Pointset must be "
                                    "parameterized and share same independent"
                                    "parameter name")
            else:
                raise TypeError("Source Pointset must be parameterized")
        else:
            if ix > 0:
                p_result = copy(self[:ix])
                p_result.append(p)
            else:
                p_result = pointsToPointset(p, self.indepvarname)
            try:
                p_result.append(self[ix:])
            except ValueError:
                # ix > greatest index, so no points left to add
                # (i.e., p was appended to end)
                pass
            self.coordarray = p_result.coordarray
            self.labels = p_result.labels
            self.indepvararray = p_result.indepvararray
        self.makeIxMaps()


    def append(self, parg, t=None, skipMatchingIndepvar=False):
        """Append individual Point or Pointset in place.

        skipMatchingIndepvar option causes a matching independent
        variable value at the beginning of p to be skipped (only
        meaningful for appending parameterized Pointsets). This
        option is mainly for internal use."""

        # test isinstance for Pointset first because it is a sub-class of Point
        # and so isinstance(p, Point) will also catch Pointsets!
        p = copy(parg)
        if isinstance(p, Pointset):
            assert p._parameterized == self._parameterized, "Parameterization mismatch"
            # check p dimension and coordnames and type
            if compareNumTypes(self.coordtype, int32) and \
               compareNumTypes(p.coordtype, float64):
                raise TypeError("Cannot add float64 pointset to an int32 Pointset")
            pdim = p.dimension
            if self._parameterized:
                if t is None:
                    if self.indepvarname in p.coordnames:
                        t = p[self.indepvarname]
                        pdim = pdim - 1
                    elif self.indepvarname == p.indepvarname:
                        t = p.indepvararray
                    else:
                        raise ValueError("Independent variable missing from Pointset")
                    if t[0] == self.indepvararray[-1] and skipMatchingIndepvar:
                        tval = t[1:]
                        start_ix = 1
                    else:
                        tval = t
                        start_ix = 0
                    if len(tval) > 0 and tval[0] <= self.indepvararray[-1]:
                        #print tval[0], " <= ", self.indepvararray[-1]
                        raise ValueError("Independent variable value too small to add pointset")
                    added_len = len(tval)
                else:
                    if t[0] == self.indepvararray[-1] and skipMatchingIndepvar:
                        tval = t[1:]
                        start_ix = 1
                    else:
                        tval = t[:]  # ensures tval is an array (t might be a Pointset)
                        start_ix = 0
                    if len(tval) > 0 and tval[0] <= self.indepvararray[-1]:
                        #print tval[0], " <= ", self.indepvararray[-1]
                        raise ValueError("Independent variable value too small to add pointset")
                    added_len = len(tval)
            else:
                if t is not None:
                    raise TypeError("t argument cannot be used for non-parameterized pointsets")
                added_len = p.coordarray.shape[1]
                start_ix = 0
            assert pdim == self.dimension, "Dimension mismatch with Pointset"
            if pdim < p.dimension:
                pcoords = copy(p.coordnames)
                pcoords.remove(p.indepvarname)
            else:
                pcoords = p.coordnames
            if remain(pcoords, self.coordnames) != []:
                raise ValueError("Coordinate name mismatch with Pointset")
            old_len = self.coordarray.shape[1]
            new_len = old_len + added_len
            old_coords = self.coordarray
            self.coordarray = zeros((self.dimension, new_len),
                                    self.coordarray.dtype)
            if self._parameterized:
                self.indepvararray.resize(new_len)
                tvals = tval[list(range(added_len))]
                self.indepvararray[old_len:] = tvals
            for tix in range(old_len):
                self.coordarray[:, tix] = old_coords[:, tix]
            pdict = p.todict()
            self.coordarray[:, old_len:] = r_[[pdict[c][start_ix:] for c in self._ix_name_map]]
            p_labels = copy(p.labels)
            pixs = p.labels.getIndices()
            if start_ix == 1:
                p_labels.mapIndices(dict(zip(pixs, [i+old_len-1 for i in pixs])))
            else:
                p_labels.mapIndices(dict(zip(pixs, [i+old_len for i in pixs])))
            self.labels.update(p_labels)
        elif isinstance(p, Point):
            # check p dimension and coordnames and type
            if compareNumTypes(self.coordtype, int32) and \
               compareNumTypes(p.coordtype, float64):
                raise TypeError("Cannot add float64 Point to an int32 Pointset")
            pdim = p.dimension
            if self._parameterized:
                if t is None:
                    if self.indepvarname not in p.coordnames:
                        raise ValueError("Independent variable missing from Point")
                    else:
                        tval = p[self.indepvarname]
                        if tval <= self.indepvararray[-1]:
                            raise ValueError("Independent variable value too small to add Point")
                        pdim = pdim - 1
                else:
                    if t <= self.indepvararray[-1]:
                        raise ValueError("Independent variable value too small to add Point")
                    tval = t
            elif t is not None:
                raise TypeError("t argument cannot be used for non-parameterized Pointsets")
            assert pdim == self.dimension, "Dimension mismatch with Point"
            if pdim < p.dimension:
                pcoords = copy(p.coordnames)
                if self._parameterized:
                    pcoords.remove(self.indepvarname)
            else:
                pcoords = p.coordnames
            if remain(pcoords, self.coordnames) != []:
                raise ValueError("Coordinate name mismatch with Point")
            new_len = self.coordarray.shape[1]+1
            old_coords = self.coordarray
            self.coordarray = zeros((self.dimension, new_len), self.coordarray.dtype)
            if self._parameterized:
                self.indepvararray.resize(new_len)
                self.indepvararray.resize(new_len)
                self.indepvararray[new_len-1] = tval
            for tix in range(new_len-1):
                self.coordarray[:, tix] = old_coords[:, tix]
            for ix in range(self.dimension):
                self.coordarray[ix,new_len-1] = p(self._ix_name_map[ix])
            if len(p.labels) > 0:
                self.labels.update({new_len-1: p.labels})
        else:
            raise TypeError("append requires Point or Pointset argument")
        self.makeIxMaps()


    extend = append   # for intuitive compatibility!


    def toarray(self, include_indepvar=False):
        """Convert the pointset to a D x L array (one variable per row),
        where D is the dimension of the pointset and L is its length.
        (This is a copy of the internal attribute 'coordarray'.)

        If the optional include_indepvar switch is set True (default False),
        the first row is the independent variable, and the whole
        array will be (D+1) x L in shape.
        """
        if self.dimension==1:
            ca = copy(self.coordarray[0])
        else:
            ca = copy(self.coordarray)
        if include_indepvar:
            ia = copy(self.indepvararray)
            ia.shape = (1, len(ia))
            return concatenate((ia,ca))
        else:
            return ca


    def todict(self, aslist=False):
        """Convert Pointset to a dictionary of arrays (or of lists with aslist=True)."""
        if aslist:
            d = dict(zip(self._ix_name_map, self.coordarray.tolist()))
        else:
            d = dict(zip(self._ix_name_map, self.coordarray))
            if self._parameterized:
                d[self.indepvarname] = self.indepvararray
        return d


    def _infostr(self, verbose=0):
        if self.name == '':
            outputStr = "Pointset <no name>"
        else:
            outputStr = "Pointset " + self.name
        if self._parameterized:
            outputStr += " (parameterized)"
        else:
            outputStr += " (non-parameterized)"
        ## The following if statement implicitly passes for
        # the verbose < 0 case
        if verbose == 0:
            outputStr += "Coordinates: %s" % str(self.coordnames)
        elif verbose > 0:
            precision = 8
            lenv = len(self)
            if lenv > 8:
                ixslo = list(range(0,2))
                ixshi = list(range(lenv-2,lenv))
            outputStr += "\n"
            if self._parameterized:
                iv = self.indepvararray
                if not isinstance(iv, ndarray):
                    iv = array(iv, self.indepvartype)  # permits slicing (lists don't)
                if lenv > 8:
                    alo = array2string(iv[ixslo],precision=precision)
                    ahi = array2string(iv[ixshi],precision=precision)
                    ivstr = alo[:-1] + ", ..., " + ahi[1:]
                else:
                    ivstr = array2string(iv,precision=precision)
                outputStr += "Independent variable:\n"
                outputStr += self.indepvarname + ':  '+ivstr+"\n"
            outputStr += "Coordinates:\n"
            for c in self.coordnames:
                v = self.coordarray[self._map_names_to_ixs(c)]
                if not isinstance(v, ndarray):
                    # only alternative is a singleton numeric value (not a list)
                    v = array([v], self.coordtype)
                if lenv > 8:
                    alo = array2string(v[ixslo],precision=precision)
                    ahi = array2string(v[ixshi],precision=precision)
                    dvstr = alo[:-1] + ", ..., " + ahi[1:]
                else:
                    dvstr = array2string(v, precision=precision)
                outputStr += c+':  '+dvstr
                if c != self.coordnames[-1]:
                    outputStr += "\n"
            outputStr += "\nLabels by index: " + self.labels._infostr(17)
        return outputStr


    def __repr__(self):
        return self._infostr(verbose=1)


    def __str__(self):
        return self._infostr(verbose=0)


    def info(self, verboselevel=1):
        print(self._infostr(verboselevel))


    def __copy__(self):
        if self._parameterized:
            return Pointset({'coordarray': copy(self.coordarray),
                         'coordnames': copy(self.coordnames),
                         'indepvarname': copy(self.indepvarname),
                         'indepvararray': copy(self.indepvararray),
                         'norm': self._normord,
                         'labels': copy(self.labels)
                         })
        else:
            return Pointset({'coordarray': copy(self.coordarray),
                         'coordnames': copy(self.coordnames),
                         'norm': self._normord,
                         'labels': copy(self.labels)})

    copy = __copy__


    def __getstate__(self):
        d = copy(self.__dict__)
        # remove reference to Cfunc types by converting them to strings
        try:
            d['indepvartype'] = _num_type2name[self.indepvartype]
        except KeyError:
            # non-parameterized Pointset
            pass
        d['coordtype'] = _num_type2name[self.coordtype]
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        # reinstate Cfunc types
        try:
            self.indepvartype = _num_name2type[self.indepvartype]
        except KeyError:
            # non-parameterized Pointset
            pass
        self.coordtype = _num_name2type[self.coordtype]

    def _match_indepvararray(self, other):
        """Verifies the matching of independent variable arrays in two pointsets.
        Does nothing if either object is not a parameterized pointset."""
        try:
            if other._parameterized and self._parameterized:
                if not all(self.indepvararray == other.indepvararray):
                    print(self.indepvararray)
                    print(other.indepvararray)
                    raise ValueError("Mismatched independent variable arrays")
        except AttributeError:
            pass

    def __add__(self, other):
        self._match_indepvararray(other)
        return Point.__add__(self, other)

    def __radd__(self, other):
        self._match_indepvararray(other)
        return Point.__radd__(self, other)

    def __sub__(self, other):
        self._match_indepvararray(other)
        return Point.__sub__(self, other)

    def __rsub__(self, other):
        self._match_indepvararray(other)
        return Point.__rsub__(self, other)

    def __mul__(self, other):
        self._match_indepvararray(other)
        return Point.__mul__(self, other)

    def __rmul__(self, other):
        self._match_indepvararray(other)
        return Point.__rmul__(self, other)

    def __div__(self, other):
        self._match_indepvararray(other)
        return Point.__div__(self, other)

    def __rdiv__(self, other):
        self._match_indepvararray(other)
        return Point.__rdiv__(self, other)

    def find(self, indepval, end=None):
        """find returns an integer index for where to place
        a point having independent variable value <indepval> in
        the Pointset, if <indepval> already exists. Otherwise, a
        pair indicating the nearest independent variable values
        present in the Pointset is returned.

        To ensure an integer is always returned, choose a left or
        right side to choose from the pair, using end=0 or 1 respectively."""
        if not self._parameterized:
            raise TypeError("Cannot find index from independent variable for "
                            "a non-parameterized Pointset")
        try:
            ix = self.indepvararray.tolist().index(indepval)
            result = ix
        except ValueError:
            cond = less(self.indepvararray, indepval).tolist()
            try:
                ix = cond.index(0)
                result = (ix-1, ix)
            except ValueError:
                result = (len(self.indepvararray)-1, len(self.indepvararray))
            if end is not None:
                result = result[end]
        return result

    # deprecated
    findIndex = find


# ----------------------------------------------------------------------------


class PointInfo(object):
    """Structure for storing individual point labels and information
    dictionaries within a Pointset object.

    This class will not know the size of the Pointset it is associated with,
    so index upper limits will not be checked in advance.

    Do not use a PointInfo object as an iterator, as it is 'infinite' in size!
    (It uses DefaultDicts as its internal storage, which return {} for
    undefined labels.)"""

    def __init__(self, ptlabels=None):
        if ptlabels is None:
            self.by_label = defaultdict(dict)
            self.by_index = defaultdict(dict)
        elif isinstance(ptlabels, PointInfo):
            self.by_label = ptlabels.by_label
            self.by_index = ptlabels.by_index
        elif isinstance(ptlabels, dict):
            # always expect the dictionary to be based on index
            self.by_label = defaultdict(dict)
            self.by_index = defaultdict(dict)
            for k, v in ptlabels.items():
                if not isinstance(k, _int_types):
                    raise TypeError("Initialization dictionary must be keyed "
                                    "by integer indices")
                if isinstance(v, str):
                    self.by_label[v][k] = {}
                    self.by_index[k][v] = {}
                else:
                    for label, infodict in v.items():
                        self.by_label[label][k] = infodict
                        self.by_index[k][label] = infodict
        else:
            raise TypeError("Invalid labels at initialization of PointInfo")


    def mapIndices(self, ixMapDict):
        by_index = {}
        ixMap = symbolMapClass(ixMapDict)
        for ix, rest in self.by_index.items():
            by_index[ixMap(ix)] = rest
        self.__init__(by_index)


    def mapNames(self, themap):
        """Map labels, using a symbol map of class symbolMapClass."""
        self.by_label = mapNames(themap, self.by_label)
        new_by_index = {}
        for ix, labdict in self.by_index.items():
            new_by_index[ix] = mapNames(themap, labdict)
        self.by_index = new_by_index


    def sortByIndex(self):
        ixkeys = sortedDictKeys(self.by_index)
        return list(zip(ixkeys,[self.by_index[ix] for ix in ixkeys]))


    def sortByLabel(self):
        labelkeys = sortedDictKeys(self.by_label)
        return list(zip(labelkeys,[self.by_label[label] for label in labelkeys]))


    def getIndices(self):
        return sortedDictKeys(self.by_index)


    def getLabels(self):
        return sortedDictKeys(self.by_label)


    def __contains__(self, key):
        return key in self.by_index or key in self.by_label


    def __getitem__(self, key):
        # indices already are enforced to be integers, and labels strings,
        # so this is a safe way to search!
        # Note: if don't use if-then test then defaultdict will
        # create an empty entry for the failed key when .values() is called!
        if isinstance(key, tuple):
            raise TypeError("Can only reference PointInfo with a single key")
        else:
            if isinstance(key, (slice, list, ndarray)):
                if isinstance(key, slice):
                    self_ixs = self.getIndices()
                    if len(self_ixs) == 0:
                        max_ixs = 0
                    else:
                        max_ixs = max(self_ixs)
                    stop = key.stop or max_ixs+1
                    try:
                        s1, s2, s3 = key.indices(stop)
                        ixs = range(s1, s2, s3)
                        key = intersect(ixs, self_ixs)
                    except TypeError:
                        key = self_ixs
                else:
                    if all([isinstance(k, str) for k in key]):
                        keylabels = intersect(key, self.getLabels())
                        key = []
                        for l in keylabels:
                            key.extend(list(self.by_label[l].keys()))
                        key = makeSeqUnique(key)
                    elif all([isinstance(k, _int_types) for k in key]):
                        key = intersect(key, self.getIndices())
                    else:
                        raise TypeError("Invalid key type for PointInfo")
                return PointInfo(dict(zip(key,[self.by_index[i] for i in key])))
            elif key in self.by_index:
                return self.by_index[key]
            elif key in self.by_label:
                return self.by_label[key]
            elif isinstance(key, int) and key < 0:
                raise IndexError("Cannot use negative indices for PointInfo")
            else:
                return {}


    def __setitem__(self, key1, the_rest):
        if isinstance(the_rest, tuple) and len(the_rest) == 2:
            if isinstance(the_rest[0], str):
                label = the_rest[0]
                ix = None
            elif isinstance(the_rest[0], _int_types):
                ix = the_rest[0]
                label = None
            else:
                raise TypeError("String expected for label")
            if isinstance(the_rest[1], dict):
                info = copy(the_rest[1])
            else:
                raise TypeError("Dictionary expected for info")
        elif isinstance(the_rest, str):
            label = the_rest
            ix = None
            info = {}
        elif isinstance(the_rest, _int_types):
            ix = the_rest
            label = None
            info = {}
        elif isinstance(the_rest, list):
            self.__setitem__(key1, the_rest[0])
            for item in the_rest[1:]:
                if isinstance(item, tuple) and len(item) == 2:
                    self.update(key1, item[0], item[1])
                else:
                    self.update(key1, item)
            return
        else:
            raise TypeError("Invalid item to set in PointInfo")
        if isinstance(key1, _int_types):
            if label is None:
                raise TypeError("Label expected")
            ix = key1
        elif isinstance(key1, str):
            if ix is None:
                raise TypeError("Index expected")
            label = key1
        if ix < 0:
            raise IndexError("Index must be non-negative")
        try:
            self.by_label[label].update({ix: info})
        except KeyError:
            self.by_label[label] = {ix: info}
        try:
            self.by_index[ix].update({label: info})
        except KeyError:
            self.by_index[ix] = {label: info}


    def __len__(self):
        return len(self.by_index)


    def remove(self, key1, *key2):
        """remove one or more items, keyed either by index or label."""
        byix = key1 in self.by_index
        if key2 == ():
            # remove all labels associated with index, or vice versa
            if byix:
                key2 = list(self.by_index[key1].keys())
            else:
                key2 = list(self.by_label[key1].keys())
        if byix:
            for k in key2:
                # have to check k in dict otherwise defaultdict creates entry!
                if k in self.by_label:
                    del self.by_index[key1][k]
                    del self.by_label[k][key1]
                else:
                    raise KeyError("Label not found")
                if self.by_label[k] == {}:
                    del self.by_label[k]
            if self.by_index[key1] == {}:
                del self.by_index[key1]
        else:
            for k in key2:
                # have to check k in dict otherwise defaultdict creates entry!
                if k in self.by_index:
                    del self.by_index[k][key1]
                    del self.by_label[key1][k]
                else:
                    raise KeyError("Index not found")
                if self.by_index[k] == {}:
                    del self.by_index[k]
            if self.by_label[key1] == {}:
                del self.by_label[key1]


    def update(self, key1, key2=None, info=None):
        if isinstance(key1, PointInfo):
            if key2 is None and info is None:
                for k, v in key1.by_index.items():
                    for vk, vv in v.items():
                        self.update(k, vk, vv)
            else:
                raise TypeError("Invalid calling sequence to update")
        elif isinstance(key1, dict):
            if key2 is None and info is None:
                for k, v in key1.items():
                    if isinstance(k, _int_types):
                        if k < 0:
                            k = k + len(self.by_index)
                        if isinstance(v, str):
                            k2 = v
                            k3 = {}
                            self.update(k, k2, k3)
                        elif isinstance(v, tuple) and len(v)==2:
                            k2 = v[0]
                            k3 = v[1]
                            self.update(k, k2, k3)
                        elif isinstance(v, dict):
                            for k2, k3 in v.items():
                                self.update(k, k2, k3)
                        else:
                            raise ValueError("Invalid data for update")
                    else:
                        raise TypeError("Invalid index for label")
            else:
                raise TypeError("Invalid calling sequence to update")
        elif isinstance(key1, _int_types):
            if info is None:
                info = {}
            if key1 in self.by_index:
                if key2 in self.by_index[key1]:
                    self.by_index[key1][key2].update(info)
                else:
                    self.__setitem__(key1, (key2, info))
            else:
                self.__setitem__(key1, (key2, info))
        elif isinstance(key1, str):
            if info is None:
                info = {}
            if key1 in self.by_label:
                if key2 in self.by_label[key1]:
                    self.by_label[key1][key2].update(info)
                else:
                    self.__setitem__(key2, (key1, info))
            else:
                self.__setitem__(key2, (key1, info))
        else:
            raise TypeError("Invalid type for update")


    def __delitem__(self, key):
        if key in self.by_index:
            labels = list(self.by_index[key].keys())
            del self.by_index[key]
            for label in labels:
                del self.by_label[label][key]
                if self.by_label[label] == {}:
                    del self.by_label[label]
        elif key in self.by_label:
            ixs = list(self.by_label[key].keys())
            del self.by_label[key]
            for ix in ixs:
                del self.by_index[ix][key]
                if self.by_index[ix] == {}:
                    del self.by_index[ix]
        else:
            raise KeyError("Index or label not found")

    __hash__ = None

    def __eq__(self, other):
        try:
            return (sorted(self.by_index.keys()) == sorted(other.by_index.keys()) and
                   sorted(self.by_label.keys()) == sorted(other.by_label.keys()))
        except AttributeError:
            raise TypeError("Invalid type for comparison to PointInfo")


    def __ne__(self, other):
        return not self.__eq__(other)

    def _infostr(self, tab=0):
        lenself = len(self)
        tabstr = " "*tab
        basestr = ",\n"+tabstr
        if lenself > 0:
            entries = self.sortByIndex()
            if lenself > 8:
                return basestr.join([_pretty_print_label(i) for i in entries[0:3]]) + ",\n" +\
                       (tabstr + " .\n")*3 + tabstr +\
                       basestr.join([_pretty_print_label(i) for i in entries[-3:]])
            else:
                return basestr.join([_pretty_print_label(i) for i in entries])
        else:
            return "Empty"


    def __repr__(self):
        return self._infostr()

    __str__ = __repr__


def _pretty_print_label(d):
    """Internal utility to pretty print point label info."""
    s = " %s: "%repr(d[0])
    entry_keys = list(d[1].keys())
    ki = 0
    kimax = len(entry_keys)
    for k in entry_keys:
        keys = list(d[1][k].keys())
        if len(keys) == 0:
            s += "{%s: {}}"%k
        else:
            s += "{%s: {keys=%s}}"%(k,",".join(keys))
        if ki < kimax-1:
            s += ', '
        ki += 1
    return s

# ------------------------------------------------


def comparePointCoords(p1, p2, fussy=False):
    """Compare two Points, Pointsets, or dictionary of point data, coordinate-wise.
    If p1 or p2 are Pointsets, their independent variable values, if present, are
    *not* compared.

    fussy option causes point norm order and coordinate types to be
    checked too (requires both arguments to be Points or Pointsets).
    """
    try:
        p1d = dict(p1)
        p1dk = list(p1d.keys())
        p2d = dict(p2)
        p2dk = list(p2d.keys())
    except:
        raise TypeError("Invalid Points, Pointsets, or dictionaries passed "
                        "to comparePointCoords")
    test1 = alltrue([ks[0]==ks[1] for ks in zip(p1dk, p2dk)])
    test2 = alltrue([vs[0]==vs[1] for vs in \
                 zip([p1d[k] for k in p1dk], [p2d[k] for k in p2dk])])
    if fussy:
        try:
            test3 = p1._normord == p2._normord
            test4 = compareNumTypes(p1.coordtype, p2.coordtype)
            return test1 and test2 and test3 and test4
        except AttributeError:
            raise TypeError("Invalid Points, Pointsets, or dictionaries passed "
                            "to comparePointCoords with fussy option")
    else:
        return test1 and test2


def isparameterized(p):
    """Returns True if Point or Pointset p is parameterized, False otherwise"""
    return p._parameterized


def makeNonParameterized(p):
    """Return a new Pointset stripped of its parameterization.
    """
    if isinstance(p, Pointset) and p._isparameterized:
        return Pointset({'coordarray': copy(p.coordarray),
                         'coordnames': copy(p.coordnames),
                         'norm': p._normord,
                         'labels': copy(p.labels)})
    else:
        raise TypeError("Must provide a parameterized Pointset")


def pointsToPointset(pointlist, indepvarname='', indepvararray=None,
                     indepvartype=float, norm=2):
    """Generate a Pointset from a list of Point objects (or a singleton Point).

    Include a name for the independent variable if constructing a
    parameterized pointset. The independent variable should be a
    coordinate of the Points passed, otherwise it can be passed as the
    optional third argument.
    """

    if not isinstance(indepvarname, str):
        raise TypeError("String expected for independent variable name")
    if isinstance(pointlist, Point):
        pointlist = [pointlist]
    coordnames = []
    ptype = ''
    paramd = indepvarname != ""
    if not paramd and indepvararray is not None:
        raise ValueError("Must supply independent variable name for "
                         "parameterized Pointset")
    if paramd and indepvararray is None:
        iv = []
    i = 0
    labels = {}
    for p in pointlist:
        assert isinstance(p, Point), \
               "pointlist argument must only contain Points"
        if coordnames == []:
            ptype = p.coordtype
            pdim = p.dimension
            coordnames = p.coordnames
            xcoordnames = copy(coordnames)
            if paramd and indepvararray is None:
                assert indepvarname in coordnames, \
                    "Independent variable name missing"
                del xcoordnames[xcoordnames.index(indepvarname)]
            dv = {}.fromkeys(xcoordnames)
            for c in xcoordnames:
                dv[c] = []
            if p.labels != {}:
                labels.update({0: p.labels})
                i += 1
        else:
            # coerce ints to float types if mixed
            if compareNumTypes(ptype, int32):
                if compareNumTypes(p.coordtype, float64):
                    ptype = float64
                elif compareNumTypes(p.coordtype, int32):
                    pass
                else:
                    raise TypeError("Type mismatch in points")
            elif compareNumTypes(ptype, float64):
                if not compareNumTypes(p.coordtype, (float64, int32)):
                    raise TypeError("Type mismatch in points")
            else:
                raise TypeError("Type mismatch in points")
            assert pdim == p.dimension, "Dimension mismatch in points"
            if remain(coordnames,p.coordnames) != []:
                raise ValueError("Coordinate name mismatch in points")
            if p.labels != {}:
                labels.update({i: p.labels})
                i += 1
        for c in xcoordnames: dv[c].append(p(c))
        if paramd and indepvararray is None:
            iv.append(p(indepvarname))
    # submit data as array to maintain coordname ordering present in Points
    dim = len(xcoordnames)
    ca = array([dv[c] for c in xcoordnames], ptype)
    argDict = {'coordarray': ca,
               'coordnames': xcoordnames,
               'coordtype': ptype,
               'labels': labels,
               'norm': norm
                 }
    if paramd:
        if indepvararray is None:
            indepvararray = array(iv, ptype)
        argDict.update({'indepvarname': indepvarname,
                         'indepvararray': indepvararray,
                         'indepvartype': indepvartype})
    return Pointset(argDict)


def arrayToPointset(a, vnames=None, ia=None, iname=""):
    """Convert an array to a non-parameterized Pointset. The inclusion of an
    optional independent variable array creates a parameterized Pointset.

    Coordinate (and independent variable) names are optional: the defaults are
    the array indices (and 't' for the independent variable).
    """
    if np.ndim(a) > 2:
        raise ValueError("Cannot convert arrays of rank > 2")
    if np.ndim(a) == 0:
        raise ValueError("Cannot convert arrays of rank 0")
    if vnames is None:
        vnames = [str(i) for i in range(shape(a)[0])]
    else:
        if len(vnames) != shape(a)[0]:
            raise ValueError("Mismatch between number of coordinate names and"
                             " number of rows in array.\nCoordinates are "
                             "assumed to be the rows of the array")
    if ia is None:
        assert iname=="", ("Independent variable name must be none if no "
                           "independent variable array provided")
        return Pointset({'coordarray': a,
                     'coordnames': vnames})
    else:
        if iname == "":
            iname = "t"
        return Pointset({'coordarray': a,
                     'coordnames': vnames,
                     'indepvararray': ia,
                     'indepvarname': iname})

def exportPointset(thepointset, infodict, separator='   ',
                   precision=12, varvaldir='col',
                   ext='', append=False):
    """Export a pointset to a set of ASCII whitespace- (or
    user-defined character-) separated data files. Option to list each
    variable's data in rows ('across') or in columns ('down').
    Existing files of the same names will be overwritten, unless the
    'append' boolean option is set.

    NB. If the file extension argument 'ext' is present without a
    leading dot, one will be added.

    infodict should consist of: keys = filenames, values = tuples of
    pointset variable names to export.
    """
    assert varvaldir in ['col', 'row'], \
           "invalid variable value write direction"
    # in order to avoid import cycles, cannot explicitly check that
    # thepointset is of type Pointset, because Points.py imports this file
    # (utils.py), so check an attribute instead.
    try:
        thepointset.coordnames
    except AttributeError:
        raise TypeError("Must pass Pointset to this function: use "
                        "arrayToPointset first!")
    infodict_usedkeys = []
    for key, info in infodict.items():
        if isinstance(info, str):
            infodict_usedkeys += [info]
        elif info == []:
            infodict[key] = copy(thepointset.coordnames)
            infodict_usedkeys.extend(thepointset.coordnames)
        else:
            infodict_usedkeys += list(info)
    allnames = copy(thepointset.coordnames)
    if thepointset._parameterized:
        allnames.append(thepointset.indepvarname)
    remlist = remain(infodict_usedkeys, allnames+list(range(len(allnames))))
    if remlist != []:
        print("Coords not found in pointset:%r" % remlist)
        raise ValueError("invalid keys in infodict - some not present "
                         "in thepointset")
    assert isinstance(ext, str), \
           "'ext' extension argument must be a string"
    if ext != '':
        if ext[0] != '.':
            ext = '.'+ext
    if append:
        assert varvaldir == 'col', ("append mode not supported for row"
                                     "format of data ordering")
        modestr = 'ab'
    else:
        modestr = 'wb'
    totlen = len(thepointset)
    if totlen == 0:
        raise ValueError("Pointset is empty")
    for fname, tup in infodict.items():
        try:
            f = open(fname+ext, modestr)
        except IOError:
            print("There was a problem opening file "+fname+ext)
            raise
        try:
            if isinstance(tup, str):
                try:
                    varray = thepointset[tup]
                except TypeError:
                    raise ValueError("Invalid specification of coordinates")
            elif isinstance(tup, int):
                try:
                    varray = thepointset[:,tup].toarray()
                except TypeError:
                    raise ValueError("Invalid specification of coordinates")
            elif isinstance(tup, (list, tuple)):
                if alltrue([isinstance(ti, str) for ti in tup]):
                    thetup = list(tup)
                    if thepointset.indepvarname in tup:
                        tix = thetup.index(thepointset.indepvarname)
                        thetup.remove(thepointset.indepvarname)
                    try:
                        vlist = thepointset[thetup].toarray().tolist()
                    except TypeError:
                        raise ValueError("Invalid specification of coordinates")
                    if len(thetup)==1:
                        vlist = [vlist]
                    if thepointset.indepvarname in tup:
                        vlist.insert(tix, thepointset.indepvararray.tolist())
                    varray = array(vlist)
                elif alltrue([isinstance(ti,_int_types) for ti in tup]):
                    try:
                        varray = thepointset[:,tup].toarray()
                    except TypeError:
                        raise ValueError("Invalid specification of coordinates")
                else:
                    raise ValueError("Invalid specification of coordinates")
            else:
                f.close()
                raise TypeError("infodict values must be singletons or "
                                "tuples/lists of strings or integers")
        except IOError:
            f.close()
            print("Problem writing to file"+fname+ext)
            raise
        except KeyError:
            f.close()
            raise KeyError("Keys in infodict not found in pointset")
        if isinstance(precision, int):
            assert precision > 0
            ps = str(precision)
        else:
            raise TypeError("precision must be a positive integer")
        if varvaldir == 'row':
            savetxt(f, varray, '%.'+ps+'f', separator)
        else:
            savetxt(f, transpose(varray), '%.'+ps+'f', separator)
        f.close()


def importPointset(xFileName, t=None, indices=None, sep=" ",
                   preamblelines=0):
    """Import ASCII format files containing data points.
    If the first row contains string names then the output
    will be a pointset, otherwise a numeric array.

    A dictionary is returned, with keys 'vararray' will point to the
    data. The 't' argument can specify one of several things:

    string: filename to read single-column of time values (same length as
            xFileName)
    sequence type: time values (same length as xFileName)
    integer: column in xFileName to treat as time data

    If used, this leads to and an additional key in the return
    dictionary where 't' points to the independent variable array.

    Specific columns can be selected for the variable data array by
    specifying a list of column indices in argument 'indices'.

    The separator used in the ASCII file can be specified by argument
    'sep' (defaults to single whitespace character).

    preamblelines (positive integer) specifies how many lines to skip before
    starting to read data (in case of preceding text) -- default 0.
    """

    if indices is None:
        indices = []
    xFile = open(xFileName, 'r')
    xFileStrList = xFile.readlines()
    filelen = len(xFileStrList)-preamblelines
    if filelen == 1 and '\r' in xFileStrList[0]:
        # fix problem when no newlines picked up, only '\r'
        xFileStrList = xFileStrList[0].split('\r')
        filelen = len(xFileStrList)
    if filelen <= 1:
        raise ValueError("Only 1 data point found in variables datafile")
    x_dummy_all = xFileStrList[preamblelines].rstrip("\n")
    x_dummy_vallist = [s for s in x_dummy_all.split(sep) if s != '']
    if t is None:
        get_t = 0
    elif isinstance(t, str):
        tFileName = t
        tFile = open(tFileName, 'r')
        tFileStrList = tFile.readlines()
        if len(tFileStrList)-preamblelines != filelen:
            raise ValueError("Length of data and time files must be equal"
                           " -- are there any blank lines in the files?")
        get_t = 1
    elif isinstance(t, _seq_types):
        if len(t) != filelen:
            raise ValueError("Length of data file and t array must be "
                       "equal -- are there any blank lines in the files?")
        tVals = t
        get_t = 0
    elif isinstance(t, _int_types):
        # t represents column index to find time data in data file
        if t >= len(x_dummy_vallist) or t < 0:
            raise ValueError("t index out of range")
        get_t = 2
    if indices == []:
        if get_t == 2:
            dim = len(x_dummy_vallist)-1
            indices = remain(range(0,dim+1),[t])
        else:
            dim = len(x_dummy_vallist)
            indices = list(range(0,dim))
    else:
        dim = len(indices)
        if get_t == 2:
            if t in indices:
                raise ValueError("You specified column "+str(t)+" as time "
                    "data, but you have specified it as a data column in "
                    "indices argument")
    # try to find variable names. if successful, start at row 1
    start = preamblelines
    # replace unnecessary quote marks in strings
    test_line = [n.strip('"').strip("'") for n in \
            xFileStrList[preamblelines].lstrip(sep).lstrip(" ").rstrip("\n").rstrip("\r").split(sep)]
    def is_float(vstr):
        try:
            val = float(vstr)
        except ValueError:
            return False
        else:
            return True
    if alltrue([not is_float(n) for n in test_line]):
        # success
        start += 1
        # replace any internal spaces with underscores, remove dots
        test_line = [n.replace(" ", "_").replace(".","") for n in test_line]
        if get_t == 2:
            t_name = test_line[t]
            varnames = test_line[0:t]+test_line[t+1:]
        else:
            if get_t == 1:
                # try first line of t file
                t_test = tFileStrList[0].lstrip(" ").rstrip("\n").rstrip("\r").replace(".","").replace(" ","_").strip('"').strip("'")
                if is_float(t_test):
                    # already checked that file lengths were the same
                    raise ValueError("First line of t file shouldn't be a number")
                else:
                    t_name = t_test
            else:
                t_name = 't'
            varnames = test_line
    else:
        t_name = 't'
        varnames = None
    tVals = zeros(filelen-start, float)
    xVals = zeros([filelen-start, dim], float)
    # read rest of file
    for i in range(filelen-start):
        vLine = xFileStrList[i+start].rstrip("\n")
        if vLine == '':
            continue
        vLineVals = [s for s in vLine.split(sep) if s != '']
        if get_t == 1:
            # Additional left strip of space char in case sep is different
            tLine = tFileStrList[i+start].rstrip("\n").lstrip(sep).lstrip(" ")
            if len(tLine.split(sep)) != 1:
                raise ValueError("Only one t value expected per line of"
                                   " datafile")
            if tLine == '':
                continue
            tVals[i] = float(tLine)
        elif get_t == 2:
            tVals[i] = float(vLineVals[t])
        try:
            xLineVals = [vLineVals[ix] for ix in indices]
        except IndexError:
            print("Valid indices were: 0 - %d" % len(vLineVals)-1)
            raise
        if len(xLineVals) != dim:
            raise ValueError("Exactly "+str(dim)+" values expected per "
                               "line of datafile")
        xVals[i] = array([float(xstr) for xstr in xLineVals], float)
    xFile.close()
    if get_t == 1:
        tFile.close()
    if get_t == 0:
        if varnames is None:
            return xVals
        else:
            # non-parameterized pointset
            return Pointset(dict(zip(varnames, xVals)))
    else:
        if varnames is None:
            return {t_name: tVals, 'vararray': xVals.T}
        else:
            return Pointset(indepvardict={t_name: tVals},
                            coorddict=dict(zip(varnames,xVals.T)))


def export_pointset_to_CSV(filename, pts):
    """Simple export that ignores all metadata in pts,
    including name, tags, norm, etc.
    Data is arranged by row only.

    Independent variable is first column, if it exists in pts.
    """
    import csv
    outfile = open(filename, 'w')
    writer = csv.writer(outfile)

    # header row
    if pts._parameterized:
        rows_header = [pts.indepvarname] + pts.coordnames
    else:
        rows_header = pts.coordnames
    writer.writerow(rows_header)

    # data rows
    if pts._parameterized:
        for i in range(len(pts)):
            writer.writerow([pts.indepvararray[i]] + list(pts.coordarray[:,i]))
    else:
        for i in range(len(pts)):
            writer.writerow(pts.coordarray[:,i])

    outfile.close()


def mergePointsets(pts1, pts2):
    """Merges two pointsets into a new pointset, preserving (merging) any
    metadata in each.

    In particular, if each have different accuracy tolerances, the
    larger of the two will be used. If each have different 'checklevel' values,
    the larger of the two will be used. Point labels will be merged. Names
    will also be merged.

    If both are parameterized, their independent variable arrays
    must be identical. If only one is parameterized, the result will be
    too. The two pointsets must be identical in length.

    The norm associated with each pointset must be the same.
    """
    len1 = len(pts1)
    len2 = len(pts2)
    assert len1 == len2, "Pointsets must have equal length"
    assert pts1._normord == pts2._normord, "Pointsets must use the same norm"
    isparam1 = isparameterized(pts1)
    isparam2 = isparameterized(pts2)
    if isparam1 and isparam2:
        assert pts1.indepvarname == pts2.indepvarname, \
               "Parameterized pointsets must have identical independent variable names"
        assert all(pts1.indepvararray == pts2.indepvararray), \
               "Parameterized pointsets must have identical independent variable values"
    common_coords = intersect(pts1.coordnames, pts2.coordnames)
    for c in common_coords:
        assert all(pts1[c] == pts2[c]), \
           "Pointsets must not share any coordinate names whose values are not identical"
    args = {}
    if isparam1 or isparam2:
        if isparam1:
            tvals = pts1.indepvararray
            tname = pts1.indepvarname
        else:
            tvals = pts2.indepvararray
            tname = pts2.indepvarname
        args['indepvardict'] = {tname: tvals}
    args['checklevel'] = max(pts1.checklevel, pts2.checklevel)
    args['tolerance'] = max(pts1._abseps, pts2._abseps)
    if name is None:
        if pts1.name == "":
            name1 = "<unnamed>"
        else:
            name1 = pts1.name
        if pts2.name == "":
            name2 = "<unnamed>"
        else:
            name2 = pts2.name
        args['name'] = "Merged %s:%s" % (name1, name2)
    coorddict = pts1.todict()
    coorddict.update(pts2.todict())
    args['coorddict'] = coorddict
    lab1 = deepcopy(pts1.labels)
    lab2 = deepcopy(pts2.labels)
    lab2.update(lab1)
    args['labels'] = lab2
    return Pointset(**args)


def padPointset(pts, pinterval, value_dict, eps=None):
    """Pad a pointset pts with values from value_dict over the interval given
    by pinterval (pair). For each side of the interval outside of the current independent
    variable domain of pts, two new points are added, one at the outer limit
    of the interval, and one a distance eps (default the abseps setting of pts)
    from the existing closest point in pts.
    """
    tlo, thi = pinterval
    ts = pts.indepvararray
    all_dict = value_dict.copy()
    assert remain(value_dict.keys(), pts.coordnames) == []
    if eps is None:
        eps = pts._abseps
    if tlo < ts[0]:
        all_dict['t'] = tlo
        pts.insert(Point(coorddict=all_dict,
                         labels='pad'))
        all_dict['t'] = ts[0]-eps
        pts.insert(Point(coorddict=all_dict,
                         labels='pad'))
    if thi > ts[-1]:
        all_dict['t'] = ts[-1]+eps
        pts.insert(Point(coorddict=all_dict,
                         labels='pad'))
        all_dict['t'] = thi
        pts.insert(Point(coorddict=all_dict,
                         labels='pad'))
    return pts
