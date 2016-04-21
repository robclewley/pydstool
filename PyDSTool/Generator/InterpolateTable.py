# Interpolated lookup table
from __future__ import division, absolute_import, print_function

from .allimports import *
from .baseclasses import ctsGen, theGenSpecHelper
from PyDSTool.utils import *
from PyDSTool.common import *

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, float64
import math, random
from copy import copy, deepcopy

# -----------------------------------------------------------------------------

class InterpolateTable(ctsGen):
    """Data lookup table with piecewise linear or piecewise constant interpolation."""

    def __init__(self, kw):
        try:
            self.tdata = kw['tdata']
            self._xdatadict = {}
            for k, v in dict(kw['ics']).items():
                self._xdatadict[str(k)] = v
            self.foundKeys = 2
            # check for other, invalid keys (but currently just ignored)
        except KeyError:
            raise PyDSTool_KeyError('Keywords missing in argument')
        self.tdomain = extent(self.tdata)
        self.xdomain = {}
        for x in self._xdatadict:
            self.xdomain[x] = extent(self._xdatadict[x])
        ctsGen.__init__(self, kw)
        self._needKeys.extend(['tdata', 'ics'])
        self._optionalKeys.append('method')
        self.funcspec = {}  # dict, not a FuncSpec instance
        if 'vars' in kw:
            raise PyDSTool_KeyError('vars option invalid for interpolated table class')
        if 'auxvars' in kw:
            raise PyDSTool_KeyError('auxvars option invalid for interpolated table class')
        if 'tdomain' in kw:
            raise PyDSTool_KeyError('tdomain option invalid for interpolated table class')
        if 'xdomain' in kw:
            raise PyDSTool_KeyError('xdomain option invalid for interpolated table class')
        if 'pdomain' in kw:
            raise PyDSTool_KeyError('pdomain option invalid for interpolated table class')
        if 'ttype' in kw:
            raise PyDSTool_KeyError('ttype option invalid for interpolated table class')
        # hack to allow xtype to run
        kw['varspecs'] = {}.fromkeys(self._xdatadict, '')
        self._kw_process_dispatch(['varspecs', 'xtype'], kw)
        del kw['varspecs']
        self.foundKeys -= 1
        if 'method' in kw:
            if kw['method']=='linear':
                interp=interp1d
            elif kw['method']=='constant':
                interp=interp0d
            else:
                raise ValueError("Invalid interpolation method")
            self.foundKeys += 1
        else:
            # default to piecewise linear interpolation
            interp=interp1d
        self.indepvartype = float
        for x in self._xdatadict:
            self.funcspec[x] = Pointset({'coordarray': self._xdatadict[x],
                                         'coordtype': self.xtype[x],
                                         'indepvararray': self.tdata,
                                         'indepvartype': self.indepvartype,
                                         'indepvarname': 't',
                                         'coordnames': x})
        self.checkArgs(kw)
        self.indepvariable = Variable(listid, Interval('t_domain',
                                                       self.indepvartype,
                                              self.tdomain, self._abseps),
                             Interval('t', self.indepvartype,
                                      extent(self.tdata),
                                      self._abseps), 't')
        self._register(self.indepvariable)
        for x in self._xdatadict:
            self.variables[x] = Variable(interp(copy(self.tdata),
                                          self.funcspec[x].toarray()), 't',
                                  Interval(x, self.xtype[x], self.xdomain[x],
                                           self._abseps), x)
        self._register(self.variables)
        self.dimension = len(self._xdatadict)
        self.validateSpec()
        self.defined = True


    def compute(self, trajname):
        return Trajectory(trajname, [copy(v) for v in self.variables.values()],
                          abseps=self._abseps, globalt0=self.globalt0,
                          checklevel=self.checklevel,
                          FScompatibleNames=self._FScompatibleNames,
                          FScompatibleNamesInv=self._FScompatibleNamesInv,
                          modelNames=self.name,
                          modelEventStructs=self.eventstruct)


    def set(self, **kw):
        if 'abseps' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, abseps=kw['abseps'])
            for x in self._xdatadict:
                self.variables[x] = Variable(interp(copy(self.tdata),
                                          self.funcspec[x].toarray()), 't',
                                  Interval(x, self.xtype[x], self.xdomain[x],
                                           self._abseps), x)
        if 'checklevel' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, checklevel=kw['checklevel'])
        if 'globalt0' in kw:
            # pass up to generic treatment for this
            ctsGen.set(self, globalt0=kw['globalt0'])

    def validateSpec(self):
        ctsGen.validateSpec(self)
        try:
            assert isoutputcts(self.indepvariable)
            for v in self.variables.values():
                assert isinstance(v, Variable)
            assert not self.inputs
        except AssertionError:
            print('Invalid system specification')
            raise


    def __del__(self):
        ctsGen.__del__(self)




# Register this Generator with the database

symbolMapDict = {}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(InterpolateTable, symbolMapDict, 'python')
