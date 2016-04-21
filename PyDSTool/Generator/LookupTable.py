# Lookup table
from __future__ import division, absolute_import, print_function

from .allimports import *
from .baseclasses import discGen, theGenSpecHelper
from PyDSTool.utils import *
from PyDSTool.common import *

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue
import math, random
from copy import copy, deepcopy

# -----------------------------------------------------------------------------

class LookupTable(discGen):
    """Lookup table trajectory with no interpolation.

    Independent and dependent variables may be integers or floats."""

    def __init__(self, kw):
        try:
            self.tdata = kw['tdata']
            self._xdatadict = {}
            for k, v in dict(kw['ics']).items():
                self._xdatadict[str(k)] = v
            self.foundKeys = 2
            # check for other, invalid keys (but currently just ignored)
        except KeyError:
            raise PyDSTool_KeyError('Invalid keyword passed')
        self.tdomain = self.tdata
        discGen.__init__(self, kw)
        self._needKeys.extend(['tdata', 'ics'])
        # return values unused (side effects only)
        # hack to allow xtype to run
        kw['varspecs'] = {}.fromkeys(self._xdatadict, '')
        self._kw_process_dispatch(['varspecs', 'ttype', 'xtype'], kw)
        del kw['varspecs']
        self.foundKeys -= 1
        self.funcspec = {}
        if 'vars' in kw:
            raise PyDSTool_KeyError("vars option invalid for lookup table "
                                    "class")
        if 'auxvars' in kw:
            raise PyDSTool_KeyError("auxvars option invalid for lookup table "
                                    "class")
        for x in self._xdatadict:
            self.funcspec[x] = Pointset({'coordarray': self._xdatadict[x],
                                         'coordtype': self.xtype[x],
                                         'indepvararray': self.tdata,
                                         'indepvartype': self.indepvartype,
                                         'indepvarname': 't',
                                         'coordnames': x})

        self.checkArgs(kw)
        self.indepvariable = Variable(listid, {'t_domain': self.tdomain},
                             {'t': self.tdata}, 't')
        self._register(self.indepvariable)
        for x in self._xdatadict:
            self.variables[x] = Variable(self.funcspec[x],
                                         {'t': copy(self.indepvariable.depdomain)},
                                         {x: self.funcspec[x].toarray()}, x)
        self._register(self.variables)
        self.dimension = len(self._xdatadict)
        self.validateSpec()
        self.defined = True


    def compute(self, trajname):
        if self.defined:
            #self.validateSpec()
            self.diagnostics.clearWarnings()
            self.diagnostics.clearErrors()
        return Trajectory(trajname, [copy(v) for v in self.variables.values()],
                          abseps=self._abseps, globalt0=self.globalt0,
                          checklevel=self.checklevel,
                          FScompatibleNames=self._FScompatibleNames,
                          FScompatibleNamesInv=self._FScompatibleNamesInv,
                          modelNames=self.name,
                          modelEventStructs=self.eventstruct)

    def validateSpec(self):
        discGen.validateSpec(self)
        try:
            assert isoutputdiscrete(self.indepvariable)
            for v in self.variables.values():
                assert isinstance(v, Variable)
            assert not self.inputs
        except AssertionError:
            print('Invalid system specification')
            raise


    def __del__(self):
        discGen.__del__(self)




# Register this Generator with the database

symbolMapDict = {}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(LookupTable, symbolMapDict, 'python')
