# Interpolated lookup table with extrapolated end points

from .allimports import *
from .baseclasses import ctsGen, theGenSpecHelper
from PyDSTool.utils import *
from PyDSTool.common import *

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, float64, array
from scipy import polyfit
import math, random
from copy import copy, deepcopy

# -----------------------------------------------------------------------------

class ExtrapolateTable(ctsGen):
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

        if 'npts' in kw:
            self.npts = kw['npts']
            if self.npts is None:
                self.npts = -1
            self.foundKeys +=1
        else:
            self.npts = -1

        tlen = len(self.tdata)
        if self.npts > 1:
            lidx = min(tlen,self.npts)
            if lidx <= 1 or lidx == tlen:
                tl = self.tdata
                th = self.tdata
            else:
                tl = self.tdata[0:lidx]
                th = self.tdata[-lidx-1:-1]
        else:
            tl = self.tdata
            th = self.tdata
            lidx = -1
        self._lapx = {}
        self._hapx = {}

        for x in self._xdatadict:
            if lidx > 1:
                self._lapx[x] = polyfit(tl, self._xdatadict[x][0:lidx], 1)
                self._hapx[x] = polyfit(th, self._xdatadict[x][-lidx-1:-1], 1)
            else:
                self._lapx[x] = polyfit(tl, self._xdatadict[x], 1)
                self._hapx[x] = self._lapx[x]

        if 'lotime' in kw:
            assert kw['lotime'] < self.tdata[0]
            loT = kw['lotime']
            for x in self._xdatadict:
                if tlen == 1:
                    newloval = self._xdatadict[x][0]
                else:
                    newloval = self._lapx[x][0]*loT + self._xdatadict[x][0]
                    #newloval = (self._xdatadict[x][1] - self._xdatadict[x][0])/(self.tdata[1] - self.tdata[0])*loT
                    #+ self._xdatadict[x][0]
                if isinstance(self._xdatadict[x], list):
                    self._xdatadict[x] = [newloval] + self._xdatadict[x]
                else:
                    temp = self._xdatadict[x].tolist()
                    self._xdatadict[x] = array([newloval] + temp)
            self.foundKeys += 1

        if 'hitime' in kw:
            assert kw['hitime'] > self.tdata[-1]
            hiT = kw['hitime']
            for x in self._xdatadict:
                if tlen == 1:
                    newhival = self._xdatadict[x][-1]
                else:
                    newhival = self._hapx[x][0]*hiT + self._xdatadict[x][-1]
                    #newhival = (self._xdatadict[x][-2] - self._xdatadict[x][-1])/(self.tdata[-2] - self.tdata[-1])*hiT
                    #+ self._xdatadict[x][-1]
                if isinstance(self._xdatadict[x], list):
                    self._xdatadict[x] = self._xdatadict[x] + [newhival]
                else:
                    temp = self._xdatadict[x].tolist()
                    self._xdatadict[x] = array(temp + [newhival])
            self.foundKeys += 1

        if 'lotime' in kw:
            if isinstance(self.tdata, list):
                self.tdata = [kw['lotime']] + self.tdata
            else:
                temp = self.tdata.tolist()
                self.tdata = [kw['lotime']] + temp

        if 'hitime' in kw:
            if isinstance(self.tdata, list):
                self.tdata = self.tdata + [kw['hitime']]
            else:
                temp = self.tdata.tolist()
                self.tdata = temp + [kw['hitime']]

        self.tdata = array(self.tdata)

        self.tdomain = extent(self.tdata)
        self.xdomain = {}
        for x in self._xdatadict:
            self.xdomain[x] = extent(self._xdatadict[x])
        ctsGen.__init__(self, kw)
        self.funcspec = {}
        if 'vars' in kw:
            raise PyDSTool_KeyError('vars option invalid for extrapolated table class')
        if 'auxvars' in kw:
            raise PyDSTool_KeyError('auxvars option invalid for extrapolated table class')
        if 'tdomain' in kw:
            raise PyDSTool_KeyError('tdomain option invalid for extrapolated table class')
        if 'xdomain' in kw:
            raise PyDSTool_KeyError('xdomain option invalid for extrapolated table class')
        if 'pdomain' in kw:
            raise PyDSTool_KeyError('pdomain option invalid for extrapolated table class')
        if 'ttype' in kw:
            raise PyDSTool_KeyError('ttype option invalid for extrapolated table class')
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
        for x in self._xdatadict:
            self.funcspec[x] = Pointset({'coordarray': self._xdatadict[x],
                                         'coordtype': float64,
                                         'indepvararray': self.tdata,
                                         'indepvartype': float64,
                                         'indepvarname': 't',
                                         'coordnames': x})
        self._needKeys.extend(['tdata', 'ics'])
        self._optionalKeys.extend(['method','lotime','hitime','npts'])
        self.indepvartype = float
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
theGenSpecHelper.add(ExtrapolateTable, symbolMapDict, 'python')
