""" Curve classes: Continuation, EquilibriumCurve, FoldCurve, HopfCurveOne, HopfCurveTwo

    Drew LaMar, 2006;
    Last edited: December 2012

    Continuation is the ancestral class of all curve classes and contains the continuation algorithms
    (Moore-Penrose, etc.)  It also contains all methods that are general to any curve found using
    continuation.

    TO DO/Notes:
        * Why are there two BranchPointFold in BifPoint.py?
        * Highlight not working in plot_cycles
        * Branch point curve
        * Phase plane stuff! (see PyCont_PredPreyI-III examples in Phage project)
        * Symbolic jacobians not working!!! (see PyCont_PredPreyI.py)
        * Modify bifurcation point locators to handle nonzero parts; check MATCONT again
        * LPC detection problem in PyCont_Logistic.py
        * Branch points on fold curve problem (PyCont_PredPrey.py)
        * Need to check update in BorderMethod.  Not currently working.  Sticking with random for now.
        * Implement pseudo-Newton method for branch point locator (BranchPoint.locate)
        * Need to revisit alternate branch locator in BranchPoint.process
        * Allow user to toggle update in BorderMethods (see BranchFold.py for example) -- how to make matrix as well-conditioned as possible?
            (In AddTestFunction:  Changed to "self.testfunc = TF_type(self.sysfunc, C, update=False)
        * Removed BP in Continuation class
        * Implement branch points for FixedPointCuspCurve [Networks/Global/dat/PyCont_Oscillator.py]
    	* FixedPointCuspCurve - Merge AddTestFunction_FixedPoint_Mult and AddTestFunction_FixedPoint
    	* FixedPointCuspCurve - Merge CP_Fold and CP_Fold2
    	* Create NSCurve (PyCont_DiscPredPrey2.py) [using NS_Bor]
    	* Allow for xdomain (see PyCont_DiscPredPrey2.py)
    	* Labels plot off screen when xlim or ylim
    	* Add cleanLabels to all children classes?  (e.g., in FixedPointNSCurve)
    	* Rename FoldCurve to LimitPointCurve
    	* Allow for PCargs to include different parameters (e.g. initpars)
"""

# -----------------------------------------------------------------------------------------

from __future__ import absolute_import, print_function

from .misc import *
from .TestFunc import *
from .BifPoint import *
from .Plotting import *

from PyDSTool import Point, Pointset, PointInfo, args
from PyDSTool.common import pickle, sortedDictValues, sortedDictKeys
from PyDSTool.errors import *
from PyDSTool.Symbolic import QuantSpec
try:
    from PyDSTool.matplotlib_import import *
except ImportError:
    from PyDSTool.matplotlib_unavailable import *
    print("Warning: matplotlib failed to import properly and so is not")
    print("  providing a graphing interface")

from numpy.random import random
from numpy import dot as matrixmultiply
from scipy import optimize, linalg
from numpy import array, float, complex, int, float64, complex64, int32, \
    zeros, divide, subtract, arange, all, any, argsort, reshape, nonzero, \
    log10, Inf, NaN, isfinite, r_, c_, sign, mod, mat, log2, \
    subtract, divide, transpose, eye, real, imag, isnan, resize
from numpy.linalg import cond # not present in scipy.linalg!

from copy import copy, deepcopy
from math import ceil

#####
_classes = ['Continuation', 'EquilibriumCurve', 'FoldCurve', 'HopfCurveOne',
            'HopfCurveTwo', 'FixedPointCurve', 'LimitCycleCurve',
            'UserDefinedCurve', 'FixedPointFoldCurve', 'FixedPointFlipCurve',
            'FixedPointNSCurve', 'FixedPointCuspCurve']

_constants = ['cont_args_list', 'cont_bif_points', 'equilibrium_args_list',
              'equilibrium_bif_points', 'fold_args_list', 'fold_bif_points',
              'hopf_args_list', 'hopf_bif_points', 'limitcycle_args_list',
              'limitcycle_bif_points', 'fixedpoint_args_list',
              'fixedpoint_bif_points', 'flip_args_list', 'flip_bif_points',
              'NS_args_list', 'NS_bif_points', 'userdefined_args_list',
              'all_args_list', 'all_point_types', 'all_curve_types',
              'bif_curve_colors', 'bif_point_colors',
              'stab_line_styles','auto_point_types', 'other_special_points',
              'solution_measures', 'solution_measures_list']

__all__ = _classes + _constants
#####

cont_args_list = ['name','force','freepars','MaxNumPoints','MaxCorrIters',
                  'MaxTestIters','MaxStepSize', 'MinStepSize', 'StepSize',
                  'VarTol','FuncTol','TestTol', 'description', 'uservars',
                  'LocBifPoints','verbosity','ClosedCurve','SaveJacobian',
                  'SaveEigen', 'Corrector', 'UseAuto', 'StopAtPoints',
                  'SPOut']

cont_bif_points = ['B', 'SP']

equilibrium_args_list = ['LocBifPoints']
equilibrium_bif_points = ['BP', 'LP', 'H']

fold_args_list = ['LocBifPoints']
fold_bif_points = ['BT', 'ZH', 'CP']
#fold_bif_points = ['BT', 'ZH', 'CP', 'BP']  # Disabling BP for now.

hopf_args_list = ['LocBifPoints']
hopf_bif_points = ['BT', 'ZH', 'GH', 'DH']

fixedpoint_args_list = ['LocBifPoints', 'period']
fixedpoint_bif_points = ['BP', 'PD', 'LPC', 'NS']

fold_map_args_list = ['LocBifPoints', 'period']
fold_map_bif_points = ['CP']

flip_args_list = ['LocBifPoints', 'period']
flip_bif_points = []

NS_args_list = ['LocBifPoints', 'period']
NS_bif_points = []

cusp_args_list = ['LocBifPoints', 'period']
cusp_bif_points = ['']

limitcycle_args_list = ['LocBifPoints', 'NumCollocation', 'NumIntervals',
                        'AdaptMesh', 'NumSPOut', 'DiagVerbosity',
                        'SolutionMeasures', 'SaveFlow']
limitcycle_bif_points = ['PD', 'LPC', 'NS']

userdefined_args_list = ['LocBifPoints']

other_special_points = ['RG', 'UZ', 'P', 'MX', 'B']

auto_point_types = {1: 'BP', 2: 'LP', 3: 'H', 4: 'RG', -4: 'UZ', 5: 'LPC',
                    6: 'BP', 7: 'PD', 8: 'NS', 9: 'P', -9: 'MX'}

solution_measures_list = ['max', 'min', 'avg', 'nm2']   # Ordering is important
solution_measures = dict(zip(solution_measures_list,[0, 0, 1, 2]))

all_args_list = unique(cont_args_list + equilibrium_args_list + fold_args_list +
                       hopf_args_list + fixedpoint_args_list + flip_args_list +
                       NS_args_list + limitcycle_args_list)
all_point_types = unique(other_special_points + cont_bif_points +
                         equilibrium_bif_points + fold_bif_points +
                         hopf_bif_points + fixedpoint_bif_points +
                         flip_bif_points + NS_bif_points +
                         limitcycle_bif_points)
all_curve_types = ['EP-C', 'LP-C', 'H-C1', 'H-C2', 'FP-C', 'LC-C', 'FD-C',
                   'FL-C', 'NS-C', 'CP-C']

bif_curve_colors = {'EP-C': 'k', 'LP-C': 'r', 'H-C1': 'b', 'H-C2': 'b',
                    'FP-C': 'k', 'LC-C': 'm', 'UD-C': 'k', 'FD-C': 'r',
                    'FL-C': 'g', 'NS-C': 'b', 'CP-C': 'c'}
bif_point_colors = {'P': 'ok', 'RG': 'ok', 'LP': 'or', 'BP': 'og',
                    'H': 'ob', 'B': 'dr', 'BT': 'sy', 'ZH': 'sk',
                    'CP': 'sr', 'GH': 'sb', 'DH': 'sg', 'LPC': 'Dr',
                    'PD': 'Dg', 'NS': 'Db', 'MX': 'xr', 'UZ': '^r',
                    'SP': '*b'}
stab_line_styles = {'S': '-', 'U': '--', 'N': '-.', 'X': ':'}


class Continuation(object):
    """Abstract continuation class

    Children: EquilibriumCurve, FoldCurve, HopfCurveOne, HopfCurveTwo,
    LimitCycleCurve
    """

    def __init__(self, model, gen, automod, plot, args=None):
        self.curvetype = args['type']
        self._ptlabel = self.curvetype.split('-')[0]

        self.model = model
        self.gensys = gen
        self._autoMod = automod
        self.UseAuto = False

        if 'description' not in args:
            self.description = 'None'
        else:
            self.description = args['description']

        if not hasattr(self, 'parsdict'):
            self.parsdict = self.model.query('pars')
        self.freepars = args['freepars']
        self.auxpars = args['auxpars']

        if hasattr(self, 'varslist'):
            # varsindices refers to the indices in the full set of variables
            # that are used in this subset
            self.varslist.sort()
            if self.curvetype == 'UD-C':
                # unused, self._systemuser -> user-supplied func will be used directly
                self.varsindices = array([])
            else:
                orig_vars = self.model.query('vars')
                # will call self._system, selecting vars from possible ones
                self.varsindices = array([orig_vars.index(v) for v in self.varslist])
        else:
            if 'uservars' in args and self.curvetype != 'UD-C':
                self.varslist = args['uservars']
                orig_vars = self.model.query('vars')
                # will call self._system, selecting vars from possible ones
                self.varsindices = array([orig_vars.index(v) for v in self.varslist])
            else:
                self.varslist = self.model.query('vars')
                self.varsindices = arange(len(self.varslist))

        if self.gensys.haveJacobian_pars():
            fargs, fspecstr = self.gensys.funcspec._auxfnspecs['Jacobian_pars']
            Jquant = QuantSpec('J', fspecstr)
            if Jquant.dim == 0:
                # dim of vars == 1 == dim of pars
                assert len(self.varslist) == 1
                assert len(self.freepars) == 1
                # Supplied Jac w.r.t. params is custom-made for only the free params in this continuation
                # (or there's only one parameter in system)
                self.parsindices = array([0])
            else:
                assert len(self.varslist) == Jquant.dim
                Jquant0 = Jquant.fromvector(0)
                if Jquant0.dim == 0:
                    # dim of free pars == 1
                    assert len(self.freepars) == 1
                    # Supplied Jac w.r.t. params is custom-made for only the free params in this continuation
                    # (or there's only one parameter in system)
                    self.parsindices = array([0])
                else:
                    if len(self.freepars) == Jquant0.dim:
                        # Supplied Jac w.r.t. params is custom-made for only the free params in this continuation
                        self.parsindices = arange(range(Jquant0.dim))
                    else:
                        # Assume supplied Jac w.r.t. params is for all params in the original system
                        # therefore there should be fewer free params than # system parameters
                        assert len(self.freepars) < Jquant0.dim
                        self.parsindices = array([list(self.parsdict.keys()).index(p) for p in self.freepars])
        else:
            self.parsindices = array([list(self.parsdict.keys()).index(p) for p in self.freepars])
        self.varsdim = len(self.varslist)
        self.freeparsdim = len(self.freepars)
        self.auxparsdim = len(self.auxpars)
        self.dim = self.varsdim + self.freeparsdim + self.auxparsdim

        if (self.curvetype != 'UD-C'):
            self.sysfunc = Function((self.dim, self.varsdim), self._system)
        else:
            self.sysfunc = Function((self.dim, self.varsdim), self._systemuser)

        if (self.curvetype != 'UD-C' and self.gensys.haveJacobian()):
            if self.gensys.haveJacobian_pars():
                self.sysfunc.jac = Function((self.sysfunc.n,
                                         (self.sysfunc.m,self.sysfunc.n)),
                                        self._systemjac_withpars)
            else:
                self.sysfunc.jac = Function((self.sysfunc.n,
                                         (self.sysfunc.m,self.sysfunc.n)),
                                        self._systemjac)
        elif (self.curvetype == 'UD-C' and hasattr(self, '_userjac')):
            self.sysfunc.jac = Function((self.sysfunc.n,
                                         (self.sysfunc.m,self.sysfunc.n)),
                                        self._systemjacuser)
        else:
            self.sysfunc.jac = Function((self.sysfunc.n,
                                         (self.sysfunc.m,self.sysfunc.n)),
                                        self.sysfunc.diff)

        self.coords = self.sysfunc.coords = arange(self.varsdim).tolist()
        self.params = self.sysfunc.params = (arange(self.freeparsdim \
                                                + self.auxparsdim) \
                                                + self.varsdim).tolist()
        self.allvars = self.sysfunc.allvars = self.coords + self.params

        # Initialize vars and pars based on initpoint
        self.initpoint = self.model.query('ics')
        for k, v in args['initpoint'].items():
            if k in self.varslist or k in args['auxpars']:
                self.initpoint[k] = v
            elif k in self.model.query('pars'):
                self.parsdict[k] = v

        for p in args['freepars']:
            self.initpoint[p] = self.parsdict[p]

        self.initpoint = tocoords(self, self.initpoint.copy())

        if 'initdirec' not in args:
            self.initdirec = None
        else:
            self.initdirec = tocoords(self, args['initdirec'])

        if 'initcycle' not in args:
            self.initcycle = None
        else:
            self.initcycle = args['initcycle']

        if not hasattr(self, "SPOut"):
            self.SPOut = None
        if not hasattr(self, "NumSPOut"):
            self.NumSPOut = 300

        self.preTF = None
        self.reset()

        # Removes extra parameters (first time parameter initpoint, system
        #   parameter auxpars, and uneditable parameter type) before sending
        #   to update() method
        args = dict(args)
        [args.pop(i) for i in ['initpoint','initdirec','initcycle','auxpars',
                               'type'] if i in args]
        self.update(args)
        self.fig = None
        self.text_handles = []
        self.plot = plot

        self._statuscodes = {0: 'Unrecognized error encountered (check stderr output).  Stopping continuation...',
                            -1: 'Do over.'}


    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def reset(self, args=None):
        """Resets curve by setting default parameters and deleting solution curve."""
        self.MaxNumPoints = 300
        self.MaxCorrIters = 5
        self.MaxTestIters = 10
        self.MaxStepSize = 0.1
        self.MinStepSize = 1e-5
        self.StepSize = 0.01
        self.VarTol = self.FuncTol = 1e-6
        self.TestTol = 1e-4
        self.ClosedCurve = 50
        self.verbosity = 1
        self.SPOut = None
        self.NumSPOut = 300
        self.sol = None
        # record of newly computed solution segment by
        # forward or backward methods
        self.new_sol_segment = None
        self.LocBifPoints = []
        self.StopAtPoints = []
        self.TestFuncs = None
        self.BifPoints = {}
        self.CurveInfo = PointInfo()
        self.SaveJacobian = False
        self.SaveEigen = False
        self.Corrector = self._MoorePenrose

        if args is not None:
            self.update(args)


    def update(self, args):
        """Update parameters for Continuation."""
        if args is not None:
            for k, v in args.items():
                if k in cont_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = cont_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = cont_bif_points
                                else:
                                    w = [w]

                        self.LocBifPoints = [bftype for bftype in v \
                                             if bftype in cont_bif_points]
                        self.StopAtPoints = [bftype for bftype in w \
                                             if bftype in cont_bif_points]
                    elif k == 'Corrector':
                        self.Corrector = getattr(self, '_' + v)
                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))
                elif k not in all_args_list:
                    print("Warning: " + k + " is either not a valid parameter or immutable.")


    def _preTestFunc(self, X, V):
        J = self.sysfunc.jac(X)
        self.sysfunc.J_coords = J[:,self.coords[0]:(self.coords[-1]+1)]
        self.sysfunc.J_params = J[:,self.params[0]:(self.params[-1]+1)]

        if self.preTF is not None:
            self.preTF(X, V)


    def _createTestFuncs(self):
        """Creates processors and test functions for Continuation class.

        Note:  In the following list, processors are in PyCont.Bifpoint
        and test functions are in PyCont.TestFunc.

        Point type (Processor): Test Function(s)
        ----------------------------------------

        BP (BranchPoint): Branch_Det
        """
        self.TestFuncs = []
        self.BifPoints = {}

        for bftype in self.LocBifPoints:
            if bftype in cont_bif_points:
                stop = bftype in self.StopAtPoints  # Set stopping flag
                #if bftype is 'BP':
                    #method = Branch_Det(self.CorrFunc, self, save=True,
                                        #numpoints=self.MaxNumPoints+1)
                    #self.TestFuncs.append(method)

                    #self.BifPoints['BP'] = BranchPoint(method, iszero,
                                                       #stop=stop)
                if bftype is 'B':
                    method = B_Check(self.CorrFunc, self, save=True,
                                     numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['B'] = BPoint(method, iszero, stop=stop)

        if self.SPOut is not None:
            # add simple "user"-defined function to catch parameter values
            # during continuation
            for par, par_vals in self.SPOut.items():
                try:
                    par_ix = self.params[self.freepars.index(par)]
                except IndexError:
                    raise ValueError("Invalid free parameter %s" % par)
                for i, pval in enumerate(par_vals):
                    method = ParTestFunc(self.sysfunc.n,
                                         self, par_ix, pval, save=True,
                                         numpoints=self.NumSPOut+1)
                    self.TestFuncs.append(method)
                    self.BifPoints['SP-%s-%i' % (par, i)] = \
                        SPoint(method, iszero, stop=False)


    def _system(self, X):
        VARS = dict(zip(self.varslist, array(X)[self.coords]))
        for i, par in enumerate(self.freepars):
            self.parsdict[par] = X[self.params[i]]
        try:
            t = self.parsdict['time']
        except KeyError:
            # autonomous system, t doesn't matter
            t = 0
        return self.gensys.Rhs(t, VARS, self.parsdict, asarray=True)[self.varsindices]


    def _systemjac(self, x0, ind=None):
        VARS = dict(zip(self.varslist, array(x0)[self.coords]))
        for i, par in enumerate(self.freepars):
            self.parsdict[par] = x0[self.params[i]]
        try:
            t = self.parsdict['time']
        except KeyError:
            # autonomous system, t doesn't matter
            t = 0
        jacx = self.gensys.Jacobian(t, VARS, self.parsdict, asarray=True)[self.varsindices]
        jacp = self.sysfunc.diff(x0, ind=self.params)
        try:
            return c_[jacx, jacp][:,ind[0]:(ind[-1]+1)]
        except:
            return c_[jacx, jacp]


    def _systemjac_withpars(self, x0, ind=None):
        VARS = dict(zip(self.varslist, array(x0)[self.coords]))
        for i, par in enumerate(self.freepars):
            self.parsdict[par] = x0[self.params[i]]
        try:
            t = self.parsdict['time']
        except KeyError:
            # autonomous system, t doesn't matter
            t = 0
        jacx = self.gensys.Jacobian(t, VARS, self.parsdict, asarray=True)[self.varsindices]
        jacp = self.gensys.JacobianP(t, VARS, self.parsdict, asarray=True)[self.parsindices]
        try:
            return c_[jacx, jacp][:,ind[0]:(ind[-1]+1)]
        except:
            return c_[jacx, jacp]


    def _systemuser(self, X):
        """Calls self._userfunc, which is assumed to return an array of RHS
        values for the relevant (possibly subset of) variables."""
        VARS = dict(zip(self.varslist, array(X)[self.coords]))
        for i, par in enumerate(self.freepars):
            self.parsdict[par] = X[self.params[i]]
        return self._userfunc(self, VARS, self.parsdict)


    def _systemjacuser(self, x0, ind=None):
        """Calls self._userjac, which is assumed to return an array of
        [Jac_x, Jac_p]."""
        VARS = dict(zip(self.varslist, array(X)[self.coords]))
        for i, par in enumerate(self.freepars):
            self.parsdict[par] = X[self.params[i]]
        return self._userjac(self, VARS, self.parsdict)


    def _checkForBifPoints(self):
        # increase efficiency by preventing many self. references
        loc = self.loc
        # these declarations just make references
        curve = self.curve
        V = self.V
        # store commonly referenced values for efficiency
        V_loc = V[loc]
        curve_loc = curve[loc]
        for bftype, bfinfo in self.BifPoints.items():
            bftype = bftype.split('-')[0]
            flag_list = []
            for i, testfunc in enumerate(bfinfo.testfuncs):
                for k in range(testfunc.m):
                    flag_list.append(bfinfo.flagfuncs[i](testfunc[loc-1][k],
                                                         testfunc[loc][k]))

                # if bftype == 'NS':
                # 	print loc, bftype, flag_list, testfunc[loc]	# DREW WUZ HERE 2012
            bfpoint_found = all(flag_list)
            if bfpoint_found:
                # Locate bifurcation point
                Xval, Vval = bfinfo.locate((curve[loc-1], V[loc-1]),
                                           (curve_loc, V_loc), self)
                found = bfinfo.process(Xval, Vval, self)

                if found:
                    # Move information one more step forward
                    if not bfinfo.stop:
                        curve[loc+1] = curve_loc
                        V[loc+1] = V_loc
                        for testfunc in self.TestFuncs:
                            testfunc[loc+1] = testfunc[loc]
                    else:
                        startx = copy(curve_loc)
                        startv = copy(V_loc)

                    curve[loc] = Xval
                    V[loc] = Vval

                    self._savePointInfo(loc)
                    self.CurveInfo[loc] = (bftype,
                                        {'data': bfinfo.found[-1],
                                         'plot': args()})
                    if not bfinfo.stop:
                        self.loc += 1
                        loc += 1  # update in sync with self.loc
                        V_loc = V[loc]
                        curve_loc = curve[loc]
                    else:
                        self.CurveInfo[loc] = ('P',
                                {'data': args(V = todict(self, startv)),
                                 'startx': todict(self, startx),
                                 'plot': args()})
                        return True

        # Do not stop computations
        return False


    def exportGeomview(self, coords=None, filename="geom.dat"):
        if  coords is not None and len(coords) == 3:
            GeomviewOutput = "(progn (geometry " + self.model.name + \
                             " { LIST {: axes_" + self.model.name + "}"
            #        for cname, curve in self.curves.iteritems():
            GeomviewOutput += " {: " + self.name + "}"
            GeomviewOutput += "}))\n\n"

            # Get axes limits
            alim = [[Inf,-Inf],[Inf,-Inf],[Inf,-Inf]]
            #  for cname, curve in self.curves.iteritems():
            for n in range(len(coords)):
                alim[n][0] = min(alim[n][0], min(self.sol[coords[n]]))
                alim[n][1] = max(alim[n][1], max(self.sol[coords[n]]))

            GeomviewOutput += "(progn (hdefine geometry axes_" + \
                self.model.name + " { appearance { linewidth 2 } SKEL 4 3 " +\
                "0 0 0 1 0 0 0 1 0 0 0 1 " + \
                "2 0 1 1 0 0 1 2 0 2 0 1 0 1 2 0 3 0 0 1 1})\n\n"

            #for cname, curve in self.curves.iteritems():
            cname = self.name
            GeomviewOutput += "(hdefine geometry " + cname + \
                " { LIST {: curve_" + cname + "} {: specpts_" + cname + "}})\n\n"

            GeomviewOutput += "(hdefine geometry curve_" + cname + \
                " { appearance { linewidth 2 } SKEL " + \
                repr(len(self.sol)) + " " + repr(len(self.sol)-1)
            for n in range(len(self.sol)):
                GeomviewOutput += " " + repr((self.sol[n][coords[0]]-alim[0][0])/(alim[0][1]-alim[0][0])) + \
                    " " + repr((self.sol[n][coords[1]]-alim[1][0])/(alim[1][1]-alim[1][0])) + \
                    " " + repr((self.sol[n][coords[2]]-alim[2][0])/(alim[2][1]-alim[2][0]))
            for n in range(len(self.sol)-1):
                GeomviewOutput += " 2 " + repr(n) + " " + repr(n+1) + " 0 0 0 1"

            GeomviewOutput += "})\n\n"

            GeomviewOutput += ")\n"

            f = open(filename, "w")
            f.write(GeomviewOutput)
            f.close()
        else:
            raise PyDSTool_ValueError("Coordinates not specified or not of correct dimension.")


    def display(self, coords=None, dirs=None, origin=None, figure=None,
                axes=None, stability=False, domain=False, init_display=True,
                points=True, **plot_args):
        """Plot curve in coordinates specified by coords.

           Inputs:

               coords -- pair of coordinates (None defaults to the first free
                   parameter and the first state variable)
                   Use a 3-tuple to export to geomview.
               dirs -- tuple of coordinate directions IF coord is not in regular coords
               origin -- Useful if want affine coordinates
        """
        # Take care of calling with state variable w/o max/min for LC
        disp_args = copy(plot_args)
        if self.sol is not None:
            if coords is None:
                coords = [self.freepars[0], self.varslist[0]]
            if self.curvetype == 'LC-C':
                coords = list(coords)
                for n in range(2):
                    if coords[n] in self.varslist:
                        # Default to max of solution
                        coords[n] = coords[n]+'_max'

            if len(coords) == 3:
                self.exportGeomview(coords=coords)
                return

            if origin is not None:
                clist = self.sol.coordnames
                clen = len(clist)
                aorigin = array([origin[k] for k in clist])

            X = zeros((2,len(self.sol)), float)
            for n in range(2):
                if coords[n] in self.sol.coordnames:
                    X[n] = self.sol[coords[n]]
                    if origin is not None:
                        X[n] = X[n] - origin[coords[n]]
                elif coords[n] in self.parsdict.keys():
                    X[n] = array([self.parsdict[coords[n]]]*len(self.sol))
                    if origin is not None:
                        X[n] = X[n] - origin[coords[n]]
                elif dirs is not None and coords[n] in dirs.keys():
                    # Project curve onto plane spanned by coordinate directions
                    #   spanning variables and free parameters
                    X[n] = array([matrixmultiply(x-aorigin, dirs[coords[n]]) \
                                  for x in self.sol])
                else:
                    raise KeyError('Coordinate ' + coords[n] + ' is not defined.')

            if init_display:
                initializeDisplay(self.plot, figure=figure, axes=axes)
            cfl = self.plot._cfl
            cal = self.plot._cal

            ## Prints curve

            # Get unique name
            name = self.name
            if name in self.plot[cfl][cal]:
                num = 0
                for k, v in self.plot[cfl][cal].items():
                    if isinstance(v, pargs) and k.split('_')[0] == name:
                        num += 1
                name = name + '_' + repr(num)

            self.plot[cfl][cal][name] = pargs()
            self.plot[cfl][cal][name].curve = []
            label = self.curvetype.split('-')[0]
            self.plot[cfl][cal][name].type = label
            if stability and self.SaveEigen:
                if 'linewidth' not in disp_args:    # Default linewidth 1
                    disp_args['linewidth'] = 1
                disp_args['label'] = '_nolegend_'
                stabdict = partition([x.labels[label]['stab'] \
                                      for x in self.sol],['S','U','N'])
                for stabtype, stablist in stabdict.items():
                    for curve in stablist:
                        self.plot[cfl][cal][name].curve.extend(plt.plot(X[0][curve[0]:curve[1]], \
                                                                          X[1][curve[0]:curve[1]], \
                            bif_curve_colors[self.curvetype]+stab_line_styles[stabtype], **disp_args))
            else:
                if 'label' not in disp_args:
                    disp_args['label'] = name
                self.plot[cfl][cal][name].curve.extend(plt.plot(X[0], X[1], \
                                bif_curve_colors[self.curvetype], **disp_args))

            # Take care of labels
            xlab = coords[0]
            ylab = coords[1]
            if self.curvetype == 'LC-C':
                for smtype in self.SolutionMeasures:
                    if xlab.rfind('_'+smtype) > 0:
                        xlab = xlab[0:xlab.rfind('_'+smtype)]
                        break

                for smtype in self.SolutionMeasures:
                    if ylab.rfind('_'+smtype) > 0:
                        ylab = ylab[0:ylab.rfind('_'+smtype)]
                        break

            plt.xlabel(xlab)
            plt.ylabel(ylab)

            # Prints special points
            if points:
                for bftype in all_point_types:
                    bflist = self.sol.bylabel(bftype)
                    if bflist is not None:
                        for point in bflist:
                            if 'name' in point.labels[bftype]:
                                X = zeros(2, float)
                                for n in range(2):
                                    if coords[n] in self.sol.coordnames:
                                        X[n] = point[coords[n]]
                                        if origin is not None:
                                            X[n] = X[n] - origin[coords[n]]
                                    elif coords[n] in self.parsdict.keys():
                                        X[n] = self.parsdict[coords[n]]
                                        if origin is not None:
                                            X[n] = X[n] - origin[coords[n]]
                                    elif dirs is not None and coords[n] in dirs.keys():
                                        # Project point onto plane spanned by coordinate directions
                                        #   spanning variables and free parameters
                                        X[n] = matrixmultiply(point-aorigin,
                                                              dirs[coords[n]])

                                # Print point
                                ptname = point.labels[bftype]['name']
                                self.plot[cfl][cal][name][ptname] = pargs()
                                self.plot[cfl][cal][name][ptname].point = \
                                        plt.plot([X[0]], [X[1]],
                                                   bif_point_colors[bftype],
                                                   label='_nolegend_')

                                # Print label
                                ha = 'left'
                                if self.curvetype in ['LP-C','H-C1','H-C2','LC-C']:
                                    va = 'top'
                                else:
                                    va = 'bottom'

                                self.plot[cfl][cal][name][ptname].text = \
                                    plt.text(X[0], X[1], '  '+ ptname,
                                               ha=ha, va=va)


    def _savePointInfo(self, loc):
        """Created a function for this since it needs to be called
        both in _compute and when a bifurcation point is found.  It
        will have conditional statements for saving of Jacobian and
        eigenvalues, as well as other possible tidbits of
        information."""
        ptlabel = self._ptlabel
        self.CurveInfo[loc] = (ptlabel, \
            {'data': args(V = todict(self, self.V[loc]),
                          ds = self.StepSize)})

        # Save domain information
        if 'B' in self.LocBifPoints:
            val = self.BifPoints['B'].testfuncs[0][loc][0]
            # if val >= 0 set domain = 'inside' otherwise 'outside'
            self.CurveInfo[loc][ptlabel]['domain'] = (val >= 0) \
                and 'inside' or 'outside'

        # Save eigenvalue information
        if self.SaveEigen:
            # May be able to use J_coords here
            jac = self.sysfunc.jac(self.curve[loc])
            jacx = jac[:,self.coords[0]:(self.coords[-1]+1)]
            jacp = jac[:,self.params[0]:(self.params[-1]+1)]
            w, vr = linalg.eig(jacx)
            self.CurveInfo[loc][ptlabel]['data'].evals = w
            self.CurveInfo[loc][ptlabel]['data'].evecs = vr

            if ptlabel == 'FP':
                inside = [abs(eig) < 1-1e-6 for eig in w]
                outside = [abs(eig) > 1+1e-6 for eig in w]
                if all(inside):
                    self.CurveInfo[loc][ptlabel]['stab'] = 'S'
                elif all(outside):
                    self.CurveInfo[loc][ptlabel]['stab'] = 'U'
                else:
                    self.CurveInfo[loc][ptlabel]['stab'] = 'N'
            else:
                realpos = [real(eig) > 1e-6 for eig in w]
                realneg = [real(eig) < -1e-6 for eig in w]
                if all(realneg):
                    self.CurveInfo[loc][ptlabel]['stab'] = 'S'
                elif all(realpos):
                    self.CurveInfo[loc][ptlabel]['stab'] = 'U'
                else:
                    self.CurveInfo[loc][ptlabel]['stab'] = 'N'

        # Save jacobian information
        if self.SaveJacobian:
            try:
                self.CurveInfo[loc][ptlabel]['data'].jacx = jacx
                self.CurveInfo[loc][ptlabel]['data'].jacp = jacp
            except:
                jac = self.sysfunc.jac(self.curve[loc])
                jacx = jac[:,self.coords[0]:(self.coords[-1]+1)]
                jacp = jac[:,self.params[0]:(self.params[-1]+1)]
                self.CurveInfo[loc][ptlabel]['data'].jacx = jacx
                self.CurveInfo[loc][ptlabel]['data'].jacp = jacp

        if ptlabel == 'UD':
            self.CurveInfo[loc][ptlabel]['data'].update(self._userdata)


    def _MoorePenrose(self, X, V):
        k, converged = 0, 0
        problem = 0     # Only necessary for UserDefined classes
        Xold = X.copy()
        x0 = 0.0
        x1 = 0.0
        diag = args()
        diag.cond = []
        diag.nrm = []
        fun = self.CorrFunc
        jac = self.CorrFunc.jac
        while not problem and not converged and k < self.MaxCorrIters:
            A = jac(X)
            B = r_[A,[V]]
            R = r_[matrixmultiply(A,V),0]
            Q = r_[fun(X),0]
            if self.curvetype == 'UD-C' and 'problem' in self._userdata \
               and self._userdata.problem:
                problem = 1
                break
            if self.verbosity >= 10:
                print("  [%d]" % k)
                u, s, vh = linalg.svd(B)
                cond = s[0]/s[-1]
                diag.cond.append(cond)
                print("    Log(Condition #) = %lf" % log10(cond))
            WX = linalg.solve(B,mat([R,Q]).T)
            subtract(V, WX[:,0], V)
            divide(V, linalg.norm(V), V)
            subtract(X, WX[:,1], X)

            # Check for convergence
            Fnrm = linalg.norm(self.CorrFunc(X))
            Vnrm = linalg.norm(WX[:,1])
            converged = Fnrm < self.FuncTol and Vnrm < self.VarTol
            if self.verbosity >= 10:
                print('    (Fnrm, Vnrm) = (%.12f,%.12f)' % (Fnrm, Vnrm))
                x0 = x1
                x1 = linalg.norm(X-Xold)
                Xold = X.copy()
                diag.nrm.append((Fnrm, Vnrm))
            k += 1
        return k, converged, problem, diag


    def _Natural(self, X, V):
        # Get index of coordinate with maximal change
        k, converged = 0, 0
        problem = 0     # Only necessary for UserDefined classes
        Xold = X.copy()    # Use for secant predictor below
        x0 = 0.0
        x1 = 0.0
        diag = args()
        diag.cond = []
        diag.nrm = []

        ind = argsort(V)[-1]
        vi = zeros(len(X), float64)
        vi[ind] = 1.0
        xi = X[ind]
        fun = self.CorrFunc
        jac = self.CorrFunc.jac
        while not problem and not converged and k < self.MaxCorrIters:
            # Newton's method: X_{n+1} = X_{n} - W, where W = B^{-1}*Q
            A = jac(X)
            B = r_[A,[vi]]
            Q = r_[fun(X), X[ind] - xi]
            if self.curvetype == 'UD-C' and 'problem' in self._userdata \
                                        and self._userdata.problem:
                problem = 1
                break
            if self.verbosity >= 10:
                print("  [%d]" % k)
                u, s, vh = linalg.svd(B)
                cond = s[0]/s[-1]
                diag.cond.append(cond)
                print("    Log(Condition #) = %lf" % log10(cond))
            W = linalg.solve(B, Q)
            subtract(X, W, X)

            # Check for convergence
            Fnrm = linalg.norm(self.CorrFunc(X))
            Vnrm = linalg.norm(W)
            converged = Fnrm < self.FuncTol and Vnrm < self.VarTol
            if self.verbosity >= 10:
                print('    (Fnrm, Vnrm) = (%.12f,%.12f)' % (Fnrm, Vnrm))
                x0 = x1
                x1 = linalg.norm(X-Xold)
                Xold = X.copy()
                diag.nrm.append((Fnrm, Vnrm))
            k += 1

        V = linalg.solve(r_[A,[V]], r_[zeros((self.varsdim, 1), float), [[1.]]])
        V = V/linalg.norm(V)
        return k, converged, problem, diag


    def _Keller(self, X, V):
        k, converged = 0, 0
        problem = 0     # Only necessary for UserDefined classes
        Xold = X.copy()
        x0 = 0.0
        x1 = 0.0
        diag = args()
        diag.cond = []
        diag.nrm = []
        fun = self.CorrFunc
        jac = self.CorrFunc.jac
        while not problem and not converged and k < self.MaxCorrIters:
            A = jac(X)
            B = r_[A,[V]]
            Q = r_[fun(X),
                   matrixmultiply(X-self.curve[self.loc-1],V)-self.StepSize]
            if self.curvetype == 'UD-C' and 'problem' in self._userdata \
               and self._userdata.problem:
                problem = 1
                break
            if self.verbosity >= 10:
                print("  [%d]" % k)
                u, s, vh = linalg.svd(B)
                cond = s[0]/s[-1]
                diag.cond.append(cond)
                print("    Log(Condition #) = %lf" % log10(cond))
            W = linalg.solve(B, Q)
            subtract(X, W, X)

            # Check for convergence
            Fnrm = linalg.norm(self.CorrFunc(X))
            Vnrm = linalg.norm(W)
            converged = Fnrm < self.FuncTol and Vnrm < self.VarTol
            if self.verbosity >= 10:
                print('    (Fnrm, Vnrm) = (%.12f,%.12f)' % (Fnrm, Vnrm))
                x0 = x1
                x1 = linalg.norm(X-Xold)
                Xold = X.copy()
                diag.nrm.append((Fnrm, Vnrm))
            k += 1

        V = linalg.solve(r_[A,[V]], r_[zeros((self.varsdim,1), float), [[1.]]])
        V = V/linalg.norm(V)
        return k, converged, problem, diag


    def _compute(self, x0=None, v0=None, direc=1):
        """Continuation using Moore-Penrose method (called by forward
        and backward methods)

        NOTE: For codimension 2 curves, CorrFunc is the augmented
        system consisting of sysfunc with testfunc associated with the
        curve given by sysfunc.  When you call CorrFunc, it calls
        sysfunc.PreTestFunc to calculate the jacobian that testfunc
        needs.  THUS, sysfunc jacobians are computed and stored in
        sysfunc.  Now, if you have test functions that require
        jacobian information from sysfunc and CorrFunc, I only call
        CorrFunc.PreTestFunc. BUT, it still works because CorrFunc was
        called just previous, thereby saving sysfunc jacobians that
        are needed by the test functions that rely on sysfunc
        jacobians.  Understand? Good, cuz I barely do.  I'm going to
        try and alleviate this confusion.  Wish me luck...  By the
        way, the example of this is in computing ZH points.  They
        require test functions for a codimension 1 curve while
        continuing a codimension 2 curve.
        """

        # Process X
        if x0 is None:
            x0 = self.initpoint

        if isinstance(x0, dict):
            x0 = tocoords(self, x0)
        elif isinstance(x0, list):
            x0 = array(x0)
        elif isinstance(x0, Point):
            x0 = x0.todict()
            for p in self.freepars:
                if p not in x0.keys():
                    x0[p] = self.parsdict[p]
            x0 = tocoords(self, x0)

        # Get Jacobian information
        if self.sysfunc == self.CorrFunc:
            # It's codim 1 and can use the existing Jac (in case it's symbolic)
            J = self.CorrFunc.jac(x0)
            J_coords = J[:,self.coords]
            J_params = J[:,self.params]
        else:
            # SHORTCOMING HERE:
            # replace Jac with numerical diff, regardless of whether
            # original sysfunc had a symbolic Jac
            self.CorrFunc.jac = Function((self.CorrFunc.n,
                                      (self.CorrFunc.m,self.CorrFunc.n)), \
                          self.CorrFunc.diff, numpoints=self.MaxNumPoints+1)
            J_coords = self.CorrFunc.jac(x0, self.coords)
            J_params = self.CorrFunc.jac(x0, self.params)

        # Process V
        if v0 is None:
            # Compute tangent vector v0 to curve at ix0 by solving:
            #   [A; rand(n)] * v0 = [zeros(n-1); 1]
            #
            singular = True
            perpvec = r_[1,zeros(self.dim-1)]
            d = 1
            while singular and d <= self.dim:
                try:
                    v0 = linalg.solve(r_[c_[J_coords, J_params],
                                         [perpvec]], \
                                      r_[zeros(self.dim-1),1])
                except:
                    perpvec = r_[0., perpvec[0:(self.dim-1)]]
                    d += 1
                    if self.verbosity >= 10:
                        print("%d: %lf" % (d,log10(cond(r_[c_[J_coords, J_params], [perpvec]]))))
                else:
                    singular = False
                    if self.verbosity >= 10:
                        print("%d: %lf" % (d,log10(cond(r_[c_[J_coords, J_params], [perpvec]]))))
                    # perpvec = r_[0., perpvec[0:(self.dim-1)]]
                    # d += 1

            if singular:
                raise PyDSTool_ExistError("Problem in _compute: Failed to compute tangent vector.")
#             v0 = zeros(self.dim, float)
#             v0 = linalg.solve(r_[c_[J_coords, J_params],
#                                  [2*(random(self.dim)-0.5)]], \
#                               r_[zeros(self.dim-1),1])
            v0 /= linalg.norm(v0)
            v0 = direc*sign([x for x in v0 if abs(x) > 1e-8][0])*v0
        elif isinstance(v0, dict):
            v0 = direc*tocoords(self, v0)
        elif isinstance(v0, list):
            v0 = direc*array(v0)
        elif isinstance(v0, Point):
            v0 = direc*v0.toarray()

        self.V = zeros((self.MaxNumPoints+1,self.dim), float)
        self.V[0] = v0

        # Start on curve
        # NOTE: If having trouble with losing branch given by initdirec, then save
        #   self.V[0] = v0.copy() and replace after.
        self.curve = zeros((self.MaxNumPoints+1, self.dim), float)
        self.curve[0] = x0

        # curve and V are arrays and so will be copied by reference only
        curve = self.curve
        V = self.V

        converged = True
        attempts = 0
        val = linalg.norm(self.CorrFunc(x0))
        while val > self.FuncTol or not converged:
            try:
                k, converged, problem, diag = self.Corrector(curve[0], V[0])
            except:
                converged = False
                print("Error occurred in dynamical system computation")
            else:
                val = linalg.norm(self.CorrFunc(curve[0]))
            attempts += 1
            if not converged and attempts > 1:
                # Stop continuation
                self.Corrector = self._Natural
                if attempts > 2:
                    print("Not converged: ", curve[0], "\n")
                    raise PyDSTool_ExistError("Could not find starting point on curve.  Stopping continuation.")
        # Initialize index location on curve data set
        self.loc = 0
        if self.verbosity >= 3:
            print('    Found initial point on curve: ' + \
                  repr(todict(self, curve[0])))

        # Initialize test functions
        self._createTestFuncs()

        # self.CorrFunc.J_coords = self.CorrFunc.jac(x0,self.coords)
        # self.CorrFunc.J_params = self.CorrFunc.jac(x0,self.params)
        # old above (2 lines); r345 below (5 lines)
        x0 = curve[0]
        v0 = V[0]
        J = self.CorrFunc.jac(x0)
        self.CorrFunc.J_coords = J[:,self.coords]
        self.CorrFunc.J_params = J[:,self.params]

        if self.TestFuncs != []:
            for testfunc in self.TestFuncs:
                if hasattr(testfunc, 'setdata'):
                    testfunc.setdata(x0, v0)
            self._preTestFunc(x0, v0)
            for testfunc in self.TestFuncs:
                testfunc[self.loc] = testfunc(x0, v0)

        # Save initial information
        self._savePointInfo(self.loc)
        self.CurveInfo[0] = ('P', {'data': args(V = todict(self, v0)), \
                                   'plot': args()})

        # Stepsize control parameters
        CorrThreshold = 6
        SSC_c, SSC_C = 0.8, 1.2

        if self.verbosity >= 3:
            print('    Beginning continuation...')
        # Continuation loop
        closed = False
        stop = False
        problem = False
        if self.curvetype == 'UD-C' and 'problem' in self._userdata:
            self._userdata.problem = False

        # de-references to improve efficiency
        loc = self.loc  # integer, so self.loc won't get updated automatically
        while loc+1 < self.MaxNumPoints and not stop:
            # Predictor
            loc += 1
            curve[loc] = curve[loc-1] + self.StepSize*V[loc-1]
            V[loc] = V[loc-1]

            # Corrector -- update self.loc for Corrector's reference
            self.loc = loc

            try:
                k, converged, problem, diag = self.Corrector(curve[loc], V[loc])
            except:
                problem = True
            #if self._userdata.has_key('problem'):  # Uncomment the these three lines to "find" the boundary between regions
            #    self._userdata.problem = False
            #    problem = False

            if self.verbosity >= 10:
                print("Step #%d:" % loc)
                print("  Corrector steps: %d/%d" % (k, self.MaxCorrIters))
                print("  Converged: %d" % (converged and not problem))

            if problem:
                stop = True
            elif not converged:
                loc -= 1
                if self.StepSize > self.MinStepSize:
                    # Reduce stepsize and try again
                    if self.verbosity >= 3:
                        print("Trouble converging.  Reducing stepsize. (ds=%lf)" % self.StepSize)
                    self.StepSize = max(self.MinStepSize, self.StepSize*SSC_c)
                else:
                    # Stop continuation
                    print("Did not converge.  Stopping continuation.  Reduce MinStepSize to continue.")
                    raise PyDSTool_ExistError("Did not converge. Stopping continuation. Reduce MinStepSize to continue")
            else:
                # Increase stepsize for fast convergence
                if self.StepSize < self.MaxStepSize and k < CorrThreshold:
                    self.StepSize = min(self.MaxStepSize, self.StepSize*SSC_C)

                # Evaluate test functions
                if self.TestFuncs is not None:
                    self._preTestFunc(curve[loc], V[loc])
                    for testfunc in self.TestFuncs:
                        testfunc[loc] = testfunc(curve[loc], V[loc])

                # DREW WUZ HERE 2012
                #if self.curvetype == 'FP-C':
                #    print array([linalg.det(r_[c_[self.sysfunc.J_coords - eye(self.sysfunc.m, self.sysfunc.m), self.sysfunc.J_params],[V[loc]]])])

                # Check for bifurcation points.
                # If _checkForBifPoints returns True, stop loop
                # update self.loc for Corrector's reference
                self.loc = loc
                if self.BifPoints != {} and self._checkForBifPoints():
                    stop = True

                # Checks to see if curve is closed and if closed, it closes the curve
                if self.ClosedCurve < loc+1 < self.MaxNumPoints and \
                   linalg.norm(curve[loc]-curve[0]) < self.StepSize:
                    # Need to be able to copy PointInfo information
                    print("Detected closed curve.  Stopping continuation...\n")
                    curve[loc+1] = curve[0]
                    V[loc+1] = V[0]

                    for testfunc in self.TestFuncs:
                        testfunc[loc+1] = testfunc[loc]

                    self._savePointInfo(loc)
                    loc += 1
                    self._savePointInfo(loc)

                    closed = True
                    break

                # Print information
                if self.verbosity >= 4:
                    print("Loc = %4d    %s = %lf" % (loc, self.freepars[0],
                                                     curve[loc][-1]))

                # Save information
                self._savePointInfo(loc)

        # Finish updating self.loc integer location
        self.loc = loc
        # Save end point information
        if problem:
            #self.CurveInfo[loc] = ('UD', {'data': args(V = todict(self,
            #                            V[self.loc]), ds=self.StepSize)})
            self.CurveInfo[loc] = ('MX', {'data': args(V = todict(self,
                                        V[loc]), ds=self.StepSize)})
        elif not closed and 'P' not in self.CurveInfo[loc]:
            self.CurveInfo[loc] = ('P', {'data': args(V = todict(self,
                                        V[loc])), 'plot': args()})


    def forward(self):
        """Computes forward along curve from initpoint if this is the first run.  Otherwise, it computes
        forward along curve from the last point on the saved solution sol.  The new curve is appended to
        the end of sol."""
        #if self.gensys.haveJacobian_pars():
        #    raise NotImplementedError("Jacobian with respect to parameters is not currently implemented in PyCont.")

        self.CurveInfo = PointInfo()
        if self.sol is None:
            self._compute(v0=self.initdirec)
            self.sol = self._curveToPointset()

            for pttype in self.LocBifPoints + other_special_points:
                bylabels = self.sol.bylabel(pttype)
                if bylabels is not None:
                    num = 1
                    for pt in bylabels:
                        if pttype in pt.labels:
                            pt.labels[pttype]['name'] = pttype + repr(num)
                        else:
                            pt.labels[pttype] = {'name': pttype + repr(num)}
                        if 'cycle' in pt.labels[pttype]:
                            pt.labels[pttype]['cycle'].name = pttype + repr(num)
                        num += 1
            self.new_sol_segment = copy(self.sol)
        else:
            # find final non-MX point that has the correct label for this curve type
            pt_type = self.curvetype.split('-')[0]
            i = -1
            while True:
                try:
                    sol1 = self.sol[i]
                except IndexError:
                    raise IndexError("Not enough points found")
                if 'MX' not in sol1.labels and pt_type in sol1.labels:
                    break
                else:
                    # !!! some points do not seem to have any labels
                    # unclear whether these are legit points or not, so
                    # ignore for now and keep going
                    i -= 1

            # Set start point (if bif point, set to startx)
            if 'startx' in sol1.labels[pt_type]:
                x0 = sol1.labels[pt_type]['startx']
            else:
                x0 = sol1

            try:
                v0 = sol1.labels[pt_type]['data'].V
            except:
                v0 = None

            if 'ds' in sol1.labels[self.curvetype.split('-')[0]]['data']:
                self.StepSize = min(self.StepSize,
                          sol1.labels[self.curvetype.split('-')[0]]['data'].ds)

            self._compute(x0=x0, v0=v0)
            sol = self._curveToPointset()[1:]

            # Fix labels
            try:
                self.sol.labels.remove(len(self.sol)-1,'P')
            except:
                pass

            # RC: Why remove the MX? I think there was a case where it was needed...
            #try:
            #    self.sol.labels.remove(len(self.sol)-1,'MX')
            #except:
            #    pass

            for pttype in self.LocBifPoints + other_special_points:
                if self.sol.bylabel(pttype) is not None:
                    num = len(self.sol.bylabel(pttype)) + 1
                else:
                    num = 1
                bylabels = sol.bylabel(pttype)
                if bylabels is not None:
                    for pt in bylabels:
                        if pttype in pt.labels:
                            pt.labels[pttype]['name'] = pttype + repr(num)
                        else:
                            pt.labels[pttype] = {'name': pttype + repr(num)}
                        if 'cycle' in pt.labels[pttype]:
                            pt.labels[pttype]['cycle'].name = pttype + repr(num)
                        num += 1

            self.new_sol_segment = copy(sol)
            self.sol.append(sol)


    def backward(self):
        """Computes backward along curve from initpoint if this is the
        first run.  Otherwise, it computes backward along curve from
        the first point on the saved solution sol.  The new curve is
        appended to the front of sol.
        """
        #if self.gensys.haveJacobian_pars():
        #    raise NotImplementedError('Jacobian with respect to parameters is not currently implemented in PyCont.')

        self.CurveInfo = PointInfo()
        if self.sol is None:
            self._compute(v0=self.initdirec, direc=-1)
            self.sol = self._curveToPointset()

            # Turn curve around
            self.sol.reverse()
            sol0 = self.sol[0]

            # Type of end point
            if 'P' in sol0.labels:
                etype0 = 'P'
            else:
                etype0 = 'MX'
            if 'P' in self.sol[-1].labels:
                etype1 = 'P'
            else:
                etype1 = 'MX'

            # Turn tangent vectors around (for non auto only)
            if not self.UseAuto:
                # Turn tangent vectors at end point type around
                for k, v in sol0.labels[etype0]['data'].V.items():
                    sol0.labels[etype0]['data'].V[k] = -1*v

                for k, v in self.sol[-1].labels[etype1]['data'].V.items():
                    self.sol[-1].labels[etype1]['data'].V[k] = -1*v

                # Turn tangent vectors at curve type around (PROBLEM AT MX POINTS)
                ctype = self.curvetype.split('-')[0]
                for pt in self.sol:
                    for k, v in pt.labels[ctype]['data'].V.items():
                        pt.labels[ctype]['data'].V[k] = -1*v

            for pttype in self.LocBifPoints + other_special_points:
                bylabels = self.sol.bylabel(pttype)
                if bylabels is not None:
                    num = 1
                    for pt in bylabels:
                        if pttype in pt.labels:
                            pt.labels[pttype]['name'] = pttype + repr(num)
                        else:
                            pt.labels[pttype] = {'name': pttype + repr(num)}
                        if 'cycle' in pt.labels[pttype]:
                            pt.labels[pttype]['cycle'].name = pttype + repr(num)
                        num += 1
            self.new_sol_segment = copy(self.sol)
        else:
            sol0 = self.sol[0]
            # Set start point (if bif point, set to startx)
            if 'startx' in sol0.labels['P']:
                x0 = sol0.labels['P']['startx']
            else:
                x0 = sol0

            try:
                v0 = sol0.labels['P']['data'].V
            except:
                try:
                    v0 = sol0.labels['MX']['data'].V
                except:
                    v0 = None

            ctype = self.curvetype.split('-')[0]
            #try:
            #    self.StepSize = self.sol[0].labels[ctype]['data'].ds
            #except:
            #    pass
            if 'ds' in sol0.labels[self.curvetype.split('-')[0]]['data']:
                self.StepSize = min(self.StepSize,
                    sol0.labels[self.curvetype.split('-')[0]]['data'].ds)

            self._compute(x0=x0, v0=v0, direc=-1)
            sol = self._curveToPointset()

            # Turn curve around
            sol.reverse()
            sol0 = sol[0]

            # Type of end point
            etype = 'P' in sol0.labels and 'P' or 'MX'

            if etype in sol0.labels:
                sol0.labels[etype]['name'] = etype+'1'
            else:
                sol0.labels[etype] = {'name': etype+'1'}
            if 'cycle' in sol0.labels[etype]:
                sol0.labels[etype]['cycle'].name = etype+'1'

            # Turn tangent vectors around (for non auto only)
            if not self.UseAuto:
                # Turn tangent vector around at point type endtype and change name
                for k, v in sol0.labels[etype]['data'].V.items():
                    sol0.labels[etype]['data'].V[k] = -1*v

                # Turn tangent vectors at curve type around
                for pt in sol:
                    for k, v in pt.labels[ctype]['data'].V.items():
                        pt.labels[ctype]['data'].V[k] = -1*v

            # Fix labels
            try:
                self.sol.labels[0].pop('P')
            except:
                self.sol.labels[0].pop('MX')

            for pttype in self.LocBifPoints + ['RG', 'UZ', 'MX']:
                bylabels = self.sol.bylabel(pttype)
                if bylabels is not None:
                    num = len(bylabels) + 1
                else:
                    num = 1
                new_bylabels = sol.bylabel(pttype)
                if new_bylabels is not None:
                    for pt in new_bylabels:
                        if pttype in pt.labels:
                            pt.labels[pttype]['name'] = pttype + repr(num)
                        else:
                            pt.labels[pttype] = {'name': pttype + repr(num)}
                        if 'cycle' in pt.labels[pttype]:
                            pt.labels[pttype]['cycle'].name = pttype + repr(num)
                        num += 1

            self.new_sol_segment = copy(sol)
            sol.append(self.sol)
            self.sol = sol


    def testdomain(self, ic=None, ind=0, direc=1):
        if ic is None:
            ic = self.initpoint
        else:
            x0 = ic.copy()
            x0 = x0.todict()
            for p in self.freepars:
                if p not in x0.keys():
                    x0[p] = self.parsdict[p]
            x0 = tocoords(self, x0)

        dy = 0.5
        Dy = 0.0
        self.CorrFunc.jac = Function((self.CorrFunc.n,
                                      (self.CorrFunc.m,self.CorrFunc.n)), \
                                     self.CorrFunc.diff, numpoints=1)

        # Process V
        try:
            icv = ic.labels['UD']['data'].V
        except:
            icv = None

        if icv is None:
            # Get Jacobian information
            print("Creating vector...\n")
            J_coords = self.CorrFunc.jac(x0, self.coords)
            J_params = self.CorrFunc.jac(x0, self.params)

            v0 = zeros(self.dim, float)
            v0 = linalg.solve(r_[c_[J_coords, J_params],
                                 [2*(random(self.dim)-0.5)]], \
                              r_[zeros(self.dim-1),1])
            v0 /= linalg.norm(v0)
            v0 = array(sign([x for x in v0 if abs(x) > 1e-8][0])*v0)
        else:
            v0 = icv.copy()
            v0 = tocoords(self, v0)

        ic = x0.copy()
        icv = v0.copy()
        #while (abs(x0[ind] - ic[ind]) < 1.0):
        out = []
        while (dy >= 1e-4):
            print("%s = %lf" % (self.varslist[ind], x0[ind]))
            # Get Jacobian information
            #J_coords = self.CorrFunc.jac(x0, self.coords)
            #J_params = self.CorrFunc.jac(x0, self.params)

            k, converged, problem, diag = self.Corrector(x0, v0)
            print('x0 = ', x0)
            if not converged:
                print("  Did not converge.")
                if (Dy >= dy):
                    print("  Changing stepsize.")
                    Dy -= dy
                    dy *= 0.5
                else:
                    print("  Minimum reached.  Stopping simulation.")
                    raise PyDSTool_ExistError("Failed to converge")
            else:
                out.append((Dy, diag))
                #print "  Converged.  Avg. cond. # = %lf" % (csum/k)
                print("  Converged.  Avg. cond. # = %lf" % (sum(diag.cond)/len(diag.cond)))

                # Takes care of "infinite" domains
                if (Dy >= 5.0):
                    Dy -= dy
                    dy *= 0.5

            self._userdata.problem = False

            x0 = ic.copy()
            Dy += dy
            x0[ind] += direc*Dy

        return out


    def testdomaingrid(self, ic=None, coords=('y', 'theta'), Dx=None, Dy=None, step=2):
        key = ['y', 'theta', 'a']
        ind = (key.index(coords[0]), key.index(coords[1]))

        if ic is None:
            ic = self.initpoint
        else:
            ic = self.sol[ic]
            x0 = ic.copy()
            x0 = x0.todict()
            for p in self.freepars:
                if p not in x0.keys():
                    x0[p] = self.parsdict[p]
            x0 = tocoords(self, x0)

        self.CorrFunc.jac = Function((self.CorrFunc.n,
                                      (self.CorrFunc.m,self.CorrFunc.n)), \
                                     self.CorrFunc.diff, numpoints=1)

        # Process V
        try:
            icv = ic.labels['UD']['data'].V
        except:
            icv = None

        if icv is None:
            # Get Jacobian information
            print("Creating vector...\n")
            J_coords = self.CorrFunc.jac(x0, self.coords)
            J_params = self.CorrFunc.jac(x0, self.params)

            v0 = zeros(self.dim, float)
            v0 = linalg.solve(r_[c_[J_coords, J_params],
                                 [2*(random(self.dim)-0.5)]], \
                              r_[zeros(self.dim-1),1])
            v0 /= linalg.norm(v0)
            v0 = array(sign([x for x in v0 if abs(x) > 1e-8][0])*v0)
        else:
            v0 = icv.copy()
            v0 = tocoords(self, v0)

        out = []
        if Dx is None:
            xmin = min([pt[coords[0]] for pt in self.sol])
            xmax = max([pt[coords[0]] for pt in self.sol])
        else:
            xmin = x0[ind[0]]-Dx
            xmax = x0[ind[0]]+Dx
        Dx = xmax-xmin
        dx = Dx/(2*step)

        if Dy is None:
            ymin = min([pt[coords[1]] for pt in self.sol])
            ymax = max([pt[coords[1]] for pt in self.sol])
        else:
            ymin = x0[ind[1]]-Dy
            ymax = x0[ind[1]]+Dy
        Dy = ymax-ymin
        dy = Dy/(2*step)

        ic = array(x0.copy())
        icv = array(v0.copy())
        for i in range(2*step+1):
            out.append([])
            for j in range(2*step+1):
                #x0 = array(ic.copy())
                #v0 = array(icv.copy())
                x0 = ic.copy()
                v0 = icv.copy()
                x0[ind[0]] = xmin + j*dx
                x0[ind[1]] = ymax - i*dy
                ix0 = x0.copy()
                print("(%d, %d) -- (%s, %s) = (%lf, %lf)" % (i, j, key[ind[0]], key[ind[1]], x0[ind[0]], x0[ind[1]]))

                # Get Jacobian information
                #J_coords = self.CorrFunc.jac(x0, self.coords)
                #J_params = self.CorrFunc.jac(x0, self.params)

                #v0 = zeros(self.dim, float)
                #v0 = linalg.solve(r_[c_[J_coords, J_params], [2*(random(self.dim)-0.5)]], \
                #    r_[zeros(self.dim-1),1])
                #v0 /= linalg.norm(v0)
                #v0 = array(sign([x for x in v0 if abs(x) > 1e-8][0])*v0)

                k, converged, problem, diag = self.Corrector(x0, v0)

                print('x0 = ', x0)
                if not converged:
                    print("  Did not converge.")
                    if len(diag.cond) > 0:
                        print("  Avg. cond. # = %lf" % (sum(diag.cond)/len(diag.cond)))
                    if problem:
                        # Did not converge due to failure in backward integration
                        out[i].append(('XB', ix0, x0, diag))
                    elif k >= self.MaxCorrIters:
                        Fnm = [nm[0] for nm in diag.nrm]
                        if Fnm[-1] < Fnm[-2] and Fnm[-2] < Fnm2[-3]:
                            # Did not converge but may have converged with more timesteps
                            out[i].append(('XC', ix0, x0, diag))
                        else:
                            # Did not converge for unknown reason
                            out[i].append(('X', ix0, x0, diag))
                else:
                    print("  Converged.  Avg. cond. # = %lf" % (sum(diag.cond)/len(diag.cond)))
                    # Converged
                    out[i].append(('C', ix0, x0, diag))

                self._userdata.problem = False

        return out


    def testdomaintangrid(self, Dx, Dy, ic=None, step=2):
        key = ['y', 'theta', 'a']

        if ic is None:
            ic = self.initpoint
        else:
            ic = self.sol[ic]
            x0 = ic.copy()
            x0 = x0.todict()
            for p in self.freepars:
                if p not in x0.keys():
                    x0[p] = self.parsdict[p]
            x0 = tocoords(self, x0)

        self.CorrFunc.jac = Function((self.CorrFunc.n,
                                      (self.CorrFunc.m,self.CorrFunc.n)), \
                                     self.CorrFunc.diff, numpoints=1)

        # Process V
        try:
            icv = ic.labels['UD']['data'].V
        except:
            icv = None

        if icv is None:
            # Get Jacobian information
            print("Creating vector...\n")
            J_coords = self.CorrFunc.jac(x0, self.coords)
            J_params = self.CorrFunc.jac(x0, self.params)

            v0 = zeros(self.dim, float)
            v0 = linalg.solve(r_[c_[J_coords, J_params],
                                 [2*(random(self.dim)-0.5)]], \
                              r_[zeros(self.dim-1),1])
            v0 /= linalg.norm(v0)
            v0 = array(sign([x for x in v0 if abs(x) > 1e-8][0])*v0)
        else:
            v0 = icv.copy()
            v0 = tocoords(self, v0)

        v1 = zeros(self.dim, float)
        v1[2] = 1.0
        v1 = v1 - v0[2]*v0
        v1 = v1/linalg.norm(v1)
        print("Checking orthonormal...")
        print("  |v0| = %lf" % linalg.norm(v0))
        print("  |v1| = %lf" % linalg.norm(v1))
        print("  <v0,v1> = %lf" % matrixmultiply(v0,v1))

        out = []
        d0 = Dx/(2*step)
        d1 = Dy/(2*step)

        print("Start x = ", x0)
        for i in range(step, -1*(step+1), -1):
            out.append([])
            for j in range(-1*step, step+1):
                x = x0 + j*d0*v0 + i*d1*v1
                print("(%d, %d) -- (y, theta, a) = (%lf, %lf, %lf)" % (i, j, x[0], x[1], x[2]))

                ix = x.copy()
                v = v0.copy()

                k, converged, problem, diag = self.Corrector(x, v)

                print('x = ', x)
                if not converged:
                    if len(diag.cond) > 0:
                        print("  Avg. cond. # = %lf" % (sum(diag.cond)/len(diag.cond)))
                    if problem:
                        # Did not converge due to failure in backward integration
                        print("  Did not converge (XB).")
                        out[step-i].append(('XB', ix, x, diag))
                    elif k >= self.MaxCorrIters:
                        Fnm_flag = monotone([nm[0] for nm in diag.nrm], -3,
                                            direc=-1)
                        Vnm_flag = monotone([nm[1] for nm in diag.nrm], -3,
                                            direc=-1)
                        cond_flag = monotone(diag.cond, -3, direc=-1)
                        if Fnm_flag and Vnm_flag and cond_flag:
                            # Did not converge but may have converged with more time steps
                            print("  Did not converge (XC).")
                            out[step-i].append(('XC', ix, x, diag))
                        else:
                            # Did not converge for unknown reason
                            print("  Did not converge (X).")
                            out[step-i].append(('X', ix, x, diag))
                else:
                    print("  Converged.  Avg. cond. # = %lf" % (sum(diag.cond)/len(diag.cond)))
                    # Converged
                    out[step-i].append(('C', ix, x, diag))

                self._userdata.problem = False

        grid = args()
        grid.v0 = v0
        grid.v1 = v1
        grid.x0 = x0
        grid.d0 = d0
        grid.d1 = d1
        grid.out = out

        return grid


    def _curveToPointset(self):
        # Saves curve
        totlen = len(self.varslist + self.freepars + self.auxpars)
        if totlen != self.curve.shape[1]:
            # Curve of cycles
            coordnames = []
            for smtype in self.SolutionMeasures:
                coordnames += [v+'_'+smtype for v in self.varslist]
            coordnames += self.freepars + self.auxpars
            totlen = len(self.SolutionMeasures)*len(self.varslist) + \
                     len(self.freepars + self.auxpars)
        else:
            coordnames = self.varslist + self.freepars + self.auxpars
        coordarray = transpose(self.curve[:self.loc+1,0:totlen]).copy()
        sol = Pointset({'coordarray': coordarray,
                        'coordnames': coordnames,
                        'name': self.name})

        # Saves bifurcation info
        sol.labels = self.CurveInfo

        return sol


    def getSpecialPoint(self, label1, label2=None):
        """Gets a point on the curve with name specified by label1 and
        label2.

           Inputs:

               label1 -- string
               label2 -- string

           Output:

               x -- Point with specified name (type Point)

           If label2 is None, then label1 needs to be the name of the
           point.  In this case, the point type should be apparent
           from the name (i.e. by stripping off digits from the
           right).

           If label2 is not None, then label1 should be the point type
           and label2 the point name.

           For example, the following two function calls are
           equivalent:

               getSpecialPoint('LP3')
               getSpecialPoint('LP','LP3')
        """
        if label2 is None:
            label2 = label1
            label1 = label2.strip('0123456789')
        if self.sol is not None:
            ixs = self.sol.labels[label1]
            l = []
            for ix in ixs:
                pt = self.sol[ix]
                if pt.labels[label1]['name'] == label2:
                    l.append(pt)
#            l = [pt for pt in self.sol if pt.labels.has_key(label1) and \
#                 pt.labels[label1].has_key('name') and \
#                 pt.labels[label1]['name'] == label2]
        else:
            raise PyDSTool_ValueError('Empty curve.  Run forward() or backward().')
        if l == []:
            return None
        elif len(l) > 1:
            raise KeyError('Ambiguous point name: More than one point with that name exists.')
        else:
            return l[0]


    def computeEigen(self):
        self.update(args(SaveEigen = True))
        ptlabel = self.curvetype.split('-')[0]
        for loc, x in enumerate(self.sol):
            jac = self.sysfunc.jac(x.toarray())
            jacx = jac[:,self.coords[0]:(self.coords[-1]+1)]
            jacp = jac[:,self.params[0]:(self.params[-1]+1)]
            w, vr = linalg.eig(jacx)
            self.sol[loc].labels[ptlabel]['data'].evals = w
            self.sol[loc].labels[ptlabel]['data'].evecs = vr

            if ptlabel == 'FP':
                inside = [abs(eig) < 1-1e-6 for eig in w]
                outside = [abs(eig) > 1+1e-6 for eig in w]
                if all(inside):
                    self.sol[loc].labels[ptlabel]['stab'] = 'S'
                elif all(outside):
                    self.sol[loc].labels[ptlabel]['stab'] = 'U'
                else:
                    self.sol[loc].labels[ptlabel]['stab'] = 'N'
            else:
                realpos = [real(eig) > 1e-6 for eig in w]
                realneg = [real(eig) < -1e-6 for eig in w]
                if all(realneg):
                    self.sol[loc].labels[ptlabel]['stab'] = 'S'
                elif all(realpos):
                    self.sol[loc].labels[ptlabel]['stab'] = 'U'
                else:
                    self.sol[loc].labels[ptlabel]['stab'] = 'N'


    def cleanLabels(self):
        for pttype in self.LocBifPoints+other_special_points:
            if pttype not in ['P', 'MX']:
                if self.sol.bylabel(pttype) is not None:
                    num = 1;
                    for pt in self.sol.bylabel(pttype):
                        pt.labels[pttype]['name'] = pttype + repr(num)
                        if 'cycle' in pt.labels[pttype]:
                            pt.labels[pttype]['cycle'].name = pttype + repr(num)
                        num += 1


    def info(self):
        print(self.__repr__())
        print("Using model: %s\n"%self.model.name)
        if self.description is not 'None':
            print('Description')
            print('----------- \n')
            print(self.description, '\n')
        print('Model Info')
        print('---------- \n')
        print("  Variables : %s"%', '.join(self.varslist))
        print("  Parameters: %s\n"%', '.join(list(self.parsdict.keys())))
        print('Continuation Parameters')
        print('----------------------- \n')
        args_list = cont_args_list[:]
        exclude = ['description']
        args_list.insert(2, 'auxpars')
        for arg in args_list:
            if hasattr(self, arg) and arg not in exclude:
                print(arg, ' = ', eval('self.' + arg))
        print('\n')

        spts = ''
        if self.sol is not None:
            for pttype in all_point_types:
                if self.sol.bylabel(pttype) is not None:
                    for pt in self.sol.bylabel(pttype):
                        if 'name' in pt.labels[pttype]:
                            spts = spts + pt.labels[pttype]['name'] + ', '
        print('Special Points')
        print('-------------- \n')
        print(spts[:-2])


    def __repr__(self):
        return 'PyCont curve %s (type %s)'%(self.name,
                                        self.curvetype)

    __str__ = __repr__



class EquilibriumCurve(Continuation):
    """Child of Continuation class that represents curves of
    equilibrium points.

    System:

            h(x,a) = f(x,a) = 0

        The continuation variables are (x,a) with

            x = State variables
            a = Free parameters

    Detection of points: LP, H, BP"""
    def __init__(self, model, gen, automod, plot, args=None):
        args['auxpars'] = []
        Continuation.__init__(self, model, gen, automod, plot, args=args)

        self.CorrFunc = self.sysfunc


    def reset(self, args=None):
        """Resets curve by setting default parameters and deleting solution curve."""
        self.SPOut = None
        Continuation.reset(self, args)


    def update(self, args):
        """Update parameters for EquilibriumCurve."""
        Continuation.update(self, args)

        if args is not None:
            for k, v in args.items():
                if k in equilibrium_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = equilibrium_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = equilibrium_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype not in cont_bif_points and \
                               bftype not in equilibrium_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve EP-C.")
                            elif bftype == 'H' and self.varsdim == 1:
                                if self.verbosity >= 1:
                                    print("Warning: Variable dimension must be larger than 1 to detect Hopf points.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))


    def _createTestFuncs(self):
        """Creates processors and test functions for EquilibriumCurve
        class.

        Note:  In the following list, processors are in
        PyCont.Bifpoint and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        LP (FoldPoint): Fold_Tan
                        Fold_Det
                        Fold_Bor
        H (HopfPoint): Hopf_Det
                       Hopf_Bor
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in equilibrium_bif_points:
                stop = bftype in self.StopAtPoints  # Set stopping flag
                if bftype is 'BP':
                    method = Branch_Det(self.CorrFunc, self, save=True,
                                        numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)
                    self.BifPoints['BP'] = BranchPoint(method, iszero,
                                                       stop=stop)
                elif bftype is 'LP':
                    #method1 = Fold_Bor(self.CorrFunc, self, corr=False,
                    #                  save=True, numpoints=self.MaxNumPoints+1)
                    #method1 = Fold_Det(self.CorrFunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    method1 = Fold_Tan(self.CorrFunc, self, save=True,
                                       numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method1)
                    if 'BP' not in self.BifPoints.keys():
                        method2 = Branch_Det(self.CorrFunc, self, save=True,
                                             numpoints=self.MaxNumPoints+1)
                        self.TestFuncs.append(method2)
                    else:
                        method2 = self.BifPoints['BP'].testfuncs[0]

                    self.BifPoints['LP'] = FoldPoint([method1, method2],
                                                     [iszero, isnotzero],
                                                     stop=stop)
                elif bftype is 'H':
                    method = Hopf_Bor(self.CorrFunc, self, corr=False,
                                      save=True, numpoints=self.MaxNumPoints+1)
                    #method = Hopf_Det(self.CorrFunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    #method = Hopf_Eig(self.CorrFunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['H'] = HopfPoint(method, iszero, stop=stop)



##############################################
# Continuation of codimension-1 bifurcations #
##############################################


class FoldCurve(Continuation):
    """Child of Continuation class that represents curves of limit
    points.

    Augmented system h(x,a): Uses single bordering on the matrix A
    given by

            A = f_{x}(x,a)

        (see class PyCont.TestFunc.BorderMethod)
        (see class PyCont.TestFunc.Fold_Bor)

        The continuation variables are (x,a) with

            x = State variables
            a = Free parameters

        and the continuation curve is defined by

            h(x,a) = { f(x,a) = 0
                     { G(x,a) = 0

    Detection of points: BT, ZH, CP (BP points not currently
    implemented).
    """

    def __init__(self, model, gen, automod, plot, args=None):
        args['auxpars'] = []
        Continuation.__init__(self, model, gen, automod, plot, args=args)

        self.CorrFunc = AddTestFunction(self, Fold_Bor)
        self.CorrFunc.setdata(self.initpoint, None)


    def update(self, args):
        """Update parameters for FoldCurve."""
        Continuation.update(self, args)
        #if 'BP' in self.LocBifPoints:
        #    print "BP point detection not implemented: ignoring this type of point"
        #    self.LocBifPoints.remove('BP')
        #if 'BP' in self.StopAtPoints:
        #    print "BP point detection not implemented: ignoring this type of point"
        #    self.StopAtPoints.remove('BP')
        #self.LocBifPoints = []  # Temporary fix: Delete after branch point implementation for fold curve

        if args is not None:
            for k, v in args.items():
                if k in fold_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = fold_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = fold_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype not in cont_bif_points \
                               and bftype not in fold_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve LP-C.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and \
                                   bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + v)


    def _createTestFuncs(self):
        """Creates processors and test functions for FoldCurve class.

        Note:  In the following list, processors are in
        PyCont.Bifpoint and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        BT (BTPoint): BT_Fold
        ZH (ZHPoint): Hopf_Det (or Hopf_Bor), BT_Fold
        CP (CPPoint): CP_Fold
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in fold_bif_points:
                stop = bftype in self.StopAtPoints
                if bftype in ('BT','ZH'):
                    method1 = BT_Fold(self.CorrFunc, self, save=True,
                                      numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method1)
                    method2 = Hopf_Bor(self.CorrFunc.sysfunc, self,
                                       save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method2)
                    if bftype is 'BT':
                        self.BifPoints['BT'] = BTPoint([method1, method2],
                                                       [iszero, iszero], stop=stop)
                    else:
                        self.BifPoints['ZH'] = ZHPoint([method1, method2],
                                                       [isnotzero, iszero], stop=stop)
                elif bftype is 'CP':
                    method = CP_Fold(self.CorrFunc, self, save=True,
                                     numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)
                    self.BifPoints['CP'] = CPPoint(method, iszero, stop=stop)
                elif bftype is 'BP':
                    method1 = BP_Fold(self.CorrFunc, self, 0, save=True,
                                      numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method1)
                    method2 = BP_Fold(self.CorrFunc, self, 1, save=True,
                                      numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method2)
                    self.BifPoints['BP-P1'] = BranchPointFold(method1,
                                                        iszero, stop=stop)
                    self.BifPoints['BP-P2'] = BranchPointFold(method2,
                                                        iszero, stop=stop)



class HopfCurveOne(Continuation):
    """Child of Continuation class that represents curves of Hopf
    points.

    Augmented system h(x,a): Uses double bordering on the matrix A
    given by

            A = 2*f_{x}(x,a) (*) I

        where (*) denotes the bi-alternate matrix product.

        (see class PyCont.TestFunc.BiAltMethod)
        (see class PyCont.TestFunc.BorderMethod)
        (see class PyCont.TestFunc.Hopf_Double_Bor_One)

        The continuation variables are (x,a) with

            x = State variables
            a = Free parameters

        and the continuation curve is defined by

            h(x,a) = { f(x,a) = 0
                     { det(G) = 0

    Detection of points: BT, ZH, GH, DH
    """

    def __init__(self, model, gen, automod, plot, _args=None):
        _args['auxpars'] = []
        Continuation.__init__(self, model, gen, automod, plot, args=_args)

        self.CorrFunc = AddTestFunction(self, Hopf_Double_Bor_One)
        self.CorrFunc.setdata(self.initpoint, None)
        self.preTF = self._bialttoeig
        self.TFdata = args()


    def update(self, args):
        """Update parameters for HopfCurveOne."""
        Continuation.update(self, args)
        if 'BP' in self.LocBifPoints:
            self.LocBifPoints.remove('BP')
        if 'BP' in self.StopAtPoints:
            self.StopAtPoints.remove('BP')
        #self.LocBifPoints = []  # Temporary fix: Delete after branch point implementation for hopf curve

        if args is not None:
            for k, v in args.items():
                if k in hopf_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = hopf_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = hopf_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype == 'BP' or bftype not in cont_bif_points \
                                             and bftype not in hopf_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve H-C1.")
                            elif bftype == 'DH' and self.varsdim <= 3:
                                if self.verbosity >= 1:
                                    print("Warning: Variable dimension must be larger than 3 to detect Double Hopf points.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))


    def _createTestFuncs(self):
        """Creates processors and test functions for HopfCurveOne class.

        Note:  In the following list, processors are in PyCont.Bifpoint
        and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------------

        BT (BTPoint): BT_Hopf_One
        DH (DHPoint): DH_Hopf
        ZH (ZHPoint): Fold_Det (or Fold_Bor)
        GH (GHPoint): GH_Hopf_One
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in hopf_bif_points:
                stop = bftype in self.StopAtPoints  # Set stopping flag
                if bftype is 'BT':
                    method = BT_Hopf_One(self.CorrFunc, self, save=True, \
                                         numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['BT'] = BTPoint(method, iszero, stop=stop)
                if bftype is 'DH':
                    method = DH_Hopf(self.CorrFunc, self, save=True, \
                                     numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['DH'] = DHPoint(method, iszero, stop=stop)
                elif bftype is 'ZH':
                    method = Fold_Det(self.CorrFunc.sysfunc, self, \
                                      save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['ZH'] = ZHPoint(method, iszero, stop=stop)
                elif bftype is 'GH':
                    method = GH_Hopf_One(self.CorrFunc, self, save=True, \
                                         numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['GH'] = GHPoint(method, iszero, stop=stop)


    def _bialttoeig(self, X, V):
        q = self.CorrFunc.testfunc.data.c
        p = self.CorrFunc.testfunc.data.b
        n = self.sysfunc.m
        A = self.sysfunc.J_coords

        v1, v2 = invwedge(q, n)
        w1, w2 = invwedge(p, n)

        A11 = bilinearform(A,v1,v1)
        A22 = bilinearform(A,v2,v2)
        A12 = bilinearform(A,v1,v2)
        A21 = bilinearform(A,v2,v1)
        v11 = matrixmultiply(transpose(v1),v1)
        v22 = matrixmultiply(transpose(v2),v2)
        v12 = matrixmultiply(transpose(v1),v2)
        D = v11*v22 - v12*v12
        k = (A11*A22 - A12*A21)/D

        self.TFdata.k = k[0][0]
        self.TFdata.v1 = v1
        self.TFdata.w1 = w1




class HopfCurveTwo(Continuation):
    """Child of Continuation class that represents curves of Hopf points.

    Augmented system h(x,a): Uses double bordering on the matrix A given by

            A = f_{x}^{2}(x,a) + _k*I_{n}

        (see class PyCont.TestFunc.BorderMethod)
        (see class PyCont.TestFunc.Hopf_Double_Bor_Two)

        The continuation variables are (x,a,_k) with

            x = State variables
            a = Free parameters
            _k = w^2 = Square of pure imaginary eigenvalue associated with Hopf point
                (auxiliary parameter)

        and the continuation curve is defined by

                        { f(x,a) = 0
            h(x,a,_k) = { G_{1,1}(x,a,_k) = 0
                        { G_{2,2}(x,a,_k) = 0

    Detection of points: BT, ZH, GH
    """
    def __init__(self, model, gen, automod, plot, args=None):
        args['auxpars'] = ['_k']
        args['initpoint']['_k'] = 0
        Continuation.__init__(self, model, gen, automod, plot, args=args)

        self.CorrFunc = AddTestFunction(self, Hopf_Double_Bor_Two)

        # Get eigenvalue information to update initpoint
        J_coords = self.sysfunc.jac(self.initpoint, self.coords)
        eigs, LV, RV = linalg.eig(J_coords,left=1,right=1)
        k = pow(imag(eigs[argsort(abs(real(eigs)))[0]]),2)
        self.initpoint[-1] = k

        self.CorrFunc.setdata(self.initpoint, None)


    def update(self, args):
        """Update parameters for HopfCurveTwo."""
        Continuation.update(self, args)
        if 'BP' in self.LocBifPoints:
            self.LocBifPoints.remove('BP')
        if 'BP' in self.StopAtPoints:
            self.StopAtPoints.remove('BP')
        #self.LocBifPoints = []  # Temporary fix: Delete after branch point implementation for hopf curve

        if args is not None:
            for k, v in args.items():
                if k in hopf_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = hopf_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = hopf_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype in ['BP', 'DH'] or bftype not in cont_bif_points and bftype not in hopf_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve H-C2.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))


    def _createTestFuncs(self):
        """Creates processors and test functions for HopfCurveTwo class.

        Note:  In the following list, processors are in PyCont.Bifpoint
        and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------------

        BT (BTPoint): BT_Hopf
        ZH (ZHPoint): Fold_Det (or Fold_Bor)
        GH (GHPoint): GH_Hopf
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in hopf_bif_points:
                stop = bftype in self.StopAtPoints  # Set stopping flag
                if bftype is 'BT':
                    method = BT_Hopf(self.CorrFunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['BT'] = BTPoint(method, iszero, stop=stop)
                elif bftype is 'GH':
                    method = GH_Hopf(self.CorrFunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['GH'] = GHPoint(method, iszero, stop=stop)
                elif bftype is 'ZH':
                    method = Fold_Det(self.CorrFunc.sysfunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['ZH'] = ZHPoint(method, iszero, stop=stop)




class FixedPointCurve(Continuation):
    def __init__(self, model, gen, automod, plot, args=None):
        args['auxpars'] = []
        self.period = 1
        Continuation.__init__(self, model, gen, automod, plot, args=args)

        self._sysfunc = self.sysfunc
        self.sysfunc = DiscreteMap(self._sysfunc, self.period)

        self.CorrFunc = FixedPointMap(self.sysfunc)
        self.CorrFunc.coords = self.sysfunc.coords = self.coords
        self.CorrFunc.params = self.sysfunc.params = self.params


    def _preTestFunc(self, X, V):
        """Need CorrFunc jacobian for BP detection.  Could compute jacobian of CorrFunc,
        since it is a FixedPointMap, but this is a waste of energy as sysfunc jacobian
        was already computed in Continuation._preTestFunc."""

        Continuation._preTestFunc(self, X, V)	# Compute jacobian of system function
        self.CorrFunc.J_coords = self.sysfunc.J_coords - eye(self.varsdim,self.varsdim)
        self.CorrFunc.J_params = self.sysfunc.J_params


    def _createTestFuncs(self):
        """Creates processors and test functions for EquilibriumCurve class.

        Note:  In the following list, processors are in PyCont.Bifpoint
        and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        LPC (FoldPoint): Fold_Tan
                         Fold_Det
                         Fold_Bor
        H (HopfPoint): Hopf_Det
                       Hopf_Bor
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in fixedpoint_bif_points:
                stop = bftype in self.StopAtPoints
                if bftype is 'LPC':
                    method1 = Fold_Tan(self.CorrFunc, self, save=True,
                                      numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method1)
                    if 'BP' not in self.BifPoints.keys():
                        method2 = Branch_Det(self.CorrFunc, self, save=True, numpoints=self.MaxNumPoints+1)
                        self.TestFuncs.append(method2)
                    else:
                        method2 = self.BifPoints['BP'].testfuncs[0]

                    self.BifPoints['LPC'] = LPCPoint([method1, method2], [iszero, isnotzero], stop=stop)
                elif bftype is 'PD':
                    method = PD_Det(self.sysfunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['PD'] = PDPoint(method, iszero, stop=stop)
                elif bftype is 'NS':
                    method = NS_Det(self.sysfunc, self, save=True, numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['NS'] = NSPoint(method, iszero, stop=stop)
                elif bftype is 'BP':
                    if 'LPC' not in self.BifPoints.keys():
                        method = Branch_Det(self.CorrFunc, self, save=True, numpoints=self.MaxNumPoints+1)
                        self.TestFuncs.append(method)
                    else:
                        method = self.BifPoints['LPC'].testfuncs[1]
                    self.BifPoints['BP'] = BranchPoint(method, iszero, stop=stop)



    def update(self, args):
        """Update parameters for FixedPointCurve."""
        Continuation.update(self, args)

        if args is not None:
            for k, v in args.items():
                if k in fixedpoint_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = fixedpoint_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = fixedpoint_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype not in cont_bif_points and bftype not in fixedpoint_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve FP-C.")
                            elif bftype == 'NS' and self.varsdim == 1:
                                if self.verbosity >= 1:
                                    print("Warning: Variable dimension must be larger than 1 to detect NS points.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))



class FixedPointFoldCurve(FixedPointCurve):
    """Child of Continuation class that represents curves of limit
    points.

    Augmented system h(x,a): Uses single bordering on the matrix A
    given by

            A = f_{x}(x,a)

        (see class PyCont.TestFunc.BorderMethod)
        (see class PyCont.TestFunc.Fold_Bor)

        The continuation variables are (x,a) with

            x = State variables
            a = Free parameters

        and the continuation curve is defined by

            h(x,a) = { f(x,a) = 0
                     { G(x,a) = 0

    Detection of points: BT, ZH, CP (BP points not currently
    implemented).
    """

    def __init__(self, model, gen, automod, plot, args=None):
        FixedPointCurve.__init__(self, model, gen, automod, plot, args=args)

        self.CorrFunc = AddTestFunction_FixedPoint(self, LPC_Bor)
        self.CorrFunc.setdata(self.initpoint, None)

    def _preTestFunc(self, X, V):
        """Need CorrFunc jacobian for BP detection."""
        Continuation._preTestFunc(self, X, V)
        self.CorrFunc.J_coords = self.sysfunc.J_coords - eye(self.varsdim,self.varsdim)
        self.CorrFunc.J_params = self.sysfunc.J_params

    def _createTestFuncs(self):
        """Creates processors and test functions for FoldCurve class.

        Note:  In the following list, processors are in
        PyCont.Bifpoint and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        BT (BTPoint): BT_Fold
        ZH (ZHPoint): Hopf_Det (or Hopf_Bor), BT_Fold
        CP (CPPoint): CP_Fold
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in fold_map_bif_points:
                stop = bftype in self.StopAtPoints
                if bftype is 'CP':
                    method = CP_Fold(self.CorrFunc, self, save=True,
                                     numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)

                    self.BifPoints['CP'] = CPPoint(method, iszero, stop=stop)

    def update(self, args):
        """Update parameters for FixedPointCurve."""
        Continuation.update(self, args)

        if args is not None:
            for k, v in args.items():
                if k in fold_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = fold_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = fold_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype not in cont_bif_points and bftype not in fold_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve FP-C.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))

class FixedPointFlipCurve(FixedPointCurve):
    """Child of Continuation class that represents curves of limit
    points.

    Augmented system h(x,a): Uses single bordering on the matrix A
    given by

            A = f_{x}(x,a)

        (see class PyCont.TestFunc.BorderMethod)
        (see class PyCont.TestFunc.Fold_Bor)

        The continuation variables are (x,a) with

            x = State variables
            a = Free parameters

        and the continuation curve is defined by

            h(x,a) = { f(x,a) = 0
                     { G(x,a) = 0

    Detection of points: BT, ZH, CP (BP points not currently
    implemented).
    """

    def __init__(self, model, gen, automod, plot, args=None):
        FixedPointCurve.__init__(self, model, gen, automod, plot, args=args)

        self.CorrFunc = AddTestFunction_FixedPoint(self, PD_Bor)
        self.CorrFunc.setdata(self.initpoint, None)

    def _preTestFunc(self, X, V):
        """Need CorrFunc jacobian for BP detection."""
        Continuation._preTestFunc(self, X, V)
        self.CorrFunc.J_coords = self.sysfunc.J_coords - eye(self.varsdim,self.varsdim)
        self.CorrFunc.J_params = self.sysfunc.J_params

    def _createTestFuncs(self):
        """Creates processors and test functions for FixedPointFoldCurve class.

        Note:  In the following list, processors are in
        PyCont.Bifpoint and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        BT (BTPoint): BT_Fold
        ZH (ZHPoint): Hopf_Det (or Hopf_Bor), BT_Fold
        CP (CPPoint): CP_Fold
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in flip_bif_points:
                stop = bftype in self.StopAtPoints

    def update(self, args):
        """Update parameters for FixedPointCurve."""
        Continuation.update(self, args)

        if args is not None:
            for k, v in args.items():
                if k in flip_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = flip_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = flip_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype not in cont_bif_points and bftype not in flip_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve FP-C.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))

class FixedPointNSCurve(FixedPointCurve):
    """Child of Continuation class that represents curves of limit
    points.

    Augmented system h(x,a): Uses single bordering on the matrix A
    given by

            A = f_{x}(x,a)

        (see class PyCont.TestFunc.BorderMethod)
        (see class PyCont.TestFunc.Fold_Bor)

        The continuation variables are (x,a) with

            x = State variables
            a = Free parameters

        and the continuation curve is defined by

            h(x,a) = { f(x,a) = 0
                     { G(x,a) = 0

    Detection of points: BT, ZH, CP (BP points not currently
    implemented).
    """

    def __init__(self, model, gen, automod, plot, args=None):
        FixedPointCurve.__init__(self, model, gen, automod, plot, args=args)

        self.CorrFunc = AddTestFunction_FixedPoint(self, NS_Det)
        self.CorrFunc.setdata(self.initpoint, None)

    def _preTestFunc(self, X, V):
        """Need CorrFunc jacobian for BP detection."""
        Continuation._preTestFunc(self, X, V)
        self.CorrFunc.J_coords = self.sysfunc.J_coords - eye(self.varsdim,self.varsdim)
        self.CorrFunc.J_params = self.sysfunc.J_params

    def _createTestFuncs(self):
        """Creates processors and test functions for FixedPointFoldCurve class.

        Note:  In the following list, processors are in
        PyCont.Bifpoint and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        BT (BTPoint): BT_Fold
        ZH (ZHPoint): Hopf_Det (or Hopf_Bor), BT_Fold
        CP (CPPoint): CP_Fold
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in NS_bif_points:
                stop = bftype in self.StopAtPoints

    def update(self, args):
        """Update parameters for FixedPointCurve."""
        Continuation.update(self, args)

        if args is not None:
            for k, v in args.items():
                if k in NS_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = NS_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = NS_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype not in cont_bif_points and bftype not in NS_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve FP-C.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))

class FixedPointCuspCurve(FixedPointCurve):
    """Child of Continuation class that represents curves of limit
    points.

    Augmented system h(x,a): Uses single bordering on the matrix A
    given by

            A = f_{x}(x,a)

        (see class PyCont.TestFunc.BorderMethod)
        (see class PyCont.TestFunc.Fold_Bor)

        The continuation variables are (x,a) with

            x = State variables
            a = Free parameters

        and the continuation curve is defined by

            h(x,a) = { f(x,a) = 0
                     { G(x,a) = 0

    Detection of points: BT, ZH, CP (BP points not currently
    implemented).
    """

    def __init__(self, model, gen, automod, plot, args=None):
        FixedPointCurve.__init__(self, model, gen, automod, plot, args=args)

        self.CorrFunc = AddTestFunction_FixedPoint_Mult(self, [LPC_Bor, CP_Fold2])
        self.CorrFunc.setdata(self.initpoint, None)

    def _preTestFunc(self, X, V):
        """Need CorrFunc jacobian for BP detection."""
        Continuation._preTestFunc(self, X, V)
        self.CorrFunc.J_coords = self.sysfunc.J_coords - eye(self.varsdim,self.varsdim)
        self.CorrFunc.J_params = self.sysfunc.J_params

    def _createTestFuncs(self):
        """Creates processors and test functions for FoldCurve class.

        Note:  In the following list, processors are in
        PyCont.Bifpoint and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        BT (BTPoint): BT_Fold
        ZH (ZHPoint): Hopf_Det (or Hopf_Bor), BT_Fold
        CP (CPPoint): CP_Fold
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            if bftype in cusp_bif_points:
                stop = bftype in self.StopAtPoints
                if bftype is 'BP':
                    method = Branch_Det(self.CorrFunc, self, save=True,
                                        numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.remove(self.TestFuncs[-1])
                    self.TestFuncs.append(method)

                    self.BifPoints['BP'] = BranchPoint(method, iszero,
                                                       stop=stop)


    def update(self, args):
        """Update parameters for FixedPointCurve."""
        Continuation.update(self, args)

        if args is not None:
            for k, v in args.items():
                if k in fold_args_list:
                    if k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = fold_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = fold_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype not in cont_bif_points and bftype not in fold_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of " + bftype + " points not implemented for curve FP-C.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)

                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))


class LimitCycleCurve(Continuation):
    """Wrapper for auto limit cycle computations.

    Reimplement these Continuation methods:
        _compute
        _curveToPointset
    """
    def __init__(self, model, gen, automod, plot, args=None):
        # Initialize initpoint
        args['auxpars'] = ['_T']
        args['initpoint']['_T'] = 0

        Continuation.__init__(self, model, gen, automod, plot, args=args)

        self.UseAuto = True
        self._AdaptCycle = False

        self.CorrFunc = self.sysfunc

        if not hasattr(self, "NumSPOut"):
            self.NumSPOut = self.MaxNumPoints

        if not hasattr(self, "SPOut"):
            self.SPOut = None


    def reset(self, args=None):
        """Resets curve by setting default parameters and deleting solution curve."""
        self.NumCollocation = 4
        self.NumIntervals = 50
        self.AdaptMesh = 3
        self.DiagVerbosity = 2
        self.SolutionMeasures = ['max','min']
        self.SaveFlow = False
        self.SPOut = None
        Continuation.reset(self, args)


    def update(self, args):
        """Update parameters for LimitCycleCurve."""
        Continuation.update(self, args)
        if 'BP' in self.LocBifPoints:
            self.LocBifPoints.remove('BP')
        if 'BP' in self.StopAtPoints:
            self.StopAtPoints.remove('BP')
        #self.LocBifPoints = []  # Temporary fix: Delete after branch fix for LimitCycleCurve

        if args is not None:
            for k, v in args.items():
                if k in limitcycle_args_list:
                    if k == 'SolutionMeasures':
                        self.SolutionMeasures = ['max', 'min']
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = list(solution_measures.keys())
                            else:
                                v = [v]

                        for smtype in v:
                            if smtype not in solution_measures.keys():
                                if self.verbosity >= 1:
                                    print("Warning: Solution measure " + smtype + " is not valid.")
                            elif smtype not in self.SolutionMeasures:
                                self.SolutionMeasures.append(smtype)

                        # Order SolutionMeasures based on solution_measures
                        self.SolutionMeasures = [v for v in solution_measures_list \
                                                 if v in self.SolutionMeasures]
                    elif k == 'LocBifPoints':
                        if isinstance(v, str):
                            if v.lower() == 'all':
                                v = limitcycle_bif_points
                            else:
                                v = [v]

                        # Handle stopping points
                        w = []
                        if 'StopAtPoints' in args:
                            w = args['StopAtPoints']
                            if isinstance(w, str):
                                if w.lower() == 'all':
                                    w = limitcycle_bif_points
                                else:
                                    w = [w]

                        for bftype in v:
                            if bftype == 'BP' or bftype not in cont_bif_points \
                               and bftype not in limitcycle_bif_points:
                                if self.verbosity >= 1:
                                    print("Warning: Detection of %s"%bftype, end=' ')
                                    print(" points not implemented for curve LC-C.")
                            else:
                                if bftype not in self.LocBifPoints:
                                    self.LocBifPoints.append(bftype)
                                if bftype in w and bftype not in self.StopAtPoints:
                                    self.StopAtPoints.append(bftype)
                    elif k in ['NumCollocation', 'NumIntervals']:
                        self._AdaptCycle = True
                        exec('self.' + k + ' = ' + repr(v))
                    elif k != 'StopAtPoints':
                        exec('self.' + k + ' = ' + repr(v))


    def plot_cycles(self, coords=None, cycles=None, exclude=None,
                    figure=None, axes=None, normalized=False, tlim=None,
                    method=None, color_method='default', **plot_args):
        if not method or method == 'highlight':
            initializeDisplay(self.plot, figure=figure, axes=axes)
            cfl = self.plot._cfl
            cal = self.plot._cal

        # Plot all cycles if cycles is None
        if cycles is None:
            cycles = []
            for pt in self.sol:
                for ptdata in pt.labels.values():
                    if 'cycle' in ptdata:
                        cycles.append(ptdata['cycle'].name)
        elif not isinstance(cycles, list):
            cycles = [cycles]

        # Initialize exclude, if necessary
        if exclude is None:
            exclude = []
        elif not isinstance(exclude, list):
            exclude = [exclude]

        # Determine labeling information
        if 'label' not in plot_args:
            UseCycleName = True
        else:
            UseCycleName = False

        # Initialize and check coords
        if coords is None:
            coords = ('t', self.varslist[0])
        for n in range(2):
            if coords[n] not in ['t']+self.varslist:
                raise KeyError('Coordinate %s does not exist'%coords[n])

        # Get cycle pointsets from cycle names, if appropriate
        pts = []
        for cyclename in cycles:
            if isinstance(cyclename, str):
                if cyclename not in exclude:
                    pointtype = cyclename.strip('0123456789')
                    if pointtype not in limitcycle_bif_points \
                                      + other_special_points:
                        raise PyDSTool_TypeError('Wrong point type')
                    cycle = self.getSpecialPoint(pointtype, cyclename)
                    if cycle is None:
                        raise KeyError('Cycle %s does not exist'%cyclename)
                    pts.append(cycle.labels[pointtype]['cycle'])
            elif isinstance(cyclename, Pointset):
                pts.append(cyclename)
            else:
                print('Point must be type(str) or type(Pointset).')
        cycles = pts

        # Get maximal period
        if coords[0] == 't':
            if tlim is not None:
                if isinstance(tlim, str) and tlim[-1] == 'T' \
                       and isinstance(eval(tlim.rstrip('T')), int):
                    tmult = eval(tlim.rstrip('T'))
                else:
                    raise PyDSTool_TypeError("tlim must be a string of the form '#T'")
            else:
                tmult = 1

            if normalized:
                t1 = tmult
            else:
                t1 = 0
                for cycle in cycles:
                    #print cycle.name
                    t1 = cycle['t'][-1] > t1 and cycle['t'][-1] or t1
                t1 *= tmult

        # Check method input
        if method is not None:
            if method == 'stack' or isinstance(method, tuple) \
                      and method[0] == 'stack':
                cycperaxes = len(cycles)
                if isinstance(method, tuple):
                    if len(method) == 1 or len(method) > 2:
                        raise TypeError('Method should be a tuple of length 2')
                    if not isinstance(method[1], int):
                        raise TypeError('Need integer number of cycles')
                    if method[1] < 1 or method[1] > 10:
                        raise TypeError('Number of cycles per axes is limited'
                                        ' between 1 and 10')
                    cycperaxes = method[1]
                    method = 'stack'

                # Check axes input
                if axes is None:
                    axes = (1,1,[1])
                elif isinstance(axes, plt.Axes):
                    raise TypeError('Axes must be a tuple or None')
                elif isinstance(axes[2], int):
                    axes = list(axes)
                    axes[2] = [axes[2]]

                # Initialize figure
                initializeDisplay(self.plot, figure=figure, \
                                  axes=(axes[0], axes[1], axes[2][0]))
                figure = plt.gcf()
            elif method != 'highlight':
                raise NotImplementedError('Requested method is not implemented')

        # Now go through cycles and plot
        for cyct, cycle in enumerate(cycles):

            # Set up axes for 'stack' method
            if method == 'stack':
                # Get axes number and cycle number
                axnum = int(cyct/cycperaxes)
                cycnum = cyct % cycperaxes

                # Handle when cycnum is 0 (i.e. new axes)
                if cycnum == 0:
                    if axnum < len(axes[2]):
                        axbox = plt.subplot(axes[0], axes[1], axes[2][axnum])
                    else:
                        figure = plt.figure()
                        axbox = plt.subplot(1,1,1)
                    initializeDisplay(self.plot, figure=figure, axes=axbox)
                    cfl = self.plot._cfl
                    cal = self.plot._cal
                    cyclenames = []

                # Add cycle name to list for y labels
                cyclenames.append(cycle.name)

            # Initialize curve
            disp_args = copy(plot_args)
            if coords[0] == 't':
                if normalized:
                    numcyc = t1
                    numpts = t1*len(cycle) - numcyc + 1
                else:
                    #numcyc = int((t1/cycle['t'][-1]))
                    #numpts = int((t1/cycle['t'][-1])*len(cycle)) - numcyc + 1
                    numcyc = int((t1/cycle['t'][-1]))+1
                    numpts = numcyc*len(cycle) - numcyc + 1
                X = zeros((2,numpts), float)
                T = normalized and cycle['t'][-1] or 1.

                for n in range(numcyc):
                    X[0][(n*len(cycle)-n):((n+1)*len(cycle)-n)] = \
                                    (cycle['t'] + n*cycle['t'][-1])/T
                    X[1][(n*len(cycle)-n):((n+1)*len(cycle)-n)] = \
                                    cycle[coords[1]]
                #X[0][(numcyc*len(cycle)-numcyc):numpts] = (numcyc*cycle['t'][-1] + cycle['t'][0:(numpts-numcyc*(len(cycle)-1))])/T
                #X[1][(numcyc*len(cycle)-numcyc):numpts] = (cycle[coords[1]].toarray())[0:(numpts-numcyc*(len(cycle)-1))]
            else:
                X = zeros((2,len(cycle)), float)
                for n in range(2):
                    if coords[n] == 't':
                        X[n] = cycle['t']
                    else:
                        X[n] = cycle[coords[n]]

            # Modify curve for specified method, if necessary
            if method == 'stack':
                X[1] = (0.95/2)*X[1]/(max(abs(X[1]))*cycperaxes) + \
                       (1-(0.5+cycnum)/cycperaxes)

            # Prints curve
            self.plot[cfl][cal][cycle.name] = pargs()
            if UseCycleName:
                disp_args['label'] = cycle.name
            if color_method == 'bytype' and 'c' not in disp_args and \
                                        'color' not in disp_args:
                disp_args['color'] = \
                    bif_point_colors[cycle.name.strip('0123456789')][-1]
            self.plot[cfl][cal][cycle.name].cycle = \
                    plt.plot(X[0], X[1], **disp_args)

            # Last curve of axes was drawn, so clean up axes
            if (cyct == len(cycles)-1) or (method=='stack' and \
                                           cycnum == cycperaxes-1):
                # Handle x labels and limits
                plt.xlabel(coords[0])
                if coords[0] == 't':
                    plt.xlim([0, t1])
                    if normalized:
                        plt.xlabel('Period')

                # Handle y labels and limits
                if not method or method == 'highlight':
                    plt.ylabel(coords[1])
                else:
                    yticklocs = (1-(0.5+arange(cycperaxes))/cycperaxes)
                    plt.yticks(yticklocs, cyclenames)
                    plt.ylim([0, 1])

                self.plot[cfl][cal].refresh()

        # Final cleanup
        if method == 'highlight':
            KeyEvent(self.plot[cfl][cal])


    def _savePointInfo(self, ds=None, evals=None, jacx=None, jacp=None,
                       flow=None, diag=None):
        """Created a function for this since it needs to be called
        both in _compute and when a bifurcation point is found.  It
        will have conditional statements for saving of Jacobian and
        eigenvalues, as well as other possible tidbits of information.
        """
        ptlabel = 'LC'
        for ind in range(self.loc+1):
            self.CurveInfo[ind] = (ptlabel, {'data': args()})

            if ds is not None:
                self.CurveInfo[ind][ptlabel]['data'].ds = ds[ind]

            # Save jacobian information
            if evals is not None:
                self.CurveInfo[ind][ptlabel]['data'].evals = evals[ind]

                if isnan(evals[ind][0]) or abs(evals[ind][0] - 1.) > 0.05:
                    # 0.05 is same tolerance used by AUTO
                    self.CurveInfo[ind][ptlabel]['stab'] = 'X'
                else:
                    inside = [abs(eig) < 1. for eig in evals[ind][1:]]
                    outside = [abs(eig) > 1. for eig in evals[ind][1:]]
                    if all(inside):
                        self.CurveInfo[ind][ptlabel]['stab'] = 'S'
                    elif all(outside):
                        self.CurveInfo[ind][ptlabel]['stab'] = 'U'
                    elif any(isnan(inside)):
                        self.CurveInfo[ind][ptlabel]['stab'] = 'X'
                    else:
                        self.CurveInfo[ind][ptlabel]['stab'] = 'N'

            if jacx is not None:
                self.CurveInfo[ind][ptlabel]['data'].jac0 = \
                    reshape(jacx[0][ind], (self.varsdim, self.varsdim))
                self.CurveInfo[ind][ptlabel]['data'].jac1 = \
                    reshape(jacx[1][ind], (self.varsdim, self.varsdim))

            if diag is not None:
                self.CurveInfo[ind][ptlabel]['data'].nit = diag[ind][0]


    def _compute(self, x0=None, v0=None, direc=1):
        """NOTE: This doesn't use v0!!!! It gets v0 from c0, which is found from x0.
        This makes sense since there can be no v0 without a c0 (v0 doesn't make sense
        for a Hopf point)."""

        ########################
        # Initialize auto module
        ########################
        if not self._autoMod.Initialize():
            raise InitError("Initialization of auto module failed...")

        #########################
        # Set data in auto module
        #########################

        # Setup index to free parameters (icp in auto)
        # BE CAREFUL OF PERIOD IN 11th SLOT!!
        freepars = []
        for P in self.freepars:
            ind = self.gensys.funcspec.pars.index(P)
            if ind < 10:
                freepars.append(ind)
            else:
                freepars.append(ind+40)
        freepars.append(10) # Add period

        # Computation flags (special points and floguet multipliers)
        if 'PD' in self.LocBifPoints or 'NS' in self.LocBifPoints:
            isp = 2
        elif self.SaveEigen:
            isp = 1
        else:
            isp = 0

        if 'LPC' in self.LocBifPoints:
            ilp = 1
        else:
            ilp = 0

        # Save jacobian info
        if self.SaveJacobian:
            sjac = 1;
        else:
            sjac = 0;

        # Save flow info
        if self.SaveFlow:
            sflow = 1;
        else:
            sflow = 0;

        # Solution measures
        nsm = 1
        for smtype in self.SolutionMeasures:
            nsm += solution_measures[smtype]

        # CHECK ME (Moved from below)
        parsdim = len(self.parsdict)
        ipar = list(range(min(parsdim,10))) + [10] + list(range(50,parsdim+40))
        parkeys = sortedDictKeys(self.parsdict)

        # INSERT SPOut STUFF HERE
        if self.SPOut is None:
            nuzr = 0
            iuz = None
            vuz = None
        else:
            iuz = []
            vuz = []
            for k, v in self.SPOut.items():
                pind = ipar[parkeys.index(k)]
                iuz.extend(len(v)*[pind])
                vuz.extend(v)
            nuzr = len(iuz)

        self._autoMod.SetData(2,    # Problem type (ips)
                              ilp,    # No fold detection (ilp)
                              1,    # Branch switching (isw) (assuming no branch point or PD)
                              isp,    # Bifurcation point detection flag!!! (isp) 0 for now...
                              sjac,
                              sflow,
                              nsm,
                              self.MaxNumPoints,
                              self.varsdim,
                              self.NumIntervals,
                              self.NumCollocation,
                              self.AdaptMesh,
                              self.FuncTol,
                              self.VarTol,
                              self.TestTol,
                              self.MaxTestIters,
                              self.MaxCorrIters,
                              direc*self.StepSize,
                              self.MinStepSize,
                              self.MaxStepSize,
                              self.NumSPOut,
                              self.DiagVerbosity,
                              self.freeparsdim + self.auxparsdim,    # len(freepars)
                              freepars,
                              nuzr,
                              iuz,
                              vuz,
                              )

        #################################################
        # Set initial point in auto module (u, par, ipar)
        #################################################
        c0 = None
        T = None
        if x0 is None:
            x0 = self.initpoint
            c0 = self.initcycle

        if isinstance(x0, dict):
            # x0 is a dict on the first initial run (x0 has been parsed a little
            #   in ContClass.newCurve()

            # FIX ME
            x0n = x0.copy()
            for k, v in x0.items():
                kn = k
                for smtype in self.SolutionMeasures:
                    if k.rfind('_'+smtype) > 0:
                        kn = k[0:k.rfind('_'+smtype)]
                        break
                x0n[kn] = v
            x0 = x0n

            x0 = tocoords(self, x0)
        elif isinstance(x0, Point):
            # Check to see if point contains a cycle.  If it does, assume
            #   we are starting at a cycle and save it in initcycle
            for v in x0.labels.values():
                if 'cycle' in v:
                    c0 = v   # Dictionary w/ cycle, name, and tangent information
                    try:
                        T = x0['_T']
                    except:
                        T = None
            x0 = x0.todict()
            for p in self.freepars:
                if p not in x0.keys():
                    x0[p] = self.parsdict[p]

            # FIX ME
            x0n = x0.copy()
            for k, v in x0.items():
                kn = k
                for smtype in self.SolutionMeasures:
                    if k.rfind('_'+smtype) > 0:
                        kn = k[0:k.rfind('_'+smtype)]
                        break
                x0n[kn] = v
            x0 = x0n

            x0 = tocoords(self, x0)

        #u = x0.resize(x0.size())
        u = x0

        # Set up ipars and pars, being careful with period in the 11th index
        for i, par in enumerate(self.freepars):
            self.parsdict[par] = u[self.params[i]]
        par = sortedDictValues(self.parsdict)

        u = u.tolist()   # Make u a list
        u.insert(0, 0.0)  # Time is in the first position

        # Set Hopf/Cycle info in auto module
        # In particular, sets period (differently depending on Hopf vs. Cycle)
        self.CurveInfo[0] = ('P', {})
        if c0 is None:
            ups = None
            udotps = None
            rldot = None
            upslen = 0
            T = CheckHopf(self, x0)
        else:
            t = c0['cycle']['t'].copy()
            if T is None:
                T = t[-1] - t[0]
            t.resize((len(t),1))
            t = (t-t[0])/T

            v = transpose(c0['cycle'].toarray())
            ups = array(c_[t, v])
            upslen = len(ups)
            ups = resize(ups, ups.size)
            ups = ups.tolist()  # One dim list

            udotps = c0['data'].V['udotps']
            if udotps is not None:
                udotps = resize(udotps, udotps.size)
                udotps = udotps.tolist()    # One dim list

            rldot = c0['data'].V['rldot']
            if rldot is not None:
                rldot = c0['data'].V['rldot'].tolist()

        # Insert period T in par and u list.
        if T is not None:
            par.insert(10, T)
        else:
            if not self._autoMod.ClearAll():
                raise RuntimeError('Cleanup of auto module failed...')
            raise PyDSTool_ExistError("Period missing in starting point... aborting.")

        # Load data into auto module and call AUTO if successful
        #print ups
        if self._autoMod.SetInitPoint(u[0:self.varsdim+1], parsdim+1,
                                      ipar, par, freepars, upslen, ups,
                                      udotps, rldot, self._AdaptCycle):
            AutoOut = self._autoMod.Compute()

            ##############
            # Unpack stuff
            ##############

            # Save curve (all solution measures and parameters at end)
            #self.curve = c_[AutoOut[0], AutoOut[1]]
            #for i in range(len(self.SolutionMeasures)-1):
            #    self.curve = c_[self.curve, AutoOut[2+i]]
            num_u = self._autoMod.getSolutionNum()[0]
            VarOut = zeros((num_u,self.varsdim,2+int(log2(nsm))))
            self._autoMod.getSolutionVar(VarOut)
            ParOut = zeros((num_u,self.freeparsdim+self.auxparsdim))
            self._autoMod.getSolutionPar(ParOut)
            self.curve = array([])
            self.curve = c_[VarOut[:,:,0],VarOut[:,:,1]]
            for i in range(len(self.SolutionMeasures)-2):
                self.curve = c_[self.curve, VarOut[:,:,i]]
            self.curve = c_[self.curve, ParOut]

            # Find bad points on curve
            self.loc = len(self.curve)-1    # This might change based on bad points
            badpts = nonzero([any(isnan(pt)) for pt in self.curve])[0]
            if len(badpts) > 0:
                # This needs to be changed to the special point just previous to this
                endpt = badpts[0]-1
            else:
                endpt = self.loc

            # Save floquet multipliers
            evals = None
            if self.SaveEigen or 'PD' in self.LocBifPoints or \
               'NS' in self.LocBifPoints:
                self.SaveEigen = True
                EV = zeros((num_u,self.varsdim,2))
                self._autoMod.getFloquetMultipliers(EV)
                evals = EV[:,:,0] + 1j*EV[:,:,1]

            # Save Jacobian info
            jacx = None
            if self.SaveJacobian:
                JacOut = zeros((2,num_u,self.varsdim*self.varsdim))
                self._autoMod.getJacobians(JacOut)
                jacx = (JacOut[0,:,:],JacOut[1,:,:])

            # Save number of iterations
            nit = None
            if 1:
                nit = zeros((num_u,1),dtype=int32)
                self._autoMod.getNumIters(nit)

            # Insert cycle info and determine stopping point
            sp_endpt = 0
            sp_endpt_type = 'P'
            num_sp = self._autoMod.getSpecPtNum()[0]

            for i in range(num_sp):
                sp_dims = zeros((5,),dtype=int32)
                self._autoMod.getSpecPtDims(i, sp_dims)
                (sp_ind,sp_type,ntpl,nar,nfpr) = sp_dims

                sp_flags = zeros((4,),dtype=int32)
                self._autoMod.getSpecPtFlags(i, sp_flags)
                (ups_flag,udotps_flag,rldot_flag,flow_flag) = sp_flags

                if ups_flag:
                    sp_ups = zeros((ntpl,nar))
                    self._autoMod.getSpecPt_ups(i, sp_ups)
                else:
                    sp_ups = None

                if udotps_flag:
                    sp_udotps = zeros((ntpl,nar-1))
                    self._autoMod.getSpecPt_udotps(i,sp_udotps)
                else:
                    sp_udotps = None

                if rldot_flag:
                    sp_rldot = zeros((nfpr,))
                    self._autoMod.getSpecPt_rldot(i,sp_rldot)
                    sp_rldot = transpose(sp_rldot)
                else:
                    sp_rldot = None

                if flow_flag:
                    sp_flow1 = zeros((self.NumIntervals,nar-1,nar-1))
                    sp_flow2 = zeros((self.NumIntervals,nar-1,nar-1))
                    self._autoMod.getSpecPt_flow1(i,sp_flow1)
                    self._autoMod.getSpecPt_flow2(i,sp_flow2)
                    sp_flow = ()
                    for i in range(self.NumIntervals):
                        sp_flow = sp_flow + (sp_flow1[i,:,:],)
                        sp_flow = sp_flow + (sp_flow2[i,:,:],)
                else:
                    sp_flow = None

                if sp_ups is not None and sp_ind <= endpt:
                    pttype = auto_point_types[sp_type]

                    # Only save those that are asked for
                    # (auto calculates PD and NS together)
                    if pttype in self.LocBifPoints + other_special_points:
                        sp_endpt = sp_ind
                        sp_endpt_type = pttype

                        # Scale time to period
                        sp_ups[:,0] *= self.curve[sp_ind][-1]

                        # Change cycle to parametrized Pointset
                        # (MAY WANT TO GENERALIZE _curveToPointset!)
                        coordnames = self.varslist
                        coordarray = transpose(sp_ups[:,1:]).copy()
                        indepvarname = 't'
                        indepvararray = transpose(sp_ups[:,0]).copy()
                        sp_ups = Pointset({'coordarray': coordarray,
                                           'coordnames': coordnames,
                                           'indepvarname': indepvarname,
                                           'indepvararray': indepvararray})

                        self.CurveInfo[sp_ind] = (pttype, \
                            {'cycle': sp_ups, 'data': args(V = \
                                {'udotps': sp_udotps, 'rldot': sp_rldot})})

                        # Save flow
                        if sp_flow is not None:
                            self.CurveInfo[sp_ind][pttype]['flow'] = sp_flow

            if sp_endpt < self.loc:
                if self.verbosity > 0:
                    print('Warning: NaNs in solution from AUTO. ', end=' ')
                    print('Reduce stepsize and try again.')
                self.loc = sp_endpt
                self.curve = self.curve[0:self.loc+1]
                self.CurveInfo[self.loc] = ('MX', \
                                    self.CurveInfo[self.loc][sp_endpt_type])
                self.CurveInfo.remove(self.loc, sp_endpt_type)

            # Initialize labels
            self._savePointInfo(evals=evals, jacx=jacx, diag=nit)
            #self.CurveInfo[i] = ('LC', {})

            self._AdaptCycle = False
        else:
            if not self._autoMod.ClearAll():
                raise RuntimeError('Cleanup of auto module failed...')
            raise PyDSTool_ExistError('Bad initial point/cycle.  No curve computed.')
        ### END CALL TO AUTO

        #########
        # Cleanup
        #########
        if not self._autoMod.ClearAll():
            raise RuntimeError('Cleanup of auto module failed...')


    def info(self):
        Continuation.info(self)

        print('\nLimit Cycle Curve Parameters')
        print('------------------------------\n')
        args_list = limitcycle_args_list[:]
        args_list.remove('LocBifPoints')
        for arg in args_list:
            if hasattr(self, arg):
                print(arg, ' = ', eval('self.' + arg))
        print('\n')




class UserDefinedCurve(Continuation):
    """User defined curve.  We'll see how this goes..."""

    def __init__(self, model, gen, automod, plot, initargs=None):
        initargs['auxpars'] = []

        # Initialize user information
        self.varslist = initargs['uservars']
        self.parsdict = initargs['userpars'].copy()
        self._userfunc = initargs['userfunc']
        self._userdata = args()
        if 'userjac' in initargs:
            self._userjac = initargs['userjac']
        if 'usertestfuncs' in initargs:
            self._usertestfuncs = initargs['usertestfuncs']
        if 'userbifpoints' in initargs:
            self._userbifpoints = initargs['userbifpoints']
        else:
            self._userbifpoints = []
        if 'userdomain' in initargs:
            self._userdomain = initargs['userdomain']

        [initargs.pop(i) for i in ['uservars', 'userpars', 'userjac',
                'userfunc', 'usertestfuncs', 'userbifpoints',
                'userdomain'] if i in initargs]
        Continuation.__init__(self, model, gen, automod, plot, args=initargs)

        self.CorrFunc = self.sysfunc


    def update(self, args):
        """Update parameters for UserDefinedCurve."""

        Continuation.update(self, args)
        if args is not None:
            for k, v in args.items():
                if k in userdefined_args_list:
                    exec('self.' + k + ' = ' + repr(v))


    def _createTestFuncs(self):
        """Creates processors and test functions for UserDefinedCurve class.

        Note:  In the following list, processors are in PyCont.Bifpoint
        and test functions are in PyCont.TestFunc.

        Point type (Processor): Test function(s)
        ----------------------------------------

        LP (FoldPoint): Fold_Tan
                        Fold_Det
                        Fold_Bor
        H (HopfPoint): Hopf_Det
                       Hopf_Bor
        """
        Continuation._createTestFuncs(self)

        for bftype in self.LocBifPoints:
            stop = bftype in self.StopAtPoints  # Set stopping flag

            if bftype in self._userbifpoints:
                tfuncs = []
                tflags = []
                if not isinstance(self._userbifpoints[bftype], list):
                    self._userbifpoints[bftype] = [self._userbifpoints[bftype]]
                for tfunc in self._userbifpoints[bftype]:
                    tfunc_name = tfunc[0]
                    tfunc_flag = tfunc[1]
                    if tfunc_flag == 0:
                        tflags.append(iszero)
                    else:
                        tflags.append(isnotzero)
                    tfunc_func = self._usertestfuncs[tfunc_name][0]
                    tfunc_dim = self._usertestfuncs[tfunc_name][1]

                    method = UserDefinedTestFunc((self.sysfunc.n, tfunc_dim),
                                                self, tfunc_func, save=True,
                                                numpoints=self.MaxNumPoints+1)
                    self.TestFuncs.append(method)
                    tfuncs.append(method)

                self.BifPoints[bftype] = BifPoint(tfuncs, tflags,
                                                  label=bftype, stop=stop)
