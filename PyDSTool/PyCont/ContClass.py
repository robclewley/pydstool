""" ContClass stores continuation curves for a specified model.

    Drew LaMar, March 2006
"""


from .Continuation import (
    EquilibriumCurve, FoldCurve, HopfCurveOne, HopfCurveTwo,
    FixedPointCurve, LimitCycleCurve, UserDefinedCurve,
    FixedPointFoldCurve, FixedPointFlipCurve, FixedPointNSCurve, \
    FixedPointCuspCurve
)
from .misc import *
from .Plotting import pargs, initializeDisplay

from PyDSTool.Model import Model, findTrajInitiator
from PyDSTool.Generator import Generator
from PyDSTool.ModelTools import embed
from PyDSTool import Point, Pointset
from PyDSTool.common import pickle, Utility, args, filteredDict, isUniqueSeq
from PyDSTool.utils import remain
from PyDSTool import utils
from PyDSTool import common
from PyDSTool.core.context_managers import RedirectStdout
from PyDSTool.errors import *
from PyDSTool.matplotlib_import import *

from numpy import dot as matrixmultiply
from numpy import array, float, complex, int, float64, complex64, int32, \
     zeros, divide, subtract, Inf, NaN, isfinite, r_, c_, sign, mod, \
     subtract, divide, transpose, eye, real, imag, all, ndarray
from numpy import get_include

from PyDSTool.parseUtils import addArgToCalls, wrapArgInCall
from PyDSTool.utils import distutil_destination
from numpy.distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
# from distutils import ccompiler  ## NEEDED?
import scipy
import scipy.io as io
import os, platform, shutil, sys
from PyDSTool import __path__ as _pydstool_path

#####
_pydstool_path = _pydstool_path[0]

_classes = ['ContClass']

_constants = ['curve_list', 'curve_args_list', 'auto_list']

__all__ = _classes + _constants
#####


curve_list = {'EP-C': EquilibriumCurve, 'LP-C': FoldCurve,
              'H-C1': HopfCurveOne, 'H-C2': HopfCurveTwo,
              'FP-C': FixedPointCurve, 'LC-C': LimitCycleCurve,
              'UD-C': UserDefinedCurve, 'FD-C': FixedPointFoldCurve,
              'FL-C': FixedPointFlipCurve, 'NS-C': FixedPointNSCurve,
              'CP-C': FixedPointCuspCurve
              }

curve_args_list = ['verbosity']

auto_list = ['LC-C']



class ContClass(Utility):
    """Stores continuation curves for a specified model."""

    curve_list = curve_list
    curve_args_list = curve_args_list
    auto_list = auto_list

    def __init__(self, model):
        if isinstance(model, Generator):
            self.model = embed(model, make_copy=False)
            self.gensys = list(self.model.registry.values())[0]
        else:
            self.model = model
            mi, swRules, globalConRules, nextModelName, reused, \
                epochStateMaps, notDone = model._findTrajInitiator(None,
                                                                   0, 0, self.model.icdict, None, None)
            self.gensys = mi.model
        self._autoMod = None
        self.curves = {}
        self.plot = pargs()

    def __getitem__(self, name):
        try:
            return self.curves[name]
        except:
            raise KeyError('No curve named ' + str(name))

    def __contains__(self, name):
        return name in self.curves

    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

    def delCurve(self, curvename):
        try:
            del self.curves[curvename]
        except KeyError:
            raise KeyError("Curve %s does not exist" % curvename)

    def newCurve(self, initargs):
        """Create new curve with arguments specified in the dictionary initargs."""
        curvetype = initargs['type'].upper()

        if curvetype not in self.curve_list:
            raise PyDSTool_TypeError(str(curvetype) + ' not an allowable curve type')

        # Check name
        cname = initargs['name']
        if 'force' in initargs:
            if initargs['force'] and cname in self.curves:
                del self.curves[cname]

        if cname in self.curves:
            raise ValueError('Ambiguous name field: ' + cname \
                             + ' already exists (use force=True to override)')

        # Check parameters
        if (curvetype != 'UD-C' and self.model.pars == {}) or \
           (curvetype == 'UD-C' and 'userpars' not in initargs):
            raise ValueError('No parameters defined for this system!')

        # Process initial point
        initargs = initargs.copy()   # ensures no side-effects outside
        if 'initpoint' not in initargs or initargs['initpoint'] is None:
            # Default to initial conditions for model
            if self.model.icdict == {}:
                raise ValueError('No initial point defined for this system!')
            elif 'uservars' in initargs:
                if remain(initargs['uservars'], self.model.icdict.keys()) == []:
                    # uservars just used to select a subset of system's regular state vars
                    initargs['initpoint'] = filteredDict(self.model.icdict,
                                                         initargs['uservars'])
                else:
                    raise ValueError('No initial point defined for this system!')
            else:
                initargs['initpoint'] = self.model.icdict.copy()
            #for p in initargs['freepars']:
            #    initargs['initpoint'][p] = self.model.pars[p]
        else:
            if isinstance(initargs['initpoint'], dict):
                initargs['initpoint'] = initargs['initpoint'].copy()
                #for p in initargs['freepars']:
                #    if p not in initargs['initpoint'].keys():
                #        initargs['initpoint'][p] = self.model.pars[p]
            elif isinstance(initargs['initpoint'], str):
                curvename, pointname = initargs['initpoint'].split(':')
                pointtype = pointname.strip('0123456789')
                if curvename not in self.curves:
                    raise KeyError('No curve of name ' + curvename + ' exists.')
                else:
                    point = self.curves[curvename].getSpecialPoint(pointtype, pointname)
                    if point is None:
                        raise KeyError('No point of name ' + pointname + ' exists.')
                    else:
                        initargs['initpoint'] = point

            # Separate from if-else above since 'str' clause returns type Point
            if isinstance(initargs['initpoint'], Point):
                # Check to see if point contains a cycle.  If it does, assume
                #   we are starting at a cycle and save it in initcycle
                for v in initargs['initpoint'].labels.values():
                    if 'cycle' in v:
                        initargs['initcycle'] = v   # Dictionary w/ cycle, name, and tangent information

                # Save initial point information
                initPoint = {}
                if 'curvename' in locals() and curvename in self.curves:
                    initPoint = self.curves[curvename].parsdict.copy()

                initPoint.update(initargs['initpoint'].copy().todict())
                initargs['initpoint'] = initPoint
                # initargs['initpoint'] = initargs['initpoint'].copy().todict()
                #for p in initargs['freepars']:
                #    if p not in initargs['initpoint'].keys():
                #        initargs['initpoint'][p] = self.model.pars[p]

        # Process cycle
        if 'initcycle' in initargs:
            if isinstance(initargs['initcycle'], ndarray):
                c0 = {}
                c0['data'] = args(V = {'udotps': None, 'rldot': None})
                c0['cycle'] = Pointset({'coordnames': self.gensys.funcspec.vars,
                                        'coordarray': initargs['initcycle'][1:,:].copy(),
                                        'indepvarname': 't',
                                        'indepvararray': initargs['initcycle'][0,:].copy()
                                        })
                initargs['initcycle'] = c0
            elif isinstance(initargs['initcycle'], Pointset):
                c0 = {}
                c0['data'] = args(V = {'udotps': None, 'rldot': None})
                c0['cycle'] = initargs['initcycle']
                initargs['initcycle'] = c0

        # Load auto module if required
        automod = None
        if curvetype in auto_list:
            if self._autoMod is None:
                self.loadAutoMod()
            automod = self._autoMod

        self.curves[cname] = self.curve_list[curvetype](self.model, self.gensys, automod, self.plot, initargs)

    # Export curve data to Matlab file format
    def exportMatlab(self, filename=None):

        if not filename:
            filename = self.model.name + '.mat'

        savedict = {}

        print(list(self.__dict__['curves'].keys()))
        # Save data for each curve with different prefix
        for name in self.__dict__['curves'].keys():
            if self[name].curvetype not in ['EP-C', 'LC-C']:
                print("Can't save curve type", self[name].curvetype, "yet. (", name, ")")
                continue
            # Can save equilibrium curves
            else:
                curve = self[name].sol
                N = len(curve)
                # Currently ignoring the extra labels (B, etx)
                if self[name].curvetype == 'EP-C':
                    label = 'EP'
                elif self[name].curvetype == 'LC-C':
                    label = 'LC'


                # Add variables
                for key in curve.coordnames:
                    savedict[name+'_'+key] = array(curve[key])

                # Add stabilities
                if 'stab' in curve[0].labels[label].keys():
                    savedict[name+'_stab'] = [curve[x].labels[label]['stab'] for x in range(N)]
                    temp = zeros(N)
                    for x in range(N):
                        if savedict[name+'_stab'][x] == 'S':
                            temp[x] = -1
                        elif savedict[name+'_stab'][x] == 'U':
                            temp[x] = 1
                    savedict[name+'_stab'] = array(temp)

                # Add domain
                if 'domain' in curve[0].labels[label].keys():
                    savedict[name+'_domain'] = [curve[x].labels[label]['domain'] for x in range(N)]
                    temp = zeros(N)
                    for x in range(N):
                        if savedict[name+'_domain'][x] == 'inside':
                            temp[x] = -1
                        elif savedict[name+'_domain'][x] == 'outside':
                            temp[x] = 1
                    savedict[name+'_domain'] = array(temp)

                # Add data
                if 'data' in curve[0].labels[label].keys():
                    # Add eigenvalues
                    if 'evals' in curve[0].labels[label]['data'].keys():
                        dim = len(curve[0].labels[label]['data']['evals'])
                        evltmp = [[] for x in range(dim)]
                        for x in range(dim):
                            evltmp[x] = array([curve[y].labels[label]['data']['evals'][x] for y in range(N)])
                        savedict[name+'_evals'] = array(evltmp)

                    # Add ds
                    if 'ds' in curve[0].labels[label]['data'].keys():
                        savedict[name+'_ds'] = array([curve[y].labels[label]['data']['ds'] for y in range(N)])

                    # Add eigenvectors
                    if 'evecs' in curve[0].labels[label]['data'].keys():
                        #dim = len(curve.coordnames) - 1
                        dim = len(curve[0].labels[label]['data']['evecs'][0])
                        evectmp = []
                        for x in range(dim):
                            for y in range(dim):
                                evectmp.append(array([curve[z].labels[label]['data']['evecs'][x][y] for z in range(N)]))
                        savedict[name+'_evecs'] = array(evectmp)

                    # Add V
                    if 'V' in curve[0].labels[label]['data'].keys():
                        for key in curve[0].labels[label]['data']['V'].keys():
                            savedict[name+'_V_'+key] = array([curve[x].labels[label]['data']['V'][key] for x in range(N)])

        # Save the dictionary in matlab format
        io.savemat(filename, savedict)


    def exportGeomview(self, coords=None, filename="geom.dat"):
        if coords is not None and len(coords) == 3:
            GeomviewOutput = "(progn (geometry " + self.model.name + " { LIST {: axes_" + self.model.name + "}"
            for cname, curve in self.curves.items():
                GeomviewOutput += " {: " + cname + "}"
            GeomviewOutput += "}))\n\n"

            # Get axes limits
            alim = [[Inf,-Inf],[Inf,-Inf],[Inf,-Inf]]
            for cname, curve in self.curves.items():
                for n in range(len(coords)):
                    alim[n][0] = min(alim[n][0], min(curve.sol[coords[n]].toarray()))
                    alim[n][1] = max(alim[n][1], max(curve.sol[coords[n]].toarray()))

            GeomviewOutput += "(progn (hdefine geometry axes_" + self.model.name + " { appearance { linewidth 2 } SKEL 4 3 " + \
                "0 0 0 1 0 0 0 1 0 0 0 1 " + \
                "2 0 1 1 0 0 1 2 0 2 0 1 0 1 2 0 3 0 0 1 1})\n\n"

            for cname, curve in self.curves.items():
                GeomviewOutput += "(hdefine geometry " + cname + " { LIST {: curve_" + cname + "} {: specpts_" + cname + "}})\n\n"

                GeomviewOutput += "(hdefine geometry curve_" + cname + " { appearance { linewidth 2 } SKEL " + \
                    repr(len(curve.sol)) + " " + repr(len(curve.sol)-1)
                for n in range(len(curve.sol)):
                    GeomviewOutput += " " + repr((curve.sol[n][coords[0]]-alim[0][0])/(alim[0][1]-alim[0][0])) + \
                        " " + repr((curve.sol[n][coords[1]]-alim[1][0])/(alim[1][1]-alim[1][0])) + \
                        " " + repr((curve.sol[n][coords[2]]-alim[2][0])/(alim[2][1]-alim[2][0]))
                for n in range(len(curve.sol)-1):
                    GeomviewOutput += " 2 " + repr(n) + " " + repr(n+1) + " 0 0 0 1"

                GeomviewOutput += "})\n\n"

            GeomviewOutput += ")\n"

            f = open(filename, "w")
            f.write(GeomviewOutput)
            f.close()
        else:
            raise Warning("Coordinates not specified or not of correct dimension.")

    def display(self, coords=None, curves=None, figure=None, axes=None, stability=False, domain=False, **plot_args):
        """Plot all curves in coordinates specified by coords.

           Inputs:

               coords -- pair of coordinates (None defaults to the first free
                   parameter and the first state variable).
                   Use a 3-tuple to export to geomview.
        """
        if coords is not None and len(coords) == 3:
            self.exportGeomview(coords)
            return

        if curves is None:
            curves = self.curves.keys()

        plot_curves = []
        for curve in curves:
            if curve in self.curves:
                plot_curves.append(curve)
            else:
                print("Warning: Curve " + curve + " does not exist.")

        if len(plot_curves) > 0:
            initializeDisplay(self.plot, figure=figure, axes=axes)

        for curve in plot_curves:
            self.curves[curve].display(coords, figure=figure, axes=axes, stability=stability, domain=domain, init_display=False, **plot_args)

    def computeEigen(self):
        for curve in self.curves.values():
            curve.computeEigen()

    def info(self):
        print(self.__repr__())
        #print "  Variables : %s"%', '.join(self.model.allvars)
        #print "  Parameters: %s\n"%', '.join(self.model.pars.keys())
        print("Containing curves: ")
        for c in self.curves:
            print("  " + c + " (type " + self.curves[c].curvetype + ")")

    def update(self, args):
        """Update parameters for all curves."""
        for c in args.keys():
            if c not in curve_args_list:
                args.pop(c)

        for v in self.curves.values():
            v.update(args)

    def loadAutoMod(self, nobuild=False):
        thisplatform = platform.system()
        if thisplatform == 'Windows':
            self._dllext = ".pyd"
        elif thisplatform in ['Linux', 'IRIX', 'Solaris', 'SunOS', 'Darwin', 'FreeBSD']:
            self._dllext = '.so'
        else:
            print("Shared library extension not tested on this platform.")
            print("If this process fails please report the errors to the")
            print("developers.")
            self._dllext = '.so'

        self._compilation_tempdir = os.path.join(os.getcwd(),
                                                 "auto_temp")
        if not os.path.isdir(self._compilation_tempdir):
            try:
                assert not os.path.isfile(self._compilation_tempdir), \
                       "A file already exists with the same name"
                os.mkdir(self._compilation_tempdir)
            except:
                print("Could not create compilation temp directory " + \
                      self._compilation_tempdir)
                raise
        self._compilation_sourcedir = os.path.join(_pydstool_path,"PyCont/auto/module")
        self._vf_file = self.gensys.name+"_vf.c"
        self._vf_filename_ext = "_"+self._vf_file[:-2]
        if not (os.path.isfile(os.path.join(os.getcwd(),
                                            "auto"+self._vf_filename_ext+".py")) and \
                os.path.isfile(os.path.join(os.getcwd(),
                                            "_auto"+self._vf_filename_ext+self._dllext))):
            self.funcspec = self.gensys.funcspec.recreate('c')
            if not nobuild:
                self.makeAutoLibSource()
                self.compileAutoLib()
            else:
                print("Build the library using the makeAutoLib method, or in ")
                print("stages using the makeAutoLibSource and compileAutoLib methods.")
                print("Then load the Auto module using the importAutoLib method")

        if not nobuild:
            self.importAutoLib()

    def importAutoLib(self):
        """
        Import the Auto library.
        This method should be called only after compiling the Auto library.

        In general, users do not need to explicitely call this method except if the "nobuild" option
        was specified in loadAutoMod.
        """
        try:
            self._autoMod = __import__("auto"+self._vf_filename_ext, globals())
        except:
            print("Error loading auto module.")
            raise

    def forceAutoLibRefresh(self):
        """forceAutoLibRefresh should be called after event contents are changed,
        or alterations are made to the right-hand side of the ODEs.

        Currently this function does NOT work!"""

        # (try to) free auto module from namespace
        delfiles = True
        try:
            del(sys.modules["_auto"+self._vf_filename_ext])
            del(sys.modules["auto"+self._vf_filename_ext])
        except NameError:
            # modules weren't loaded, so nothing to do
            delfiles = False
        if delfiles:
            gc.collect()
            # still not able to delete these files!!!!! Argh!
        print("Cannot rebuild library without restarting session. Sorry.")
        print("Try asking the Python developers to make a working module")
        print("unimport function!")

    def makeAutoLib(self, libsources=[], libdirs=[], include=[]):
        """makeAutoLib calls makeAutoLibSource and then the compileAutoLib method.
        To postpone compilation of the source to a DLL, call makeAutoLibSource()
        separately."""
        self.makeAutoLibSource(include)
        self.compileAutoLib(libsources, libdirs)

    def makeAutoLibSource(self, include=[]):
        """makeAutoLibSource generates the C source for the vector field specification.
        It should be called only once per vector field."""

        # Make vector field (and event) file for compilation
        assert isinstance(include, list), "includes must be in form of a list"
        # codes for library types (default is USERLIB, since compiler will look in standard library d
        STDLIB = 0
        USERLIB = 1
        libinclude = dict([('math.h', STDLIB), ('stdio.h', STDLIB), ('stdlib.h', STDLIB),
                           ('string.h', STDLIB), ('autovfield.h', USERLIB),
                           ('auto_c.h', USERLIB)])
        include_str = '#include "auto_f2c.h"\n' # This must come first
        for libstr, libtype in libinclude.items():
            if libtype == STDLIB:
                quoteleft = '<'
                quoteright = '>'
            else:
                quoteleft = '"'
                quoteright = '"'
            include_str += "#include " + quoteleft + libstr + quoteright + "\n"
        if include != []:
            assert isUniqueSeq(include), "list of library includes must not contain repeats"
            for libstr in include:
                if libstr in libinclude:
                    # don't repeat libraries
                    print("Warning: library '" + libstr + "' already appears in list"\
                          + " of imported libraries")
                else:
                    include_str += "#include " + '"' + libstr + '"\n'

        # f2c auto conventions (dirty trick!)
        #define_str = "\n#define double    doublereal\n#define int   integer\n\n"
        define_str = ""

        allfilestr = "/*  Vector field and other functions for Auto continuer.\n " \
            + "  This code was automatically generated by PyDSTool, but may be modified " \
            + "by hand. */\n\n" + include_str + define_str + """
double *gICs;
double **gBds;
double globalt0;

static double pi = 3.1415926535897931;

"""
        pardefines = ""
##        parundefines = ""
        vardefines = ""
##        varundefines = ""
        inpdefines = ""
##        inpundefines = ""
        # sorted version of var, par, and input names
        vnames = self.gensys._var_ixmap
        pnames = self.funcspec.pars
        inames = self.funcspec.inputs
        pnames.sort()
        inames.sort()
        for i in range(self.gensys.numpars):
            p = pnames[i]
            # add to defines (WATCH OUT FOR PERIOD _T!!!)
            if (i < 10):
                pardefines += self.funcspec._defstr+" "+p+"\tp_["+str(i)+"]\n"
            elif (i >= 10):
                pardefines += self.funcspec._defstr+" "+p+"\tp_["+str(i+40)+"]\n"
        # add period _T
        pardefines += self.funcspec._defstr+" _T\tp_[10]\n"
        for i in range(self.gensys.dimension):
            v = vnames[i]
            # add to defines
            vardefines += self.funcspec._defstr+" "+v+"\tY_["+str(i)+"]\n"
##            # add to undefines
##            varundefines += self.funcspec._undefstr+" "+v+"\n"
        for i in range(len(self.funcspec.inputs)):
            inp = inames[i]
            # add to defines
            inpdefines += self.funcspec._defstr+" "+inp+"\txv_["+str(i)+"]\n"
##            # add to undefines
##            inpundefines += self.funcspec._undefstr+" "+inp+"\n"
        allfilestr += "\n/* Variable, parameter, and input definitions: */ \n" \
            + pardefines + vardefines + inpdefines + "\n"
        # add signature for auxiliary functions
        if self.funcspec.auxfns:
            allfilestr += "\n"
            for finfo in self.funcspec.auxfns.values():
                allfilestr += finfo[1] + ";\n"
        allfilestr += "\nvoid auxvars(unsigned, unsigned, double, double*, double*, " \
            + "double*, unsigned, double*, unsigned, double*);\n" \
            + """void jacobian(unsigned, unsigned, double, double*, double*, double**, unsigned, double*, unsigned, double*);
void jacobianParam(unsigned, unsigned, double, double*, double*, double**, unsigned, double*, unsigned, double*);
"""
        if self.funcspec.auxvars == []:
            allfilestr += "int N_AUXVARS = 0;\n\n\n"
        else:
            allfilestr += "int N_AUXVARS = " + str(len(self.funcspec.auxvars)) \
                + ";\n\n\n"
        allfilestr += self.funcspec.spec[0] + "\n\n"

        if self.funcspec.auxfns:
            for fname, finfo in self.funcspec.auxfns.items():
                fbody = finfo[0]
                # subs _p into auxfn-to-auxfn calls (but not to the signature)
                fbody_parsed = addArgToCalls(fbody,
                                             list(self.funcspec.auxfns.keys()),
                                             "p_, wk_, xv_", notFirst=fname)
                if 'initcond' in self.funcspec.auxfns:
                    # convert 'initcond(x)' to 'initcond("x")' for
                    # compatibility with C syntax, but don't affect the
                    # function signature!
                    fbody_parsed = wrapArgInCall(fbody_parsed,
                                                 'initcond', '"', notFirst=True)
                allfilestr += "\n" + fbody_parsed + "\n\n"
        # add auxiliary variables (shell of the function always present)
        # add event functions
        allfilestr += self.funcspec.auxspec[0]
        # if jacobians or mass matrix not present, fill in dummy
        if not self.gensys.haveJacobian():
            allfilestr += """
void jacobian(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {
}
"""
        if not self.gensys.haveJacobian_pars():
            allfilestr += """
void jacobianParam(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {
}
""" #+ "\n/* Variable and parameter substitutions undefined:*/\n" + parundefines + varundefines + "\n"
        # write out C file
        vffile = os.path.join(self._compilation_tempdir, self._vf_file)
        try:
            file = open(vffile, 'w')
            #allfilestr = allfilestr.replace("double ","doublereal ")
            #allfilestr = allfilestr.replace("double*","doublereal *")
            #allfilestr = allfilestr.replace("double**","doublereal **")
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file "+self._vf_file+" for writing")
            raise IOError(e)

    def compileAutoLib(self, libsources=[], libdirs=[]):
        """compileAutoLib generates a python extension DLL with continuer and vector
        field compiled and linked.

        libsources list allows additional library sources to be linked.
        libdirs list allows additional directories to be searched for
          precompiled libraries."""

        if os.path.isfile(os.path.join(os.getcwd(),
                                       "_auto"+self._vf_filename_ext+self._dllext)):
            # then DLL file already exists and we can't overwrite it at this
            # time
            proceed = False
            print("\n")
            print("-----------------------------------------------------------")
            print("Present limitation of Python: Cannot rebuild library")
            print("without exiting Python and deleting the shared library")
            print("   " + str(os.path.join(os.getcwd(),
                                           "_auto"+self._vf_filename_ext+self._dllext)))
            print("by hand! If you made any changes to the system you should")
            print("not proceed with running the integrator until you quit")
            print("and rebuild.")
            print("-----------------------------------------------------------")
            print("\n")
        else:
            proceed = True
        if not proceed:
            print("Did not compile shared library.")
            return
        if self._autoMod is not None:
            self.forceAutoLibRefresh()
        vffile = os.path.join(self._compilation_tempdir, self._vf_file)
        try:
            ifacefile_orig = open(os.path.join(self._compilation_sourcedir,
                                               "automod.i"), 'r')
            ifacefile_copy = open(os.path.join(self._compilation_tempdir,
                                               "auto_"+self._vf_file[:-2]+".i"), 'w')
            firstline = ifacefile_orig.readline()
            ifacefile_copy.write('%module auto_'+self._vf_file[:-2]+'\n')
            iffilestr = ifacefile_orig.read()
            ifacefile_copy.write(iffilestr)
            ifacefile_orig.close()
            ifacefile_copy.close()
        except IOError:
            print("automod.i copying error in auto compilation directory")
            raise

        swigfile = os.path.join(self._compilation_tempdir,
                                "auto"+self._vf_filename_ext+".i")
        automodfile = os.path.join(self._compilation_sourcedir, "automod.c")
        interfacefile = os.path.join(self._compilation_sourcedir, "interface.c")

        # source files
        if not (all([os.path.isfile(os.path.join(self._compilation_tempdir,
                                                 sf)) for sf in ['auto'+self._vf_filename_ext+'_wrap.o',
                                                                 'auto'+self._vf_filename_ext+'.py',
                                                                 '_auto'+self._vf_filename_ext+'.def']])):
            modfilelist = [swigfile]
        else:
            modfilelist = []
        # FOR DIST (ADD)
        modfilelist.extend([os.path.join(self._compilation_sourcedir, "../src/"+x) \
                            for x in ['auto.c','autlib1.c','autlib2.c','autlib3.c','autlib4.c','autlib5.c', \
                                      'eispack.c', 'conpar.c','setubv.c','reduce.c','dmatrix.c','fcon.c','libf2c/cabs.c','libf2c/d_lg10.c', \
                                      'libf2c/i_nint.c','libf2c/pow_di.c','libf2c/r_lg10.c','libf2c/z_exp.c','libf2c/d_imag.c', \
                                      'libf2c/d_sign.c','libf2c/i_dnnt.c','libf2c/pow_dd.c','libf2c/pow_ii.c','libf2c/z_abs.c', \
                                      'libf2c/z_log.c']])

        modfilelist.extend([automodfile, interfacefile, vffile])
        # FOR DIST (SUBTRACT)
        #modfilelist.extend(libsources)

        # script args
        script_args = ['--verbose', 'build', '--build-lib=.', #+os.getcwd(), # '-t/',
                       '-tauto_temp', #+self._compilation_tempdir,
                       '--build-base=auto_temp'] #+self._compilation_sourcedir]
        if self.gensys._compiler != '':
            script_args.append('-c'+str(self.gensys._compiler))

        # include directories for libraries
        incdirs = [get_include()]
        incdirs.extend([os.getcwd(), os.path.join(self._compilation_sourcedir,"include"),
                   self._compilation_tempdir, os.path.join(_pydstool_path,"PyCont/auto/src/include")])
        incdirs.extend(libdirs)

        # libraries
        # FOR DIST (SUBTRACT)
        #libdirs.append(os.path.join(_pydstool_path, "PyCont/auto/lib"))
        #libsources.append('auto2000')

        # Use distutils to perform the compilation of the selected files
        with RedirectStdout(os.path.join('auto_temp', 'auto.log')):
            setup(name="Auto 2000 continuer",
                  author="PyDSTool (automatically generated)",
                  script_args=script_args,
                  ext_modules=[Extension(
                      "_auto" + self._vf_filename_ext,
                      sources=modfilelist,
                      include_dirs=incdirs,
                      extra_compile_args=utils.extra_arch_arg([
                          '-w', '-D__PYTHON__', '-std=c99']),
                      extra_link_args=utils.extra_arch_arg(['-w']),
                      library_dirs=libdirs + ['./'],
                      libraries=libsources)])
        try:
            # move library files into the user's CWD
            distdestdir = distutil_destination()
            if swigfile in modfilelist or not \
               os.path.isfile(os.path.join(self._compilation_tempdir,
                                           "auto"+self._vf_filename_ext+".py")):
                shutil.move(os.path.join(os.getcwd(),
                                         self._compilation_tempdir, distdestdir,
                                         "auto_temp",
                                         "auto"+self._vf_filename_ext+".py"),
                            os.path.join(os.getcwd(),
                                         "auto"+self._vf_filename_ext+".py"))
        except:
            print("\nError occurred in generating Auto system")
            print("(while moving library extension modules to CWD)")
            print(sys.exc_info()[0], sys.exc_info()[1])
            raise RuntimeError

    def __repr__(self):
        return 'ContClass of model %s'%self.model.name

    __str__ = __repr__
