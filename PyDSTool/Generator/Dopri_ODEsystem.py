# Dopri ODE system
from __future__ import division, absolute_import, print_function

from .allimports import *
from PyDSTool.Generator import ODEsystem as ODEsystem
from .baseclasses import Generator, theGenSpecHelper, genDB, _pollInputs
from PyDSTool.utils import *
from PyDSTool.common import *
# for future cleanup of * imports
from PyDSTool import utils
from PyDSTool import common
from PyDSTool.integrator import integrator
from PyDSTool.parseUtils import addArgToCalls, wrapArgInCall
import PyDSTool.Redirector as redirc
import numpy as npy

# Other imports
from numpy import Inf, NaN, isfinite, int, int32, float, float64, \
     sometrue, alltrue, any, all, concatenate, transpose, array, zeros
import math, random
import operator
from copy import copy, deepcopy
import os, platform, shutil, sys, gc
#import distutils
from numpy.distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from time import clock, sleep

# path to the installation
import PyDSTool
_pydstool_path = PyDSTool.__path__[0]


rout = redirc.Redirector(redirc.STDOUT)
rerr = redirc.Redirector(redirc.STDERR)


class dopri(integrator):
    """Dopri 853 specialization of the basic integrator class."""

    def __init__(self, modname, rhs='default_name', phaseDim=0, paramDim=0,
                 nAux=0, nEvents=0, nExtInputs=0,
                 hasJac=0, hasJacP=0, hasMass=0, extraSpace=0,
                 defaultBound=1e8):

        integrator.__init__(self, rhs=rhs, phaseDim=phaseDim, paramDim=paramDim,
                            nAux=nAux, nEvents=nEvents, nExtInputs=nExtInputs,
                            hasJac=hasJac, hasJacP=hasJacP, hasMass=hasMass,
                            extraSpace=extraSpace, defaultBound=defaultBound)
        self.modname = modname
        try:
            self._integMod = __import__(modname, globals())
        except:
            print("Error in importing compiled vector field and integrator.")
            print("Did you compile the RHS C code?")
            raise
        # check module's directory
        assert 'Integrate' in dir(self._integMod), \
               "dopri853 library does not contain Integrate()"

        self.fac1 = []
        self.fac2 = []
        self.safety = []
        self.beta = []
        self.checkBounds = 0
        self.boundsCheckMaxSteps = 1000
        self.magBound = 1000000

        retval = self._integMod.InitBasic(self.phaseDim, self.paramDim, self.nAux,
                                          self.nEvents, self.nExtInputs, self.hasJac,
                                          self.hasJacP, self.hasMass, self.extraSpace)

        if retval[0] != 1:
            raise PyDSTool_InitError('Call to InitBasic failed! (dopri)')

        self.initBasic = True


    def Run(self, hinit=0, hmax=1.0, checkAux=0, calcSpecTimes=0, verbose=0,
            fac1=0.2, fac2=10.0, safety=0.9, beta=0.04, checkBounds=0,
            boundsCheckMaxSteps=1000, magBound=1000000):
        if not self.initBasic:
            raise PyDSTool_InitError('initBasic is False (dopri)')
        if not self.initEvents:
            raise PyDSTool_InitError('initEvents is False (dopri)')
        if not self.initIntegrate:
            raise PyDSTool_InitError('initInteg is False (dopri)')
        if not self.setParams:
            raise PyDSTool_InitError('setParams is False (dopri)')
        if self.nExtInputs > 0 and not self.initExtInputs:
            raise PyDSTool_InitError('initExtInputs is False (dopri)')

        self.setDopriParams(hinit=hinit, hmax=hmax, checkAux=checkAux,
                            calcSpecTimes=calcSpecTimes,
                            verbose=verbose, fac1=fac1,
                            fac2=fac2, safety=safety, beta=beta,
                            checkBounds=checkBounds,
                            boundsCheckMaxSteps=boundsCheckMaxSteps,
                            magBound=magBound)

        # For a run, we want to ensure indices are set to 0
        self.Reset()
        T, P, A, Stats, H, Err, EvtT, EvtP = self._integMod.Integrate(self.ic,
                                                          self.t0,
                                                          self.hinit,
                                                          self.hmax,
                                                          self.safety,
                                                          self.fac1,
                                                          self.fac2,
                                                          self.beta,
                                                          self.verbose,
                                                          self.checkAux,
                                                          self.calcSpecTimes,
                                                          self.checkBounds,
                                                          self.boundsCheckMaxSteps,
                                                          self.magBound)
        self.points = P
        self.times = T
        self.auxPoints = A
        self.eventTimes = EvtT
        self.eventPoints = EvtP
        self.errors = Err
        self.stats = Stats
        self.step = H

        try:
            self.lastTime = self.times[-1]
            self.lastPoint = [self.points[i][-1] for i in range(self.phaseDim)]
            self.lastStep = self.step
        except IndexError:
            self.lastTime = self.t0
            self.lastPoint = self.ic
            self.lastStep = self.hinit
        self.numRuns += 1
        self.canContinue = True

        return T, P, A, Stats, H, Err, EvtT, EvtP


    def Continue(self, tend, params=[], calcSpecTimes=0, verbose=0, extInputChanged=False,
                 extInputVals=[], extInputTimes=[], bounds=[]):
        if not self.initBasic:
            raise PyDSTool_InitError('initBasic is False (dopri)')
        if not self.initEvents:
            raise PyDSTool_InitError('initEvents is False (dopri)')
        if not self.initIntegrate:
            raise PyDSTool_InitError('initInteg is False (dopri)')
        if not self.setParams:
            raise PyDSTool_InitError('setParams is False (dopri)')
        if self.nExtInputs > 0 and not self.initExtInputs:
            raise PyDSTool_InitError('initExtInputs is False (dopri)')

        if not self.canContinue:
            raise PyDSTool_ContError('Unable to continue trajectory -- '
                    'have you run the integrator and reset events, etc?')

        self.setContParams(tend=tend, params=copy(params),
                           calcSpecTimes=calcSpecTimes, verbose=verbose,
                           extInputChanged=extInputChanged,
                           extInputVals=copy(extInputVals),
                           extInputTimes=copy(extInputTimes),
                           bounds=copy(bounds))

        # For a continue, we do not set indices to 0
        T, P, A, Stats, H, Err, EvtT, EvtP = self._integMod.Integrate(self.lastPoint,
                                                          self.lastTime,
                                                          self.lastStep,
                                                          self.hmax,
                                                          self.safety,
                                                          self.fac1,
                                                          self.fac2,
                                                          self.beta,
                                                          self.verbose,
                                                          self.checkAux,
                                                          self.calcSpecTimes,
                                                          self.checkBounds,
                                                          self.boundsCheckMaxSteps,
                                                          self.magBound)
        self.points = P
        self.times = T
        self.auxPoints = A
        self.eventTimes = EvtT
        self.eventPoints = EvtP
        self.errors = Err
        self.stats = Stats
        self.step = H

        try:
            self.lastTime = self.times[-1]
            self.lastPoint = [self.points[i][-1] for i in range(self.phaseDim)]
            self.lastStep = self.step
        except IndexError:
            self.lastTime = self.t0
            self.lastPoint = self.ic
            self.lastStep = self.hinit
        self.numRuns += 1
        self.numContinues += 1
        self.canContinue = True

        return T, P, A, Stats, H, Err, EvtT, EvtP


    def setDopriParams(self,hinit,hmax,checkAux,calcSpecTimes,verbose,
                       fac1,fac2,safety,beta,checkBounds,boundsCheckMaxSteps,
                       magBound):
        checkAux = int(checkAux)
        calcSpecTimes = int(calcSpecTimes)

        if not isinstance(hinit, _num_types):
            raise TypeError("hinit must be int, float")

        if not isinstance(hmax, _num_types):
            raise TypeError("hmax must be int, float")

        if abs(hinit) > abs(hmax):
            raise ValueError("Abs value of hinit (%g) must be less than hmax (%g)"%(hinit,hmax))

        if not isinstance(checkAux, _int_types):
            raise TypeError("checkAux must be int")
        if checkAux not in (0,1):
            raise TypeError("checkAux must be 0 or 1")
        if checkAux == 1 and self.nAux <= 0:
            raise ValueError("checkAux cannot be 1 if nAux is 0")

        if not isinstance(verbose, _int_types):
            raise TypeError("verbose must be int")
        if verbose not in (0,1):
            if verbose >= 2:
                # interpret all greater values as 1
                verbose = 1
            else:
                raise TypeError("verbose must be 0 or 1")

        if not isinstance(calcSpecTimes, _int_types):
            raise TypeError("calcSpecTimes must be int")
        if calcSpecTimes not in (0,1):
            raise TypeError("calcSpecTimes must be 0 or 1")
        if calcSpecTimes == 1 and len(self.specTimes) <= 0:
            raise ValueError("calcSpecTimes cannot be 1 if specTimes is empty")

        if fac1 < 0:
            raise ValueError("fac1 must be non-negative")
        if fac2 < 0:
            raise ValueError("fac2 must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if safety < 0:
            raise ValueError("safety must be non-negative")

        if not isinstance(checkBounds, _int_types):
            raise TypeError("checkBounds must be int")
        if checkBounds not in (0,1,2):
            raise ValueError("checkBounds must be 0, 1, or 2")

        if not isinstance(boundsCheckMaxSteps, _int_types):
            raise TypeError("boundsCheckMaxSteps must be int")
        if boundsCheckMaxSteps < 0:
            raise ValueError("boundsCheckMaxSteps must be non-negative")

        if isinstance(magBound, _num_types):
            if magBound <= 0:
                raise ValueError("magBound must be positive")
            mbound = [float(magBound) for x in range(self.phaseDim)]
            self.magBound = mbound
        else:
            for x in magBound:
                if x <= 0:
                    raise ValueError("All magBound components must be positive")
            self.magBound = magBound

        self.boundsCheckMaxSteps = boundsCheckMaxSteps
        self.checkBounds = checkBounds
        self.hinit = hinit
        self.hmax = hmax
        self.checkAux = checkAux
        self.verbose = verbose
        self.calcSpecTimes = calcSpecTimes
        self.fac1 = fac1
        self.fac2 = fac2
        self.beta = beta
        self.safety = safety



class Dopri_ODEsystem(ODEsystem):
    """Wrapper for Dopri853 integrator.

    Uses C target language only for functional specifications."""
    _paraminfo = {'rtol': 'Relative error tolerance.',
                  'atol': 'Absolute error tolerance.',
                  'safety': 'Safety factor in the step size prediction, default 0.9.',
                  'fac1': 'Parameter for step size selection; the new step size is chosen subject to the restriction  fac1 <= new_step/old_step <= fac2. Default value is 0.333.',
                  'fac2': 'Parameter for step size selection; the new step size is chosen subject to the restriction  fac1 <= new_step/old_step <= fac2. Default value is 6.0.',
                  'beta': 'The "beta" for stabilized step size control. Larger values for beta ( <= 0.1 ) make the step size control more stable. Negative initial value provoke beta=0; default beta=0.04',
                  'max_step': 'Maximal step size, default tend-tstart.',
                  'init_step': 'Initial step size, default is a guess computed by the function init_step.',
                  'refine': 'Refine output by adding points interpolated using the RK4 polynomial (0, 1 or 2).',
                  'use_special': "Switch for using special times",
                  'specialtimes': "List of special times to use during integration",
                  'check_aux': "Switch",
                  'extraspace': "",
                  'magBound': "The largest variable magnitude before a bounds error flags (if checkBound > 0). Defaults to 1e7",
                  'checkBounds': "Switch to check variable bounds: 0 = no check, 1 = check up to 'boundsCheckMaxSteps', 2 = check for every point",
                  'boundsCheckMaxSteps': "Last step to bounds check if checkBound==1. Defaults to 1000."
                  }

    def __init__(self, kw):
        """Use the nobuild key to postpone building of the library, e.g. in
        order to provide additional build options to makeLibSource and
        compileLib methods or to make changes to the C code by hand.
        No build options can be specified otherwise."""

        if 'nobuild' in kw:
            nobuild = kw['nobuild']
            # delete because not covered in ODEsystem
            del kw['nobuild']
        else:
            nobuild = False
        ODEsystem.__init__(self, kw)
        self.diagnostics.outputStatsInfo = {
            'last_step': 'Predicted step size of the last accepted step (useful for a subsequent call to dop853).',
            'num_steps': 'Number of used steps.',
            'num_accept': 'Number of accepted steps.',
            'num_reject': 'Number of rejected steps.',
            'num_fcns': 'Number of function calls.',
            'errorStatus': 'Error status on completion.'
                        }
        self.diagnostics._errorcodes = {
             0 : 'Unrecognized error code returned (see stderr output)',
            -1 : 'input is not consistent',
            -2 : 'larger nmax is needed',
            2 : 'larger nmax or maxevtpts is probably needed (error raised by solout)',
            -3 : 'step size becomes too small',
            -4 : 'the problem is probably stiff (interrupted)',
            -8 : 'The solution exceeded a magbound (poor choice of initial step)'}
        self._solver = None
        algparams_def = {'poly_interp': False,
                        'init_step': 0,
                        'max_step': 0,
                        'rtol': [1e-9 for i in range(self.dimension)],
                        'atol': [1e-12 for i in range(self.dimension)],
                        'fac1': 0.2,
                        'fac2': 10.0,
                        'safety': 0.9,
                        'beta': 0.04,
                        'max_pts': 10000,
                        'refine': 0,
                        'maxbisect': [], # for events
                        'maxevtpts': 1000, # for events
                        'eventInt': [],  # set using setEventInterval only
                        'eventDelay': [], # set using setEventDelay only
                        'eventTol': [], # set using setEventTol only
                        'use_special': 0,
                        'specialtimes': [],
                        'check_aux': 1,
                        'extraspace': 100,
                        'verbose': 0,
                        'hasJac': 0,
                        'hasJacP': 0,
                        'magBound': 1e7,
                        'boundsCheckMaxSteps': 1000,
                        'checkBounds': self.checklevel
                        }
        for k, v in algparams_def.items():
            if k not in self.algparams:
                self.algparams[k] = v
        # verify that no additional keys are present in algparams, after
        # defaults are added above
        if len(self.algparams) != len(algparams_def):
            raise ValueError("Invalid keys present in algparams argument: " \
                     + str(remain(self.algparams.keys(),algparams_def.keys())))
        thisplatform = platform.system()
        if thisplatform == 'Windows':
            self._dllext = ".pyd"
        elif thisplatform in ['Linux', 'IRIX', 'Solaris', 'SunOS', 'MacOS', 'Darwin', 'FreeBSD']:
            self._dllext = '.so'
        else:
            print("Shared library extension not tested on this platform.")
            print("If this process fails please report the errors to the")
            print("developers.")
            self._dllext = '.so'
        self._compilation_tempdir = os.path.join(os.getcwd(),
                                                      "dopri853_temp")
        if not os.path.isdir(self._compilation_tempdir):
            try:
                assert not os.path.isfile(self._compilation_tempdir), \
                     "A file already exists with the same name"
                os.mkdir(self._compilation_tempdir)
            except:
                print("Could not create compilation temp directory %s" % self._compilation_tempdir)
                raise
        self._compilation_sourcedir = os.path.join(_pydstool_path,"integrator")
        self._vf_file = self.name+"_vf.c"
        self._vf_filename_ext = "_"+self._vf_file[:-2]
        self._prepareEventSpecs()
        if not (os.path.isfile(os.path.join(os.getcwd(),
                                "dop853"+self._vf_filename_ext+".py")) and \
                os.path.isfile(os.path.join(os.getcwd(),
                                "_dop853"+self._vf_filename_ext+self._dllext))):
            if not nobuild:
                self.makeLibSource()
                self.compileLib()
            else:
                print("Build the library using the makeLib method, or in ")
                print("stages using the makeLibSource and compileLib methods.")
        self._inputVarList = []
        self._inputTimeList = []


    def forceLibRefresh(self):
        """forceLibRefresh should be called after event contents are changed,
        or alterations are made to the right-hand side of the ODEs.

        Currently this function does NOT work!"""

        # (try to) free dopri module from namespace
        delfiles = True
        self._solver = None
        try:
            del(sys.modules["_dop853"+self._vf_filename_ext])
            del(sys.modules["dop853"+self._vf_filename_ext])
##            del(self._integMod)
        except NameError:
            # modules weren't loaded, so nothing to do
            delfiles = False
        if delfiles:
            gc.collect()
            # still not able to delete these files!!!!! Argh!
##            if os.path.isfile(os.path.join(os.getcwd(),
##                                    "dop853"+self._vf_filename_ext+".py")):
##                os.remove(os.path.join(os.getcwd(),
##                                    "dop853"+self._vf_filename_ext+".py"))
##            if os.path.isfile(os.path.join(os.getcwd(),
##                                 "_dop853"+self._vf_filename_ext+self._dllext)):
##                os.remove(os.path.join(os.getcwd(),
##                                 "_dop853"+self._vf_filename_ext+self._dllext))
        print("Cannot rebuild library without restarting session. Sorry.")
        print("Try asking the Python developers to make a working module")
        print("unimport function!")
##        self.makeLibSource()


    def _prepareEventSpecs(self):
        eventActive = []
        eventTerm = []
        eventDir = []
        eventDelay = []
        eventTol = []
        maxbisect = []
        eventInt = []
        # convert event specs (term, active, etc.) into integparam specs
        self._eventNames = self.eventstruct.sortedEventNames()
        for evname in self._eventNames:
            ev = self.eventstruct.events[evname]
            assert isinstance(ev, LowLevelEvent), ("Dopri can only "
                                                "accept low level events")
        # if event 'precise' flags set to False then set their tolerances
        # to be > max_step
        maxstep = self.algparams['max_step']
        for evname in self._eventNames:
            ev = self.eventstruct.events[evname]
            eventActive.append(int(ev.activeFlag))
            eventTerm.append(int(ev.termFlag))
            eventDir.append(ev.dircode)
            eventInt.append(ev.eventinterval)
            eventDelay.append(ev.eventdelay)
            if ev.preciseFlag:
                eventTol.append(ev.eventtol)
                maxbisect.append(ev.bisectlimit)
            else:
                eventTol.append(maxstep*1.5)
                maxbisect.append(1)
        self.algparams['eventTol'] = eventTol
        self.algparams['eventDelay'] = eventDelay
        self.algparams['eventInt'] = eventInt
        self.algparams['maxbisect'] = maxbisect
        self.algparams['eventActive'] = eventActive
        self.algparams['eventTerm'] = eventTerm
        self.algparams['eventDir'] = eventDir


    def makeLib(self, libsources=[], libdirs=[], include=[]):
        """makeLib calls makeLibSource and then the compileLib method.
        To postpone compilation of the source to a DLL, call makelibsource()
        separately."""
        self.makeLibSource(include)
        self.compileLib(libsources, libdirs)


    def makeLibSource(self, include=[]):
        """makeLibSource generates the C source for the vector field specification.
        It should be called only once per vector field."""

        # Make vector field (and event) file for compilation
        assert isinstance(include, list), "includes must be in form of a list"
        # codes for library types (default is USERLIB, since compiler will look in standard library d
        STDLIB = 0
        USERLIB = 1
        libinclude = dict([('Python.h', STDLIB), ('math.h', STDLIB), ('stdio.h', STDLIB),
                    ('stdlib.h', STDLIB), ('string.h', STDLIB), ('vfield.h', USERLIB),
                    ('events.h', USERLIB), ('signum.h', USERLIB), ('maxmin.h', USERLIB)])
        include_str = ''
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
                    print("Warning: library '%s'' already appears in list" % libstr\
                          + " of imported libraries")
                else:
                    include_str += "#include " + '"' + libstr + '"\n'
        allfilestr = "/*  Vector field function and events for Dopri853 integrator.\n " \
            + "  This code was automatically generated by PyDSTool, but may be modified " \
            + "by hand. */\n\n" + include_str + """
extern double *gICs;
extern double **gBds;
extern double globalt0;

static double pi = 3.1415926535897931;

double signum(double x)
{
  if (x<0) {
    return -1;
  }
  else if (x==0) {
    return 0;
  }
  else if (x>0) {
    return 1;
  }
  else {
    /* must be that x is Not-a-Number */
    return x;
  }
}

"""
        pardefines = ""
        vardefines = ""
        auxvardefines = ""
        inpdefines = ""
        # sorted version of var, par, and input names
        vnames = self._var_ixmap
        pnames = self.funcspec.pars
        inames = self.funcspec.inputs
        pnames.sort()
        inames.sort()
        for i in range(self.numpars):
            p = pnames[i]
            # add to defines
            pardefines += self.funcspec._defstr+" "+p+"\tp_["+str(i)+"]\n"
        for i in range(self.dimension):
            v = vnames[i]
            # add to defines
            vardefines += self.funcspec._defstr+" "+v+"\tY_["+str(i)+"]\n"
        for i, v in enumerate(self.funcspec.auxvars):
            auxvardefines += self.funcspec._defstr+" "+v+"\t("+self.funcspec._auxdefs_parsed[v]+")\n"
        for i in range(len(self.funcspec.inputs)):
            inp = inames[i]
            # add to defines
            inpdefines += self.funcspec._defstr+" "+inp+"\txv_["+str(i)+"]\n"
        allfilestr += "\n/* Variable, aux variable, parameter, and input definitions: */ \n" \
                      + pardefines + vardefines + auxvardefines + inpdefines + "\n"
        # preprocess event code
        allevs = ""
        if self._eventNames == []:
            numevs = 0
        else:
            numevs = len(self._eventNames)
        for evname in self._eventNames:
            ev = self.eventstruct.events[evname]
            evfullfn = ""
            assert isinstance(ev, LowLevelEvent), ("Dopri can only "
                                                "accept low level events")
            evsig = ev._LLreturnstr + " " + ev.name + ev._LLargstr
            assert ev._LLfuncstr.index(';') > 1, ("Event function code "
                    "error: Have you included a ';' character at the end of"
                                            "your 'return' statement?")
            fbody = ev._LLfuncstr
            # check fbody for calls to user-defined aux fns
            # and add hidden p argument
            if self.funcspec.auxfns:
                fbody_parsed = addArgToCalls(fbody,
                                        list(self.funcspec.auxfns.keys()),
                                        "p_, wk_, xv_")
                if 'initcond' in self.funcspec.auxfns:
                    # convert 'initcond(x)' to 'initcond("x")' for
                    # compatibility with C syntax
                    fbody_parsed = wrapArgInCall(fbody_parsed,
                                        'initcond', '"')
            else:
                fbody_parsed = fbody
            evbody = " {\n" + fbody_parsed + "\n}\n\n"
            allevs += evsig + evbody
            allfilestr += evsig + ";\n"
        # add signature for auxiliary functions
        if self.funcspec.auxfns:
            allfilestr += "\n"
            for finfo in self.funcspec.auxfns.values():
                allfilestr += finfo[1] + ";\n"
        assignEvBody = ""
        for evix in range(numevs):
            assignEvBody += "events[%d] = &%s;\n"%(evix,self._eventNames[evix])
        allfilestr += "\nint N_EVENTS = " + str(numevs) + ";\nvoid assignEvents(" \
              + "EvFunType *events){\n " + assignEvBody  \
              + "\n}\n\nvoid auxvars(unsigned, unsigned, double, double*, double*, " \
              + "double*, unsigned, double*, unsigned, double*);\n" \
              + """void jacobian(unsigned, unsigned, double, double*, double*, double**, unsigned, double*, unsigned, double*);
void jacobianParam(unsigned, unsigned, double, double*, double*, double**, unsigned, double*, unsigned, double*);
"""
        if self.funcspec.auxvars == []:
            allfilestr += "int N_AUXVARS = 0;\n\n\n"
        else:
            allfilestr += "int N_AUXVARS = " + str(len(self.funcspec.auxvars)) \
                       + ";\n\n\n"
        if self.funcspec.inputs == []:
            allfilestr += "int N_EXTINPUTS = 0;\n\n\n"
        else:
            allfilestr += "int N_EXTINPUTS = " + str(len(self.funcspec.inputs)) \
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
        allfilestr += self.funcspec.auxspec[0] + allevs
        # if jacobians or mass matrix not present, fill in dummy
        if self.haveMass():
            raise ValueError("Mass matrix declaration is incompatible with "
                             "Dopri integrator system specification")
        else:
            allfilestr += """
void massMatrix(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {
}
"""
        if not self.haveJacobian():
            allfilestr += """
void jacobian(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {
}
"""
        if not self.haveJacobian_pars():
            allfilestr += """
void jacobianParam(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {
}
""" #+ "\n/* Variable and parameter substitutions undefined:*/\n" + parundefines + varundefines + "\n"
        # write out C file
        vffile = os.path.join(self._compilation_tempdir, self._vf_file)
        try:
            file = open(vffile, 'w')
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file %s for writing" % self._vf_file)
            raise IOError(e)


    def compileLib(self, libsources=[], libdirs=[]):
        """compileLib generates a python extension DLL with integrator and vector
        field compiled and linked.

        libsources list allows additional library sources to be linked.
        libdirs list allows additional directories to be searched for
          precompiled libraries."""

        if os.path.isfile(os.path.join(os.getcwd(),
                                "_dop853"+self._vf_filename_ext+self._dllext)):
            # then DLL file already exists and we can't overwrite it at this
            # time
            proceed = False
            print("\n")
            print("-----------------------------------------------------------")
            print("Present limitation of Python: Cannot rebuild library")
            print("without exiting Python and deleting the shared library")
            print("   %s" % str(os.path.join(os.getcwd(),
                                "_dop853"+self._vf_filename_ext+self._dllext)))
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
        if self._solver is not None:
            self.forceLibRefresh()
        vffile = os.path.join(self._compilation_tempdir, self._vf_file)
        try:
            ifacefile_orig = open(os.path.join(self._compilation_sourcedir,
                                               "dop853.i"), 'r')
            ifacefile_copy = open(os.path.join(self._compilation_tempdir,
                                       "dop853_"+self._vf_file[:-2]+".i"), 'w')
            firstline = ifacefile_orig.readline()
            ifacefile_copy.write('%module dop853_'+self._vf_file[:-2]+'\n')
            iffilestr = ifacefile_orig.read()
            ifacefile_copy.write(iffilestr)
            ifacefile_orig.close()
            ifacefile_copy.close()
        except IOError:
            print("dop853.i copying error in dopri853 compilation directory")
            raise
        swigfile = os.path.join(self._compilation_tempdir,
                                "dop853"+self._vf_filename_ext+".i")
        dopwrapfile = os.path.join(self._compilation_sourcedir, "dop853mod.c")
        dopfile = os.path.join(self._compilation_sourcedir, "dop853.c")
        integfile = os.path.join(self._compilation_sourcedir, "integration.c")
        interfacefile = os.path.join(self._compilation_sourcedir, "interface.c")
        eventfile = os.path.join(self._compilation_sourcedir, "eventFinding.c")
        memfile = os.path.join(self._compilation_sourcedir, "memory.c")
        # The following if statement attempts to avoid recompiling the SWIG wrapper
        # if the files mentioned already exist, because in principle the SWIG interface
        # only needs compiling once. But this step doesn't seem to work yet.
        # Instead, it seems that SWIG always gets recompiled with everything else
        # (at least on Win32). Maybe the list of files is incorrect...
        if not (all([os.path.isfile(os.path.join(self._compilation_tempdir,
                           sf)) for sf in ['dop853'+self._vf_filename_ext+'_wrap.o',
                                           'lib_dop853'+self._vf_filename_ext+'.a',
                                           'dop853'+self._vf_filename_ext+'.py',
                                           '_dop853'+self._vf_filename_ext+'.def']])):
            modfilelist = [swigfile]
        else:
            modfilelist = []
        modfilelist.extend([dopwrapfile, dopfile, vffile, integfile, eventfile,
                           interfacefile, memfile])
        modfilelist.extend(libsources)
        script_args = ['-q', 'build', '--build-lib=.', #+os.getcwd(), # '-t/',
                 '-tdopri853_temp',#+self._compilation_tempdir,
                 '--build-base=dopri853_temp', '--build-purelib=dopri853_temp']#+self._compilation_sourcedir]
        #script_args = ['-q', 'build', '--build-lib='+os.getcwd(), '-t/',
        #               '--build-base='+self._compilation_tempdir]
        if self._compiler != '':
            script_args.append('-c'+str(self._compiler))

        # include directories for libraries
        narraydir = npy.get_numarray_include()
        npydir = npy.get_include()

        incdirs = [npydir, narraydir, os.getcwd(), self._compilation_sourcedir] #_compilation_tempdir]
        incdirs.extend(libdirs)
        # Use distutils to perform the compilation of the selected files
        try:
            distobject = setup(name = "Dopri 853 integrator",
                  author = "PyDSTool (automatically generated)",
                  script_args = script_args,
                  ext_modules = [Extension("_dop853"+self._vf_filename_ext,
                                 sources=modfilelist,
                                 include_dirs=incdirs,
#                                 library_dirs=['./'],
                                 extra_compile_args=utils.extra_arch_arg(['-w', '-D__DOPRI__', '-Wno-return-type']),
                                 extra_link_args=utils.extra_arch_arg(['-w']))])
        except:
            print("\nError occurred in generating Dopri system...")
            print("%s %s" % (sys.exc_info()[0], sys.exc_info()[1]))
            raise RuntimeError
        rout.start()    # redirect stdout
        try:
            # move library files into the user's CWD
            distdestdir = distutil_destination()
            if swigfile in modfilelist or not \
               os.path.isfile(os.path.join(self._compilation_tempdir,
                                "dop853"+self._vf_filename_ext+".py")):
                # temporary hack to fix numpy_distutils bug
                shutil.move(os.path.join(os.getcwd(),
                                  self._compilation_tempdir, distdestdir,
                                  "dopri853_temp",
                                 "dop853"+self._vf_filename_ext+".py"),
                            os.path.join(os.getcwd(),
                                 "dop853"+self._vf_filename_ext+".py"))
        except:
            rout.stop()
            print("\nError occurred in generating Dopri system")
            print("(while moving library extension modules to CWD)")
            print("%s %s" % (sys.exc_info()[0], sys.exc_info()[1]))
            raise RuntimeError
        rout.stop()    # restore stdout


    def compute(self, trajname, dirn='f', ics=None):
        continue_integ = ODEsystem.prepDirection(self, dirn)
        if ics is not None:
            self.set(ics=ics)
        self.validateICs()
        self.diagnostics.clearWarnings()
        self.diagnostics.clearErrors()
        if isinstance(self.algparams['rtol'], list):
            if len(self.algparams['rtol']) != self.dimension:
                raise ValueError('rtol list must have same length as phase dimension')
        else:
            rtol = self.algparams['rtol']
            self.algparams['rtol'] = [rtol for dimix in range(self.dimension)]
        if isinstance(self.algparams['atol'], list):
            if len(self.algparams['atol']) != self.dimension:
                raise ValueError('atol list must have same length as phase dimension')
        else:
            atol = self.algparams['atol']
            self.algparams['atol'] = [atol for dimix in range(self.dimension)]
        anames = self.funcspec.auxvars
        # Check i.c.'s are well defined (finite)
        self.checkInitialConditions()
        self.setEventICs(self.initialconditions, self.globalt0)
        # update event params in case changed since last run
        self._prepareEventSpecs()
        # Main integration
        t0 = self.indepvariable.depdomain[0]
        t1 = self.indepvariable.depdomain[1]
        plist = sortedDictValues(self.pars)
        self.algparams['hasJac'] = self.haveJacobian()
        self.algparams['hasJacP'] = self.haveJacobian_pars()
        if self._solver is None:
            self._solver = dopri("dop853"+self._vf_filename_ext,
                                 rhs=self.name, phaseDim=self.dimension,
                                 paramDim=len(plist), nAux=len(anames),
                                 nEvents=len(self._eventNames),
                                 nExtInputs=len(self.inputs),
                                 hasJac=self.algparams['hasJac'],
                                 hasJacP=self.algparams['hasJacP'],
                                 hasMass=self.haveMass(),
                                 extraSpace=self.algparams['extraspace'],
                                 )
            try:
                genDB.register(self)
            except PyDSTool_KeyError:
                errstr = "Generator " + self.name + ": this vector field's " +\
                         "DLL is already in use"
                raise RuntimeError(errstr)
        if self._dircode == 1:
            tbegin = t0
            tend = t1
        elif self._dircode == -1:
            # dopri does reverse time integration simply by switching t0 and t1
            # and using negative steps
            tbegin = t1
            tend = t0
        if len(self.algparams['specialtimes'])>0:
            use_special = self.algparams['use_special']
        else:
            use_special = 0
        bounds = [[],[]]  # lower, then upper
        for v in self.funcspec.vars:
            bds = self.xdomain[v]
            bounds[0].append(bds[0])
            bounds[1].append(bds[1])
        for p in self.funcspec.pars:
            bds = self.pdomain[p]
            bounds[0].append(bds[0])
            bounds[1].append(bds[1])
        if continue_integ:
            x0 = self._solver.lastPoint
            # overwrite t0 from self.indepvariable.domain, but use its t1
            tbegin = self._solver.lastTime
            if abs(self._solver.lastStep) < abs(self.algparams['init_step']):
                self.algparams['init_step'] = self._solver.lastStep
            if abs(t1-tbegin) < abs(self.algparams['init_step']):
                raise ValueError("Integration end point too close to initial "
                                 "point")
#            if self.inputs and self._extInputsChanged:
#                self._extInputsChanged = False
#                self._solver.setContParams(tend, plist,
#                                           use_special,
#                                           self.algparams['verbose'],
#                                           True, deepcopy(self._inputVarList),
#                                           deepcopy(self._inputTimeList))
        else:
            if self._solver.numRuns > 0:
                self._solver.clearAll()
            x0 = sortedDictValues(self.initialconditions, self.funcspec.vars)
            self._solver.setInteg(maxpts=self.algparams['max_pts'],
                rtol=self.algparams['rtol'], atol=self.algparams['atol'])
            self._solver.setRunParams(ic=x0, params=plist,
                                  t0=tbegin, tend=tend, gt0=self.globalt0,
                                  refine=self.algparams['refine'],
                                  specTimes=self.algparams['specialtimes'],
                                  bounds=bounds)
        if self.inputs:
            # self._extInputsChanged if global t0 changed so that can
            # adjust times given to the integrator (it is blind to global t0
            # when accesses input variable times)
            self._ensure_inputs(self._extInputsChanged)
        # hinit only set if not continue_integ
        if len(anames)>0:
            check_aux = self.algparams['check_aux']
        else:
            check_aux = 0
        if self.algparams['max_step'] == 0:
            max_step = tend-tbegin
        else:
            max_step = self.algparams['max_step']
        init_step = self.algparams['init_step']
        if self._dircode == 1:
            if init_step < 0:
                init_step = -init_step
            if max_step < 0:
                max_step = -max_step
        else:
            if init_step > 0:
                init_step = -init_step
            if max_step > 0:
                max_step = -max_step
        if continue_integ:
            # record needed for bounds checking and truncation
            old_highest_ix = self._solver.points.shape[1]
            alltData, X, A, Stats, H, Err, Evtimes, \
                 Evpoints = self._solver.Continue(tend, plist,
                                  use_special, self.algparams['verbose'],
                                  self._extInputsChanged,
                                  deepcopy(self._inputVarList),
                                  deepcopy(self._inputTimeList),
                                  bounds)
        else:
            old_highest_ix = 0
            self._solver.setEvents(eventActive=self.algparams['eventActive'],
                eventTerm=self.algparams['eventTerm'],
                eventDir=self.algparams['eventDir'],
                eventDelay=self.algparams['eventDelay'],
                eventInt=self.algparams['eventInt'],
                eventTol=self.algparams['eventTol'],
                maxevtpts=self.algparams['maxevtpts'],
                maxbisect=self.algparams['maxbisect'])
            alltData, X, A, Stats, H, Err, Evtimes, \
                 Evpoints = self._solver.Run(init_step,
                                    max_step,
                                    check_aux,
                                    use_special,
                                    self.algparams['verbose'],
                                    self.algparams['fac1'],
                                    self.algparams['fac2'],
                                    self.algparams['safety'],
                                    self.algparams['beta'],
                                    self.algparams['checkBounds'],
                                    self.algparams['boundsCheckMaxSteps'],
                                    self.algparams['magBound'])
        self._extInputsChanged = False    # reset this now
        self.diagnostics.outputStats = {'last_step': H,
                            'last_time': self._solver.lastTime,
                            'last_point': self._solver.lastPoint,
                            'num_fcns': Stats[0],
                            'num_steps': Stats[1],
                            'num_accept': Stats[2],
                            'num_reject': Stats[3],
                            'errorStatus': Err
                            }
        if self._dircode == -1:
            # reverse the array object (no reverse method!)
            alltData = alltData[::-1]
            X = X[:,::-1]
            if anames != []:
                A = A[:,::-1]
        xnames = self._var_ixmap
        # Package up computed trajectory in Variable variables
        # Add external inputs warnings to self.diagnostics.warnings, if any
        # (not presently supported)
##        for f in inputVarList:
##            for winfo in f.diagnostics.warnings:
##                self.diagnostics.warnings.append((W_NONTERMSTATEBD,
##                                     (winfo[0], f.name, winfo[1],
##                                      f.depdomain)))
        eventslist = self.eventstruct.query(['lowlevel', 'active'])
        termevents = self.eventstruct.query(['term'], eventslist)
        if self._eventNames != []:
            # build self.diagnostics.warnings because events happened --
            # and keep a record of which times terminal events happened because
            # Model.py's event handling procedure assumes multiple events
            # happening at one time are listed in one warning
            termevtimes = {}
            nontermevtimes = {}
            try:
                for evix in range(len(self._eventNames)):
                    if Evpoints[evix] is None:
                        continue
                    evname = self._eventNames[evix]
                    numevs = len(Evtimes[evix])
                    if self.algparams['eventTerm'][evix]:
                        if numevs > 1:
                            print("Event info:%r, %r" % (Evpoints, Evtimes))
                        assert numevs <= 1, ("Internal error: more than one "
                                         "terminal event of same type found")
                        # For safety, we should assert that this event
                        # also appears in termevents, but we don't
                        if Evtimes[evix][0] in termevtimes.keys():
                            # append event name to this warning
                            warning_ix = termevtimes[Evtimes[evix][0]]
                            self.diagnostics.warnings[warning_ix][1][1].append(evname)
                        else:
                            # make new termevtime entry for the new warning
                            termevtimes[Evtimes[evix][0]] = len(self.diagnostics.warnings)
                            self.diagnostics.warnings.append((W_TERMEVENT,
                                             (Evtimes[evix][0],
                                             [self._eventNames[evix]])))
                    else:
                        for ev in range(numevs):
                            if Evtimes[evix][ev] in nontermevtimes.keys():
                                # append event name to this warning
                                warning_ix = nontermevtimes[Evtimes[evix][ev]]
                                self.diagnostics.warnings[warning_ix][1][1].append(evname)
                            else:
                                # make new nontermevtime entry for the new warning
                                nontermevtimes[Evtimes[evix][ev]] = \
                                                            len(self.diagnostics.warnings)
                                self.diagnostics.warnings.append((W_NONTERMEVENT,
                                                 (Evtimes[evix][ev],
                                                  [evname])))
            except IndexError:
                print("Events returned from integrator are the wrong size.")
                print("  Did you change the system and not refresh the C " \
                      + "library using the forcelibrefresh() method?")
                raise
        termcount = 0
        for (w,i) in self.diagnostics.warnings:
            if w == W_TERMEVENT or w == W_TERMSTATEBD:
                if termcount > 0:
                    raise ValueError("Internal error: more than one terminal "
                                     "event found")
                termcount += 1
        # post-process check of variable bounds (if defined and algparams['checkBounds'] True)
        if self._dircode > 0:
            compare = operator.lt
            last_ix = Inf
        else:
            compare = operator.gt
            last_ix = -Inf
        highest_ix = X.shape[1]-1
        last_t = Inf
        if self.algparams['checkBounds'] > 0:
            # temp storage for repeatedly used object attributes (for lookup efficiency)
            depdomains = dict(zip(range(self.dimension),
                                  [self.variables[xn].depdomain for xn in xnames]))
            offender_ix = None
            for xi in range(self.dimension):
                if not any(depdomains[xi].isfinite()):
                    # no point in checking when the bounds are +/- infinity
                    continue
                next_last_ix = array_bounds_check(X[xi][old_highest_ix:],
                                    depdomains[xi], self._dircode) + old_highest_ix
                if compare(next_last_ix, last_ix):
                    # won't count as truncating unless the following checks
                    # hold
                    last_ix = next_last_ix
                    offender_ix = xi
            if not isfinite(last_ix) and last_ix < 0:
                # only use +Inf hereon to flag no truncation needed
                last_ix = Inf
            elif last_ix >= 0 and last_ix < highest_ix:
                # truncate data
                last_t = alltData[last_ix]
                print("Warning; domain bound reached (because algparams['checkBounds'] > 0)")
                self.diagnostics.warnings.append((W_TERMSTATEBD,
                                    (last_t, xnames[offender_ix],
                                     X[offender_ix, last_ix],
                                     depdomains[offender_ix].get())))
        # Create variables (self.variables contains no actual data)
        variables = copyVarDict(self.variables)
        # build event pointset information (reset previous trajectory's)
        # don't include events after any truncation due to state bound violation
        self.trajevents = {}
        for evix in range(len(self._eventNames)):
            evname = self._eventNames[evix]
            if Evpoints[evix] is None:
                self.trajevents[evname] = None
            else:
                try:
                    ev_a_list = []
                    for t in Evtimes[evix]:
                        tix = find(alltData, t)
                        ev_a_list.append(A[:,tix])
                    ev_array = concatenate((Evpoints[evix],
                                         transpose(array(ev_a_list, 'd'))))
                    del ev_a_list, tix
                except TypeError:
                    # A is empty
                    ev_array = Evpoints[evix]
                if last_ix >= 0 and last_ix < highest_ix:
                    # don't count last_ix = -1 which is the same as highest_ix
                    last_ev_tix = npy.argmax(Evtimes[evix] >= alltData[last_ix])
                    if last_ev_tix == 0 and Evtimes[evix][0] >= last_t:
                        # checks that there was actually a violation
                        # - so no events to record
                        self.trajevents[evname] = None
                    else:
                        # truncation needed
                        ev_array = ev_array[:, :last_ev_tix+1]
                        ev_times = Evtimes[evix][:last_ev_tix+1]
                        self.trajevents[evname] = Pointset({'coordnames': xnames+anames,
                                               'indepvarname': 't',
                                               'coordarray': ev_array,
                                               'indepvararray': ev_times})
                else:
                    # no truncation needed
                    self.trajevents[evname] = Pointset({'coordnames': xnames+anames,
                                               'indepvarname': 't',
                                               'coordarray': ev_array,
                                               'indepvararray': Evtimes[evix]})
        if last_ix >= 0 and last_ix < highest_ix:
            # truncate
            X = X[:, :last_ix]
            alltData = alltData[:last_ix]
        try:
            allxDataDict = dict(zip(xnames,X))
        except IndexError:
            print("Integration returned variable values of unexpected dimensions.")
            print("  Did you change the system and not refresh the C library" \
                  + " using the forcelibrefresh() method?")
            raise
        # storage of all auxiliary variable data
        try:
            if anames != []:
                if last_ix < highest_ix:
                    A = A[:, :last_ix]
                try:
                    allaDataDict = dict(zip(anames,A))
                except TypeError:
                    print("Internal error!  Type of A: %s" % type(A))
                    raise
        except IndexError:
            print("Integration returned auxiliary values of unexpected dimensions.")
            print("  Did you change the system and not refresh the C library" \
                  + " using the forcelibrefresh() method?")
            raise
        if int(Err) == 1 or (int(Err) == 2 and termcount == 1):
            # output OK
            if self.algparams['poly_interp']:
                rhsfn = self._solver.Rhs
                # when Dopri can output the Rhs values alongside variable
                # values then this won't be necessary
                dxvals = zeros((len(alltData),self.dimension),float)
                for tix, tval in enumerate(alltData):
                    # solver's Rhs function already contains the inputs so no
                    # need to recompute and provide here.
                    #i = _pollInputs(sortedDictValues(self.inputs), tval,
                    #                        self.checklevel)
                    # X is the output variable array, but rhsfn demands a list
                    dxvals[tix] = rhsfn(tval, list(X[:,tix]), plist)[0]
            for xi, x in enumerate(xnames):
                if len(alltData) > 1:
                    if self.algparams['poly_interp']:
                        interp = PiecewisePolynomial(alltData,
                                    array([allxDataDict[x], dxvals[:,xi]]).T, 2)
                    else:
                        interp = interp1d(alltData, allxDataDict[x])
                    variables[x] = Variable(interp, 't', x, x)
                else:
                    raise PyDSTool_ValueError("Fewer than 2 data points "
                                              "computed")
            for a in anames:
                if len(alltData) > 1:
                    variables[a] = Variable(interp1d(alltData,allaDataDict[a]),
                                             't', a, a)
                else:
                    raise PyDSTool_ValueError("Fewer than 2 data points "
                                              "computed")
            # final checks
            #self.validateSpec()
            self.defined = True
            return Trajectory(trajname, list(variables.values()),
                              abseps=self._abseps, globalt0=self.globalt0,
                              checklevel=self.checklevel,
                              FScompatibleNames=self._FScompatibleNames,
                              FScompatibleNamesInv=self._FScompatibleNamesInv,
                              modelNames=self.name, events=self.trajevents,
                              modelEventStructs=self.eventstruct)
        else:
            try:
                diagnost_info = self.diagnostics._errorcodes[int(Err)]
            except TypeError:
                # errcode messed up from Dopri
                print("Error code: %d" % Err)
                diagnost_info = self.diagnostics._errorcodes[0]
            if self._solver.verbose:
                info(self.diagnostics.outputStats, "Output statistics")
            self.defined = False
            # Did the solver run out of memory?
            if (len(alltData) == self.algparams['max_pts'] or \
                self.diagnostics.outputStats['num_steps'] >= self.algparams['max_pts']) \
                   and alltData[-1] < tend:
                print("max_pts algorithmic parameter too small: current " + \
                      "value is %i"%self.algparams['max_pts'])
#                avstep = (self.algparams['init_step']+self.diagnostics.outputStats['last_step'])/2.
                if self.diagnostics.outputStats['last_time']-tbegin > 0:
                    ms = str(int(round(self.algparams['max_pts'] / \
                              (self.diagnostics.outputStats['last_time'] - \
                               tbegin)*(tend-tbegin))))
                else:
                    ms = 'Inf'
                print("(recommended value for this trajectory segment is " + \
                      "estimated to be %s (saved in diagnostics.errors attribute))"%str(ms))
                diagnost_info += " -- recommended value is " + ms
            self.diagnostics.errors.append((E_COMPUTFAIL,
                                    (self._solver.lastTime, diagnost_info)))
            raise PyDSTool_ExistError("No trajectory created")


    def Rhs(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with Model.Rhs"""
        # must convert names to FS-compatible as '.' sorts before letters
        # while '_' sorts after!
        x = sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                          self.funcspec.vars))
        if pdict is None:
            pdict = self.pars
            # internal self.pars already is FS-compatible
            p = sortedDictValues(pdict)
        else:
            p = sortedDictValues(self._FScompatibleNames(pdict))
        i = _pollInputs(sortedDictValues(self.inputs),
                        t, self.checklevel)
        self._ensure_solver({'params': p, 't0': 0, 'tend': 1})
        self._ensure_inputs()
        return self._solver.Rhs(t, x, p+i)[0]


    def Jacobian(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.Jacobian"""
        if self.haveJacobian():
            x = sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                              self.funcspec.vars))
            if pdict is None:
                pdict = self.pars
                # internal self.pars already is FS-compatible
                p = sortedDictValues(pdict)
            else:
                p = sortedDictValues(self._FScompatibleNames(pdict))
            i = _pollInputs(sortedDictValues(self.inputs),
                            t, self.checklevel)
            self._ensure_solver({'params': p, 't0': 0, 'tend': 1})
            self._ensure_inputs()
            return self._solver.Jacobian(t, x, p+i)[0]
        else:
            raise PyDSTool_ExistError("Jacobian not defined")


    def JacobianP(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.JacobianP"""
        if self.haveJacobian_pars():
            x = sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                              self.funcspec.vars))
            if pdict is None:
                pdict = self.pars
                # internal self.pars already is FS-compatible
                p = sortedDictValues(pdict)
            else:
                p = sortedDictValues(self._FScompatibleNames(pdict))
            i = _pollInputs(sortedDictValues(self.inputs),
                            t, self.checklevel)
            self._ensure_solver({'params': p, 't0': 0, 'tend': 1})
            self._ensure_inputs()
            return self._solver.JacobianP(t, x, p+i)[0]
        else:
            raise PyDSTool_ExistError("Jacobian w.r.t. parameters not defined")


    def AuxVars(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.AuxVars"""
        x = sortedDictValues(filteredDict(self._FScompatibleNames(xdict),
                                          self.funcspec.vars))
        if pdict is None:
            pdict = self.pars
            # internal self.pars already is FS-compatible
            p = sortedDictValues(pdict)
        else:
            p = sortedDictValues(self._FScompatibleNames(pdict))
        i = _pollInputs(sortedDictValues(self.inputs),
                        t, self.checklevel)
        self._ensure_solver({'params': p, 't0': 0, 'tend': 1})
        self._ensure_inputs()
        return self._solver.AuxFunc(t, x, p+i)[0]


    def _ensure_solver(self, pars=None):
        if self._solver is None:
            sortedDictValues(filteredDict(self.initialconditions, self.funcspec.vars))
#            _integMod = self._ensureLoaded("dopri853"+self._vf_filename_ext)
            self._solver = dopri("dop853"+self._vf_filename_ext,
                                 rhs=self.name, phaseDim=self.dimension,
                                 paramDim=self.numpars,
                                 nAux=len(self.funcspec.auxvars),
                                 nEvents=len(self._eventNames),
                                 nExtInputs=len(self.inputs),
                                 hasJac=self.haveJacobian(),
                                 hasJacP=self.haveJacobian_pars(),
                                 hasMass=self.haveMass(),
                                 extraSpace=self.algparams['extraspace'])
            try:
                genDB.register(self)
            except PyDSTool_KeyError:
                errstr = "Generator " + self.name + ": this vector field's " +\
                         "DLL is already in use"
                raise RuntimeError(errstr)
            if pars is not None:
                # tend value doesn't matter
                self._solver.setRunParams(
                              ic=sortedDictValues(filteredDict(self.initialconditions,
                                                               self.funcspec.vars)),
                              params=pars['params'],
                              t0=pars['t0'], tend=pars['tend'],
                              gt0=self.globalt0,
                              refine=0, specTimes=[])

    def _ensure_inputs(self, force=False):
        if not self.inputs:
            return
        if force:
            listOK = False
        else:
            try:
                listOK = self._inputTimest0 == self.globalt0
            except AttributeError:
                # not yet defined, so proceed
                listOK = False
        if not listOK:
            self._inputVarList = []
            self._inputTimeList = []
            self._inputTimest0 = self.globalt0
            # inputVarList is a list of Variables or Pointsets
            for inp in sortedDictValues(self.inputs):
                if isinstance(inp, Variable):
                    pts = inp.getDataPoints()
                    if pts is None:
                        raise TypeError("Can only pass external input Variable objects if based on"
                                        " an underlying mesh")
                    else:
                        tvals = copy(pts[inp.indepvarname])
                        tvals -= self.globalt0
                    self._inputVarList.append(pts[inp.coordname].tolist())
                    self._inputTimeList.append(tvals.tolist())
                elif isinstance(inp, Pointset):
                    tvals = copy(inp.indepvararray)
                    tvals -= self.globalt0
                    self._inputVarList.append(inp[inp.coordname].tolist())
                    self._inputTimeList.append(tvals.tolist())
                else:
                    raise TypeError("Invalid type of input")
        if not self._solver.initExtInputs:
            self._solver.setExtInputs(True, deepcopy(self._inputVarList),
                                        deepcopy(self._inputTimeList))
        elif not listOK:
            self._solver.clearExtInputs()
            self._solver.setExtInputs(True, deepcopy(self._inputVarList),
                                    deepcopy(self._inputTimeList))
            self._solver.canContinue=True


    def __del__(self):
        genDB.unregister(self)
        ODEsystem.__del__(self)



# Register this Generator with the database

symbolMapDict = {'abs': 'fabs', 'sign': 'signum', 'mod': 'fmod'}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(Dopri_ODEsystem, symbolMapDict, 'c')
