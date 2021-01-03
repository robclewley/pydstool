# Radau ODE system

import importlib

from .allimports import *
from PyDSTool.Generator import ODEsystem as ODEsystem
from .baseclasses import theGenSpecHelper, genDB, _pollInputs
from .mixins import CompiledMixin, full_path
from PyDSTool.utils import *
from PyDSTool.common import *
# for future cleanup of * imports
from PyDSTool import utils
from PyDSTool import common
from PyDSTool.ModelSpec import QuantSpec
from PyDSTool.integrator import integrator
import numpy as np

# Other imports
from numpy import isfinite, int, int32, float, float64, \
    sometrue, alltrue, any, all, concatenate, transpose, array, zeros
import operator
from copy import copy, deepcopy


class radau(integrator):
    """Radau 5 specialization of the basic integrator class."""

    def __init__(self, modname, rhs='default_name', phaseDim=0, paramDim=0,
                 nAux=0, nEvents=0, nExtInputs=0, hasJac=0, hasJacP=0,
                 hasMass=0, extraSpace=0, defaultBound=1e8):
        integrator.__init__(self, rhs=rhs, phaseDim=phaseDim, paramDim=paramDim,
                            nAux=nAux, nEvents=nEvents, nExtInputs=nExtInputs, hasJac=hasJac,
                            hasJacP=hasJacP, hasMass=hasMass, extraSpace=extraSpace,
                            defaultBound=defaultBound)
        self.modname = modname
        try:
            self._integMod = importlib.import_module(
                '.' + modname,
                package='radau5_temp',
            )
        except:
            print("Error in importing compiled vector field and integrator.")
            print("Did you compile the RHS C code?")
            raise
        # check module's directory
        assert 'Integrate' in dir(self._integMod), \
               "radau library does not contain Integrate()"

        self.safety = []
        self.jacRecompute = []
        self.newtonStop = []
        self.stepChangeLB = []
        self.stepChangeUB = []
        self.stepSizeLB = []
        self.stepSizeUB = []
        self.hessenberg = []
        self.maxNewton = []
        self.newtonStart = []
        self.index1dim = []
        self.index2dim = []
        self.index3dim = []
        self.stepSizeStrategy = []
        self.DAEstructureM1 = []
        self.DAEstructureM2 = []

        retval = self._integMod.InitBasic(self.phaseDim, self.paramDim, self.nAux,
                                          self.nEvents, self.nExtInputs, self.hasJac,
                                          self.hasJacP, self.hasMass, self.extraSpace)

        if retval[0] != 1:
            raise PyDSTool_InitError('Call to InitBasic failed! (radau)')

        self.initBasic = True


    def Run(self, hinit=0, hmax=1.0, checkAux=0, calcSpecTimes=0, verbose=0,
            safety=0.9, jacRecompute=0.001, newtonStop=-1, stepChangeLB=1,
            stepChangeUB=1.2, stepSizeLB=0.2, stepSizeUB=8.0, hessenberg=0,
            maxNewton=7, newtonStart=0, index1dim=-1, index2dim=0, index3dim=0,
            stepSizeStrategy=1, DAEstructureM1=0, DAEstructureM2=0, useJac=0, useMass=0):
        if not self.initBasic:
            raise PyDSTool_InitError('initBasic is False (radau)')
        if not self.initEvents:
            raise PyDSTool_InitError('initEvents is False (radau)')
        if not self.initIntegrate:
            raise PyDSTool_InitError('initInteg is False (radau)')
        if not self.setParams:
            raise PyDSTool_InitError('setParams is False (radau)')
        if self.nExtInputs > 0 and not self.initExtInputs:
            raise PyDSTool_InitError('initExtInputs is False (radau)')

        self.setRadauParams(hinit=hinit, hmax=hmax, checkAux=checkAux,
                            calcSpecTimes=calcSpecTimes,
                            verbose=verbose, safety=safety,
                            jacRecompute=jacRecompute, newtonStop=newtonStop,
                            stepChangeLB=stepChangeLB, stepChangeUB=stepChangeUB,
                            stepSizeLB=stepSizeLB, stepSizeUB=stepSizeUB,
                            hessenberg=hessenberg,maxNewton=maxNewton,
                            newtonStart=newtonStart, index1dim=index1dim,
                            index2dim=index2dim, index3dim=index3dim,
                            stepSizeStrategy=stepSizeStrategy,
                            DAEstructureM1=DAEstructureM1,
                            DAEstructureM2=DAEstructureM2,
                            useJac=useJac,useMass=useMass)

        # For a run, we want to ensure indices are set to 0
        self.Reset()
        T, P, A, Stats, H, Err, EvtT, EvtP = self._integMod.Integrate(self.ic,
                                                      self.t0,
                                                      self.hinit,
                                                      self.hmax,
                                                      self.safety,
                                                      self.jacRecompute,
                                                      self.newtonStop,
                                                      self.stepChangeLB,
                                                      self.stepChangeUB,
                                                      self.stepSizeLB,
                                                      self.stepSizeUB,
                                                      self.hessenberg,
                                                      self.maxNewton,
                                                      self.newtonStart,
                                                      self.index1dim,
                                                      self.index2dim,
                                                      self.index3dim,
                                                      self.stepSizeStrategy,
                                                      self.DAEstructureM1,
                                                      self.DAEstructureM2,
                                                      self.useJac,
                                                      self.useMass,
                                                      self.verbose,
                                                      self.checkAux,
                                                      self.calcSpecTimes)
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


    def Continue(self, tend, params=[], calcSpecTimes=0, verbose=0,
                 extInputChanged=False, extInputVals=[], extInputTimes=[],
                 bounds=[]):
        if not self.initBasic:
            raise PyDSTool_InitError('initBasic is False (radau)')
        if not self.initEvents:
            raise PyDSTool_InitError('initEvents is False (radau)')
        if not self.initIntegrate:
            raise PyDSTool_InitError('initInteg is False (radau)')
        if not self.setParams:
            raise PyDSTool_InitError('setParams is False (radau)')
        if self.nExtInputs > 0 and not self.initExtInputs:
            raise PyDSTool_InitError('initExtInputs is False (radau)')

        if not self.canContinue:
            raise PyDSTool_ContError('Unable to continue trajectory -- '
                        'have you run the integrator and reset events, etc?')

        self.setContParams(tend=tend, params=copy(params),
                           calcSpecTimes=calcSpecTimes, verbose=verbose, extInputChanged=extInputChanged,
                           extInputVals=copy(extInputVals), extInputTimes=copy(extInputTimes),
                           bounds=copy(bounds))

        # For a continue, we do not set indices to 0
        T, P, A, Stats, H, Err, EvtT, EvtP = \
                self._integMod.Integrate(self.lastPoint,
                                      self.lastTime,
                                      self.lastStep, self.hmax,
                                      self.safety,
                                      self.jacRecompute,
                                      self.newtonStop,
                                      self.stepChangeLB,
                                      self.stepChangeUB,
                                      self.stepSizeLB,
                                      self.stepSizeUB,
                                      self.hessenberg,
                                      self.maxNewton,
                                      self.newtonStart,
                                      self.index1dim,
                                      self.index2dim,
                                      self.index3dim,
                                      self.stepSizeStrategy,
                                      self.DAEstructureM1,
                                      self.DAEstructureM2,
                                      self.useJac,
                                      self.useMass,
                                      self.verbose,
                                      self.checkAux,
                                      self.calcSpecTimes)

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


    def setRadauParams(self, hinit, hmax, checkAux, calcSpecTimes,
                       verbose, safety, jacRecompute, newtonStop,
                       stepChangeLB, stepChangeUB, stepSizeLB, stepSizeUB,
                       hessenberg, maxNewton, newtonStart, index1dim,
                       index2dim, index3dim, stepSizeStrategy,
                       DAEstructureM1, DAEstructureM2, useJac, useMass):
        useJac = int(useJac)
        useMass = int(useMass)
        checkAux = int(checkAux)
        calcSpecTimes = int(calcSpecTimes)
        hessenberg = int(hessenberg)

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

        if safety < 0:
            raise ValueError("safety must be non-negative")
        if jacRecompute <= 0.0:
            raise ValueError("jacRecompute must be positive")
        if newtonStop < 0:
            newtonStop = 0
        if stepChangeLB <= 0:
            raise ValueError("stepChangeLB must be positive")
        if stepChangeUB <= 0:
            raise ValueError("stepChangeUB must be positive")
        if stepSizeLB <= 0:
            raise ValueError("stepSizeLB must be positive")
        if stepSizeUB <= 0:
            raise ValueError("stepSizeUB must be positive")

        if stepChangeLB > stepChangeUB:   # was >= but this allows fac1=fac2=1
            raise ValueError("stepChangeLB must be less than stepChangeUB")
        if stepSizeLB >= stepSizeUB:
            raise ValueError("stepSizeLB must be less than stepSizeUB")

        if hessenberg not in (0,1):
            raise ValueError("hessenberg must be 0 or 1")
        if hessenberg == 1 and useMass != 0:
            raise ValueError("hessenberg form cannot be used for implicit systems (mass matrix)")
        if not isinstance(maxNewton, _int_types):
            raise TypeError("maxNewton must be int")
        if maxNewton <= 0:
            raise ValueError("maxNewton must be positive")

        if newtonStart not in (0,1):
            raise ValueError("newtonStart must be 0 or 1")

        if index1dim <= 0:
            index1dim = self.phaseDim
        if index2dim != 0:
            raise ValueError("Currently index2dim must be 0")
        if index3dim != 0:
            raise ValueError("Currently index3dim must be 0")

        if stepSizeStrategy not in (1,2):
            raise ValueError("stepSizeStrategy must be 1 or 2")

        if DAEstructureM1 != 0:
            raise ValueError("Currently DAEstructureM1 must be 0")
        if DAEstructureM2 != 0:
            raise ValueError("Currently DAEstructureM2 must be 0")

        if useJac not in (0,1):
            raise ValueError("useJac must be 0 or 1")
        if useMass not in (0,1):
            raise ValueError("useMass must be 0 or 1")

        if useJac == 1 and self.hasJac != 1:
            raise ValueError("useJac must be 0 if hasJac is not 1")
        if useMass == 1 and self.hasMass != 1:
            raise ValueError("useMass must be 0 if hasMass is not 1")

        self.hinit = hinit
        self.hmax = hmax
        self.safety = safety
        self.jacRecompute = jacRecompute
        self.newtonStop = newtonStop
        self.stepChangeLB = stepChangeLB
        self.stepChangeUB = stepChangeUB
        self.stepSizeLB = stepSizeLB
        self.stepSizeUB = stepSizeUB
        self.hessenberg = hessenberg
        self.maxNewton = maxNewton
        self.newtonStart = newtonStart
        self.index1dim = index1dim
        self.index2dim = index2dim
        self.index3dim = index3dim
        self.stepSizeStrategy = stepSizeStrategy
        self.DAEstructureM1 = DAEstructureM1
        self.DAEstructureM2 = DAEstructureM2
        self.useJac = useJac
        self.useMass = useMass
        self.verbose = verbose
        self.checkAux = checkAux
        self.calcSpecTimes = calcSpecTimes


class Radau_ODEsystem(ODEsystem, CompiledMixin):
    """Wrapper for Radau integrator (with support for differential-algebraic equations).

    Uses C target language only for functional specifications"""
    _paraminfo = {'rtol': 'Relative error tolerance.',
                  'atol': 'Absolute error tolerance.',
                  'safety': 'Safety factor in the step size prediction, default 0.9.',
                  'max_step': 'Maximal step size, default tend-tstart.',
                  'init_step': 'Initial step size, default is a guess computed by the function init_step.',
                  'fac1': 'Parameter for step size selection; the new step size is chosen subject to the restriction  fac1 <= new_step/old_step <= fac2. Default value is 1.0.',
                  'fac2': 'Parameter for step size selection; the new step size is chosen subject to the restriction  fac1 <= new_step/old_step <= fac2. Default value is 1.2.',
                  'stepLB': '',
                  'stepUB': '',
                  'refine': 'Refine output by adding points interpolated using the RK4 polynomial (0, 1 or 2).',
                  'step_strategy': """Switch for step size strategy;
If step_strategy=1  mod. predictive controller (Gustafsson).
If step_strategy=2  classical step size control.
The default value (for step_strategy=0) is step_strategy=1.
the choice step_strategy=1 seems to produce safer results;
for simple problems, the choice step_strategy=2 produces
often slightly faster runs.""",
                  'jac_recompute': """Decides whether the Jacobian should be recomputed;
increase jac_recompute to 0.1 say, when Jacobian evaluations
are costly. for small systems jac_recompute should be smaller
(0.001, say). negative jac_recompute forces the code to
compute the Jacobian after every accepted step.
Default 0.001.""",
                  'newton_start': "",
                  'newton_stop': "",
                  'max_newton': "Maximum number of Newton iterations to take in solving the implicit system at each step (default 7)",
                  'DAEstructureM1': "",
                  'DAEstructureM2': "",
                  'hessenberg': "",
                  'index1dim': "",
                  'index2dim': "",
                  'index3dim': "",
                  'use_special': "Switch for using special times",
                  'specialtimes': "List of special times to use during integration",
                  'check_aux': "Switch",
                  'extraspace': ""
                  }

    def __init__(self, kw):
        """Use the nobuild key to postpone building of the library, e.g. in
        order to provide additional build options to makeLibSource and
        compileLib methods or to make changes to the C code by hand.
        No build options can be specified otherwise."""

        # delete because not covered in ODEsystem
        nobuild = kw.pop('nobuild', False)
        ODEsystem.__init__(self, kw)
        self._solver = None
        self.diagnostics._errorcodes = {
              0: 'Unrecognized error code returned (see stderr output)',
            -1 : 'input is not consistent',
            -2 : 'larger nmax is needed',
             2 : 'larger nmax or maxevtpts is probably needed (error raised by solout)',
            -3 : 'step size becomes too small',
            -4 : 'the matrix is repeatedly singular (interrupted)',
            -8 : 'The solution exceeded a magbound (poor choice of initial step)'}
        self.diagnostics.outputStatsInfo = {
            'last_step': 'Predicted step size of the last accepted step (useful for a subsequent call to radau).',
            'num_steps': 'Number of used steps.',
            'num_accept': 'Number of accepted steps.',
            'num_reject': 'Number of rejected steps.',
            'num_fcns': 'Number of function evaluations.',
            'num_jacs': 'Number of Jacobian evaluations.',
            'num_dec': 'Number of LU-decompositions',
            'num_subs': 'Number of forward-backward substitutions',
            'errorStatus': 'Error status on completion.'
             }

        # currently the final four of these params are for event handling
        algparams_def = {'poly_interp': False,
                        'init_step': 0,
                        'max_step': 0,
                        'rtol': [1e-9 for i in range(self.dimension)],
                        'atol': [1e-12 for i in range(self.dimension)],
                        'fac1': 1.0,
                        'fac2': 1.2,
                        'stepLB': 0.2,
                        'stepUB': 8.0,
                        'safety': 0.9,
                        'max_pts': 10000,
                        'refine': 0,
                        'maxbisect': [], # for events
                        'maxevtpts': 1000, # for events
                        'eventInt': [], # set using setEventInterval only
                        'eventDelay': [], # set using setEventDelay only
                        'eventTol': [], # set using setEventTol only
                        'use_special': 0,
                        'specialtimes': [],
                        'check_aux': 1,
                        'extraspace': 100,
                        'verbose': 0,
                        'jac_recompute': 0.001,
                        'step_strategy': 1,
                        'index1dim': -1,
                        'index2dim': 0,
                        'index3dim': 0,
                        'DAEstructureM1': 0,
                        'DAEstructureM2': 0,
                        'hessenberg': 0,
                        'newton_start': 0,
                        'newton_stop': -1,
                        'max_newton': 7,
                        'hasJac': 0,
                        'hasJacP': 0,
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
        # Check for non-constant mass matrix
        if self.haveMass():
            mspec = self.funcspec.auxfns['massMatrix']
            lensig = len(mspec[1])
            body_str = mspec[0][lensig:].replace('\n','')
            qbody = QuantSpec('__body__', body_str, treatMultiRefs=False,
                              ignoreSpecial=['[',']','{','}'])
            self._const_massmat = intersect(['Y_','t'], qbody.usedSymbols) == []
        else:
            self._const_massmat = True

        self._prepareEventSpecs()
        self._inputVarList = []
        self._inputTimeList = []

        if nobuild:
            print("Build the library using the makeLib method, or in ")
            print("stages using the makeLibSource and compileLib methods.")
        else:
            self.makeLib()

    @property
    def integrator(self):
        return {
            'name': ('radau5' if self._const_massmat else 'radau5v', 'Radau'),
            'description': "Radau5 integrator" + \
            "" if self._const_massmat else " (version for non-constant mass matrices)",
            'src': ["radau5mod.c"],
            'cflags': ["-D__RADAU__"],
            'libs': [
                ('radau5', {
                    'sources': full_path(['radau5.f' if self._const_massmat else 'radau5v.f']),
                    'extra_f77_compile_args': utils.extra_arch_arg(['-w']),
                }),
                ('lapack_lite', {
                    'sources': full_path(['lapackc.f', 'lapack.f', 'dc_lapack.f']),
                    'extra_f77_compile_args': utils.extra_arch_arg(['-w']),
                })
            ],
        }

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
            assert isinstance(ev, LowLevelEvent), ("Radau can only "
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
            self.algparams['rtol'] = [rtol for i in range(self.dimension)]
        if isinstance(self.algparams['atol'], list):
            if len(self.algparams['atol']) != self.dimension:
                raise ValueError('atol list must have same length as phase dimension')
        else:
            atol = self.algparams['atol']
            self.algparams['atol'] = [atol for i in range(self.dimension)]
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
        self._ensure_solver()
        if self._dircode == 1:
            tbegin = t0
            tend = t1
        elif self._dircode == -1:
            # radau does reverse time integration simply by switching t0 and t1
            tbegin = t1
            tend = t0
        if len(self.algparams['specialtimes'])>0:
            use_special = self.algparams['use_special']
        else:
            use_special = 0
        bounds = [[],[]]  # lower, then upper
        for v in self.funcspec.vars:
            bds = self.xdomain[v]
            try:
                bounds[0].append(bds[0])
                bounds[1].append(bds[1])
            except TypeError:
                print("%r %s %r" % (v, type(bds), bds))
                print(self.xdomain)
                raise
        for p in self.funcspec.pars:
            bds = self.pdomain[p]
            try:
                bounds[0].append(bds[0])
                bounds[1].append(bds[1])
            except TypeError:
                print("%s %r" % (type(bds), bds))
                raise
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
#                                           deeppcopy(self._inputTimeList))
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
            max_step = abs(tend-tbegin)
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
                                  deepcopy(self._inputTimeList), bounds)
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
                                    self.algparams['safety'],
                                    self.algparams['jac_recompute'],
                                    self.algparams['newton_stop'],
                                    self.algparams['fac1'],
                                    self.algparams['fac2'],
                                    self.algparams['stepLB'],
                                    self.algparams['stepUB'],
                                    self.algparams['hessenberg'],
                                    self.algparams['max_newton'],
                                    self.algparams['newton_start'],
                                    self.algparams['index1dim'],
                                    self.algparams['index2dim'],
                                    self.algparams['index3dim'],
                                    self.algparams['step_strategy'],
                                    self.algparams['DAEstructureM1'],
                                    self.algparams['DAEstructureM2'],
                                    self.haveJacobian(),
                                    self.haveMass())
        self._extInputsChanged = False    # reset this now
        self.diagnostics.outputStats = {'last_step': H,
                            'last_time': self._solver.lastTime,
                            'last_point': self._solver.lastPoint,
                            'num_fcns': Stats[0],
                            'num_jacs': Stats[1],
                            'num_steps': Stats[2],
                            'num_accept': Stats[3],
                            'num_reject': Stats[4],
                            'num_dec': Stats[5],
                            'num_subs': Stats[6],
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
                            print("Event info: %r %r" % (Evpoints, Evtimes))
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
                            termevtimes[Evtimes[evix][0]] = \
                                       len(self.diagnostics.warnings)
                            self.diagnostics.warnings.append((W_TERMEVENT,
                                             (Evtimes[evix][0],
                                             [evname])))
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
            last_ix = np.Inf
        else:
            compare = operator.gt
            last_ix = -np.Inf
        highest_ix = X.shape[1]-1
        last_t = np.Inf
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
                last_ix = np.Inf
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
                    last_ev_tix = np.argmax(Evtimes[evix] >= alltData[last_ix])
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
        anames = self.funcspec.auxvars
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
                    raise PyDSTool_ValueError("Fewer than 2 data points computed")
            for a in anames:
                if len(alltData) > 1:
                    variables[a] = Variable(interp1d(alltData,allaDataDict[a]),
                                             't', a, a)
                else:
                    raise PyDSTool_ValueError("Fewer than 2 data points computed")
            # final checks
            #self.validateSpec()
            self.defined = True
            return Trajectory(trajname, list(variables.values()),
                              abseps=self._abseps, globalt0=self.globalt0,
                              checklevel=self.checklevel,
                              FScompatibleNames=self._FScompatibleNames,
                              FScompatibleNamesInv=self._FScompatibleNamesInv,
                              events=self.trajevents,
                              modelNames=self.name,
                              modelEventStructs=self.eventstruct)
        else:
            try:
                diagnost_info = self.diagnostics._errorcodes[int(Err)]
            except TypeError:
                # errcode messed up from Radau
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


    def MassMatrix(self, t, xdict, pdict=None, asarray=True):
        """asarray is an unused, dummy argument for compatibility with
        Model.MassMatrix"""
        if self.haveMass():
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
            return self._solver.MassMatrix(t, x, p+i)[0]
        else:
            raise PyDSTool_ExistError("Mass matrix not defined")


    def _ensure_solver(self, pars=None):
        if self._solver is None:
            x0 = sortedDictValues(filteredDict(self.initialconditions, self.funcspec.vars))
#            _integMod = self._ensureLoaded(self.modname)
            self._solver = radau(self.modname,
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
theGenSpecHelper.add(Radau_ODEsystem, symbolMapDict, 'c')
