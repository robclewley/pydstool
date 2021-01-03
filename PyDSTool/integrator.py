"""Basic integrator interface class
   Erik Sherwood, September 2006
"""

from .errors import PyDSTool_InitError as InitError
from .errors import PyDSTool_ClearError as ClearError
from .common import _all_int, _real_types, \
     verify_intbool, verify_pos, verify_nonneg, verify_values
import numpy as np
import operator
import sys

# Need to add checks for when tolerances that should be nonneg are
# less than 0
class integrator:
    def __init__(self, rhs='default_name', phaseDim=0, paramDim=0, nAux=0,
                 nEvents=0, nExtInputs=0, hasJac=0, hasJacP=0, hasMass=0,
                 extraSpace=0, defaultBound=1e8):

        # Internal state variables
        self.initBasic = False
        self.initIntegrate = False
        self.initEvents = False
        self.initExtInputs = False
        self.setParams = False
        self.numRuns = 0
        self.numContinues = 0
        self.canContine = False

        # Run output
        self.times = []
        self.points = []
        self.auxPoints = []
        self.eventTimes = []
        self.eventPoints = []
        self.errors = []
        self.stats = []
        self.step = []

        self.lastPoint = []
        self.lastTime = []

        # Run parameters
        self.ic = []; self.params = []
        self.t0 = []; self.tend = []; self.gt0 = []
        self.refine = []
        self.upperBounds = []
        self.lowerBounds = []
        self.defaultBound = abs(float(defaultBound))

        # Event variables
        self.maxevtpts = []
        self.eventActive = []; self.eventDir = []; self.eventTerm = []
        self.eventInt = []; self.eventDelay=[]; self.eventTol=[]
        self.maxbisect = []

        # Integ variables
        self.maxpts = []; self.rtol = []; self.atol = []

        # Specific to integrator
        self.hinit = []; self.hmax = []; self.verbose = []
        self.specTimes = []
        self.direction = 0

        self.checkAux = 0
        self.calcSpecTimes = 0

        self.checkBasic(rhs=rhs, phaseDim=phaseDim, paramDim=paramDim,
                           nAux=nAux, nEvents=nEvents, nExtInputs=nExtInputs,
                           hasJac=hasJac, hasJacP=hasJacP, hasMass=hasMass,
                           extraSpace=extraSpace)

        self.rhs = rhs
        self.phaseDim = phaseDim
        self.paramDim = paramDim
        self.extraSpace = extraSpace
        self.nAux = nAux
        self.nEvents = nEvents
        self.nExtInputs = nExtInputs

        self.hasJac = int(hasJac)
        self.hasJacP = int(hasJacP)
        self.hasMass = int(hasMass)

        # At this point, we expect the child class to set the self._integ field
        # and then call the initBasic method of the shared library.
        # The child class sets self.initBasic to True


    def __del__(self):
        try:
            self._integMod.CleanUp()
        except:
            pass


    def checkBasic(self, rhs, phaseDim, paramDim, nAux, nEvents, nExtInputs,
                   hasJac, hasJacP, hasMass, extraSpace):
        # Check that inputs to this function are correct
        try:
            if not isinstance(rhs, str):
                raise TypeError("right hand side rhs must be a string")
            verify_nonneg('phaseDim', phaseDim, _all_int)
            verify_nonneg('paramDim', paramDim, _all_int)
            verify_intbool('hasJac', hasJac)
            verify_intbool('hasJacP', hasJacP)
            verify_intbool('hasMass', hasMass)
            verify_nonneg('nAux', nAux, _all_int)
            verify_nonneg('nEvents', nEvents, _all_int)
            verify_nonneg('nExtInputs', nExtInputs, _all_int)
            verify_nonneg('extraSpace', extraSpace, _all_int)
        except:
            print("%s %s" % (sys.exc_info()[0], sys.exc_info()[1]))
            raise InitError('Integrator initialization failed!')


    def setEvents(self, maxevtpts=1000, eventActive=None, eventDir=None,
                  eventTerm=None, eventInt=0.005, eventDelay=1e-4,
                  eventTol=1e-6, maxbisect=100, eventNearCoef=1000):

        if not self.initBasic:
            raise InitError('You must initialize the integrator before setting events. (initBasic)')

        if self.initEvents:
            raise InitError('You must clear events before setting them. Use clearEvents()')

        # Currently we will not raise an error, but instead ignore setting events
        # if nEvents is zero. Just set to some default values and pass to the
        # shared library.
        if self.nEvents <= 0:
            maxevtpts = 0
            eventActive = []
            eventDir = []
            eventTerm = []
            eventInt = 0.005
            eventDelay = 1e-4
            eventTol = 1e-6
            maxbisect = 100
            eventNearCoef = 1000

        if eventActive is None:
            eventActive = []
        if eventDir is None:
            eventDir = []
        if eventTerm is None:
            eventTerm = []

        self.checkEvents(maxevtpts, eventActive, eventDir, eventTerm,
                         eventInt, eventDelay, eventTol, maxbisect, eventNearCoef)
        self.maxevtpts = maxevtpts

        if isinstance(eventActive, list):
            self.eventActive = [int(ea) for ea in eventActive]
        else:
            self.eventActive = [int(eventActive)]*self.nEvents

        if isinstance(eventDir, list):
            self.eventDir = eventDir
        else:
            self.eventDir = [eventDir]*self.nEvents

        if isinstance(eventTerm, list):
            self.eventTerm = [int(et) for et in eventTerm]
        else:
            self.eventTerm = [int(eventTerm)]*self.nEvents

        if isinstance(maxbisect, list):
            self.maxbisect = maxbisect
        else:
            self.maxbisect = [maxbisect]*self.nEvents

        if isinstance(eventInt, list):
            self.eventInt = [float(ei) for ei in eventInt]
        else:
            self.eventInt = [float(eventInt)]*self.nEvents

        if isinstance(eventDelay, list):
            self.eventDelay = [float(ed) for ed in eventDelay]
        else:
            self.eventDelay = [float(eventDelay)]*self.nEvents

        if isinstance(eventTol, list):
            self.eventTol = [float(et) for et in eventTol]
        else:
            self.eventTol = [float(eventTol)]*self.nEvents

        self.eventNearCoef = eventNearCoef

        retval = self._integMod.InitEvents(self.maxevtpts, self.eventActive,
                                           self.eventDir, self.eventTerm,
                                           self.eventInt, self.eventDelay,
                                           self.eventTol, self.maxbisect,
                                           self.eventNearCoef)

        if retval[0] != 1:
            raise InitError('InitEvents call failed!')

        self.canContinue = False
        self.initEvents = True


    def clearEvents(self):
        if not self.initBasic:
            raise ClearError('You must initialize the integrator before clearing events.')
        if self.initEvents:
            if self._integMod.ClearEvents()[0] != 1:
                raise ClearError('ClearEvents call failed!')
            self.canContinue = False
            self.initEvents = False


    def checkEvents(self, maxevtpts, eventActive, eventDir, eventTerm,
                    eventInt, eventDelay, eventTol, maxbisect, eventNearCoef):

        if not self.initBasic:
            raise InitError('You must initialize the integrator before ' + \
                            'checking events. (initBasic)')

        verify_nonneg('maxevtpts', maxevtpts, _all_int)

        list_len_val = (self.nEvents, 'nEvents')

        verify_intbool('eventActive', eventActive,
                      list_ok=True, list_len=list_len_val)

        verify_values('eventDir', eventDir, [-1, 0, 1], list_ok=True,
                      list_len=list_len_val)

        verify_intbool('eventTerm', eventTerm,
                       list_ok=True, list_len=list_len_val)

        verify_nonneg('eventInt', eventInt, _real_types,
                      list_ok=True, list_len=list_len_val)

        verify_nonneg('eventDelay', eventDelay, _real_types,
                      list_ok=True, list_len=(self.nEvents, 'nEvents'))

        verify_pos('eventTol', eventTol, _real_types,
                   list_ok=True, list_len=list_len_val)

        verify_nonneg('maxbisect', maxbisect, _all_int,
                      list_ok=True, list_len=list_len_val)

        verify_pos('eventNearCoef', eventNearCoef, _real_types)


    def setInteg(self, maxpts=100000, rtol=1e-9, atol=1e-12):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before setting integ. (initBasic)')

        if self.initIntegrate:
            raise InitError('You must clear integ before setting it. Use clearInteg()')

        self.checkInteg(maxpts, rtol, atol)
        self.maxpts = maxpts

        if isinstance(rtol, list):
            self.rtol = rtol
        else:
            self.rtol = [rtol]*self.phaseDim

        if isinstance(atol, list):
            self.atol = atol
        else:
            self.atol = [atol]*self.phaseDim

        if self._integMod.InitInteg(self.maxpts, self.atol, self.rtol)[0] != 1:
            raise InitError('InitInteg call failed!')

        self.canContinue = False
        self.initIntegrate = True


    def clearInteg(self):
        if not self.initBasic:
            raise ClearError('You must initialize the integrator before clearing events.')
        if self.initIntegrate:
            if self._integMod.ClearInteg()[0] != 1:
                raise ClearError('ClearInteg call failed!')
            self.canContinue = False
            self.initIntegrate = False


    def checkInteg(self, maxpts, rtol, atol):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before checking integ. (initBasic)')

        verify_pos('maxpts', maxpts, _all_int)

        list_len_val = (self.phaseDim, 'phaseDim')
        verify_pos('rtol', rtol, _real_types,
                   list_ok=True, list_len=list_len_val)
        verify_pos('atol', atol, _real_types,
                   list_ok=True, list_len=list_len_val)


    def setExtInputs(self, doCheck, extInputVals, extInputTimes):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before setting external Inputs. (initBasic)')

        if self.initExtInputs:
            raise InitError('You must clear extInputs before setting it. Use clearInteg()')

        if self.nExtInputs > 0:
            if doCheck:
                self.checkExtInputs(extInputVals, extInputTimes)
            self.extInputLens = []
            for x in range(self.nExtInputs):
                self.extInputLens.append(len(extInputTimes[x]))

            IVals = extInputVals[0]
            ITimes = extInputTimes[0]
            for x in range(self.nExtInputs - 1):
                IVals += extInputVals[x+1]
                ITimes += extInputTimes[x+1]

            self.extInputVals = extInputVals
            self.extInputTimes = extInputTimes

            if self._integMod.InitExtInputs(self.nExtInputs,
                           self.extInputLens, IVals, ITimes)[0] != 1:
                raise InitError('InitExtInputs call failed!')

            self.canContinue = False
            self.initExtInputs = True


    def clearExtInputs(self):
        if not self.initBasic:
            raise ClearError('You must initialize the integrator before clearing external inputs.')
        if self.nExtInputs <= 0:
            self.extInputLens = []
            self.extInputVals = []
            self.extInputTimes = []
            self.initExtInputs = False
        elif self.initExtInputs:
            if self._integMod.ClearExtInputs()[0] != 1:
                raise ClearError('ClearExtInputs call failed!')
            self.extInputLens = []
            self.extInputVals = []
            self.extInputTimes = []
            self.canContinue = False
            self.initExtInputs = False


    def checkExtInputs(self, extInputVals, extInputTimes):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before checking external inputs. (initBasic)')

        if not isinstance(extInputVals, list):
            raise TypeError("extInputVals must be list.")
        if len(extInputVals) != self.nExtInputs:
            raise ValueError("length of extInputVals must match nExtInputs")
        if len(extInputTimes) != self.nExtInputs:
            raise ValueError("length of extInputTimes must match nExtInputs")
        if not all([isinstance(v, _real_types) for v in np.array(extInputVals).flatten()]):
            raise TypeError("extInputVals entries must be real values")
        if not all([isinstance(v, _real_types) for v in np.array(extInputTimes).flatten()]):
            raise TypeError("extInputTimes entries must be real values")

        for ti, time_array in enumerate(extInputTimes):
            if len(time_array) > 1:
                # Check orientation
                orientation = time_array[-1] - time_array[0]
                if orientation == 0:
                    raise ValueError("Each extInputTime must be distinct; first and last times are identical with len > 1")
                if orientation < 0:
                    if np.any(np.diff(time_array) >= 0):
                        print("External input #{}: {}".format(ti, time_array))
                        raise ValueError("extInputTimes must be strictly monotonic")
                if orientation > 0:
                    if np.any(np.diff(time_array) <= 0):
                        print("External input #{}: {}".format(ti, time_array))
                        raise ValueError("extInputTimes must be strictly monotonic")


    def setRunParams(self, ic=[], params=[], t0=[], tend=[], gt0=[], refine=0,
                     specTimes=[], bounds=[]):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before setting params. (initBasic)')

        #if self.initParams:
        #    raise InitError('You must clear params before setting them. Use clearParams()')
        self.checkRunParams(ic, params, t0, tend, gt0, refine, specTimes,
                               bounds)
        self.ic = ic
        self.params = params
        self.t0 = float(t0)
        self.tend = float(tend)
        self.gt0 = float(gt0)
        self.refine = int(refine)
        self.specTimes = list(specTimes)

        if self.t0 < self.tend:
            self.direction = 1
        else:
            self.direction = -1

        self.set_bounds(bounds)

        retval = self._integMod.SetRunParameters(self.ic, self.params,
                             self.gt0, self.t0, self.tend, self.refine,
                             len(self.specTimes), self.specTimes,
                             self.upperBounds, self.lowerBounds)

        if retval[0] != 1:
            raise InitError('SetRunParameters call failed!')

        self.canContinue = False
        self.setParams = True

    def set_bounds(self, bounds):
        """Sets self.upperBounds and self.lowerBounds.
        """
        def_bd = abs(float(self.defaultBound))

        if bounds != []:
            lobd = bounds[0]
            upbd = bounds[1]
            # use the zipped transpose values for easy single for loop.
            # explicit list is OK since these are relatively small objects
            bounds_T = list(zip(*bounds))
            for i, (l, u) in enumerate(bounds_T):
                if np.isinf(u):
                    upbd[i] = np.sign(u)*def_bd
                if np.isinf(l):
                    lobd[i] = np.sign(l)*def_bd
            self.upperBounds = upbd
            self.lowerBounds = lobd
        else:
            self.upperBounds = [def_bd] * (self.phaseDim + self.paramDim)
            self.lowerBounds = [-def_bd] * (self.phaseDim + self.paramDim)


    def clearRunParams(self):
        if not self.initBasic:
            raise ClearError('You must initialize the integrator before clearing params.')
        if self.setParams:
            if self._integMod.ClearParams()[0] != 1:
                raise ClearError('ClearParams call failed!')
            self.canContinue = False
            self.setParams = False


    def check_params(self, params):
        if not isinstance(params, list):
            raise TypeError("params must be list")
        if len(params) != self.paramDim:
            raise ValueError("params must have length equal to phaseDim")
        if len(params) > 0:
            for x in params:
                if not isinstance(x, _real_types):
                    raise TypeError("params entries must be real values")


    def check_bounds(self, bounds):
        """Check parameter and phase space bounds.
        """
        if not isinstance(bounds, list):
            raise TypeError("bounds must be list")
        if bounds != []:
            if len(bounds) != 2:
                raise TypeError("non-empty bounds must be a 2-list")
            for i, bd_i in enumerate(bounds):
                if type(bd_i) is not list:
                    raise TypeError("non-empty bounds must be a 2-list")
                if len(bd_i) != self.phaseDim + self.paramDim:
                    raise ValueError("bounds have incorrect size")
                if not all([isinstance(bd_ij, _real_types) for bd_ij in bd_i]):
                    # TEMP
                    print(i, bd_i)
                    raise TypeError("bounds entries must be real valued")


    def checkRunParams(self, ic, params, t0, tend, gt0, refine, specTimes,
                       bounds):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before checking run params. (initBasic)')

        if not isinstance(ic, list):
            raise TypeError("ic must be list")
        if len(ic) != self.phaseDim:
            print("IC length %i didn't match phaseDim %i %r"%(len(ic), self.phaseDim, ic))
            raise ValueError('ic must have length equal to phaseDim')
        for x in ic:
            if not isinstance(x, _real_types):
                raise TypeError("ic entries must be real values")

        self.check_params(params)
        self.check_bounds(bounds)
        verify_nonneg('refine', refine, _all_int)

        if not isinstance(t0, _real_types):
            raise TypeError("t0 must be real valued")
        if not isinstance(tend, _real_types):
            raise TypeError("tend must be real valued")
        if t0 == tend:
            raise ValueError("t0 must differ from tend")
        if t0 < tend:
            direction = 1
        else:
            direction = -1

        try:
            specTimes = list(specTimes)
        except:
            raise TypeError("specTimes must be a sequence type")

        if len(specTimes) > 0:
            if not all([isinstance(t, _real_types) for t in specTimes]):
                raise TypeError("specTimes entries must be real valued")
            if direction == 1:
                tlo = t0
                thi = tend
                if np.any(np.diff(specTimes) < 0):
                    raise ValueError("specTimes must be non-decreasing")
            else:
                tlo = tend
                thi = t0
                if np.any(np.diff(specTimes) > 0):
                    raise ValueError("specTimes must be non-increasing")
            if any([t < tlo or t > thi for t in specTimes]):
                raise ValueError("specTimes entries must be within [%.8f,%.8f]"%(t0, tend))


    def setContParams(self, tend, params, calcSpecTimes, verbose,
                      extInputChanged, extInputVals, extInputTimes, bounds):
        if self.direction > 0:
            if tend < self.tend:
                raise ValueError("new tend must be > old tend")
        if self.direction < 0:
            if tend > self.tend:
                raise ValueError("new tend must be < old tend")

        self.check_params(params)
        # in case params blank, leave alone
        if params != []:
            self.params = params

        self.check_bounds(bounds)
        self.set_bounds(bounds)

        if calcSpecTimes not in (0, 1):
            raise TypeError("calcSpecTimes must be 0 or 1")
        if calcSpecTimes == 1 and len(self.specTimes) <= 0:
            raise ValueError("calcSpecTimes cannot be 1 if specTimes is empty")

        if verbose not in (0, 1):
            raise TypeError("verbose must be 0 or 1")

        if extInputChanged:
            if self.nExtInputs <= 0:
                raise ValueError("Cannot change extInputs if nExtInputs is 0")
            if self.initExtInputs:
                self.checkExtInputs(extInputVals, extInputTimes)
                self.clearExtInputs()
                self.setExtInputs(False, extInputVals, extInputTimes)
            else:
                self.setExtInputs(True, extInputVals, extInputTimes)

        self.tend = tend

        retval = self._integMod.SetContParameters(self.tend, self.params,
                                         self.upperBounds, self.lowerBounds)
        if retval[0] != 1:
            raise InitError('Call to SetContParams failed!')


    def clearAll(self):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before clearing')
        self.clearRunParams()
        self.clearEvents()
        self.clearInteg()
        self.clearExtInputs()


    def Run(*args):
        if self.__class__==integrator:
            raise NotImplementedError("Call Run on a concrete subclass")

    def Continue(*args):
        if self.__class__==integrator:
            raise NotImplementedError("Call Continue on a concrete subclass")

    def Reset(self):
        if not self.initBasic:
            raise InitError('You must initialize the integrator before resetting')
        self._integMod.Reset()
        # What to do now?

    def Rhs(self, time, x, p):
        if self.initBasic:
            if self.nExtInputs > 0 and not self.initExtInputs:
                return None
            else:
                return self._integMod.Vfield(time, x, p)
        return None

    def Jacobian(self, time, x, p):
        if self.initBasic:
            if self.nExtInputs > 0 and not self.initExtInputs:
                return None
            if self.hasJac:
                return self._integMod.Jacobian(time, x, p)
        return None

    def JacobianP(self, time, x, p):
        if self.initBasic:
            if self.nExtInputs > 0 and not self.initExtInputs:
                return None
            if self.hasJacP:
                return self._integMod.JacobianP(time, x, p)
        return None

    def MassMatrix(self, time, x, p):
        if self.initBasic:
            if self.nExtInputs > 0 and not self.initExtInputs:
                return None
            if self.hasMass:
                return self._integMod.MassMatrix(time, x, p)
        return None

    def AuxFunc(self, time, x, p):
        if self.initBasic:
            if self.nExtInputs > 0 and not self.initExtInputs:
                return None
            if self.nAux > 0:
                return self._integMod.AuxFunc(time, x, p)
        return None

