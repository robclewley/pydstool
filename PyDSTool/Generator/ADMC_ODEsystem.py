# ADMC++ ODE system
from __future__ import division, absolute_import, print_function

from .allimports import *
from PyDSTool.Generator import ODEsystem as ODEsystem
from .baseclasses import Generator, theGenSpecHelper
from PyDSTool.utils import *
from PyDSTool.common import *
from PyDSTool.integrator import integrator

# Other imports
from numpy import Inf, NaN, isfinite, sometrue, alltrue, isnan, zeros
import math, random
from copy import copy, deepcopy
import os, platform, shutil, sys, gc
import distutils
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from time import clock, sleep

# path to the installation
import PyDSTool
_pydstool_path = PyDSTool.__path__[0]

# Removed integrator subclass since this just generates code for
# ADMC++ for matlab to deal with

class ADMC_ODEsystem(ODEsystem):
    """Wrapper for code generator for ADMC++32 and Matlab.
    Uses Matlab functional specifications only."""

    def __init__(self, kw):
        """Use the nobuild key to postpone building of the library, e.g. in
        order to provide additional build options to makeLibSource and
        compileLib methods or to make changes to the C code by hand.
        No build options can be specified otherwise."""

        # Building is just doing make
        if 'nobuild' in kw:
            nobuild = kw['nobuild']
            del kw['nobuild']
        else:
            nobuild = False
        ODEsystem.__init__(self, kw)
        self._solver = None
        assert self.funcspec.targetlang == 'matlab', \
               ('Wrong target language for functional specification. '
                'matlab needed for this class')
        assert isinstance(self.funcspec, RHSfuncSpec), ('ADMC++ '
                                    'requires RHSfuncSpec type to proceed')
        assert not self.inputs, \
                        'ADMC++ does not support external inputs feature'
        self._errorcodes = {}
        self._paraminfo = {}

        self.vftype = 'vfieldts'

        # currently the final four of these params are for event handling
        # NEED TO CHECK WHICH ONES ARE SUPPORTED BY ADMC -- LOOKS LIKE EVTOLS ONLY FOR NOW
        # HACK: vftype is alg param for now, tells us whether parent class is hybridvf, vfieldts, etc.
        algparams_def = {'evtols' : 0.0001, 'vftype' : 'vfieldts'}

        # Remove this later
        for k, v in algparams_def.items():
            if k not in self.algparams:
                self.algparams[k] = v

        # verify that no additional keys are present in algparams, after
        # defaults are added above
        if len(self.algparams) != len(algparams_def):
            raise ValueError("Invalid keys present in algparams argument: " \
                     + str(remain(self.algparams.keys(),algparams_def.keys())))

        thisplatform = platform.system()

        self._compilation_tempdir = os.path.join(os.getcwd(),
                                                      "admcpp_temp")
        if not os.path.isdir(self._compilation_tempdir):
            try:
                assert not os.path.isfile(self._compilation_tempdir), \
                     "A file already exists with the same name"
                os.mkdir(self._compilation_tempdir)
            except:
                print("Could not create compilation temp directory " + \
                      self._compilation_tempdir)
                raise

        # ADMC targets must go in their own directories with appropriate names
        self._model_dir = "@"+self.name
        self._target_dir = os.path.join(self._compilation_tempdir,self._model_dir)
        # Make the target directory
        if not os.path.isdir(self._target_dir):
            try:
                assert not os.path.isfile(self._target_dir), \
                       "A file already exists with the same name"
                os.mkdir(self._target_dir)
            except:
                print("Could not creat target ADMC model directory " + \
                      self._target_dir)
                raise


        """ An ADMC model has the following files:
        vfield.m -- contains code for the RHS of the vector field
        set.m -- a generic method that overload matlab's set method; only need to insert vfield name
        get.m -- a generic method that overloads matlab's get method; only need to insert appropriate parent name
        """

        # model.m, get.m, set.m, vfield.m are minimal files required. TO DO: EVENTS
        self._model_file = self.name+".m"
        self._ic_file = self.name+"_ics.m"
        self._param_file = self.name+"_params.m"
        self._set_file = "set.m"
        self._get_file = "get.m"
        self._vfield_file = "vfield.m"
        self._events_file = "events.m"

        self._vf_filename_ext = "_"+self._model_file[:-2]

        if not nobuild:
            self.makeLibSource()
        else:
            print("Build the library using the makeLib method, or in ")
            print("stages using the makeLibSource and compileLib methods.")


    def _prepareEventSpecs(self):
        # in admc++, all events are terminal, must be 0 or 1 for direction, no delay, no tolerances, etc.
        eventDir = []

#       eventTol = []

        # convert event specs (term, active, etc.) into integparam specs
        self._eventNames = self.eventstruct.sortedEventNames()
        for evname in self._eventNames:
            ev = self.eventstruct.events[evname]
            assert isinstance(ev, MatlabEvent), ("ADMC++ can only "
                                                 "accept matlab events")

        for evname in self._eventNames:
            ev = self.eventstruct.events[evname]
            #assert ev.dircode in [-1,1], ("ADMC++ requires events to have direction -1 or 1")
            #eventDir.append(ev.dircode)

        #self.algparams['eventDir'] = eventDir
        #self.algparams['eventTol'] = eventTol
        #self.algparams['eventDelay'] = eventDelay
        #self.algparams['eventInt'] = eventInt
        #self.algparams['maxbisect'] = maxbisect
        #self.algparams['eventActive'] = eventActive
        #self.algparams['eventTerm'] = eventTerm

    def _prepareEventsFileContents(self):

        allfilestr = ""

        evname = self._eventNames
        evcount = len(evname)
        if evcount < 1:
            return allfilestr

        topstr = "function [vf_, ev_] = events(vf_, t_, x_, p_, state_)"
        commentstr = "\n% Events method for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"
        alldefines = self._prepareVfieldDefines()

        allfilestr = topstr + commentstr + alldefines

        evassign = "\tev_ = zeros(1," + str(evcount) + ");\n"
        for x in range(evcount):
            ev = self.eventstruct.events[evname[x]]
            evassign += "\tev_(" + str(x+1) + ") = " + ev.name + ev._LLargstr + ";\n"

        allfilestr += evassign + "\n\n"

        for x in range(evcount):

            ev = self.eventstruct.events[evname[x]]
            evfullfn = ""
            evsig = "function " + ev._LLreturnstr + ev.name + ev._LLargstr + "\n"
            assert ev._LLfuncstr.index(';') > 1, ("End your event function with a ';'")

            fbody =  "%BEGIN func " + ev.name + "\n" + alldefines
            fbody += ev._LLfuncstr

            if self.funcspec.auxfns:
                fbody_parsed = addArgToCalls(fbody, list(self.funcspec.auxfns.keys()), "p_")
            else:
                fbody_parsed = fbody

            evbody = "\n % Function definition\n" + fbody_parsed + "\n\n\n"
            allevs = evsig + evbody
            allfilestr += allevs

        return allfilestr

    def _prepareSetFileContents(self):
        allfilestr = ""

        topstr = "function a = set(ain, varargin)\n"
        commentstr = "% Set method for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"

        bodystr = "if nargin == 1\n" \
                  + "\t % Show input fields\n" \
                  + "\t todisp = structsub(struct(ain), ain.protectedfields{:});\n" \
                  + "\t disp(todisp);\n\t return \n end\n\n" \
                  + "a = ain;\n\n" \
                  + "if nargout < 1\n" + "\t warning('SET method invoked without output.');\n" \
                  + "\t disp(' ');\n" + "\t return\n" + "end\n\n" \
                  + "if rem(nargin-1, 2)\n" + "\t error('Wrong number of input arguments.');\n" + "end\n\n" \
                  + "args = {varargin{:}};\n\n" \
                  + "% Process input arguments\n" \
                  + "while ~isempty(args)\n" + "\t nam = args{1};\n\t val = args{2};\n\t args = args(3:end);\n\n" \
                  + "\t if any( strcmp( nam, a.privatefields ) )\n" + "\t\t warning(['Field ' nam ' is private -- unchanged.']);\n\n" \
                  + "\t elseif isfield(struct(a), nam)\n\t\t eval(['a.' name '= val;']);\n\n"

        parentfieldstr = "\t % Set parent field\n" + "\t else\n" \
                         + "\t\t a." + str(self.vftype) + " = set(a." + str(self.vftype) + ", nam, val);\n\n" \
                         + "\t end\n" + "end\n\n" + "varargout{1} = a;\n"

        allfilestr = topstr + commentstr + bodystr + parentfieldstr

        return allfilestr

    def _prepareGetFileContents(self):
        allfilestr = ""

        topstr = "function varargout = get(a, nam)\n"
        commentstr = "% Get method for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"

        bodystr = "if nargin < 2\n" \
                  + "\t % Show structure info\n" \
                  + "\t disp(struct(a));\n\t return \n end\n\n" \
                  + "if nargin > 2\n" + "\t warning('GET method invoked with more than two input arguments.');\n" \
                  + "\t disp(' ');\n" + "end\n\n" \
                  + "% Check for fields in this class; return values\n" \
                  + "if any( strcmp( nam, fieldnames(a) ) )\n" + "\t v = getfield(struct(a), nam);\n\n" \

        parentfieldstr = "% Try parent field\n" + "else\n" \
                         + "\t v = get(a." + str(self.vftype) + ", nam);\n\n" \
                         + "end\n\n" + "varargout{1} = v;\n"

        allfilestr = topstr + commentstr + bodystr + parentfieldstr

        return allfilestr


    def _prepareVfieldDefines(self):
        pardefines = ""
        vardefines = ""

        vnames = self._var_ixmap
        pnames = self.funcspec.pars
        pnames.sort()

        for i in range(self.numpars):
            p = pnames[i]
            pardefines += "\t" + p + " = p_(" + str(i+1) + ");\n"

        for i in range(self.dimension):
            v = vnames[i]
            vardefines += "\t" + v + " = x_(" + str(i+1) + ");\n"

        alldefines = "\n% Parameter definitions\n\n" + pardefines \
                     + "\n% Variable definitions\n\n" + vardefines

        allfilestr = alldefines

        return allfilestr

    def _prepareVfieldContents(self, vfdefines):
        allfilestr = ""

        topstr = "function [vf_, y_] = vfield(vf_, t_, x_, p_)\n"
        commentstr = "% Vector field definition for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"

        bodystr = vfdefines

        # Process the vector field stuff here

        # That's it, except we may need aux functions as well!
        allfilestr = topstr + commentstr + bodystr + self.funcspec.spec[0]

        return allfilestr

    def _prepareAuxContents(self):

        allfilestr = ""

        if self.funcspec.auxfns:
            for finfo in self.funcspec.auxfns.values():
                fbody = finfo[0]
                # subs _p into auxfn-to-auxfn calls (but not to the signature)
                fbody_parsed = addArgToCalls(fbody,
                                        list(self.funcspec.auxfns.keys()),
                                        "p_", notFirst=True)

                allfilestr += "\n" + fbody_parsed + "\n\n"
        # add auxiliary variables (shell of the function always present)
        # add event functions
        # allfilestr += self.funcspec.auxspec[0]

        return allfilestr



    def _prepareModelContents(self):
        allfilestr = ""
        topstr = "function a = " + self.name + "(varargin)\n"
        commentstr = "% Vf object definition for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"

        if len(self._eventNames) > 0:
            objectstr =  "\t\t vf = vfieldts('dimx'," + str(self.dimension) + ", 'eventdim', " \
                        + str(len(self._eventNames)) + ");\n\n"
        else:
            objectstr = "\t\t vf = vfieldts('dimx'," + str(self.dimension) + ");\n\n"

        bodystr = "nargs = nargin\n\n" + "switch nargs\n" \
                  + "\t case 0\n" + "\t\t a.publicfields = {};\n" \
                  + "\t\t a.protectedfields = {};\n" + "\t\t a.privatefields = {};\n\n" \
                  + objectstr \
                  + "\t\t a = class(a, '" + self.name + "', vf);\n\n" \
                  + "\t case 1\n" + "\t\t if (isa(varargin{1}, '" + self.name + "'))\n" \
                  + "\t\t\t a = varargin{1};\n" + "\t\t else\n" \
                  + "\t\t\t error('Wrong argument type');\n" + "\t\t end\n\n" \
                  + "\t otherwise\n" + "\t\t a = " + self.name + ";\n" \
                  + "\t\t a = set(a, varargin{:});\n" + "end\n"

        allfilestr = topstr + commentstr + bodystr

        return allfilestr

    def _prepareICContents(self):
        allfilestr = ""
        topstr = "function ics_ = " + self.name +"_ics()\n"
        commentstr = "% Initial conditions for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"

        bodystr = "ics_ = [ ...\n"
        if self.initialconditions:
            icnames = list(self.initialconditions.keys())
            icnames.sort()

            for i in range(len(icnames)-1):
                if isnan(self.initialconditions[icnames[i]]):
                    val = str(0.0)
                else:
                    val = str(self.initialconditions[icnames[i]])

                bodystr += val + ", ... % " + icnames[i] + "\n"

            if isnan(self.initialconditions[icnames[len(icnames)-1]]):
                val = str(0.0)
            else:
                val = self.initialconditions[icnames[len(icnames)-1]]

            bodystr += val + " % " + icnames[len(icnames)-1] + " ...\n"

        bodystr += "];\n"

        allfilestr = topstr + commentstr + bodystr

        return allfilestr

    def _prepareParamContents(self):
        allfilestr = ""
        topstr = "function pars__ = " + self.name +"_params()\n"
        commentstr = "% Parameters for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"

        bodystr = "pars_ = [ ...\n"
        if self.pars:
            pnames = list(self.pars.keys())
            pnames.sort()

            for i in range(len(pnames)-1):
                bodystr += str(self.pars[pnames[i]]) + ", ... % " + pnames[i] + "\n"

            bodystr += str(self.pars[pnames[len(pnames)-1]]) + " % " + pnames[len(pnames)-1] + " ...\n"

        bodystr += "];\n"

        allfilestr = topstr + commentstr + bodystr

        return allfilestr


    def _prepareEventFuncStrings(self, vfdefines):
        allevs = ""

        if self._eventNames == []:
            numevs = 0
        else:
            numevs = len(self._eventNames)
        for ev in self._eventNames:
            ev = self.eventstruct.events[evname]
            evfullfn = ""
            assert isinstance(ev, MatlabEvent), ("ADMC can only accept matlab events")
            evsig = "function y_ = " + ev.name + "(vf_, t_, x_, p_)\n"

            assert ev._LLfuncstr.index(';') > 1, ("Event function code error: "
                                                  "Have you included a ';' character at the end of "
                                                  "your function?")
            fbody = ev._LLfuncstr
            # NEED TO CHECK WHETHER THIS IS APPROPRIATELY DEFINED
            # check for calls to user-defined functions and add hidden p_ argument
            if self.funcspec.auxfns:
                fbody_parsed = addArgToCalls(fbody, list(self.funcspec.auxfns.keys()), "p_")
                if 'initcond' in self.funcspec.auxfns:
                    fbody_parsed = wrapArgInCall(fbody_parsed, 'initcond', ' ')

            else:
                fbody_parsed = fbody

            allevs += evsig + vfdefines + fbody_parsed + "\n\n"

            return allevs


#    def _prepareAuxFuncStrings(self, vfdefines):
#        allaux =


    def makeLib(self, libsources=[], libdirs=[], include=[]):
        """makeLib calls makeLibSource and then the compileLib method.
        To postpone compilation of the source to a DLL, call makelibsource()
        separately."""
        self.makeLibSource()


    def makeLibSource(self):
        """makeLibSource generates the MATLAB source for the vector field specification.
        It should be called only once per vector field."""

        # Make vector field (and event) file for compilation
        # This sets the field self._eventNames
        self._prepareEventSpecs()

        # Write the model.m file
        allfilestr = self._prepareModelContents()
        modelfile = os.path.join(self._target_dir, self._model_file)
        try:
            file = open(modelfile, 'w')
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file "+self._model_file+" for writing")
            raise IOError(e)

        # Write the events.m file
        if len(self._eventNames) > 0:
            allfilestr = self._prepareEventsFileContents() + self._prepareAuxContents()
            eventsfile = os.path.join(self._target_dir, self._events_file)
            try:
                file = open(eventsfile, 'w')
                file.write(allfilestr)
                file.close()
            except IOError as e:
                print("Error opening file "+self._events_file+" for writing")
                raise IOError(e)


        # Write the initialconditions.m file
        allfilestr = self._prepareICContents()
        icfile = os.path.join(self._target_dir, self._ic_file)
        try:
            file = open(icfile, 'w')
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file "+self._ic_file+" for writing")
            raise IOError(e)

        # Write the pars.m file
        allfilestr = self._prepareParamContents()
        paramfile = os.path.join(self._target_dir, self._param_file)
        try:
            file = open(paramfile, 'w')
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file "+self._param_file+" for writing")
            raise IOError(e)

        # Write the get.m file
        allfilestr = self._prepareGetFileContents()
        getfile = os.path.join(self._target_dir, self._get_file)
        try:
            file = open(getfile, 'w')
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file "+self._get_file+" for writing")
            raise IOError(e)

        # Write the set.m file
        allfilestr = self._prepareSetFileContents()
        setfile = os.path.join(self._target_dir, self._set_file)
        try:
            file = open(setfile, 'w')
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file "+self._set_file+" for writing")
            raise IOError(e)

        # Write the vfield.m file
#        vfdefines = self._prepareVfieldDefines()
#        allfilestr = self._prepareVfieldContents(vfdefines)
        allfilestr = self.funcspec.spec[0] + self._prepareAuxContents()
        vffile = os.path.join(self._target_dir, self._vfield_file)
        try:
            file = open(vffile, 'w')
            file.write(allfilestr)
            file.close()
        except IOError as e:
            print("Error opening file "+self._vfield_file+" for writing")
            raise IOError(e)


    # We have omitted methods: RHS, compute, etc. because this
    # class is intended solely to generate code for ADMC++, not do any integrations
    # etc.

    def __del__(self):
        ODEsystem.__del__(self)


# Register this Generator with the database

symbolMapDict = {}
# in future, provide appropriate mappings for libraries math,
# random, etc. (for now it's left to FuncSpec)
theGenSpecHelper.add(ADMC_ODEsystem, symbolMapDict, 'matlab')
