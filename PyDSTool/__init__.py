"""PyDSTool initialization script.

print PyDSTool.__LICENSE__    for the terms of use.
"""


__LICENSE__ = """\
Copyright (C) 2007-2020, Copyright (C) 2007-2020, Robert Clewley
All rights reserved.

Parts of this distribution that originate from different authors are
individually marked as such. Copyright and licensing of those parts remains
with the original authors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of Robert Clewley, or of his affiliations may not be used to
      endorse or promote products derived from this software without specific
      prior written permission.

THIS SOFTWARE IS PROVIDED BY ROBERT CLEWLEY ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ROBERT CLEWLEY BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys, os, gc

try:
    import scipy
except ImportError:
    raise RuntimeError("SciPy is required")

import math, random
import types, time


# PyDSTool imports
from .Events import *
from .Interval import *
from .Points import *
from .Variable import *
from .Trajectory import *
from .FuncSpec import *
# \begin{hacksRus}
from . import Generator as GenModule
from .Generator import Generator as Generator_
from .Generator import *
Generator = GenModule
from . import Model as ModelModule
from .Model import Model as Model_
from .Model import *
Model = ModelModule
# \end{hacksRus}
from .ModelTools import *
from .Toolbox.ParamEst import ParamEst
from .Toolbox.ModelEst import ModelEst
from .ModelContext import *
from .Symbolic import *
from .ModelSpec import *
from .parseUtils import auxfnDBclass, protected_allnames, protected_auxnamesDB, \
         convertPowers
from .PyCont import *
import numpy as np
import scipy as sp
from math import *
try:
    from matplotlib import pyplot as plt
    from matplotlib.pyplot import figure, plot, show, draw, hold
except ImportError:
    pass

from numpy import *

from copy import copy

# note that the names with leading underscores will not be exported by
# "from PyDSTool import *"
# diff overwrites numpy diff
from .common import Continuous, Discrete, targetLangs, _seq_types, \
              _num_types, _int_types, _float_types, _complex_types, \
              _real_types, _all_numpy_int, _all_numpy_float, \
              _all_numpy_complex, _all_int, _all_float, _all_complex, \
              LargestInt32, diff, diff2
from scipy import who as scipy_who
from .utils import *

_pyDSToolTypes = [ndarray, Generator_, Variable, Trajectory, Event,
                EventStruct, Point, Pointset, Interval, ParamEst,
                Model_, Quantity, ModelSpec, QuantSpec, ModelEst,
                ModelConstructor, auxfnDBclass, nameResolverClass,
                ContClass, GeneratorConstructor, condition, feature,
                GeneratorInterface, ModelInterface, ModelManager,
                ModelTransform, GDescriptor, MDescriptor, context,
                ModelLibrary]


def who(typelist=None, objdict=None, verboselevel=0, returnlevel=0,
        deepSearch=False, _localCall=False):
    """Information about the PyDSTool user-created objects of types
    specified by typelist (defaults to all PyDSTool types and
    numpy arrays), from the objdict dictionary (or from globals() if this
    is not given).

    returnlevel > 0 puts who() into silent mode, and it just returns
    either (1) a list of the objects found, or (2) a dictionary of object
    names ->  objects found.

    deepSearch = True causes a shallow search of list, tuple and dictionary
    objects defined at the topmost level of objdict. Otherwise only
    PyDSTool objects visible at the topmost level will be seen.
    """
    objdict_out = {}   # for returning if returnlevel > 0
    if objdict is None:
        if _localCall:
            # e.g. from saveSession or restart, then need to look one
            # stack frame further back.
            frame = sys._getframe().f_back.f_back
        else:
            frame = sys._getframe().f_back
        objdict = frame.f_globals
    # Generator_ is an alias for PyDSTool.Generator.baseclasses.Generator
    # Model_ is an alias for PyDSTool.Model.Model
    if typelist is None:
        typelist_actual = _pyDSToolTypes
    elif isinstance(typelist, list):
        # hacks for taking care of class naming problems
        typelist_actual = []
        for t in typelist:
            if t == Model:
                # user meant Model.Model, i.e. local name Model_
                typelist_actual.append(Model_)
            elif t == Generator:
                # user meant Generator.Generator, i.e. local name Generator_
                typelist_actual.append(Generator_)
            elif t == array:
                # user meant numpy.ndarray
                typelist_actual.append(numpy.ndarray)
            else:
                typelist_actual.append(t)
                #if compareClassAndBases(t, _pyDSToolTypes):
                #    typelist_actual.append(t)
                #else:
                #    raise TypeError("Invalid PyDSTool object types passed")
    else:
        # hacks for taking care of class naming problems
        # when typelist is a singleton (type)
        if typelist == Model:
            # user meant Model.Model, i.e. local name Model_
            typelist_actual = [Model_]
        elif typelist == Generator:
            # user meant Generator.Generator, i.e. local name Generator_
            typelist_actual = [Generator_]
        elif typelist == array:
            typelist_actual = [numpy.ndarray]
        else:
            typelist_actual = [typelist]
        #elif compareClassAndBases(typelist, _pyDSToolTypes):
        #    typelist_actual = [typelist]
        #else:
        #    raise TypeError("Invalid PyDSTool object types passed")
    for objname, obj in objdict.items():
        if not isinstance(obj, (type, types.ModuleType)):
            if compareClassAndBases(obj, typelist_actual):
                if isinstance(obj, QuantSpec) and objname in protected_allnames:
                    # don't display internally-created QuantSpecs (i.e. all
                    # of the wrappers of the math functions and constants)
                    continue
                objdict_out[objname] = obj
            elif deepSearch:
                if isinstance(obj, (list, tuple)):
                    if any([compareClassAndBases(x, typelist_actual) \
                                 for x in obj]):
                        objdict_out[objname] = obj
                elif isinstance(obj, dict):
                    if any([compareClassAndBases(x, typelist_actual) \
                                 for x in obj.values()]):
                        objdict_out[objname] = obj
    if returnlevel == 1:
        # silent mode -- just return the objects
        return list(objdict_out.values())
    elif returnlevel == 2:
        # silent mode -- return the objects mapped by their names
        return objdict_out
    else:
        for objname, obj in objdict_out.items():
            # make appropriate separation between output items
            if verboselevel > 0:
                print("\n"*(verboselevel-1))
            if hasattr(obj, '_infostr') and not isinstance(obj, type):
                try:
                    print(objname + ": " + obj._infostr(verboselevel))
                except:
                    print("Problem with: %s, %s, %s" %(objname, className(obj), obj.info))
                    raise
            else:
                print(objname + " (Class " + className(obj) + ")" \
                      + (verboselevel > 0)*":")
                if verboselevel > 0:
                    info(obj, objname, recurseDepthLimit=verboselevel-1)


__session_ext = 'ses'
__symbolic_ext = 'sym'

def saveSession(sessionName=None, force=False, silent=False, deepSearch=False):
    if sessionName is None:
        datestr = time.strftime("%Y %m %d _ %Hh%Mm").replace(" ","")[2:]
        sessionName = "Session_" + datestr
    objdict = who(returnlevel=2, _localCall=True, deepSearch=deepSearch)
    objnamelist, objlist = sortedDictLists(objdict, byvalue=False)
    # objnamelist stores the original names of the objects saved, for use
    # when restoring the session
    objlist.append(objnamelist)
    saveObjects(objlist, sessionName+'.'+__session_ext, force)
    if not silent:
        print("Important!")
        print("If you used any user-defined classes for ModelSpec, these need to ")
        print("be recreated by running their definition scripts when you restore ")
        print("the session. saveSession only saves class _instances_.")
    #Symbolic.saveDiffs(sessionName+'.'+__symbolic_ext)


def loadSession(sessionName, tolocals=False):
    """Use tolocals boolean option if loading a session into the local
    namespace of the caller (i.e. if calling this from within a function
    rather than interactively at the prompt)"""

    sessionName_split = sessionName.split('.')
    if sessionName_split[-1] != __session_ext:
        sessionName = sessionName + '.' + __session_ext
    try:
        loadlist = loadObjects(sessionName)
    except:
        print("Problem loading session " + sessionName)
        raise
    numobjs = len(loadlist) - 1   # last entry is obj name list
    if len(loadlist) <= 0:
        raise ValueError("Session was empty!")
    objnamelist = loadlist[-1]
    objlist = loadlist[:-1]
    frame = sys._getframe().f_back
    if tolocals:
        nspace = frame.f_locals
    else:
        nspace = frame.f_globals
    # bind the original session names for the objects to the objects
    try:
        for i in range(numobjs):
            nspace[objnamelist[i]] = objlist[i]
    except:
        print("Problem recreating objects")
        print("Debug info: %d, %d, %d" % (len(objnamelist), len(objlist), numobjs))
        raise
    # load any symbolic derivatives previously auto-saved
    #symbolic.loadDiffs(sessionName+'.'+__symbolic_ext)


def restart(delall=0):
    """restart clears out global databases of PyDSTool objects,
    and with the optional argument delall=1 will delete all PyDSTool
    objects found at the top-level of the caller's namespace (not including
    numpy arrays).

    delall=2 will cause a one-level deeper search of lists, tuples, and
    dictionaries for PyDSTool objects, and the lists etc. will be deleted.
    Additionally, numpy arrays will be deleted.
    """
    nameResolver.clearall()
    genDB.clearall()
    protected_auxnamesDB.clearall()
    if delall>0:
        deep = delall==2
        objdict = who(returnlevel=2, _localCall=True, deepSearch=deep)
        frame = sys._getframe().f_back
        nspace = frame.f_globals
        for objname, obj in objdict.items():
            if objname not in ['nameResolver', 'protected_auxnamesDB'] and \
               (not isinstance(obj, ndarray) or delall==2):
                # don't delete those types of global objects
                del nspace[objname]
        del objdict, frame, nspace
        gc.collect()
