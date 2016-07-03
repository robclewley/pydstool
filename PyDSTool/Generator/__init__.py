"""Trajectory generator classes.

   Robert Clewley, September 2005
"""
from __future__ import absolute_import

from .baseclasses import *
from .ODEsystem import *
from .Euler_ODEsystem import *
from .Vode_ODEsystem import *
from .Dopri_ODEsystem import *
from .Radau_ODEsystem import *
from .ADMC_ODEsystem import *
from .ExplicitFnGen import *
from .ImplicitFnGen import *
from .EmbeddedSysGen import *
from .LookupTable import *
from .InterpolateTable import *
from .ExtrapolateTable import *
from .MapSystem import *
import six

def findGenSubClasses(superclass):
    """Find all Generator sub-classes of a certain class, e.g. ODEsystem."""
    assert isinstance(superclass, six.string_types), \
           "findGenSubClasses requires a string as the name of the class to search for subclasses."
    subclasslist = []
    sc = eval(superclass)
    for x in theGenSpecHelper.gshDB.keys():
        if compareClassAndBases(theGenSpecHelper.gshDB[x].genClass,sc):
            subclasslist.append(x)
    return subclasslist
