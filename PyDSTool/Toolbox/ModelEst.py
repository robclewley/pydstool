"""Model estimation classes for ODEs.

   Robert Clewley.
"""

from __future__ import absolute_import, print_function

# PyDSTool imports
from PyDSTool.Points import Point, Pointset
from PyDSTool.Model import Model
from PyDSTool.ModelSpec import *
from PyDSTool.common import Utility, _seq_types, metric, args
from PyDSTool.utils import intersect, remain, filteredDict
from PyDSTool.errors import *
from PyDSTool.Toolbox.ParamEst import *

# Other imports from scipy import isfinite, mean
from scipy.linalg import norm, eig, eigvals, svd, svdvals

from numpy import linspace, array, arange, zeros, sum, power, \
     swapaxes, asarray, ones, alltrue, concatenate, ravel, argmax
import math, types
from copy import copy, deepcopy

# ----------------------------------------------------------------------

# !! In development
class ModelEst(Utility):
    """General-purpose model estimation class.
    """
    _needKeys = ['manager', 'context']
    _optionalKeys = ['libraries', 'verboselevel']

    def __init__(self, **kw):
        # Model manager object
        self.manager = kw['manager']
        # Model library object
        if 'libraries' in kw:
            self.libraries = kw['libraries']
        else:
            self.libraries = None
        # ModelContext.context class
        self.context = kw['context']
        if 'verboselevel' in kw:
            self.verboselevel = kw['verboselevel']
        else:
            self.verboselevel = 1

    def fit(self, name, free_pars, parest=None):
        """Fit an individual candidate model in its context,
        peforming parameter optimization on the given free parameters.
        """
        model = self.manager[name]
        assert remain(free_pars, model.pars) == [], "Invalid free variables"
        model.set(verboselevel=self.verboselevel)
        # get initial results of context at default parameters
        # if success then done already
        if self.context.evaluate(model):
            return filteredDict(model.pars, free_pars)

        # determine parameter sensitivity of the free parameters
        # dict of par -> sensitivity
        #sens = param_sensitivity(model, self.context, free_pars)

        print("Get pre-defined tols from features somehow")
        print("Establish parameter estimation class and run it")
        #pest = LMpest(freeParams=free_pars,
        #              testModel=model,
        #              context=self.context,
        #              verbose_level=self.verboselevel)
        #return pest.run(parDict={'ftol': 1e-5,
        #                        'xtol': 1e-5,
        #                        'args': (self.context,)},
        #                verbose=self.verboselevel>0)

