"""Library of common model equation primitives.

Contains functions, expressions, etc.
"""


from numpy import inf
from PyDSTool.ModelTools import ModelLibrary
import PyDSTool.ModelSpec
from PyDSTool.Symbolic import Fun

__all__ = ['threshold', 'linear_decay']

# -------------------------------------------------------------------------

threshold = ModelLibrary('threshold',
                        spectype = Fun,
                        pars={'k': [-inf,inf]},
                        indepdomain={'x': [-inf,inf]},
                        depdomain=[0,1],
                        description="""Smooth R -> [0,1] functions of
                        one parameter that represent thresholds""")
tanh_spec = Fun('1+tanh(x/k)', ['x'], 'tanh_thresh')
logistic_spec = Fun('1/(1+exp(-k*x))', ['x'], 'logistic_thresh')
alg_spec = Fun('0.5*(1+x/(sqrt(k+x*x)))', ['x'], 'algebraic_thresh')
# not yet supported
#heav_spec = Fun('if(x>=0,1,0)', ['x'], 'heav_thresh')
threshold.add_spec([tanh_spec, logistic_spec, alg_spec])


linear_decay = ModelLibrary('linear_decay',
                            spectype = Fun,
                            pars={'tau': [0,inf], 'x_inf': [-inf,inf]},
                            indepdomain={'x': [-inf,inf]},
                            depdomain=[-inf,inf],
                            description="""Linear decay given by two
                            parameters""")
linear_decay.add_spec(Fun('(x_inf-x)/tau', ['x'], 'linear'))
