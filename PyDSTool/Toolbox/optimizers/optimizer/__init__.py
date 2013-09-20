
# Matthieu Brucher
# Last Change : 2007-07-23 10:50

"""
Module containing the core optimizers

Optimizers :
  - Optimizer
    - a skeletton for defining a custom optimizer
    - calls iterate until the criterion indicates that the optimization has converged
  - StandardOptimizer
    - takes a full step after a line search
  - StandardOptimizerModifying
    - takes a full step after a line search
    - modifies the resulting parameters
"""

from standard_optimizer import *
from standard_optimizer_modifying import *

optimizer__all__ = ['StandardOptimizer', 'StandardOptimizerModifying']

__all__ = optimizer__all__