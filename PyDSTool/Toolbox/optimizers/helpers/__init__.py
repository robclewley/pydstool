"""
Helper functions

Fitting functions :
  - Quadratic defines a quadratic cost

NB : the first dimension of the cost, gradient or hessian is the number of
points to fit, the second is the dimension of the point if there is one.
This leads to the fact that the gradient returns in fact the jacobian of
the function.

Finite Difference functions :
  - ForwardFiniteDifferences
  - CenteredFiniteDifferences
  - also, versions with in-built caching of previous values
"""


from .quadratic import *

from .finite_difference import *
from .levenberg_marquardt import *

helpers__all__ = ['Quadratic', 'FiniteDifferencesFunction',
                  'ForwardFiniteDifferences',
                  'CenteredFiniteDifferences', 'ForwardFiniteDifferencesCache',
                  'LMQuadratic']

__all__ = helpers__all__
