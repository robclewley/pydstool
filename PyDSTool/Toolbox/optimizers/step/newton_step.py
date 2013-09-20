
# Matthieu Brucher
# Last Change : 2007-08-29 15:48

"""
Computes a Newton step for a specific function at a specific point
"""

import numpy
import numpy.linalg

class NewtonStep(object):
  """
  The Newton step
  """
  def __call__(self, function, point, state):
    """
    Computes a Newton step based on a function and a point
    """
    hessian = function.hessian(point)
    gradient = function.gradient(point)
    step = (-numpy.linalg.solve(hessian, gradient)).reshape(point.shape)
    state['hessian'] = hessian
    state['gradient'] = gradient
    state['direction'] = step
    return step
