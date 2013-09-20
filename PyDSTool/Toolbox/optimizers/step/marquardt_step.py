
# Matthieu Brucher
# Last Change : 2007-08-29 13:42

"""
Computes a Marquardt step for a specific function at a specific point
"""

import numpy
import numpy.linalg

class MarquardtStep(object):
  """
  The simple gradient step
  """
  def __init__(self, **kwargs):
    """
    Computes the Marquardt step
      - gamma is the initial Marquardt correction factor (gamma = 1e4)
      - c1 is the decreasing factor (c1 = 0.5)
      - c2 is the increasing factor (c2 = 2.)
    """
    self.gamma = kwargs.get('gamma', 1.e4)
    self.c1 = kwargs.get('c1', 0.5)
    self.c2 = kwargs.get('c2', 2.)

  def __call__(self, function, point, state):
    """
    Computes a gradient step based on a function and a point
    """
    if 'gamma' in state:
      gamma = state['gamma']
    else:
      gamma = self.gamma

    hessian = function.hessian(point)
    gradient = function.gradient(point)
    approx_hessian = hessian + numpy.diag([gamma] * len(gradient))
    step = (-numpy.linalg.solve(approx_hessian, gradient)).reshape(point.shape)

    f0 = function(point)
    f1 = function(point + step)
    if f1 < f0:
      gamma = self.c1 * gamma
    else:
      gamma = self.c2 * gamma

    state['hessian'] = hessian
    state['approx_hessian'] = approx_hessian
    state['gradient'] = gradient
    state['direction'] = step
    state['gamma'] = gamma
    return step
