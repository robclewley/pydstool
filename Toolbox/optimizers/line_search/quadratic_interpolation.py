
# Matthieu Brucher
# Last Change : 2007-08-26 19:51

"""
Line Search with the quadratic interpolation method with the computation of the gradient of the function
"""

import math
import numpy

class QuadraticInterpolationSearch(object):
  """
  Line Search with the quadratic interpolation when the gradient of the function is provided
  """
  def __init__(self, min_alpha_step, alpha_step = 1., **kwargs):
    """
    Needs to have :
      - a minimum step size (min_alpha_step)
    Can have :
      - a step modifier, a factor to modulate the step (alpha_step = 1.)
    """
    self.minStepSize = min_alpha_step
    self.stepSize = alpha_step

  def __call__(self, origin, function, state, **kwargs):
    """
    Returns a good candidate
    Parameters :
      - origin is the origin of the search
      - function is the function to minimize
      - state is the state of the optimizer
    """
    direction = state['direction']
    ak = 0.
    if 'initial_alpha_step' in state:
      bk = state['initial_alpha_step']
    else:
      bk = self.stepSize
    v_bk = function(origin + bk * direction)

    while abs(bk - ak) > self.minStepSize:
      v_ak = function(origin + ak * direction)
      g_ak = numpy.dot(function.gradient(origin + ak * direction), direction)
      ck = ak - .5 * (ak - bk) * g_ak / (g_ak - (v_ak - v_bk) / (ak - bk))
      v_ck = function(origin + ck * direction)

      bk = ak
      ak = ck
      v_bk = v_ak
      v_ak = v_ck

    state['alpha_step'] = ck
    return origin + ck * direction
