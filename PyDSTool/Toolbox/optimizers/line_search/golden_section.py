
# Matthieu Brucher
# Last Change : 2007-08-26 19:50

"""
Line Search with the golden section method
"""

import math

class GoldenSectionSearch(object):
  """
  Line Search with the golden section method
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
    self.goldenNumber = (math.sqrt(5) - 1) / 2.

  def __call__(self, origin, function, state, **kwargs):
    """
    Returns a good candidate
    Parameters :
      - origin is the origin of the search
      - function is the function to minimize
      - state is the state of the optimizer
    """
    direction = state['direction']
    ak = 0
    v_ak = function(origin + ak * direction)
    if 'initial_alpha_step' in state:
      bk = state['initial_alpha_step']
    else:
      bk = self.stepSize
    v_bk = function(origin + bk * direction)

    uk = ak + self.goldenNumber * (bk - ak)
    v_uk = function(origin + uk * direction)
    lk = ak + (1 - self.goldenNumber) * (bk - ak)
    v_lk = function(origin + lk * direction)

    while True:
      if v_uk < v_lk:
        if (bk - lk) < self.minStepSize:
          state['alpha_step'] = uk
          return origin + uk * direction
        ak = lk
        v_ak = v_lk
        lk = uk
        v_lk = v_uk
        uk = ak + self.goldenNumber * (bk - ak)
        v_uk = function(origin + uk * direction)
      else:
        if (uk - ak) < self.minStepSize:
          state['alpha_step'] = lk
          return origin + lk * direction
        bk = uk
        v_bk = v_uk
        uk = lk
        v_uk = v_lk
        lk = ak + (1 - self.goldenNumber) * (bk - ak)
        v_lk = function(origin + lk * direction)
