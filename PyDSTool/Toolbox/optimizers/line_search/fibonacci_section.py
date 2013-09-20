
# Matthieu Brucher
# Last Change : 2007-08-26 19:48

"""
Line Search with the Fibonacci section method
"""

import math

class FibonacciSectionSearch(object):
  """
  Line Search with the Fibonacci section method, optimal section method
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

    limit = self.stepSize / self.minStepSize
    self.fibonacci = self.computeFibonacci(limit)

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

    k = 1

    uk = ak + (self.fibonacci[-2] / self.fibonacci[-1]) * (bk - ak)
    v_uk = function(origin + uk * direction)
    lk = ak + self.fibonacci[-3] / self.fibonacci[-1] * (bk - ak)
    v_lk = function(origin + lk * direction)

    while True:
      if v_uk < v_lk:
        if k > len(self.fibonacci) - 2:
          state['alpha_step'] = uk
          return origin + uk * direction
        ak = lk
        v_ak = v_lk
        lk = uk
        v_lk = v_uk
        uk = ak + self.fibonacci[-k - 1] / self.fibonacci[-k] * (bk - ak)
        v_uk = function(origin + uk * direction)
      else:
        if k > len(self.fibonacci) - 2:
          state['alpha_step'] = lk
          return origin + lk * direction
        bk = uk
        v_bk = v_uk
        uk = lk
        v_uk = v_lk
        lk = ak + self.fibonacci[-k - 2] / self.fibonacci[-k] * (bk - ak)
        v_lk = function(origin + lk * direction)
      k = k + 1

  def computeFibonacci(self, limit):
    fibonacci = [1., 1.]
    while (fibonacci[-1] < limit):
      fibonacci.append(fibonacci[-2] + fibonacci[-1])

    return fibonacci