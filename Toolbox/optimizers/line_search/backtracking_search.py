
# Matthieu Brucher
# Last Change : 2007-08-26 19:43

import numpy

class BacktrackingSearch(object):
  """
  The backtracking algorithm for enforcing Armijo rule
  """
  def __init__(self, rho = 0.1, alpha_step = 1., alpha_factor = 0.5, **kwargs):
    """
    Can have :
      - a coefficient for the Armijo rule (rho = 0.1)
      - an alpha factor to modulate the step (alpha_step = 1.)
      - an alpha factor < 1 that will decrease the step size until the rule is valid (alpha_factor = 0.5)
    """
    self.rho = rho
    self.stepSize = alpha_step
    self.stepFactor = alpha_factor

  def __call__(self, origin, function, state, **kwargs):
    """
    Tries to find an acceptable candidate
    """
    direction = state['direction']
    if 'initial_alpha_step' in state:
      alpha = state['initial_alpha_step']
    else:
      alpha = self.stepSize

    f1temp = function(origin)
    gradient = state['gradient']
    while(True):
      ftemp = function(origin + alpha * direction)
      #Armijo rule
      if ftemp <= f1temp + self.rho * alpha * numpy.dot(gradient, direction):
        state['alpha_step'] = alpha
        return origin + alpha * direction
      alpha = alpha * self.stepFactor
