
# Matthieu Brucher
# Last Change : 2007-08-26 19:47

"""
A damped line search
"""

class DampedLineSearch(object):
  """
  A damped line search, takes a point and a direction.
  Tests a new point for minimization, if it is greater that the current cost times (1 + error),
    the step is divided by two, until the step is too small
  """
  def __init__(self, min_alpha_step, damped_error, alpha_step = 1., **kwargs):
    """
    Needs to have :
      - a minimum step size (min_alpha_step)
      - a factor to allow the cost to rise a little bit (damped_error)
    Can have :
      - a step modifier, a factor to modulate the step (alpha_step = 1.)
    """
    self.minStepSize = min_alpha_step
    self.dampedError = damped_error
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
    if 'initial_alpha_step' in state:
      stepSize = state['initial_alpha_step']
    else:
      stepSize = self.stepSize
    currentValue = function(origin)
    optimalPoint = origin + stepSize * direction
    newValue = function(optimalPoint)

    while(newValue > currentValue * (1. + self.dampedError)):
      stepSize /= 2.
      if stepSize < self.minStepSize:
        break
      optimalPoint = origin + stepSize * direction
      newValue = function(optimalPoint)
    else:
      state['alpha_step'] = stepSize
      return optimalPoint
    state['alpha_step'] = 0.
    return origin

