
# Matthieu Brucher
# Last Change : 2007-08-29 11:48

"""
Restarts a conjugate gradient search by deleting the step key in the state dictionary
"""

import random
import numpy
import numpy.linalg

class RestartPeriodicallyConjugateGradientStep(object):
  """
  A step decorator that periodically deletes the direction key in the dictionary so that the CG search can be restarted
  """
  def __init__(self, step, iteration_period = 10):
    """
    Constructs the decorator
      - step is the step that will be decorated
      - iteration_period is the number of iteration between the deletion
    """
    self.step = step
    self.iteration_period = iteration_period

  def __call__(self, function, point, state):
    """
    Computes a step based on a function and a point
    """
    if (state['iteration'] % self.iteration_period == 0) & ('direction' in state):
      del state['direction']
    step = self.step(function, point, state)
    return step

class RestartNotOrthogonalConjugateGradientStep(object):
  """
  A step decorator that deletes the direction key in the dictionary if the last gradients are not orthogonal enough so that the CG search can be restarted
  """
  def __init__(self, step, orthogonal_coeff = 0.1):
    """
    Constructs the decorator
      - step is the step that will be decorated
      - orthogonal_coeff is the orthogonal limit
    """
    self.step = step
    self.orthogonal_coeff = orthogonal_coeff

  def __call__(self, function, point, state):
    """
    Computes a step based on a function and a point
    """
    newGradient = function.gradient(point)
    if 'gradient' in state:
      if abs(numpy.dot(state['gradient'], newGradient) / numpy.linalg.norm(newGradient)) > self.orthogonal_coeff:
        del state['direction']
    step = self.step(function, point, state)
    return step
