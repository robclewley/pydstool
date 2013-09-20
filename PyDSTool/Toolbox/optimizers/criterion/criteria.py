
# Matthieu Brucher
# Last Change : 2007-08-24 14:19

"""
A list of standard convergence criteria based on the number of iterations, the last values taken by the cost function and the associated points
"""

import math
import numpy
import defaults

class IterationCriterion(object):
  """
  A simple criterion that stops when the iteration limit is reached
  """
  def __init__(self, iterations_max):
    """
    Initializes the criterion with a max number of iterations (iterations_max)
    """
    self.iterations_max = iterations_max

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    value = (state['iteration'] > self.iterations_max)
    if value:
      state['istop'] = defaults.IS_MAX_ITER_REACHED
    return value

class MonotonyCriterion(object):
  """
  A simple criterion that stops when the values of the function starts to rise again
  """
  def __init__(self, ftol):
    """
    Initializes the criterion with an error fraction for the monotony test (ftol)
    """
    self.error = ftol

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    value = (state['new_value'] > state['old_value'] * (1. + self.error))
    if value:
      state['istop'] = defaults.SMALL_DELTA_F
    return value

class RelativeValueCriterion(object):
  """
  The relative criterion stops the optimization when the relative error of the value is below a certain level (ftol)
  """
  def __init__(self, ftol):
    """
    Initializes the criterion with an error fraction
    """
    self.error = ftol

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    old_value = state['old_value']
    new_value = state['new_value']
    value = (abs(new_value - old_value) / (new_value + old_value + 10e-127) < self.error)
    if value:
      state['istop'] = defaults.SMALL_DELTA_F_F
    return value

class AbsoluteValueCriterion(object):
  """
  The absolute criterion stops the optimization when the absolute error of the value is below a certain level (ftol)
  """
  def __init__(self, ftol):
    """
    Initializes the criterion with an error fraction
    """
    self.error = ftol

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    value = (abs(state['new_value'] - state['old_value']) < self.error)
    if value:
      state['istop'] = defaults.SMALL_DELTA_F
    return value

class RelativeParametersCriterion(object):
  """
  The relative criterion stops the optimization when the relative error of the parameters is below a certain level (xtol)
  """
  def __init__(self, xtol, weight = None):
    """
    Initializes the criterion with an error fraction and the weight assigned for each parameter
    """
    self.error = xtol
    if weight != None:
      self.weight = weight
    else:
      self.weight = 1

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    old_parameters = state['old_parameters']
    new_parameters = state['new_parameters']
    value = ((self.weight * numpy.abs(new_parameters - old_parameters) / (new_parameters + old_parameters + 10e-127)) < self.error).all()
    if value:
      state['istop'] = defaults.SMALL_DELTA_X_X
    return value

class AbsoluteParametersCriterion(object):
  """
  The absolute criterion stops the optimization when the relative error of the parameters is below a certain level (xtol)
  """
  def __init__(self, xtol, weight = None):
    """
    Initializes the criterion with an error fraction and the weight assigned for each parameter
    """
    self.error = xtol
    if weight != None:
      self.weight = weight
    else:
      self.weight = 1

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    old_parameters = state['old_parameters']
    new_parameters = state['new_parameters']
    value = ((self.weight * numpy.abs(new_parameters - old_parameters)) < self.error).all()
    if value:
      state['istop'] = defaults.SMALL_DELTA_X
    return value

class GradientCriterion(object):
  """
  The gradient criterion stops the optimization when the gradient at the current point is less that a given tolerance
  """
  def __init__(self, gtol, weight = None):
    """
    Initializes the criterion with an error fraction and the weight assigned for each parameter
    """
    self.error = gtol
    if weight != None:
      self.weight = weight
    else:
      self.weight = 1

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    gradient = state['function'].gradient(state['new_parameters'])
    value = ((self.weight * numpy.abs(gradient)) < self.error).all()
    if value:
      state['istop'] = defaults.SMALL_DF
    return value
