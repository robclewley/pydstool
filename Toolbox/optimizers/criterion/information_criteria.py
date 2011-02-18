
# Matthieu Brucher
# Last Change : 2007-08-28 14:42

import math

class AICCriterion(object):
  """
  The Akaike information criterion
  """
  def __init__(self, ftol, iterations_max, correct_factor = 1.):
    """
    Initializes the criterion with a max number of iterations and an error fraction for the monotony test
    - ftol is the relative tolerance of the AIC criterion
    - correct_factor is the modifiying factor for the weight of the parameters size
    """
    self.error = ftol
    self.iterationMax = iterations_max
    self.correct_factor = correct_factor

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    iteration = state['iteration']
    old_value = state['old_value']
    new_value = state['new_value']
    old_parameters = state['old_parameters']
    new_parameters = state['new_parameters']
    return ((iteration > self.iterationMax) or ((self.correct_factor * len(new_parameters) + new_value) > (self.correct_factor * len(old_parameters) + old_value) * (1. + self.error)))

class ModifiedAICCriterion(object):
  """
  The Akaike information criterion with several trials authorized
  """
  def __init__(self, ftol, iterations_max, correct_factor = 1., trials = 5):
    """
    Initializes the criterion with a max number of iterations and an error fraction for the monotony test
    - ftol is the relative tolerance of the AIC criterion
    - trials indicates how many time the criterion will return false when in fact the criterion was true
    - correct_factor is the modifiying factor for the weight of the parameters size
    """
    self.error = ftol
    self.iterationMax = iterations_max
    self.trials = trials
    self.correct_factor = correct_factor

  def __call__(self, state, **kwargs):
    """
    Computes the stopping criterion
    """
    iteration = state['iteration']
    old_value = state['old_value']
    new_value = state['new_value']
    old_parameters = state['old_parameters']
    new_parameters = state['new_parameters']
    if not 'trial' in state:
      state['trial']=0
    criteria = ((self.correct_factor * len(new_parameters) + new_value) > (self.correct_factor * len(old_parameters) + old_value) * (1. + self.error))
    if criteria:
      state['trial'] += 1
    return ((iteration > self.iterationMax) or (criteria and (state['trial'] >= self.trials)))
