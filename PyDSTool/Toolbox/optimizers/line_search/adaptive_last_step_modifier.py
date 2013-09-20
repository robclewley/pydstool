
# Matthieu Brucher
# Last Change : 2007-08-28 00:34

"""
Line search decorator that overrides the default alpha_step value with a factor times the last alpha_step, the factor being a function of the current direction and the last direction
"""

import numpy

class AdaptiveLastStepModifier(object):
  """
  Overrides the default step size and replaces it with a factor times the last one
  """
  def __init__(self, line_search, **kwargs):
    """
    Needs to have :
      - the decorated line search (line_search)
    """
    self.line_search = line_search

  def __call__(self, origin, function, state, **kwargs):
    """
    Returns a good candidate
    Parameters :
      - origin is the origin of the search
      - function is the function to minimize
      - state is the state of the optimizer
    """
    if 'alpha_step' in state: # in this case, last_direction and last_gradient should be defined
      state['initial_alpha_step'] = numpy.dot(state['last_gradient'], state['last_direction']) / numpy.dot(state['gradient'], state['direction']) * state['alpha_step']
    state['last_direction'] = state['direction']
    state['last_gradient'] = state['gradient']
    return self.line_search(origin = origin, function = function, state = state, **kwargs)
