
# Matthieu Brucher
# Last Change : 2007-08-28 00:34

"""
Line search decorator that overrides the default alpha_step value with a factor times the last alpha_step
"""

class FixedLastStepModifier(object):
  """
  Overrides the default step size and replaces it with a factor times the last one
  """
  def __init__(self, line_search, alpha_factor = 2., **kwargs):
    """
    Needs to have :
      - the decorated line search (line_search)
    Can have :
      - an alpha modifier, a factor to modulate the last step length (alpha_factor = 2.)
    """
    self.line_search = line_search
    self.alpha_factor = alpha_factor

  def __call__(self, origin, function, state, **kwargs):
    """
    Returns a good candidate
    Parameters :
      - origin is the origin of the search
      - function is the function to minimize
      - state is the state of the optimizer
    """
    if 'alpha_step' in state:
      state['initial_alpha_step'] = self.alpha_factor * state['alpha_step']
    return self.line_search(origin = origin, function = function, state = state, **kwargs)
