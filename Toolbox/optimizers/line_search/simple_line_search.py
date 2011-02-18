
# Matthieu Brucher
# Last Change : 2007-08-28 00:36

"""
A simple line search, in fact no searches at all
"""

class SimpleLineSearch(object):
  """
  A simple line search, takes a point, adds a step and returns it
  """
  def __init__(self, alpha_step = 1., **kwargs):
    """
    Needs to have :
      - nothing
    Can have :
      - a step modifier, a factor to modulate the step (alpha_step = 1.)
    """
    self.stepSize = alpha_step

  def __call__(self, origin, state, **kwargs):
    """
    Returns a good candidate
    Parameters :
      - origin is the origin of the search
      - state is the state of the optimizer
    """
    direction = state['direction']
    if 'initial_alpha_step' in state:
      state['alpha_step'] = state['initial_alpha_step']
    else:
      state['alpha_step'] = self.stepSize
    return origin + state['alpha_step'] * direction

