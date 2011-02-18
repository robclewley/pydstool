
# Matthieu Brucher
# Last Change : 2007-12-18 09:47

"""
An hyperbolic line search, in fact no searches at all
"""

class HyperbolicLineSearch(object):
  """
  An inverse line search, takes a point, adds a step (1/(1+iterations)) and returns it
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
      alpha = state['initial_alpha_step'] /( 1 + state['iteration'])
      state['alpha_step'] = alpha
    else:
      alpha = self.stepSize /( 1 + state['iteration'])
      state['alpha_step'] = alpha
    return origin + alpha * direction

