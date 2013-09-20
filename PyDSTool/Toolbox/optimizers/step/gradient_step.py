
# Matthieu Brucher
# Last Change : 2007-08-10 23:15

"""
Computes a gradient step for a specific function at a specific point
"""

class GradientStep(object):
  """
  The simple gradient step
  """
  def __call__(self, function, point, state):
    """
    Computes a gradient step based on a function and a point
    """
    gradient = function.gradient(point)
    state['gradient'] = gradient
    step = - gradient
    state['direction'] = step
    return step
