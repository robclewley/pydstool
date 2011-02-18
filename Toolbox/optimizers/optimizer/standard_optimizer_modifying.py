
# Matthieu Brucher
# Last Change : 2007-08-10 23:10

"""
A standard optimizer with a special object that modifies the resulting set of parameters
"""

import optimizer

class StandardOptimizerModifying(optimizer.Optimizer):
  """
  A standard optimizer, takes a step and finds the best candidate
  Must give in self.optimalPoint the optimal point after optimization
  After each iteration the resulting optimization point is modified by a call to a function
  """
  def __init__(self, **kwargs):
    """
    Needs to have :
      - an object function to optimize (function), alternatively a function ('fun'), gradient ('gradient'), ...
      - a way to get a new point, that is a step (step)
      - a criterion to stop the optimization (criterion)
      - a starting point (x0)
      - a way to find the best point on a line (line_search)
    Can have :
      - a step modifier, a factor to modulate the step (step_size = 1.)
      - a pre-modifier, that acts on the set of parameters before an iteration (pre_modifier)
      - a post-modifier, that acts on the set of parameters after an iteration (post_modifier)
    """
    optimizer.Optimizer.__init__(self, **kwargs)
    self.stepKind = kwargs['step']
    self.optimalPoint = kwargs['x0']
    self.lineSearch = kwargs['line_search']
    self.pre_modifier = kwargs.get('pre_modifier')
    self.post_modifier = kwargs.get('post_modifier')

    self.state['new_parameters'] = self.optimalPoint
    self.state['new_value'] = self.function(self.optimalPoint)

    self.recordHistory(**self.state)

  def iterate(self):
    """
    Implementation of the optimization. Does one iteration
    """
    self.state['unmodified_old_parameters'] = self.optimalPoint
    self.state['unmodified_old_value'] = self.state['new_value']

    if self.pre_modifier:
      self.optimalPoint = self.pre_modifier(self.optimalPoint)
    self.state['old_parameters'] = self.optimalPoint
    self.state['old_value'] = self.function(self.optimalPoint)

    direction = self.stepKind(self.function, self.optimalPoint, state = self.state)

    self.optimalPoint = self.lineSearch(origin = self.optimalPoint, function = self.function, state = self.state)

    self.state['unmodified_new_parameters'] = self.optimalPoint
    self.state['unmodified_new_value'] = self.function(self.optimalPoint)

    if self.post_modifier:
      self.optimalPoint = self.post_modifier(self.optimalPoint)
    self.state['new_parameters'] = self.optimalPoint
    self.state['new_value'] = self.function(self.optimalPoint)

    self.recordHistory(**self.state)

