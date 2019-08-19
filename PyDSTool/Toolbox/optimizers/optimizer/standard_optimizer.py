
# Matthieu Brucher
# Last Change : 2007-08-10 23:13


"""
A standard optimizer
"""

from . import optimizer

class StandardOptimizer(optimizer.Optimizer):
  """
  A standard optimizer, takes a step and finds the best candidate
  Must give in self.optimalPoint the optimal point after optimization
  """
  def __init__(self, **kwargs):
    """
    Needs to have :
      - an object function to optimize (function), alternatively a function ('fun'), gradient ('gradient'), ...
      - a way to get a new point, that is a step (step)
      - a criterion to stop the optimization (criterion)
      - a starting point (x0)
      - a way to find the best point on a line (lineSearch)
    Can have :
      - a step modifier, a factor to modulate the step (stepSize = 1.)
    """
    optimizer.Optimizer.__init__(self, **kwargs)
    self.stepKind = kwargs['step']
    self.optimalPoint = kwargs['x0']
    self.lineSearch = kwargs['line_search']

    self.state['new_parameters'] = self.optimalPoint
    self.state['new_value'] = self.function(self.optimalPoint)

    self.recordHistory(**self.state)


  def iterate(self, forceDir=None):
    """
    Implementation of the optimization. Does one iteration.
    (Optional) Provide known direction to overide step call in 'forceDir'.
    """
    self.state['old_parameters'] = self.optimalPoint
    self.state['old_value'] = self.state['new_value']

    if forceDir is None:
      step = self.stepKind(self.function, self.optimalPoint, state = self.state)
    else:
      self.state['direction'] = forceDir
      self.state['gradient'] = -forceDir
      step = forceDir

    self.optimalPoint = self.lineSearch(origin = self.optimalPoint,
                                        function = self.function,
                                        state = self.state)
    try:
      pest = self.function.pest
    except AttributeError:
      new_pars = self.optimalPoint
    else:
      # this would include self.optimalPoint if the lowest was found by the linesearch
      new_pars = pest.pars_dict_to_array(pest.log[pest._lowest_res_log_ix].pars)
      print("*** CHOSE pars with residual %.8f" % pest.log[pest._lowest_res_log_ix].residual_norm)
    self.state['new_parameters'] = new_pars

    self.state['new_value'] = self.function(new_pars)

    self.recordHistory(**self.state)

