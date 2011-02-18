
# Matthieu Brucher
# Last Change : 2007-08-22 14:04

import numpy

class GoldsteinRule(object):
  """
  The Goldstein rule for a inexact line search
  """
  def __init__(self, rho = 0.1, alpha_min = 0., alpha_max = 1., **kwargs):
    """
    Initializes the search
    Parameters :
      - rho is the factor
      - alpha_min is the inf limit of the search interval (0.)
      - alpha_max is the max limit of the search interval (1.)
    Those parameters should be tweaked depending on the function to optimize
    """
    self.rho = rho
    self.alpha_min = alpha_min
    self.alpha_max = alpha_max

  def __call__(self, origin, function, state, **kwargs):
    """
    Tries to find an acceptable candidate
    """
    direction = state['direction']
    gradient = state['gradient']
    ak = self.alpha_min
    if 'initial_alpha_step' in state:
      bk = state['initial_alpha_step']
    else:
      bk = self.alpha_max
    alpha = (ak + bk)/2.

    f1temp = function(origin)
    ft = function(origin)
    while(True):
      #print ak, bk, alpha
      if numpy.isnan(alpha):
        state['alpha_step'] = 0
        return origin

      ftemp = function(origin + alpha * direction)
      #First rule test
      if ftemp <= ft + self.rho * alpha * numpy.dot(gradient, direction):

        #Second rule test
        if ftemp >= ft + (1 - self.rho) * alpha * numpy.dot(gradient, direction):
          state['alpha_step'] = alpha
          return origin + alpha * direction
        else:
          ak = alpha
          alpha = (ak + bk)/2.
      else:
        bk = alpha
        alpha = (ak + bk)/2.
        