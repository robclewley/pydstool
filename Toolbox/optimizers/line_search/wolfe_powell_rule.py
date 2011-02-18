
# Matthieu Brucher
# Last Change : 2007-08-22 14:05

import numpy

class WolfePowellRule(object):
  """
  The standard Wolfe-Powell rule for a inexact line search
  """
  def __init__(self, alpha = 1., rho = 0.1, sigma = 0.4, alpha_min = 0., alpha_max = 1., alpha_limit = 0.1, **kwargs):
    """
    Initializes the search
    Parameters :
      - alpha is the first step size that will be tried (1.)
      - rho is the rhos acceptation factor (0.1)
      - sigma is the factor for the Wolfe-Powell rule (0.4)
      - alpha_min is the inf limit of the search interval (0.)
      - alpha_max is the max limit of the search interval (1.)
      - alpha_limit is a factor so that the estimated alpha is not near the limts of the tested bracket, leading to a divergence in the algorithm (alpha_limit = 0.1)
    Those parameters should be tweaked depending on the function to optimize
    """
    self.alpha = alpha
    self.rho = rho
    self.sigma = sigma
    self.alpha_min = alpha_min
    self.alpha_max = alpha_max
    self.alpha_limit = alpha_limit

  def __call__(self, origin, function, state, **kwargs):
    """
    Tries to find an acceptable candidate
    """
    direction = state['direction']
    ak = self.alpha_min
    bk = self.alpha_max
    if 'initial_alpha_step' in state:
      alpha = state['initial_alpha_step']
    else:
      alpha = self.alpha
    gradient = state['gradient']

    f1temp = function(origin)
    f1ptemp = numpy.dot(gradient, direction)
    while(True):
      #print ak, bk, alpha
      if numpy.isnan(alpha):
        state['alpha_step'] = 0
        return origin

      ftemp = function(origin + alpha * direction)
      #First rule test
      if ftemp <= function(origin) + self.rho * alpha * numpy.dot(gradient, direction):
        fptemp = numpy.dot(function.gradient(origin + alpha * direction).T, direction)
        #Second rule, Wolfe-Powell, test
        if fptemp >= self.sigma * numpy.dot(gradient, direction):
          state['alpha_step'] = alpha
          return origin + alpha * direction
        else:
          alphap = alpha + (alpha - ak) * fptemp / (f1ptemp - fptemp)
          ak = alpha
          alpha = alphap
          f1temp = ftemp
          f1ptemp = fptemp
      else :
        bracketSize = abs(bk - ak)
        alphap = ak + (alpha - ak) / (2. * (1. + (f1temp - ftemp) / ((alpha - ak) * f1ptemp)))
        bk = alpha
        if abs(alphap - ak) < self.alpha_limit * bracketSize:
          alpha = ak + self.alpha_limit * bracketSize
        else:
          alpha = alphap
