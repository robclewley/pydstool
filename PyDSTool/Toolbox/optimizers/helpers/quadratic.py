
# Matthieu Brucher
# Last Change : 2007-07-20 15:04

import numpy.linalg

class Quadratic(object):
  """
  Defines a cost function with a quadratic cost
  """
  def __init__(self, x, y, f):
    """
    Creates the function :
    - x is the parameters for the function
    - y is the data to approximate
    - f is an object function defining the gradient and the Hessian if needed
      It takes two parameters when called, the values x where the function should be computed and an array params that contains the parameters. The gradient of the function is the gradient for the parameters
    """
    self.x = x
    self.y = y
    self.f = f

  def __call__(self, params):
    """
    Computes the cost for the specified parameters
    """
    return numpy.sum((self.y-self.f(self.x, params))**2)

  def gradient(self, params):
    """
    Computes the gradient of the function
    """
    inter = -2 * self.f.gradient(self.x, params) * (self.y-self.f(self.x, params))[..., numpy.newaxis]
    inter.shape=(-1, inter.shape[-1])
    return numpy.sum(inter, axis = 0)

  def hessian(self, params):
    """
    Compute sthe hessian of the fit function
    """
    inter = -2 * self.f.hessian(self.x, params) * (self.y-self.f(self.x, params))[..., numpy.newaxis, numpy.newaxis] + 2 * self.f.gradient(self.x, params)[..., numpy.newaxis] * self.f.gradient(self.x, params)[..., numpy.newaxis, :]
    shape = inter.shape[-2], inter.shape[-1]
    inter.shape = (-1, inter.shape[-2] * inter.shape[-1])
    temp = numpy.sum(inter, axis = 0)
    temp.shape = shape
    return temp