
# Matthieu Brucher
# Last Change : 2007-08-23 10:12

import quadratic
import numpy

class LMQuadratic(quadratic.Quadratic):
  """
  Defines a cost function with a quadratic cost but the Levenberg-Marquardt approximation of the hessian
  """
  def hessian(self, params):
    """
    Compute sthe hessian of the fit function
    """
    inter = 2 * self.f.gradient(self.x, params)[..., numpy.newaxis] * self.f.gradient(self.x, params)[..., numpy.newaxis, :]
    shape = inter.shape[-2], inter.shape[-1]
    inter.shape = (-1, inter.shape[-2] * inter.shape[-1])
    temp = numpy.sum(inter, axis = 0)
    temp.shape = shape
    return temp