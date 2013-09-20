
# Matthieu Brucher
# Last Change : 2007-08-30 09:23

"""
Computes a quasi-Newton step for a specific function at a specific point
"""

import numpy
import numpy.linalg

class DFPNewtonStep(object):
  """
  The Davidson-Fletcher-Powell Quasi-Newton step
  """
  def __init__(self, hessian_approx):
    """
    Construct a DFP module
      - hessian_approx is an approximation of the hessian around the starting point
    """
    self.B0 = numpy.linalg.inv(hessian_approx)

  def __call__(self, function, point, state):
    """
    Computes a gradient step based on a function and a point
    """
    if 'Bk' not in state:
      Bk = self.B0.copy()
      gradient = function.gradient(point)
    else:
      Bk = state['Bk']
      oldParams = state['old_parameters']
      newParams = state['new_parameters']
      gradient = function.gradient(point)
      oldGradient = state['gradient']

      yk = gradient - oldGradient
      sk = newParams - oldParams
      rho = 1/numpy.dot(yk.T, sk)
      Bk = numpy.dot(numpy.eye(len(gradient), len(gradient)) - rho * numpy.dot(yk[:, numpy.newaxis], sk[numpy.newaxis,:]),  numpy.dot(Bk, numpy.eye(len(gradient), len(gradient)) - rho * numpy.dot(sk[:, numpy.newaxis], yk[numpy.newaxis,:]))) + rho * numpy.dot(yk[:, numpy.newaxis], yk[numpy.newaxis,:])

    step = -numpy.dot(Bk, gradient)

    state['Bk'] = Bk
    state['gradient'] = gradient
    state['direction'] = step
    return step

