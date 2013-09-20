
# Matthieu Brucher
# Last Change : 2007-08-31 14:01

"""
Computes Goldstein-Price step for a specific function at a specific point
"""

import math
from numpy import dot
from numpy.linalg import solve as n_solve
from numpy.linalg import norm, cholesky

class GoldsteinPriceStep(object):
  """
  The Goldstein-Price step
  """
  def __init__(self, mu = 0.1):
    """
    """
    self.mu = mu
    self.nu = math.cos(math.pi/2.0 - mu)

  def __call__(self, function, point, state):
    """
    Computes Goldstein-Price step 
    """
    g = function.gradient(point)
    state['gradient'] = g
    G = function.hessian(point)
    state['hessian'] = G

    isPositiveDefinite = True
    d0 = None

    try:
        L = cholesky(G)
        d0 = n_solve(L.T, n_solve(L, -g))
    except:
        isPositiveDefinite = False

    if isPositiveDefinite:
        cosTheta = dot(d0, -g) / (norm(d0) * norm(g))
        if cosTheta >= self.nu:
            step = d0
        else:
            step = -g
    else:
        step = -g

    state['direction'] = step
    return step
