
"""
Computes Goldfeld step for a specific function at a specific point
"""
from numpy import pi,  dot, cos
from numpy.linalg import solve as n_solve
from numpy.linalg import norm, cholesky, eigvalsh

class GoldfeldStep(object):
    
  """
  The Goldfeld step
  """
  def __call__(self, function, point, state):
    """
    Computes Goldfeld step 
    """
    g = function.gradient(point)
    state['gradient'] = g
    G = function.hessian(point)
    state['hessian'] = G
    c = 1e-8 # is this one best?
    
    
    
    d0 = None
    
    try:
        L = cholesky(G)
        # reach here => isPositiveDefinite = True
        step = n_solve(L.T, n_solve(L, -g))
    except:
        # isPositiveDefinite = False
        G_eigvals = eigvalsh(G)
        minEig = min(G_eigvals)
        if minEig < 0:
            shift = -minEig + c
            
            #avoiding sparse case with big nVars
            for i in xrange(point):  G[i,i] += shift
                
        step = n_solve(G, -g)
 
    
    state['direction'] = step
    return step
