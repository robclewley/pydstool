##class Function(object):
##  def __call__(self, x):
##    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2
##
##  def gradient(self, x):
##    return numpy.array((2 * (x[0] - 2), 4 * (2 * x[1] + 4)))

import sys
sys.path.append('/home/dmitrey/scikits/openopt/scikits/openopt/solvers/optimizers')

    
from line_search import CubicInterpolationSearch
from numpy import *
from numpy.linalg import norm

def BarzilaiBorwein_nonmonotone(function, x0, df = None, maxIter = 1000):
    x0 = asfarray(x0)
    lineSearch = CubicInterpolationSearch(min_step_size = 0.0001)
    
  
    #TODO: get gradtol as self.gradtol
    gradtol = 1e-6
    

    g0 = function.gradient(x0)
    state0 = {'direction' : g0}
    x1 = lineSearch(origin = x0, state = state0, function = function)
    
    alpha_k = abs(x1 - x0).sum() / abs(g0).sum()



    #default settings
    eps = 1e-5 #must be 0 < eps << 1
    M = 8
    rho = 0.5 # must be from (0,1)
    sigma1, sigma2  = 0.015, 0.8 # must be 0 < sigma1 < sigma2 < 1
    sigma = 0.015
    #alpha_min, alpha_max = 1e-6, 1.5 # should be alpha_min <= sigma <= alpha_max
    alpha_min, alpha_max = 0.015, 1e6

    xk = x0.copy()
    last_M_iter_objFun_values = []
    
    for k in xrange(maxIter):
        gk = function.gradient(xk)
        if norm(gk) <= gradtol: break
        
        fk = function(xk)
        last_M_iter_objFun_values.append(fk)
        if len(last_M_iter_objFun_values) > M: last_M_iter_objFun_values.pop()
        maxObjFunValue = max(last_M_iter_objFun_values)
        
        if not alpha_min <= alpha_k <= alpha_max: alpha_k = sigma
        Lambda = 1.0 / alpha_k
        
        
        while True:
            if function(xk - Lambda * gk) < maxObjFunValue - rho * Lambda * dot(gk.T, gk): 
                Lambda_k = Lambda
                yk = -Lambda_k * gk
                xk += yk
                break
            sigma = (sigma1+sigma2)/2 # here alg says "choose sigma from [sigma1, sigma2]"
            Lambda *= sigma
        
        alpha_k = -dot(gk.T, yk) / (Lambda_k * dot(gk.T, gk))
        
    #state['gradient'] = gk #TODO: uncomment me!
    return xk

if __name__ == '__main__':
    class Function:
        def __call__(self, x): return ((x-arange(x.size))**6).sum()
        def gradient(self, x): return 6*(x-arange(x.size))**5
    
    x0 = sin(arange(300))
    fun = Function()
    x_opt = BarzilaiBorwein_nonmonotone(fun, x0)
    print x_opt
    print fun(x_opt)
    
