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

def BarzilaiBorwein(function, x0, df = None, maxIter = 1000):
    x0 = asfarray(x0)
    lineSearch = CubicInterpolationSearch(min_step_size = 0.0001)
    g0 = function.gradient(x0)
    
    #if norm(g0) <= self.gradtol: return x0
    if norm(g0) <= 1e-6: return x0
    
    state0 = {'direction' : g0}

    x1 = lineSearch(origin = x0, state = state0, function = function)
    print x1
    s0 = x1 - x0
    
    y0 = state['gradient'] - g0
    
    #if norm(state['gradient']) <= self.gradtol: return newX
    if norm(state['gradient']) <= 1e-6: return x1
    
    xk = x1
    sk_ = s0
    yk_ = y0
    gk_ = state['gradient']
    
    for k in xrange(maxIter):
        alpha_k = dot(sk_, sk_) / dot(sk_,yk_)
        sk_ = -alpha_k * gk_
        xk += sk_
        gk_prev = gk_.copy()
        gk_ = function.gradient(xk)
        yk_ = gk_ - gk_prev
        #if norm(gk_) < self.gradtol: return xk
        if norm(gk_) <= 1e-6: 
            print 'k=', k
            return xk
    return xk

if __name__ == '__main__':
    class Function:
        def __call__(self, x): return ((x-arange(x.size))**2).sum()
        def gradient(self, x): return 2*(x-arange(x.size))
    
    x0 = sin(arange(1000))
    fun = Function()
    x_opt = BarzilaiBorwein(fun, x0)
    print x_opt
    print fun(x_opt)
    
