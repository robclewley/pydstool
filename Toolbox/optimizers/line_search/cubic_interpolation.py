
# Matthieu Brucher
# Last Change : 2007-08-26 19:45

"""
Line Search with the cubic interpolation method with the computation of the gradient of the function
"""

import numpy
from numpy.linalg import norm

class CubicInterpolationSearch(object):
  """
  Line Search with the cubic interpolation when the gradient of the function is provided
  """
  def __init__(self, min_alpha_step, alpha_step = 1., grad_tolerance = 1.e-6, **kwargs):
    """
    Needs to have :
      - a minimum step size (min_alpha_step)
    Can have :
      - a step modifier, a factor to modulate the step (alpha_step = 1.)
      - the tolerance on the gradient (grad_tolerance = 1e-6)
    """
    self.minStepSize = min_alpha_step
    self.stepSize = alpha_step
    self.gradtol =  grad_tolerance

  def __call__(self, origin, function, state, **kwargs):
    """
    Returns a good candidate
    Parameters :
      - origin is the origin of the search
      - function is the function to minimize
      - state is the state of the optimizer
    """
    direction = state['direction']
    ak = 0.
    if 'initial_alpha_step' in state:
      h0 = state['initial_alpha_step']
    else:
      h0 = self.stepSize

    istop = -1

    x0 = 0

    f = lambda x : function(origin + x * direction)
    df = lambda x : numpy.dot(function.gradient(origin + x * direction), direction)

    def f_and_df(x, fv = None):
        if fv is None:
          return f(x), df(x)
        else:
          return fv, df(x)

    f0 = None#TODO: extract f0 from state
    f0, df0 = f_and_df(x0, f0)

    #TODO: rename something, step is inconvenient
    xtol = self.minStepSize

    #if norm(df0) < gradtol: return origin + x0 * direction
    h = h0

    m1 = x0
    f1 = f0
    df1 = df0

    while True:
        if df1 > 0: h = -abs(h)
        else: h = abs(h)

        while True:
            #iter += 1
            #if iter > maxiter: istop = 0; break
            m2 = m1 + h
            f2, df2 = f_and_df(m2)
            if df1 * df2 <= 0:
              break# in the origin algorithm here is '<' but I think here should be '<='
            h *= 2.0
            m1, f1, df1 = m2, f2, df2

        if h>0:
          a, b = m1, m2
        else: 
            a, b = m2, m1
            f1, f2 = f2, f1
            df1, df2 = df2, df1 # in the origin algorithm ithe line is absent but I think it should be present elseware error occures

        S = 3.0 * (f2-f1) / (b-a)
        z = S - df1 -df2
        w = numpy.sqrt(z**2 - df1*df2)
        z = 1.0 - (df2 + w + z) / (df2 - df1 + 2*w)
        m1 += (b-a)*z

        f1, df1 = f_and_df(m1)
        h /= 10.0

        if abs(m1-m2) <= xtol:
          istop = 3
        elif norm(df1) <= self.gradtol:
          istop = 2
        if istop>0:
          break

    state['alpha_step'] = m1 - x0
    return origin + m1 * direction
