#!/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-24 10:59

"""
Class defining the Rosenbrock function
"""

import numpy
from numpy.testing import *
set_package_path()
from optimizers import criterion, step, optimizer, line_search
restore_path()

class Rosenbrock:
  """
  The Rosenbrock function
  """
  def __init__(self, dimension):
    """
    Constructor
    """
    self.dimension = dimension

  def __call__(self, x):
    """
    Get the value of the Rosenbrock function at a specific point
    """
    return numpy.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1. - x[:-1])**2.0)

  def gradient(self, x):
    """
    Evaluates the gradient of the function
    """
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = numpy.zeros(x.shape, x.dtype)
    der[1:-1] = 200. * (xm - xm_m1**2.) - 400. * (xm_p1 - xm**2.) * xm - 2. * (1. - xm)
    der[0] = -400. * x[0] * (x[1] - x[0]**2.) - 2. * (1. - x[0])
    der[-1] = 200. * (x[-1] - x[-2]**2.)
    return der

  def hessian(self, x):
    """
    Evaluates the gradient of the function
    """
    H = numpy.diag(-400. * x[:-1], 1) - numpy.diag(400. * x[:-1],-1)
    diagonal = numpy.zeros(len(x), x.dtype)
    diagonal[0] = 1200. * x[0]**2. - 400. * x[1] + 2.
    diagonal[-1] = 200.
    diagonal[1:-1] = 202. + 1200. * x[1:-1]**2. - 400. * x[2:]
    H += numpy.diag(diagonal)
    return H

class test_Rosenbrock(NumpyTestCase):
  """
  Global test class with the Rosenbrock function
  """
  def check_simple_gradient_monotony(self):
    startPoint = numpy.empty(2, numpy.float)
    startPoint[0] = -1.01
    startPoint[-1] = 1.01
    optimi = optimizer.StandardOptimizer(function = Rosenbrock(2), step = step.GradientStep(), criterion = criterion.OrComposition(criterion.MonotonyCriterion(0.00001), criterion.IterationCriterion(10000)), x0 = startPoint, line_search = line_search.SimpleLineSearch(alpha_step = 0.001))
    assert_almost_equal(optimi.optimize(), numpy.ones(2, numpy.float), decimal=1)

  def check_simple_gradient_relative(self):
    startPoint = numpy.empty(2, numpy.float)
    startPoint[0] = -1.01
    startPoint[-1] = 1.01
    optimi = optimizer.StandardOptimizer(function = Rosenbrock(2), step = step.GradientStep(), criterion = criterion.RelativeValueCriterion(0.00001), x0 = startPoint, line_search = line_search.SimpleLineSearch(alpha_step = 0.001))
    assert_almost_equal(optimi.optimize(), numpy.ones(2, numpy.float), decimal=1)

  def check_simple_newton_relative(self):
    startPoint = numpy.empty(2, numpy.float)
    startPoint[0] = -1.01
    startPoint[-1] = 1.01
    optimi = optimizer.StandardOptimizer(function = Rosenbrock(2), step = step.NewtonStep(), criterion = criterion.RelativeValueCriterion(0.00001), x0 = startPoint, line_search = line_search.SimpleLineSearch())
    assert_almost_equal(optimi.optimize(), numpy.ones(2, numpy.float))

  def check_wpr_cwgradient(self):
    startPoint = numpy.empty(2, numpy.float)
    startPoint[0] = -1.01
    startPoint[-1] = 1.01
    optimi = optimizer.StandardOptimizer(function = Rosenbrock(2), step = step.CWConjugateGradientStep(), criterion = criterion.criterion(iterations_max = 1000, ftol = 0.00000001, gtol = 0.0001), x0 = startPoint, line_search = line_search.WolfePowellRule())
    assert_array_almost_equal(optimi.optimize(), numpy.ones(2, numpy.float))

  def check_swpr_dygradient(self):
    startPoint = numpy.empty(2, numpy.float)
    startPoint[0] = -1.01
    startPoint[-1] = 1.01
    optimi = optimizer.StandardOptimizer(function = Rosenbrock(2), step = step.DYConjugateGradientStep(), criterion = criterion.criterion(iterations_max = 1000, ftol = 0.00000001, gtol = 0.0001), x0 = startPoint, line_search = line_search.StrongWolfePowellRule())
    assert_array_almost_equal(optimi.optimize(), numpy.ones(2, numpy.float), decimal = 4)

if __name__ == "__main__":
  NumpyTest().run()
