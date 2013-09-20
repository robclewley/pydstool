#!/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-10-16 20:48

"""
Class defining the Powell function
"""

import numpy
from numpy.testing import *
set_package_path()
from optimizers import criterion, step, optimizer, line_search
restore_path()


class Powell:
  """
  The Powell function
  """
  def __call__(self, x):
    """
    Get the value of the Powell function at a specific point
    """
    return (x[0] + 10* x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2 * x[2])**4 + 10 * (x[0] - x[3])**4

  def gradient(self, x):
    """
    Evaluates the gradient of the function
    """
    return numpy.array([2 * (x[0] + 10 * x[1]) + 40 * (x[0] - x[3])**3, 20 * (x[0] + 10 * x[1]) + 4 * (x[1] - x[2])**3, 10 * (x[2] - x[3]) - 8 * (x[1] - 2 * x[2])**3, 10 * (x[2] - x[3]) - 40 * (x[0] - x[3])**3], dtype = numpy.float)

  def hessian(self, x):
    """
    Evaluates the gradient of the function
    """
    print x, self(x)
    return numpy.array([[2 + 120 * (x[0] - x[3]) ** 2, 20, 0, -120 * (x[0] - x[3]) ** 2], [20, 200 + 12 * (x[1] - 2 * x[2]) ** 2, -24 * (x[1] - 2 * x[2]) ** 2, 0], [0, -24 * (x[1] - 2 * x[2]) ** 2, 10 + 48 * (x[1] - 2 * x[2]) ** 2, -10], [-120 * (x[0] - x[3]) ** 2, 0, -10, -10 + 120 * (x[0] - x[3])**2]], dtype = numpy.float)

class test_Powell(NumpyTestCase):
  """
  Global test class with the Powell function
  """
  def check_simple_newton(self):
    startPoint = numpy.empty(4, numpy.float)
    startPoint[0] = 3.
    startPoint[1] = -1.
    startPoint[2] = 0.
    startPoint[3] = 1.
    optimi = optimizer.StandardOptimizer(function = Powell(), step = step.NewtonStep(), criterion = criterion.OrComposition(criterion.RelativeValueCriterion(0.00000001), criterion.IterationCriterion(1000)), x0 = startPoint, line_search = line_search.SimpleLineSearch())
    assert_array_almost_equal(optimi.optimize(), numpy.zeros(4, numpy.float), decimal=2)

if __name__ == "__main__":
    NumpyTest().run()
