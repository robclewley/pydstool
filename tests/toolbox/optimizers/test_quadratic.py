#!/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-24 11:20

"""
Class defining a quadratic function
"""

from __future__ import absolute_import

import numpy
from numpy.testing import assert_almost_equal
from PyDSTool.Toolbox.optimizers import criterion, step, optimizer, line_search

class Quadratic:
  """
  A simple quadratic function
  """
  def __call__(self, x):
    """
    Get the value of the quadratic function at a specific point
    """
    return (x[0] + 2* x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

  def gradient(self, x):
    """
    Evaluates the gradient of the function
    """
    return numpy.array([2 * (x[0] + 2* x[1] - 7) + 4 * (2 * x[0] + x[1] - 5), 4 * (x[0] + 2* x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)], dtype = numpy.float)

  def hessian(self, x):
    """
    Evaluates the gradient of the function
    """
    return numpy.array([[10, 8], [8, 10]], dtype = numpy.float)

def test_simple_gradient():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.GradientStep(), criterion = criterion.RelativeValueCriterion(0.00001), x0 = startPoint, line_search = line_search.SimpleLineSearch(alpha_step = 0.001))
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float))

def test_simple_newton():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.NewtonStep(), criterion = criterion.RelativeValueCriterion(0.00001), x0 = startPoint, line_search = line_search.SimpleLineSearch())
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float))

def test_damped_gradient_relative():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.GradientStep(), criterion = criterion.RelativeValueCriterion(0.00001), x0 = startPoint, line_search = line_search.DampedLineSearch(damped_error = 0.001, min_alpha_step = 0.0001))
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float))

def test_swp_frgradient_relative():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.FRConjugateGradientStep(), criterion = criterion.criterion(ftol=0.000001, iterations_max=1000, gtol = 0.0001), x0 = startPoint, line_search = line_search.StrongWolfePowellRule())
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float))

def test_swp_cwgradient_relative():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.CWConjugateGradientStep(), criterion = criterion.criterion(ftol=0.0000001, iterations_max=1000, gtol = 0.0001), x0 = startPoint, line_search = line_search.StrongWolfePowellRule())
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float), decimal=4)

def test_swp_prpgradient_relative():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.PRPConjugateGradientStep(), criterion = criterion.criterion(ftol=0.000001, iterations_max=1000, gtol = 0.0001), x0 = startPoint, line_search = line_search.StrongWolfePowellRule())
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float))

def test_swp_dpgradient_relative():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.DConjugateGradientStep(), criterion = criterion.criterion(ftol=0.000001, iterations_max=1000, gtol = 0.0001), x0 = startPoint, line_search = line_search.StrongWolfePowellRule())
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float))

def test_swp_dypgradient_relative():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.DYConjugateGradientStep(), criterion = criterion.criterion(ftol=0.000001, iterations_max=1000, gtol = 0.0001), x0 = startPoint, line_search = line_search.StrongWolfePowellRule())
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float))

def test_simple_marquardt():
  startPoint = numpy.zeros(2, numpy.float)
  optimi = optimizer.StandardOptimizer(function = Quadratic(), step = step.MarquardtStep(), criterion = criterion.criterion(gtol=0.00001, iterations_max=200), x0 = startPoint, line_search = line_search.BacktrackingSearch())
  opt = optimi.optimize()
  assert_almost_equal(optimi.optimize(), numpy.array([1, 3], dtype = numpy.float), decimal = 5)
