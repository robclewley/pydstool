#!/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-24 10:19

import unittest
import numpy
import numpy.random

import os.path

from numpy.testing import *
set_package_path()

from criterion import *
from helpers import ForwardFiniteDifferences, CenteredFiniteDifferences
from line_search import *
from optimizer import *
from step import *

restore_path()

class Function(ForwardFiniteDifferences):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

class test_ForwardFiniteDifferences(unittest.TestCase):
  def test_gradient_optimization(self):
    startPoint = numpy.zeros(2, numpy.float)
    optimi = StandardOptimizer(function = Function(),
                                         step = FRConjugateGradientStep(),
                                         criterion = criterion(iterations_max = 100, ftol = 0.0000001, gtol=0.0001),
                                         x0 = startPoint,
                                         line_search = StrongWolfePowellRule())
    assert_almost_equal(optimi.optimize(), numpy.array((2., -2)))

  def test_hessian_optimization(self):
    startPoint = numpy.zeros(2, numpy.float)
    optimi = StandardOptimizer(function = Function(),
                                         step = NewtonStep(),
                                         criterion = criterion(iterations_max = 100, ftol = 0.0000001),
                                         x0 = startPoint,
                                         line_search = SimpleLineSearch())
    assert_almost_equal(optimi.optimize(), numpy.array((2., -2)))

class Function2(CenteredFiniteDifferences):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

class test_CenteredFiniteDifferences(unittest.TestCase):
  def test_gradient_optimization(self):
    startPoint = numpy.zeros(2, numpy.float)
    optimi = StandardOptimizer(function = Function2(),
                                         step = FRConjugateGradientStep(),
                                         criterion = criterion(ftol = 0.0000001, gtol=0.0001),
                                         x0 = startPoint,
                                         line_search = StrongWolfePowellRule())
    assert_almost_equal(optimi.optimize(), numpy.array((2., -2)))

  def test_hessian_optimization(self):
    startPoint = numpy.zeros(2, numpy.float)
    optimi = StandardOptimizer(function = Function2(),
                                         step = NewtonStep(),
                                         criterion = RelativeValueCriterion(0.0000001),
                                         x0 = startPoint,
                                         line_search = SimpleLineSearch())
    assert_almost_equal(optimi.optimize(), numpy.array((2., -2)))

if __name__ == "__main__":
  unittest.main()
