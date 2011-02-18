#!/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-23 20:49

import unittest
import numpy
import numpy.random

import os.path

from numpy.testing import *
set_package_path()

from criterion import *
from helpers import Quadratic
from line_search import *
from optimizer import *
from step import *

restore_path()

class F1(object):
  def __call__(self, x, params):
    return params[0] + params[1] * x

  def gradient(self, x, params):
    return numpy.array((numpy.ones(len(x)), x)).T

  def hessian(self, x, params):
    return numpy.zeros((len(x), 2, 2))

class F2(object):
  def __call__(self, x, params):
    return params[0:2] + params[2:4] * x[:,numpy.newaxis]

  def gradient(self, x, params):
    grad = numpy.array(((numpy.ones(len(x)), numpy.zeros(len(x)), x, numpy.zeros(len(x))), (numpy.zeros(len(x)), numpy.ones(len(x)), numpy.zeros(len(x)), x)))
    return grad.transpose((2, 0, 1))

  def hessian(self, x, params):
    return numpy.zeros((len(x), 2, 4, 4))

class test_LinearFunction(unittest.TestCase):
  def setUp(self):
    self.x = numpy.random.random_sample((10))
    self.y = 3*self.x+4
    self.function = Quadratic(self.x, self.y, F1())

    self.x0 = numpy.zeros(2)

  def test_gradient_optimization(self):
    opt = StandardOptimizer(function = self.function,
                            x0 = self.x0,
                            step = GradientStep(),
                            line_search = FibonacciSectionSearch(min_alpha_step=0.000001),
                            criterion = criterion(ftol = 0.00001, iterations_max = 1000))
    optimum = opt.optimize()
    assert_array_almost_equal(optimum, numpy.array((4., 3.)))

  def test_newton_optimization(self):
    opt = StandardOptimizer(function = self.function,
                            x0 = self.x0,
                            step = NewtonStep(),
                            line_search = SimpleLineSearch(),
                            criterion = criterion(ftol = 0.00001, iterations_max = 1000))
    optimum = opt.optimize()
    assert_array_almost_equal(optimum, numpy.array((4., 3.)))

class test_LinearFunction2D(unittest.TestCase):
  def setUp(self):
    self.x = numpy.random.random_sample((10))
    self.y = numpy.array((3, -2))*self.x[:,numpy.newaxis]+4
    self.function = Quadratic(self.x, self.y, F2())

    self.x0 = numpy.zeros(4)

  def test_gradient_optimization(self):
    opt = StandardOptimizer(function = self.function,
                            x0 = self.x0,
                            step = GradientStep(),
                            line_search = FibonacciSectionSearch(min_alpha_step=0.000001),
                            criterion = criterion(ftol = 0.0001, iterations_max = 1000))
    optimum = opt.optimize()
    assert_array_almost_equal(optimum, numpy.array((4., 4., 3., -2.)))

  def test_newton_optimization(self):
    opt = StandardOptimizer(function = self.function,
                            x0 = self.x0,
                            step = NewtonStep(),
                            line_search = SimpleLineSearch(),
                            criterion = criterion(ftol = 0.0001, iterations_max = 1000))
    optimum = opt.optimize()
    assert_array_almost_equal(optimum, numpy.array((4., 4., 3., -2.)))

class Fexp(object):
  def __call__(self, x, params):
    return params[0] * numpy.exp(- params[1] * x)

  def gradient(self, x, params):
    return numpy.array((numpy.exp(- params[1] * x), - x * params[0] * numpy.exp(- params[1] * x))).T

  def hessian(self, x, params):
    return numpy.array(((numpy.zeros(len(x)), - x * numpy.exp(- params[1] * x)),
                       (- x * numpy.exp(- params[1] * x), x**2 * params[0] * numpy.exp(- params[1] * x)))).T

class test_ExponentialFunction(unittest.TestCase):
  def setUp(self):
    self.x = numpy.random.random_sample((10))
    self.y = 4*numpy.exp(- self.x*3)
    self.function = Quadratic(self.x, self.y, Fexp())

    self.x0 = numpy.ones(2)

  def test_gradient_optimization(self):
    opt = StandardOptimizer(function = self.function,
                            x0 = self.x0,
                            step = GradientStep(),
                            line_search = FibonacciSectionSearch(min_alpha_step=0.000001),
                            criterion = criterion(ftol = 0.00000001, iterations_max = 1000))
    optimum = opt.optimize()
    assert_array_almost_equal(optimum, numpy.array((4., 3.)), decimal = 3)

  def test_newton_optimization(self):
    opt = StandardOptimizer(function = self.function,
                            x0 = self.x0,
                            step = MarquardtStep(),
                            line_search = SimpleLineSearch(),
                            criterion = criterion(ftol = 0.00000001, iterations_max = 1000))
    optimum = opt.optimize()
    assert_array_almost_equal(optimum, numpy.array((4., 3.)))

if __name__ == "__main__":
  unittest.main()
