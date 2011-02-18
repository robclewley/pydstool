#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-31 14:01

import unittest
import numpy

from numpy.testing import *
set_package_path()

from step import GoldsteinPriceStep

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2.) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2. * (x[0] - 2), 4 * (2 * x[1] + 4)))

  def hessian(self, x):
    return numpy.diag((2., 8.))

class Function2(object):
  def __call__(self, x):
    return (x[0] - 2.) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2. * (x[0] - 2), 4 * (2 * x[1] + 4)))

  def hessian(self, x):
    return numpy.diag((2., -1.))

class test_GoldsteinPriceStep(unittest.TestCase):
  def test_call_GP(self):
    step = GoldsteinPriceStep()
    state = {}
    function = Function()
    assert_almost_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((2., -2.)))

  def test_call_GP_bis(self):
    step = GoldsteinPriceStep()
    state = {}
    function = Function2()
    assert_almost_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

if __name__ == "__main__":
  unittest.main()