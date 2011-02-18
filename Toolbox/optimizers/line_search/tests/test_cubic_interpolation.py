#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-22 14:01

import unittest
import numpy

from numpy.testing import *
set_package_path()

from line_search import CubicInterpolationSearch

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2 * (x[0] - 2), 4 * (2 * x[1] + 4)))

class test_CubicInterpolationSearch(unittest.TestCase):
  def test_create(self):
    lineSearch = CubicInterpolationSearch(min_alpha_step = 0.0001)
    assert_equal(lineSearch.stepSize, 1.)

  def test_call(self):
    lineSearch = CubicInterpolationSearch(min_alpha_step = 0.0001)
    state = {'direction' : numpy.ones((2))}
    function = Function()
    assert_array_less(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.ones((2)) * 0.0001)
    assert(state['alpha_step'] < 0.0001)

  def test_call_gradient_direction(self):
    lineSearch = CubicInterpolationSearch(min_alpha_step = 0.0001)
    state = {'direction' : numpy.array((4., -8.))}
    function = Function()
    assert_almost_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.array((1.0588, -2.1176)), decimal = 4)
    assert_almost_equal(state['alpha_step'], 1.0588/4, decimal = 4)

if __name__ == "__main__":
  unittest.main()
