#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-22 14:01

import unittest
import numpy

from numpy.testing import *
set_package_path()

from line_search import DampedLineSearch

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2 * (x[0] - 2), 2 * (2 * x[1] + 4)))

class test_DampedLineSearch(unittest.TestCase):
  def test_create(self):
    lineSearch = DampedLineSearch(min_alpha_step = 0.0001, damped_error = 1.)
    assert_equal(lineSearch.stepSize, 1.)

  def test_call(self):
    lineSearch = DampedLineSearch(min_alpha_step = 0.0001, damped_error = 0.1)
    state = {'direction' : numpy.ones((2))}
    function = Function()
    assert_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.ones((2)) * 0.125)
    assert_equal(state['alpha_step'], 0.125)

  def test_call_damped(self):
    lineSearch = DampedLineSearch(min_alpha_step = 0.0001, damped_error = 1.)
    state = {'direction' : numpy.ones((2))}
    function = Function()
    assert_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.ones((2)))
    assert_equal(state['alpha_step'], 1.)

  def test_call_gradient_direction(self):
    lineSearch = DampedLineSearch(min_alpha_step = 0.0001, damped_error = 0.1)
    state = {'direction' : numpy.array((4, -8))}
    function = Function()
    assert_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.array((2, -4)))
    assert_equal(state['alpha_step'], 0.5)

if __name__ == "__main__":
  unittest.main()