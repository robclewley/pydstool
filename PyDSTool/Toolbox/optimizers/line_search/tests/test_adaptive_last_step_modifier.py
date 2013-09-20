#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-28 00:41

import unittest
import numpy

from numpy.testing import *
set_package_path()

from line_search import SimpleLineSearch, AdaptiveLastStepModifier

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2 * (x[0] - 2), 4 * (2 * x[1] + 4)))

class test_AdaptiveLastStepModifier(unittest.TestCase):

  def test_call(self):
    lineSearch = AdaptiveLastStepModifier(SimpleLineSearch())
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.ones((2))}
    function = Function()
    assert_array_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.ones((2)))
    assert(state['alpha_step'] == 1.)

  def test_call_twice(self):
    lineSearch = AdaptiveLastStepModifier(SimpleLineSearch())
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.array((4., -8.)), 'alpha_step' : 0.5,
             'last_gradient' : numpy.array((4., -8.)), 'last_direction' : numpy.array((4., -8.))}
    function = Function()
    assert_array_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.array((2., -4.)))
    assert(state['alpha_step'] == 0.5)

  def test_call_twice2(self):
    lineSearch = AdaptiveLastStepModifier(SimpleLineSearch())
    state = {'gradient' : numpy.array((-4., 8.)), 'direction' : numpy.ones((2)), 'alpha_step' : 0.5,
             'last_gradient' : numpy.array((4., -8.)), 'last_direction' : numpy.array((4., -8.))}
    function = Function()
    assert_array_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.array((10., 10.)))
    assert(state['alpha_step'] == 10.)

if __name__ == "__main__":
  unittest.main()
