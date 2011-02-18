#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-28 00:24

import unittest
import numpy

from numpy.testing import *
set_package_path()

from line_search import SimpleLineSearch, FixedLastStepModifier

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2 * (x[0] - 2), 4 * (2 * x[1] + 4)))

class test_FixedLastStepModifier(unittest.TestCase):

  def test_call(self):
    lineSearch = FixedLastStepModifier(SimpleLineSearch())
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.ones((2))}
    function = Function()
    assert_array_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.ones((2)))
    assert(state['alpha_step'] == 1.)

  def test_call_twice(self):
    lineSearch = FixedLastStepModifier(SimpleLineSearch())
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.array((4., -8.)), 'alpha_step' : 0.5}
    function = Function()
    assert_array_equal(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.array((4., -8.)))
    assert(state['alpha_step'] == 1.)

if __name__ == "__main__":
  unittest.main()
