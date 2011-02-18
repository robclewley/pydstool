#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-22 13:58

import unittest
import numpy

from numpy.testing import *
set_package_path()

from line_search import BacktrackingSearch

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2 * (x[0] - 2), 4 * (2 * x[1] + 4)))

class test_BacktrackingSearch(unittest.TestCase):

  def test_call(self):
    lineSearch = BacktrackingSearch()
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.ones((2))}
    function = Function()
    assert_array_less(lineSearch(origin = numpy.zeros((2)), state = state, function = function), numpy.ones((2)) * 0.0001)
    assert(state['alpha_step'] < 0.0001)

  def test_call_gradient_direction(self):
    lineSearch = BacktrackingSearch()
    state = {'gradient' : numpy.array((4., -8.)), 'direction' : numpy.array((4., -8.))}
    function = Function()
    x = lineSearch(origin = numpy.zeros((2)), state = state, function = function)
    assert(function(x) <= function(numpy.zeros((2))) + 0.1 * state['alpha_step'] * numpy.dot(numpy.array((4., -8.)), numpy.array((4., -8.))))
    assert(state['alpha_step'] > 0)

if __name__ == "__main__":
  unittest.main()
