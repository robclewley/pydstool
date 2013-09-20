#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-29 17:29

import unittest
import numpy

from numpy.testing import *
set_package_path()

from step import DFPNewtonStep

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2.) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2. * (x[0] - 2), 4 * (2 * x[1] + 4)))

  def hessian(self, x):
    return numpy.diag((2., 8.))

class test_DFPNewtonStep(unittest.TestCase):
  def test_call_DFP(self):
    step = DFPNewtonStep(numpy.eye(2, 2))
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_DFP_bis(self):
    step = DFPNewtonStep(numpy.eye(2, 2))
    function = Function()
    state = {'old_parameters' : numpy.zeros((2)), 'old_value' : function(numpy.zeros((2)))}
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.178571428571 * direction
    state['new_parameters'] = origin
    state['new_value'] = function(origin)
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

if __name__ == "__main__":
  unittest.main()