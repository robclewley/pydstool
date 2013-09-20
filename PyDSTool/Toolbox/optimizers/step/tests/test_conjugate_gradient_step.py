#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-29 17:14

import unittest
import numpy

from numpy.testing import *
set_package_path()

from step import CWConjugateGradientStep, DYConjugateGradientStep, DConjugateGradientStep, FRConjugateGradientStep, PRPConjugateGradientStep, FRPRPConjugateGradientStep, HZConjugateGradientStep
restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2.) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2. * (x[0] - 2), 4 * (2 * x[1] + 4)))

class test_CWConjugateGradientStep(unittest.TestCase):
  def test_call_gradient(self):
    step = CWConjugateGradientStep()
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_conjugate_gradient(self):
    step = CWConjugateGradientStep()
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.178571428571 * direction
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

class test_DYConjugateGradientStep(unittest.TestCase):
  def test_call_gradient(self):
    step = DYConjugateGradientStep()
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_conjugate_gradient(self):
    step = DYConjugateGradientStep()
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.178571428571 * direction
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

class test_DConjugateGradientStep(unittest.TestCase):
  def test_call_gradient(self):
    step = DConjugateGradientStep()
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_conjugate_gradient(self):
    step = DConjugateGradientStep()
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.178571428571 * direction
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

class test_FRConjugateGradientStep(unittest.TestCase):
  def test_call_gradient(self):
    step = FRConjugateGradientStep()
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_conjugate_gradient(self):
    step = FRConjugateGradientStep()
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.178571428571 * direction
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

class test_PRPConjugateGradientStep(unittest.TestCase):
  def test_call_gradient(self):
    step = PRPConjugateGradientStep()
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_conjugate_gradient(self):
    step = PRPConjugateGradientStep()
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.1 * direction
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

class test_FRPRPConjugateGradientStep(unittest.TestCase):
  def test_call_gradient(self):
    step = FRPRPConjugateGradientStep()
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_conjugate_gradient(self):
    step = FRPRPConjugateGradientStep()
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.178571428571 * direction
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

class test_HZConjugateGradientStep(unittest.TestCase):
  def test_call_gradient(self):
    step = HZConjugateGradientStep()
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., -16.)))

  def test_call_conjugate_gradient(self):
    step = HZConjugateGradientStep()
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    origin = 0.178571428571 * direction
    newDirection = step(function = function, point = origin, state = state)
    assert(function(origin + 0.01*newDirection) < function(origin))

if __name__ == "__main__":
  unittest.main() 
