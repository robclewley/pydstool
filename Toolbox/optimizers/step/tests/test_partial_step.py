#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-28 15:01

import unittest
import numpy

from numpy.testing import *
set_package_path()

from step import PartialStep, GradientStep

restore_path()

class Function(object):
  def __call__(self, x):
    return (x[0] - 2.) ** 2 + (2 * x[1] + 4) ** 2

  def gradient(self, x):
    return numpy.array((2. * (x[0] - 2), 4 * (2 * x[1] + 4)))

class test_PartialStep(unittest.TestCase):
  def test_call_1(self):
    step = PartialStep(GradientStep(), nb_chunks = 2, indice = 0)
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((4., 0.)))

  def test_call_2(self):
    step = PartialStep(GradientStep(), nb_chunks = 2, indice = 1)
    state = {}
    function = Function()
    assert_equal(step(function = function, point = numpy.zeros((2)), state = state), numpy.array((0., -16.)))

  def test_call_random(self):
    step = PartialStep(GradientStep(), nb_chunks = 2)
    state = {}
    function = Function()
    direction = step(function = function, point = numpy.zeros((2)), state = state)
    assert((direction == numpy.array((4., 0.))).all() or (direction == numpy.array((0., -16.))).all())

if __name__ == "__main__":
  unittest.main()