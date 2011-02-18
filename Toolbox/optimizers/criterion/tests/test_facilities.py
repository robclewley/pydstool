#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-28 14:32

import unittest
import numpy

from numpy.testing import *
set_package_path()

from criterion import criterion

restore_path()

class Function(object):
  def __init__(self, value):
    self.value = value

  def gradient(self, x):
    return numpy.ones((1,)) * self.value

class test_criterion(unittest.TestCase):
  def test_relative_value(self):
    crit = criterion(iterations_max = 1000, ftol = 0.1)
    state = {'iteration' : 1001, 'old_value' : 1., 'new_value' : 0.}
    assert(crit(state))
    state = {'iteration' : 5, 'old_value' : 1., 'new_value' : 0.}
    assert(not crit(state))
    state = {'iteration' : 5, 'old_value' : 1., 'new_value' : 0.9}
    assert(crit(state))

  def test_relative_parameters(self):
    crit = criterion(iterations_max = 1000, xtol = 0.1)
    state = {'iteration' : 1001, 'old_parameters' : numpy.ones((1,)), 'new_parameters' : numpy.zeros((1,))}
    assert(crit(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.ones((1,)), 'new_parameters' : numpy.zeros((1,))}
    assert(not crit(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.ones((1,)), 'new_parameters' : numpy.ones((1,)) * 0.9}
    assert(crit(state))

  def test_relative_gradient(self):
    crit = criterion(iterations_max = 1000, gtol = 0.1)
    state = {'iteration' : 1001, 'function' : Function(1.), 'new_parameters' : numpy.zeros((1,))}
    assert(crit(state))
    state = {'iteration' : 5, 'function' : Function(1.), 'new_parameters' : numpy.zeros((1,))}
    assert(not crit(state))
    state = {'iteration' : 5, 'function' : Function(0.09), 'new_parameters' : numpy.zeros((1,))}
    assert(crit(state))


if __name__ == "__main__":
  unittest.main()
