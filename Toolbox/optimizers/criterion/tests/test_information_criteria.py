#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-28 14:55

import unittest
import numpy

from numpy.testing import *
set_package_path()

from criterion import AICCriterion, ModifiedAICCriterion

restore_path()

class test_AICCriterion(unittest.TestCase):
  def test_call(self):
    criterion = AICCriterion(ftol = 0.1, iterations_max = 1000)
    state = {'iteration' : 1001, 'old_value' : 0, 'new_value' : 1., 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_value' : 10., 'new_value' : 0., 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2.5, 2.5, 2.5))}
    assert(not criterion(state))
    state = {'iteration' : 5, 'old_value' : 0., 'new_value' : 1., 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_value' : 0., 'new_value' : 0., 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2., 2.))}
    assert(criterion(state))

class test_ModifiedAICCriterion(unittest.TestCase):
  def test_call(self):
    criterion = ModifiedAICCriterion(ftol = 0.1, iterations_max = 1000)
    state = {'iteration' : 1001, 'old_value' : 0, 'new_value' : 1., 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_value' : 0., 'new_value' : 1., 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(not criterion(state))
    assert(state['trial'] == 1)
    state = {'iteration' : 5, 'old_value' : 0., 'new_value' : 1., 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.)), 'trial' : 5.}
    assert(criterion(state))

if __name__ == "__main__":
  unittest.main()
