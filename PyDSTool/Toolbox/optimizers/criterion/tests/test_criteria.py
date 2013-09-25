#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-23 09:45

import unittest
import numpy

from numpy.testing import *
set_package_path()

from criterion import IterationCriterion, MonotonyCriterion, RelativeValueCriterion, AbsoluteValueCriterion, RelativeParametersCriterion, AbsoluteParametersCriterion

restore_path()

class test_IterationCriterion(unittest.TestCase):
  def test_call(self):
    criterion = IterationCriterion(1000)
    state = {'iteration' : 1001, 'old_value' : 0, 'new_value' : 1.}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_value' : 0, 'new_value' : 0.}
    assert(not criterion(state))

class test_MonotonyCriterion(unittest.TestCase):
  def test_call(self):
    criterion = MonotonyCriterion(0.1)
    state = {'iteration' : 5, 'old_value' : 0, 'new_value' : 1.}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_value' : 0, 'new_value' : 0.}
    assert(not criterion(state))

class test_RelativeValueCriterion(unittest.TestCase):
  def test_call(self):
    criterion = RelativeValueCriterion(0.1)
    state = {'iteration' : 5, 'old_value' : 2., 'new_value' : 2.}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_value' : 2., 'new_value' : 2.5}
    assert(not criterion(state))

class test_AbsoluteValueCriterion(unittest.TestCase):
  def test_call(self):
    criterion = AbsoluteValueCriterion(0.1)
    state = {'iteration' : 5, 'old_value' : 2., 'new_value' : 2.}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_value' : 2., 'new_value' : 2.1}
    assert(not criterion(state))

class test_RelativeParametersCriterion(unittest.TestCase):
  def test_call(self):
    criterion = RelativeParametersCriterion(0.1)
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2.5, 2.5, 2.5))}
    assert(not criterion(state))

  def test_call_weight(self):
    criterion = RelativeParametersCriterion(0.1, weight = numpy.array((0., 1., 0.)))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((0., 2., 20.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2.5, 2.))}
    assert(not criterion(state))

class test_AbsoluteParametersCriterion(unittest.TestCase):
  def test_call(self):
    criterion = AbsoluteParametersCriterion(0.1)
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2.1, 2.1, 2.1))}
    assert(not criterion(state))

  def test_call_weight(self):
    criterion = AbsoluteParametersCriterion(0.1, weight = numpy.array((0., 1., 0.)))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2., 2.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((0., 2., 20.))}
    assert(criterion(state))
    state = {'iteration' : 5, 'old_parameters' : numpy.array((2., 2., 2.)), 'new_parameters' : numpy.array((2., 2.1, 2.))}
    assert(not criterion(state))

if __name__ == "__main__":
  unittest.main()
