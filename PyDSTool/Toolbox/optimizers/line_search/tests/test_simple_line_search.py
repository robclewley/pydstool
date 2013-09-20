#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-22 14:02

import unittest
import numpy

from numpy.testing import *
set_package_path()

from line_search import SimpleLineSearch

restore_path()

class test_SimpleLineSearch(unittest.TestCase):
  def test_create(self):
    lineSearch = SimpleLineSearch()
    assert_equal(lineSearch.stepSize, 1.)

  def test_call(self):
    lineSearch = SimpleLineSearch(alpha_step = 0.1)
    state = {'direction' : numpy.ones((5))}
    assert_equal(lineSearch(origin = numpy.zeros((5)), state = state), numpy.ones((5)) * 0.1)
    assert_equal(state['alpha_step'], 0.1)

if __name__ == "__main__":
  unittest.main()