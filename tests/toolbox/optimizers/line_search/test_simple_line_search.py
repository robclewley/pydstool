#/usr/bin/env python

# Matthieu Brucher
# Last Change : 2007-08-22 14:02

from __future__ import absolute_import

import unittest
import numpy

from numpy.testing import assert_equal
from PyDSTool.Toolbox.optimizers.line_search import SimpleLineSearch


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
