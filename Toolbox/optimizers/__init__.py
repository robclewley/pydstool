
# Matthieu Brucher
# Last Change : 2007-08-24 11:16

"""
Optimization module
"""

# Needed so that default paramaters are accessible from everywhere in the submodule
def mod_path():
  import sys
  import os.path
  sys.path.append(os.path.abspath(os.path.dirname(__file__)))

mod_path()

import defaults
import criterion
import line_search
import optimizer
import step
import helpers

__all__= ['defaults', 'criterion', 'line_search', 'optimizer', 'step', 'helpers']

def test(level = -5, verbosity = 1):
  from numpy.testing import NumpyTest
  return NumpyTest().test(level, verbosity)

def testall(level = -5, verbosity = 1):
  from numpy.testing import NumpyTest
  return NumpyTest().testall(level, verbosity)
