
# Matthieu Brucher
# Last Change : 2007-08-24 10:25

"""
Composite criteria allow to use several criteria together, with and/or composition
"""

import sys

if sys.version < '2.5':
  def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True

  def any(iterable):
      for element in iterable:
          if element:
              return True
      return False

class OrComposition(object):
  """
  Compose several criteria with an or rule
  """
  def __init__(self, *args, **kwargs):
    """
    Collects the different criteria
    """
    self.criteria = kwargs.values() + list(args)

  def __call__(self, state, **kwargs):
    """
    Evaluates each criterion (no lazy evaluation) and returns True if one of them is True
    """
    r = [criterion(state, **kwargs) for criterion in self.criteria]
    return any(r)

class AndComposition(object):
  """
  Compose several criteria with an and rule
  """
  def __init__(self, *args, **kwargs):
    """
    Collects the different criteria
    """
    self.criteria = kwargs.values() + list(args)

  def __call__(self, state, **kwargs):
    """
    Evaluates each criterion (no lazy evaluation) and returns True if one of them is True
    """
    r = [criterion(state, **kwargs) for criterion in self.criteria]
    return all(r)
