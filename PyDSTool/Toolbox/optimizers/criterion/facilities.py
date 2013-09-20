
# Matthieu Brucher
# Last Change : 2007-08-24 10:05

"""
Proposes a way to create a composite criterion
"""

__all__ = ['criterion']

from criteria import IterationCriterion, RelativeValueCriterion, RelativeParametersCriterion, GradientCriterion
from composite_criteria import OrComposition

def criterion(**kwargs):
  """
  Creates a composite criterion based on the formal parameters :
    - iterations_max indicates the maximum number of iteration
    - ftol is the maximum relative change of the value function
    - xtol is the maximum relative change of the parameters
    - gtol is the maximum gradient
  """
  l = []
  if 'iterations_max' in kwargs:
    l.append(IterationCriterion(kwargs['iterations_max']))
  if 'ftol' in kwargs:
    l.append(RelativeValueCriterion(kwargs['ftol']))
  if 'xtol' in kwargs:
    l.append(RelativeParametersCriterion(kwargs['xtol']))
  if 'gtol' in kwargs:
    l.append(GradientCriterion(kwargs['gtol']))

  return OrComposition(*l)
