
# Matthieu Brucher
# Last Change : 2007-08-24 10:04

"""
Module containing every criteria for converge test

Functions :
  - crietrion() creates a composite criterion

Criteria :
  - IterationCriterion
    - stops when the iteration limit is reached
  - MonotonyCriterion
    - stops when the cost rises again
  - RelativeValueCriterion
    - stops when the relative value error is below a certain level
  - AbsoluteValueCriterion
    - stops when the absolute value error is below a certain level
  - RelativeParametersCriterion
    - stops when the relative parameters error is below a certain level
  - AbsoluteParametersCriterion
    - stops when the absolute parameters error is below a certain level
  - GradientCriterion
    - stops when the gradient is below a certain level

Composite criteria :
  - OrComposition
    - returns True if one of the criteria returns True
  - AndComposition
    - returns True if all the criteria return True

Information criteria :
  - AICCriterion
    - stops when the cost function increases again
    - is dependent of the number of parameters if it changes
  - ModifiedAICCriterion
    - Identical to AICCriterion except that a number of increases are tolerated
"""

from criteria import *
from composite_criteria import *
from information_criteria import *
from facilities import *

criterion__all__ = ['IterationCriterion', 'MonotonyCriterion', 'RelativeValueCriterion', 'AbsoluteValueCriterion', 'RelativeParametersCriterion', 'AbsoluteParametersCriterion', 'GradientCriterion',
                    'OrComposition', 'AndComposition',
                    'AICCriterion', 'ModifiedAICCriterion',
                    'criterion', ]

__all__ = criterion__all__