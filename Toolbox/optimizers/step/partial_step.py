
# Matthieu Brucher
# Last Change : 2007-08-10 23:15

"""
Computes a partial step for a specific function at a specific point, acting like a decorator for other steps
"""

import random
import numpy

class PartialStep(object):
  """
  A partial step
  """
  def __init__(self, step, nb_chunks, indice = None):
    """
    Allows only a part of the set of parameters to be updated
      - step is the step that will be decorated
      - nb_chunks is the number of chunks in the parameter set
      - indice is the chunk to update each time. if None, a random indice is drawn each time
    """
    self.step = step
    self.nb_chunks = nb_chunks
    self.indice = indice

  def __call__(self, function, point, state):
    """
    Computes a step based on a function and a point
    """
    step = self.step(function, point, state)
    state['old_direction'] = step
    if self.indice is None:
      indice = random.randint(0, self.nb_chunks - 1)
    else:
     indice = self.indice
    new_step = numpy.zeros(step.shape)
    new_step.shape = (self.nb_chunks, -1)
    step.shape = (self.nb_chunks, -1)
    new_step[indice] = step[indice]

    new_step.shape = -1
    step.shape = -1
    state['direction'] = new_step
    return new_step
