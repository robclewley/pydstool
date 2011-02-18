
# Matthieu Brucher
# Last Change : 2007-08-29 15:05

"""
Computes the conjugate gradient steps for a specific function at a specific point
"""

import numpy

class ConjugateGradientStep(object):
  """
  The basic conjugate gradient step
  """
  def __init__(self, coeff_function):
    """
    Initialization of the gradient step
      - coeff_function is the function that will compute the appropriate coefficient
    """
    self.oldStep = None
    self.oldGradient = None
    self.coeff_function = coeff_function

  def __call__(self, function, point, state):
    """
    Computes a gradient step based on a function and a point
    """
    newGradient = function.gradient(point)

    if 'direction' in state:
      oldGradient = state['gradient']
      oldStep = state['direction']
      coeff = self.coeff_function(newGradient, oldGradient, oldStep)
      step = - newGradient + coeff * state['direction']
    else:
      coeff = 0
      step = - newGradient
    self.oldGradient = newGradient
    state['gradient'] = newGradient
    state['conjugate_coefficient'] = coeff
    state['direction'] = step
    return step

def CWConjugateGradientStep():
  """
  The Crowder-Wolfe or Hestenes-Stiefel conjugate gradient step
  """
  def function(newGradient, oldGradient, oldStep):
    return numpy.dot(newGradient.T, (newGradient - oldGradient)) / numpy.dot(oldStep.T, (newGradient - oldGradient))
  return ConjugateGradientStep(function)

def DConjugateGradientStep():
  """
  The Dixon conjugate gradient step
  """
  def function(newGradient, oldGradient, oldStep):
    return - numpy.dot(newGradient.T, newGradient) / numpy.dot(oldStep.T, oldGradient)
  return ConjugateGradientStep(function)

def DYConjugateGradientStep():
  """
  The Dai Yan conjugate gradient step
  Has good convergence capabilities (same as the FR-PRP gradient)
  """
  def function(newGradient, oldGradient, oldStep):
    return numpy.dot(newGradient.T, newGradient) / numpy.dot(oldStep.T, (newGradient - oldGradient))
  return ConjugateGradientStep(function)

def FRConjugateGradientStep():
  """
  The Fletcher Reeves conjugate gradient step
  Needs an exact line search for convergence or the strong Wolfe-Powell rules for an inexact line search
  """
  def function(newGradient, oldGradient, oldStep):
    return numpy.dot(newGradient.T, newGradient) / numpy.dot(oldGradient.T, oldGradient)
  return ConjugateGradientStep(function)

def PRPConjugateGradientStep():
  """
  The Polak-Ribiere-Polyak conjugate gradient step
  Can restart automatically, but needs an exact line search with a uniformely convex function to globally converge
  """
  def function(newGradient, oldGradient, oldStep):
    return numpy.dot(newGradient.T, (newGradient - oldGradient)) / numpy.dot(oldGradient.T, oldGradient)
  return ConjugateGradientStep(function)

def FRPRPConjugateGradientStep():
  """
  The Fletcher-Reeves modified Polak-Ribiere-Polyak conjugate gradient step
  Can restart automatically and has the advantages of the PRP gradient and of the FR gradient
  """
  def function(newGradient, oldGradient, oldStep):
    beta = numpy.dot(newGradient.T, (newGradient - oldGradient)) / numpy.dot(oldGradient.T, oldGradient)
    betafr = numpy.dot(newGradient.T, newGradient) / numpy.dot(oldGradient.T, oldGradient)
    if beta < -betafr:
      beta = -betafr
    elif beta > betafr:
      beta = betafr
    return beta
  return ConjugateGradientStep(function)

def HZConjugateGradientStep():
  """
  The Hager-Zhang conjugate gradient step
  Has good convergence capabilities (same as the FR-PRP gradient)
  """
  def function(newGradient, oldGradient, oldStep):
    yk = newGradient - oldGradient
    beta = numpy.dot((yk - 2*numpy.linalg.norm(yk)/numpy.dot(yk.T, oldStep) * oldStep).T, newGradient) / numpy.dot(yk.T, oldStep)
    return beta
  return ConjugateGradientStep(function)
