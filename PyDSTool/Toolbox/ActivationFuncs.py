
from PyDSTool import *
from PyDSTool.Toolbox import *
from time import perf_counter

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def Sigma(vo, theta, k):
    return 1.0 / ( 1.0 + Exp( ( vo - theta ) / k ) )

def Sigma2(vo, a, b):
    return 1.0 / ( 1.0 + Exp( -( a + b*vo  ) ) )


def SigmaNeg(vo, theta, k):
    return 1.0 / ( 1.0 + Exp( -( vo + theta ) / k ) )

def Beta(vo, theta, k, c):
    return c / ( Cosh((vo-theta)/(2*k)) )

def SigmaV(vo, a, b, c, d, e):
    return ((a*v0 - b)/(c*(1-Exp(-(d+e*v0)))))

def Gamma(vo, a, b, c):
    return (a * Exp(-(b+c*vo)))
