
# Matthieu Brucher
# Last Change : 2007-08-24 15:03

"""
Defines the defaults parameters for the generic optimizer framework
"""

__all__ = ['parameters', 'errors']

SMALL_DF = 2
SMALL_DELTA_X = 3
SMALL_DELTA_F = 4
SMALL_DELTA_X_X = 5
SMALL_DELTA_F_F = 6
FVAL_IS_ENOUGH = 10
SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON = 1000

IS_NAN_IN_X = -4
IS_LINE_SEARCH_FAILED = -5
IS_MAX_ITER_REACHED = -7
IS_MAX_CPU_TIME_REACHED = -8
IS_MAX_TIME_REACHED = -9
IS_MAX_FUN_EVALS_REACHED = -10
IS_ALL_VARS_FIXED = -11

FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON = -1000

parameters = {
              'alpha_step' : 1.,
              'ftol' : 0.001,
              'gtol' : 0.001,
              'iterations_max' : 1000,
              'min_alpha_step' : 1.,
              'xtol' : 0.001,
              }

errors = {
          SMALL_DF : "gradient norm is small enough",
          SMALL_DELTA_X : "absolute X difference is small enough",
          SMALL_DELTA_F : "absolute F(X) difference is small enough",
          SMALL_DELTA_X_X : "relative X difference is small enough",
          SMALL_DELTA_F_F : "relative F(X) difference is small enough",
          FVAL_IS_ENOUGH : "F(X) is small enough",
          SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON : "Unknown reason of convergence",

          FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON : "Unknown reason of failure",
          }