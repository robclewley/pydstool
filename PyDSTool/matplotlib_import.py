"""
    Plotting imports for PyDSTool, from Matplotlib's pyplot library.

    Robert Clewley, March 2006.
"""

from __future__ import absolute_import, print_function

from numpy import Inf, NaN, isfinite, int, int8, int16, int32, int64, float, float32, float64
try:
    import matplotlib
    ver = matplotlib.__version__.split(".")
    if int(ver[0]) == 0 and int(ver[1]) < 65:
        import matplotlib.matlab as plt
        from matplotlib.matlab import *
    else:
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import *
except RuntimeError as err:
    if str(err) == 'could not open display':
        failed=True
    else:
        raise
except ImportError:
    failed=True
else:
    failed=False

if failed:
    # Dummy plot overrides for PyDSTool when matplotlib fails to import

    def plot(*args, **kw):
        print("Warning: plot does not work!")

    def save_fig(fignum, fname, formats=[]):
        print("Warning: plot does not work!")

    print("Warning: matplotlib failed to import properly and so is not")
    print("  providing a graphing interface")
    plt = None   # will cause an error if someone tries to access in order to plot
    gca = None
else:
    import os
    from .Trajectory import Trajectory
    from .common import _num_types

    # Convenient shorthand to permit singleton numeric types and Trajectories
    # in the plot arguments without first converting them to lists or arrays.
    def plot(*args, **kw):
        new_args = list(args)
        if isinstance(args[0], _num_types):
            new_args[0] = [args[0]]
        elif isinstance(args[0], Trajectory):
            try:
                new_args[0] = args[0].sample()
            except:
                raise RuntimeError("Could not sample trajectory with default "
                                   "options for plotting")
        if len(args) > 1:
            if isinstance(args[1], _num_types):
                new_args[1] = [args[1]]
            elif isinstance(args[1], Trajectory):
                try:
                    new_args[1] = args[1].sample()
                except:
                    raise RuntimeError("Could not sample trajectory with "
                                       "default options for plotting")
        return plt.plot(*tuple(new_args), **kw)

    def save_fig(fignum, fname, formats=['png','svg','eps']):
        """Save figure fignum to multiple files with different formats
        and extensions given by the formats argument.
        These are platform-dependent and are specific to matplotlib's support.
        """
        for f in formats:
            plt.figure(fignum).savefig(fname+'.'+f)

