import numpy as np
from PyDSTool import common

class LocalBruteForce1DStep(object):
    """Local brute force search for 1D parameter (sub-)space, making no
    use of gradient information. Takes a "step" in a local neighbourhood to
    the minimum in that neighbourhood. Specify the neighbourhood by absolute
    limits as a pair or Interval object, and either the resolution (for uniform
    sampling of the interval) or the explicit sample values (as a strictly
    increasing sequence), via keyword args 'resolution' or 'samples'.

    Use in conjunction with SimpleLineSearch.
    """
    def __init__(self, interval, **kwargs):
        self.interval = interval
        if 'samples' in kwargs:
            self.samples = kwargs['samples']
            assert self.samples[0] >= self.interval[0]
            assert self.samples[-1] <= self.interval[-1]
            assert common.isincreasing(self.samples)
        elif 'resolution' in kwargs:
            numpoints = 1 + (self.interval[1]-self.interval[0])/kwargs['resolution']
            self.samples = np.linspace(self.interval[0], self.interval[1],
                                          numpoints)
        if 'smooth' in kwargs:
            self.smooth = kwargs['smooth']
        else:
            self.smooth = True
        if 'index' in kwargs:
            self.index = kwargs['index']
        else:
            self.index = 0
        if self.smooth:
            self.quadratic = common.fit_quadratic()
        else:
            # not used
            self.quadratic = None


    def __call__(self, function, point, state):
        """Assumes 1D parameter input"""
        res = []
        for pval in self.samples:
            res.append(function.residual(np.array([pval]))[self.index])
        if self.smooth:
            ixlo, ixhi = common.nearest_2n_indices(self.samples,
                                                   np.argmin(res), 2)
            smooth_res = common.smooth_pts(self.samples[ixlo:ixhi+1],
                                           res[ixlo:ixhi+1],
                                           self.quadratic)
            pmin = smooth_res.results.peak[0]
        else:
            pmin = self.samples[np.argmin(res)]
        state['direction'] = pmin - point
        return pmin
