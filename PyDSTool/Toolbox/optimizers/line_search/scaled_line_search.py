
import numpy as np


class ScaledLineSearch(object):
    """
    A simple line search, takes a point, adds a step and returns it
    Scales step according to given scales of the parameters and ignores
    *magnitude* of gradient.

    (in early development and experimental only at this point)
    """
    def __init__(self, max_step=1, step_mod=3, **kwargs):
        """
        Needs to have :
        - nothing
        Can have :
        - max_step: a maximum step control, a scalar or vector to restrict step size
        in each direction (default 1)
        - step_mod: a factor to divide the step when back-tracking (default 3)
        - max_reduce_fac: max_step divided by this is the smallest step that will be tried (default 2000),
        """
        self.maxStepSize = max_step
        self.stepMod = step_mod
        if np.isscalar(max_step):
            self.basis = None
            self.dim = None
        else:
            self.dim = len(max_step)
            self.basis = np.identity(self.dim)
        try:
            self.filter = kwargs['filter']
        except KeyError:
            self.filter = False
        try:
            self.maxReduceFac = kwargs['max_reduce_fac']
        except KeyError:
            self.maxReduceFac = 7 #5000
#        try:
#            self.use_dirs = kwargs['use_directions']
#        except KeyError:
#            self.use_dirs = np.zeros((self.dim,),float)


    def __call__(self, origin, state, **kwargs):
        """
        Returns a good candidate
        Parameters :
        - origin is the origin of the search
        - state is the state of the optimizer
        """
        fun = kwargs['function']
        d = state['direction']/np.linalg.norm(state['direction'])
        # filter directions that are too large
        if self.filter:
            ndabs_log = -np.log10(np.abs(d))
            mean_log = np.mean(ndabs_log)
            #print "\n ** MEAN =", mean_log
            direction = (ndabs_log > mean_log-1.5).astype(int)*d
        else:
            direction = d
        state['direction'] = direction
##        for pos, d in enumerate(direction):
##            use_dir = self.use_dirs[pos]
##            if use_dir * d < 0:
##                # directions don't match so don't move in this direction
##                direction[pos] = 0
        maxStepSize = self.maxStepSize
        if np.isscalar(maxStepSize):
            stepSize = maxStepSize
        else:
            stepfacs = np.zeros(self.dim)
            for d in range(self.dim):
                # explicit loop so as to catch any ZeroDivisionErrors
                try:
                    stepfacs[d] = abs(maxStepSize[d] / direction[d])
                except ZeroDivisionError:
                    # Direction is orthogonal to this parameter direction,
                    # so ensure won't choose this as the minimum step size
                    stepfacs[d] = Inf
            # Stop stepping with giant sizes if direction vector has strong
            # separation of scales
            stepSize = min(stepfacs)
#            print "direction = ", direction
#            print "step = ", step
        i = 1
        old_value = state['old_value']
        not_done = True
#        print "** TEMP: Hardwiring step size to be 0.0005"
#        stepSize = 0.0005
        init_step = stepSize
        while not_done:
            print("\nLinestep: i =", i, "step size =", stepSize, "direction =\n", end='')
            print(direction)
            p = origin + i * stepSize * direction
            print("Testing p = ", p)
            new_value = fun(p)
            if new_value < old_value:
                i += 1
                old_value = new_value
            else:
                if i == 1:
                    # don't shrink step size to be less than 1/maxReduceFac of initial
                    if stepSize*self.maxReduceFac < init_step:
                        not_done = False
                        p = origin + (i-1) * stepSize * direction
                    else:
                        stepSize /= self.stepMod
                else:
                    # had found a working step but it's no longer stepping to lower residuals
                    not_done = False
                    p = origin + (i-1) * stepSize * direction
        state['alpha_step'] = stepSize
        return p
