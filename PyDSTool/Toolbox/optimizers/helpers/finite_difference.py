from __future__ import division, absolute_import

from PyDSTool import common
import numpy as np

class FiniteDifferencesFunction(object):
    def residual(p, extra_args=None):
        raise NotImplementedError("Define in concrete sub-class")

class ForwardFiniteDifferences(FiniteDifferencesFunction):
    """
    A function that will be able to computes its derivatives with a forward difference formula
    """
    def __call__(self, params):
        return np.linalg.norm(self.residual(params))

    def __init__(self, eps=1e-7, *args, **kwargs):
        """
        Creates the function :
        - eps is the amount of difference that will be used in the computations
        """
        self.eps = eps
        self.inveps = 1 / eps

    def gradient(self, params):
        """
        Computes the gradient of the function
        """
        grad = np.empty(params.shape)
        curValue = self(params)
        for i in range(0, len(params)):
            #inveps = self.inveps[i]
            paramsb = params.copy()
            paramsb[i] += self.eps
            grad[i] = (self(paramsb) - curValue) / self.eps
        return grad

    def hessian(self, params):
        """
        Computes the hessian of the function
        """
        hess = np.empty((len(params), len(params)))
        curGrad = self.gradient(params)
        for i in range(0, len(params)):
            #inveps = self.inveps[i]
            paramsb = params.copy()
            paramsb[i] -= self.eps
            hess[i] = - (self.gradient(paramsb) - curGrad) / self.eps
        return hess

    def hessianvect(self, params):
        """
        Computes the hessian times a vector
        """
        raise NotImplementedError


class CenteredFiniteDifferences(FiniteDifferencesFunction):
    """
    A function that will be able to computes its derivatives with a centered difference formula
    """
    def __init__(self, eps=1e-7, *args, **kwargs):
        """
        Creates the function :
        - eps is the amount of difference that will be used in the computations
        """
        self.eps = eps # see the way this is used differently to inveps in gradient() below
        self.inveps = 1 / (2 * eps)

    def gradient(self, params):
        """
        Computes the gradient of the function
        """
        grad = np.empty(params.shape)
        for i in range(0, len(params)):
            paramsa = params.copy()
            paramsb = params.copy()
            paramsa[i] -= self.eps
            paramsb[i] += self.eps
            grad[i] = self.inveps * (self(paramsb) - self(paramsa))
        return grad

    def hessian(self, params):
        """
        Computes the hessian of the function
        """
        hess = np.empty((len(params), len(params)))
        for i in range(0, len(params)):
            paramsa = params.copy()
            paramsb = params.copy()
            paramsa[i] -= self.eps
            paramsb[i] += self.eps
            hess[i] = self.inveps * (self.gradient(paramsb) - self.gradient(paramsa))
        return hess

    def hessianvect(self, params):
        """
        Computes the hessian times a vector
        """
        raise NotImplementedError



class FiniteDifferencesCache(FiniteDifferencesFunction):
    """General class for recognition by ParamEst as a function
    with a non-explicit derivative. Uses a cache to save recomputation
    of most recent values.
    """
    def __call__(self, params):
        return np.linalg.norm(self.residual(params))

    def __init__(self, eps=1e-7, *args, **kwargs):
        """
        Creates the function :
        - eps is the scale at which the function varies by O(1) in each
        parameter direction
        - grad_ratio_tol (optional, default = 10) is the relative change in any direction
        after which the function is deemed to have changed non-smoothly, so that gradient in
        that direction will be ignored this step
        """
        self.eps = eps
        try:
            self.pest = kwargs['pest']
        except KeyError:
            # must set it using pest.setFn
            self.pest = None
        try:
            self.grad_ratio_tol = kwargs['grad_ratio_tol']
        except KeyError:
            self.grad_ratio_tol = 10

    def residual(self, p, extra_args=None):
        # tuple(p) works for multi-parameter otherwise falls through to just p
        try:
            pars = tuple(p)
        except TypeError:
            pars = (p,)
        try:
            r = self.pest.key_logged_residual(pars, self.pest.context.weights)
        except KeyError:
            r = self._res_fn(p, extra_args)
        return r

    def _res_fn(self, p, extra_args=None):
        raise NotImplementedError("Define in concrete sub-class")


class ForwardFiniteDifferencesCache(FiniteDifferencesCache):
    """
    A function that will be able to computes its derivatives with a
    forward difference formula.
    """
    def jacobian(self, params, extra_args=None):
        """
        Computes the jacobian of the function
        """
        return common.diff2(self.residual, params, eps=self.eps)

    def gradient(self, params):
        """
        Computes the gradient of the function
        """
        # turn the matrix into a flat array
        #return (common.diff2(self.__call__, params, eps=self.eps)).A1
        grad = np.empty(params.shape)
        res_x = self.residual(params)  # will look up cache if available
        for i in range(0, len(params)):
            eps = self.eps[i]
            paramsa = params.copy()
            paramsa[i] += eps
            res_a = self.residual(paramsa)
            # filter out steps that caused non-smooth changes in gradient (as per relative change)
            # i.e. make those directions count as zero
            filt = (abs(res_x / res_a) < self.grad_ratio_tol).astype(int)
            grad[i] = (np.linalg.norm(res_a*filt)-np.linalg.norm(res_x*filt))/eps
        return grad

    def hessian(self, params):
        """
        Computes the hessian of the function approximated by
        J^T * T
        """
        J = self.jacobian(params)
        return J.T * J
