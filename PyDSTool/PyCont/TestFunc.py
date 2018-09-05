""" Test functions

    Drew LaMar, March 2006
"""

# ----------------------------------------------------------------------------
# Version History:
#   February 2007
#       Works with SciPy 0.5.2
#
#   September 2006
#       Added Hopf_Eig test function
#       Added UserDefinedTestFunc
#
#   April 2006
#       TestFunc now stores curve class C.
#       Added DiscreteMap and FixedPointMap classes.
#       Added LPC_Det, NS_Det, PD_Det test functions for FP-C
#
#    May 2012
#		Added AddTestFunction_FixedPoint and LPC_Bor
#
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

from .misc import *
from PyDSTool.common import args, copy

from numpy.linalg import cond
from scipy import optimize, linalg
from numpy import dot as matrixmultiply
from numpy import array, float, complex, int, float64, complex64, int32, \
     zeros, divide, subtract, any, argsort, product, Inf, NaN, isfinite, \
     r_, c_, sign, mod, subtract, divide, transpose, eye, real, imag, \
     conjugate, shape, reshape, sqrt, random, spacing
from numpy.random import random

#####
_classes = ['Function', 'TestFunc', 'AddTestFunction', 'AddTestFunction_FixedPoint', 'DiscreteMap', 'FixedPointMap',
            'BiAltMethod', 'BorderMethod', 'B_Check', 'Fold_Tan',
            'Fold_Det', 'Fold_Bor', 'Branch_Det', 'Branch_Bor', 'Hopf_Det', 'Hopf_Bor', 'Hopf_Eig',
            'Hopf_Double_Bor_One', 'Hopf_Double_Bor_Two', 'BT_Fold', 'BT_Hopf', 'BT_Hopf_One',
            'CP_Fold', 'CP_Fold2', 'BP_Fold', 'DH_Hopf', 'GH_Hopf', 'GH_Hopf_One', 'LPC_Det', 'LPC_Bor', 'PD_Det', 'PD_Bor', 'NS_Det',
            'NS_Bor', 'ParTestFunc', 'UserDefinedTestFunc', 'AddTestFunction_FixedPoint_Mult']

__all__ = _classes
#####

class Function(object):
    """ F: R^n --> R^m

    Note: The function func needs to act on arrays, not numbers. Thus, in the case where n=1, define
    the function as acting on a 1 dimensional array, i.e. reference x[0] NOT x.  This is so numjac works
    without having to test cases (most generality)."""

    def __init__(self, dims, func=None, save=False, numpoints=None):
        (n, m) = dims
        if not hasattr(self, 'data'):
            self.data = None

        if not hasattr(self, 'n'):
            self.n, self.m = n, m

        if save and not hasattr(self, 'vals'):
            self.numpoints = numpoints

            if isinstance(m, int):
                self.vals = zeros((numpoints, m), float)
            else:
                self.vals = zeros((numpoints,) + m, float)

        if not hasattr(self, 'func') or self.func is None:
            self.func = func

        self.lastval = None
        self.jac = self.diff
        self.hess = self.numhess

    def __getitem__(self, loc):
        return self.vals[loc]

    def __setitem__(self, loc, vals):
        self.vals[loc] = vals

    def __call__(self, *cargs):
        self.lastval = self.func(*cargs)
        return self.lastval

    def findzero(self, X):
        return optimize.fsolve(self.func, X)

    def diff(self, x0, ind=None):
        eps = 1e-6
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        J = zeros((self.m, n), float)
        for i in range(n):
            ei = zeros(self.n, float)
            ei[ind[i]] = 1.0
            J[:,i] = (self.func(x0+eps*ei)-self.func(x0-eps*ei))/(2*eps)
        return J


    def numhess(self, x0, ind1=None, ind2=None):
        """Computes second derivative using 2nd order centered finite difference scheme.
        MAKE MORE EFFICIENT IN FUTURE (i.e. when an index is in both ind1 and ind2)

        Thus, for F: R^n ---> R^m, H is an (m,n1,n2) matrix, where n1, n2 are subsets of [1,...,n]
        """
        eps = 1e-3
        try:
            n1 = len(ind1)
        except:
            n1 = self.n
            ind1 = list(range(n1))

        try:
            n2 = len(ind2)
        except:
            n2 = self.n
            ind2 = list(range(n2))

        H = zeros((self.m,n1,n2), float)
        for i in range(n1):
            ei = zeros(self.n, float)
            ei[ind1[i]] = 1.0
            for j in range(n2):
                ej = zeros(self.n, float)
                ej[ind2[j]] = 1.0
                if ind1[i] == ind2[j]:
                    H[:,i,j] = (-1*self.func(x0+2*eps*ei) + 16*self.func(x0+eps*ei) - 30*self.func(x0) + \
                        16*self.func(x0-eps*ei) - self.func(x0-2*eps*ei))/(12*eps*eps)
                else:
                    H[:,i,j] = (self.func(x0+eps*(ei+ej)) - self.func(x0+eps*(ei-ej)) - \
                        self.func(x0+eps*(ej-ei)) + self.func(x0-eps*(ei+ej)))/(4*eps*eps)
        return H


class TestFunc(Function):
    """You need to define the function yourself within an inherited class."""
    def __init__(self, dims, F, C, save=False, numpoints=None):
        Function.__init__(self, dims, func=None, save=save, numpoints=numpoints)

        if not hasattr(self, "F"):
            self.F = F

        if not hasattr(self, "C"):
            self.C = C

    def findzero(self, P1, P2, ind):
        """Uses secant method to find zero of test function."""

        X1, V1 = P1
        X2, V2 = P2

        Z1 = copy(X1)
        Z2 = copy(X2)

        self.C._preTestFunc(X1, V1)
        T1 = self.func(X1, V1)[ind]
        # print 'X1 = ', repr(X1)
        # print 'T1 = ', repr(T1)

        self.C._preTestFunc(X2, V2)
        T2 = self.func(X2, V2)[ind]
        # print 'X2 = ', repr(X2)
        # print 'T2 = ', repr(T2)

        Tmax = 10*max(abs(T1),abs(T2))
        p = 1
        for i in range(self.C.MaxTestIters):
            if (Tmax < Inf) and (abs(T1-T2) > spacing(1)):
                r = pow(abs(T1/(T1-T2)),p)
                if r >= 1:
                    r = 0.5
            else:
                r = 0.5

            X = X1 + r*(X2-X1)
            V = V1 + r*(V2-V1)

            self.C.Corrector(X,V)

            self.C._preTestFunc(X, V)
            T = self.func(X, V)[ind]
            # print 'X = ', repr(X)
            # print 'T = ', repr(T)

            if abs(T) < self.C.TestTol and min(linalg.norm(X-X1),linalg.norm(X-X2)) < self.C.VarTol:
                break
            elif abs(T) > Tmax:
                print('Test function going crazy: ', self, '\n')
                break
            else:
                if sign(T) == sign(T2):
                    X2 = X
                    V2 = V
                    T2 = T
                    p = 1.02
                else:
                    X1 = X
                    V1 = V
                    T1 = T
                    p = 0.98

        if self.C.verbosity >= 2 and i == self.C.MaxTestIters-1:
            print('Maximum test function iterations reached.\n')

        return X, V

    def diff(self, x0, ind=None):
        """This is recreated from Function class above for TestFunc class.  Why, you ask?  Because for
        test functions, the jacobian of the parent function F needs to be computed before every test
        function call.  Unforunately, this slows things down, but that is unavoidable."""
        eps = 1e-6
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        J = zeros((self.m, n), float)
        for i in range(n):
            ei = zeros(self.n, float)
            ei[ind[i]] = 1.0

            self.C._preTestFunc(x0+eps*ei, None)
            F1 = self.func(x0+eps*ei, None)

            self.C._preTestFunc(x0-eps*ei, None)
            F2 = self.func(x0-eps*ei, None)

            J[:,i] = (F1-F2)/(2*eps)
        return J


class BorderMethod(TestFunc):
    """Border method:

            [A   B][V] = [0]
            [C^T D][G]   [1]

        where r = corank and:
            A = (n,m)
            s = max(n,m)
            p = s-m+r, q = s-n+r
            B = (n,p)
            C = (m,q)
            D = 0_(q,p)
            V = (m,q)
            G = (p,q)
            0 = (n,q)
            1 = (q,q)

        F:R^nm --> R^pq,  F(A) = G

        It can also be written as:

            [W^T G][A   B] = [0 1]
                   [C^T D]

        where now:
            W = (n,p)
            0 = (p,m)
            1 = (p,p)

        This is important for calculating derivatives:

            S_{z} = -W^T x A_{z} x V

        This is implemented in the diff method.

        In order to use the Function class, the matrices are put into vector form.

    """

    def __init__(self, dims1, dims2, F, C, r=1, update=False, corr=False, save=False, numpoints=None):
        """F: R^a --> R^b

        Note: I did not assign b automatically (although I could - it would just be b = p*q) since
        you may not want to use all of the entries of S.  Some bordering methods are not
        minimally augmented systems and thus only require certain entries of S."""
        (n,m) = dims2
        TestFunc.__init__(self, dims1, F, C, save=save, numpoints=numpoints)

        self.update = update
        self.corr = corr

        s = max(n,m)

        if self.data is None:
            self.data = args()
        self.data.n, self.data.m = n, m
        self.data.p = s-m+r
        self.data.q = s-n+r

    def setdata(self, A):
        """Note: p, q <= min(n,m)"""
        self.data.Brand = 2*(random((A.shape[0],self.data.p))-0.5)
        self.data.Crand = 2*(random((A.shape[1],self.data.q))-0.5)
        self.data.D = zeros((self.data.q,self.data.p), float)

        if self.update:
            U, S, Vh = linalg.svd(A)
            self.data.B = U[:,-1*self.data.p:]
            self.data.C = transpose(Vh)[:,-1*self.data.q:]
        else:
                # self.data.B = eye(self.data.Brand.shape)
                # self.data.C = eye(self.data.Crand.shape)
                # USE OF RANDOM
            self.data.B = self.data.Brand
            self.data.C = self.data.Crand
            #self.data.B = zeros(self.data.Brand.shape, float)
            #self.data.C = zeros(self.data.Crand.shape, float)
            #self.data.B[0,0] = self.data.C[0,0] = self.data.B[1,1] = self.data.C[1,1] = 1.0

    def updatedata(self, A):
        if self.update:
            if self.corr:
                B = self.data.w
                C = self.data.v
            else:
                # Note: Problem when singular vectors switch smallest singular value (See NewLorenz).
                #       To overcome this, I have implemented a 1e-8 random nudge.
                try:
                    ALU = linalg.lu_factor(A)
                    # BC = linalg.lu_solve(ALU, c_[linalg.lu_solve(ALU, self.data.B), self.data.C], trans=1)
                    # USE OF RANDOM NUDGE
                    BC = linalg.lu_solve(ALU, c_[linalg.lu_solve(ALU, self.data.B + 1e-8*self.data.Brand), \
                                                 self.data.C + 1e-8*self.data.Crand], trans=1)
                    C = linalg.lu_solve(ALU, BC[:,-1*self.data.q:])
                    B = BC[:,0:self.data.p]
                except:
                    if self.C.verbosity >= 1:
                        print('Warning: Problem updating border vectors.  Using svd...')
                    U, S, Vh = linalg.svd(A)
                    B = U[:,-1*self.data.p:]
                    C = transpose(Vh)[:,-1*self.data.q:]

            bmult = cmult = 1
            if matrixmultiply(transpose(self.data.B), B) < 0:
                bmult = -1
            if matrixmultiply(transpose(self.data.C), C) < 0:
                cmult = -1
            self.data.B = bmult*B*(linalg.norm(A,1)/linalg.norm(B))
            self.data.C = cmult*C*(linalg.norm(A,Inf)/linalg.norm(C))

    def func(self, A):
        V, W = self.getVW(A)

        self.data.v = V[0:self.data.m,:]
        self.data.w = W[0:self.data.n,:]
        self.data.g = V[-1*self.data.p:,:]

        self.updatedata(A)

        return self.data.g

    def getVW(self, A):
        # V --> m, W --> n
        #print self.data
        MLU = linalg.lu_factor(c_[r_[A,transpose(self.data.C)], r_[self.data.B,self.data.D]])
        V = linalg.lu_solve(MLU,r_[zeros((self.data.n,self.data.q), float), eye(self.data.q)])
        W = linalg.lu_solve(MLU,r_[zeros((self.data.m,self.data.p), float), eye(self.data.p)],trans=1)

        return V, W

class BiAltMethod(TestFunc):
    def __init__(self, dims, F, C, save=False, numpoints=None):
        (n,m) = dims
        TestFunc.__init__(self, dims, F, C, save=save, numpoints=numpoints)

        if self.data is None:
            self.data = args()
        n = len(self.F.coords)
        self.data.P = zeros((n*(n-1)//2, n*(n-1)//2), float)

    def bialtprod(self, A, B):
        n = A.shape[0]
        for p in range(1,n):
            for q in range(p):
                for r in range(1,n):
                    for s in range(r):
                        self.data.P[p*(p-1)//2 + q][r*(r-1)//2 + s] = 0.5*(A[p][r]*B[q][s] - A[p][s]*B[q][r] + \
                                                                         B[p][r]*A[q][s] - B[p][s]*A[q][r])
        return self.data.P

    def bialtprodeye(self, A):
        n = A.shape[0]
        for p in range(1,n):
            for q in range(p):
                for r in range(1,n):
                    for s in range(r):
                        if r == q:
                            self.data.P[p*(p-1)//2 + q][r*(r-1)//2 + s] = -1*A[p][s]
                        elif r != p and s == q:
                            self.data.P[p*(p-1)//2 + q][r*(r-1)//2 + s] = A[p][r]
                        elif r == p and s == q:
                            self.data.P[p*(p-1)//2 + q][r*(r-1)//2 + s] = A[p][p] + A[q][q]
                        elif r == p and s != q:
                            self.data.P[p*(p-1)//2 + q][r*(r-1)//2 + s] = A[q][s]
                        elif s == p:
                            self.data.P[p*(p-1)//2 + q][r*(r-1)//2 + s] = -1*A[q][r]
                        else:
                            self.data.P[p*(p-1)//2 + q][r*(r-1)//2 + s] = 0
        return self.data.P

class AddTestFunction(Function):
    """Only works with testfuncs that don't have PreTestFunc and rely only on sysfunc.
    BorderMethod update is set to False for now since updating is not working."""
    def __init__(self, C, TF_type):
        self.sysfunc = C.sysfunc
        self.testfunc = TF_type(self.sysfunc, C, update=False)

        self.coords = self.sysfunc.coords
        self.params = self.sysfunc.params

        Function.__init__(self, (self.sysfunc.n, self.sysfunc.m + self.testfunc.m))

    def setdata(self, X, V):
        if hasattr(self.testfunc, "setdata"):
            self.sysfunc.J_coords = self.sysfunc.jac(X, self.sysfunc.coords)
            self.sysfunc.J_params = self.sysfunc.jac(X, self.sysfunc.params)
            self.testfunc.setdata(X, V)

    def func(self, X):
        # This is for the testfunc
        self.sysfunc.J_coords = self.sysfunc.jac(X, self.sysfunc.coords)
        self.sysfunc.J_params = self.sysfunc.jac(X, self.sysfunc.params)
        tmp1 = self.sysfunc(X)
        tmp2 = self.testfunc(X, None)
        # Broke with move to SciPy 0.5.2
        return c_[[self.sysfunc(X)], [self.testfunc(X, None)]][0]

    def diff(self, X, ind=None):
        return r_[self.sysfunc.jac(X, ind), self.testfunc.jac(X, ind)]

# DISCRETE MAPS

class DiscreteMap(Function):
    """Turns a function into a map composed with itself period times.  Chain rule gives jacobian.
    Note that F: R^n+m --> R^n, where m is the number of free parameters.  Thus, we transform to
    G: R^n+m --> R^n+m given by G(x,p) = [F(x,p), p], where x is state variable and p parameters.
    This gives

        DG = [ DF_x DF_p ]
             [ 0    I    ]

    Chain rule on F gives DF^n = DF(F^n-1)*DF(F^n-2)*...*DF(F)*DF(X).  Chain rule on G gives
    the same thing as F, and when you keep track of the upper left and upper right blocks of DG composed
    with itself, you arrive at

        DF^n_x = DF_x(F^n-1)*DF_x(F^n-2)*...*DF_x(F)*DF_x(X)
        DF^n_a = DF_x(F^n-1)*DF_a(F^n-2) + DF_a(F^n-1) (defined recursively with F^0 = X, F^-1 = 0)
    """
    def __init__(self, F, period=1):
        self.F = F

        Function.__init__(self, (self.F.n, self.F.m))
        self.period = period

    def func(self, X):
        FX = self.F(X)
        if self.period >= 2:
            for k in range(2,self.period+1):
                FX = self.F(c_[[FX], [X[self.F.params]]][0])
        return FX

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        F_n = zeros((self.period, self.n), float)
        F_n[0] = X
        for k in range(1,self.period):
            F_n[k] = c_[[self.F(F_n[k-1])], [X[self.F.params]]][0]

        xslice = slice(self.F.coords[0], self.F.coords[-1]+1, 1)
        pslice = slice(self.F.params[0], self.F.params[-1]+1, 1)
        J = self.F.jac(F_n[0])
        for k in range(1,self.period):
            J2 = self.F.jac(F_n[k])
            J[:,xslice] = matrixmultiply(J2[:,xslice] ,J[:,xslice])
            J[:,pslice] = matrixmultiply(J2[:,xslice], J[:,pslice]) + J2[:,pslice]

        return J[:, ind[0]:ind[-1]+1]

class FixedPointMap(Function):
    """Turns a discrete map into a fixed point map."""
    def __init__(self, F):
        self.F = F

        Function.__init__(self, (self.F.n, self.F.m))

    def func(self, X):
        return self.F(X) - X[0:self.F.m]

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        return self.F.jac(X, ind) - eye(self.F.m, self.F.n)[:,ind[0]:ind[-1]+1]

class AddTestFunction_FixedPoint(FixedPointMap):
    """Only works with testfuncs that don't have PreTestFunc and rely only on sysfunc."""
    def __init__(self, C, TF_type):
        self.sysfunc = C.sysfunc
        self.testfunc = TF_type(self.sysfunc, C)

        self.coords = self.sysfunc.coords
        self.params = self.sysfunc.params

        Function.__init__(self, (self.sysfunc.n, self.sysfunc.m + self.testfunc.m))
        FixedPointMap.__init__(self, self.sysfunc)

    def setdata(self, X, V):
        if hasattr(self.testfunc, "setdata"):
            self.F.J_coords = self.F.jac(X, self.F.coords)
            self.F.J_params = self.F.jac(X, self.F.params)
            self.testfunc.setdata(X, V)

    def func(self, X):
        self.F.J_coords = self.F.jac(X, self.F.coords)
        self.F.J_params = self.F.jac(X, self.F.params)
        tmp1 = self.F(X)
        tmp2 = self.testfunc(X, None)
        # Broke with move to SciPy 0.5.2
        return c_[[self.F(X) - X[0:self.F.m]], [self.testfunc(X, None)]][0]

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        return r_[self.F.jac(X, ind) - eye(self.F.m, self.F.n)[:,ind[0]:ind[-1]+1], self.testfunc.jac(X, ind)]

class AddTestFunction_FixedPoint_Mult(FixedPointMap):
    """Only works with testfuncs that don't have PreTestFunc and rely only on sysfunc."""
    def __init__(self, C, TF_types):
        self.sysfunc = C.sysfunc

        self.testfunc = []
        tf_m = 0
        for tftype in TF_types:
            self.testfunc.append(tftype(self.sysfunc, C))
            tf_m = tf_m + self.testfunc[-1].m

        self.coords = self.sysfunc.coords
        self.params = self.sysfunc.params

        Function.__init__(self, (self.sysfunc.n, self.sysfunc.m + tf_m))
        FixedPointMap.__init__(self, self.sysfunc)

    def setdata(self, X, V):
        self.F.J_coords = self.F.jac(X, self.F.coords)
        self.F.J_params = self.F.jac(X, self.F.params)
        for tf in self.testfunc:
            if hasattr(tf, "setdata"):
                tf.setdata(X, V)

    def func(self, X):
        self.F.J_coords = self.F.jac(X, self.F.coords)
        self.F.J_params = self.F.jac(X, self.F.params)
        tf_vals = []
        for tf in self.testfunc:
            tf_vals = r_[tf_vals, tf(X,None)]
        # Broke with move to SciPy 0.5.2
        return c_[[self.F(X) - X[0:self.F.m]], [tf_vals]][0]

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        tf_jacs = self.testfunc[0].jac(X,ind)[0]
        for tf in self.testfunc[1:]:
            tf_jacs = r_[[tf_jacs], [tf.jac(X, ind)[0]]]

        return r_[self.F.jac(X, ind) - eye(self.F.m, self.F.n)[:,ind[0]:ind[-1]+1], tf_jacs]

# BOUNDARY POINTS

class B_Check(TestFunc):
    """There is an attempt here to be a little efficient.  Instead of having a test
    function for every variable and free parameter, just have one test function that
    monitors when one of the boundaries is crossed (self.all).  Once this is triggered,
    find the var/par that crossed and locate using that specific var/par (self.one)."""
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)
        # Get the domain for the parameters and variables
        # We want to make sure we stay in these bounds
        lower = {}
        upper = {}
        for par in C.freepars:
            if C.curvetype == 'UD-C':
                if hasattr(C, '_userdomain') and par in C._userdomain.keys():
                    domain = C._userdomain[par]
                else:
                    domain = [-1*Inf, Inf]
            else:
                domain = C.gensys.query('pardomains')[par] #pdomain[par]
            lower[par] = domain[0]
            upper[par] = domain[1]

        for par in C.varslist:
            if C.curvetype == 'UD-C':
                if hasattr(C, '_userdomain') and par in C._userdomain.keys():
                    domain = C._userdomain[par]
                else:
                    domain = [-1*Inf, Inf]
            else:
                domain = C.gensys.query('vardomains')[par] #xdomain[par]
            lower[par] = domain[0]
            upper[par] = domain[1]

        for par in C.auxpars:
            lower[par] = -1*Inf
            upper[par] = Inf

        # We can compare a row of values to the two following arrays
        # to determine if we are in bounds.
        self.lower = tocoords(C,lower)
        self.upper = tocoords(C,upper)

        self.func = self.all
        self.ind = None

    def one(self, X, V):
        return array((X[self.ind] - self.lower[self.ind])*(self.upper[self.ind] - X[self.ind]))

    def all(self, X, V):
        return array((any(X < self.lower) or any(X > self.upper)) and -1 or 1)

# BRANCH POINTS

class Branch_Det(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array([linalg.det(r_[c_[self.F.J_coords,self.F.J_params],[V]])])

class Branch_Bor(BorderMethod):
    def __init__(self, F, C, update=True, save=False, numpoints=None):
        BorderMethod.__init__(self, (F.n, 2), (F.n-1,F.n), F, C, update=update, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        BorderMethod.setdata(self, c_[self.F.J_coords,self.F.J_params])

    def func(self, X, V):
        return array(BorderMethod.func(self, c_[self.F.J_coords,self.F.J_params])[0])

# FOLD POINTS

class Fold_Det(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array([linalg.det(self.F.J_coords)])

class Fold_Tan(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array([V[-1]])

class Fold_Bor(BorderMethod):
    def __init__(self, F, C, update=True, corr=True, save=False, numpoints=None):
        BorderMethod.__init__(self, (F.n, 1), (F.m,F.m), F, C, update=update, corr=corr, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        BorderMethod.setdata(self, self.F.J_coords)

    def func(self, X, V):
        return array(BorderMethod.func(self, self.F.J_coords)[0])

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        V, W = self.getVW(self.F.jac(X, self.F.coords))
        H = self.F.hess(X, self.F.coords, ind)
        return -1*reshape([bilinearform(H[:,:,i], V[0:self.data.m,:], W[0:self.data.n,:]) \
                           for i in range(n)],(1,len(ind)))

# HOPF POINTS

class Hopf_Det(BiAltMethod):
    def __init__(self, F, C, save=False, numpoints=None):
        BiAltMethod.__init__(self, (F.n,1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        self.bialtprodeye(2*self.F.J_coords)
        return array([linalg.det(self.data.P)])

class Hopf_Bor(BorderMethod, BiAltMethod):
    def __init__(self, F, C, update=False, corr=False, save=False, numpoints=None):
        n = F.m
        BiAltMethod.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)
        BorderMethod.__init__(self, (F.n, 1), (n*(n-1)//2, n*(n-1)//2), F, C, update=update, corr=corr, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        self.bialtprodeye(2*self.F.J_coords)
        BorderMethod.setdata(self, self.data.P)

    def func(self, X ,V):
        self.bialtprodeye(2*self.F.J_coords)
        return array(BorderMethod.func(self, self.data.P)[0])

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        self.bialtprodeye(2*self.F.jac(X, self.F.coords))
        V, W = self.getVW(self.data.P)
        H = self.F.hess(X, self.F.coords, ind)
        return -1*reshape([bilinearform(self.bialtprodeye(2*H[:,:,i]), V[0:self.data.m,:], W[0:self.data.n,:]) \
                           for i in range(n)],(1,len(ind)))

    def __str__(self):
        return 'class Hopf_Bor(BorderMethod, BiAltMethod)'

class Hopf_Eig(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)
        self.sgnnum = 0
        self.sgn = 1

    def func(self, X, V):
        eigs, LV = linalg.eig(self.F.J_coords)
        reigs = [real(eig) for eig in eigs if abs(imag(eig)) >= 0.0]
        sgnnum = len([z for z in reigs if z < 0.0])
        if self.sgnnum != sgnnum:
            self.sgnnum = sgnnum
            self.sgn = -1*self.sgn
        return array([self.sgn*min([abs(r) for r in reigs])])

# Codimension-1 continuation

class Hopf_Double_Bor_One(BorderMethod, BiAltMethod):
    def __init__(self, F, C, update=False, save=False, numpoints=None):
        n = F.m
        BiAltMethod.__init__(self, (F.n,1), F, C, save=save, numpoints=numpoints)
        BorderMethod.__init__(self, (F.n,1), (n*(n-1)//2, n*(n-1)//2), F, C, r=2, corr=True, update=update, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        A = self.bialtprodeye(2*self.F.J_coords)
        """Note: p, q <= min(n,m)"""

        self.data.Brand = 2*(random((A.shape[0],self.data.p))-0.5)
        self.data.Crand = 2*(random((A.shape[1],self.data.q))-0.5)
        self.data.B = zeros((A.shape[0],self.data.p), float)
        self.data.C = zeros((A.shape[1],self.data.q), float)
        self.data.D = zeros((self.data.q,self.data.p), float)

        U, S, Vh = linalg.svd(A)
        self.data.b = U[:,-1:]
        self.data.c = transpose(Vh)[:,-1:]

        if self.update:
            self.data.B[:,1] = self.data.b
            self.data.C[:,1] = self.data.c

            U2, S2, Vh2 = linalg.svd(c_[r_[A, transpose(self.data.C[:,1])], r_[self.data.B[:,1], [[0]]]])
            self.data.B[:,2] = U2[0:A.shape[0],-1:]
            self.data.C[:,2] = transpose(Vh2)[0:A.shape[1],-1:]
            self.data.D[0,1] = U2[A.shape[0],-1]
            self.data.D[1,0] = transpose(Vh2)[A.shape[1],-1]
        else:
                # self.data.B = eye(self.data.Brand.shape)
                # self.data.C = eye(self.data.Crand.shape)
                # USE OF RANDOM
            self.data.B = self.data.Brand
            self.data.C = self.data.Crand
            #self.data.B[0,0] = self.data.C[0,0] = self.data.B[1,1] = self.data.C[1,1] = 1.0

    def updatedata(self, A):
        # Update b, c
        try:
            ALU = linalg.lu_factor(A)
            # BC = linalg.lu_solve(ALU, c_[linalg.lu_solve(ALU, self.data.b), \
            #                      self.data.c], trans=1)
            # USE OF RANDOM NUDGING
            BC = linalg.lu_solve(ALU, c_[linalg.lu_solve(ALU, self.data.b + 1e-8*self.data.Brand[:,:1]), \
                                         self.data.c + 1e-8*self.data.Crand[:,:1]], trans=1)
            C = linalg.lu_solve(ALU, BC[:,-1:])
            B = BC[:,:1]
        except:
            if self.C.verbosity >= 1:
                print('Warning: Problem updating border vectors.  Using svd...')
            U, S, Vh = linalg.svd(A)
            B = U[:,-1:]
            C = transpose(Vh)[:,-1:]

        bmult = cmult = 1
        if matrixmultiply(transpose(self.data.b), B) < 0:
            bmult = -1
        if matrixmultiply(transpose(self.data.c), C) < 0:
            cmult = -1
        self.data.b = bmult*B*(linalg.norm(A,1)/linalg.norm(B))
        self.data.c = cmult*C*(linalg.norm(A,Inf)/linalg.norm(C))

        # Update
        if self.update:
            self.data.B[:,0] = self.data.b*(linalg.norm(A,1)/linalg.norm(self.data.b))
            self.data.C[:,0] = self.data.c*(linalg.norm(A,Inf)/linalg.norm(self.data.c))

            self.data.B[:,1] = self.data.w[:,2]*(linalg.norm(A,1)/linalg.norm(self.data.w,1))
            self.data.C[:,1] = self.data.v[:,2]*(linalg.norm(A,Inf)/linalg.norm(self.data.v,1))

            self.data.D[0,1] = self.data.g[0,1]
            self.data.D[1,0] = self.data.g[1,0]

    def func(self, X, V):
        self.bialtprodeye(2*self.F.J_coords)
        return array([linalg.det(BorderMethod.func(self, self.data.P))])

class Hopf_Double_Bor_Two(BorderMethod):
    # Define diff at some point
    def __init__(self, F, C, update=False, save=False, numpoints=None):
        BorderMethod.__init__(self, (F.n, 2), (F.m,F.m), F, C, r=2, corr=True, update=update, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        BorderMethod.setdata(self, matrixmultiply(self.F.J_coords,self.F.J_coords) + X[-1]*eye(self.F.m))

    def func(self, X, V):
        BorderMethod.func(self, matrixmultiply(self.F.J_coords,self.F.J_coords) + X[-1]*eye(self.F.m))
        return array([self.data.g[0,0], self.data.g[1,1]])

    # def diff(self, X, ind=None):
        # try:
            # n = len(ind)
        # except:
            # n = self.n
            # ind = range(n)
#
        # J_coords = self.F.jac(X, self.F.coords)
        # V, W = self.getVW(matrixmultiply(J_coords,J_coords) + X[-1]*eye(self.F.m))
        # H = self.F.hess(X, self.F.coords, ind)
        # dG = -1*array([bilinearform(H[:,:,i], V[0:self.data.m,:], W[0:self.data.n,:]) \
            # for i in range(n)])
        # #return r_[reshape(dG[:,0,0], (1,len(ind))), reshape(dG[:,1,1], (1,len(ind)))]
        # return reshape(r_[dG[:,0,0], dG[:,1,1]], (2, len(ind)))

# Codimension-2 bifurcations

class BT_Fold(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array(matrixmultiply(transpose(self.F.testfunc.data.w), self.F.testfunc.data.v)[0])

class CP_Fold(TestFunc):
    """self.F is of type AddTestFunction***, with testfunc of type LP_Bor (or) LPC_Bor"""

    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        H = self.F.sysfunc.hess(X, self.F.coords, self.F.coords)
        v = self.F.testfunc.data.v
        w = self.F.testfunc.data.w
        return array(matrixmultiply(transpose(w), reshape([bilinearform(H[i,:,:], v, v) \
                                                           for i in range(H.shape[0])],(H.shape[0],1)))[0])

        """This commented text is normal form information (I think).  It's a temporary commment.  Delete when comfortable."""
#q = self.F.testfunc.data.v/linalg.norm(self.F.testfunc.data.v)
#p = self.F.testfunc.data.w/matrixmultiply(transpose(self.F.testfunc.data.w),q)
#return array(0.5*matrixmultiply(transpose(p), reshape([bilinearform(H[i,:,:], q, q) \
#     for i in range(H.shape[0])],(H.shape[0],1)))[0])

class BP_Fold(TestFunc):
    def __init__(self, F, C, pind=0, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)
        self.pind = pind;

    def func(self, X, V):
        J_params = self.F.sysfunc.J_params
        return [matrixmultiply(transpose(self.F.testfunc.data.w), J_params[:,self.pind])[0]]

class CP_Fold2(TestFunc):
    """Used for Cusp Curve"""
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        F = self.C.CorrFunc
        # print F.testfunc[0]
        # print F.testfunc[0].data
        F.testfunc[0](X,V)
        # print F.testfunc[0].data
        H = F.sysfunc.hess(X, self.F.coords, self.F.coords)
        q = F.testfunc[0].data.v/linalg.norm(F.testfunc[0].data.v)
        p = F.testfunc[0].data.w/matrixmultiply(transpose(F.testfunc[0].data.w),q)

        return array(0.5*matrixmultiply(transpose(p), reshape([bilinearform(H[i,:,:], q, q) \
             for i in range(H.shape[0])],(H.shape[0],1)))[0])

# Test functions for HopfCurveOne

class DH_Hopf(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array([self.F.testfunc.data.g[1,1]])

class BT_Hopf_One(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        k = self.C.TFdata.k

        return array([k])

class GH_Hopf_One(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        k = self.C.TFdata.k
        v1 = self.C.TFdata.v1
        w1 = self.C.TFdata.w1

        if k >=0:
            J_coords = self.F.sysfunc.J_coords
            w = sqrt(k)

            q = v1 - (1j/w)*matrixmultiply(self.F.sysfunc.J_coords,v1)
            p = w1 + (1j/w)*matrixmultiply(transpose(self.F.sysfunc.J_coords),w1)

            p /= linalg.norm(p)
            q /= linalg.norm(q)

            p = reshape(p,(p.shape[0],))
            q = reshape(q,(q.shape[0],))

            direc = conjugate(1/matrixmultiply(transpose(conjugate(p)),q))
            p = direc*p

            l1 = firstlyapunov(X, self.F.sysfunc, w, J_coords=J_coords, p=p, q=q)

            return array([l1])
        else:
            return array([1])

# Test functions for HopfCurveTwo

class BT_Hopf(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array([X[-1]])

class GH_Hopf(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        if X[-1] >=0:
            J_coords = self.F.sysfunc.J_coords
            w = sqrt(X[-1])

            l1 = firstlyapunov(X, self.F.sysfunc, w, J_coords=J_coords, V=self.F.testfunc.data.v, W=self.F.testfunc.data.w)

            return array([l1])
        else:
            return array([1])

# Test functions for FixedPointCurve

# class Branch_FP_Det(TestFunc):
#     def __init__(self, F, C, save=False, numpoints=None):
#         TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)
#
#     def func(self, X, V):
#         return array([linalg.det(r_[c_[self.F.J_coords - eye(self.F.m, self.F.m),self.F.J_params],[V]])])

class LPC_Det(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array([linalg.det(self.F.J_coords - eye(self.F.m, self.F.m))])

class LPC_Bor(BorderMethod):
    def __init__(self, F, C, update=True, corr=True, save=False, numpoints=None):
        BorderMethod.__init__(self, (F.n, 1), (F.m,F.m), F, C, update=update, corr=corr, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        BorderMethod.setdata(self, self.F.J_coords-eye(self.F.m, self.F.m))

    def func(self, X, V):
        return array(BorderMethod.func(self, self.F.J_coords-eye(self.F.m, self.F.m))[0])

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        V, W = self.getVW(self.F.jac(X, self.F.coords)-eye(self.F.m, self.F.m))
        H = self.F.hess(X, self.F.coords, ind)
        return -1*reshape([bilinearform(H[:,:,i], V[0:self.data.m,:], W[0:self.data.n,:]) \
                           for i in range(n)],(1,len(ind)))

class PD_Det(TestFunc):
    def __init__(self, F, C, save=False, numpoints=None):
        TestFunc.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        return array([linalg.det(self.F.J_coords + eye(self.F.m, self.F.m))])

class PD_Bor(BorderMethod):
    def __init__(self, F, C, update=True, corr=True, save=False, numpoints=None):
        BorderMethod.__init__(self, (F.n, 1), (F.m,F.m), F, C, update=update, corr=corr, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        BorderMethod.setdata(self, self.F.J_coords+eye(self.F.m, self.F.m))

    def func(self, X, V):
        return array(BorderMethod.func(self, self.F.J_coords+eye(self.F.m, self.F.m))[0])

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.n
            ind = list(range(n))

        V, W = self.getVW(self.F.jac(X, self.F.coords)+eye(self.F.m, self.F.m))
        H = self.F.hess(X, self.F.coords, ind)
        return -1*reshape([bilinearform(H[:,:,i], V[0:self.data.m,:], W[0:self.data.n,:]) \
                           for i in range(n)],(1,len(ind)))

class NS_Det(BiAltMethod):
    def __init__(self, F, C, save=False, numpoints=None):
        BiAltMethod.__init__(self, (F.n,1), F, C, save=save, numpoints=numpoints)

    def func(self, X, V):
        n = self.F.m*(self.F.m-1)//2
        self.bialtprod(self.F.J_coords,self.F.J_coords)
        return array([linalg.det(self.data.P - eye(n,n))])

class NS_Bor(BorderMethod, BiAltMethod):
    def __init__(self, F, C, update=True, corr=True, save=False, numpoints=None):
        n = F.m
        BiAltMethod.__init__(self, (F.n, 1), F, C, save=save, numpoints=numpoints)
        BorderMethod.__init__(self, (F.n, 1), (n*(n-1)//2, n*(n-1)//2), F, C, update=update, corr=corr, save=save, numpoints=numpoints)

    def setdata(self, X, V):
        n = self.F.m*(self.F.m-1)//2
        self.bialtprod(self.F.J_coords,self.F.J_coords)
        BorderMethod.setdata(self, self.data.P - eye(n, n))

    def func(self, X, V):
        n = self.F.m*(self.F.m-1)//2
        self.bialtprod(self.F.J_coords,self.F.J_coords)
        return array(BorderMethod.func(self, self.data.P - eye(n, n))[0])

    def diff(self, X, ind=None):
        try:
            n = len(ind)
        except:
            n = self.F.m*(self.F.m-1)//2
            ind = list(range(n))

        self.F.J_coords = self.F.jac(X, self.F.coords)
        self.bialtprod(self.F.J_coords,self.F.J_coords)
        V, W = self.getVW(self.data.P - eye(n, n))
        H = self.F.hess(X, self.F.coords, ind)
        return -1*reshape([bilinearform(H[:,:,i], V[0:self.data.m,:], W[0:self.data.n,:]) \
                           for i in range(n)],(1,len(ind)))

# Test function for stopping at parameter values (SPOut for PyCont)

class ParTestFunc(TestFunc):
    def __init__(self, n, C, pix, pval, save=False, numpoints=None):
        TestFunc.__init__(self, (n, 1), C.sysfunc, C, save=save, numpoints=numpoints)
        self._pval = pval
        self._pix = pix

    def func(self, X, V):
        return array([X[self._pix] - self._pval])


# Test function for UserDefinedCurve

class UserDefinedTestFunc(TestFunc):
    def __init__(self, dims, C, tfunc, save=False, numpoints=None):
        TestFunc.__init__(self, dims, C.sysfunc, C, save=save, numpoints=numpoints)

        self.tfunc = tfunc

    def setdata(self, X, V):
        self.C._userdata.sgn = -1
        self.C._userdata.val = 1

    def func(self, X, V):
        return self.tfunc(self.C._userdata, X, V)
