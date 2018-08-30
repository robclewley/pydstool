""" Bifurcation point classes.  Each class locates and processes bifurcation points.

    * _BranchPointFold is a version based on BranchPoint location algorithms
    * BranchPoint: Branch process is broken (can't find alternate branch -- see MATCONT notes)

    Drew LaMar, March 2006
"""

from __future__ import absolute_import, print_function

from .misc import *
from PyDSTool.common import args
from .TestFunc import DiscreteMap, FixedPointMap

from numpy import Inf, NaN, isfinite, r_, c_, sign, mod, \
    subtract, divide, transpose, eye, real, imag, \
    conjugate, average
from scipy import optimize, linalg
from numpy import dot as matrixmultiply
from numpy import array, float, complex, int, float64, complex64, int32, \
    zeros, divide, subtract, reshape, argsort, nonzero

#####
_classes = ['BifPoint', 'BPoint', 'BranchPoint', 'FoldPoint', 'HopfPoint',
            'BTPoint', 'ZHPoint', 'CPPoint',
            'BranchPointFold', '_BranchPointFold', 'DHPoint',
            'GHPoint', 'LPCPoint', 'PDPoint', 'NSPoint', 'SPoint']

__all__ = _classes
#####

class BifPoint(object):
    def __init__(self, testfuncs, flagfuncs, label='Bifurcation', stop=False):
        self.testfuncs = []
        self.flagfuncs = []
        self.found = []
        self.label = label
        self.stop = stop
        self.data = args()

        if not isinstance(testfuncs, list):
            testfuncs = [testfuncs]
        if not isinstance(flagfuncs, list):
            flagfuncs = [flagfuncs]

        self.testfuncs.extend(testfuncs)
        self.flagfuncs.extend(flagfuncs)
        self.tflen = len(self.testfuncs)

    def locate(self, P1, P2, C):
        pointlist = []

        for i, testfunc in enumerate(self.testfuncs):
            if self.flagfuncs[i] == iszero:
                for ind in range(testfunc.m):
                    X, V = testfunc.findzero(P1, P2, ind)
                    pointlist.append((X,V))

        X = average([point[0] for point in pointlist], axis=0)
        V = average([point[1] for point in pointlist], axis=0)
        C.Corrector(X,V)

        return X, V

    def process(self, X, V, C):
        data = args()
        data.X = todict(C, X)
        data.V = todict(C, V)

        self.found.append(data)

    def info(self, C, ind=None, strlist=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        if C.verbosity >= 1:
            print(self.label + ' Point found ')
        if C.verbosity >= 2:
            print('========================== ')
            for n, i in enumerate(ind):
                print(n, ': ')
                Xd = self.found[i].X
                for k, j in Xd.items():
                    print(k, ' = ', j)
                print('')
                if hasattr(self.found[i], 'eigs'):
                    print('Eigenvalues = \n')
                    for x in self.found[i].eigs:
                        print('     (%f,%f)' % (x.real, x.imag))
                    print('\n')
                if strlist is not None:
                    for string in strlist:
                        print(string)
                    print('')


class SPoint(BifPoint):
    """Special point that represents user-selected free parameter values."""
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'S', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)
        self.info(C, -1)
        return True


class BPoint(BifPoint):
    """Special point that represents boundary of computational domain."""
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'B', stop=stop)

    def locate(self, P1, P2, C):
        # Find location that triggered testfunc and initialize testfunc to that index
        val1 = (P1[0]-self.testfuncs[0].lower)*(self.testfuncs[0].upper-P1[0])
        val2 = (P2[0]-self.testfuncs[0].lower)*(self.testfuncs[0].upper-P2[0])
        ind = nonzero(val1*val2 < 0)
        self.testfuncs[0].ind = ind
        self.testfuncs[0].func = self.testfuncs[0].one

        X, V = BifPoint.locate(self, P1, P2, C)

        # Set testfunc back to monitoring all
        self.testfuncs[0].ind = None
        self.testfuncs[0].func = self.testfuncs[0].all

        return X, V

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        BifPoint.info(self, C, ind)



class BranchPoint(BifPoint):
    """May only work for EquilibriumCurve ... (needs fixing)"""
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'BP', stop=stop)

    def __locate_newton(self, X, C):
        """x[0:self.dim] = (x,alpha)
           x[self.dim] = beta
           x[self.dim+1:2*self.dim] = p
        """
        J_coords = C.CorrFunc.jac(X[0:C.dim], C.coords)
        J_params = C.CorrFunc.jac(X[0:C.dim], C.params)

        return r_[C.CorrFunc(X[0:C.dim]) + X[C.dim]*X[C.dim+1:], \
                  matrixmultiply(transpose(J_coords),X[C.dim+1:]), \
                  matrixmultiply(transpose(X[C.dim+1:]),J_params), \
                  matrixmultiply(transpose(X[C.dim+1:]),X[C.dim+1:]) - 1]

    def locate(self, P1, P2, C):
        # Initiliaze p vector to eigenvector with smallest eigenvalue
        X, V = P1
        X2, V2 = P2

        J_coords = C.CorrFunc.jac(X, C.coords)

        W, VL = linalg.eig(J_coords, left=1, right=0)
        ind = argsort([abs(eig) for eig in W])[0]
        p = real(VL[:,ind])

        initpoint = zeros(2*C.dim, float)
        initpoint[0:C.dim] = X
        initpoint[C.dim+1:] = p

        X = optimize.fsolve(self.__locate_newton, initpoint, C)
        self.data.psi = X[C.dim+1:]
        X = X[0:C.dim]

        V = 0.5*(V+V2)

        return X, V

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        # Finds the new branch
        J_coords = C.CorrFunc.jac(X, C.coords)
        J_params = C.CorrFunc.jac(X, C.params)


        singular = True
        perpvec = r_[1,zeros(C.dim-1)]
        d = 1
        while singular and d <= C.dim:
            try:
                v0 = linalg.solve(r_[c_[J_coords, J_params],
                                  [perpvec]], \
                                  r_[zeros(C.dim-1),1])
            except:
                perpvec = r_[0., perpvec[0:(C.dim-1)]]
                d += 1
            else:
                singular = False

        if singular:
            raise PyDSTool_ExistError("Problem in _compute: Failed to compute tangent vector.")
        v0 /= linalg.norm(v0)
        V = sign([x for x in v0 if abs(x) > 1e-8][0])*v0

        A = r_[c_[J_coords, J_params], [V]]
        W, VR = linalg.eig(A)
        W0 = [ind for ind, eig in enumerate(W) if abs(eig) < 5e-5]
        V1 = real(VR[:,W0[0]])

        H = C.CorrFunc.hess(X, C.coords+C.params, C.coords+C.params)
        c11 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V, V) for i in range(H.shape[0])])
        c12 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V, V1) for i in range(H.shape[0])])
        c22 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V1, V1) for i in range(H.shape[0])])

        beta = 1
        alpha = -1*c22/(2*c12)
        V1 = alpha*V + beta*V1
        V1 /= linalg.norm(V1)

        self.found[-1].eigs = W
        self.found[-1].branch = todict(C, V1)

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        for n, i in enumerate(ind):
            strlist.append('branch angle = ' + repr(matrixmultiply(tocoords(C, self.found[i].V), \
                tocoords(C, self.found[i].branch))))

        X = tocoords(C, self.found[-1].X)
        V = tocoords(C, self.found[-1].V)
        C._preTestFunc(X, V)
        strlist.append('Test function #1: ' + repr(self.testfuncs[0](X,V)[0]))

        BifPoint.info(self, C, ind, strlist)



class FoldPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'LP', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        # Compute normal form coefficient
        # NOTE: These are for free when using bordering technique!)
        # NOTE: Does not agree with MATCONT output! (if |p| = |q| = 1, then it does)
        J_coords = C.CorrFunc.jac(X, C.coords)
        W, VL, VR = linalg.eig(J_coords, left=1, right=1)
        minW = min(abs(W))
        ind = [(abs(eig) < minW+1e-8) and (abs(eig) > minW-1e-8) for eig in W].index(True)
        p, q = real(VL[:,ind]), real(VR[:,ind])
        p /= matrixmultiply(p,q)

        B = C.CorrFunc.hess(X, C.coords, C.coords)
        self.found[-1].a = abs(0.5*matrixmultiply(p,[bilinearform(B[i,:,:], q, q) for i in range(B.shape[0])]))
        self.found[-1].eigs = W

        numzero = len([eig for eig in W if abs(eig) < 1e-4])
        if numzero > 1:
            if C.verbosity >= 2:
                print('Fold-Fold!\n')
            del self.found[-1]
            return False
        elif numzero == 0:
            if C.verbosity >= 2:
                print('False positive!\n')
            del self.found[-1]
            return False

        if C.verbosity >= 2:
            print('\nChecking...')
            print('  |q| = %f' % linalg.norm(q))
            print('  <p,q> = %f' % matrixmultiply(p,q))
            print('  |Aq| = %f' % linalg.norm(matrixmultiply(J_coords,q)))
            print('  |transpose(A)p| = %f\n' % linalg.norm(matrixmultiply(transpose(J_coords),p)))

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        for n, i in enumerate(ind):
            strlist.append('a = ' + repr(self.found[i].a))

        BifPoint.info(self, C, ind, strlist)



class HopfPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'H', stop=stop)

    def process(self, X, V, C):
        """Tolerance for eigenvalues a possible problem when checking for neutral saddles."""
        BifPoint.process(self, X, V, C)

        J_coords = C.CorrFunc.jac(X, C.coords)
        eigs, LV, RV = linalg.eig(J_coords,left=1,right=1)

        # Check for neutral saddles
        found = False
        for i in range(len(eigs)):
            if abs(imag(eigs[i])) < 1e-5:
                for j in range(i+1,len(eigs)):
                    if C.verbosity >= 2:
                        if abs(eigs[i]) < 1e-5 and abs(eigs[j]) < 1e-5:
                            print('Fold-Fold point found in Hopf!\n')
                        elif abs(imag(eigs[j])) < 1e-5 and abs(real(eigs[i]) + real(eigs[j])) < 1e-5:
                            print('Neutral saddle found!\n')
            elif abs(real(eigs[i])) < 1e-5:
                for j in range(i+1, len(eigs)):
                    if abs(real(eigs[j])) < 1e-5 and abs(real(eigs[i]) - real(eigs[j])) < 1e-5:
                        found = True
                        w = abs(imag(eigs[i]))
                        if imag(eigs[i]) > 0:
                            p = conjugate(LV[:,j])/linalg.norm(LV[:,j])
                            q = RV[:,i]/linalg.norm(RV[:,i])
                        else:
                            p = conjugate(LV[:,i])/linalg.norm(LV[:,i])
                            q = RV[:,j]/linalg.norm(RV[:,j])

        if not found:
            del self.found[-1]
            return False

        direc = conjugate(1/matrixmultiply(conjugate(p),q))
        p = direc*p

        # Alternate way to compute 1st lyapunov coefficient (from Kuznetsov [4])

        #print (1./(w*w))*real(1j*matrixmultiply(conjugate(p),b1)*matrixmultiply(conjugate(p),b3) + \
        #   w*matrixmultiply(conjugate(p),trilinearform(D,q,q,conjugate(q))))

        self.found[-1].w = w
        self.found[-1].l1 = firstlyapunov(X, C.CorrFunc, w, J_coords=J_coords, p=p, q=q, check=(C.verbosity==2))
        self.found[-1].eigs = eigs

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        for n, i in enumerate(ind):
            strlist.append('w = ' + repr(self.found[i].w))
            strlist.append('l1 = ' + repr(self.found[i].l1))

        BifPoint.info(self, C, ind, strlist)


# Codimension-2 bifurcations

class BTPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'BT', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        J_coords = C.CorrFunc.sysfunc.jac(X, C.coords)

        W, VL, VR = linalg.eig(J_coords, left=1, right=1)

        self.found[-1].eigs = W

        if C.verbosity >= 2:
            if C.CorrFunc.testfunc.data.B.shape[1] == 2:
                b = matrixmultiply(transpose(J_coords), C.CorrFunc.testfunc.data.w[:,0])
                c = matrixmultiply(J_coords, C.CorrFunc.testfunc.data.v[:,0])
            else:
                b = C.CorrFunc.testfunc.data.w[:,0]
                c = C.CorrFunc.testfunc.data.v[:,0]
            print('\nChecking...')
            print('  <b,c> = %f' % matrixmultiply(transpose(b), c))
            print('\n')

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        BifPoint.info(self, C, ind)



class ZHPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'ZH', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        J_coords = C.CorrFunc.sysfunc.jac(X, C.coords)

        W, VL, VR = linalg.eig(J_coords, left=1, right=1)

        self.found[-1].eigs = W

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        BifPoint.info(self, C, ind)



class CPPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'CP', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        J_coords = C.CorrFunc.sysfunc.jac(X, C.coords)
        B = C.CorrFunc.sysfunc.hess(X, C.coords, C.coords)

        W, VL, VR = linalg.eig(J_coords, left=1, right=1)

        q = C.CorrFunc.testfunc.data.C/linalg.norm(C.CorrFunc.testfunc.data.C)
        p = C.CorrFunc.testfunc.data.B/matrixmultiply(transpose(C.CorrFunc.testfunc.data.B),q)

        self.found[-1].eigs = W

        a = 0.5*matrixmultiply(transpose(p), reshape([bilinearform(B[i,:,:], q, q) \
                for i in range(B.shape[0])],(B.shape[0],1)))[0][0]
        if C.verbosity >= 2:
            print('\nChecking...')
            print('  |a| = %f' % a)
            print('\n')

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        BifPoint.info(self, C, ind)


class BranchPointFold(BifPoint):
    """Check Equilibrium.m in MATCONT"""
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'BP', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        pind = self.testfuncs[0].pind

        # Finds the new branch
        J_coords = C.CorrFunc.jac(X, C.coords)
        J_params = C.CorrFunc.jac(X, C.params)

        A = r_[c_[J_coords, J_params[:,pind]]]
        #A = r_[c_[J_coords, J_params], [V]]

        W, VR = linalg.eig(A)
        W0 = [ind for ind, eig in enumerate(W) if abs(eig) < 5e-5]
        tmp = real(VR[:,W0[0]])
        V1 = r_[tmp[:-1], 0, 0]
        V1[len(tmp)-1+pind] = tmp[-1]

        """NEED TO FIX THIS!"""
        H = C.CorrFunc.hess(X, C.coords+C.params, C.coords+C.params)
        # c11 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V, V) for i in range(H.shape[0])])
        # c12 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V, V1) for i in range(H.shape[0])])
        # c22 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V1, V1) for i in range(H.shape[0])])

        # beta = 1
        # alpha = -1*c22/(2*c12)
        # V1 = alpha*V + beta*V1
        # V1 /= linalg.norm(V1)

        self.found[-1].eigs = W
        self.found[-1].branch = None
        self.found[-1].par = C.freepars[self.testfuncs[0].pind]
        # self.found[-1].branch = todict(C, V1)

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        #for n, i in enumerate(ind):
        #    strlist.append('branch angle = ' + repr(matrixmultiply(tocoords(C, self.found[i].V), \
        #        tocoords(C, self.found[i].branch))))

        X = tocoords(C, self.found[-1].X)
        V = tocoords(C, self.found[-1].V)
        C._preTestFunc(X, V)
        strlist.append('Test function #1: ' + repr(self.testfuncs[0](X,V)[0]))

        BifPoint.info(self, C, ind, strlist)

class _BranchPointFold(BifPoint):
    """Check Equilibrium.m in MATCONT"""
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'BP', stop=stop)

    def __locate_newton(self, X, C):
        """Note:  This is redundant!! B is a column of A!!!  Works for now, though..."""
        pind = self.testfuncs[0].pind

        J_coords = C.CorrFunc.jac(X[0:C.dim], C.coords)
        J_params = C.CorrFunc.jac(X[0:C.dim], C.params)

        A = c_[J_coords, J_params[:,pind]]
        B = J_params[:,pind]

        return r_[C.CorrFunc(X[0:C.dim]) + X[C.dim]*X[C.dim+1:], \
                  matrixmultiply(transpose(A),X[C.dim+1:]), \
                  matrixmultiply(transpose(X[C.dim+1:]),B), \
                  matrixmultiply(transpose(X[C.dim+1:]),X[C.dim+1:]) - 1]

    def locate(self, P1, P2, C):
        # Initiliaze p vector to eigenvector with smallest eigenvalue
        X, V = P1

        pind = self.testfuncs[0].pind

        J_coords = C.CorrFunc.jac(X, C.coords)
        J_params = C.CorrFunc.jac(X, C.params)

        A = r_[c_[J_coords, J_params[:,pind]]]

        W, VL = linalg.eig(A, left=1, right=0)
        ind = argsort([abs(eig) for eig in W])[0]
        p = real(VL[:,ind])

        initpoint = zeros(2*C.dim, float)
        initpoint[0:C.dim] = X
        initpoint[C.dim+1:] = p

        X = optimize.fsolve(self.__locate_newton, initpoint, C)
        self.data.psi = X[C.dim+1:]
        X = X[0:C.dim]

        return X, V

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        pind = self.testfuncs[0].pind

        # Finds the new branch
        J_coords = C.CorrFunc.jac(X, C.coords)
        J_params = C.CorrFunc.jac(X, C.params)

        A = r_[c_[J_coords, J_params[:,pind]]]
        #A = r_[c_[J_coords, J_params], [V]]

        W, VR = linalg.eig(A)
        W0 = [ind for ind, eig in enumerate(W) if abs(eig) < 5e-5]
        tmp = real(VR[:,W0[0]])
        V1 = r_[tmp[:-1], 0, 0]
        V1[len(tmp)-1+pind] = tmp[-1]

        """NEED TO FIX THIS!"""
        H = C.CorrFunc.hess(X, C.coords+C.params, C.coords+C.params)
        c11 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V, V) for i in range(H.shape[0])])
        c12 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V, V1) for i in range(H.shape[0])])
        c22 = matrixmultiply(self.data.psi,[bilinearform(H[i,:,:], V1, V1) for i in range(H.shape[0])])

        beta = 1
        alpha = -1*c22/(2*c12)
        V1 = alpha*V + beta*V1
        V1 /= linalg.norm(V1)

        self.found[-1].eigs = W
        self.found[-1].branch = None
        self.found[-1].par = C.freepars[self.testfuncs[0].pind]
        self.found[-1].branch = todict(C, V1)

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        #for n, i in enumerate(ind):
        #    strlist.append('branch angle = ' + repr(matrixmultiply(tocoords(C, self.found[i].V), \
        #        tocoords(C, self.found[i].branch))))

        X = tocoords(C, self.found[-1].X)
        V = tocoords(C, self.found[-1].V)
        C._preTestFunc(X, V)
        strlist.append('Test function #1: ' + repr(self.testfuncs[0](X,V)[0]))

        BifPoint.info(self, C, ind, strlist)


class DHPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'DH', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        J_coords = C.CorrFunc.sysfunc.jac(X, C.coords)
        eigs, LV, RV = linalg.eig(J_coords,left=1,right=1)

        self.found[-1].eigs = eigs

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        BifPoint.info(self, C, ind)



class GHPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'GH', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        J_coords = C.CorrFunc.sysfunc.jac(X, C.coords)
        eigs, LV, RV = linalg.eig(J_coords,left=1,right=1)

        # Check for neutral saddles
        found = False
        for i in range(len(eigs)):
            if abs(imag(eigs[i])) < 1e-5:
                for j in range(i+1,len(eigs)):
                    if C.verbosity >= 2:
                        if abs(eigs[i]) < 1e-5 and abs(eigs[j]) < 1e-5:
                                print('Fold-Fold point found in Hopf!\n')
                        elif abs(imag(eigs[j])) < 1e-5 and abs(real(eigs[i]) + real(eigs[j])) < 1e-5:
                                print('Neutral saddle found!\n')
            elif abs(real(eigs[i])) < 1e-5:
                for j in range(i+1, len(eigs)):
                    if abs(real(eigs[j])) < 1e-5 and abs(real(eigs[i]) - real(eigs[j])) < 1e-5:
                        found = True
                        w = abs(imag(eigs[i]))
                        if imag(eigs[i]) > 0:
                            p = conjugate(LV[:,j]/linalg.norm(LV[:,j]))
                            q = RV[:,i]/linalg.norm(RV[:,i])
                        else:
                            p = conjugate(LV[:,i]/linalg.norm(LV[:,i]))
                            q = RV[:,j]/linalg.norm(RV[:,j])

        if not found:
            del self.found[-1]
            return False

        direc = conjugate(1/matrixmultiply(conjugate(p),q))
        p = direc*p

        # Alternate way to compute 1st lyapunov coefficient (from Kuznetsov [4])

        #print (1./(w*w))*real(1j*matrixmultiply(conjugate(p),b1)*matrixmultiply(conjugate(p),b3) + \
        #   w*matrixmultiply(conjugate(p),trilinearform(D,q,q,conjugate(q))))

        self.found[-1].w = w
        self.found[-1].l1 = firstlyapunov(X, C.CorrFunc.sysfunc, w, J_coords=J_coords, p=p, q=q, check=(C.verbosity==2))
        self.found[-1].eigs = eigs

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        for n, i in enumerate(ind):
            strlist.append('w = ' + repr(self.found[i].w))
            strlist.append('l1 = ' + repr(self.found[i].l1))

        BifPoint.info(self, C, ind, strlist)



# Discrete maps

class LPCPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'LPC', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        J_coords = C.sysfunc.jac(X, C.coords)

        W, VL, VR = linalg.eig(J_coords, left=1, right=1)

        self.found[-1].eigs = W

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        X = tocoords(C, self.found[-1].X)
        V = tocoords(C, self.found[-1].V)
        C._preTestFunc(X, V)
        strlist.append('Test function #1: ' + repr(self.testfuncs[0](X,V)[0]))
        strlist.append('Test function #2: ' + repr(self.testfuncs[1](X,V)[0]))

        BifPoint.info(self, C, ind, strlist)

class PDPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'PD', stop=stop)

    def process(self, X, V, C):
        """Do I need to compute the branch, or will it always be in the direction of freepar = constant?"""
        BifPoint.process(self, X, V, C)

        F = DiscreteMap(C.sysfunc, period=2*C.sysfunc.period)
        FP = FixedPointMap(F)

        J_coords = FP.jac(X, C.coords)
        J_params = FP.jac(X, C.params)

        # Locate branch of double period map
        W, VL = linalg.eig(J_coords, left=1, right=0)
        ind = argsort([abs(eig) for eig in W])[0]
        psi = real(VL[:,ind])

        A = r_[c_[J_coords, J_params], [V]]
        W, VR = linalg.eig(A)
        W0 = argsort([abs(eig) for eig in W])[0]
        V1 = real(VR[:,W0])

        H = FP.hess(X, C.coords+C.params, C.coords+C.params)
        c11 = matrixmultiply(psi,[bilinearform(H[i,:,:], V, V) for i in range(H.shape[0])])
        c12 = matrixmultiply(psi,[bilinearform(H[i,:,:], V, V1) for i in range(H.shape[0])])
        c22 = matrixmultiply(psi,[bilinearform(H[i,:,:], V1, V1) for i in range(H.shape[0])])

        beta = 1
        alpha = -1*c22/(2*c12)
        V1 = alpha*V + beta*V1
        V1 /= linalg.norm(V1)

        J_coords = C.sysfunc.jac(X, C.coords)
        W = linalg.eig(J_coords, right=0)

        self.found[-1].eigs = W
        self.found[-1].branch_period = 2*C.sysfunc.period
        self.found[-1].branch = todict(C, V1)

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        strlist = []
        for n, i in enumerate(ind):
            strlist.append('Period doubling branch angle = ' + repr(matrixmultiply(tocoords(C, self.found[i].V), \
                tocoords(C, self.found[i].branch))))

        BifPoint.info(self, C, ind, strlist)



class NSPoint(BifPoint):
    def __init__(self, testfuncs, flagfuncs, stop=False):
        BifPoint.__init__(self, testfuncs, flagfuncs, 'NS', stop=stop)

    def process(self, X, V, C):
        BifPoint.process(self, X, V, C)

        J_coords = C.sysfunc.jac(X, C.coords)

        eigs, VL, VR = linalg.eig(J_coords, left=1, right=1)

        # Check for nonreal multipliers
        found = False
        for i in range(len(eigs)):
            for j in range(i+1,len(eigs)):
                if abs(imag(eigs[i])) > 1e-10 and \
                   abs(imag(eigs[j])) > 1e-10 and \
                   abs(eigs[i]*eigs[j] - 1) < 1e-5:
                    found = True

        if not found:
            del self.found[-1]
            return False

        self.found[-1].eigs = eigs

        self.info(C, -1)

        return True

    def info(self, C, ind=None):
        if ind is None:
            ind = list(range(len(self.found)))
        elif isinstance(ind, int):
            ind = [ind]

        BifPoint.info(self, C, ind)
