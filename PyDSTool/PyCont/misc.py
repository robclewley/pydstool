""" Common functions

    Drew LaMar, March 2006
"""

# ----------------------------------------------------------------------------
# Version History:
#   February 2007
#       Works with SciPy 0.5.2
#
#   September 2006
#       Modified todict() to not update and return entire parsdict.  NOT COMPLETELY
#           SURE OF EFFECT THIS WILL CREATE!
#
#   April 2006
#       Added partition() function used in display() for plotting stability info
#
#   March 2006
#       Added unique() function to make a list of items remove duplicate elements
#
# ----------------------------------------------------------------------------

from __future__ import absolute_import, print_function

from PyDSTool import pointsToPointset, Point, Pointset
from PyDSTool.common import args
from PyDSTool.matplotlib_import import *
from PyDSTool.errors import PyDSTool_ValueError

# THESE ARE REPEATS FROM CONTINUATION!  MAKE SURE AND UPDATE!!!
all_point_types = ['P', 'RG', 'LP', 'BP', 'H', 'BT', 'ZH', 'CP', 'GH',
                   'DH', 'LPC', 'PD', 'NS', 'MX', 'UZ']
all_curve_types = ['EP', 'LP', 'H', 'FP', 'LC', 'FD']

from time import clock
from scipy import linalg
from numpy import dot as matrixmultiply
from numpy import array, float, complex, int, float64, complex64, int32, \
     zeros, divide, subtract, argmax, identity, argsort, conjugate, sqrt, \
     arange, Inf, NaN, isfinite, r_, c_, sign, mod, mat, sum, \
     multiply, transpose, eye, real, imag, ndarray
from math import pi as PI
from copy import copy

#####
_classes = ['IterationError']

_functions = ['iszero', 'isnotzero', 'todict', 'tocoords', 'jac', 'hess',
              'hess3', 'bilinearform', 'trilinearform', 'ijtoind', 'indtoij',
              'testindij', 'wedge', 'invwedge', 'bialttoeig',
              'getFlowMaps', 'getFlowJac', 'getLeftEvecs',
              'firstlyapunov', 'unique', 'partition', 'CheckHopf', 'monotone']

__all__ = _classes + _functions
#####

FLOQ_TOL = 0.01

class IterationError(Exception):
    pass


iszero = lambda x, y: x*y < 0
isnotzero = lambda x, y: x*y > 0
#anyiszero = lambda x, y: (x[0]*y[0] < 0) or (x[1]*y[1] < 0)

def monotone(x, num=None, direc=1):
    """Checks to see if the list 'x' is increasing/decreasing ('direc'
    = 1/-1, respectively). If 'num' is specified, then it checks the
    first 'num' indices in 'x' if num > 0 and the last 'num' indices
    in 'x' if num < 0.
    """
    if num is None:
        ind = list(range(len(x)-1))
    elif num > 1:
        ind = list(range(num-1))
    elif num < -1:
        ind = list(range(num, -1))
    else:
        raise PyDSTool_ValueError('Number of indices must be larger than 1.')

    mon = True
    for i in ind:
        if direc*(x[i+1]-x[i]) < 0:
            mon = False
            break

    return mon

def todict(C, X):
    # Variables
    VARS = dict(zip(C.varslist,array(X)[C.coords]))

    # Free parameters
    for i, par in enumerate(C.freepars):
        VARS.update({par: X[C.params[i]]})
    #for i, par in enumerate(C.freepars):
    #   C.parsdict[par] = X[C.params[i]]
    #VARS.update(C.parsdict)

    # Auxiliary parameters
    for i, par in enumerate(C.auxpars):
        VARS.update({par: X[C.params[C.freeparsdim+i]]})
    return VARS

def tocoords(C, D):
    X = zeros(C.dim, float)
    for i in range(C.varsdim):
        X[i] = D[C.varslist[i]]
    for i in range(C.freeparsdim):
        X[C.varsdim + i] = D[C.freepars[i]]
    for i in range(C.auxparsdim):
        X[C.varsdim + C.freeparsdim + i] = D[C.auxpars[i]]
    return X

def jac(func, x0, ind):
    """Compute (n-1) x m Jacobian of function func at x0, where n =
    len(x0) and m = len(ind).  ind denotes the indices along which to
    compute the Jacobian.

    NOTE:  This assumes func has 1 more variable than it has equations!!
    """

    eps = 1e-6
    n = len(x0)
    m = len(ind)
    J = zeros((n-1, m), float)
    for i in range(m):
        ei = zeros(n, float)
        ei[ind[i]] = 1.0
        J[:,i] = (func(x0+eps*ei)-func(x0-eps*ei))/(2*eps)
    return J

def hess(func, x0, ind):
    """Computes second derivative using 2nd order centered finite difference scheme."""
    eps = 1e-3
    n = len(x0)
    m = len(ind)
    H = zeros((func.m,m,m), float)
    for i in range(m):
        ei = zeros(n, float)
        ei[ind[i]] = 1.0
        for j in range(i,m):
            ej = zeros(n, float)
            ej[ind[j]] = 1.0
            if i == j:
                H[:,i,j] = (-1*func(x0+2*eps*ei) + 16*func(x0+eps*ei) - \
                            30*func(x0) + 16*func(x0-eps*ei) - \
                            func(x0-2*eps*ei))/(12*eps*eps)
            else:
                H[:,i,j] = H[:,j,i] = (func(x0+eps*(ei+ej)) - \
                            func(x0+eps*(ei-ej)) - func(x0+eps*(ej-ei)) + \
                            func(x0-eps*(ei+ej)))/(4*eps*eps)
    return H

def hess3(func, x0, ind):
    """Computes third derivative using hess function."""
    eps = sqrt(1e-3)
    n = len(x0)
    m = len(ind)
    C = zeros((func.m,m,m,m), float)
    for i in range(m):
        ei = zeros(n, float)
        ei[ind[i]] = 1.0
        C[i,:,:,:] = (hess(func,x0+eps*ei,ind) - hess(func,x0-eps*ei,ind))/(2*eps)
    return C

def bilinearform(A, x1, x2):
    return matrixmultiply(transpose(x2),matrixmultiply(A,x1))

def trilinearform(A, x1, x2, x3):
    dim = A.shape
    return matrixmultiply(transpose([bilinearform(A[i,:,:],x1,x2) \
                                     for i in range(dim[0])]),x3)

def ijtoind(i, j):
    """ 0 <= j < i """
    return i*(i-1)/2 + j

def indtoij(ind):
    #size = array([n*(n-1)/2 - k*(k-1)/2 for k in range(1,n+1)])
    #temp = nonzero(size <= ind)[0]
    #j = n - temp
    #i = j+1 + ind - size[temp]

    x = 0.5*(1 + sqrt(1 + 8*ind))
    i = int(x)
    j = int(round(i*(x-i)))

    return i, j

def testindij(n):
    bn = n*(n-1)/2
    print("Testing %d..." % n)
    for ind in range(bn):
        i, j = indtoij(ind)
        ind2 = ijtoind(i, j)
        if ind != ind2:
            print("DAMNIT!\n")
        #print "  %d ---> (%d,%d) ---> %d" % (ind,i,j,ind2)

def wedge(u, v):
    n = u.shape[0]
    bn = n*(n-1)/2
    q = zeros((bn,1), float)

    for ind in range(bn):
        i, j = indtoij(ind)
        q[ind] = u[j]*v[i] - u[i]*v[j]

    return q

def invwedge(q, n):
    ind = argmax(abs(q),0)[0]
    q = q/q[ind]

    i, j = indtoij(ind)

    v1 = zeros((n,1), float)
    v2 = zeros((n,1), float)

    v1[i,0] = 0
    v2[j,0] = 0
    v1[j,0] = 1
    v2[i,0] = 1
    for k in range(0,i):
        if k != j:
            v1[k,0] = q[ijtoind(i, k),0]
    for k in range(i+1,n):
        v1[k,0] = -1*q[ijtoind(k, i),0]

    for k in range(0,j):
        v2[k,0] = -1*q[ijtoind(j, k),0]
    for k in range(j+1,n):
        if k != i:
            v2[k,0] = q[ijtoind(k, j),0]

    return v1, v2

def bialttoeig(q, p, n, A):
    v1, v2 = invwedge(q, n)
    w1, w2 = invwedge(p, n)

    A11 = bilinearform(A,v1,v1)
    A22 = bilinearform(A,v2,v2)
    A12 = bilinearform(A,v1,v2)
    A21 = bilinearform(A,v2,v1)
    v11 = matrixmultiply(transpose(v1),v1)
    v22 = matrixmultiply(transpose(v2),v2)
    v12 = matrixmultiply(transpose(v1),v2)
    D = v11*v22 - v12*v12
    k = (A11*A22 - A12*A21)/D

    return k[0][0], v1, w1

def firstlyapunov(X, F, w, J_coords=None, V=None, W=None, p=None, q=None,
                  check=False):
    if J_coords is None:
        J_coords = F.jac(X, F.coords)

    if p is None:
        alpha = bilinearform(transpose(J_coords),V[:,0],V[:,1]) - \
                1j*w*matrixmultiply(V[:,0],V[:,1])
        beta = -1*bilinearform(transpose(J_coords),V[:,0],V[:,0]) + \
                1j*w*matrixmultiply(V[:,0],V[:,0])
        q = alpha*V[:,0] + beta*V[:,1]

        alpha = bilinearform(J_coords,W[:,0],W[:,1]) + \
                1j*w*matrixmultiply(W[:,0],W[:,1])
        beta = -1*bilinearform(J_coords,W[:,0],W[:,0]) - \
                1j*w*matrixmultiply(W[:,0],W[:,0])
        p = alpha*W[:,0] + beta*W[:,1]

        p /= linalg.norm(p)
        q /= linalg.norm(q)

        direc = conjugate(1/matrixmultiply(conjugate(p),q))
        p = direc*p

    if check:
        print('Checking...')
        print('  |q| = %f' % linalg.norm(q))
        temp = matrixmultiply(conjugate(p),q)
        print('  |<p,q> - 1| = ', abs(temp-1))
        print('  |Aq - iwq| = %f' % linalg.norm(matrixmultiply(J_coords,q) - 1j*w*q))
        print('  |A*p + iwp| = %f\n' % linalg.norm(matrixmultiply(transpose(J_coords),p) + 1j*w*p))

    # Compute first lyapunov coefficient
    B = F.hess(X, F.coords, F.coords)
    D = hess3(F, X, F.coords)
    b1 = array([bilinearform(B[i,:,:], q, q) for i in range(B.shape[0])])
    b2 = array([bilinearform(B[i,:,:], conjugate(q),
                          linalg.solve(2*1j*w*eye(F.m) - J_coords, b1)) \
                 for i in range(B.shape[0])])
    b3 = array([bilinearform(B[i,:,:], q, conjugate(q)) \
                 for i in range(B.shape[0])])
    b4 = array([bilinearform(B[i,:,:], q, linalg.solve(J_coords, b3)) \
                 for i in range(B.shape[0])])
    temp = array([trilinearform(D[i,:,:,:],q,q,conjugate(q)) \
                 for i in range(D.shape[0])]) + b2 - 2*b4

    l1 = 0.5*real(matrixmultiply(conjugate(p), temp))

    return l1

def CheckHopf(C, X):
    J_coords = C.CorrFunc.jac(X, C.coords)
    eigs = linalg.eig(J_coords,left=0,right=0)

    # Check for neutral saddles
    found = False
    for i in range(len(eigs)):
        if abs(real(eigs[i])) < 1e-5:
            for j in range(i+1, len(eigs)):
                if abs(real(eigs[j])) < 1e-5 and \
                   abs(real(eigs[i]) - real(eigs[j])) < 1e-5:
                    found = True
                    T = 2*PI/abs(imag(eigs[i]))

    if found:
        return T
    else:
        return None

def unique(s):
    """
    Return a list of the elements in s, but without duplicates.

    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],
    unique("abcabc") some permutation of ["a", "b", "c"], and
    unique(([1, 2], [2, 3], [1, 2])) some permutation of
    [[2, 3], [1, 2]].

    For best speed, all sequence elements should be hashable.  Then
    unique() will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if list(s).sort() doesn't raise TypeError it's
    assumed that they do enjoy a total ordering.  Then unique() will
    usually work in O(N*log2(N)) time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then unique() will usually work in quadratic
    time.

    Author: Tim Peters (2001/04/06)
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        del u  # move on to the next method
    else:
        return list(u.keys())

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.
    try:
        t = list(s)
        t.sort()
    except TypeError:
        del t  # move on to the next method
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.
    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u

def partition(a, elems):
    """Current issue with neutral points changing past bifurcation
    point. False advertising as well: Not really a general function
    (specialized for my use)
    """
    delems = {'X': []}
    for n in range(len(elems)):
        delems[elems[n]] = []

    loc1 = loc2 = 0
    elem = a[0]
    while loc2 < len(a):
        loc2 += 1
        if loc2 == len(a) or a[loc2] != elem:
            if a[loc2-1] == 'N':
                delems[elem].append([loc1,loc2])
                loc1 = loc2-1
            else:
                delems[elem].append([loc1,loc2+1])
                loc1 = loc2

            if loc2 != len(a):
                # Move loc2 to list of elems larger than 1 in length and save as unknown curve
                if a[loc2] != 'N':
                    while loc2 < len(a)-1 and a[loc2] != a[loc2+1]:
                        loc2 += 1
                    if loc2-loc1 > 1:
                        delems['X'].append([loc1,loc2+1])
                        loc1 = loc2

                elem = a[loc2]

    return delems

def negate(x):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = -1*v
    elif isinstance(x, Point):
        for i in range(len(x)):
            x[i] = -1*x[i]
    elif isinstance(x, Pointset):
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] = -1*x[i][j]
    elif isinstance(x, ndarray):
        x = -1*x
    elif isinstance(x, args):
        for k, v in x.items():
            x[k] = negate(x[k])
    else:
        raise TypeError("Invalid argument type given")

    return x


def getFlowJac(pt, verbose=False):
    try:
        jac0 = pt.labels['LC']['data'].jac0
        jac1 = pt.labels['LC']['data'].jac1
    except AttributeError:
        raise RuntimeError("Malformed point -- no Jacobian information.")

    J = linalg.solve(jac1, jac0)
    if verbose:
        print("Jacobian J*x")
        print("------------\n")
        print(J)
        print("\n")

        print("Check Jacobian")
        print("--------------\n")
        print("   eigs = ", linalg.eig(J)[0])
        print("   eigs = ", pt.labels['LC']['data'].evals)

    return J


def getFlowMaps(n, pt, pttype, method='standard'):
    """
    method: 'standard' is currently the only working method for calculating
      the flow maps.
    """

    try:
        flow = pt.labels[pttype]['flow']  # flow maps (matrices)
    except:
        raise RuntimeError("Malformed point -- no flow map information.")
    ntst = len(flow)//2
    maps = []
    if method=='standard':
        for i in range(ntst):
            I = identity(n)
            for j in mod(arange(i, i + ntst), ntst):
                j = int(j)
                I = linalg.solve(flow[2*j+1], matrixmultiply(flow[2*j], I))
            maps.append(I)
    else:
        raise RuntimeError('method %s not supported'%method)
    return maps, ntst


def getLeftEvecs(n, ntst, maps, flow_vecs, method='standard', verbose=False):
    """Get left eigenvetors w corresponding to the unit eigenvalue of flow,
    normalized so that w.v = 1, where v are flow vectors.

    method: 'standard' is currently the only working method for calculating
      the flow maps and left eigenvectors.
    """
    evals = []
    levecs = []
    revecs = []   # unused unless verbose (for debugging)
    for m in maps:
        w, vl, vr = linalg.eig(m, left=1, right=1)
        evals.append(w)
        levecs.append(vl)
        revecs.append(vr)  # unused unless verbose (for debugging)
    idxs = list(range(ntst))

    if verbose:
        print("Eigenvalues:", [evals[i] for i in idxs])
        check1 = linalg.norm(matrixmultiply(maps[i], revecs[i][:,ind]) - \
                        evals[i][ind]*revecs[i][:,ind])
        check2 =  linalg.norm(matrixmultiply(transpose(levecs[i][:,ind]),
                        maps[i]) - evals[i][ind]*transpose(levecs[i][:,ind]))
        if check1 > 1e-5 or check2 > 1e-5:
            raise RuntimeError("Bad eigenvectors of monodromy matrix")

    if method == 'standard':
        # all left eigenvectors w are given in the array evec1
        evec1 = evec1_standard(idxs, evals, levecs)
    else:
        raise RuntimeError('Method %s not supported'%method)

    ### Normalization
    # (flow vectors eval'd at limit cycle points, assuming mesh points are evenly spaced)
    evn = multiply(evec1, flow_vecs)
    # Divide each column of evec1 through by the length of w*v
    return evec1/array([sum(evn, 1)]*n).T



def evec1_standard(idxs, evals, levecs):
    """Standard method"""
    evec1 = []
    for i in idxs:
        ind = argsort(abs(evals[i]-1.))[0]
        if abs(evals[i][ind]-1) > FLOQ_TOL:
            raise RuntimeError("Bad Floquet multipliers")

        vsgn = 1
        if i > 1:
            vsgn = sign(matrixmultiply(transpose(evec1[-1]),
                                       levecs[i][:,ind]))
        evec1.append(vsgn*levecs[i][:,ind])
    return array(evec1)
