# Library functions for building matrices for rigid body mechanics
# Robert Clewley, (c) 2005, 2006

from __future__ import absolute_import

from numpy import mat, identity, shape, resize, dot, array, transpose
import math
import copy

__all__ = ['augment_3x3_matrix', 'augment_3_vector', 'ZeroPos', 'Id',
           'make_rot', 'make_disp', 'make_T', 'cross', 'dot']

# ---------------------------------------------------------------------------

def augment_3_vector(v=array([0., 0., 0.]), free=False):
    if free:
        freeval = 0.
    else:
        freeval = 1.
    assert shape(v) == (3,), "v argument must be 3-vector -- found "+str(shape(v))
    vaug = resize(v,(4,))
    vaug[3] = freeval
    return vaug

ZeroPos = augment_3_vector([0., 0., 0.])

def augment_3x3_matrix(M=mat(identity(3,'f')),
                   disp=array([0., 0., 0.]), free=False):
    if free:
        freeval = 0.
    else:
        freeval = 1.
    assert shape(M) == (3,3), "M argument must be 3 x 3 matrix"
    assert shape(disp) == (3,), "displacement argument must be 1 x 3 array"
    # resize works properly only on rows ... therefore,
    # use transpose twice. (resize on columns messes up matrix contents)
    Mxr = mat(resize(M.array, (4,3)))
    Mxr[3,:] = [0.,0.,0.]
    MxrcT = mat(resize((Mxr.T).array, (4,4)))
    try:
        # array
        dcol = copy.copy(disp).resize(4)
        dcol[3] = freeval
    except AttributeError:
        # list
        dcol = copy.copy(disp)
        dcol.append(freeval)
    MxrcT[3,:] = dcol
    return MxrcT.T


def make_disp(r):
    """Make 3D displacement
    """
    return augment_3x3_matrix(disp=r)


def make_rot(a):
    """Make 3D rotation (around z, then new x, then new y)
    """
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]

    ca1 = math.cos(a1)
    sa1 = math.sin(a1)
    ca2 = math.cos(a2)
    sa2 = math.sin(a2)
    ca3 = math.cos(a3)
    sa3 = math.sin(a3)

    # rot: new y, by a3
    roty = augment_3x3_matrix(mat([[ca3, 0., sa3], [0., 1., 0.],
                                              [-sa3, 0, ca3]]))

    # rot: new x, by a2
    rotx = augment_3x3_matrix(mat([[1., 0., 0.], [0., ca2, -sa2],
                                              [0., sa2, ca2]]))

    # rot: z, by a1
    rotz = augment_3x3_matrix(mat([[ca1, -sa1, 0.], [sa1, ca1, 0.],
                                              [0., 0., 1.]]))
    return roty*rotx*rotz


def make_T(r, a):
    """Make general 3D transformation (displacement then rotation)
    """
    R = make_rot(a)
    D = make_disp(r)
    return R*D

# Identity matrix
Id = mat(identity(4,'f'))

def cross(a,b):
    """Cross product of two 3D vectors
    """
    c = array([0., 0., 0.])
    c[0] = a[1]*b[2]-a[2]*b[1]
    c[1] = a[2]*b[0]-a[0]*b[2]
    c[2] = a[0]*b[1]-a[1]*b[0]
    return c
