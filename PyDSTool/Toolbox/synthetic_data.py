"""Helper functions for creating synthetic data

Robert Clewley, 2006, 2007.

"""


from PyDSTool import *
from numpy import random, array, dot, zeros, transpose
from numpy.linalg import norm, inv

_generators = ['generate_swirl', 'generate_discspiral', 'generate_hypercube',
              'generate_spiral']

_hypercuboid_utils = ['sup_norm', 'inf_norm', 'maxs', 'mins',
                      'max_euclidean_distance']

__all__ = _generators + _hypercuboid_utils

# ----------------------------------------------------------------------------

def generate_swirl(N, L, zmin, zmax, cycle_rate, bias_x, bias_y, roll_num=0,
                   noise=0, fromcentre=True):
    """Generate N points on a 3D 'swirl' (a cyclic trajectory on a
    1 x L "swiss roll" manifold).

    cycle_rate is measured in radians.
    roll_num is the number of full rolls in the swiss roll.

    The origin is at the centre of the set unless fromcentre=False.
    """
    X = zeros((N, 3), float)
    r = 0.5
    assert L > 0
    assert zmax >= 1
    assert zmin > 0
    assert zmax > zmin
    zscale = (zmax - zmin)/L
    def roll(x):
        r = zmax - x*zscale
        rho = x/L*(roll_num*2*pi)
        return (r*cos(rho), r*sin(rho))
    theta = 0
    for i in range(N):
        theta += cycle_rate
        x = r*cos(theta)+bias_x*i
        x_rolled, z_rolled = roll(x)
        X[i,:] = array([x_rolled + random.normal(0,noise),
                        r*sin(theta) + bias_y*i + random.normal(0,noise),
                        z_rolled + random.normal(0,noise)])
    return X


def generate_discspiral(N1, N2, D, radius, cycle_rate,
                        num_spirals=1, noise=0):
    """Generate N1 points on a planar disc and N2 points on up to two spirals
    attached to it, also in the plane. The data are embedded in an ambient
    space of dimension D.
    The cycle_rate (radians per point) determines how much of the spiral is
    created.
    num_spirals can be 0, 1, or 2
    """
    assert D > 1
    assert num_spirals in [0,1,2]
    X = zeros((N1+num_spirals*N2, D), float)
    # first take care of the disc
    assert radius >= 1
    X[0:N1,:2] = generate_ball(N1, 2, radius)
    for i in range(N1):
        X[i,2:D] = array([random.normal(0,noise)]*(D-2), float)
    # now do the spiral(s)
    theta = sqrt(radius*radius-1)
    for i in range(N2):
        theta += cycle_rate
        r = sqrt(theta*theta+0.8)
        X[N1+i,:] = array([r*cos(theta)+random.normal(0,noise),
                           r*sin(theta)+random.normal(0,noise)] + \
                          [random.normal(0,noise)]*(D-2))
        if num_spirals == 2:
            X[N1+N2+i,:] = array([r*cos(theta+pi)+random.normal(0,noise),
                               r*sin(theta+pi)+random.normal(0,noise)] + \
                                 [random.normal(0,noise)]*(D-2))
    return X


def generate_spiral(N, D, radius, cycle_rate, expand_rate=1.,
                        num_spirals=1, noise=0):
    """Generate N points on up to two spirals in the plane.
    The data are embedded in an ambient space of dimension D.
    The cycle_rate (radians per point) determines how much of the spiral is
    created.
    radius is the starting radius of the spiral.
    num_spirals can be 0, 1, or 2
    """
    assert D > 1
    assert num_spirals in [0,1,2]
    X = zeros((num_spirals*N, D), float)
    theta = sqrt(radius*radius-1)
    for i in range(N):
        theta += cycle_rate
        r = sqrt(theta*theta+0.8)*expand_rate
        X[i,:] = array([r*cos(theta)+random.normal(0,noise),
                           r*sin(theta)+random.normal(0,noise)] + \
                          [random.normal(0,noise)]*(D-2))
        if num_spirals == 2:
            X[N+i,:] = array([r*cos(theta+pi)+random.normal(0,noise),
                               r*sin(theta+pi)+random.normal(0,noise)] + \
                                 [random.normal(0,noise)]*(D-2))
    return X

def generate_hypercube(N, D, length, fromcentre=True):
    """Generate a length-N uniformly-distributed random set of data points in a
    D-dimensional hypercube having side-length given by third parameter.

    The origin is at the centre of the set unless fromcentre=False.
    """
    X = zeros((N, D), float)
    if fromcentre:
        for i in range(N):
            X[i,:] = array([random.uniform(-length/2., length/2.) for j in range(D)],
                           float)
    else:
        for i in range(N):
            X[i,:] = array([random.uniform(0, length) for j in range(D)],
                           float)
    return X

def generate_ball(N, D, radius):
    """Generate a length-N uniformly-distributed random set of data points in a
    D-dimensional ball of radius given by the third parameter. Points are found
    by elimination of points from hypercubes and thus this method is not
    recommended for large D!
    """
    X = zeros((N, D), float)
    twopi = 2*pi
    assert D > 1
    foundpts = 0
    outsidepts = 0
    while foundpts < N:
        hcube = generate_hypercube(N, D, 2*radius)
        for x in hcube:
##            xr = norm(x)
            if norm(x) < radius:
                X[foundpts,:] = x
                foundpts += 1
##                print x, xr, "inside"
                if foundpts == N:
                    break
            else:
                outsidepts += 1
##                print x, xr, "outside"
    print("Number of points outside ball that were discarded = ", outsidepts)
    return X


# -----------------------------------------------------------------------------

# Utilities for metrics on synthetic data distributed in a hypercuboid

def sup_norm(p, lengths=None):
    """sup norm for data distributed in a hypercuboid with given lengths."""
    return max(maxs(p), lengths)

def inf_norm(p, lengths=None):
    """inf norm for data distributed in a hypercuboid with given lengths."""
    return min(mins(p), lengths)

def maxs(p, lengths=None):
    """Return ordered array of max distances from a point to the edges of
    a hypercuboid.
    """
    if lengths is None:
        lengths = [1.] * len(p)
    a = array([max([p[i],lengths[i]-p[i]]) for i in range(len(p))])
    a.sort()
    return a

def mins(p, lengths=None):
    """Return ordered array of min distances from a point to the edges of
    a hypercuboid.
    """
    if lengths is None:
        lengths = [1.] * len(p)
    a = array([min([p[i],lengths[i]-p[i]]) for i in range(len(p))])
    a.sort()
    return a

def max_euclidean_distance(lengths):
    """Returns the largest Euclidean distance possible in a D-dimensional
    hypercuboid with given side-lengths."""
    return norm(lengths, 2)
