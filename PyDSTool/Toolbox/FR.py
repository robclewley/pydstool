# Fruchterman-Reingold graph drawing algorithm in Python
# R. Clewley, November 2005

from __future__ import division, absolute_import, print_function
from numpy import array, alltrue, arange, sign
from numpy.linalg import norm
import math
from numpy.random import uniform, normal
from PyDSTool import interp1d, sortedDictLists

class vertex(object):
    def __init__(self, x, y):
        self.pos = array([x, y], 'd')
        self.disp = array([0., 0.])

    def __eq__(self, other):
        return alltrue(self.pos == other.pos)

    def __ne__(self, other):
        return not self.__eq__(other)


class edge(object):
    """v -> u directed edge"""
    def __init__(self, u, v):
        self.u = u #vertex(u.pos[0], u.pos[1])
        self.v = v #vertex(v.pos[0], v.pos[1])

    def __eq__(self, other):
        return self.u == other.u and self.v == other.v

    def __ne__(self, other):
        return not self.__eq__(other)



# very simple cooling scheme
def cool_simple(t, t0, cfac):
    return t*math.exp(-cfac*(t0-t)-0.05)


class cool(object):
    def __init__(self, t0, tarray):
        n = len(tarray)-1
        trange = arange(0., 1+1./n, 1./n)
        self.coolfn = interp1d(trange, t0*tarray)
        self.t0 = t0

    def __call__(self, t):
        # temp t = t0 -> 0, rescaled to 0 -> 1
        return self.coolfn((self.t0-t)/self.t0)


tarray = array([0.98, 0.92, 0.86, 0.78, 0.72,
                0.67, 0.64, 0.60, 0.50, 0.45,
                0.40, 0.35, 0.30, 0.26, 0.20,
                0.16, 0.12, 0.09, 0.04, 0.00])

# for testing purposes
def itercool(t, t0, n=30):
    cooler = cool(t0, tarray)
    for i in range(n+1):
        print(t)
        t=cooler(t)


def out_degree(V, E):
    outd = {}
    for vn in V:
        try:
            outd[vn] = len(E[vn])
        except KeyError:
            outd[vn] = 0
    return outd

def in_degree(V, E):
    ind = {}
    for vn, vv in V.items():
        count = 0
        for vert in V.keys():
            try:
                es = E[vert]
            except KeyError:
                continue
            for e in es:
                if vv == e.u:
                    count += 1
        ind[vn] = count
    return ind


def FR(V, E, W=[0, 1], L=[0, 1], num_its=30, fixed=[]):
    """Fruchterman - Reingold graphing algorithm.

    Changes V and E in-place."""

    area = (W[1]-W[0]) * (L[1]-L[0])
    k = math.sqrt(area/len(V))

    def fa(x):
        return x*x/k

    def fr(x):
        return k*k/x

    od = out_degree(V, E)
    fixedverts = list(fixed.values())

    # initial temperature
    t = t0 = max([W[1]-W[0], L[1]-L[0]])/3.5
#    coolfac = 1./num_its
    # num_its unused now -- assumed 30
    cooler = cool(t0, tarray)

    for i in range(num_its):
#        print "t = %.4f"%t
        # calculate repulsive forces (with weak Gaussian noise)
        for vname, v in V.items():
            # each vertex has two vectors: .pos and .disp
            v.disp = array([0., 0.])
            if vname in fixed:
                continue
            for uname, u in V.items():
                if u != v:
                    D = v.pos - u.pos
                    aD = norm(D)
                    try:
                        v.disp = v.disp + (D/aD) * fr(aD) \
                            + array([normal(0, t), normal(0, t)])
                    except ZeroDivisionError:
#                        print "Clipping to graphics frame caused vertices to",\
#                              "coalesce on iteration %d"%i
#                        print " @ v.pos=(%.4f,%.4f), u.pos=(%.4f,%.4f)"%(v.pos[0],
#                                v.pos[1],u.pos[0],u.pos[1])
#                        print "Trying again..."
                        dx = uniform(W[0],W[1])
                        dy = uniform(L[0],L[1])
                        du = array([dx/3., dy/3.])
                        u.pos = u.pos + du
                        D = v.pos - u.pos
                        aD = norm(D)
                        v.disp = v.disp + (D/aD) * fr(aD) \
                             + array([normal(0, t), normal(0, t)])
        for vname, v in V.items():
            # penalize edges from common vertex that are very close to
            # eachother in angle
            if od[vname] > 2:
                angle_dict = {}
                for e in E[vname]:
                    D = e.u.pos - e.v.pos
                    try:
                        angle_dict[e] = math.atan(D[1]/D[0])
                    except ZeroDivisionError:
                        sgnD = sign(D[1])  # get weird errors from this sometimes
                        if sgnD > 0:
                            angle_dict[e] = math.pi/2
                        else:
                            angle_dict[e] = -math.pi/2
                angles_sorted, vals_sorted = sortedDictLists(angle_dict)
                num_outs = od[vname]
                for i in range(num_outs):
                    dtheta1 = angle_dict[angles_sorted[i]] - angle_dict[angles_sorted[i-1]]
                    ed = angles_sorted[i].u.pos - angles_sorted[i].v.pos
                    if abs(dtheta1) < 0.4:
                        # swap x, y of target vertex to get orthogonal direction
                        # (rotating in -ve angle direction)
                        vd = array([ed[1], -ed[0]])
                        vd = vd/norm(vd)
                        try:
                            c = max([0.07, 0.03/abs(dtheta1)])
                        except ZeroDivisionError:
                            c = 0.07
                        angles_sorted[i].u.disp = angles_sorted[i].u.disp - c*vd
                        # angles are close enough that use same direction for other vertex
                        angles_sorted[i-1].u.disp = angles_sorted[i-1].u.disp + c*vd
                    ni = (i+1)%num_outs
                    dtheta2 = angle_dict[angles_sorted[ni]] - angle_dict[angles_sorted[i]]
                    if abs(dtheta2) < 0.4:
                        # swap x, y of target vertex to get orthogonal direction
                        # (rotating in -ve angle direction)
                        vd = array([ed[1], -ed[0]])
                        vd = vd/norm(vd)
                        try:
                            c = max([0.07, 0.03/abs(dtheta2)])
                        except ZeroDivisionError:
                            c = 0.07
                        angles_sorted[i].u.disp = angles_sorted[i].u.disp + c*vd
                        # angles are close enough that use same direction for other vertex
                        angles_sorted[ni].u.disp = angles_sorted[ni].u.disp - c*vd
        # calculate attractive forces along edges
        for vname, elist in E.items():
            if od[vname] > 3:
                att_scale = 1.75
            else:
                att_scale = 2.5
            for e in elist:
                # each edge is an ordered pair of vertices .v and .u
                D  = e.v.pos - e.u.pos
                aD = norm(D)
                disp = (D/aD) * att_scale*fa(aD)
                if e.v not in fixedverts:
                    e.v.disp = e.v.disp - disp
                if e.u not in fixedverts:
                    e.u.disp = e.u.disp + disp

        # limit the maximum displacement to the temperature t
        # and then prevent vertex being displaced outside frame
        for v in V.values():
            if v in fixedverts:
                continue
            v.pos = v.pos + (v.disp/norm(v.disp)) * min([norm(v.disp), t])
            v.pos[0] = min([W[1], max([W[0], v.pos[0]])])
            v.pos[1] = min([L[1], max([W[0], v.pos[1]])])
        # reduce the temperature as the layout approaches a better configuration
        t = cooler(t)

