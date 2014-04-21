from __future__ import print_function

from numpy import arange
from numpy.linalg import norm
import numpy.random as rand

# from PyDSTool import *
from PyDSTool.Toolbox.data_analysis import (
    find_nearby_ball,
    find_diameter,
    find_recurrences,
)
from PyDSTool.Toolbox.synthetic_data import (
    # generate_swirl,
    generate_spiral,
)


def test_find_nearby_ball():

    data = rand.random_sample((30, 2))  # 2D points uniform over [0,1] x [0,1]
    refix = 10
    r = 0.2

    ixs = find_nearby_ball(data, refix, r)
    pts = data[ixs]

    for p in pts:
        assert norm(p - data[refix]) <= r


def test_find_diameter(with_plot=False):
    ##swirl = generate_swirl(1000, 1, 0.1, 1.5, 0.0064, 0.00005, 0.000, 2, 0.01)
    #  Generating 2D spiral data with added noise (s.d. 0.05)
    spiral = generate_spiral(150, 2, 1., 0.1, 0.1, 1, 0.05)

    # test function for radius of data set
    diam = find_diameter(spiral, 0.5)
    print("\nDiameter of spiral to within +/- 0.5 is", diam)
    assert abs(diam - 2.75) < 0.5
    #centre_ix = find_central_point(spiral)
    centre_ix = 80  # pick a point out on the arm

    times = 0.1 * arange(0, len(spiral))
    rec_info = find_recurrences(spiral, centre_ix, diam / 3, times)

    print("\nall data indices in ball: ", rec_info.ball_ixs)

    print("\npartitions found (by index): ", rec_info.partitions)

    print("\nnumber of contiguous traj points in a ball near centre_ix = ", rec_info.partition_lengths)
    print("\nrecurrence times for a ball around the black square of radius %.3f" % (diam / 3))
    print("for the partitions found are ", rec_info.rec_times)

    if with_plot:
        from matplotlib.pyplot import figure, plot, show
        figure(1)
        plot(spiral[:, 0], spiral[:, 1], 'ro')
        plot([spiral[centre_ix, 0]], [spiral[centre_ix, 1]], 'ks')
        for p in rec_info.partitions:
            x = spiral[p[0]:p[1]][:, 0]
            y = spiral[p[0]:p[1]][:, 1]
            plot(x, y, 'kx')
        show()

if __name__ == '__main__':
    # x-y projection
    test_find_diameter(True)
