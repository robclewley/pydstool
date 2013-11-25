from numpy import (
    arange,
    array,
    linspace,
)

from PyDSTool import (
    args,
    numeric_to_traj,
    boundary_containment_by_postproc,
    domain_test,
    Interval,
)


def test_bd_containment():
    """Basic tests for boundary containment features."""

    vals1 = linspace(-3, 3, 20)
    test_traj1 = numeric_to_traj([vals1],
                                 'test', 'x', arange(len(vals1), dtype='d'))

    vals2 = array(
        [1., 1., 1.00000000001, 1.0000001, 1., 0.9999999999, 0.99999])
    test_traj2 = numeric_to_traj([vals2],
                                 'test', 'x', arange(len(vals2), dtype='d'))

    bc = boundary_containment_by_postproc('bc_test', description='Test BCs',
                                          pars=args(thresh=1, interior_dirn=-1, abseps=1e-4))

    # Test whether test trajectory 1 remains below x_ref=1
    # up to rounding error tolerance of 1e-4...
    assert not bc(test_traj1)

    # print "... failed at index", bc._find_idx()
    # print "\nbc.results -->\n"
    # print bc.results

    # Test whether test trajectory 2 remains below x_ref=1
    # up to rounding error tolerance of 1e-4...
    assert bc(test_traj2)

    # Test whether initial condition lies within a domain
    # (includes transversality condition: derivative must point into interior
    # of domain):

    ic_inside = numeric_to_traj([[0.3], [-10.5]], 'test', ['x', 'dxdt'], 0.)
    ic_crit_ok = numeric_to_traj([[1], [-1.]], 'test', ['x', 'dxdt'], 0.)
    ic_crit_not_ok = numeric_to_traj([[1], [1.]], 'test', ['x', 'dxdt'], 0.)
    ic_outside = numeric_to_traj([[1.01], [-1.]], 'test', ['x', 'dxdt'], 0.)

    in_domain = domain_test('domain_test', pars=args(coordname='x',
                                                     derivname='dxdt', interval=[-1, 1], verbose_level=3))

    assert in_domain(ic_inside)
    assert in_domain(ic_crit_ok)
    assert not in_domain(ic_crit_not_ok)
    assert not in_domain(ic_outside)

    # Check that exterior IC failed at trajectory point index '0' with
    #     in_domain._find_idx()==0

    # ------------------------------------------------------------------
    # Test for domain with rounding-error tolerated boundaries:

    in_domain_loose = domain_test('domain_test', pars=args(coordname='x',
                                                           derivname='dxdt', interval=Interval('x', float, [-1, 1], abseps=1e-4),
                                                           verbose_level=0))

    ic_crit_round_ok = numeric_to_traj(
        [[1.000001], [-1.]], 'test', ['x', 'dxdt'], 0.)
    ic_crit_round_not_ok = numeric_to_traj(
        [[1.000001], [1.]], 'test', ['x', 'dxdt'], 0.)

    assert in_domain_loose(ic_inside)
    assert in_domain_loose(ic_crit_ok)
    assert not in_domain_loose(ic_crit_not_ok)
    assert not in_domain_loose(ic_outside)
    assert in_domain_loose(ic_crit_round_ok)
    assert not in_domain_loose(ic_crit_round_not_ok)

    zbd1 = domain_test(
        'z', pars=args(coordname='z', derivname='D_z', interval=[0.0, 1.0], verbose_level=0))
    zbd2 = domain_test(
        'z', pars=args(coordname='z', derivname='D_z', interval=1.0, verbose_level=0))
    ztraj = {}
    ztraj[0] = numeric_to_traj([[1], [0.437]], 'test', ['z', 'D_z'], 0)
    ztraj[1] = numeric_to_traj([[1], [1]], 'test', ['z', 'D_z'], 0)
    ztraj[2] = numeric_to_traj([[1.0], [1]], 'test', ['z', 'D_z'], 0)
    ztraj[3] = numeric_to_traj([[1], [1.0]], 'test', ['z', 'D_z'], 0)
    ztraj[4] = numeric_to_traj([[1], [1.0]], 'test', ['z', 'D_z'], 0)
    ztraj[5] = numeric_to_traj([[1.0], [5.0]], 'test', ['z', 'D_z'], 0)
    assert not zbd1(ztraj[0])
    for i in range(1, 6):
        assert zbd2(ztraj[i])


def test_discrete_domain():
    disc_dom = domain_test('disc_dom_test', pars=args(
        coordname='x', derivname='dxdt', interval=1, verbose_level=3))
    ic_disc = numeric_to_traj([[1], [0]], 'test', ['x', 'dxdt'], 0.)
    assert disc_dom(ic_disc)
