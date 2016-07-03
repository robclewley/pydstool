from __future__ import absolute_import, print_function

from PyDSTool import *
from PyDSTool.PyCont.misc import getFlowMaps, getFlowJac, getLeftEvecs
import six

__all__ = ['adjointPRC', 'rotate_phase']

def adjointPRC(model, limit_cycle, vname, freepar, numIntervals=500,
               numCollocation=7, method='standard', spike_est=None,
               saveData=False, doPlot=False, verbosity=0, fname=None,
               force=False):
    """Create adjoint PRC from a limit cycle using AUTO, given an
    ODE_Generator or Model object.

    spike_est (default None) provides an estimate of the spike time in
    the limit cycle, if applicable. If set, this can be useful to
    ensure the PRC is phase-shifted to start at the associated zero
    (in the variable given by vname).

    method: 'standard' is currently the only working method for calculating
      the eigenvetors corresponding to the unit eigenvalue. 'cyclic' selects
      the cyclic method.

    Optionally save the PRC data and make plots.
    Returns a dictionary containing the PRC, cycles with independent
    variable values scaled to [0,1], the Jacobian, the flow maps,
    the eigenvector corresponding to the unit eigenvalue, and
    other algorithmic information.
    """

    assert isinstance(freepar, six.string_types), "Free parameter must be a single string"

    try:
        MDCont = ContClass(model)
    except:
        print("Problem with model argument")
        raise

    # If given an LC, use it for continuation
    if limit_cycle is None:

        PCargs = args(name='EQ1', type='EP-C')
        PCargs.freepars = [freepar]
        PCargs.StepSize = 1e-2
        PCargs.MaxNumPoints = 300
        PCargs.LocBifPoints = 'all'
        PCargs.NumIntervals = numIntervals
        PCargs.NumCollocation = numCollocation
        PCargs.FuncTol = 1e-10
        PCargs.VarTol = 1e-10
        PCargs.SaveEigen = True
        PCargs.SaveFlow = True
        PCargs.SaveJacobian = True
        PCargs.verbosity = 2
        MDCont.newCurve(PCargs)

        print('Computing curve...')
        start = clock()
        MDCont['EQ1'].forward()
        print('done in %.3f seconds!' % (clock()-start))

        #PCargs.name = 'LC1'
        #PCargs.type = 'LC-C'
        #PCargs.MaxNumPoints = 250
        #PCargs.NumIntervals = 20
        #PCargs.NumCollocation = 4
        #PCargs.initpoint = 'EQ1:H1'
        #PCargs.SolutionMeasures = 'all'
        #PCargs.NumSPOut = 50
        #PCargs.FuncTol = 1e-10
        #PCargs.VarTol = 1e-10
        #PCargs.TestTol = 1e-7
        #PyCont.newCurve(PCargs)

        # Create bifn curve for limit cycle
        PCargs.name = 'LC1'
        PCargs.type = 'LC-C'
        PCargs.verbosity = verbosity
        #PCargs.freepars = [freepar]
        #PCargs.initcycle = copy(cycle)
        PCargs.initpoint = 'EQ1:H1'
        PCargs.MinStepSize = 8e-3
        PCargs.MaxStepSize = 8e-3
        PCargs.StepSize = 8e-3
        PCargs.MaxNumPoints = 4
        PCargs.LocBifPoints = []
        PCargs.StopAtPoints = 'B'
        PCargs.NumIntervals = numIntervals
        PCargs.NumCollocation = numCollocation
        PCargs.FuncTol = 1e-10
        PCargs.VarTol = 1e-10
        PCargs.NumSPOut = 3
        PCargs.SolutionMeasures = 'all'
        PCargs.SaveEigen = True
        PCargs.SaveFlow = True
        PCargs.SaveJacobian = True
        MDCont.newCurve(PCargs)

    # Otherwise, find an LC to use
    else:
        try:
            cycle = limit_cycle[MDCont.model.query('variables')]
        except:
            raise ValueError("Model variables must match cycle coordinate names")
        assert vname in cycle.coordnames, "vname argument must be a coordinate"

        assert cycle[-1] == cycle[0], "First and last point of cycle must be equal"

        assert isinstance(limit_cycle, Pointset), \
               "cycle argument must be a Pointset"
        assert remain(MDCont.model.query('variables'),limit_cycle.coordnames)==[], \
               "Model variables must match cycle coordinate names"

        # Create bifn curve for limit cycle
        PCargs = args(name = 'LC1')
        PCargs.type = 'LC-C'
        PCargs.verbosity = verbosity
        PCargs.freepars = [freepar]
        PCargs.initcycle = copy(cycle)
        PCargs.MinStepSize = 1e-6
        PCargs.MaxStepSize = 1e-6
        PCargs.StepSize = 1e-6
        PCargs.MaxNumPoints = 4
        PCargs.LocBifPoints = []
        PCargs.StopAtPoints = 'B'
        PCargs.NumIntervals = numIntervals
        PCargs.NumCollocation = numCollocation
        PCargs.FuncTol = 1e-10
        PCargs.VarTol = 1e-10
        PCargs.NumSPOut = 3
        PCargs.SolutionMeasures = 'all'
        PCargs.SaveEigen = True
        PCargs.SaveFlow = True
        PCargs.SaveJacobian = True
        MDCont.newCurve(PCargs)


    # Perform continuation to calculate PRC by adjoint method
    MDCont['LC1'].forward()
    pt = MDCont['LC1'].getSpecialPoint('RG1')
    if verbosity > 1:
        print("\nRegular point chosen:", pt, "\n")
    J = getFlowJac(pt, verbose=verbosity>1)
    # n = dimension of system
    n = J.shape[0]

    if verbosity > 1:
        print("Computing flow maps...")
    maps, ntst = getFlowMaps(n, pt, 'RG', method=method)

    cycle_true = pt.labels['RG']['cycle']
    nint = MDCont['LC1'].NumIntervals
    ncol = MDCont['LC1'].NumCollocation

    assert isinstance(MDCont.model, NonHybridModel), \
           "Only use a single vector field for the model"
    pars = MDCont.model.query('pars')
    idxs = ncol*arange(nint)
    flow_vecs = array([MDCont.model.Rhs(0, x, pars) for x in \
                       cycle_true[idxs]])

    if verbosity > 1:
        print("Computing left eigenvetors corresponding to the unit ", \
          "eigenvalue along flow")
    evec1 = getLeftEvecs(n, ntst, maps, flow_vecs, method=method,
                         verbose=verbosity>1)
    t = cycle_true['t'][idxs]
    PRC = Pointset(indepvararray=t/t[-1],
                   coordarray=evec1.T/t[-1],
                   coordnames=cycle.coordnames)

    if spike_est is not None:
        assert spike_est >= 0 and spike_est <= t[-1]
        spike_est_phase = spike_est*1./cycle_true['t'][-1]
        res = PRC.find(spike_est_phase)
        if isinstance(res, tuple):
            spike_est_ix = res[0]
        else:
            spike_est_ix = res
        prcv = PRC[vname]
        phase0_ix = find_zero_phase(prcv, spike_est_ix)
        PRC = rotate_phase(PRC, phase0_ix)
        phase0_t = PRC['t'][phase0_ix]*t[-1]
        if verbosity > 1:
            print("spike_est_ix =", spike_est_ix)
            print("phase0_ix =", phase0_ix)
            print("phase0_t =", phase0_t)
        # PRC pointset used a uniform sampling of
        # cycle's time mesh points, so get ix in cycle by
        # multiplying up phase0_ix by the sampling rate
        cycle_true_ph0_ix = phase0_ix*ncol
        cycle_true = rotate_phase(cycle_true, cycle_true_ph0_ix)

    cycle.indepvararray /= cycle['t'][-1]
    T = cycle_true['t'][-1]
    cycle_true.indepvararray /= T

    if doPlot:
        cyc_offset = (max(cycle_true[vname])+min(cycle_true[vname]))/2.
        cyc_scale = abs(max(cycle_true[vname])-min(cycle_true[vname]))
        PRC_scale = abs(max(PRC[vname])-min(PRC[vname]))
        cyc_rescale = PRC_scale/cyc_scale
        plt.figure()
        plot(PRC['t'], PRC[vname], 'r', linewidth=2)
        plot(cycle_true['t'], cyc_rescale*(cycle_true[vname]-cyc_offset))
        plt.title('PRC overlay on (cycle%+.2e)*%.2e'%(cyc_offset,
                                                  cyc_rescale))
        show()

    save_objs = {'starting cycle': cycle, 'PRC cycle': cycle_true,
                   'PyCont args': PCargs, 'starting point': pt,
                   'PRC': PRC, 'nint': nint, 'ncol': ncol,
                   'Jacobian': J, 'maps': maps, 'evec1': evec1,
                   'period': T}
    if saveData:
        # Save the data
        if fname is None:
            savefile = MDCont.model.name + '_adjointPRC.pkl'
        else:
            savefile = str(fname)
        try:

            saveObjects(objlist=save_objs, filename=savefile, force=force)
        except ValueError:
            print("File already exists -- not saving return objects")

    return_objs = save_objs
    return_objs['PyCont'] = MDCont

    return return_objs


def rotate_phase(pts, phase0_ix):
    """Phase shift a pointset (assumed to be a cycle) about index
    phase0_ix, i.e. 'rotate' it to put the point at phase0_ix at the
    beginning of the pointset.

    NOTE: Does not update any label info that might be attached to pts!
    """
    assert phase0_ix > 0 and phase0_ix < len(pts), "phase 0 index out of range"
    try:
        t0 = pts['t'][phase0_ix]
        parameterized = True
    except PyDSTool_KeyError:
        parameterized = False
    pts_0 = pts[phase0_ix:]
    if parameterized:
        pts_0.indepvararray -= t0
    pts_1 = pts[:phase0_ix]
    if parameterized:
        pts_1.indepvararray += (pts['t'][-1]-pts['t'][phase0_ix-1])
    pts_0.extend(pts_1)
    return pts_0


def find_zero_phase(prc, est_ix):
    """prc should be of a single variable of interest"""
    len_prc = len(prc)
    assert est_ix < len(prc)
    assert est_ix >= 0
    zero_ixs = { 1: len_prc,
                -1:-len_prc}  # out of bounds initial values
    for dirn in [1, -1]:
        ix = est_ix + dirn
        while ix >= 0 and ix < len_prc:
            if prc[ix]*prc[est_ix] < 0:
                zero_ixs[dirn] = ix
                break
            else:
                ix += dirn
    dist_forward = zero_ixs[1] - est_ix
    dist_backward = est_ix - zero_ixs[-1]
    if dist_forward < len_prc:
        phase0_ix = zero_ixs[1]
    if dist_backward < dist_forward:
        phase0_ix = zero_ixs[-1]
    else:
        phase0_ix = est_ix
    return phase0_ix
