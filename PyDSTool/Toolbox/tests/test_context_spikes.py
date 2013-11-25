import os.path as path
from numpy.random import normal
from numpy.testing import assert_allclose
from PyDSTool import importPointset, numeric_to_traj, args
# common.interp1d almost identical to scipy's
# importPointset or use scipy's text file -> array reading functions
from PyDSTool.Toolbox.neuro_data import get_spike_data


def test_get_spike_data():
    """Test Context class and example of qualitative fitting of spikes"""

    sigma = 0.05

    # import doesn't actually create a Pointset (it's slightly misnamed for this
    # purpose but it can do other things...)
    data = importPointset(path.join(path.dirname(__file__), 'test_spikes.dat'),t=0,sep='\t')

    vs = data['vararray'][0]  # pick one of the signals
    ts = data['t']

    traj = numeric_to_traj([vs], 'test_traj', ['x'], ts, discrete=False)


    # ---------------------------------------------------------------------

    # set debug = True and verbose_level = 2 for plot of fit
    is_spike = get_spike_data('one_spike', pars=args(
        height_tol=2000., thresh_pc=0.15,
        fit_width_max=20, weight=0, noise_tol=300,
        tlo=260, width_tol=ts[-1], coord='x', eventtol=1e-2,
        verbose_level=0, debug=False))


    # compare
    # plot(ts, vs)

    # assert ensures that is_spike returns True when passed the particular
    # set of data
    assert is_spike(traj)

    # given that it returned true, try another time for next spike
    is_spike.pars.tlo = 268
    assert is_spike(traj)

    # introspect the is_spike object with info(is_spike)
    # .results values now guaranteed to exist
    assert_allclose(is_spike.results.tlo, 269.98922, rtol=1e-5)
    assert_allclose(is_spike.results.thi, 272.0108, rtol=1e-5)
    assert_allclose(is_spike.results.spike_time, 270.5574, rtol=1e-5)
    assert_allclose(is_spike.results.spike_val, 5117.07697, rtol=1e-4)
