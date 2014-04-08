"""
    Tests for the Generator class.  #3.
    Test MapSystem with external input (an InterpolateTable) and events.

    Robert Clewley, June 2005.
"""

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from PyDSTool import (
    args,
    Events,
    Generator,
    MapSystem,
    Point,
    Pointset,
)


def test_mapsystem():

    # datafn provides an external input to the map system -- this could have
    # been from an experimentally measured signal
    datafn = Generator.InterpolateTable(
        {'name': 'datafunction',
         'tdata': [-30., 0., 5., 10., 30., 100., 180., 400.],
         'ics': {'x': [4., 1., 0., 1., 2., 3., 4., 8.]}
         }
    )

    fvarspecs = {
        "w": "15-a*w + 2*x",
        "v": "1+k*w/10",
        'aux_wdouble': 'w*2 + globalindepvar(t)',
        'aux_other': 'myauxfn(2*t) + initcond(w)'
    }
    fnspecs = {
        'myauxfn': (['t'], '.5*cos(3*t)'),
    }

    # targetlang is optional if default=python is OK
    DSargs = args(name='maptest', fnspecs=fnspecs)
    DSargs.varspecs = fvarspecs
    DSargs.tdomain = [0, 400]
    DSargs.pars = {'k': 2.1, 'a': -0.5}
    DSargs.vars = ['w', 'v']
    DSargs.ttype = int  # force independent variable type to be integer
    DSargs.checklevel = 2
    DSargs.inputs = datafn.variables
    testmap = MapSystem(DSargs)
    assert testmap.pars == DSargs.pars
    assert not testmap.defined
    assert_array_almost_equal([0, 400], testmap.tdata)
    testmap.set(
        ics={'w': 3.0, 'v': 2.},
        tdata=[10, 400])
    assert_array_almost_equal([10, 400], testmap.tdata)
    assert_almost_equal(3.0, testmap.initialconditions['w'])
    assert_almost_equal(2.0, testmap.initialconditions['v'])

    traj1 = testmap.compute('traj1')
    assert testmap.defined
    p = Point({'coorddict': {
        'aux_other': 3.34962540324,
        'aux_wdouble': 98.1981323242,
        'v': 8.64360778809,
        'w': 36.5990661621,
    }})
    assert_almost_equal(p.toarray(), traj1(25).toarray())
    assert testmap.diagnostics.hasWarnings()
    assert_array_almost_equal(testmap.tdata, traj1.indepdomain.get())
    assert_almost_equal(2.70076996547, traj1(30, 'aux_other'))
    ps = traj1(list(range(10, 40)))
    assert isinstance(ps, Pointset)
    assert ps._parameterized

    ev_args = {'name': 'threshold',
               'eventtol': 1e-4,
               'eventdelay': 1e-5,
               'starttime': 0,
               'active': True,
               'term': True,
               'precise': False}
    thresh_ev = Events.makePythonStateZeroCrossEvent('w', 58, 1, ev_args)
    testmap.eventstruct.add(thresh_ev)
    traj2 = testmap.compute('traj2')
    assert testmap.getEventTimes()['threshold'] == [347.]
    assert_array_almost_equal([10, 347], traj2.indepdomain.get())

    assert testmap.diagnostics.hasWarnings()
    assert testmap.diagnostics.findWarnings(10) != []
    assert_almost_equal(58.0, traj2(traj2.indepdomain[1], 'w'))
