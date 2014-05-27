"""Test pickling for saving and loading various PyDSTool objects"""

import os
from tempfile import mkstemp
from numpy import (
    array,
    float64,
    Inf,
)
from PyDSTool import (
    Interval,
    loadObjects,
    Point,
    Pointset,
    saveObjects,
    Variable,
    Trajectory,
    Events,
)
from PyDSTool.Generator import (
    InterpolateTable,
    Vode_ODEsystem,
    ExplicitFnGen,
    ImplicitFnGen,
)
import pytest

@pytest.fixture
def fname():
    _, fname = mkstemp()
    return fname


def test_saveload_array(fname):
    """Test pickling for saving and loading array"""
    a = array([1, Inf])
    b = [Inf, 0]

    saveObjects([a, b], fname, True)
    loadedObjs = loadObjects(fname)
    assert a[0] == loadedObjs[0][0]
    assert a[1] == loadedObjs[0][1]
    assert b[0] == loadedObjs[1][0]
    os.remove(fname)


def test_saveload_interval(fname):
    """Test pickling for saving and loading 'Interval'"""

    m = Interval('test1', float, (-Inf, 1))
    s = Interval('a_singleton', float, 0.4)
    saveObjects([m, s], fname, True)
    objs_ivals = loadObjects(fname)
    assert objs_ivals[0].get(1) == 1

    # Try loading partial list from a larger file
    objs_part = loadObjects(fname, ['a_singleton'])
    assert objs_part[0] == s
    os.remove(fname)


def test_saveload_point_and_pointset(fname):
    """Test pickling for saving and loading 'Point' and 'Pointset'"""

    x = Point(
        coorddict={
            'x0': [1.123456789],
            'x1': [-0.4],
            'x2': [4000]
        },
        coordtype=float64
    )

    v = Pointset(
        coorddict={
            'x0': 0.2,
            'x1': -1.2
        },
        indepvardict={'t': 0.01},
        coordtype=float,
        indepvartype=float
    )

    saveObjects([x, v], fname, True)
    objs_pts = loadObjects(fname)
    assert objs_pts[0] == x
    assert objs_pts[1] == v
    os.remove(fname)


def test_saveload_variable(fname):
    """Test pickling for saving and loading 'Variable'"""

    var1 = Variable(
        Pointset(
            coordarray=array(range(10), float) * 0.1,
            indepvararray=array(range(10), float) * 0.5
        ),
        name='v1'
    )
    saveObjects(var1, fname, True)
    obj_var = loadObjects(fname)[0]
    assert obj_var(1.5) == var1(1.5)
    os.remove(fname)


def test_saveload_trajectory(fname):
    """Test pickling for saving and loading 'Trajectory'"""

    var1 = Variable(
        Pointset(
            coordarray=array(range(10), float) * 0.1,
            indepvararray=array(range(10), float) * 0.5
        ),
        name='v1'
    )
    var2 = Variable(
        Pointset(
            coordarray=array(range(10), float) * 0.25 + 1.0,
            indepvararray=array(range(10), float) * 0.5
        ),
        name='v2'
    )
    traj = Trajectory('traj1', [var1, var2])
    saveObjects(traj, fname, True)
    traj_loaded = loadObjects(fname)[0]
    assert traj_loaded(2.0) == traj(2.0)
    os.remove(fname)


@pytest.fixture
def interptable():
    timeData = array([0.1, 1.1, 2.1])
    xData = dict(zip(
        ['x1', 'x2'],
        [array([10.2, -1.4, 4.1]), array([0.1, 0.01, 0.4])]))
    itableArgs = {
        'tdata': timeData,
        'ics': xData,
        'name': 'interp',
    }
    return InterpolateTable(itableArgs)


def test_saveload_interpolated_table_generator(interptable, fname):
    """Test pickling for saving and loading 'InterpolateTable' Generator"""
    itabletraj = interptable.compute('itable')
    saveObjects(itabletraj, fname, True)
    obj_itab = loadObjects(fname)
    t = 0.1
    while t < 2.1:
        assert obj_itab[0](t) == itabletraj(t)
        t += 0.1
    os.remove(fname)


def test_saveload_vode_odesystem(interptable, fname):
    """Test pickling for saving and loading 'Vode_ODEsystem' Generator"""

    # Vode object with event and external input trajectory (defined earlier)
    fvarspecs = {
        "w": "k*w + a*itable + sin(t) + myauxfn1(t)*myauxfn2(w)",
        'aux_wdouble': 'w*2 + globalindepvar(t)',
        'aux_other': 'myauxfn1(2*t) + initcond(w)'
    }
    fnspecs = {
        'myauxfn1': (['t'], '2.5*cos(3*t)'),
        'myauxfn2': (['w'], 'w/2')
    }
    ev_args = {
        'name': 'threshold',
        'eventtol': 1e-4,
        'eventdelay': 1e-5,
        'starttime': 0,
        'term': True,
    }
    thresh_ev = Events.makePythonStateZeroCrossEvent('w', 20, 1, ev_args)
    DSargs = {
        'tdomain': [0.1, 2.1],
        'tdata': [0.11, 2.1],
        'ics': {'w': 3.0},
        'pars': {'k': 2, 'a': -0.5},
        'inputs': {'itable': interptable.variables['x1']},
        'auxvars': ['aux_wdouble', 'aux_other'],
        'algparams': {'init_step': 0.01, 'strict': False},
        'events': thresh_ev,
        'checklevel': 2,
        'name': 'ODEtest',
        'fnspecs': fnspecs,
        'varspecs': fvarspecs
    }
    testODE = Vode_ODEsystem(DSargs)
    odetraj = testODE.compute('testode')
    saveObjects([odetraj, testODE], fname, True)
    objs_ode = loadObjects(fname)
    objs_ode[1].diagnostics.clearWarnings()
    assert len(objs_ode[1].diagnostics.warnings) == 0
    odetraj2 = objs_ode[1].compute('testode2')
    assert odetraj2(0.6) == odetraj(0.6)
    assert len(objs_ode[1].diagnostics.warnings) == 1
    os.remove(fname)


def test_saveload_explicitfngen(fname):
    """Test pickling for saving and loading 'ExplicitFnGen'"""

    args = {
        'tdomain': [-50, 50],
        'pars': {'speed': 1},
        'xdomain': {'s': [-1., 1.]},
        'name': 'sine',
        'globalt0': 0.4,
        'pdomain': {'speed': [0, 200]},
        'varspecs': {'s': "sin(globalindepvar(t)*speed)"}
    }
    sin_gen = ExplicitFnGen(args)
    sintraj1 = sin_gen.compute('sine1')
    sin_gen.set(pars={'speed': 2})
    sintraj2 = sin_gen.compute('sine2')
    saveObjects([sin_gen, sintraj1, sintraj2], fname, True)
    objs_sin = loadObjects(fname)
    assert sintraj1(0.55) == objs_sin[1](0.55)
    assert sintraj2(0.55) == objs_sin[2](0.55)
    os.remove(fname)


def test_saveload_implicitfngen(fname):
    """Test pickling for saving and loading 'ImplicitFnGen'"""

    argsi = {
        'varspecs': {
            "y": "t*t+y*y-r*r",
            "x": "t"
        },
        'algparams': {'solvemethod': 'newton', 'atol': 1e-4},
        'xdomain': {'y': [-2, 2]},
        'ics': {'y': 0.75},
        'tdomain': [-2, 0],
        'pars': {'r': 2},
        'vars': ['y'],
        'checklevel': 2,
        'name': 'imptest',
    }

    testimp = ImplicitFnGen(argsi)
    traj1 = testimp.compute('traj1')
    saveObjects([testimp, traj1], fname, True)
    objs_imp = loadObjects(fname)
    assert objs_imp[0].xdomain['y'] == [-2, 2]
    assert traj1(-0.4) == objs_imp[1](-0.4)
    os.remove(fname)


def test_saveload_model(fname):
    """Test pickling for saving and loading 'Model'"""
    pass
