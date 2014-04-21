"""
Test of the derived_parameters feature (i.e. "RHSdefs" is True)
and also the feature that allows inclusion of the right-hand of an ODE into another ODE.

This is essentially a small extension on the following tutorial example:
    http://www.ni.gsu.edu/~rclewley/PyDSTool/Tutorial/Tutorial_linear.html
"""
from PyDSTool.Toolbox.makeSloppyModel import makeSloppyModel
from numpy.testing import assert_allclose


def test_Sloppy_derived_parameters():
    sloppyModelEg = {
     'assignments': {},
     'derived_params': {'k': 's1/s2', 'q': 's3*s4', 'm': 'q+s5'},
     'functions': {},
     'odes': {'x':'y + _y_RHS', 'y': '-k*x/m'},
     'parameters':{'s1': 1, 's2': 10., 's3': 0.25, 's4': 1, 's5': 0.25},
     'events': {},
     'domains': {}
     }

    model_name = 'test_derived_parameters'
    ics = {'x': 1, 'y': 0.4}
    algparams = {'init_step': 0.1, 'stiff': True}
    sModel = makeSloppyModel(model_name, sloppyModelEg, 'Vode_ODEsystem',
                             algParams=algparams, silent=False,
                             containsRHSdefs=True)

    sModel.compute(trajname='test_derived_params',
                   force=True,
                   ics=ics,
                   tdata=[0, 20],
                   verboselevel=0
                  )
    pts = sModel.sample('test_derived_params')
    assert_allclose(pts[-1]['x'], -0.042398, rtol=1e-4)
    assert_allclose(pts[-1]['y'], -0.073427, rtol=1e-4)
