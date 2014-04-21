"""
Test of importing a model from PySCes systems biology tool. Requires PySCes to have been installed.

R. Clewley, 2012
"""
from PyDSTool import *
from PyDSTool.Toolbox.PySCes_SBML import *

print("Modify the path variable to indicate where your PySCes models are...")
path = '/pysces/pscmodels/'
#fname = 'pysces_test_linear1.psc'
fname = 'pysces_test_branch1.psc'
#fname = 'pysces_test_pitcon.psc'

gen = get_pysces_model(path+fname, 'Vode')
gen.set(tdata=[0,10])
gen.set(algparams={'init_step': 0.03})
traj=gen.compute('test')
pts=traj.sample()
for x in pts.coordnames:
    plot(pts['t'],pts[x])
