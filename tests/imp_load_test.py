"""Implicit trajectory load test"""

from PyDSTool import *
import os

print "Implicit trajectory load test"

impgen, imptraj = loadObjects('temp_imp.pkl')
assert impgen.xdomain['y'] == [-2, 2]
assert allclose(imptraj(-0.4)['y'], 1.85903)

impgen.set(pars={'r':10.}, xdomain={'y': [-10,10]})
imptraj2 = impgen.compute('test2')
try:
    imptraj2(-0.4)
except PyDSTool_BoundsError:
    pass
else:
    raise AssertionError("Bounds checking messed up")

impgen.set(xdomain={'z': [-5,5]})
imptraj2 = impgen.compute('test2')
assert allclose(imptraj2(-0.4)['y'], 9.47924)

os.remove('temp_imp.pkl')

print "   ...passed"
