"""
    Test hybrid system with external inputs and mixed classes of
    integrator.

    Robert Clewley, September 2006
"""

from PyDSTool import *
from test_hybrid_extinputs import makeModel, test1, test2, doPlot

m1 = makeModel('Dopri','Dopri')
m2 = makeModel('Vode','Radau')

print("Testing dopri integrator")
plotData1 = test1(m1)
test2(m1)

print("Testing vode + radau integrator")
plotData2 = test1(m2)
test2(m2)
