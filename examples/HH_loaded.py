"""HH Re-load test. Only run after test 'HH_model.py'
"""

from PyDSTool import *

try:
    objs = loadObjects('temp_HH.pkl')
except:
    print("Only run after test 'HH_model.py'")
    raise
HH = objs[0]
HHtraj1 = objs[1]
plotData1 = HHtraj1.sample(dt=0.05)
evt1=HH.getEventTimes()['thresh_ev'] # HH was last used to compute HHtraj1 in HH_model.py
assert evt1 == []

# test user interface to auxiliary functions
C=HH.pars['C']
assert HH.auxfns.ptest(4) == 1+4+HH.auxfns.ma(-50)+C, "Failure of user interface to auxiliary functions"
HH.set(pars={'C':101})
assert HH.auxfns.ptest(4) == 1+4+HH.auxfns.ma(-50)+101, "Failure of user interface to auxiliary functions"
HH.set(pars={'C':C})

# other tests
HH.set(tdata=[0, 10], pars={'C': 1.2})
HHtraj2 = HH.compute('test2')
plotData2 = HHtraj2.sample(dt=0.05)
evt2=HH.getEventTimes()['thresh_ev']
assert len(evt2) > 0

yaxislabelstr = 'v'
plt.ylabel(yaxislabelstr)
plt.xlabel('t')

vline1=plot(plotData1['t'], plotData1['v'])
vline2=plot(plotData2['t'], plotData2['v'])

try:
    HHtraj1(evt1, 'v')
except ValueError:
    # OK
    pass
else:
    raise RuntimeError("calling trajectory with empty time sequence should have failed")
plot(evt2, HHtraj2(evt2, 'v'), 'ro')
show()
