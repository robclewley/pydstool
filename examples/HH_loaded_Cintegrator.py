"""HH (C integrator) re-load test.  Only run after test 'HH_model_Cintegrator.py'
Essentially the same as HH_loaded.py, just using the other .pkl file
"""

from PyDSTool import *

try:
    objs = loadObjects('temp_HH_Cintegrator.pkl')
except:
    print("Only run after test 'HH_model_Cintegrator.py'")
    raise
HH = objs[0]
HHtraj1 = objs[1]
plotData1 = HHtraj1.sample(dt=0.05)
evt1=HH.getEventTimes()['thresh_ev']

HH.set(tdata=[0, 10], pars={'C': 1.2})
HHtraj2 = HH.compute('test2')
plotData2 = HHtraj2.sample(dt=0.05)
evt2=HH.getEventTimes()['thresh_ev']

yaxislabelstr = 'v'
plt.ylabel(yaxislabelstr)
plt.xlabel('t')

vline1=plot(plotData1['t'], plotData1['v'])
vline2=plot(plotData2['t'], plotData2['v'])

plot(evt1, HHtraj1(evt1, 'v'), 'ro')
plot(evt2, HHtraj2(evt2, 'v'), 'ro')
show()
