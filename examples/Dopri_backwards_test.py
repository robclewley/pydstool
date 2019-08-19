"""A demonstration of backwards integration using Dopri or Radau.
   Substitute Radau for Dopri in the code.

   Robert Clewley, August 2005.
"""

from PyDSTool import *
from time import perf_counter


# ------------------------------------------------------------

DSargs = args()
DSargs.varspecs = {'x': 'k*x'}
DSargs.pars = {'k': .1}
DSargs.xdomain = {'x': [0, 1e15]}
DSargs.tdomain = [-20, 20]
DSargs.algparams = {'max_step': 1, 'init_step': 0}
DSargs.ics = {'x': 0.1}
DSargs.name = 'my_ode'
ode = Generator.Dopri_ODEsystem(DSargs)
ode.set(tdata=[0, 10])
trajf = ode.compute('testf')
print("======================")
DSargsB = args()
DSargsB.varspecs = {'x': 'k*x'}
DSargsB.pars = {'k': .1}
DSargsB.xdomain = {'x': [0, 1e15]}
DSargsB.tdomain = [-20, 20]
DSargsB.algparams = {'max_step': 1, 'init_step': 0}
DSargsB.ics = {'x': 0.1}
DSargsB.name = 'my_ode_back'
odeb = Generator.Dopri_ODEsystem(DSargsB)
odeb.set(tdata=[-10, 0])
trajb = odeb.compute('testb', 'b')

print('Preparing plot (curves should align to be a single smooth curve)')

plotDataf = trajf.sample(dt=0.1)
plotDatab = trajb.sample(dt=0.1)
yaxislabelstr = 'x'
plt.ylabel(yaxislabelstr)
plt.xlabel('t')
vlinef=plot(plotDataf['t'], plotDataf['x'])
vlineb=plot(plotDatab['t'], plotDatab['x'])
show()
