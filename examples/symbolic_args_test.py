
"""

Regression test showing Generator and Event set up when using
Symbolic objects for all var/par references

"""

from PyDSTool import *

x = Var('x')
p = Par('p')

x_dot = Fun(-2 * x + p, [x], 'x_dot')
x_cross = Fun(0.01*Sin(x) + x**2 - 10, [x], 'x_cross')
x_reset = Fun(10 * x, [x], 'x_reset')

xargs = args(name='xgen')
xargs.vars = [x]
xargs.pars = {p: 0}
xargs.ics = {x: 100}
xargs.fnspecs = [x_dot]
xargs.tdomain = [0,365 * 0.1]
#xargs.tdata = [0,1]

xargs.varspecs = {x:x_dot(x)}

xargs.algparams = {'init_step':0.01}
xargs.checklevel = 2

# detect when x crosses 1 by decreasing values
xargs.events = Events.makeZeroCrossEvent(x_cross(x), -1,
                {'name':'cross', 'term':True, 'active':True},
                varnames=[x],
                targetlang='c')

mc = ModelConstructor("mc")
mc.addModelInfo(xargs, 'Dopri_ODEsystem')
mc.mapEvent('xgen', 'cross', 'xgen', {x:x_reset(x)})

model = mc.getModel()
info(model)

model.compute(trajname='testin', tdata=xargs.tdomain,
              ics={x: 100})
smpl = model.sample('testin')

plt.plot(smpl['t'], smpl['x'], 'b-')
plt.show()
