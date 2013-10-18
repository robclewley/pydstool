import PyDSTool as dst
import numpy as np
from matplotlib import pyplot as plt

# we must give a name
DSargs = dst.args(name='Calcium channel model')
# parameters
DSargs.pars = { 'vl': -60,
               'vca': 120,
                 'i': 0,
                'gl': 2,
               'gca': 4,
                 'c': 20,
                'v1': -1.2,
                'v2': 18  }
# auxiliary helper function(s) -- function name: ([func signature], definition)
DSargs.fnspecs  = {'minf': (['v'], '0.5 * (1 + tanh( (v-v1)/v2 ))') }
# rhs of the differential equation, including dummy variable w
DSargs.varspecs = {'v': '( i + gl * (vl - v) - gca * minf(v) * (v-vca) )/c',
                   'w': 'v-w' }
# initial conditions
DSargs.ics      = {'v': 0, 'w': 0 }

DSargs.tdomain = [0,30]                         # set the range of integration.
ode  = dst.Generator.Vode_ODEsystem(DSargs)     # an instance of the 'Generator' class.
traj = ode.compute('polarization')              # integrate ODE
pts  = traj.sample(dt=0.1)                      # Data for plotting

# PyPlot commands
plt.plot(pts['t'], pts['v'])
plt.xlabel('time')                              # Axes labels
plt.ylabel('voltage')                           # ...
plt.ylim([0,65])                                # Range of the y axis
plt.title(ode.name)                             # Figure title from model name

plt.figure()
plt.hold(True)                                  # Sequences of plot commands will not clear the existing figures
for i, v0 in enumerate(np.linspace(-80,80,20)):
    ode.set( ics = { 'v': v0 } )                # Initial condition
    # Trajectories are called pol0, pol1, ...
    # sample them on the fly to create Pointset tmp
    tmp = ode.compute('pol%3i' % i).sample()
    plt.plot(tmp['t'], tmp['v'])
plt.xlabel('time')
plt.ylabel('voltage')
plt.title(ode.name + ' multi ICs')
plt.show()

# Prepare the system to start close to a steady state
ode.set(pars = {'i': -220} )       # Lower bound of the control parameter 'i'
ode.set(ics =  {'v': -170} )       # Close to one of the steady states present for i=-220

PC = dst.ContClass(ode)                 # Set up continuation class

PCargs = dst.args(name='EQ1', type='EP-C')  # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
PCargs.freepars     = ['i']                      # control parameter(s) (it should be among those specified in DSargs.pars)
PCargs.MaxNumPoints = 450                        # The following 3 parameters are set after trial-and-error
PCargs.MaxStepSize  = 2
PCargs.MinStepSize  = 1e-5
PCargs.StepSize     = 2e-2
PCargs.LocBifPoints = 'LP'                       # detect limit points / saddle-node bifurcations
PCargs.SaveEigen    = True                       # to tell unstable from stable branches

PC.newCurve(PCargs)
PC['EQ1'].forward()
PC['EQ1'].display(['i','v'], stability=True, figure=3)        # stable and unstable branches as solid and dashed curves, resp.

PCargs = dst.args(name='SN1', type='LP-C')
PCargs.initpoint    = 'EQ1:LP2'
PCargs.freepars     = ['i', 'gca']
PCargs.MaxStepSize  = 2
PCargs.LocBifPoints = ['CP']
PCargs.MaxNumPoints = 200
PC.newCurve(PCargs)
PC['SN1'].forward()
PC['SN1'].backward()
PC['SN1'].display(['i','gca'], figure=4)