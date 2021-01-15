"""Demonstration of using scipy splines as a callback function from the RHS of an ODE definition,
for an application with Predator-Prey models.

Currently only works for VODE integrator. Dopri version in development.
2010.
"""

from PyDSTool import *
import sys
from scipy.interpolate import UnivariateSpline


x = linspace(0,80,20)
y = 3.3*x/(4.3+x)
pred_spline = spy.interpolate.UnivariateSpline(x,y,s=0)


DSargs = args(name='Calcium')
DSargs.pars = { 'delta': 1.25,
                'Nin' : 80,
                'mort': 0.055,
                'lambd': 0.4,
                'bB': 2.25,
                'Kb': 15,
                'epsilon': .25}



DSargs.varspecs = { 'N': '(delta * (Nin - N) - (pred_spline_val * D))',
                    'D': '(pred_spline_val * D) - (( ( (bB * D)/(Kb + D)) * B / epsilon)) - (delta * D)',
                    'S': '((((bB * D)/(Kb + D)) * S ) - (delta + mort + lambd) * S)',
                    'B': '((((bB * D)/(Kb + D)) * S ) - (delta + mort) * B)'}

DSargs.ics      = {'N': 6.3121172183, 'D':46.9266049603, 'S':3.75421042894, 'B':4.90492627563}

DSargs.vfcodeinsert_start = 'pred_spline_val = ds.pred_spline(N)'
DSargs.ignorespecial = ['pred_spline_val']
DSargs.tdomain = [0,60]
ode  = Generator.Vode_ODEsystem(DSargs)

# provide the callback function for the spline
ode.pred_spline = pred_spline
traj = ode.compute('polarization')
pd   = traj.sample()

arraysize=10
maxD=linspace(1,arraysize,arraysize)
minD=linspace(1,arraysize,arraysize)
finalD=linspace(1,arraysize,arraysize)
deltaD=linspace(1,arraysize,arraysize)
maxB=linspace(1,arraysize,arraysize)
minB=linspace(1,arraysize,arraysize)
finalB=linspace(1,arraysize,arraysize)

plt.clf()                   # Clear the screen
plt.hold(True)              # Sequences of plot commands will not clear the existing figures
for j, v0 in enumerate(linspace(.001,1.75,arraysize)):
    ode.set( pars = { 'delta': v0 } )                     # Dilution Rates
    tmp = ode.compute('pol%3f' % j).sample()     # Trajectories are called pol0, pol1, ...


### Prepare the system to start close to a steady state
ode.set(pars = {'delta': 1.25} )       # Lower bound of the control parameter 'i'  , 'w':6.31211705861
ode.set(ics =  {'N': 6.3121172183, 'D':46.9266049603, 'S':3.75421042894, 'B':4.90492627563} )       # Close to one of the steady states present for i=-220

PyCont = ContClass(ode)                 # Set up continuation class
PCargs = args(name='EQ1', type='EP-C')  # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
PCargs.freepars     = ['delta']                      # control parameter(s) (it should be among those specified in DSargs.pars)
PCargs.MaxNumPoints = 400# 300                     # The following 3 parameters are set after trial-and-error
PCargs.MaxStepSize  = 2
PCargs.MinStepSize  = 1e-5
PCargs.StepSize     = 2e-3
PCargs.LocBifPoints = 'all' #['BP', 'LP','H', 'B']                       # detect limit points / saddle-node bifurcations
PCargs.SaveEigen    = True                       # to tell unstable from stable branches
PCargs.verbosity = 2

PyCont.newCurve(PCargs)
PyCont['EQ1'].forward()
PyCont['EQ1'].backward()
PyCont['EQ1'].info()

PyCont['EQ1'].display(('delta','D'), stability=True, color='green')

print("Avoiding computing the Hopf curves until C code can be augmented with equivalent spline code")
plt.show()
# STOP HERE
1/0

###############

PCargs = args(name='HO1', type='H-C2')
PCargs.initpoint = 'EQ1:H1'
PCargs.freepars = ['delta', 'Nin']
PCargs.MaxStepSize = 1e-3
PCargs.LocBifPoints = ['GH', 'BT', 'ZH']
PCargs.MaxNumPoints = 500   # 1000


PyCont.newCurve(PCargs)
PyCont['HO1'].backward()
PyCont['HO1'].info()
PyCont.plot.clearall()


PCargs = args(name='FO1', type='LC-C')
PCargs.initpoint = 'EQ1:H1'
PCargs.freepars = ['delta']
PCargs.LocBifPoints = 'all'
PCargs.MaxNumPoints = 1500  #1000
PCargs.SolutionMeasures = 'all'


PyCont.newCurve(PCargs)
PyCont['FO1'].forward()
PyCont['FO1'].info()
PyCont.plot.clearall()




PCargs = args(name='FO2', type='LC-C')
PCargs.initpoint = 'EQ1:H2'
PCargs.freepars = ['delta']
PCargs.LocBifPoints = 'all'
PCargs.MaxNumPoints = 500 #1000
PCargs.SolutionMeasures = 'all'

PyCont.newCurve(PCargs)
PyCont['FO2'].forward()
PyCont['FO2'].info()







print('done in %.3f seconds!' % (perf_counter()-start))

ax1=subplot(111)
PyCont['FO1'].display(('delta','D_max'),axes=ax1, stability=True, color='green')
hold(True)
PyCont['FO1'].display(('delta','D_min'),axes=ax1, stability=True, color='green')
hold(True)
PyCont['FO2'].display(('delta','D_max'),axes=ax1, stability=True, color='green')
hold(True)
PyCont['FO2'].display(('delta','D_min'),axes=ax1, stability=True, color='green')

hold(True)
PyCont['EQ1'].display(('delta','D'),axes=ax1, stability=True, color='green')
ylim(0,80)
ylabel('Chlorella vulgaris (5 X 10^4 cells/ml)')
xlabel('Dilution rate delta(per day)')
title('Fussmann Hopf Science 2001')


ax2=twinx()
hold(True)
PyCont['FO1'].display(('delta','B_max'),axes=ax2, stability=True, color='black', linewidth=4)
hold(True)
PyCont['FO1'].display(('delta','B_min'),axes=ax2, stability=True, color='black', linewidth=4)
hold(True)
PyCont['FO2'].display(('delta','B_max'),axes=ax2, stability=True, color='black', linewidth=4)
hold(True)
PyCont['FO2'].display(('delta','B_min'),axes=ax2, stability=True, color='black', linewidth=4)

hold(True)
PyCont['EQ1'].display(('delta','B'),axes=ax2, stability=True, color='black', linewidth=4)
xlim(0, 1.75)
ylim(0,12)
ylabel('Brachionus calyciflorus (females/0.2 ml)')
title('Fussmann Hopf Science 2001')
xlabel('Dilution rate delta(per day)')

PyCont.plot.toggleLabels(visible='off',  bylabel=None, byname=None, bytype=None)
PyCont.plot.togglePoints(visible='off',  bylabel=None, byname=None, bytype=None)

show()





