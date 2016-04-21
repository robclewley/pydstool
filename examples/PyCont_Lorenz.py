""" EXAMPLE: "New" Lorenz system found in [4], p. 374, Exercise 8.7.12
	Also in [3], p. 153, Exercise 5.6.18

    Drew LaMar, December 2005
"""

from PyDSTool import *

pars = {'p1': 280., 'p2': 8./3., 'p3': 10.}

icdict = {'u1': 0.0, 'u2': 0.0, 'u3': 0.0}

# Set up model
u1str = 'p3*(u2 - u1)'
u2str = 'p1*u1 - u2 - u1*u3'
u3str = 'u1*u2 - p2*u3'

DSargs = args(name='Lorenz')
DSargs.fnspecs = {'Jacobian': (['t', 'u1', 'u2', 'u3'],
                        """[[-1*p3, p3, 0.0],
                            [p1-u3, -1.0, -1*u1],
                            [u2, u1, -1*p2]]""")}
                  #'Jacobian_pars': (['t', 'p1', 'p2', 'p3'],
                  #      """[[0.0, 0.0, u2-u1],
                  #          [u1, 0.0, 0.0],
                  #          [0.0, -1*u3, 0.0]]""")}
DSargs.pars = pars
DSargs.varspecs = {'u1': u1str, 'u2': u2str, 'u3': u3str}
DSargs.ics = icdict

testDS = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

# Read cycle data from lor.dat
cycle=transpose(importPointset('../PyDSTool/PyCont/auto/demos/lor/lor.dat'))

# Set up continuation parameters
PCargs = args(name='LC1', type='LC-C')
PCargs.freepars = ['p1']
PCargs.StepSize = 0.5
PCargs.MinStepSize = 0.01
PCargs.MaxStepSize = 25.0
PCargs.NumCollocation = 4
PCargs.NumIntervals = 20
PCargs.initcycle = cycle
PCargs.SolutionMeasures = 'all'
PCargs.LocBifPoints = 'all'
PCargs.FuncTol = 1e-7
PCargs.VarTol = 1e-7
PCargs.TestTol = 1e-4
PCargs.MaxNumPoints = 70
PCargs.SaveJacobian = True
PyCont.newCurve(PCargs)

print("Beginning computation of curve in backward and then forward direction...")
start = clock()
PyCont['LC1'].backward()
PyCont['LC1'].update({'MaxNumPoints': 90, 'NumSPOut': 90})
PyCont['LC1'].forward()
print("\nComputation complete in %.3f seconds.\n" % (clock()-start))

# Plot
PyCont['LC1'].cleanLabels()
PyCont['LC1'].display(stability=True, axes=(1,2,1))
PyCont['LC1'].plot_cycles(cycles=['P1', 'PD1', 'LPC1'], coords=('u1','u2'), color_method='bytype', linewidth=1, axes=(1,2,2))
PyCont['LC1'].plot_cycles(cycles=['PD2', 'P2'], coords=('u1','u2'), color_method='bytype', linestyle='--', linewidth=1, axes=(1,2,2))
plt.legend(loc=2)

PyCont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
PyCont.plot.fig1.axes1.axes.set(xlim=(50, 375), ylim=(14, 50))

PyCont.plot.fig1.axes2.axes.set_title('Cycles')
X = array(PyCont.plot.fig1.axes2.axes.get_position()).flatten()
PyCont.plot.fig1.axes2.axes.set_position((X[0]*1.05, X[1], X[2], X[3]))

PyCont.plot.refresh()
show()
