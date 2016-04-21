""" EXAMPLE: Catalytic Oscillator system found in [3]

    Drew LaMar, December 2005
"""

from PyDSTool import *

pars = {'q1': 2.5, 'q2': 1.92373, 'q3': 10, 'q4': 0.0675, 'q5': 1, 'q6': 0.1, 'k': 0.4}

icdict = {'x': 0.0014673, 'y': 0.826167, 's': 0.123119}

# Set up model
auxfndict = {'z': (['x', 'y', 's'], '1 - x - y - s')}

xstr = '2*q1*z(x,y,s)*z(x,y,s) - 2*q5*x*x - q3*x*y'
ystr = 'q2*z(x,y,s) - q6*y - q3*x*y'
sstr = 'q4*z(x,y,s) - k*q4*s'

DSargs = args(name='CatalyticOscillator')
DSargs.pars = pars
DSargs.varspecs = {'x': xstr, 'y': ystr, 's': sstr}
DSargs.fnspecs = auxfndict
DSargs.ics = icdict

testDS = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['q2']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 175
PCargs.MaxStepSize = 1e-2
PCargs.LocBifPoints = ['LP','H']
PCargs.verbosity = 2
PCargs.SaveEigen = True
PyCont.newCurve(PCargs)

print('Computing equilibrium curve...')
start = clock()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Hopf curve
PCargs.name = 'HO1'
PCargs.type = 'H-C1'
PCargs.initpoint = 'EQ1:H2'
PCargs.freepars = ['q2', 'k']
PCargs.MaxStepSize = 1e-2
PCargs.LocBifPoints = 'all'
PCargs.MaxNumPoints = 100

PyCont.newCurve(PCargs)

print('Computing hopf curve...')
start = clock()
PyCont['HO1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Fold curve
PCargs.name = 'FO1'
PCargs.type = 'LP-C'
PCargs.initpoint = 'EQ1:LP1'
PCargs.freepars = ['q2', 'k']
PCargs.LocBifPoints = 'all'
PCargs.MaxNumPoints = 110

PyCont.newCurve(PCargs)

print('Computing fold curve...')
start = clock()
PyCont['FO1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont.display(('q2','x'), axes=(3,1,1))
plt.xlabel('')
plt.xticks([])
plt.title('')

PyCont.display(('q2','y'), axes=(3,1,2))
plt.xlabel('')
plt.xticks([])
plt.title('')

PyCont.display(('q2','s'), axes=(3,1,3))
plt.title('')

PyCont.plot.toggleAll('off', bytype='P')
plt.show()
