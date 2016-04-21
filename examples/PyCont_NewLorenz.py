""" EXAMPLE: "New" Lorenz system found in [4], p. 374, Exercise 8.7.12
	Also in [3], p. 153, Exercise 5.6.18

    Drew LaMar, December 2005
"""

from PyDSTool import *

pars = {'a': 0.25,'b': 4.0,'F': 0.0,'G': 0.0}

icdict = {'x': 0.0,'y': 0.0,'z': 0.0}

# Set up model
xstr = '-y*y - z*z - a*x + a*F'
ystr = 'x*y - b*x*z - y + G'
zstr = 'b*x*y + x*z - z'

DSargs = args(name='NewLorenz')
DSargs.pars = pars
DSargs.varspecs = {'x': xstr, 'y': ystr, 'z': zstr}
DSargs.ics = icdict

testDS = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['F']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 200
PCargs.MaxStepSize = 1e-2
PCargs.LocBifPoints = 'H'
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing equilibrium curve...')
start = clock()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Hopf curve
PCargs = args(name='HO1', type='H-C2')
PCargs.initpoint = 'EQ1:H1'
PCargs.freepars = ['F','G']
PCargs.MaxNumPoints = 40

PyCont.newCurve(PCargs)

print('Computing hopf curve...')
start = clock()
PyCont['HO1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont.display(('F','x'))
show()
