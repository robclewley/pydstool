""" EXAMPLE: Simple Hamiltonian system (see [10] for continuation setup details)

    Hamiltonian: H(x,y) = y^2/2 - x^2/2 + x^4/4

    Model:  dx/dt = dH/dy + a*dH/dx
            dy/dt = -dH/dx + a*dH/dy
"""

from PyDSTool import *

pars = {'a': -1.}

icdict = {'x': -1., 'y': 0.}

# Set up model
xstr = 'y + a*(x*x*x - x)'
ystr = 'x - x*x*x + a*y'

DSargs = args(name='Hamiltonian')
DSargs.pars = pars
DSargs.varspecs = {'x': xstr, 'y': ystr}
DSargs.ics = icdict

testDS = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['a']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 150
PCargs.MaxStepSize = 1e-2
PCargs.LocBifPoints = 'H'
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing equilibrium curve...')
start = clock()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs.initpoint = {'x': 1, 'y': 0}
PCargs.name = 'EQ2'
PyCont.newCurve(PCargs)

print('Computing equilibrium curve...')
start = clock()
PyCont['EQ2'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'LC1'
PCargs.type = 'LC-C'
PCargs.initpoint = 'EQ1:H1'
PCargs.MinStepSize = 0.000001
PCargs.MaxStepSize = 0.01
PCargs.StepSize = 0.01
PCargs.MaxNumPoints = 100
PCargs.NumSPOut = 10
PCargs.SaveEigen = False
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['LC1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'LC2'
PCargs.initpoint = 'EQ2:H1'
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['LC2'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont.display(curves=['EQ1', 'EQ2'], coords=('x','y'),stability=True)

PyCont['LC1'].plot_cycles(coords=('x','y'))
PyCont['LC2'].plot_cycles(coords=('x','y'))

PyCont.plot.fig1.axes1.EQ1.toggleAll('off', bytype='P')
PyCont.plot.fig1.axes1.EQ2.toggleAll('off', bytype='P')
show()
