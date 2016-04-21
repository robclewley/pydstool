"""PyCont test for saddle-node bifurcations.

   Drew LaMar 2006
"""

from PyDSTool import *

pars = {'r': 1}

icdict = {'x': -1.0}

# Set up model
xstr = 'r - x*x'

DSargs = args(name='SaddleNode')
DSargs.pars = pars
DSargs.varspecs = {'x': xstr}
DSargs.ics = icdict

testDS = Generator.Radau_ODEsystem(DSargs)

#testDS.haveJacobian()

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['r']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 50
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'LP'
PCargs.verbosity = 2
PCargs.SaveEigen = True
PCargs.Corrector = 'Natural'
PyCont.newCurve(PCargs)

print('Computing equilibrium curve...')
start = clock()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont.display(('r','x'), stability=True, linewidth=0.5)
show()
