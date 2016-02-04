""" EXAMPLE: ABC Reaction  system found in [7]

    Drew LaMar, January 2006
"""

from PyDSTool import *

pars = {'p1': 0.1, 'p2': 1, 'p3': 1.5, 'p4': 8, 'p5': 0.04}

icdict = {'u1': 0.13304, 'u2': 0.13223, 'u3': 0.42833}

# Set up model
u1str = '-1*u1 + p1*(1 - u1)*exp(u3)'
u2str = '-1*u2 + p1*exp(u3)*(1 - u1 - p5*u2)'
u3str = '-1*u3 - p3*u3 + p1*p4*exp(u3)*(1 - u1 + p2*p5*u2)'

DSargs = args(name='ABCReaction')
DSargs.pars = pars
DSargs.varspecs = {'u1': u1str, 'u2': u2str, 'u3': u3str}
DSargs.ics = icdict

testDS = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['p1']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 580
PCargs.MaxStepSize = 1e-2
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PCargs.SaveEigen = True
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Hopf Curve -- Curiously screws up further out on curve.  Will take a closer look later .....
PCargs = args(name='HO1', type='H-C2')
PCargs.initpoint = 'EQ1:H1'
PCargs.freepars = ['p1', 'p5']
PCargs.MaxNumPoints = 40
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing Hopf curve...')
start = clock()
PyCont['HO1'].backward()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont.display(('p1','u1'), stability=True)
show()
