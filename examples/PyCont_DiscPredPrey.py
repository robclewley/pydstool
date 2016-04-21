""" EXAMPLE: Discrete Predator-Prey system (from [9])

    Drew LaMar, March 2006
"""

from PyDSTool import *

pars = {'p1': 0., 'p2': 2., 'p3': 1.}

icdict = {'u1': 0., 'u2': 0.}

# Set up model
u1str = 'p1*u1*(1-u1) - p2*u1*u2'
u2str = '(1-p3)*u2 + p2*u1*u2'

DSargs = args(name='DiscretePredatorPrey')
DSargs.pars = pars
DSargs.varspecs = {'u1': u1str, 'u2': u2str}
DSargs.ics = icdict
DSargs.ttype = int

testDS = Generator.MapSystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='FP1', type='FP-C')
PCargs.freepars = ['p1']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 56
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['FP1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs = args(name='FP2', type='FP-C')
PCargs.freepars = ['p1']
PCargs.initpoint = 'FP1:BP1'
PCargs.initdirec = PyCont['FP1'].getSpecialPoint('BP1').labels['BP']['data'].branch
PCargs.MaxNumPoints=50
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing second branch...')
start = clock()
PyCont['FP2'].forward()
PyCont['FP2'].update(args(MaxNumPoints=10))
PyCont['FP2'].backward()
PyCont['FP2'].cleanLabels()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'FP3'
PCargs.initpoint = 'FP2:BP2'
PCargs.initdirec = PyCont['FP2'].getSpecialPoint('BP2').labels['BP']['data'].branch
PCargs.MaxNumPoints = 50
PyCont.newCurve(PCargs)

print('Computing third branch...')
start = clock()
PyCont['FP3'].forward()
PyCont['FP3'].backward()
PyCont['FP3'].cleanLabels()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'FP4'
PCargs.initpoint = 'FP2:PD1'
PCargs.initdirec = PyCont['FP2'].getSpecialPoint('PD1').labels['PD']['data'].branch
PCargs.period = PyCont['FP2'].getSpecialPoint('PD1').labels['PD']['data'].branch_period
PCargs.MaxNumPoints = 100
PyCont.newCurve(PCargs)

print('Computing third branch...')
start = clock()
PyCont['FP4'].forward()
PyCont['FP4'].backward()
PyCont['FP4'].cleanLabels()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont.display(('p1','u1'))
PyCont.plot.fig1.axes1.FP4.BP2.toggleLabel('off')
PyCont.plot.fig1.axes1.FP3.BP1.toggleLabel('off')
show()
