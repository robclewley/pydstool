""" EXAMPLE: Logistic map

    Drew LaMar, March 2006
"""

from PyDSTool import *

pars = {'r': 0.}

icdict = {'x': 0.}

# Set up model
xstr = 'r*x*(1-x)'

DSargs = args(name='LogisticMap')
DSargs.pars = pars
DSargs.varspecs = {'x': xstr}
DSargs.ics = icdict
DSargs.ttype = int
DSargs.pdomain = {'r': [0.0, 4.0]}

testDS = Generator.MapSystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='FP1', type='FP-C')
PCargs.freepars = ['r']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 70
PCargs.MaxStepSize = 1e-1
PCargs.verbosity = 2
PCargs.LocBifPoints = 'all'
PCargs.StopAtPoints = 'B'
PCargs.SaveEigen = True
PCargs.SPOut = {'r': [0.1919191, 1.5353535]}
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['FP1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'FP2'
PCargs.initpoint = 'FP1:BP1'
PCargs.initdirec = PyCont['FP1'].getSpecialPoint('BP1').labels['BP']['data'].branch
PCargs.MaxNumPoints=50
PCargs.LocBifPoints = ['PD', 'B']
PyCont.newCurve(PCargs)

print('Computing second branch...')
start = clock()
PyCont['FP2'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'FP3'
PCargs.initpoint = 'FP2:PD1'
PCargs.initdirec = PyCont['FP2'].getSpecialPoint('PD1').labels['PD']['data'].branch
PCargs.MaxNumPoints = 40
PCargs.LocBifPoints = ['PD', 'B']
PCargs.period = 2
PyCont.newCurve(PCargs)

print('Computing 2-cycle branch...')
start = clock()
PyCont['FP3'].forward()
PyCont['FP3'].backward()
PyCont['FP3'].cleanLabels()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name='FP4'
PCargs.initpoint = 'FP3:PD1'
PCargs.initdirec = PyCont['FP3'].getSpecialPoint('PD1').labels['PD']['data'].branch
PCargs.period = 4
PyCont.newCurve(PCargs)

print('Computing 1st 4-cycle branch...')
start = clock()
PyCont['FP4'].forward()
PyCont['FP4'].backward()
PyCont['FP4'].cleanLabels()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'FP5'
PCargs.initpoint = 'FP3:PD2'
PCargs.initdirec = PyCont['FP3'].getSpecialPoint('PD2').labels['PD']['data'].branch
PyCont.newCurve(PCargs)

print('Computing 2nd 4-cycle branch...')
start = clock()
PyCont['FP5'].forward()
PyCont['FP5'].backward()
PyCont['FP5'].cleanLabels()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont.display(stability=True)
plt.xlim([1, 4])
PyCont.plot.toggleAll('off')
plt.title('Logistic map')
show()
