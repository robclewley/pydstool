""" EXAMPLE: AB Reaction system (from [9])

	H-C1 does not currently work on this example.

    Drew LaMar, January 2006
"""

from PyDSTool import *

pars = {'p1': 0., 'p2': 14., 'p3': 2.}

icdict = {'u1': 0., 'u2': 0.}

# Set up model
u1str = '-1*u1 + p1*(1 - u1)*exp(u2)'
u2str = '-1*u2 + p1*p2*(1 - u1)*exp(u2) - p3*u2'

DSargs = args(name='ABReaction')
DSargs.pars = pars
DSargs.varspecs = {'u1': u1str, 'u2': u2str}
DSargs.ics = icdict
#DSargs.pdomain = {'p1': [0, 0.08]}
#DSargs.xdomain = {'u1': [0.2, 0.8]}

testDS = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['p1']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 56
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
#PCargs.StopAtPoints = 'B'
PCargs.SaveJacobian = True
PCargs.SaveEigen = True
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Option to delay build of system if want to add user-defined C code
#PyCont.loadAutoMod(nobuild=True)
#PyCont.makeAutoLibSource()
# Make edits to the ABReaction_vf.c file by hand
#PyCont.compileAutoLib()

PCargs.name = 'LC1'
PCargs.type = 'LC-C'
PCargs.initpoint = 'EQ1:H1'
PCargs.MinStepSize = 0.005
PCargs.MaxStepSize = 0.5
PCargs.StepSize = 0.01
PCargs.MaxNumPoints = 150
PCargs.LocBifPoints = []
PCargs.NumSPOut = 30;
PCargs.SolutionMeasures = 'all'
PCargs.SaveEigen = True
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['LC1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'FO1'
PCargs.type = 'LP-C'
PCargs.initpoint = 'EQ1:LP1'
PCargs.freepars = ['p1','p3']
PCargs.MaxNumPoints = 70
PCargs.MaxStepSize = 0.1
PCargs.StepSize = 1e-2
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing Fold curve...')
start = clock()
PyCont['FO1'].forward()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PyCont['EQ1'].display(stability=True, axes=(1,2,1))
PyCont['FO1'].display()
PyCont['LC1'].display(('p1','u1_nm2'), stability=True)

PyCont['LC1'].plot_cycles(coords=('u1','u2'), axes=(1,2,2))

PyCont.plot.fig1.toggleAll('off', bytype=['P', 'MX'], byname='RG4')
PyCont.plot.setLegends('_nolegend_', bytype=['P', 'MX'], byname='RG4')
PyCont.plot.fig1.axes1.EQ1.LP1.text.set_ha('right')
pos = PyCont.plot.fig1.axes1.EQ1.LP1.text.get_position()
PyCont.plot.fig1.axes1.EQ1.LP1.text.set_position((pos[0]*0.95, pos[1]))
PyCont.plot.fig1.axes1.axes.set(xticks=[0.0, 0.04, 0.08, 0.12, 0.16])
PyCont.plot.fig1.axes2.axes.set(xticks=[0.2, 0.4, 0.6, 0.8, 1.0])
PyCont.plot.fig1.refresh()
plt.legend(loc=2)

PyCont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
PyCont.plot.fig1.axes2.axes.set_title('Cycles')
PyCont.plot.refresh()
show()
