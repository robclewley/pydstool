""" EXAMPLE: Morris-Lecar (Type I)

    Drew LaMar, December 2005
"""

from PyDSTool import *

# "Computational Cell Biology", Fall (Type I)
pars = {'Iapp': 0.0,
        'C': 20.,
        'vK': -84.,
        'gK': 8.,
        'vCa': 120.,
        'gCa': 4.,
        'vL': -60.,
        'gL': 2.,
        'v1': -1.2,
        'v2': 18.,
        'v3': 12.,
        'v4': 17.4,
        'phi': 0.066}

icdict = {'v': -60., 'w': 0.01}

# Set up model
auxfndict = {'minf': (['v'], '0.5*(1 + tanh((v-v1)/v2))'), \
			 'winf': (['v'], '0.5*(1 + tanh((v-v3)/v4))'), \
			 'wtau': (['v'], '1/cosh((v-v3)/(2*v4))') \
			}

vstr = '(Iapp - gCa*minf(v)*(v-vCa) - gK*w*(v-vK) - gL*(v-vL))/C'
wstr = 'phi*(winf(v)-w)/wtau(v)'

DSargs = args(name='MorrisLecar')
DSargs.pars = pars
DSargs.varspecs = {'v': vstr, 'w': wstr}
DSargs.fnspecs = auxfndict
DSargs.ics = icdict

testDS = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['Iapp']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 350
PCargs.MaxStepSize = 1.
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PCargs.SaveEigen = True
PyCont.newCurve(PCargs)

print('Computing curve...')
start = perf_counter()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (perf_counter()-start))

PCargs.name = 'LC1'
PCargs.type = 'LC-C'
PCargs.initpoint = 'EQ1:H1'
PCargs.MinStepSize = 0.005
PCargs.MaxStepSize = 1.0
PCargs.StepSize = 0.01
PCargs.MaxNumPoints = 200
PCargs.NumSPOut = 40
PCargs.LocBifPoints = 'LPC'
PCargs.SolutionMeasures = 'all'
PCargs.SaveEigen = True
PyCont.newCurve(PCargs)

print('Computing curve...')
start = perf_counter()
PyCont['LC1'].forward()
print('done in %.3f seconds!' % (perf_counter()-start))

# Plot
PyCont.display(('Iapp','v'),stability=True)
PyCont['LC1'].display(('Iapp','v_min'),stability=True)

plt.xlim([-25, 150])
PyCont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
PyCont.plot.fig1.toggleAll('off', bytype=['P', 'MX'])

PyCont['LC1'].plot_cycles(normalized=True, figure='fig2', method='highlight', exclude='MX1')
PyCont.plot.fig2.axes1.axes.set_title('Cycles')

#PyCont.plot.setLegends('_nolegend_', bytype=['P', 'MX'])
#PyCont.plot.fig2.refresh()
#plt.legend()
show()
