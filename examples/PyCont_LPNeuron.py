""" EXAMPLE: Lateral Pyloric neuron model found in [6] and [3] (sec. 5.4)

    Drew LaMar, December 2005
"""

from PyDSTool import *

pars = {'Cm': 0.0017,
	'kh': 500.0,
	'kCa': 360.0,
	'kKCa': 45.0,
	'kAf': 30.0,
	'kAs': 10.0,
	'kr': 0.1,
	'gNa': 2300.0,
	'gCa1': 0.21,
	'gCa2': 0.047,
	'gK': 0.841,
	'gKCa': 5.0,
	'gAf': 1.0,
	'gAs': 1.3,
	'gh': 0.1,
	'ENa': 50.0,
	'ECa': 140.0,
	'EK': -86.0,
	'Eh': -10.0,
	'El': -50.0,
	'Iext': 0.551636,
	'vr': -110.0,
	'sr': 12.0,
	'ciCa': 300.0,
	'vA': -43.0,
	'vkr': -100.0,
	'skr': -13.0,
	'gl': 0.1,
	'vb': -62.0}

icdict = {'v': -38.6522,
          'h': 0.119015,
          'Ca': 0.166282,
          'aCa1': 0.0188852,
          'aCa2': 0.00017256,
          'bCa1': 0.1949,
          'n': 0.309369,
          'aKCa': 0.00134052,
          'bKCa': 0.783001,
          'aA': 0.56752,
          'bAf': 0.0200101,
          'bAs': 0.0200101,
          'ah': 0.00261036}

# Set up model
auxfndict = {'minf': (['v'], '1/(1 + 15*exp(-1*((v+34)/13))*(1-exp(-0.05*(v+6)))/(0.11*v+0.66))')}

v_str = '(Iext + (' + \
       'gNa*pow(minf(v),3)*h*(ENa - v) + ' + \
	   '(gCa1*aCa1*bCa1 + gCa2*aCa2)*(ECa - v) + ' + \
	   'gK*pow(n,4)*(EK - v) + ' + \
	   'gKCa*aKCa*bKCa*(EK - v) + ' + \
	   '(gAf*bAf + gAs*bAs)*pow(aA,3)*(EK - v) + ' + \
	   'gh*ah*(Eh - v) + ' + \
	   'gl*(El - v)' + \
       '))/Cm'

h_str = 'kh*(0.08*(1-h)*exp(-0.125*(v+39.)) - h/(1+exp(-1*(0.2*v + 8.))))'

Ca_str = '-1*ciCa*(gCa1*aCa1*bCa1 + gCa2*aCa2)*(v - ECa) + kCa*(0.05-Ca)'
aCa1_str = '50./(1 + exp(-1*(v+11.)/7.)) - 50.*aCa1'
aCa2_str = '10./(1 + exp(-1*(v-22.)/7.)) - 10.*aCa2'
bCa1_str = '16./(1 + exp(0.125*(v + 50.))) - 16.*bCa1'

n_str = '(282./(1 + exp(-1*(v+25.)/17.)) - 282.*n)/(1 + exp(-1*(v-10.)/22.))'

aKCa_str = 'kKCa*Ca/(' + \
		   '(1 + exp(-1*(0.0434783*v + 0.026087*Ca + 0.869565)))*' + \
		   '(1 + exp(-1*(0.166667*v + 0.1*Ca + 3.83333)))*' + \
		   '(2.5 + Ca)' + \
		   ') - kKCa*aKCa'
bKCa_str = '21./(0.6 + Ca) - 35.*bKCa'

aA_str = '140.*(1./(1 + exp(-1*(v - vA)/16.)) - aA)'
bAf_str = 'kAf*(1./(1 + exp((v - vb)/6.)) - bAf)'
bAs_str = 'kAs*(1./(1 + exp((v - vb)/6.)) - bAs)'

ah_str = 'kr*(1 + exp((v - vkr)/skr))*(1./(1 + exp((v - vr)/sr)) - ah)'

DSargs = args(name='LPneuron')
DSargs.pars = pars
DSargs.varspecs = {'v': v_str,
                   'h': h_str,
                   'Ca': Ca_str,
                   'aCa1': aCa1_str,
                   'aCa2': aCa2_str,
                   'bCa1': bCa1_str,
                   'n': n_str,
                   'aKCa': aKCa_str,
                   'bKCa': bKCa_str,
                   'aA': aA_str,
                   'bAf': bAf_str,
                   'bAs': bAs_str,
                   'ah': ah_str}
DSargs.fnspecs = auxfndict
DSargs.ics = icdict
DSargs.algparams = {'refine': True,
                    }
testDS = Generator.Radau_ODEsystem(DSargs)

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['Iext']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 300
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PyCont.newCurve(PCargs)

print('Computing curve...')
start = clock()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PyCont['EQ1'].display(('Iext','v'), figure='new')
PyCont.plot.toggleAll('off', bytype='P')

# Limit cycle curve
PCargs.name = 'LC1'
PCargs.type = 'LC-C'
PCargs.MaxNumPoints = 250
PCargs.NumIntervals = 20
PCargs.NumCollocation = 6
PCargs.initpoint = 'EQ1:H1'
PCargs.SolutionMeasures = 'all'
PCargs.NumSPOut = 50
PCargs.FuncTol = 1e-10
PCargs.VarTol = 1e-10
PCargs.TestTol = 1e-8
PyCont.newCurve(PCargs)

print('Computing limit-cycle curve...')
start = clock()
PyCont['LC1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PyCont['LC1'].display(('Iext','v'), stability=True)

PyCont['LC1'].display(stability=True, figure='new', axes=(1,2,1))
PyCont['LC1'].plot_cycles(coords=('v','h'), linewidth=1, axes=(1,2,2), figure='fig2')
show()

# Hopf curve
# PCargs.name = 'HO1'
# PCargs.type = 'H-C1'
# PCargs.initpoint = 'EQ1:H1'
# PCargs.freepars = ['Iext','gAf']
# PCargs.MaxStepSize = 1e-2
# PCargs.LocBifPoints = 'all'
# PCargs.MaxNumPoints = 2
#
# PyCont.newCurve(PCargs)
#
# print 'Computing hopf curve...'
# start = clock()
# PyCont['HO1'].forward()
# print 'done in %.3f seconds!' % (clock()-start)

# Fold curve
# PCargs.name = 'FO1'
# PCargs.type = 'LP-C'
# PCargs.initpoint = 'EQ1:LP1'
# PCargs.freepars = ['Iext','gAf']
# PCargs.MaxNumPoints = 100
# PCargs.LocBifPoints = ['BT','ZH','Cusp']
# PyCont.newCurve(PCargs)
#
# print 'Computing fold curve...'
# start = clock()
# PyCont['FO1'].backward()
# print 'done in %.3f seconds!' % (clock()-start)
