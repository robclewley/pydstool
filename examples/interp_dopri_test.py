"""
    Tests for the Generator class.
    This tests:
     (1) InterpolateTable and Dopri_ODEsystem generators with external input
         (where the ODE has discontinuous right-hand side).
     (2) continuing orbit integration from previous state,
         by comparing to single-stage integration over same interval.

    Robert Clewley, February/August 2005.
"""

from PyDSTool import *
from time import perf_counter
from copy import copy


print('-------- Test: InterpolateTable')
xnames = ['x1', 'x2']
timeData = array([0., 11., 20., 30.])
x1data = array([10.2, -1.4, 4.1, 6.])
x2data = array([0.1, 0.01, 0.4, -0.8])
print("names for variables: xnames = ", xnames)
print("timeData = ", timeData)
print("x1data = ", x1data)
print("x2data = ", x2data)
xData = dict(list(zip(xnames, [x1data, x2data])))
interptable = InterpolateTable({'tdata': timeData,
                              'ics': xData,
                              'name': 'interp'
                              })
itabletraj = interptable.compute('interp')
print("interptable.variables['x1'](0.4) = ", interptable.variables['x1'](0.4), '\n')
print("itabletraj(11, ['x1']) = ", itabletraj(11, ['x1']), " (preferred syntax)\n")


# ------------------------------------------------------------

print('-------- Test: ODE system')

wfn_str = '100 -(1+a)*w*heav(0.2-sin(t+1)) -2*x1 -k*auxval1(t, w)*auxval2(y)'
yfn_str = '50 - (w/100)*y'
fnspecs = {'auxval1': (['t', 'x'], 'if(x>100,2*cos(10*t/pi),-0.5)'),
           'auxval2': (['x'], 'x/2')}

DSargs = args()
DSargs.tdomain = [0,20]
DSargs.tdata = [0,10]
DSargs.pars = {'k':1, 'a':2}
DSargs.inputs = itabletraj.variables['x1']
DSargs.varspecs = {'w': wfn_str, 'y': yfn_str}
DSargs.xdomain = {'w': [0, 340], 'y': [0, 150]}
DSargs.fnspecs = fnspecs
DSargs.algparams = {'init_step' :0.02}
DSargs.checklevel = 2
DSargs.ics = {'w': 30.0, 'y': 80}
DSargs.name = 'ODEtest'

print("\n")
info(DSargs)
print("\n")

testODE = Dopri_ODEsystem(DSargs)

print('Integrating...')
start = perf_counter()
testtraj = testODE.compute('test')
print('  ... finished in %.3f seconds.\n' % (perf_counter()-start))

print("\nTesting direct call to vector field function:")
print("""testODE.Rhs(0.3, {'w':10., 'y':0.3}, DSargs.pars,
                     {'itable': itabletraj.variables['x1'](0.3)}) -->""")
print(testODE.Rhs(0.3, {'w':10., 'y':0.3}, DSargs.pars))

plotData = testtraj.sample(dt=0.1)
yaxislabelstr = 'w, y, for k = ' + str(DSargs.pars['k'])

print("\nTesting continued integration -- with different input")
testODE.set(tdata=[10,20], inputs={'x1': itabletraj.variables['x2']})
testtraj2 = testODE.compute('test2', 'c')
plotData2 = testtraj2.sample(dt=0.1)

testODE.set(tdata=[0,20])
fulltraj = testODE.compute('fulltraj')
plotData_full = fulltraj.sample(dt=0.1)

plt.ylabel(yaxislabelstr)
plt.xlabel('t')

wline_full=plot(plotData_full['t'], plotData_full['w'])
yline_full=plot(plotData_full['t'], plotData_full['y'])
wline1=plot(plotData['t'], plotData['w'])
yline1=plot(plotData['t'], plotData['y'])
wline2=plot(plotData2['t'], plotData2['w'])
yline2=plot(plotData2['t'], plotData2['y'])

w = testODE.diagnostics.warnings
assert len(w)==1
warning = w[0]
assert warning[0] == 11, \
       "Trajectory should have been truncated by reaching a state bound"
assert allclose(warning[1][0], 14.44, 0.1), \
       "Trajectory should have been truncated at approximately t = 14.44"

print("\n--------------------------------------\ntestODE.showSpec() -->")
testODE.showSpec()
print("testODE.showAuxFnSpec() -->")
testODE.showAuxFnSpec()

show()
