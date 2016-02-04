""" EXAMPLE: Catalytic Oscillator system found in [3]

    Drew LaMar, December 2005
"""

d = lambda x: max([len(n) for n in x.splitlines()])
c = '#'

str = """
from PyDSTool import *

pars = {'q1': 2.5, 'q2': 1.92373, 'q3': 10, 'q4': 0.0675, 'q5': 1, 'q6': 0.1, 'k': 0.4}

icdict = {'x': 0.0014673, 'y': 0.826167, 's': 0.123119}

auxfndict = {'z': (['x', 'y', 's'], '1 - x - y - s')}

xstr = '2*q1*z(x,y,s)*z(x,y,s) - 2*q5*x*x - q3*x*y'
ystr = 'q2*z(x,y,s) - q6*y - q3*x*y'
sstr = 'q4*z(x,y,s) - k*q4*s'

DSargs = args(name='CatalyticOscillator')
DSargs.pars = pars
DSargs.varspecs = {'x': xstr, 'y': ystr, 's': sstr}
DSargs.fnspecs = auxfndict
DSargs.ics = icdict

testDS = Generator.Dopri_ODEsystem(DSargs)
PyCont = ContClass(testDS)
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['q2']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 175
PCargs.MaxStepSize = 1e-2
PCargs.LocBifPoints = ['LP','H']

PyCont.newCurve(PCargs)
PyCont['EQ1'].forward()
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
print('Computing curve...')
exec(str)

str = """
PyCont['EQ1'].info()
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
X = PyCont['EQ1'].getSpecialPoint('LP1')
print(X)
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
print(X.labels['LP']['data'].a)
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PyCont['EQ1'].display(axes=(1,2,1))
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)
show()

str = """
PyCont['EQ1'].display(('q2','x'), axes=(1,2,2))
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PyCont.plot.info()
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PyCont.info()
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PCargs = args(name='HO1', type='H-C2')
PCargs.initpoint = 'EQ1:H2'
PCargs.freepars = ['q2', 'k']
PCargs.MaxStepSize = 1e-2
PCargs.LocBifPoints = ['GH', 'BT', 'ZH']
PCargs.MaxNumPoints = 100

PyCont.newCurve(PCargs)
PyCont['HO1'].forward()
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
print('Computing curve...')
exec(str)

str = """
PyCont.plot.clearall()
PyCont.display(('q2','x'))
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PCargs = args(name='FO1', type='LP-C')
PCargs.initpoint = 'EQ1:LP1'
PCargs.freepars = ['q2', 'k']
PCargs.LocBifPoints = 'all'
PCargs.MaxNumPoints = 45

PyCont.newCurve(PCargs)
PyCont['FO1'].forward()
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
print('Computing curve...')
exec(str)

str = """
PyCont.plot.clearall()
PyCont.display(('q2','x'))
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PyCont['FO1'].backward()
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
print('Computing curve...')
exec(str)

str = """
PyCont.plot.fig1.axes1.FO1.delete()
PyCont['FO1'].display(('q2','x'))
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PyCont.plot.toggleLabels('off')
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PyCont.plot.toggleLabels('on')
PyCont.plot.togglePoints('off')
PyCont.plot.toggleCurves('off', byname = 'HO1')
PyCont.plot.fig1.axes1.HO1.toggleCurve('off')
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

str = """
PyCont.computeEigen()
PyCont.display(coords=('q2','x'), stability=True, figure='fig2')
PyCont.plot.fig2.toggleAll('off', bytype='P')
"""

print('\n' + c*d(str) + str + c*d(str))
#dummy = raw_input()
exec(str)

print('\n')

#dummy = raw_input()
