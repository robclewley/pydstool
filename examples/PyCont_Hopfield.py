#Author:    David Sterratt
#Date:      May 2007
""" EXAMPLE: Hopfield.
    Uses special math functions and user-defined stopping criteria for PyCont.

    David Sterratt, May 2007
"""

from PyDSTool import *
from scipy.special import *

pars = {'alpha': 0.1,
        'p': 0.5,
        'theta': 0.,
        'gam': 0.,
        'lambda': 0.,
        'chi': 1.}

icdict = {'m': 1., 'r': 1, 'C': 0., 's': 0}

# Set up model
auxfndict = {'sigsq': (['s', 'r'], 'alpha*(4.*p*(1.-p)*r+pow((1.-chi)*s+lambda,2))'), \
             'yp': (['m', 's', 'r'], '(2.*(1.-p)*m-gam*s-theta)/sqrt(2*alpha*(4*p*(1-p)*r+pow((1-chi)*s+lambda,2)))'), \
             'ym': (['m', 's', 'r'], '(2.*(-p)*m-gam*s-theta)/sqrt(2*alpha*(4*p*(1-p)*r+pow((1-chi)*s+lambda,2)))')}

mstr = '-m + 0.5*special_erf(yp(m,s,r)) - 0.5*  special_erf(ym(m,s,r))'
sstr = '-s + p*  special_erf(yp(m,s,r)) + (1-p)*special_erf(ym(m,s,r))'
rstr = '-r + (1-pow(s,2))/(4*p*(1-p)*(pow(1-C,2)))'
Cstr = '-C + sqrt(2/(pi*sigsq(s,r)))*(p*exp(-pow(yp(m,s,r),2))+(1-p)*exp(-pow(ym(m,s,r),2)))'

DSargs = args(name='Hopfield')
DSargs.pars = pars
DSargs.varspecs = {'m': mstr, 's': sstr, 'r': rstr, 'C': Cstr}
DSargs.fnspecs = auxfndict
DSargs.ics = icdict
DSargs.tdomain = [0,100]
DSargs.pdomain = {'theta': [-0.6,0.6], 'alpha': [0.001, 20]}
DSargs.algparams = {'init_step' :0.2, 'strictopt':False}

testDS = Generator.Vode_ODEsystem(DSargs)

print('Integrating...')
start = perf_counter()
testtraj = testDS.compute('testDS')
print('  ... finished in %.3f seconds.\n' % (perf_counter()-start))

if 0:
    plotData=testtraj.sample(dt=0.1)
    mline=plt.plot(plotData['t'],plotData['m'])
    rline=plt.plot(plotData['t'],plotData['r'])
    Cline=plt.plot(plotData['t'],plotData['C'])
    sline=plt.plot(plotData['t'],plotData['s'])
    plt.legend([mline,rline,Cline,sline],['m','r','C','s'])
    show()

# Set up continuation class
PyCont = ContClass(testDS)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['alpha']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 350
PCargs.MaxStepSize = 1.
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PCargs.SaveEigen = True
PCargs.MaxTestIters = 200
PyCont.newCurve(PCargs)

print('Computing curve...')
start = perf_counter()
PyCont['EQ1'].forward()
print('done in %.3f seconds!' % (perf_counter()-start))

if 0:
    PyCont.display(('alpha','m'),stability=True)
    show()

PCargs.name = 'FO1'
PCargs.type = 'LP-C'
PCargs.freepars = ['alpha','theta']
PCargs.initpoint = 'EQ1:LP1'
PCargs.StopAtPoints = ['B']
PyCont.newCurve(PCargs)

print('Computing curve...')
start = perf_counter()
PyCont['FO1'].forward()
PyCont['FO1'].backward()
print('done in %.3f seconds!' % (perf_counter()-start))

PyCont['FO1'].display(('theta','alpha'),stability=True)
show()