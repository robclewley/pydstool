""" EXAMPLE: Two patch Rosenzweig-MacArthur predator-prey model (see [8])

    When continuue fold curve from ZH point, it detects the ZH point again.  More elegant way to handle
    this?

    Detects DH point

    Drew LaMar, March 2006
"""

from PyDSTool import *

pars = {'k': 7, 'D': 0.5, 'theta': 1, 'h': 0.5, 'mu': 0.8}

# Finds nontrivial boundary equilibrium from parameters
v1 = pars['mu']*(2*pars['D'] + pars['mu'])/ \
     (pars['D']*(1-pars['h']*pars['mu']-pars['theta']*pars['h']*pars['mu']) + pars['mu']*(1-pars['h']*pars['mu']))
v2 = 0.0
p1 = (1+pars['h']*v1)*(1-v1/pars['k'])
p2 = (pars['D']/(pars['D']+pars['mu']))*(1+pars['theta']*pars['h']*v1)*(1-v1/pars['k'])
icdict = {'v1': v1, 'v2': v2, 'p1': p1, 'p2': p2}

# Set up model
v1str = 'v1*(1-v1/k) - v1*p1/(1+h*v1)'
v2str = 'v2*(1-v2/k) - v2*p2/(1+h*v2)'
p1str = '-1*mu*p1 + v1*p1/(1+h*v1) + D*(((1+theta*h*v2)/(1+h*v2))*p2 - ((1+theta*h*v1)/(1+h*v1))*p1)'
p2str = '-1*mu*p2 + v2*p2/(1+h*v2) + D*(((1+theta*h*v1)/(1+h*v1))*p1 - ((1+theta*h*v2)/(1+h*v2))*p2)'

DSargs = args(name='PredatorPrey')
DSargs.pars = pars
DSargs.varspecs = {'v1': v1str, 'v2': v2str, 'p1': p1str, 'p2': p2str}
DSargs.ics = icdict

ode = Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PC = ContClass(ode)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['k']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 50
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PC.newCurve(PCargs)

print('Computing curve...')
start = clock()
PC['EQ1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs.name = 'HO1'
PCargs.type = 'H-C2'
PCargs.initpoint = 'EQ1:H1'
PCargs.freepars = ['k','D']
PCargs.MaxNumPoints = 50
PCargs.MaxStepSize = 0.1
PCargs.LocBifPoints = ['ZH']
PC.newCurve(PCargs)

print('Computing Hopf curve...')
start = clock()
PC['HO1'].forward()
print('done in %.3f seconds!' % (clock()-start))

PCargs = args(name = 'FO1', type = 'LP-C')
PCargs.initpoint = 'HO1:ZH1'
PCargs.freepars = ['k','D']
PCargs.MaxNumPoints = 25
PCargs.MaxStepSize = 0.1
PCargs.LocBifPoints = 'all'
PC.newCurve(PCargs)

print('Computing fold curve (forward)...')
start = clock()
PC['FO1'].forward()
print('done in %.3f seconds!' % (clock()-start))

print('Computing fold curve (backward)...')
start = clock()
PC['FO1'].backward()
print('done in %.3f seconds!' % (clock()-start))

# Plot
PC.display(('k','D'))
show()
