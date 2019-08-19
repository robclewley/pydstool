""" EXAMPLE: User-defined continuation.
             [Unforced van der Pol oscillator]

    Described and discussed in detail in a research article submitted
    for publication in 2007 by John Guckenheimer and Drew LaMar.

    Drew LaMar, April 2007
"""

from PyDSTool import *
import sys

vdp_filename = 'vanderPol.dat'
print("At end of testing you can delete the temp file", vdp_filename)

def integrate(DS, t1=200, name='traj'):
    DS.set(tdata=[0, t1])
    return DS.compute(name)


def create_system(sysname='vanderPol', eps=1.0):
    pars = {'eps': eps, 'a': 0.5, 'y1': -0.708}

    icdict = {'x': pars['a'], 'y': pars['a'] - pars['a']*pars['a']*pars['a']/3}

    # Set up models
    xstr = '(y - (x*x*x/3 - x))/eps'
    ystr = 'a - x'

    DSargs = args(name=sysname)
    """ (function, direction, arguments, variables, language) """
    event_x_a = Events.makeZeroCrossEvent('x-a', 0,
                                {'name': 'event_x_a',
                                 'eventtol': 1e-10,
                                 'term': False,
                                 'active': True}, varnames=['x'],
                                 parnames=['a'],
                                 targetlang='c')
    DSargs.events = [event_x_a]
    DSargs.pars = pars
    DSargs.tdata = [0, 200]
    DSargs.algparams = {'max_pts': 300000}
    DSargs.varspecs = {'x': xstr, 'y': ystr}
    DSargs.ics = icdict
    DSargs.pdomain = {'a': [0, 1.2]}

    return Generator.Radau_ODEsystem(DSargs)

def find_cycle(DS, disp=True):
    DS.set(ics = {'x': 0.2, 'y': 0.2},
           algparams={'poly_interp': True})
    traj = integrate(DS)
    DS.set(algparams={'poly_interp': False})
    if disp:
        plotData = traj.sample(dt=0.1)
        plot(plotData['t'], plotData['x'])
        return traj
    else:
        t0 = 97
        T0 = 7
        T = findApproxPeriod(traj, t0, t0+T0, coordname='y', rtol=0.005)
        cycle = traj.sample(dt=0.01, tlo=t0, thi=t0+T, precise=True)
        cycle.indepvararray = cycle.indepvararray.flatten()
        return cycle


def continuation(PyCont, cycle=None, compute=False, disp=False):
    # Initial continuation to find epsilon curve
    PCargs = args(name='LC_eps', type='LC-C')
    PCargs.description = """One-parameter continuation with epsilon as free parameter.  Gives
    four special points UZ1-UZ4 w/ epsilon = (1e-1, 1e-2, 1e-3, 1e-4), respectively."""
    PCargs.freepars = ['eps']
    PCargs.StepSize = 1e-2
    PCargs.MaxStepSize = 1e-2
    PCargs.MaxNumPoints = 200
    PCargs.SPOut = {'eps': [1e-1, 1e-2, 1e-3, 1e-4]}
    PCargs.NumIntervals = 100
    PCargs.SolutionMeasures = 'all'
    PCargs.SaveEigen = True
    if cycle is not None:
        PCargs.initcycle = cycle
    try:
        PyCont.newCurve(PCargs)
    except:
        PyCont['LC_eps'].initcycle = cycle

    if compute:
        PyCont['LC_eps'].backward()
        PCargs.initpoint = PyCont['LC_eps'].getSpecialPoint('UZ2')

    PCargs.name='LC_a'
    PCargs.description = """One-parameter continuation with a as free parameter.  The parameter
    epsilon is fixed at epsilon = 0.01 (starts at UZ2 from LC_eps)."""
    PCargs.freepars = ['a']
    PCargs.StepSize = 5e-3
    PCargs.MaxStepSize = 1e-2
    PCargs.MaxNumPoints = 800
    PCargs.MinStepSize = 1e-6
    PCargs.NumSPOut = 50
    PCargs.SPOut = {'eps': [1e-1, 1e-2, 1e-3, 1e-4]}
    PCargs.NumIntervals = 100
    PCargs.SolutionMeasures = 'all'
    PCargs.SaveEigen = True
    try:
        PyCont.newCurve(PCargs)
    except:
        PyCont['LC_a'].initcycle = cycle

    if compute:
        PyCont['LC_a'].forward()
        save_continuation(PyCont, ['LC_eps', 'LC_a'], vdp_filename);
    else:
        filename=vdp_filename
        load_continuation(PyCont, filename=filename)

    if disp:
        PyCont['LC_eps'].display(('eps', 'x_nm2'), stability=True)
        PyCont.plot.toggleAll('off', bytype='P')
        PyCont['LC_a'].plot_cycles(('x','y'), method='highlight', figure='new')
        plot_manifold()


def user_continuation(PyCont):
    name = 'RG13'
    point = PyCont['LC_a'].getSpecialPoint(name)
    cycle = point.labels[name.rstrip('0123456789')]['cycle']
    PyCont.gensys.pars['a'] = point.todict()['a']
    PyCont.gensys.pars['eps'] = PyCont['LC_a'].parsdict['eps']

    PCargs = args(name='UD1', type='UD-C')
    PCargs.description = """Continuation of canards.  Cross your fingers..."""
    PCargs.uservars = ['a']
    PCargs.userpars = PyCont.gensys.pars
    PCargs.userfunc = cont_func
    PCargs.FuncTol = 1e-10
    PCargs.VarTol = 1e-10
    PCargs.freepars = ['y1']
    PCargs.StepSize = 1e-2
    PCargs.MaxStepSize = 5e-2
    PCargs.MaxNumPoints = 50
    PCargs.SaveJacobian = True
    PCargs.verbosity = 4
    PCargs.initpoint = {'a': PyCont.gensys.pars['a']}
    PyCont.newCurve(PCargs)

    PCargs = args(name='UD2', type='UD-C')
    PCargs.description = """Continuation of canards.  Cross your fingers..."""
    PCargs.uservars = ['a']
    PCargs.userpars = PyCont.gensys.pars
    PCargs.userfunc = cont_func
    PCargs.freepars = ['y1']
    PCargs.StepSize = 1e-2
    PCargs.MaxStepSize = 5e-2
    PCargs.MaxNumPoints = 40
    PCargs.SaveJacobian = True
    PCargs.verbosity = 4
    PCargs.initpoint = {'a': PyCont.gensys.pars['a']}
    PyCont.newCurve(PCargs)
    PyCont['UD2']._userdata.sgn = 1


def plot_manifold():
    x = linspace(-3.,-1., 200)
    y = x*x*x/3 - x
    plot(x, y, 'k-', linewidth=1.5)

    x = linspace(1., 3., 200)
    y = x*x*x/3 - x
    plot(x, y, 'k-', linewidth=1.5)

    x = linspace(-1., 1., 200)
    y = x*x*x/3 - x
    plot(x, y, 'k--', linewidth=1.5)


def save_continuation(PyCont, sessions, filename='foo', force=True):
    data = []
    for sess in sessions:
        data.extend([PyCont[sess].sol, PyCont[sess].parsdict])
    saveObjects(data, filename, force=force)


def load_continuation(PyCont, filename='foo'):
    data = loadObjects(filename)
    for i in range(len(data)//2):
        name = data[2*i].name
        PyCont[name].sol = data[2*i]
        PyCont[name].parsdict = data[2*i+1]


def set_initpoint(DS, PyCont, name='RG13'):
    point = PyCont['LC_a'].getSpecialPoint(name)
    cycle = point.labels[name.rstrip('0123456789')]['cycle']
    DS.pars['a'] = point.todict()['a']
    DS.pars['eps'] = PyCont['LC_a'].parsdict['eps']

    across = []
    monecross = []
    for i in range(len(cycle)-1):
        if (cycle['x'][i]-DS.pars['a'])*(cycle['x'][i+1]-DS.pars['a']) <= 0:
            across.append(i)
        elif (cycle['x'][i]+1.0)*(cycle['x'][i+1]+1.0) <= 0:
            monecross.append(i)

    # x0
    ind = across[1]
    iy = cycle['y'][ind] + ((cycle['y'][ind+1] - cycle['y'][ind])/(cycle['x'][ind+1] - cycle['x'][ind]))*(DS.pars['a'] - cycle['x'][ind])
    x0 = [DS.pars['a'], iy]

    # set up for initial backwards run
    PyCont.gensys.pars['y1'] = iy
    x = array([DS.pars['a']], float)

##    # x1
##    ind = monecross[0]
##    iy = cycle['y'][ind] + ((cycle['y'][ind+1] - cycle['y'][ind])/(cycle['x'][ind+1] - cycle['x'][ind]))*(-1.0 - cycle['x'][ind])
##    x1 = [-1.0, iy]

    return x


def cont_func(C, pt, pars):
    plot = False
    DS = C.gensys
    DS.pars['a'] = pt['a']

    if (('sgn' not in C._userdata) or C._userdata.sgn == -1):
        x1 = {'x': -1.0, 'y': pars['y1']}
    else:
        x1 = {'x': DS.pars['a'], 'y': pars['y1']}

    # BACKWARD
    DS.eventstruct['event_x_a'].dircode = 1
    try:
        tx1b = DS.compute('x1b', dirn='b', ics=x1)
    except PyDSTool_ExistError:
        print('Lost canard!\n')
        C._userdata.problem = True
        return array([0], float)
    x1b = DS.getEvents()['event_x_a'][0].toarray()

    # FORWARD
    DS.eventstruct['event_x_a'].dircode = -1
    tx1f = DS.compute('x1f', ics=x1)
    x1f = DS.getEvents()['event_x_a'][0].toarray()

    # Store cycle information in userdata
    cycleb = tx1b.sample()
    cycleb['t'] = cycleb['t'] - min(cycleb['t'])

    cyclef = tx1f.sample()
    cyclef['t'] = cyclef['t'] + cycleb['t'][-1]

    cycleb.append(cyclef[1:])
    C._userdata.cycle = cycleb

    if plot:
        px1f = tx1f.sample(dt=0.01)
        px1b = tx1b.sample(dt=0.01)
        plt.plot(px1f['x'], px1f['y'])
        plt.plot(px1b['x'], px1b['y'])
        input()
        plt.close()

    F = array([x1f[1]-x1b[1]], float)
    return F


def plot_cycles(PyCont, name='UD1', meas=None):
    """Plot cycles from cycle data saved from _userdata argument list.  Plots cycles
    if meas is None and solution measure nm2 or max if specified."""
    if meas is None:
        for pt in PyCont[name].sol:
            if 'UD' in pt.labels and 'cycle' in pt.labels['UD']['data']:
                cycle = pt.labels['UD']['data'].cycle
                plt.plot(cycle['x'], cycle['y'], '-b')
    else:
        a = PyCont[name].sol['a']
        solmeas = []
        ind = []
        if meas == 'nm2':
            for i, pt in enumerate(PyCont[name].sol):
                if 'UD' in pt.labels and 'cycle' in pt.labels['UD']['data']:
                    cycle = pt.labels['UD']['data'].cycle
                    dt = (cycle['t'][1:]-cycle['t'][0:-1])/(cycle['t'][-1]-cycle['t'][0])
                    solmeas.append(sqrt(0.5*(sum(dt*(cycle['x'][1:]*cycle['x'][1:] + \
                                                     cycle['x'][:-1]*cycle['x'][:-1])))))
                    ind.append(i)
        elif meas == 'max':
            for i, pt in enumerate(PyCont[name].sol):
                if 'UD' in pt.labels and 'cycle' in pt.labels['UD']['data']:
                    solmeas.append(max(abs(pt.labels['UD']['data'].cycle['x'])))
                    ind.append(i)

        solmeas = array(solmeas)
        ind = array(ind)
        plt.plot(a[ind], solmeas)


# --------------------------------------------------------------------------

DS = create_system()

# Compute limit cycle
cycle = find_cycle(DS, disp=False)
print("Finding limit cycle using AUTO")
C = ContClass(DS)
continuation(C, cycle=cycle, compute=True, disp=False)

# Display results
C['LC_eps'].display(('eps', 'x_nm2'))
C.plot.toggleAll('off', bytype='P')
C['LC_a'].plot_cycles(('x', 'y'), figure='mew', method='highlight')
plot_manifold()
plt.xlim([-2.5, 2.5])
plt.ylim([-1, 1])

print("Continuing limit cycle")
continuation(C, cycle=None, compute=False, disp=False)
DS.set(tdata=[0, 400])

# Set initial point and initialize user-defined continuation
name = 'RG13'
x = set_initpoint(DS, C, name=name)
user_continuation(C)

# Activate user-defined events
C['UD1'].gensys.eventstruct['event_x_a'].activeFlag = True
C['UD1'].gensys.eventstruct['event_x_a'].termFlag = True

print('UD1: Integrating backward...')
try:
    C['UD1'].backward()
except:
    exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
    print("Error: ", exceptionType, ": message:", exceptionValue)
print('done!\n\n')
print('UD1: Integrating forward...')
try:
    C['UD1'].forward()
except:
    exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
    print("Error: ", exceptionType, ": message:", exceptionValue)
print('done!\n\n')

## Temp removed UD2 calculation (broken)

### Find starting point for UD2
### start from pt # 1 in case previous run ended with an MX point
##x1 = {'x': -1.0, 'y':  C['UD1'].sol[1]['y1']}
##C['UD1'].gensys.eventstruct['event_x_a'].dircode = 1
##tx1f = C['UD1'].gensys.compute('x1f', ics=x1)
##C['UD1'].gensys.eventstruct['event_x_a'].dircode = -1
##y1 = C['UD1'].gensys.getEvents()['event_x_a']['y'][0]

##C['UD2'].initpoint = {'a': C['UD1'].sol[1]['a'], 'y1': y1}

##print 'UD2: Integrating forward...'
##try:
##    C['UD2'].forward()
##except:
##    exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
##    print "Error: ", exceptionType, ": message:", exceptionValue
##print 'done!\n\n'

# Display results
plot_manifold()
plot_cycles(C, 'UD1')
#plot_cycles(C, 'UD2')
print("At end of testing you can delete the temp file", vdp_filename)
show()
