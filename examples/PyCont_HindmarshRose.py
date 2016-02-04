""" EXAMPLE: Hindmarsh-Rose

    Drew LaMar, July 2006
"""

from PyDSTool import *

def create_fast_subsystem():
    x = -0.5*(1+sqrt(5))

    pars = {'I': 2., 'z': 0.}

    icdict = {'x': x, 'y': 1-5*x*x}

    # Set up model
    xstr = 'y - x*x*x + 3*x*x + I - z'
    ystr = '1 - 5*x*x - y'

    DSargs = args(name='HindmarshRoseFast')
    DSargs.fnspecs = {'Jacobian': (['t', 'x', 'y'], """[[-3*x*x + 6*x, 1.0],
                                                        [-10*x, -1.0]]""")}
    DSargs.pars = pars
    DSargs.varspecs = {'x': xstr, 'y': ystr}
    DSargs.ics = icdict
    DSargs.pdomain = {'z': [-14, 4]}

    testDS = Generator.Radau_ODEsystem(DSargs)

    # Set up continuation class
    return ContClass(testDS)

def create_fast_diagram(HR_fast):
    PCargs = args(name='EQ1', type='EP-C')
    PCargs.freepars = ['z']
    PCargs.StepSize = 2e-3
    PCargs.MaxNumPoints = 450
    PCargs.MaxStepSize = 1e-1
    PCargs.LocBifPoints = 'all'
    PCargs.StopAtPoints = 'B'
    PCargs.SaveEigen = True
    PCargs.verbosity = 2
    HR_fast.newCurve(PCargs)

    print('Computing curve...')
    start = clock()
    HR_fast['EQ1'].forward()
    HR_fast['EQ1'].backward()
    print('done in %.3f seconds!' % (clock()-start))

    PCargs.name = 'LC1'
    PCargs.type = 'LC-C'
    PCargs.freepars = ['z']
    PCargs.initpoint = 'EQ1:H2'
    PCargs.MinStepSize = 1e-4
    PCargs.MaxStepSize = 0.1
    PCargs.StepSize = 0.1
    PCargs.MaxNumPoints = 240
    PCargs.LocBifPoints = []
    PCargs.NumIntervals = 200
    PCargs.NumCollocation = 6
    PCargs.NumSPOut = 30;
    PCargs.SolutionMeasures = 'all'
    PCargs.SaveEigen = True
    HR_fast.newCurve(PCargs)

    print('Computing LC-C from H1...')
    start = clock()
    HR_fast['LC1'].forward()
    print('done in %.3f seconds!' % (clock()-start))

def plot_fast_subsystem(HR_fast):
    HR_fast['EQ1'].display(figure='new', coords=('z', 'x'), stability=True)
    HR_fast['LC1'].display(figure='fig1', coords=('z','x_max'), stability=True)
    HR_fast['LC1'].display(figure='fig1', coords=('z','x_min'), stability=True)

    HR_fast.plot.toggleAll('off', bytype='P')

    HR_fast['LC1'].plot_cycles(figure='new', cycles='RG7', tlim='10T')

def create_system():
    x1 = -0.5*(1+sqrt(5))
    x = x1
    y = 1-5*x*x

    pars = {'I': 2., 'r': 0.001, 's': 4., 'x1': x1}

    icdict = {'x': x, 'y': y, 'z': pars['I']}

    # Set up model
    xstr = 'y - x*x*x + 3*x*x + I - z'
    ystr = '1 - 5*x*x - y'
    zstr = 'r*(s*(x-x1)-z)'

    DSargs = args(name='HindmarshRose')
    DSargs.fnspecs = {'Jacobian': (['t', 'x', 'y', 'z'],
                            """[[-3*x*x + 6*x, 1.0, -1.0],
                                [-10*x, -1.0, 0.0],
                                [r*s, 0.0, -1*r]]""")}
                      #'Jacobian_pars': (['t', 'I', 'r', 's', 'x1'],
                      #      """[[1.0, 0.0, 0.0, 0.0],
                      #          [0.0, 0.0, 0.0, 0.0],
                      #          [0.0, s*(x-x1)-z, r*(x-x1), -1*r*s]]""")}
    DSargs.pars = pars
    DSargs.varspecs = {'x': xstr, 'y': ystr, 'z': zstr}
    DSargs.ics = icdict
    DSargs.pdomain = {'I': [0., 6.]}

    testDS = Generator.Radau_ODEsystem(DSargs)

    return ContClass(testDS)

def create_diagram(HR, cycle=None, EQ=True, LC=True, initpoint='H1'):
    PCargs = args(name='EQ1', type='EP-C')
    PCargs.freepars = ['I']
    PCargs.StepSize = 2e-3
    PCargs.MaxNumPoints = 450
    PCargs.MaxStepSize = 1e-1
    PCargs.LocBifPoints = 'all'
    PCargs.StopAtPoints = 'B'
    PCargs.SaveEigen = True
    PCargs.verbosity = 2
    HR.newCurve(PCargs)

    if EQ:
        print('Computing curve...')
        start = clock()
        HR['EQ1'].forward()
        HR['EQ1'].backward()
        print('done in %.3f seconds!' % (clock()-start))

    PCargs.name = 'LC1'
    PCargs.type = 'LC-C'
    PCargs.freepars = ['I']
    if cycle is not None:
        PCargs.initcycle = cycle
    else:
        PCargs.initpoint = 'EQ1:' + initpoint
    PCargs.MinStepSize = 1e-4
    PCargs.MaxStepSize = 1e-1
    PCargs.StepSize = 1e-2
    PCargs.MaxNumPoints = 100
    PCargs.LocBifPoints = []
    PCargs.NumIntervals = 350
    PCargs.NumCollocation = 4
    PCargs.NumSPOut = 20;
    PCargs.SolutionMeasures = 'all'
    PCargs.SaveEigen = True
    PCargs.SaveFlow = True
    PCargs.SaveJacobian = True
    HR.newCurve(PCargs)

    if LC:
        print('Computing LC-C from H1...')
        start = clock()
        HR['LC1'].forward()
        print('done in %.3f seconds!' % (clock()-start))

def create_HR_cycle(HR, t1=3000):
    DS = HR.gensys
    DS.set(tdata=[0, t1])
    return DS.compute('BurstI2')

def fast_subsystem():
    # Bifurcation diagram for fast subsystem
    HR_fast = create_fast_subsystem()
    create_fast_diagram(HR_fast)
    plot_fast_subsystem(HR_fast)

    # Create full HR system, compute cycle and plot over bifurcation diagram
    HR = create_system()
    HRtraj = create_HR_cycle(HR, t1=1000)
    plotData = HRtraj.sample()
    plt.figure(HR_fast.plot.fig1.fig.number)
    plot(plotData['z'], plotData['x'])
    plt.xlim([1.5, 2.5])

    return HR_fast

def full_system():
    # 9 spikes: I <= 2.131; 10 spikes: I >= 2.132 (AUTO: 10 spikes > 2.131891)
    # AGREEMENT WITH RADAU, i.e. I = 2.131891 = 9 spikes, I = 2.131892 = 10 spikes
    #HR.gensys.pars['I'] = 2.1373
    #HRtraj2 = create_HR_cycle(HR)
    #plotData2 = HRtraj2.sample()
    #plot(plotData2['t'], plotData2['x'])
    #HR.gensys.pars['I'] = 2.

    # Play with bifurcation diagram in HR system

    HR = create_system()
    HRtraj = create_HR_cycle(HR)
    plotData = HRtraj.sample()
    cycle = plotData[4067:5421]
    cycle[-1] = cycle[0]
    t0 = 1310
    T0 = 452.84
    T = findApproxPeriod(HRtraj, t0, t0+T0, v='z', rtol=0.015)
    cycle = HRtraj.sample(dt=.01, tlo=t0, thi=t0+T)

    create_diagram(HR, cycle=cycle, EQ=False, LC=False)

    return HR

if __name__ == '__main__':
    HR = fast_subsystem()
    show()
