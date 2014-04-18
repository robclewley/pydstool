"""
    Test hybrid system with external inputs for vode integrator.

    Robert Clewley, September 2006
"""
from PyDSTool import *
from copy import copy

# ------------------------------------------------------------

tdomain = [0,100]
timeData = linspace(tdomain[0], tdomain[1], 500)
sindata = sin(timeData)+2*cos(timeData/2+pi/5)
xData = {'in': sindata}
# This is an archaic way to generate this
# Better now is to create a Pointset of data then
# use pointset_to_traj, and pass the variable from
# that in to the creation of the Generator
my_input = InterpolateTable({'tdata': timeData,
                              'ics': xData,
                              'name': 'interp1d',
                              'method': 'linear',
                              'checklevel': 1,
                              'abseps': 1e-6
                              }).compute('interp')

def makeModel(target1, target2):
    targetlang = ['','']
    for i, target in enumerate([target1, target2]):
        if target == 'Vode':
            targetlang[i] = 'python'
        elif target in ['Dopri', 'Radau']:
            targetlang[i] = 'c'
        else:
            raise ValueError("Invalid target")

    DSargs = args()
    DSargs.varspecs = {'x': 'in+1', 'inval': 'in', 'state': '0'}
    DSargs.vars = ['x', 'state']  # inval is an aux variables
    DSargs.algparams = {'init_step': 0.01, 'max_step': 0.5,
                           'max_pts': 20000}
    DSargs.xdomain = {'x': [-Inf, 5], 'state': 0}
    DSargs.xtype = {'state': int}
    DSargs.tdomain = tdomain
    DSargs.checklevel = 2
    DSargs.ics = {'x': 0, 'state': 0}
    DSargs.inputs = {'in': my_input.variables['in']}
    DSargs.name = 'gen1'
    # switch to gen2 when x increases through 5
    DSargs.events = Events.makeZeroCrossEvent('x-5', 1,
                                        {'name': 'ev_togen2',
                                         'eventtol': 1e-8,
                                         'precise': True,
                                         'term': True}, ['x'],
                                        targetlang=targetlang[0])

    DSargs2=copy(DSargs)
    DSargs2.name = 'gen2'
    # aux var 'state' == 1 refers to this, gen2
    DSargs2.varspecs = {'x': '-abs(in)/2', 'inval': 'in', 'state': '0'}
    # gen2 will only make steps at points given in timeData, ignoring
    # init_step parameter. However, since the event requires 'precise'
    # finding, any events found at non-'special' times will also be
    # included in the output
    DSargs2.xdomain = {'x': [-5, Inf], 'state': 1}
    DSargs2.ics['state'] = 1

    # switch to gen1 when x decreases through -5
    DSargs2.events = Events.makeZeroCrossEvent('x+5', -1,
                                        {'name': 'ev_togen1',
                                         'eventtol': 1e-8,
                                         'precise': True,
                                         'term': True}, ['x'],
                                        targetlang=targetlang[1])

    mc = ModelConstructor('test_inputs')
    mc.addModelInfo(DSargs, target1+'_ODEsystem')
    mc.addModelInfo(DSargs2, target2+'_ODEsystem')
    # now that all generators declared, can introduce mappings
    mc.mapEvent('gen1', 'ev_togen2', 'gen2', {"state": "1"})
    mc.mapEvent('gen2', 'ev_togen1', 'gen1', {"state": "0"})
    return mc.getModel()

def test1(m):
    print('\nTest 1, integrating... ')
    m.compute(trajname='test1', tdata=tdomain, ics={'x':1.1, 'state': 1},
          verboselevel=2)
    assert len(m.trajectories['test1'].trajSeq) == 9, \
       "Incorrect number of hybrid segments"
    assert allclose(m('test1', [16,30,90])['x'], [2.9, -5.2, 4.4], 0.1), \
           "Inaccurate hybrid computation"

    # test transferral of input values through wrappers without being
    # affected
    diffs = []
    for t in range(tdomain[0], tdomain[1], 10):
        diffs.append(abs(my_input(t)-m('test1',t)['inval']))
    assert all(array(diffs)<2e-2), \
           "Error in hybrid system's version of external input is too large"
    return m.sample('test1')

def test2(m):
    print("\nTest 2, integrating ...")
    m.compute(trajname='test2', tdata=tdomain, ics={'x':5.1, 'state': 1})
    assert len(m.trajectories['test2'].trajSeq) == 8, \
       "Incorrect number of hybrid segments"
    assert allclose(m('test2', [16,30,90])['x'], [-3.4, 1.7, -4.3], 0.1), \
           "Inaccurate hybrid computation"

def doPlot(plotData):
    print('Preparing plot')
    plt.ylabel('x')
    plt.xlabel('t')
    xline=plot(plotData['t'], plotData['x'])
    iline=plot(plotData['t'], plotData['inval'])
    iline_true=plot(timeData, sindata)

if __name__=='__main__':
    m = makeModel('Vode','Vode')

    print("Testing vode integrator")
    plotData = test1(m)
    test2(m)

    doPlot(plotData)
    show()
