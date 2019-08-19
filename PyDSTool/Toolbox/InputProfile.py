
from PyDSTool import *

def makeSpikeProfile(rawT, rawX, baseline=0.0, upshift=0.0, peak='max', endTime=1000000):

    profile = {}
    assert (len(rawT) > 2), 'profiles must have length > 2'
    assert (len(rawT) == len(rawX)), 'dep. and indep. variable profiles must have same length.'

    if peak == 'max':
        n = rawX.index(max(rawX))
        m = rawX.index(min(rawX))
    else:
        n = rawX.index(min(rawX))
        m = rawX.index(max(rawX))

    # Determine starting time from secant to peak
    if rawX[0] == baseline:
        startT = [rawT[0] - (rawT[1] - rawT[0])]
    else:
        startT = [(baseline - rawX[0] - upshift) * (rawT[n] - rawT[0])/(rawX[n] - rawX[0])
                  + rawT[0]]
    if rawX[-1] == baseline:
        endT = [rawT[-1] + rawT[-1] - rawT[-2]]
    else:
        endT =  [(baseline - rawX[n] - upshift) * (rawT[-1] - rawT[n])/(rawX[-1] - rawX[n])
                 + rawT[n]]

    profile['t'] = array( startT + rawT + endT ) - startT[0]
    if upshift != 0:
        shiftX = [x + upshift for x in rawX]
    else:
        shiftX = rawX
    profile['x'] = array( [baseline] + shiftX + [baseline] )

    profile['peakT'] = profile['t'][n + 1]
    profile['peakX'] = profile['x'][n + 1]
    profile['peakIdx'] = n+1

    profile['perturbWhole'] = profile['t'][-1] - profile['t'][0]
    profile['perturbMain'] = profile['t'][-2] - profile['t'][1]

    profile['spikeHeight_base'] = abs(rawX[n] - baseline)
    profile['spikeHeight_spike'] = abs(rawX[n] - rawX[m])

    profile['halfHeight_base'] = 0.5 * profile['spikeHeight_base'] + baseline
    profile['halfHeight_spike'] = 0.5 * profile['spikeHeight_spike'] + baseline

    profile['baseWidth'] = abs(rawT[-1] - rawT[0])
    profile['baseline'] = baseline

    # Use interpolation table to determine spike half-width
    upsign = 1
    downsign = 1

    if peak == 'max':
        downsign = -1
    else:
        upsign = -1

    upT = profile['t'][0:n+2]
    downT = profile['t'][n+1:-1]

    upX = upsign*profile['x'][0:n+2]
    upX = upX.tolist()
    downX = downsign*profile['x'][n+1:-1]
    downX = downX.tolist()

    uTable = InterpolateTable({'tdata': upX, 'ics': makeDataDict(['up'], [upT]), 'name': 'up'})
    dTable = InterpolateTable({'tdata': downX, 'ics': makeDataDict(['down'], [downT]), 'name': 'down'})
    uTraj = uTable.compute('up')
    dTraj = dTable.compute('down')

    profile['halfWidth_up'] = (uTraj(upsign*profile['halfHeight_base']), uTraj(upsign*profile['halfHeight_spike']))
    profile['halfWidth_down'] = (dTraj(downsign*profile['halfHeight_base']), dTraj(downsign*profile['halfHeight_spike']))
    profile['halfWidth_base'] = abs(profile['halfWidth_up'][0] - profile['halfWidth_down'][0])
    profile['halfWidth_spike'] = abs(profile['halfWidth_up'][1] - profile['halfWidth_down'][1])

    profile['eSynRev'] = 0.88317057588863412 * (profile['peakX'] - profile['baseline']) + profile['baseline']
    profile['iSynRev'] = profile['baseline']
    profile['halfAct'] = 0.54989866045896085 * (profile['peakX'] - profile['baseline']) + profile['baseline']

    if endTime is not None and endTime > profile['t'][-1]:
        profile['t'] = array( profile['t'].tolist() + [endTime] )
        profile['x'] = array( profile['x'].tolist() + [profile['x'][-1]] )

    return profile

def makeInputTable(profile, name):

    xData = makeDataDict([name], [profile['x']])
    tData = profile['t']
    iTable = InterpolateTable({'tdata': tData, 'ics': xData, 'name': name})

    iTableTraj = iTable.compute(name)

    return iTableTraj


def plotProfile(profile, figure='new'):

    if figure == 'new':
        plt.figure()

    plot(profile['t'], profile['x'], 'b-')
    plot(profile['halfWidth_up'][0], profile['halfHeight_base'], 'b^')
    plot(profile['halfWidth_up'][1], profile['halfHeight_spike'], 'r^')
    plot(profile['halfWidth_down'][0], profile['halfHeight_base'], 'bo')
    plot(profile['halfWidth_down'][1], profile['halfHeight_spike'], 'ro')

    plot(profile['peakT'], profile['peakX'], 'go')
