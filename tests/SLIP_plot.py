"""Plot library function for SLIP model.

    Robert Clewley, May 2005.
"""

from PyDSTool import *

# ----------------------------------------------------------------

def SLIP_plot(SLIP, trajname, plottype=['plane'], legs=True, verboselevel=0):

    if not isinstance(plottype, list):
        plottype = [plottype]

    plotData = SLIP.sample(trajname, dt=0.025)

    if 'velocity' in plottype:
        pylab.figure(2)
        pylab.ylabel('velocity')
        pylab.xlabel('t')
        ydline = plot(plotData['t'], plotData['ydot'])
        zdline = plot(plotData['t'], plotData['zdot'])

    if 'position' in plottype:
        pylab.figure(3)
        pylab.ylabel('position')
        pylab.xlabel('t')
        yline = plot(plotData['t'], plotData['y'])
        zline = plot(plotData['t'], plotData['z'])

    if 'plane' in plottype:
        pylab.figure(1)
        pylab.ylabel('z')
        pylab.xlabel('y')
        pline=plot(plotData['y'], plotData['z'])

        evs = SLIP.getTrajEventTimes(trajname)

        numTDs = len(evs['touchdown'])
        numLOs = len(evs['liftoff'])

        if legs:
            for evix in range(numTDs):
                TDev = evs['touchdown'][evix]
                pt1 = SLIP(trajname,TDev)
                y1 = pt1('y')
                z1 = pt1('z')
                beta = SLIP.query('pars')['beta']
                y2 = y1+cos(beta)
                z2 = 0
                plot([y1,y2],[z1,z2],'r-', linewidth=3)
                if evix < numLOs:
                    LOev = evs['liftoff'][evix]
                    pt2 = SLIP(trajname,LOev)
                    y3 = pt2('y')
                    z3 = pt2('z')
                    if verboselevel > 0:
                        delta = math.asin(z3)
                        print "|delta - beta| =", abs(delta - beta)
                    plot([y2,y3],[z2,z3],'r-',linewidth=3)
