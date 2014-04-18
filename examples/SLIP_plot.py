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
        plt.figure(2)
        plt.ylabel('velocity')
        plt.xlabel('t')
        ydline = plot(plotData['t'], plotData['ydot'])
        zdline = plot(plotData['t'], plotData['zdot'])

    if 'position' in plottype:
        plt.figure(3)
        plt.ylabel('position')
        plt.xlabel('t')
        yline = plot(plotData['t'], plotData['y'])
        zline = plot(plotData['t'], plotData['z'])

    if 'plane' in plottype:
        plt.figure(1)
        plt.ylabel('z')
        plt.xlabel('y')
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
                        print("|delta - beta| =", abs(delta - beta))
                    plot([y2,y3],[z2,z3],'r-',linewidth=3)
