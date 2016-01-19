#!/usr/bin/python
import sys
from sys import argv
import os

from matplotlib import rc
rc('font', size=20)

from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp
from PyDSTool.Toolbox.phaseplane import *

# ================================================================================


class DSystem:

    """ General class to initialize PyDSTool Generators and ContClass 
    for ode models"""

    def __init__(self,varspecs,parameters,ics = None,name = 'model'):

        DSargs = args(name=name)
        DSargs.pars = parameters
        DSargs.varspecs = varspecs

        # set default xdomains
        ddict = { var : [-50,50] for var in varspecs.keys() }
        DSargs.xdomain = ddict

        self.ode = Generator.Vode_ODEsystem(DSargs)
        self.PC = ContClass(self.ode)
        self.DSargs = DSargs
        self.origpars = parameters.copy() # copy dict
        self.ics = ics
        self.fps = None # fix points not yet calculated

    def defpars(self):
        self.ode.pars = self.origpars.copy() # return to default values
        self.PC = ContClass(self.ode) # update PyContClass

    def set_par(self,par,val):
        self.ode.pars[par] = val
        self.PC = ContClass(self.ode) # update PyContClass

    def set_pdomain(self,domain_dic):
        self.DSargs.pdomain = domain_dic
        self.reinitialize()

    def set_xdomain(self,domain_dic):
        self.DSargs.xdomain = domain_dic
        self.reinitialize()

    def set_ics(self,ic_dic):
        self.ics = ic_dic

    # helper function to start with new DSargs like DSargs.pdomain
    def reinitialize(self):
        self.ode = Generator.Vode_ODEsystem(self.DSargs)
        self.PC = ContClass(self.ode) # update PyContClass


    def compute_fps(self, n = 13, eps = 1e-8, maxsearch = 3000):
        fp_coord = pp.find_fixedpoints(self.ode, n=n, eps=eps,maxsearch = maxsearch)
        self.fps = fp_coord
        print 'Found',len(self.fps),'fixed point(s):'
        print self.fps

    def integrate(self,ttime,dispvar = None,ics = None, color = 'crimson', fignum = 0):

        if ics is None and self.ics is None:
            print
            print 'Provide initial conditions!'
            return
        
        if ics is None:
            ics = self.ics

        self.ode.set(tdata = [0,ttime])

        #=================Initial Conditions=================
        self.ode.set( ics =  ics)
       #====================================================

        traj = self.ode.compute('pol')              # integrate SELF.ODE
        pts  = traj.sample(dt=0.01)                      # Data for plotting
        

        if dispvar:

                fig = figure(fignum)
                plt.xlabel( 't',fontsize = 20)
                plt.ylabel( dispvar,fontsize = 20)
                p = plt.plot(pts['t'], pts[dispvar],'-',label=dispvar,linewidth=2.,color = color,alpha = 0.8)
                plt.title('Trajectories')

        return pts

      #=================BIFURCATION ANALYSIS===========================================================
 


    def dobif(self,bpar,dispvar,ini_fp_ind = 0,num=1000, Plot = True, stop = ['B'], direction = 'both', log = False, fignum = 100):


        if self.fps is None:
            self.compute_fps()

        if len(self.fps) == 0:
            print
            print 'No fixed points found, exiting..'
            return

        if 'EQ1' in self.PC.curves.keys():
            del self.PC.curves['EQ1']

        PCargs = args(name='EQ1', type='EP-C')
        PCargs.initpoint = self.fps[ini_fp_ind]
        PCargs.freepars = [bpar]
        PCargs.StepSize = 1e-3
        PCargs.MaxNumPoints = num
        PCargs.MaxStepSize = 1e-2
        PCargs.MinStepSize = 1e-20
        PCargs.LocBifPoints = 'all'
        PCargs.verbosity = 2
        PCargs.SaveEigen    = True                       # to tell unstable from stable branches
        PCargs.SaveJacobian    = True
        PCargs.StopAtPoints = stop
        PCargs.FuncTol = 1e-8
        PCargs.VarTol = 1e-8
        self.PC.newCurve(PCargs)

        print 'Computing curve...'
        start = clock()
        if direction == 'both':
            self.PC['EQ1'].forward()
            self.PC['EQ1'].backward()

        elif direction == 'backward':
            self.PC['EQ1'].backward()

        elif direction == 'forward':
            self.PC['EQ1'].forward()

        else: print 'no direction given'

        print 'done in %.3f seconds!' % (clock()-start)


        if Plot:
            self.PC['EQ1'].display((bpar,dispvar),stability=True,figure = fignum,linewidth = 2.5)
            figure(fignum )
            plt.title("Bifurcations Analysis")

            if log:
                plt.xscale('log')
                plt.xlim( (1e-6,1.1*max(self.PC['EQ1'].sol[bpar])) )

            #self.PC.plot.togglePoints('off')
            self.PC.plot.toggleLabels('off')

            self.PC.plot.togglePoints(visible='on', bylabel=None,bytype=['H','LP'])
            self.PC.plot.toggleLabels(visible='on', bylabel=None,bytype=['H','LP'])

                

    def follow_SN(self,bpar,bpar2,num = 150, initpoint = 'LP1', Plot = True, log = False, fignum = 101):


        if 'EQ1' not in self.PC.curves.keys():
            print
            print 'Run equilibrium bifurcation analysis first !'
            return

        if 'SN1' in self.PC.curves.keys():
            del self.PC.curves['SN1']

        PCargs = args(name='SN1', type='LP-C')
        PCargs.initpoint = 'EQ1:' + initpoint
        PCargs.freepars = [bpar,bpar2]
        PCargs.StepSize = 0.001
        PCargs.MaxNumPoints = num
        PCargs.MaxStepSize = 5e-1
        PCargs.MinStepSize = 1e-40
        PCargs.LocBifPoints = 'all'
        PCargs.verbosity = 2
        PCargs.SaveEigen    = True                       # to tell unstable from stable branches
        PCargs.SaveJacobian    = True
        PCargs.StopAtPoints = ['B']
        self.PC.newCurve(PCargs)

        print self.PC.curves

        print 'Computing LP 1 curve...'
        start = clock()
        self.PC['SN1'].forward()
        self.PC['SN1'].backward()
        print 'done in %.3f seconds!' % (clock()-start)

        if Plot:
            self.PC['SN1'].display((bpar,bpar2),'--',stability=False, color = 'orange',figure=fignum,linewidth = 2.5,label = 'Saddle-node' )
            plt.title('Codimension-2 Bifurcations')
            if log:
                plt.xscale('log')
                plt.xlim( (1e-6,1.1*max(self.PC['SN1'].sol[bpar])) )

            self.PC['SN1'].plot.togglePoints(visible = 'off', bytype = ['P','B'])
            self.PC['SN1'].plot.toggleLabels(visible = 'off', bytype = ['P','B'])

            self.PC['SN1'].plot.togglePoints(visible='on', bylabel=None,bytype=['BT','GH','ZH','CP'])
            self.PC['SN1'].plot.toggleLabels(visible='on', bylabel=None,bytype=['BT','GH','ZH','CP'])


    def follow_H(self,bpar,bpar2,num = 150, initpoint = 'H1',Plot = True, log = False, fignum = 101):

        if 'EQ1' not in self.PC.curves.keys():
            print
            print 'run equilibrium bifurcation analysis first !'
            return

        if 'HC1' in self.PC.curves.keys():
            del self.PC.curves['HC1']

        PCargs = args(name='HC1', type='H-C2')
        PCargs.initpoint = 'EQ1:' + initpoint
        PCargs.freepars = [bpar,bpar2]
        PCargs.StepSize = 0.001
        PCargs.MaxNumPoints = num
        PCargs.MaxStepSize = 1e-1
        PCargs.MinStepSize = 1e-40
        #PCargs.LocBifPoints = 'all'
        PCargs.LocBifPoints = ['BT']
        PCargs.verbosity = 2
        PCargs.SaveEigen    = True                       # to tell unstable from stable branches
        PCargs.SaveJacobian    = True
        PCargs.StopAtPoints = ['B','BT']
        self.PC.newCurve(PCargs)

        print 'Computing HC1 curve...'
        start = clock()
        self.PC['HC1'].forward()
        self.PC['HC1'].backward()
        print 'done in %.3f seconds!' % (clock()-start)
    
        if Plot:
            self.PC['HC1'].display((bpar,bpar2),figure=fignum, color = 'darkseagreen', label = 'Hopf', linewidth = 2.5)
            plt.title('Codimension-2 Bifurcations')

            if log:
                plt.xscale('log')
                plt.xlim( (1e-6,1.1*max(self.PC['HC1'].sol[bpar])) )

        self.PC.plot.togglePoints('off')
        self.PC.plot.toggleLabels('off')

        self.PC.plot.togglePoints(visible='on', bylabel=None,bytype=['BT','GH','ZH','CP'])
        self.PC.plot.toggleLabels(visible='on', bylabel=None,bytype=['BT','GH','ZH','CP'])


    def compute_LC(self,bpar, dispvar,num=500, initpoint = 'H1', Plot = True,period = False):

        if 'EQ1' not in self.PC.curves.keys():
            print
            print 'run equilibrium bifurcation analysis first !'
            return


        if 'LC1' in self.PC.curves.keys():
            del self.PC.curves['LC1']

        # remove old stuff
        os.system('rm -rf *dop*')
        os.system('rm -Rf *auto*')


        PCargs = args(name='LC1', type='LC-C')
        PCargs.freepars = [bpar]
        PCargs.initpoint = 'EQ1:' + initpoint
        PCargs.MinStepSize = 1e-20
        PCargs.MaxStepSize = 0.01
        PCargs.StepSize = 0.00001
        PCargs.MaxNumPoints = num
        PCargs.NumSPOut = 10000
        PCargs.LocBifPoints = 'LPC'
        PCargs.verbosity = 0 #2
        PCargs.SolutionMeasures = 'all'
        PCargs.SaveEigen    = True                       # to tell unstable from stable branc
        self.PC.newCurve(PCargs)


        print 'Computing LC1 curve...'
        start = clock()
        self.PC['LC1'].forward()
        self.PC['LC1'].backward()


        if Plot:
        
            self.PC['LC1'].display((bpar,dispvar), figure=100,stability = True,color = 'royalblue',linewidth = 2.5, label = 'Limit cycle')
            self.PC['LC1'].display((bpar,dispvar+'_min'), figure=100,stability = True,color = 'royalblue',linewidth = 2.5,label = None)


        if period:
            self.PC['LC1'].display((bpar,'_T'), figure=66,linewidth = 2,color = 'blue',label = 'Period of limit cycle')
            figure(66)
            plt.title(' ')
            plt.tick_params(axis='both', which='major', labelsize=17)
            plt.xlabel(bpar,fontsize = 20)
            leg = plt.legend(prop = {'size': 16}, loc = 1,fancybox = True)
            if leg:
                leg.get_frame().set_alpha(0.6)
            plt.ylabel('Period (h)')

        self.PC.plot.togglePoints('off')
        self.PC.plot.toggleLabels('off')

        self.PC.plot.togglePoints(visible='on', bylabel=None,bytype=['H','LP','LPC','PD','NS'])
        self.PC.plot.toggleLabels(visible='on', bylabel=None,bytype=['H','LP','LPC','PD','NS'])


