from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp
from PyDSTool.Toolbox.phaseplane import *
from Macros import DSystem

#=========SIMPLE BISTABLE ELEMENT================

pars = {'b' : 0.01, 'TX' : 1, 'k_X' : 1, 'd_X' : 1}
Xeq = 'TX*(b + pow(X,2))/(k_X + pow(X,2)) - d_X*X'
varspecs = {'X' : Xeq}
ics = {'X' : 0.2}
BE = DSystem(varspecs,pars,ics=ics,name = 'BistableElement')

BE.dobif('TX','X',num = 2000,fignum = 1)

# follow the saddle-node bifurcations
BE.follow_SN('TX','k_X',initpoint = 'LP1',fignum = 2)
BE.follow_SN('TX','k_X',initpoint = 'LP3',fignum = 2)

BE.set_par('k_X',2.)
BE.dobif('TX','X',num = 2000, fignum = 1)

# directly access the codim-1 figure
figure(1)
plt.ylim( (-3,3) )
plt.xlim( (-11,11) )

# directly access the codim-2 figure
figure(2)
plt.ylim( (0,7) )
plt.xlim( (-15,15) )


# move to bistable regime
BE.set_par('TX',6)

# integrate the system
iclist = linspace(0,0.5,4)
for ic in iclist:
    ics = {'X' : ic}
    BE.integrate(ttime = 10,dispvar = 'X',ics = ics)


# move around bistable regime
BE.set_par('TX',8)

# integrate the system at the new position in the bistable regime
iclist = linspace(0,0.5,4)
for ic in iclist:
    ics = {'X' : ic}
    BE.integrate(ttime = 10,dispvar = 'X',ics = ics, color = 'royalblue')


show()
