import PyDSTool as dst
from PyDSTool import args
import numpy as np
from matplotlib import pyplot as plt

# The approximate map for the triple-zero unfolding of the Lorenz ODEs
# see http://www.math.pitt.edu/~bard/bardware/tut/newstyle.html#triple

# Map given by:
# z(n+1) = a+b*(z-c)^2
# with z(0)=2

# For Lyapunov exponent:
# zp(n+1) = zp+log(abs(2*b*(z-c)))
# with zp(0) = 0, so that Lyapunov Exponent L will be zp/(t+1)
# (because t starts at 0)

DSargs = args(name='Lorenz map')
DSargs.varspecs = {'z': 'a+b*(z-c)**2',   # note syntax for power
                   'zp': 'zp+log(abs(2*b*(z-c)))',
                   'L': 'zp/(t+1)'}  # L is an auxiliary variable
DSargs.tdomain = [0, 200]
DSargs.pars = args(a=2.93, b=-1.44, c=1.85)
DSargs.vars = ['z', 'zp']   # Implicitly, then, L must be an auxiliary variable
DSargs.ics = {'z': 2, 'zp': 0}
DSargs.ttype = int  # force independent variable type to be integer (discrete time)
lmap = dst.MapSystem(DSargs)

traj = lmap.compute('test')
pts = traj.sample()

plt.figure(1)
plt.plot(pts['t'], pts['z'], 'ko-')
plt.title(lmap.name)
plt.xlabel('t')
plt.ylabel('z')

plt.figure(2)
plt.plot(pts['t'], pts['L'], 'ro')
plt.title('Lyapunov exponent')
plt.ylabel('L')
plt.xlabel('t')
plt.show()