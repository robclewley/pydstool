"""
    Tests for the Generator class InterpolateTable with
    piecewise-constant vs. piecewise-linear interpolation.

    Robert Clewley, March 2006.
"""

from PyDSTool import *


print('Test InterpolateTable comparing piecewise-constant and piecewise-')
print('linear interpolation of sine function')

timeData = linspace(0, 10, 30)
sindata = sin(timeData)
xData = {'sinx': sindata}
pcwc_interp = InterpolateTable({'tdata': timeData,
                              'ics': xData,
                              'name': 'interp0d',
                              'method': 'constant',
                              'checklevel': 1,
                              'abseps': 1e-5
                              }).compute('interp')

pcwl_interp = InterpolateTable({'tdata': timeData,
                              'ics': xData,
                              'name': 'interp1d',
                              'method': 'linear',
                              'checklevel': 1,
                              'abseps': 1e-5
                              }).compute('interp')


assert np.allclose(pcwc_interp(5.68)['sinx'], -0.551, atol=0.001)
assert np.allclose(pcwl_interp(5.68)['sinx'], -0.559, atol=0.001)

### DEMO
# x = linspace(0,10,300)
# plot(x, pcwc_interp(x, ['sinx']))
# plot(x, pcwl_interp(x, ['sinx']))
# plot(x, sin(x))
# show()

print("   ...passed")
