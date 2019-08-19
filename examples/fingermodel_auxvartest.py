"""Testing auxiliary variables for a model of a stick-finger on a wheel."""

from PyDSTool import *

# test part of a finger model (version 3) -- for auxiliary variable testing
pars = {'vx': 10.4707, 'vz': 70.574, 'r': 44.7, 'h0': 70.6765, 'l': 40.2,
        'd0': 12.5, 'k': .215, 'alpha1': 1.7904, 'ad0': 40., 'alpha0': 0.8,
        'theta0': 0.8676, 'kf': .6, 'decay_end_t': 0.8}
spec_decay = {'a': 'initcond(a) + ad0*(1-exp(-k*t))/k',
                  'px': '-sin(alpha0)+cos(a)',
                  'ox': '30-vx*t',
                  }
time_ev = Events.makePythonStateZeroCrossEvent('t', 'decay_end_t', 1,
                                            {'precise': True,
                                             'active': True,
                                             'term': True,
                                             'name': 'decay_time'
                                             })
decayargs = {'name': 'testexplicitfn',
             'vars': ['a'],
             'tdomain': [0, 1.0],
             'ics': {'a': 1.7},
             'varspecs': spec_decay,
             'pars': pars,
             'events': [time_ev]
             }
decay = ExplicitFnGen(decayargs)
dtraj1 = decay.compute('testtraj1')
decay.set(ics={'a': 0.5})
dtraj2 = decay.compute('testtraj2')
print("To test that separate trajectories output from generator actually")
print(" have different values we compare two traj's values")
print("These should be different:")
result = dtraj1.variables['a'](0.2) != dtraj2.variables['a'](0.2)
print("dtraj1.variables['a'](0.2) != dtraj2.variables['a'](0.2)? =>", \
      result)
print("This demonstrates avoiding an insufficiently deep 'copy' of new")
print(" Variables into new trajectories whereby new Variables overwrite")
print(" old ones in other Trajectory objects")
assert result
