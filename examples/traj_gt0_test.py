from PyDSTool import *
vals = [0.1, 0.2, 0.4, 0.8, 1.0, 1.1, 0.97, 0.85, 0.7, 0.67]
traj = numeric_to_traj([vals], 'v', coordnames=['v'], indepvar=arange(10),
                       discrete=False)
p1 = traj(0)
q1 = traj(1.7)

vals0 = traj.sample(tlo=1.8, thi=5.8)
vals1 = traj.sample(tlo=1.8, thi=5.8)
vals2 = traj.sample(tlo=1.8, thi=5.8, dt=0.1)  # dt makes no diff here
assert all(vals1['t'] == vals2['t'])
assert all(vals1['v'] == vals2['v'])

traj.globalt0 = 1.7
try:
    shouldfail = traj(0, asGlobalTime=True)
except PyDSTool_BoundsError:
    # 0 in global time would be -1.7 in local time, for which there is no
    # defined point in vals
    pass
else:
    raise AssertionError
p2 = traj(0)
p3 = traj(traj.globalt0, asGlobalTime=True)
q2 = traj(traj.globalt0*2, asGlobalTime=True)

assert p1 == p2 == p3
assert q1 == q2

# sampling happens in global time by default
vals3 = traj.sample(tlo=1.8, thi=5.8, asGlobalTime=False)
assert all(traj.underlyingMesh()['v'][0] == arange(10))
assert all(vals0 == vals3)
assert vals3['t'][0] == 2

vals1_2 = traj.sample(tlo=1.8, thi=5.8)
assert all(traj.underlyingMesh()['v'][0] == arange(10))
vals2_2 = traj.sample(tlo=1.8, thi=5.8, dt=0.1)  # dt makes no diff here
assert all(vals1_2['t'] == vals2_2['t'])
assert all(vals1_2['v'] == vals2_2['v'])
assert vals1_2['v'][0] == 0.2

vals4 = traj.sample()
assert vals4[0] == p1

vals5 = traj.sample(tlo=1.8, thi=5.8, dt=0.1, precise=True)
assert vals5[2]['v'] == 0.13
assert all(traj.underlyingMesh()['v'][0] == arange(10))
assert allclose(vals5['t'], linspace(1.8, 5.8, int((5.8-1.8)/0.1+1)))
vals6 = traj.sample(tlo=1.8, thi=5.8, dt=2.1)  # skips some b/c not precise
assert all(vals6['t'] == array([2.7, 3.7, 5.7]))
assert vals6[0]['v'] == 0.2

print("\nAll trajectory sampling tests passed")
