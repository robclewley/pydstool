"""
    Test Vode_ODEsystem with Jacobian, using "single-step" integration
     of a stiff system, and using the 'continue' option of the integrator.
    Corresponds to the system tested at the end of /PyDSTool/scipy_ode.py
     except with an added non-terminal event.

    Note: This system does not integrate successfully using Radau!

    Contrast the specification method for this system to that in the
     similar test that uses ModelSpecs to build the expressions.

    Robert Clewley, August 2005.
"""

from PyDSTool import *

DSargs = args(fnspecs={'Jacobian': (['t','y0','y1','y2'],
                                """[[-0.04,  1e4*y2       ,  1e4*y1 ],
                                    [ 0.04, -1e4*y2-6e7*y1, -1e4*y1 ],
                                    [ 0.0 ,  6e7*y1       ,  0.0    ]]"""),
                        'ydot0': (['y0', 'y1', 'y2'], "-0.04*y0 + 1e4*y1*y2"),
                        'ydot2': (['y0', 'y1', 'y2'], "3e7*y1*y1")}
              )
DSargs.varspecs = {"y0": "ydot0(y0,y1,y2)",
                      "y2": "ydot2(y0,y1,y2)",
                      "y1": "-ydot0(y0,y1,y2)-ydot2(y0,y1,y2)"}
DSargs.tdomain = [0.,1e20]
DSargs.ics = {'y0': 1.0, 'y1': 0., 'y2': 0.}
DSargs.algparams = {'init_step':0.4, 'strictdt': True, 'stiff': True,
                    'rtol': 1e-4, 'atol': [1e-8,1e-14,1e-6]}
DSargs.events = makeZeroCrossEvent('y0-0.001', -1, {'name': 'thresh_ev',
                       'eventtol': 10,
                       'bisectlimit': 20,
                       'eventinterval': 500,
                       'eventdelay': 0,  #otherwise cannot catch event with only one step per run
                       'term': False}, ['y0'])
DSargs.checklevel = 2
DSargs.name = 'jactest'
testODE = Vode_ODEsystem(DSargs)

print("Defined the following internal Python function for Jacobian:")
print(testODE.funcspec.auxfns['Jacobian'][0], "\n")


tvals = [0.4*10**i for i in range(0,12)]
t0 = 0.

for t1 in tvals:
    dt = t1-t0
    print("\n============================================\nAt t=", end=' ')
    print(t1, "using dt =", dt)
    testODE.set(tdata=[t0,t1],
                    algparams={'init_step': dt}
                   )
    if t0 >0.:
        print(testODE._solver.y)
    traj = testODE.compute('test', 'c')  # c for continue
    testODE.diagnostics.showWarnings()
    et = testODE.getEventTimes()['thresh_ev']
    if et != []:
        print("\n****** Event found at t =", et, "\n")
    t0 = t1
    print(traj(t1))


print("\nCompare results with the output directly from the scipy_ode.py test")
print("The values from a test integration performed with scipy_ode.py " \
      + "are listed in the comments at the end of the script")

##At t=0.0  y=[ 1.  0.  0.]
##At t=0.4  y=[ 9.85172114e-001  3.38639538e-005  1.47940221e-002]
##At t=4.0  y=[ 9.05518679e-001  2.24047569e-005  9.44589166e-002]
##At t=40.0  y=[ 7.15827070e-001  9.18553479e-006  2.84163745e-001]
##At t=400.0  y=[ 4.50518662e-001  3.22290136e-006  5.49478115e-001]
##At t=4000.0  y=[ 1.83202242e-001  8.94237031e-007  8.16796863e-001]
##At t=40000.0  y=[ 3.89833646e-002  1.62176779e-007  9.61016473e-001]
##At t=400000.0  y=[ 4.93828363e-003  1.98499788e-008  9.95061697e-001]
##At t=4000000.0  y=[ 5.16819063e-004  2.06833253e-009  9.99483179e-001]
##At t=40000000.0  y=[ 5.20200703e-005  2.08090952e-010  9.99947980e-001]
##At t=400000000.0  y=[ 5.21215549e-006  2.08487297e-011  9.99994788e-001]
##At t=4000000000.0  y=[ 5.25335126e-007  2.10134087e-012  9.99999475e-001]
##At t=40000000000.0  y=[ 5.63729748e-008  2.25491907e-013  9.99999944e-001]
