"""Test of a 2D kinematic stick-finger model in contact with a wheel.
  (Two events here to test Generator's ability to cope with
   multiple event occurrences.)

   R. Clewley, March 2005
"""

from PyDSTool import *
from copy import copy


# -----------------------------------------------------------------


def makeModel3(parDict, dt=0.1):
    pd = copy(parDict)

    rhsstr = "(vy*vy*t+(-r*sin(a)+h0)*vy+vx*r*cos(a)"+\
      "-vx*r*cos(alpha0)+vx*l*sin(theta0)+vx*vx*t)/(r*(-r*sin(a)*cos(alpha0)+"+\
      "sin(a)*l*sin(theta0)+sin(a)*vx*t+cos(a)*h0+cos(a)*vy*t))"

    # c = centre of mass
    # p = point of contact
    # a = alpha
    # o = joint of finger
    # f = end of finger (=p during contact)
    specdict_contact = {'a': "adot(a, t)",
                'px': '-r*cos(alpha0)+r*cos(a)',
                'py': '-h0+r*sin(a)',
                'cx': '-r*cos(alpha0)',
                'cy': '-h0',
                'ox': '-l*sin(theta0)-vx*t',
                'oy': 'vy*t',
                'fx': '-r*cos(alpha0)+r*cos(a)',
                'fy': '-h0+r*sin(a)',
                'ad': 'adot(a,t)',
                'incontact': '0'}  # 0 derivative, i.e. constant
    fnspecs = {'adot': (['a', 't'], rhsstr)}

    # post-contact -- finger decay rate to rest position
    kf = pd['kf']

    # contact phase
    vx=pd['vx']
    r=pd['r']
    h0=pd['h0']
    l=pd['l']
    d0=pd['d0']
    k=pd['k']
    theta0=pd['theta0']
    vy=pd['vy']
    alpha0=pd['alpha0']
    alpha1=pd['alpha1']

    if l*cos(theta0) + r < h0: # then alpha_0 has no real solution
        raise ValueError('theta0 too large or h0 too large')

    # Example using the python-specific event factory function
    tang_ev = Events.makePythonStateZeroCrossEvent('a', "alpha1", 1,
                                                {'name': 'tangency',
                                                   'eventtol': 1e-5,
                                                   'eventdelay': 1e-6,
                                                   'starttime': 0,
                                                   'term': True,
                                                   })

    d_ev = Events.makeZeroCrossEvent(rhsstr+"-deriv_thresh", 1,
                            {'eventtol': 1e-5,
                             'eventdelay': 1e-6,
                             'starttime': 0,
                             'term': True,
                             'name': 'deriv'}, ['a'], list(pd.keys()))
    DSargs = {'varspecs': specdict_contact,
              'fnspecs': fnspecs,
              'xdomain': {'a': [0, 2*pi], 'incontact': 1},
              'vars': ['a', 'incontact'],
              'ics': {'a': alpha0, 'incontact': 1},
              'xtype': {'incontact': int},
              'algparams': {'init_step': dt, 'stiff': True,
                              },
              'pars': pd,
              'name': 'contact_epoch',
              'events': [tang_ev, d_ev],
              }
    ode = embed(Generator.Vode_ODEsystem(DSargs), tdata=[0,200])
    print(ode.query('ics'))

    ode_MI = intModelInterface(ode)

    # add post-contact phase to traj
    specdict_decay = {'a': 'initcond(a) + initcond(ad)*(1-exp(-k*t))/k',
                      'ad': 'initcond(ad)*exp(-k*t)',
                      'incontact': '0',
                      'cx': '-r*cos(alpha0)',
                      'cy': '-h0',
                      'px': '-r*cos(alpha0)+r*cos(a)',
                      'py': '-h0+r*sin(a)',
                      'ox': 'initcond(ox)-vx*t',
                      'oy': 'initcond(oy)*exp(-kf*t)',
                      'fx': 'initcond(fx)-vx*t',
                      'fy': 'initcond(oy)*exp(-kf*t) - abs(initcond(oy)-initcond(fy))' #'initcond(fy)'
                      }

    decayargs = {'name': 'decay_epoch',
                 'tdomain': [0, 1.5],
                 'xdomain': {'a': [0, 2*pi], 'incontact': 0},     # not redundant: crucial for initial gen selection
                 'vars': ['a', 'incontact'],
                 'ics': {'a': alpha1, 'incontact': 0},
                 'xtype': {'incontact': int},
                 'varspecs': specdict_decay,
                 'pars': pd
                 }
    decay = embed(Generator.ExplicitFnGen(decayargs), tdata=decayargs['tdomain'])
    decay_MI = intModelInterface(decay)

    # epoch map to new contact phase
    epmapping = EvMapping({"a": "alpha0",
        "incontact": "1"}, model=decay)

    allnames = ['contact_epoch', 'decay_epoch']
    contactInfo = makeModelInfoEntry(ode_MI, allnames,
                                           [('tangency', 'decay_epoch'),
                                            ('deriv', 'decay_epoch')])
    decayInfo = makeModelInfoEntry(decay_MI, allnames,
                                   [('time', ('contact_epoch', epmapping))])
    modelInfoDict = makeModelInfo([contactInfo, decayInfo])

    fullmodel = Model.HybridModel({'name': 'finger_model3',
           'modelInfo': modelInfoDict,
           'abseps': 1e-5})
    # promote aux vars to be internal vars in the model, leaving only 'a' and 'incontact' as observable vars
    fullmodel.forceIntVars(['ad', 'cx', 'cy', 'px', 'py', 'ox', 'oy', 'fx', 'fy'])
    return fullmodel


def compute_alphas(pd):
##    vx=pd['vx']
    r=pd['r']
    h0=pd['h0']
    l=pd['l']
    d0=pd['d0']
##    k=pd['k']
    theta0=pd['theta0']
##    vy=pd['vy']
    pd['alpha0'] = asin((h0-l*cos(theta0))/r)
    pd['alpha1'] = (pi-asin(h0/(r+l)))*0.8
    # alpha1 is a fudge so that stop just before singularity (tangency)


# ----------------------------------------------------------------------

pars = {'vx': 4., 'r': 2.31, 'h0': 3.3, 'l': 3.2, 'd0': 2., 'k': 2.,
        'theta0': 0.8*pi/3, 'vy': 2., 'kf': 1., 'deriv_thresh': 200}

compute_alphas(pars)   # changes pars

fullmodel = makeModel3(pars, dt=0.01)

print("Computing trajectory")
print("Using verboselevel=2 for debug-level info ...")
fullmodel.compute(trajname='run1',
                      tdata=[0, 3.0],
                      ics={'a': pars['alpha0'],
                                 'incontact': 1},
                      verboselevel=2)

print("\nPlotting output")
plotData = fullmodel.sample('run1', coords=['a','ad'],
                                           dt=0.01)
afig = plt.figure()
aline = plot(plotData['t'], plotData['a'])
##    adotfig = plt.figure()
adline = plot(plotData['t'], plotData['ad'])
show()
