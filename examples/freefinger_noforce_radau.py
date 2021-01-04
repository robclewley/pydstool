"""Implementation of a model for a freely moving finger, with
 no forces.

   Robert Clewley and Madhu Venkadesan, January 2006.
"""

from PyDSTool import *
from time import perf_counter
from scipy.io import *

# ------------------------------------------------------------

def make_finger_massmatrix(name, par_args, ic_args, evs=None, nobuild=False):

    phi1ddot_str = convertPowers('l1*phi2dot*m2*l2*s2*phi1dot-b1*phi1dot+1/2*m1*g*l1*c1+1/2*m2*g*l2*c12+1/2*m3*g*l3*c123-k1*phi1+k1*phi1ref+m2*g*l1*c1+m3*g*l1*c1+m3*g*l2*c12+m3*l3*phi3dot*l2*phi1dot*s3+m3*l3*phi3dot*phi1dot*l1*s23+m3*l3*phi3dot*l2*phi2dot*s3+1/2*m3*l3*phi3dot^2*l1*s23+1/2*m3*l3*phi3dot^2*l2*s3+1/2*l1*phi2dot^2*m2*l2*s2+2*l1*phi2dot*m3*l2*phi1dot*s2+l1*phi2dot^2*m3*l2*s2+l1*phi2dot*m3*phi1dot*l3*s23+1/2*l1*phi2dot^2*m3*l3*s23+l1*phi2dot*m3*l3*phi3dot*s23')
    phi2ddot_str = convertPowers('-b2*phi2dot-1/2*m2*l2*l1*s2*phi1dot^2-k2*phi2+k2*phi2ref+1/2*m2*g*l2*c12+1/2*m3*g*l3*c123+m3*g*l2*c12+m3*l3*phi3dot*l2*phi1dot*s3+m3*l3*phi3dot*l2*phi2dot*s3+1/2*m3*l3*phi3dot^2*l2*s3-m3*phi1dot^2*l1*l2*s2-1/2*m3*phi1dot^2*l1*l3*s23')
    phi3ddot_str = convertPowers('-b3*phi3dot-m3*l3*l2*phi1dot*phi2dot*s3-1/2*m3*l3*l2*phi2dot^2*s3+1/2*m3*g*l3*c123-k3*phi3+k3*phi3ref-1/2*m3*phi1dot^2*l1*l3*s23-1/2*m3*l3*phi1dot^2*l2*s3')

    auxdict = {}
    # var ordering will be phi1, phi1dot, phi2, phi2dot, phi3, phi3dot (alphabetical)
    # row 0
    a00 = "1";a01 = "0";a02 = "0";a03 = "0";a04 = "0";a05 = "0"
    # row 1
    a10 = "0"
    a11 = convertPowers('m2*l2*l1*c2+1/4*m2*l2^2+m2*l1^2+m3*l1^2+m3*l2^2+1/4*m3*l3^2+1/4*m1*l1^2+I_1+I_2+I_3+2*m3*l2*l1*c2+m3*l3*l1*c23+m3*l2*l3*c3')
    a12 = "0";a13 = 'mterm1';a14 = "0";a15 = 'mterm2'
    # row 2
    a20 = "0";a21 = "0";a22 = "1";a23 = "0";a24 = "0";a25 = "0"
    # row 3
    a30 = "0";a31 = 'mterm1';a32 = "0"
    a33 = convertPowers('1/4*m2*l2^2+1/4*m3*l3^2+m3*l2^2+m3*l2*l3*c3+I_2+I_3')
    a34 = "0";a35 = 'mterm3'
    # row 4
    a40 = "0";a41 = "0";a42 = "0";a43 = "0";a44 = "1";a45 = "0"
    # row 5
    a50 = "0";a51 = 'mterm2';a52 = "0";a53 = 'mterm3';a54 = "0"
    a55 = convertPowers('1/4*m3*l3^2+I_3')
    #
    M = "["
    env = locals()
    for r in range(6):
        M += "[" + ", ".join([eval("a"+str(r)+str(c), env) for c in range(6)]) +"]"
        if r < 5:
            M += ", "
    M += "]"
##    print M
    auxdict['massMatrix'] = (['t', 'phi1', 'phi2', 'phi3', 'phi1dot', 'phi2dot', 'phi3dot'], M)

    DSargs = {'nobuild': nobuild}
    DSargs['varspecs'] = {'phi1dot': phi1ddot_str, 'phi2dot': phi2ddot_str,
                             'phi3dot': phi3ddot_str, 'phi1': 'phi1dot',
                             'phi2': 'phi2dot', 'phi3': 'phi3dot'
                             }
    DSargs['pars'] = par_args
    DSargs['reuseterms'] = {'cos(phi1)': 'c1', 'cos(phi2)': 'c2', 'cos(phi3)': 'c3',
                            'cos(phi2+phi3)': 'c23',
                            'cos(phi1+phi2)': 'c12',
                            'cos(phi1+phi2+phi3)': 'c123',
                            'sin(phi2)': 's2',
                            'sin(phi3)': 's3',
                            'sin(phi2+phi3)': 's23',
                            convertPowers('1/4*m2*l2^2+1/4*m3*l3^2+m3*l2^2+1/2*m2*l2*l1*c2+m3*l2*l1*c2+m3*l2*l3*c3+1/2*m3*l3*l1*c23+I_2+I_3'): 'mterm1',
                            convertPowers('1/4*m3*l3^2+1/2*m3*l3*l1*c23+1/2*m3*l2*l3*c3+I_3'): 'mterm2',
                            convertPowers('1/4*m3*l3^2+1/2*m3*l2*l3*c3+I_3'): 'mterm3'}
    DSargs['vars'] = ['phi1', 'phi2', 'phi3', 'phi1dot', 'phi2dot', 'phi3dot']
    DSargs['fnspecs'] = auxdict
    DSargs['xdomain'] = {'phi1': [-.5, 1.], 'phi2': [0,2.], 'phi3': [0,1.]}
    DSargs['algparams'] = {'init_step': 0.001, 'refine': 0, 'max_step': 0.01,
                           'rtol': 1e-4, 'atol': 1e-4}
    DSargs['checklevel'] = 2
    DSargs['ics'] = ic_args
    DSargs['name'] = name
    if evs is not None:
        DSargs['events'] = evs
    return Generator.Radau_ODEsystem(DSargs)


# ------------------------------------------------------------


if __name__=='__main__':
    print('-------- Finger Test')
    # spatial units in mm, angular units in radians
    # Ix = 1/12 * mx * lx^2
    # radii were 10mm, 8mm, 7mm
    len1 = 27.0e-3 # m
    len2 = 34.4e-3
    len3 = 43.1e-3
    m1 = 54096*pi*len1*1e-6
    m2 = 70656*pi*len2*1e-6
    m3 = 110400*pi*len3*1e-6
    # k in Nm/rad, b in N m s / rad -- as reported by D. L. Jindrich et al.,
    # J. Biomech. 37 (2004) 1589-1596
    par_args = {'tau1': 0, 'tau2': 0, 'tau3': 0,  # these don't affect anything
                'k1': 0.12, 'k2': 0.28, 'k3': 0.54,
                'b1': 0.9e-3,
                'b2': 2.2e-3,
                'b3': 3.1e-3,
                'm1': m1,
                'm2': m2,
                'm3': m3,
                'l1': len1, 'l2': len2, 'l3': len3,
                'I_1': (m1*len1*len1)/12.,
                'I_2': (m2*len2*len2)/12.,
                'I_3': (m3*len3*len3)/12.,
                'eta': 0., 'F': 0., 'g': 9.81, # F, eta don't affect anything
                'phi1ref': 10*pi/180, 'phi2ref': 10*pi/180, 'phi3ref': 10*pi/180}
    ic_args = {'phi1': 0.01, 'phi2': 0.01, 'phi3': 0.5,
               'phi1dot': 0., 'phi2dot': 0., 'phi3dot': 0.,
               }

    def updateMass(m1,m2,m3,len1,len2,len3,pars):
        pars['I_1'] = (m1*len1*len1)/12.
        pars['I_2'] = (m2*len2*len2)/12.
        pars['I_3'] = (m3*len3*len3)/12.
        pars['m1'] = m1
        pars['m2'] = m2
        pars['m3'] = m3

    print("Making Radau finger model using mass matrix")
    finger = make_finger_massmatrix('freefinger_noforce_massmatrix', par_args, ic_args)
    finger.set(tdata=[0, 3])
#    saveObjects(finger, 'fingergen', force=True)

    print('Integrating...')
    start = perf_counter()
    ftraj = finger.compute('test')
    print('Computed trajectory in %.3f seconds.\n' % (perf_counter()-start))
    plotData = ftraj.sample(dt=.001)

    exportPointset(plotData, {'varvals': ['phi1','phi2','phi3'], 'tvals':'t'},
                   ext='dat')

    print('Preparing plot')

    yaxislabelstr = 'angles'
    plt.ylabel(yaxislabelstr)
    plt.xlabel('t')
    phi1_line=plot(plotData['t'], plotData['phi1'])
    phi2_line=plot(plotData['t'], plotData['phi2'])
    phi3_line=plot(plotData['t'], plotData['phi3'])
    show()
