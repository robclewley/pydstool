"""A demonstration of a Hodgkin-Huxley model with phase plane tools.

   Robert Clewley, April 2009.
"""

from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
import time
from copy import copy
global plotter

# ------------------------------------------------------------


def makeHHneuron(name, dt, par_args, ic_args, evs=None, extra_terms='',
                 with_flow_event=True):
    # extra_terms must not introduce new variables!
    vfn_str = '(I'+extra_terms+'-ionic(v,m,h,n))/C'
    mfn_str = 'ma(v)*(1-m)-mb(v)*m'
    nfn_str = 'na(v)*(1-n)-nb(v)*n'
    hfn_str = 'ha(v)*(1-h)-hb(v)*h'
    aux_str = 'm*m*m*h'

    # Testing: ptest in RHS function to test Jacobian computation of ionic with embedded aux function
    # ptest has no numeric effect on the ionic function otherwise
    auxdict = {'ionic': (['vv', 'mm', 'hh', 'nn'],
                              'gna*mm*mm*mm*hh*ptest(0)*(vv-vna) + gk*nn*nn*nn*nn*(vv-vk) + gl*(vv-vl)'),
               'ma': (['v'], '0.32*(v+54)/(1-exp(-(v+54)/4.0))'),
               'mb': (['v'], '0.28*(v+27)/(exp((v+27)/5.0)-1)'),
               'ha': (['v'], '.128*exp(-(50+v)/18.0)'),
               'hb': (['v'], '4/(1+exp(-(v+27)/5.0))'),
               'na': (['v'], '.032*(v+52)/(1-exp(-(v+52)/5.0))'),
               'nb': (['v'], '.5*exp(-(57+v)/40.0)'),
               'ptest': (['p'], '-C+(1+p+C)*1')} # use model parameter in this function to test jacobian creation below

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'm': mfn_str,
                       'h': hfn_str, 'n': nfn_str,
                       'v_bd0': 'getbound("v",0)',
                       'v_bd1': 'getbound("v",1)'}
    DSargs.pars = par_args
    DSargs.auxvars = ['v_bd0','v_bd1']
    DSargs.fnspecs = auxdict
    DSargs.xdomain = {'v': [-130, 70], 'm': [0,1], 'h': [0,1], 'n': [0,1]}
    DSargs.algparams = {'init_step':dt, 'max_step': dt*1.5,
                        'max_pts': 300000, 'refine': 1}

    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = name

    if with_flow_event:
        ev, ev_helper=make_flow_normal_event('v', 'm', vfn_str, mfn_str,
                          targetlang='c',
                          fnspec={'vars': ['v', 'm', 'h', 'n'],
                                  'pars': list(par_args.keys()),
                                  'inputs': [],
                                  'auxfns': auxdict},
                          evtArgs={'eventtol': 1e-8,
                                   'eventdelay': 1e-7,
                                   'starttime': 0,
                                   'term': False})
        for p in ev_helper['pars_to_vars']:
            if p in DSargs:
                raise ValueError("Parameter name %s already in system specification"%p)
            DSargs.pars[p] = 0
        if evs is None:
            evs = [ev]
        else:
            evs.append(ev)
    else:
        ev_helper = None

    if evs is not None:
        DSargs.events = evs
    return (Generator.Vode_ODEsystem(DSargs), ev_helper)


# ------------------------------------------------------------



print('Phase Plane test using Hodgkin-Huxley system during action potential')
par_args = {'gna': 100, 'gk': 80, 'gl': 0.1,
            'vna': 50, 'vk': -100, 'vl': -67,
            'I': 1.75, 'C': 1.0}

# very close to a periodic orbit
ic_args = {'v_bd0': -130.0, 'v_bd1': 70.0, 'h': 0.99599864212873856,
           'm': 0.00050362509755992027, 'n': 0.00557358064026849,
           'v': -82.018860813828837}

HH, ev_helper = makeHHneuron('HH_PP_test', 0.1, par_args, ic_args)
HH.set(tdata=[0, 100], ics={'h':0.7, 'n': 0.2})

print("Finding analytic Jacobian w.r.t. phase-plane variables v, m...")
jac, new_fnspecs = prepJacobian(HH.funcspec._initargs['varspecs'], ['m', 'v'],
                                HH.funcspec._initargs['fnspecs'])

scope = copy(HH.pars)
scope.update({'n': HH.initialconditions['n'], 'h': HH.initialconditions['h']})
scope.update(new_fnspecs)
jac_fn = expr2fun(jac, ensure_args=['t'], **scope)

print("Use of Jacobian speeds up finding of nullclines and fixed points by")
print("nearly a factor of two (not including time to plot results)...")
fp_coords = find_fixedpoints(HH, n=4, jac=jac_fn, eps=1e-8,
                             subdomain={'v':HH.xdomain['v'],'m':HH.xdomain['m'],
                             'h': HH.initialconditions['h'], 'n': HH.initialconditions['n']})
nulls_x, nulls_y = find_nullclines(HH, 'v', 'm', n=3, jac=jac_fn, eps=1e-8, max_step=0.5,
                             subdomain={'v':HH.xdomain['v'],'m':HH.xdomain['m'],
                             'h': HH.initialconditions['h'], 'n': HH.initialconditions['n']},
                             fps=fp_coords)

N_x = nullcline('v', 'm', nulls_x)
N_y = nullcline('v', 'm', nulls_y)


print("Fixed points for (v,m) phase plane sub-system when h=%.2f and n=%.2f: " % (HH.initialconditions['h'], HH.initialconditions['n']))
print("For classification and stability, we use the fixedpoint class...")

fps=[]
fps.append(fixedpoint_2D(HH, Point(fp_coords[0]), coords=['v', 'm'],
                         jac=jac_fn, description='bottom', eps=1e-8))
fps.append(fixedpoint_2D(HH, Point(fp_coords[1]), coords=['v', 'm'],
                         jac=jac_fn, description='middle', eps=1e-8))
fps.append(fixedpoint_2D(HH, Point(fp_coords[2]), coords=['v', 'm'],
                         jac=jac_fn, description='top', eps=1e-8))

for fp in fps:
    print("F.p. at (%.5f, %.5f) is a %s and has stability %s" % (fp.point['v'],
                            fp.point['m'], fp.classification, fp.stability))

plotter.fig_directory = {'PP_large': 1,
                         'PP_small': 2}
plotter.do_display=True
plotter.set_curr_fig('PP_large')
plotter.plot_nullcline(N_x, 'g')
plotter.plot_nullcline(N_y, 'r')
plotter.plot_fps(fps, 'v', 'm')
# scale_exp needs to be adjusted to get arrow sizes right for scale of vector field
plotter.plot_vf(HH, 'v', 'm', subdomain={'v': [-100, 50]}, scale_exp=1, N=50)
plt.xlabel('V')
plt.ylabel('m')
plt.title('Phase plane')

plotter.set_curr_fig('PP_small')
plotter.xdom = [-80, -50]
plotter.plot_nullcline(N_x, 'g')
plotter.plot_nullcline(N_y, 'r', N=200)
plotter.plot_fps(fps, 'v', 'm')
plotter.plot_vf(HH, 'v', 'm', subdomain={'v': [-80, -50], 'm': [0,0.14]}, scale_exp=0.2, N=50)
plt.xlabel('V')
plt.ylabel('m')
plt.title('Phase plane: zoom')
plt.show()
