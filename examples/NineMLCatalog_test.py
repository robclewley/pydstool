"""
A module for loading and testing basic neuron models as described in the NineML
Catalog (see http://github.com/INCF/NineMLCatalog)
"""
import ninemlcatalog
from PyDSTool import Par
from PyDSTool.Toolbox.NineML import get_nineml_model
import numpy as np
from matplotlib import pyplot as plt


def test_HH():
    c = ninemlcatalog.lookup('/neuron/HodgkinHuxley/HodgkinHuxley')

    # Convert to PyDSTool.ModelSpec and create NonHybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    HHmodel = get_nineml_model(c, 'HH_9ML', extra_args=[Par('iSyn')])

    HHmodel.set(pars={'C': 1.0,
                      'iSyn': 20.0,
                      'celsius': 20.0,
                      'ek': -90,
                      'el': -65,
                      'ena': 80,
                      'gkbar': 30.0,
                      'gl': 0.3,
                      'gnabar': 130.0,
                      'v_threshold': -40.0,
                      'qfactor': 6.3,
                      'tendegrees': 10.0,
                      'm_alpha_A': -0.1,
                      'm_alpha_V0': -40.0,
                      'm_alpha_K': 10.0,
                      'm_beta_A': 4.0,
                      'm_beta_V0': -65.0,
                      'm_beta_K': 18.0,
                      'h_alpha_A': 0.07,
                      'h_alpha_V0': -65.0,
                      'h_alpha_K': 20.0,
                      'h_beta_A': 1.0,
                      'h_beta_V0': -35.0,
                      'h_beta_K': 10.0,
                      'n_alpha_A': -0.01,
                      'n_alpha_V0': -55.0,
                      'n_alpha_K': 10.0,
                      'n_beta_A': 0.125,
                      'n_beta_V0': -65.0,
                      'n_beta_K': 80.0},
                ics={'V': -70, 'm': 0.1, 'n': 0, 'h': 0.9},
                tdata=[0, 15])

    HHmodel.compute('HHtest', force=True)
    pts = HHmodel.sample('HHtest')
    plt.figure(1)
    plt.plot(pts['t'], pts['V'], 'k')
    plt.title('Hodgkin-Huxley membrane potential')

    ev_info = pts.labels.by_label['Event:spikeoutput']
    for ev_ix, ev_tdata in list(ev_info.items()):
        plt.plot(ev_tdata['t'], pts[ev_ix]['V'], 'ko')

    plt.xlabel('t')
    plt.ylabel('V')


def test_aeIF():
    """Adaptive Integrate and Fire"""
    c = ninemlcatalog.lookup('/neuron/AdaptiveExpIntegrateAndFire/'
                             'AdaptiveExpIntegrateAndFire')

    # Convert to PyDSTool.ModelSpec and create HybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    aeIF = get_nineml_model(c, 'aeIF_9ML', extra_args=[Par('Isyn')],
                            max_t=100)

    aeIF.set(pars=dict(
        C_m=1,
        g_L=0.1,
        E_L=-65,
        Delta=1,
        V_T=-58,
        S=0.1,
        tspike=0.5,
        trefractory=0.25,
        tau_w=4,
        a=1,
        b=2,
        Isyn=5))

    aeIF.set(ics={'V': -70, 'w': 0.1, 'regime_': 0},
             tdata=[0, 30],
             algparams={'init_step': 0.04})

    aeIF.compute('test', verboselevel=0)

    pts = aeIF.sample('test', dt=0.1)

    plt.figure(2)
    plt.plot(pts['t'], pts['V'])
    plt.xlabel('t')
    plt.ylabel('V')
    plt.title('adaptive IF model')
    plt.figure(3)
    plt.plot(pts['t'], pts['w'])
    plt.xlabel('t')
    plt.ylabel('w')
    plt.title('adaptive IF model')


def test_Izh():
    """Basic Izhikevich hybrid model"""
    c = ninemlcatalog.lookup('/neuron/Izhikevich/Izhikevich')

    # Convert to PyDSTool.ModelSpec and create HybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    izh = get_nineml_model(c, 'izh_9ML', extra_args=[Par('Isyn')],
                            max_t=100)

    izh.set(pars=dict(a=0.2, b=0.025, c=-75, d=0.2, theta=-50,
                      Isyn=20, alpha=0.04, beta=5, zeta=140.0, C_m=1.0))
    izh.set(ics={'V': -70, 'U': -1.625, 'regime_': 0},
             tdata=[0, 80],
             algparams={'init_step': 0.04})

    izh.compute('test', verboselevel=0)

    pts = izh.sample('test')

    evs = izh.getTrajEventTimes('test')['spike']

    theta = izh.query('pars')['theta']
    plt.figure(4)
    plt.plot(pts['t'], pts['V'], 'k')
    plt.plot(evs, [theta] * len(evs), 'ko')
    plt.title('Izhikevich model')
    plt.xlabel('t')
    plt.ylabel('V')

    plt.figure(5)
    plt.plot(pts['t'], pts['U'])
    plt.xlabel('t')
    plt.ylabel('U')
    plt.title('Izhikevich model')

# ========


def test_Izh_FS(Iexts=None):
    """Izhikevich Fast Spiker model"""
    c = ninemlcatalog.lookup('/neuron/Izhikevich/IzhikevichFastSpiking')

    # Convert to PyDSTool.ModelSpec and create HybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    izh = get_nineml_model(c, 'izh_9ML', extra_args=[Par('iSyn'), Par('iExt')],
                            max_t=100)

    if Iexts is None:
        Iexts = [200]

    izh.set(pars=dict(a=0.2, b=0.025, c=-45, k=1, Vpeak=25,
                      Vb=-55, Cm=20, Vr=-55, Vt=-40))
    # set starting regime to be sub-threshold (PyDSTool will check consistency
    # with V initial condition)
    izh.set(ics={'V': -65, 'U': -1.625, 'regime_': 0},
             tdata=[0, 80],
             algparams={'init_step': 0.03})

    for Iext in Iexts:
        izh.set(pars={'iExt': Iext})
        name = 'Iext=%.1f' % (float(Iext))
        izh.compute(name, verboselevel=0)
        pts = izh.sample(name)
        evs = izh.getTrajEventTimes(name)['spikeOutput']
        ISIs = np.diff(evs)
        print("Iext ={}:\n  Mean ISI = {}, variance = {}"
              .format(Iext, np.mean(ISIs), np.var(ISIs)))

        Vp = izh.query('pars')['Vpeak']
        plt.figure(6)
        plt.plot(pts['t'], pts['V'], label=name)
        plt.plot(evs, [Vp] * len(evs), 'ko')
        plt.title('Izhikevich fast spiking model')
        plt.xlabel('t')
        plt.ylabel('V')
        plt.legend()

        plt.figure(7)
        plt.plot(pts['t'], pts['U'], label=name)
        plt.xlabel('t')
        plt.ylabel('U')
        plt.legend()
    plt.title('Izhikevich FS model')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('test', type=str, nargs='+',
                        help=("The name of the model to test. Can be either "
                              "'hh', 'izhi', 'izhiFS' or 'aeif'."))
    parser.add_argument('--injected_current', type=float, default=None,
                        help=("The amount of current injected into the "
                              "model(s) in nA."))
    args = parser.parse_args()
    for test in args.test:
        if test == 'izhi':
            test_Izh()
        elif test == 'izhiFS':
            if args.injected_current is None:
                injected_current = [100, 200, 400]
            else:
                injected_current = [args.injected_current]
            test_Izh_FS(injected_current)
        elif test == 'aeif':
            test_aeIF()
        elif test == 'hh':
            test_HH()
        else:
            raise Exception(
                "Unrecognised test option '{}', can be either 'hh', 'izhi', "
                "'izhiFS' or 'aeif'.")
    plt.show()
