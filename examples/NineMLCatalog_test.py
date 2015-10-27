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
    cc = ninemlcatalog.load('/neuron/HodgkinHuxley', 'HodgkinHuxley')
    comp = ninemlcatalog.load('neuron/HodgkinHuxley', 'SampleHodgkinHuxley')

    # Convert to PyDSTool.ModelSpec and create NonHybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    HHmodel = get_nineml_model(cc, 'HH_9ML', extra_args=[Par('isyn')])

    pars = dict((p.name, p.value) for p in comp.properties)
    pars['isyn'] = 20.0
    ics = dict((i.name, i.value) for i in comp.initial_values)
    HHmodel.set(pars=pars, ics=ics, tdata=[0, 15])

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
    cc = ninemlcatalog.load('/neuron/AdaptiveExpIntegrateAndFire',
                            'AdaptiveExpIntegrateAndFire')
    comp = ninemlcatalog.load('/neuron/AdaptiveExpIntegrateAndFire',
                              'SampleAdaptiveExpIntegrateAndFire')

    # Convert to PyDSTool.ModelSpec and create HybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    aeIF = get_nineml_model(cc, 'aeIF_9ML', extra_args=[Par('Isyn')],
                            max_t=100)

    pars = dict((p.name, p.value) for p in comp.properties)
    pars['Isyn'] = 5.0
    ics = dict((i.name, i.value) for i in comp.initial_values)
    ics['regime_'] = 0
    aeIF.set(pars=pars, ics=ics, tdata=[0, 30], algparams={'init_step': 0.04})

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
    cc = ninemlcatalog.load('/neuron/Izhikevich', 'Izhikevich')
    comp = ninemlcatalog.load('/neuron/Izhikevich', 'SampleIzhikevich')

    # Convert to PyDSTool.ModelSpec and create HybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    izh = get_nineml_model(cc, 'izh_9ML', extra_args=[Par('Isyn')],
                           max_t=100)

    pars = dict((p.name, p.value) for p in comp.properties)
    pars['Isyn'] = 20.0
    ics = dict((i.name, i.value) for i in comp.initial_values)
    ics['regime_'] = 0
    izh.set(pars=pars, ics=ics, tdata=[0, 80], algparams={'init_step': 0.04})

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


def test_Izh_FS(iSyns=None):
    """Izhikevich Fast Spiker model"""
    if iSyns is None:
        iSyns = [200]

    cc = ninemlcatalog.load('/neuron/Izhikevich', 'IzhikevichFastSpiking')
    comp = ninemlcatalog.load('/neuron/Izhikevich',
                              'SampleIzhikevichFastSpiking')

    # Convert to PyDSTool.ModelSpec and create HybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    izh = get_nineml_model(cc, 'izh_FS_9ML', extra_args=[Par('iSyn')],
                           max_t=100)
    pars = dict((p.name, p.value) for p in comp.properties)
    ics = dict((i.name, i.value) for i in comp.initial_values)
    ics['regime_'] = 0
    # set starting regime to be sub-threshold (PyDSTool will check consistency
    # with V initial condition)
    izh.set(pars=pars, ics=ics, tdata=[0, 80], algparams={'init_step': 0.03})

    for iSyn in iSyns:
        izh.set(pars={'iSyn': iSyn})
        name = 'iSyn=%.1f' % (float(iSyn))
        izh.compute(name, verboselevel=0)
        pts = izh.sample(name)
        evs = izh.getTrajEventTimes(name)['spikeOutput']
        ISIs = np.diff(evs)
        print("iSyn ={}:\n  Mean ISI = {}, variance = {}"
              .format(iSyn, np.mean(ISIs), np.var(ISIs)))

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
