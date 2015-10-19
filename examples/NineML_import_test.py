from __future__ import print_function

from PyDSTool import *  # @UnusedWildImport
from nineml import units as un
from PyDSTool.Toolbox.NineML import *  # @UnusedWildImport
from nineml import abstraction as al  # @Reimport
import nineml


def get_HH_component():
    """A Hodgkin-Huxley single neuron model.
    Written by Andrew Davison.
    See http://phobos.incf.ki.se/src_rst/
              examples/examples_al_python.html#example-hh
    """
    aliases = [
        "q10 := 3.0**((celsius - qfactor)/tendegrees)",  # temperature correction factor @IgnorePep8
        "alpha_m := alpha_m_A*(V-alpha_m_V0)/(exp(-(V-alpha_m_V0)/alpha_m_K) - 1.0)",  # @IgnorePep8
        "beta_m := beta_m_A*exp(-(V-beta_m_V0)/beta_m_K)",
        "mtau := 1.0/(q10*(alpha_m + beta_m))",
        "minf := alpha_m/(alpha_m + beta_m)",
        "alpha_h := alpha_h_A*exp(-(V-alpha_h_V0)/alpha_h_K)",
        "beta_h := beta_h_A/(exp(-(V-beta_h_V0)/beta_h_K) + 1.0)",
        "htau := 1.0/(q10*(alpha_h + beta_h))",
        "hinf := alpha_h/(alpha_h + beta_h)",
        "alpha_n := alpha_n_A*(V-alpha_n_V0)/(exp(-(V-alpha_n_V0)/alpha_n_K) - 1.0)",  # @IgnorePep8
        "beta_n := beta_n_A*exp(-(V-beta_n_V0)/beta_n_K)",
        "ntau := 1.0/(q10*(alpha_n + beta_n))",
        "ninf := alpha_n/(alpha_n + beta_n)",
        "gna := gnabar*m*m*m*h",
        "gk := gkbar*n*n*n*n",
        "ina := gna*(ena - V)",
        "ik := gk*(ek - V)",
        "il := gl*(el - V )"]

    hh_regime = al.Regime(
        "dn/dt = (ninf-n)/ntau",
        "dm/dt = (minf-m)/mtau",
        "dh/dt = (hinf-h)/htau",
        "dV/dt = (ina + ik + il + Isyn)/C",
        transitions=al.On("V > theta", do=al.SpikeOutputEvent())
    )

    state_variables = [
        al.StateVariable('V', un.voltage),
        al.StateVariable('m', un.dimensionless),
        al.StateVariable('n', un.dimensionless),
        al.StateVariable('h', un.dimensionless)]

    # the rest are not "parameters" but aliases, assigned vars, state vars,
    # indep vars, analog_analog_ports, etc.
    parameters = [
        al.Parameter('el', un.voltage),
        al.Parameter('C', un.capacitance),
        al.Parameter('ek', un.voltage),
        al.Parameter('ena', un.voltage),
        al.Parameter('gkbar', un.conductance),
        al.Parameter('gnabar', un.conductance),
        al.Parameter('theta', un.voltage),
        al.Parameter('gl', un.conductance),
        al.Parameter('celsius', un.temperature),
        al.Parameter('qfactor', un.temperature),
        al.Parameter('tendegrees', un.temperature),
        al.Parameter('alpha_m_A', un.dimensionless / (un.time * un.voltage)),
        al.Parameter('alpha_m_V0', un.voltage),
        al.Parameter('alpha_m_K', un.voltage),
        al.Parameter('beta_m_A', un.dimensionless / un.time),
        al.Parameter('beta_m_V0', un.voltage),
        al.Parameter('beta_m_K', un.voltage),
        al.Parameter('alpha_h_A', un.dimensionless / un.time),
        al.Parameter('alpha_h_V0', un.voltage),
        al.Parameter('alpha_h_K', un.voltage),
        al.Parameter('beta_h_A', un.dimensionless / un.time),
        al.Parameter('beta_h_V0', un.voltage),
        al.Parameter('beta_h_K', un.voltage),
        al.Parameter('alpha_n_A', un.dimensionless / (un.time * un.voltage)),
        al.Parameter('alpha_n_V0', un.voltage),
        al.Parameter('alpha_n_K', un.voltage),
        al.Parameter('beta_n_A', un.dimensionless / un.time),
        al.Parameter('beta_n_V0', un.voltage),
        al.Parameter('beta_n_K', un.voltage)]

    analog_ports = [al.AnalogSendPort("V", un.voltage),
                    al.AnalogReducePort("Isyn", un.current, operator="+")]

    c1 = al.DynamicsClass("HodgkinHuxley",
                          parameters=parameters,
                          state_variables=state_variables,
                          regimes=(hh_regime,),
                          aliases=aliases,
                          analog_ports=analog_ports)
    return c1


def get_Izh_component():
    subthreshold_regime = al.Regime(
        name="subthreshold_regime",
        time_derivatives=[
            "dV/dt = alpha*V*V + beta*V + zeta - U + Isyn / C_m",
            "dU/dt = a*(b*V - U)", ],

        transitions=[al.On("V > theta",
                           do=["V = c",
                               "U =  U+ d",
                               al.OutputEvent('spike')],
                           to='subthreshold_regime')]
    )

    ports = [al.AnalogSendPort("V", un.voltage),
             al.AnalogReducePort("Isyn", un.current, operator="+")]

    parameters = [
        al.Parameter('theta', un.voltage),
        al.Parameter('a', un.per_time),
        al.Parameter('b', un.per_time),
        al.Parameter('c', un.voltage),
        al.Parameter('d', un.voltage / un.time),
        al.Parameter('C_m', un.capacitance),
        al.Parameter('alpha', un.dimensionless / (un.voltage * un.time)),
        al.Parameter('beta', un.per_time),
        al.Parameter('zeta', un.voltage / un.time)]

    state_variables = [
        al.StateVariable('V', un.voltage),
        al.StateVariable('U', un.voltage / un.time)]

    c1 = al.DynamicsClass(
        name="Izhikevich",
        parameters=parameters,
        state_variables=state_variables,
        regimes=[subthreshold_regime],
        analog_ports=ports

    )
    return c1


def get_Izh_FS_component():
    """
    Load Fast spiking Izhikevich XML definition from file and parse into
    Abstraction Layer of Python API.
    """
    izhi_fs = al.DynamicsClass(
        name='IzhikevichFS',
        parameters=[
            al.Parameter('a', un.per_time),
            al.Parameter('b', un.conductance / (un.voltage ** 2)),
            al.Parameter('c', un.voltage),
            al.Parameter('k', un.conductance / un.voltage),
            al.Parameter('Vr', un.voltage),
            al.Parameter('Vt', un.voltage),
            al.Parameter('Vb', un.voltage),
            al.Parameter('Vpeak', un.voltage),
            al.Parameter('Cm', un.capacitance)],
        analog_ports=[
            al.AnalogReducePort('iSyn', un.current, operator="+"),
            al.AnalogReducePort('iExt', un.current, operator="+"),
            al.AnalogSendPort('U', un.current),
            al.AnalogSendPort('V', un.voltage)],
        event_ports=[
            al.EventSendPort("spikeOutput")],
        state_variables=[
            al.StateVariable('V', un.voltage),
            al.StateVariable('U', un.current)],
        regimes=[
            al.Regime(
                'dU/dt = a * (b * pow(V - Vb, 3) - U)',
                'dV/dt = V_deriv',
                transitions=[
                    al.On('V > Vpeak',
                          do=['V = c', al.OutputEvent('spikeOutput')],
                          to='subthreshold')],
                name="subthreshold"),
            al.Regime(
                'dU/dt = - U * a',
                'dV/dt = V_deriv',
                transitions=[al.On('V > Vb', to="subthreshold")],
                name="subVb")],
        aliases=["V_deriv := (k * (V - Vr) * (V - Vt) - U + iExt + iSyn) / Cm"])  # @IgnorePep8
    return izhi_fs


def get_aeIF_component():
    """
    Adaptive exponential integrate-and-fire neuron as described in
    A. Destexhe, J COmput Neurosci 27: 493--506 (2009)

    Author B. Kriener (Jan 2011)

    ## neuron model: aeIF

    ## variables:
    ## V: membrane potential
    ## w: adaptation variable

    ## parameters:
    ## C_m     # specific membrane capacitance [muF/cm**2]
    ## g_L     # leak conductance [mS/cm**2]
    ## E_L     # resting potential [mV]
    ## Delta   # steepness of exponential approach to threshold [mV]
    ## V_T     # spike threshold [mV]
    ## S       # membrane area [mum**2]
    ## trefractory # refractory time [ms]
    ## tspike  # spike time [ms]
    ## tau_w   # adaptation time constant
    ## a, b    # adaptation parameters [muS, nA]
    """
    aeIF = al.DynamicsClass(
        name="aeIF",
        parameters=[
            al.Parameter('C_m', un.capacitance),
            al.Parameter('g_L', un.conductance),
            al.Parameter('E_L', un.voltage),
            al.Parameter('Delta', un.voltage),
            al.Parameter('V_T', un.voltage),
            al.Parameter('S'),
            al.Parameter('trefractory', un.time),
            al.Parameter('tspike', un.time),
            al.Parameter('tau_w', un.time),
            al.Parameter('a', un.dimensionless / un.voltage),
            al.Parameter('b')],
        state_variables=[
            al.StateVariable('V', un.voltage),
            al.StateVariable('w')],
        regimes=[
            al.Regime(
                name="subthresholdregime",
                time_derivatives=[
                    "dV/dt = -g_L*(V-E_L)/C_m + Isyn/C_m + g_L*Delta*exp((V-V_T)/Delta-w/S)/C_m",  # @IgnorePep8
                    "dw/dt = (a*(V-E_L)-w)/tau_w", ],
                transitions=al.On("V > V_T",
                                  do=["V = E_L", "w = w + b",
                                      al.OutputEvent('spikeoutput')],
                                  to="refractoryregime")),
            al.Regime(
                name="refractoryregime",
                transitions=al.On("t>=tspike+trefractory",
                                  to="subthresholdregime"))],
        analog_ports=[al.AnalogReducePort("Isyn", un.current, operator="+")])
    return aeIF


def get_compound_component():
    """Cannot yet be implemented in PyDSTool
    """
    from nineml.abstraction.testing_utils import RecordValue
    from nineml.abstraction import DynamicsClass, Regime, On, OutputEvent, AnalogSendPort, AnalogReducePort

    emitter = DynamicsClass(
            name='EventEmitter',
            parameters=['cyclelength'],
            regimes=[
                Regime(
                    transitions=On(
                        't > tchange + cyclelength', do=[OutputEvent('emit'), 'tchange=t'])),
            ])

    ev_based_cc = DynamicsClass(
            name='EventBasedCurrentClass',
            parameters=['dur', 'i'],
            analog_ports=[AnalogSendPort('I')],
            regimes=[
                Regime(
                    transitions=[
                        On('inputevent', do=['I=i', 'tchange = t']),
                        On('t>tchange + dur', do=['I=0', 'tchange=t'])
                    ]
                )
            ]
        )

    pulsing_emitter = DynamicsClass(name='pulsing_cc',
                                         subnodes={'evs': emitter, 'cc': ev_based_cc},
                                         portconnections=[('evs.emit', 'cc.inputevent')]
                                         )

    nrn = DynamicsClass(
            name='LeakyNeuron',
            parameters=['Cm', 'gL', 'E'],
            regimes=[Regime('dV/dt = (iInj + (E-V)*gL )/Cm'), ],
            aliases=['iIn := iInj'],
            analog_ports=[AnalogSendPort('V'),
                          AnalogReducePort('iInj', operator='+')],
        )

    combined_comp = DynamicsClass(name='Comp1',
                                       subnodes={
                                       'nrn': nrn,  'cc1': pulsing_emitter, 'cc2': pulsing_emitter},
                                       portconnections=[('cc1.cc.I', 'nrn.iInj'),
                                                        ('cc2.cc.I', 'nrn.iInj')]
                                       )

    combined_comp = al.flattening.flatten(combined_comp)

##        records = [
##            RecordValue(what='cc1_cc_I', tag='Current', label='Current Clamp 1'),
##            RecordValue(what='cc2_cc_I', tag='Current', label='Current Clamp 2'),
##            RecordValue(what='nrn_iIn', tag='Current', label='Total Input Current'),
##            RecordValue(what='nrn_V', tag='Voltage', label='Neuron Voltage'),
##            RecordValue(what='cc1_cc_tchange', tag='Tchange', label='tChange CC1'),
##            RecordValue(what='cc2_cc_tchange', tag='Tchange', label='tChange CC2'),
##            RecordValue(what='regime',     tag='Regime',  label='Regime'),
##        ]

    parameters = al.flattening.ComponentFlattener.flatten_namespace_dict({
        'cc1.cc.i': 13.8,
        'cc1.cc.dur': 10,
        'cc1.evs.cyclelength': 30,
        'cc2.cc.i': 20.8,
        'cc2.cc.dur': 5.0,
        'cc2.evs.cyclelength': 20,
        'nrn.gL': 4.3,
        'nrn.E': -70})

    return combined_comp, parameters


# -------------------------------------------------------------------------------

def test_HH():
    c = get_HH_component()

    # Convert to PyDSTool.ModelSpec and create NonHybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    HHmodel = get_nineml_model(c, 'HH_9ML', extra_args=[Par('Isyn')])

    HHmodel.set(pars={'C': 1.0,
                      'Isyn': 20.0,
                      'celsius': 20.0,
                      'ek': -90,
                      'el': -65,
                      'ena': 80,
                      'gkbar': 30.0,
                      'gl': 0.3,
                      'gnabar': 130.0,
                      'theta': -40.0,
                      'qfactor': 6.3,
                      'tendegrees': 10.0,
                      'alpha_m_A': -0.1,
                      'alpha_m_V0': -40.0,
                      'alpha_m_K': 10.0,
                      'beta_m_A': 4.0,
                      'beta_m_V0': -65.0,
                      'beta_m_K': 18.0,
                      'alpha_h_A': 0.07,
                      'alpha_h_V0': -65.0,
                      'alpha_h_K': 20.0,
                      'beta_h_A': 1.0,
                      'beta_h_V0': -35.0,
                      'beta_h_K': 10.0,
                      'alpha_n_A': -0.01,
                      'alpha_n_V0': -55.0,
                      'alpha_n_K': 10.0,
                      'beta_n_A': 0.125,
                      'beta_n_V0': -65.0,
                      'beta_n_K': 80.0},
                ics={'V': -70, 'm': 0.1, 'n': 0, 'h': 0.9},
                tdata=[0,15])

    HHmodel.compute('HHtest', force=True)
    pts = HHmodel.sample('HHtest')
    plt.figure(1)
    plt.plot(pts['t'], pts['V'],'k')
    plt.title('Hodgkin-Huxley membrane potential')

    ev_info = pts.labels.by_label['Event:spikeoutput']
    for ev_ix, ev_tdata in list(ev_info.items()):
        plt.plot(ev_tdata['t'], pts[ev_ix]['V'], 'ko')

    plt.xlabel('t')
    plt.ylabel('V')


def test_aeIF():
    """Adaptive Integrate and Fire"""
    c = get_aeIF_component()

    # Convert to PyDSTool.ModelSpec and create HybridModel object
    # Provide extra parameter Isyn which is missing from component definition
    # in absence of any synaptic inputs coupled to the model membrane
    aeIF = get_nineml_model(c, 'aeIF_9ML', extra_args=[Par('Isyn')],
                            max_t=100)

    aeIF.set(pars=dict(
        C_m = 1,
        g_L = 0.1,
        E_L = -65,
        Delta = 1,
        V_T = -58,
        S = 0.1,
        tspike = 0.5,
        trefractory = 0.25,
        tau_w = 4,
        a = 1,
        b = 2,
        Isyn = 5
        ))

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
    c = get_Izh_component()

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
    plt.plot(evs, [theta]*len(evs), 'ko')
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
    c = get_Izh_FS_component()

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
        print("Iext =", Iext, ":")
        print("  Mean ISI = %.3f, variance = %.6f" % (np.mean(ISIs), np.var(ISIs)))

        Vp = izh.query('pars')['Vpeak']
        plt.figure(6)
        plt.plot(pts['t'], pts['V'], label=name)
        plt.plot(evs, [Vp]*len(evs), 'ko')
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


def test_compound():
    """Not yet implemented"""
    raise NotImplementedError("Come back soon!")

    c, pardict = get_compound_component()
    cm = get_nineml_model(c, 'comb_9ML',
                            max_t=100)

    cm.set(pars=pardict)
    cm.set(ics={'V': -70, 'regime_': 0},
             tdata=[0, 30],
             algparams={'init_step': 0.04})

    cm.compute('test', verboselevel=0)

    pts = xm.sample('test')

    evs = cm.getTrajEventTimes('test')

    plt.figure(8)
    plt.plot(pts['t'], pts['V'], 'k')
    plt.title('Combined passive response model')
    plt.xlabel('t')
    plt.ylabel('V')

# ==========


print("Testing Hodgkin Huxley cell model")
test_HH()

print("Testing adaptive Integrate and Fire cell model")
test_aeIF()

#print("Testing compound cell model")
#test_compound()

print("Testing basic Izhikevich cell model")
test_Izh()

print("Testing Izhikevich fast spiking cell model from XML import")
print("   at three input current levels")
test_Izh_FS([100,200,400])

plt.show()


