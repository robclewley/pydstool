from __future__ import print_function

from PyDSTool import *
import nineml.abstraction_layer as al
from PyDSTool.Toolbox.NineML import *
import nineml


def get_HH_component():
    """A Hodgkin-Huxley single neuron model.
    Written by Andrew Davison.
    See http://phobos.incf.ki.se/src_rst/examples/examples_al_python.html#example-hh
    """
    aliases = [
        "q10 := 3.0**((celsius - 6.3)/10.0)",  # temperature correction factor
        "alpha_m := -0.1*(V+40.0)/(exp(-(V+40.0)/10.0) - 1.0)",  # m
        "beta_m := 4.0*exp(-(V+65.0)/18.0)",
        "mtau := 1/(q10*(alpha_m + beta_m))",
        "minf := alpha_m/(alpha_m + beta_m)",
        "alpha_h := 0.07*exp(-(V+65.0)/20.0)",               # h
        "beta_h := 1.0/(exp(-(V+35)/10.0) + 1.0)",
        "htau := 1.0/(q10*(alpha_h + beta_h))",
        "hinf := alpha_h/(alpha_h + beta_h)",
        "alpha_n := -0.01*(V+55.0)/(exp(-(V+55.0)/10.0) - 1.0)", # n
        "beta_n := 0.125*exp(-(V+65.0)/80.0)",
        "ntau := 1.0/(q10*(alpha_n + beta_n))",
        "ninf := alpha_n/(alpha_n + beta_n)",
        "gna := gnabar*m*m*m*h",                       #
        "gk := gkbar*n*n*n*n",
        "ina := gna*(ena - V)",                 # currents
        "ik := gk*(ek - V)",
        "il := gl*(el - V )"]

    hh_regime = al.Regime(
        "dn/dt = (ninf-n)/ntau",
        "dm/dt = (minf-m)/mtau",
        "dh/dt = (hinf-h)/htau",
        "dV/dt = (ina + ik + il + Isyn)/C",
        transitions=al.On("V > theta",do=al.SpikeOutputEvent() )
    )

    # the rest are not "parameters" but aliases, assigned vars, state vars,
    # indep vars, analog_analog_ports, etc.
    parameters = ['el', 'C', 'ek', 'ena', 'gkbar', 'gnabar', 'theta', 'gl', 'celsius']

    analog_ports = [al.AnalogSendPort("V"), al.AnalogReducePort("Isyn",reduce_op="+")]

    c1 = al.DynamicsClass("HodgkinHuxley",
                          parameters=parameters,
                          regimes=(hh_regime,),
                          aliases=aliases,
                          analog_ports=analog_ports)
    return c1

def get_Izh_component():
    subthreshold_regime = al.Regime(
        name="subthreshold_regime",
        time_derivatives=[
            "dV/dt = 0.04*V*V + 5*V + 140.0 - U + Isyn",
            "dU/dt = a*(b*V - U)", ],

        transitions=[al.On("V > theta",
                           do=["V = c",
                                   "U =  U+ d",
                                   al.OutputEvent('spike'), ],
                           to='subthreshold_regime')]
    )

    ports = [al.AnalogSendPort("V"),
             al.AnalogReducePort("Isyn", reduce_op="+")]

    c1 = al.DynamicsClass(
        name="Izhikevich",
        regimes=[subthreshold_regime],
        analog_ports=ports

    )
    return c1

def get_Izh_FS_component():
    """
    Load Fast spiking Izhikevich XML definition from file and parse into
    Abstraction Layer of Python API.
    """
    return nineml.read('NineML_Izh_FS.xml')['IzhikevichClass']

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
    parameters = ['C_m', 'g_L', 'E_L', 'Delta', 'V_T', 'S',
                  'trefractory', 'tspike', 'tau_w', 'a', 'b']

    aeIF = al.DynamicsClass("aeIF",
                     regimes=[
                         al.Regime(
                                name="subthresholdregime",
                                time_derivatives = [
                                    "dV/dt = -g_L*(V-E_L)/C_m + Isyn/C_m + g_L*Delta*exp((V-V_T)/Delta-w/S)/C_m",
                                    "dw/dt = (a*(V-E_L)-w)/tau_w", ],
                                transitions=al.On("V > V_T",
                                               do=["V = E_L",
                                                   "w = w + b",
                                                   al.OutputEvent('spikeoutput')],
                                               to="refractoryregime"),
                                ),

                         al.Regime(
                                name="refractoryregime",
                                transitions=al.On("t>=tspike+trefractory",
                                               to="subthresholdregime"),
                                )
                               ],
                         analog_ports=[al.AnalogReducePort("Isyn", reduce_op="+")]
                     )

    return aeIF


def get_compound_component():
    """Cannot yet be implemented in PyDSTool
    """
    from nineml.abstraction_layer.testing_utils import RecordValue
    from nineml.abstraction_layer import DynamicsClass, Regime, On, OutputEvent, AnalogSendPort, AnalogReducePort

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
                          AnalogReducePort('iInj', reduce_op='+')],
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
                      'theta': -40.0},
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
                      Isyn=20))
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
    # Provide extra parameters iSyn and iExt which are missing from
    # component definition in absence of any synaptic inputs coupled
    # to the model membrane
    izh = get_nineml_model(c, 'izh_9ML', extra_args=[Par('iExt'), Par('iSyn')],
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
        name = 'iExt=%.1f'%(float(Iext))
        izh.compute(name, verboselevel=0)
        pts = izh.sample(name)
        evs = izh.getTrajEventTimes(name)['spikeOutput']
        ISIs = np.diff(evs)
        print("iExt =", Iext, ":")
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
#test_HH()

print("Testing adaptive Integrate and Fire cell model")
#test_aeIF()

#print("Testing compound cell model")
#test_compound()

print("Testing basic Izhikevich cell model")
#test_Izh()

fs = nineml.read('NineML_Izh_FS.xml')

print("Testing Izhikevich fast spiking cell model from XML import")
print("   at three input current levels")
test_Izh_FS([100,200,400])

plt.show()


