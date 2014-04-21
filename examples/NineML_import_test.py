from PyDSTool import *
import nineml.abstraction_layer as al
from PyDSTool.Toolbox.NineML import *

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

    analog_ports = [al.SendPort("V"), al.ReducePort("Isyn",reduce_op="+")]

    c1 = al.ComponentClass("HodgkinHuxley",
                          parameters=parameters,
                          regimes=(hh_regime,),
                          aliases=aliases,
                          analog_ports=analog_ports)
    return c1



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

    aeIF = al.ComponentClass("aeIF",
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
                         analog_ports=[al.ReducePort("Isyn", reduce_op="+")]
                     )

    return aeIF


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

    HHmodel.compute('test', force=True)
    pts = HHmodel.sample('test')
    plt.plot(pts['t'], pts['V'],'k')
    plt.title('Hodgkin-Huxley membrane potential')

    ev_info = pts.labels.by_label['Event:spikeoutput']
    for ev_ix, ev_tdata in list(ev_info.items()):
        plt.plot(ev_tdata['t'], pts[ev_ix]['V'], 'ko')

    plt.xlabel('t')
    plt.ylabel('V')


# ========

def test_aeIF():
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

    aeIF.set(ics={'V': -70, 'w': 0.1, '_regime_': 0},
             tdata=[0, 10],
             algparams={'init_step': 0.01})

    aeIF.compute('test', verboselevel=0)

    pts = aeIF.sample('test', dt=0.1)

    plt.figure(3)
    plt.plot(pts['t'], pts['V'])
    plt.figure(4)
    plt.plot(pts['t'], pts['w'])


test_HH()
test_aeIF()

plt.show()


