"""
    An example set of basic compartmental model ModelSpec classes
    for use in computational neuroscience modeling.

    Basic classes provided:
    channel, compartment

    For multi-compartment models:
    soma, dendr_compartment, synapse, neuron, network, exc_synapse, inh_synapse

Rob Clewley, September 2005.
"""
from __future__ import division, absolute_import, print_function

from PyDSTool import *
from copy import copy

# which generators will specs made from these classes be compatible with
compatGens = findGenSubClasses('ODEsystem')

class compatODEComponent(Component):
    compatibleGens=compatGens
    targetLangs=targetLangs  # (from common.py) -- all are compatible with ODEs

class compatODELeafComponent(LeafComponent):
    compatibleGens=compatGens
    targetLangs=targetLangs  # (from common.py) -- all are compatible with ODEs

# -----------------------------------------------------------------------------

# basic classes -- not intended for direct use
# indicates that only ODE-based Generators are compatible
class channel(compatODELeafComponent):
    pass

class compartment(compatODEComponent):
    pass

compartment.compatibleSubcomponents=(channel,)
channel.compatibleContainers=(compartment,)

class channel_on(channel):
    pass

class channel_off(channel):
    pass


# use these for multi-compartment cell model:
class soma(compartment):
    pass

class dendr_compartment(compartment):
    pass

class neurite_compartment(compartment):
    pass

class neuron(compatODEComponent):
    pass


# coupling
class synapse(compatODELeafComponent):
    pass

class exc_synapse(synapse):
    pass

class inh_synapse(synapse):
    pass

# point-neuron network
class pnnetwork(compatODEComponent):
    pass

# regular network
class network(compatODEComponent):
    pass

soma.compatibleContainers=(neuron, pnnetwork)
neuron.compatibleSubcomponents=(soma, dendr_compartment, neurite_compartment,
                                synapse)
neuron.compatibleContainers=(network,)
synapse.compatibleContainers=(neuron, pnnetwork)
pnnetwork.compatibleSubcomponents=(soma, synapse)
network.compatibleSubcomponents=(neuron,)

# helpful defaults exported for end-users
global voltage, V
voltage = 'V'
V = Var(voltage)


# ----------------------------------------------------------------------------
## Object factory functions, and related utilities

def disconnectSynapse(syn, mspec):
    """Disconnect synapse object syn from ModelSpec object mspec in
    which it has been declared, and remove its target cell's synaptic
    channels."""
    targsfound = []
    for targname in syn.connxnTargets:
        targsfound.extend(searchModelSpec(mspec, targname))
    if len(targsfound) == len(syn.connxnTargets):
        if remain(targsfound, syn.connxnTargets) == []:
            # all targets were found
            mspec.remove(targsfound)
        else:
            raise ValueError("synapse targets were not found in model spec"
                             "argument")
    else:
        raise ValueError("synapse targets were not found in model spec"
                         "argument")
    syn.delConnxnTarget(targsfound)
    mspec.remove(syn)


def connectWithSynapse(synname, syntypestr, source_cell, dest_cell,
                       dest_compartment_name="",
                       threshfun=None, alpha=None, beta=None,
                       threshfun_d=None, alpha_d=None, beta_d=None,
                       adapt_typestr=None, vrev=None, g=None,
                       noauxs=True, subclass=channel):
    """Make a chemical or electrical synapse between two neurons. For
    gap junctions this function returns None, otherwise will return the
    synapse gating variable object.

    Valid source_cell and dest_cell is a neuron or, for point neurons only,
      a soma is allowed.
    The optional argument dest_compartment_name is a declared name in
    dest_cell, or by default will be the soma.

    The standard assumed rate equation for a chemical synapse conductance is

       s' = (1-alpha) * s - beta * s

    Use the _d versions of threshfun, alpha, and beta, for an adapting chemical
    synapse type's adapting variable.
    """
    if isinstance(source_cell, compartment):
        addedNeuron1Name = ""
        soma1name = source_cell.name
    elif isinstance(source_cell, neuron):
        addedNeuron1Name = source_cell.name + '.'
        soma1name = source_cell._componentTypeMap['soma'][0].name
    else:
        raise TypeError("Invalid cell type to connect with synapse")
    if isinstance(dest_cell, compartment):
        addedNeuron2Name = ""
        if dest_compartment_name == "":
            comp2name = dest_cell.name
        else:
            raise ValueError("Cannot specify destination compartment name "
                             "when dest_cell has type soma")
    elif isinstance(dest_cell, neuron):
        addedNeuron2Name = dest_cell.name + '.'
        if dest_compartment_name == "":
            comp2name = dest_cell._componentTypeMap['soma'][0].name
        else:
            comp_objlist = dest_cell._componentTypeMap['compartment']
            comp_names = [o.name for o in comp_objlist]
            # check legitimacy of dest_compartment_name in dest_cell
            if dest_cell._componentTypeMap['soma'][0].name == dest_compartment_name:
                comp2name = dest_compartment_name
            elif dest_compartment_name in comp_names:
                comp2name = dest_compartment_name
            else:
                raise ValueError("Destination compartment name " \
                                 + dest_compartment_name \
                                 + "not found in destination cell")
    else:
        raise TypeError("Invalid cell type to connect with synapse")

    if addedNeuron1Name != "":
        # we were given a neuron for source_cell, so should use its name
        # to help distinguish multiple occurrences of svar name at time
        # of registration in a network component.
        svarname1 = source_cell.name+"_"+soma1name
    else:
        svarname1 = soma1name
    if addedNeuron2Name != "":
        # we were given a neuron for dest_cell, so should use its name
        # to help distinguish multiple occurrences of svar name at time
        # of registration in a network component.
        svarname2 = dest_cell.name+"_"+comp2name
    else:
        svarname2 = comp2name

    if syntypestr == 'gap':
        # make the first 1/2 of the channel going from source -> dest
        channel_s12 = makeSynapseChannel(synname+'_1', None,
                                         comp2name+'.'+voltage,
                                         syntypestr, vrev=soma1name+'.'+voltage,
                                         g=g, noauxs=noauxs,
                                         subclass=subclass,
                                         gamma1={voltage: (synname+'_1',soma1name+'.'+voltage),
                                                 synname+'_1': (voltage,)})
        channel_name = nameResolver(channel_s12, dest_cell)
        # update with globalized name
        targetChannel = addedNeuron2Name+comp2name+'.'+channel_name
        channel_s12.name = channel_name
        # make the other 1/2 channel going in the other direction
        channel_s21 = makeSynapseChannel(synname+'_2', None,
                                         soma1name+'.'+voltage,
                                         syntypestr, vrev=comp2name+'.'+voltage,
                                         g=g, noauxs=noauxs,
                                         subclass=subclass,
                                         gamma1={voltage: (synname+'_2',comp2name+'.'+voltage),
                                                 synname+'_2': (voltage,)})
        channel_name = nameResolver(channel_s21, source_cell)
        # update with globalized name
        targetChannel = addedNeuron1Name+soma1name+'.'+channel_name
        channel_s21.name = channel_name
    else:
        # chemical synapses
        svar = Var('s_'+svarname1+'_'+svarname2)
        svarname = nameResolver(svar, dest_cell)
        channel_s12 = makeSynapseChannel(synname,
                                         addedNeuron1Name+synname+'.'+svarname,
                                         comp2name+'.'+voltage,
                                         syntypestr, vrev=vrev, g=g, noauxs=noauxs,
                                         subclass=subclass,
                                         gamma1=args(voltage=(synname,),
                                                     synname=(voltage,)))
        channel_name = nameResolver(channel_s12, dest_cell)
        # update with globalized name
        targetChannel = addedNeuron2Name+comp2name+'.'+channel_name
        channel_s12.name = channel_name
        # check that this source synaptic gating variable hasn't been made before,
        # otherwise re-use that variable?
        if threshfun_d == alpha_d == beta_d == None:
            syn12 = makeSynapse(synname, svarname,
                            addedNeuron1Name+soma1name+'.'+voltage,
                            syntypestr, threshfun=threshfun, alpha=alpha,
                            beta=beta, targetchannel=targetChannel, noauxs=noauxs)
        else:
            syn12 = makeAdaptingSynapse(synname, svarname, 'd',
                            addedNeuron1Name+soma1name+'.'+voltage,
                            syntypestr, adapt_typestr,
                            threshfun=threshfun, alpha=alpha,  beta=beta,
                            threshfun_d=threshfun_d, alpha_d=alpha_d, beta_d=beta_d,
                            targetchannel=targetChannel, noauxs=noauxs)
    if addedNeuron2Name == "":
        # we were given a soma for dest_cell
        dest_cell.add(channel_s12)
    else:
        comp = dest_cell.components[comp2name]
        dest_cell.remove(comp2name)
        comp.add(channel_s12)
        dest_cell.add(comp)
    if syntypestr == 'gap':
        source_cell.addConnxnTarget(synname+'_1')
        source_cell.add(channel_s21)
        dest_cell.addConnxnTarget(synname+'_2')
        return None
    else:
        source_cell.addConnxnTarget(synname)
        if addedNeuron1Name != "":
            # we were given a neuron class for source_cell, so we have to
            # add the synaptic variable component to that cell
            source_cell.add(syn12)
        return syn12


# ----------------------------------------------------------------------------


def makeSynapseChannel(name, gatevarname=None, voltage=voltage, typestr=None,
                       vrev=None, g=None, parlist=None, noauxs=True, subclass=channel,
                       gamma1=None, gamma2=None):
    """Make a chemical or electrical (gap junction) synapse channel in a soma.
    To select these, the typestr argument must be one of 'exc', 'inh', or 'gap'.
    For a user-defined chemical synapse use a different string name for typestr.
    For gap junctions, use the pre-synaptic membrane potential's soma ModelSpec
    object or its full hierarchical name string for the argument vrev.

    Returns a channel-type object by default, or some user-defined subclass if desired.
    """
    channel_s = subclass(name)
    if vrev is None:
        if typestr == 'gap':
            raise ValueError("Must provide vrev for gap junctions")
        elif typestr == 'exc':
            vrevpar = Par('0', 'vrev')
        elif typestr == 'inh':
            vrevpar = Par('-75', 'vrev')
        elif isinstance(typestr, str):
            vrevpar = Par('vrev')
        else:
            raise TypeError("Invalid type for synapse")
    elif typestr == 'gap':
        if isinstance(vrev, str):
            vrevpar = Var(vrev)
        else:
            vrevpar = vrev
    elif isinstance(vrev, str):
        vrevpar = Par(vrev, 'vrev')
    elif isinstance(vrev, float) or isinstance(vrev, int):
        vrevpar = Par(repr(vrev), 'vrev')
    else:
        # allow for ModelSpec function of other variables
        vrevpar = vrev
    v_pot = Var(voltage)   # this is to provide the local name only
    if g is None:
        gpar = Par('g')
    elif isinstance(g, str):
        gpar = Par(g, 'g')
    else:
        gpar = Par(repr(g), 'g')
    if typestr == 'gap':
        condQ = gpar
        if gatevarname is not None:
            raise ValueError("gatevarname must be None for gap junctions")
    else:
        gatevar = Var(gatevarname)
        condQ = gpar*gatevar

    if noauxs:
        I = Var(condQ*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
        if typestr == 'gap':
            # don't include vrevpar, which is really an already existing
            # V variable in a different cell
            channel_s.add([gpar,I])
        else:
            channel_s.add([gpar,vrevpar,I])
    else:
        cond = Var(condQ, 'cond', [0,Inf], specType='ExpFuncSpec')
        shunt = Var(cond*vrevpar, 'shunt', specType='ExpFuncSpec')
        I = Var(cond*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
        if typestr == 'gap':
            arg_list = []
        else:
            arg_list = [gatevarname]
        if not isinstance(vrevpar, Par):
            if parlist is not None:
                avoid_pars = [str(q) for q in parlist if isinstance(q, Par)]
            else:
                avoid_pars = []
            arg_list.extend(remain(vrevpar.freeSymbols, avoid_pars))
        cond_fn = Fun(condQ(), arg_list, 'dssrt_fn_cond')
        shunt_fn = Fun(condQ()*vrevpar, arg_list, 'dssrt_fn_shunt')
        I_fn = Fun(condQ()*(v_pot-vrevpar), arg_list+[voltage],
                   'dssrt_fn_I')
        channel_s.add([gpar,I,cond,shunt,cond_fn,shunt_fn,I_fn])
        if typestr != 'gap':
            # don't include vrevpar, which is really an already existing
            # V variable in a different cell
            channel_s.add([vrevpar])
        if gamma1 is None:
            gamma1 = {}
        if gamma2 is None:
            gamma2 = {}
        channel_s.gamma1 = gamma1
        channel_s.gamma2 = gamma2
    return channel_s


def makeExtInputCurrentChannel(name, noauxs=True, subclass=channel,
                               gamma1=None, gamma2=None):
    """External input signal used directly as a current. Supply the Input having
    coordinate name 'ext_input'.

    Returns a channel-type object by default, or some user-defined subclass if desired.
    """
    channel_Iext = subclass(name)
    # Input doesn't need a name for the internal signal (confused ModelSpec as an
    # unresolved name!)
    inp = Input('', 'ext_input', [0,Inf], specType='ExpFuncSpec')
    I_ext = Var(-inp, 'I', specType='ExpFuncSpec')
    if noauxs:
        channel_Iext.add([inp, I_ext])
    else:
        arg_list = ['ext_input']
        I_cond = Var('0', 'cond', [0,Inf], specType='ExpFuncSpec')   # no conductance for this plain current
        I_shunt = Var(inp, 'shunt', specType='ExpFuncSpec')    # technically not a shunt but acts like a V-independent shunt!
        cond_fn = Fun(inp(), arg_list, 'dssrt_fn_cond')
        shunt_fn = Fun(inp()*vrevpar, arg_list, 'dssrt_fn_shunt')
        I_fn = Fun(inp()*(v_pot-vrevpar), arg_list+[voltage],
                   'dssrt_fn_I')
        channel_Iext.add([inp, I_ext, I_cond, I_shunt,
                          cond_fn, shunt_fn, I_fn])
        if gamma1 is not None:
            raise ValueError("gamma1 must be None")
        channel_Iext.gamma1 = {}
        if gamma2 is None:
            gamma2 = {}
        channel_Iext.gamma2 = gamma2
    return channel_Iext


def makeExtInputConductanceChannel(name, voltage=voltage,
                                   g=None, vrev=None, parlist=None,
                                   noauxs=True, subclass=channel,
                                   gamma1=None, gamma2=None):
    """External input signal used as a conductance. Supply the Input having
    coordinate name 'ext_input'.

    Returns a channel-type object by default, or some user-defined subclass if desired.
    """
    if g is None:
        gpar = Par('g')
    elif isinstance(g, str):
        gpar = Par(g, 'g')
    else:
        gpar = Par(repr(g), 'g')
    if vrev is None:
        vrevpar = Par('vrev')
    elif isinstance(vrev, str):
        try:
            val = float(vrev)
        except ValueError:
            # allow for ModelSpec reference to or function of other variables
            vrevpar = QuantSpec('vrev', vrev)
        else:
            vrevpar = Par(vrev, 'vrev')
    elif isinstance(vrev, float) or isinstance(vrev, int):
        vrevpar = Par(repr(vrev), 'vrev')
    else:
        # allow for ModelSpec function of other variables
        vrevpar = vrev
    if hasattr(vrevpar, 'name'):
        declare_list = [gpar, vrevpar]
    else:
        declare_list = [gpar]
    channel_Iext = subclass(name)
    v_pot = Var(voltage)   # this is to provide the local name only
    # Input doesn't need a name for the internal signal (confused ModelSpec as an
    # unresolved name!)
    inp = Input('', 'ext_input', [0,Inf], specType='ExpFuncSpec')
    I_ext = Var(gpar*inp*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
    channel_Iext.add(declare_list)
    channel_Iext.add([inp, I_ext])
    if not noauxs:
        I_cond = Var(gpar*inp, 'cond', [0,Inf], specType='ExpFuncSpec')   # no conductance for this plain current
        I_shunt = Var(I_cond*vrevpar, 'shunt', specType='ExpFuncSpec')    # technically not a shunt but acts like a V-independent shunt!
        arg_list = ['ext_input']
        if not isinstance(vrevpar, Par):
            if parlist is not None:
                avoid_pars = [str(q) for q in parlist if isinstance(q, Par)]
            else:
                avoid_pars = []
            arg_list.extend(remain(vrevpar.freeSymbols, avoid_pars))
        cond_fn = Fun(gpar*inp(), arg_list, 'dssrt_fn_cond')
        shunt_fn = Fun(gpar*inp()*vrevpar, arg_list, 'dssrt_fn_shunt')
        I_fn = Fun(gpar*inp()*(v_pot-vrevpar), arg_list+[voltage],
                   'dssrt_fn_I')
        channel_Iext.add([I_cond, I_shunt, cond_fn, shunt_fn, I_fn])
        if gamma1 is not None:
            raise ValueError("gamma1 must be None")
        channel_Iext.gamma1 = {}
        if gamma2 is None:
            gamma2 = {}
        channel_Iext.gamma2 = gamma2
    return channel_Iext


def makeFunctionConductanceChannel(name, parameter_name,
                                   func_def_str,
                                   voltage=voltage,
                                   g=None, vrev=None, parlist=None,
                                   noauxs=True, subclass=channel,
                                   gamma1=None, gamma2=None):
    """Explicit function waveform used as a conductance, e.g. for an alpha-function
    post-synaptic response. Creates a time-activated event to trigger the waveform
    based on a named parameter. The function will be a function of (local/relative)
    time t only.

    Returns a channel-type object by default, or some user-defined subclass if desired.
    """
    if g is None:
        gpar = Par('g')
    elif isinstance(g, str):
        gpar = Par(g, 'g')
    else:
        gpar = Par(repr(g), 'g')
    if vrev is None:
        vrevpar = Par('vrev')
    elif isinstance(vrev, str):
        try:
            val = float(vrev)
        except ValueError:
            # allow for ModelSpec reference to or function of other variables
            vrevpar = QuantSpec('vrev', vrev)
        else:
            vrevpar = Par(vrev, 'vrev')
    elif isinstance(vrev, float) or isinstance(vrev, int):
        vrevpar = Par(repr(vrev), 'vrev')
    else:
        # allow for ModelSpec function of other variables
        vrevpar = vrev
    if hasattr(vrevpar, 'name'):
        declare_list = [gpar, vrevpar]
    else:
        declare_list = [gpar]
    func_def = QuantSpec('f', func_def_str)
    # define parameters based on remaining free names in definition
    declare_list.extend([Par(0, pname) for pname in remain(func_def.freeSymbols,
                                                           ['t'])])
    fun_par = Par(0, parameter_name)
    inp = Fun('if(t > %s, %s, 0)' % (parameter_name, func_def_str),
              ['t'], 'func', [-Inf,Inf], specType='ExpFuncSpec')
    v_pot = Var(voltage)   # this is to provide the local name only
    I_ext = Var(gpar*'func(t)'*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
    channel_Ifunc = subclass(name)
    declare_list.extend([fun_par, inp, I_ext])
    channel_Ifunc.add(declare_list)
    if not noauxs:
        I_cond = Var(gpar*'func(t)', 'cond', [0,Inf], specType='ExpFuncSpec')   # no conductance for this plain current
        I_shunt = Var(I_cond*vrevpar, 'shunt', specType='ExpFuncSpec')    # technically not a shunt but acts like a V-independent shunt!
        arg_list = ['t']
        if not isinstance(vrevpar, Par):
            if parlist is not None:
                avoid_pars = [str(q) for q in parlist if isinstance(q, Par)]
            else:
                avoid_pars = []
            arg_list.extend(remain(vrevpar.freeSymbols, avoid_pars))
        cond_fn = Fun(gpar*'func(t)', arg_list, 'dssrt_fn_cond')
        shunt_fn = Fun(gpar*'func(t)'*vrevpar, arg_list, 'dssrt_fn_shunt')
        I_fn = Fun(gpar*'func(t)'*(v_pot-vrevpar), arg_list+[voltage],
                   'dssrt_fn_I')
        channel_Ifunc.add([I_cond, I_shunt, cond_fn, shunt_fn, I_fn])
        if gamma1 is not None:
            raise ValueError("gamma1 must be None")
        channel_Ifunc.gamma1 = {}
        if gamma2 is None:
            gamma2 = {}
        channel_Ifunc.gamma2 = gamma2
    return channel_Ifunc


def makeBiasChannel(name, I=None, noauxs=True, subclass=channel,
                    gamma1=None, gamma2=None):
    """Constant bias / applied current "channel".

    Returns a channel-type object by default, or some user-defined subclass if desired.
    """
    channel_Ib = subclass(name)
    if I is None:
        Ibias = Par('Ibias')
    elif isinstance(I, str):
        Ibias = Par(I, 'Ibias')
    else:
        Ibias = Par(repr(I), 'Ibias')
    Ib = Var(-Ibias, 'I', specType='ExpFuncSpec')
    if noauxs:
        channel_Ib.add([Ibias,Ib])
    else:
        Ibias_cond = Var('0', 'cond', [0,Inf], specType='ExpFuncSpec')   # no conductance for this plain current
        Ibias_shunt = Var(Ibias, 'shunt', specType='ExpFuncSpec')    # technically not a shunt but acts like a V-independent shunt!
        cond_fn = Fun('0', [], 'dssrt_fn_cond')
        shunt_fn = Fun(Ibias, [], 'dssrt_fn_shunt')
        I_fn = Fun(Ibias, [], 'dssrt_fn_I')
        channel_Ib.add([Ibias,Ibias_cond,Ibias_shunt,Ib, cond_fn, shunt_fn, I_fn])
        if gamma1 is not None:
            raise ValueError("gamma1 must be None")
        channel_Ib.gamma1 = {}
        if gamma2 is None:
            gamma2 = {}
        channel_Ib.gamma2 = gamma2
    return channel_Ib


def makeChannel_halfact(name,voltage=voltage,s=None,isinstant=False,sinf=None,
                taus=None,spow=1,s2=None,isinstant2=False,sinf2=None,taus2=None,
                spow2=1,vrev=None,g=None,
                parlist=None, noauxs=True, subclass=channel,
                gamma1=None, gamma2=None, nonlocal_variables=None):
    """Make an ionic membrane channel using the steady state and rate function formalism.

    i.e., that the gating variable s has a differential equation in the form:
       s' = (sinf(v) - s)/taus(v)
    The channel may have up to two gating variables, each of which is given by an ODE.

    If either gating variable has its corresponding 'isinstant' argument set to True, then
    that variable is set to be instananeous (algebraic, not an ODE), i.e. s = sinf(v). The
    taus function will then be ignored.

    The resulting channel current will be of the form
        name.I = g * s^spow * s2^spow2 * (voltage - vrev)

    Provide any additional Par or Fun objects necessary for the complete definition of the
    channel kinetics in the optional parlist argument.

    nonlocal_variables list (optional) provides string names of any dynamic variables
    referenced in the declared specifications that are not local to this channel.

    Returns a channel-type object by default, or some user-defined subclass if desired.
    """
    if g is None:
        gpar = Par('g')
    elif isinstance(g, str):
        gpar = Par(g, 'g')
    else:
        gpar = Par(repr(g), 'g')
    if vrev is None:
        vrevpar = Par('vrev')
    elif isinstance(vrev, str):
        try:
            val = float(vrev)
        except ValueError:
            # allow for ModelSpec reference to or function of other variables
            vrevpar = QuantSpec('vrev', vrev)
        else:
            vrevpar = Par(vrev, 'vrev')
    elif isinstance(vrev, float) or isinstance(vrev, int):
        vrevpar = Par(repr(vrev), 'vrev')
    else:
        # allow for ModelSpec function of other variables
        vrevpar = vrev
    if hasattr(vrevpar, 'name'):
        declare_list = [gpar, vrevpar]
    else:
        declare_list = [gpar]
    thischannel = subclass(name)
    v_pot = Var(voltage)   # this is to provide the local name only
    if nonlocal_variables is None:
        nonlocal_variables = []
    # deal with gating variable, if present
    if s is None:
        condQ = gpar
        assert taus==sinf==None
        assert spow==1, "power should be unused when gating variable is None"
        # override
        spow=0
    else:
        if taus==sinf==None:
            assert isinstant == False
            gatevar = Par(s)
        else:
            if isinstance(sinf, (Quantity, QuantSpec)):
                sinf_term = Var(sinf(), name=s+'inf')
            elif isinstance(sinf, str):
                sinf_term = Var(sinf, name=s+'inf')
            else:
                raise TypeError("sinf argument must be a string, Quantity or a QuantSpec")
            if isinstant:
                if noauxs:
                    gatevar = Var(sinf_term(), name=s, domain=[0,1],
                                 specType='ExpFuncSpec')
                else:
                    taus_term = Var('0', name='tau'+s)
                    taus_fn = Fun( '0', [v_pot]+nonlocal_variables,
                                   'dssrt_fn_tau'+s )
                    sinf_fn = Fun( sinf_term(), [v_pot]+nonlocal_variables,
                                'dssrt_fn_'+s+'inf' )
                    gatevar = Var(sinf_term(), name=s, domain=[0,1],
                                 specType='ExpFuncSpec')
            else:
                if isinstance(taus, (Quantity, QuantSpec)):
                    taus_term = Var(taus(), name='tau'+s)
                elif isinstance(taus, str):
                    taus_term = Var(taus, name='tau'+s)
                else:
                    raise TypeError("taus argument must be a string, Quantity or a QuantSpec")
                # temporary declaration of symbol s (the argument string) as a Var
                s_ = Var(s)
                if noauxs:
                    gatevar = Var( (sinf_term() - s_) / taus_term(), name=s,
                                 domain=[0,1], specType='RHSfuncSpec')
                else:
                    taus_fn = Fun( taus_term(), [v_pot]+nonlocal_variables,
                                   'dssrt_fn_tau'+s )
                    sinf_fn = Fun( sinf_term(), [v_pot]+nonlocal_variables,
                                'dssrt_fn_'+s+'inf' )
                    gatevar = Var( (sinf_term() - s_) / taus_term(), name=s,
                                 domain=[0,1], specType='RHSfuncSpec')
            if not noauxs:
                declare_list.extend([taus_term,sinf_term,taus_fn,sinf_fn])
        declare_list.append(gatevar)
        condQ = gpar*makePowerSpec(gatevar,spow)

    # deal with second gating variable, if present
    if s2 is not None:
        assert s is not None, "Must use first gating variable name first!"
        assert spow2 != 0, "Do not use second gating variable with spow2 = 0!"
        s2_ = Var(s2)
        if isinstance(sinf2, QuantSpec):
            sinf2_term = Var(sinf2(), name=s2+'inf', specType='ExpFuncSpec')
        elif isinstance(sinf2, str):
            sinf2_term = Var(sinf2, name=s2+'inf', specType='ExpFuncSpec')
        else:
            raise TypeError("sinf2 argument must be a string or a QuantSpec")
        if isinstant2:
            if noauxs:
                gatevar2 = Var(sinf2_term(), name=s2, domain=[0,1],
                              specType='ExpFuncSpec')
            else:
                taus2_term = Var('0', name='tau'+s2)
                taus2_fn = Fun( '0', [v_pot]+nonlocal_variables,
                                'dssrt_fn_tau'+s2 )
                sinf2_fn = Fun( sinf2_term(), [v_pot]+nonlocal_variables,
                                'dssrt_fn_'+s2+'inf' )
                gatevar2 = Var(sinf2_term(), name=s2, domain=[0,1],
                              specType='ExpFuncSpec')
        else:
            if isinstance(taus2, QuantSpec):
                taus2_term = Var(taus2(), name='tau'+s2, specType='ExpFuncSpec')
            elif isinstance(taus2, str):
                taus2_term = Var(taus2, name='tau'+s2, specType='ExpFuncSpec')
            else:
                raise TypeError("taus2 argument must be a string or a QuantSpec")
            if noauxs:
                gatevar2 = Var( (sinf2_term() - s2_) / taus2_term(), name=s2,
                              domain=[0,1], specType='RHSfuncSpec')
            else:
                taus2_fn = Fun( taus2_term(), [v_pot]+nonlocal_variables,
                                'dssrt_fn_tau'+s2 )
                sinf2_fn = Fun( sinf2_term(), [v_pot]+nonlocal_variables,
                                'dssrt_fn_'+s2+'inf' )
                gatevar2 = Var( (sinf2_term() - s2_) / taus2_term(), name=s2,
                              domain=[0,1], specType='RHSfuncSpec')
        declare_list.append(gatevar2)
        if not noauxs:
            declare_list.extend([taus2_term, sinf2_term, taus2_fn, sinf2_fn])
        condQ = condQ * makePowerSpec(gatevar2, spow2)

    if noauxs:
        I = Var(condQ*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
        declare_list.append(I)
    else:
        cond = Var(condQ, 'cond', [0,Inf], specType='ExpFuncSpec')
        shunt = Var(cond*vrevpar, 'shunt', specType='ExpFuncSpec')
        I = Var(cond*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
        if s is None:
            arg_list = []
        else:
            arg_list = [s]
        if s2 is not None:
            arg_list.append(s2)
        # vrev may be a Par, a singleton reference to another parameter already defined,
        # which counts the same; or else a compound reference (function of parameters)
        if not isinstance(vrevpar, Par) and vrevpar.freeSymbols != [str(vrevpar)]:
            if parlist is not None:
                avoid_pars = [str(q) for q in parlist if isinstance(q, Par)]
            else:
                avoid_pars = []
            arg_list.extend(remain(vrevpar.freeSymbols, avoid_pars))
        condQ_var_free = intersect(condQ.freeSymbols, nonlocal_variables)
        cond_var_free = intersect(cond.freeSymbols, nonlocal_variables)
        cond_fn = Fun(condQ(), arg_list+condQ_var_free, 'dssrt_fn_cond')
        shunt_fn = Fun(cond()*vrevpar, arg_list+cond_var_free, 'dssrt_fn_shunt')
        I_fn = Fun(cond()*(v_pot-vrevpar), arg_list+[voltage]+cond_var_free,
                   'dssrt_fn_I')
        declare_list.extend([I,cond,shunt, cond_fn, shunt_fn, I_fn])
        if gamma1 is None:
            gamma1 = {}
        if gamma2 is None:
            gamma2 = {}
        thischannel.gamma1 = gamma1
        thischannel.gamma2 = gamma2
    if parlist is not None:
        declare_list.extend(parlist)
    thischannel.add(declare_list)
    return thischannel


def makeChannel_rates(name,voltage=voltage,
                      s=None,isinstant=False,arate=None,brate=None,spow=1,
                      s2=None,isinstant2=False,arate2=None,brate2=None,spow2=1,
                      vrev=None,g=None,parlist=None,noauxs=True,subclass=channel,
                      gamma1=None, gamma2=None, nonlocal_variables=None):
    """Make an ionic membrane channel using the forward and backward rate formalism.

    i.e., that the gating variable s has a differential equation in the form:
       s' = sa(v) * (1-s) - sb(v) * s
    The channel may have up to two gating variables, each of which is given by an ODE.

    If either gating variable has its corresponding 'isinstant' argument set to True, then
    that variable is set to be instananeous (algebraic, not an ODE),
    i.e. s = sinf(v) = a(v) / (a(v) + b(v))

    The resulting channel current will be of the form
        name.I = g * s^spow * s2^spow2 * (voltage - vrev)

    Provide any additional Par or Fun objects necessary for the complete definition of the
    channel kinetics in the optional parlist argument.

    nonlocal_variables list (optional) provides string names of any dynamic variables
    referenced in the declared specifications that are not local to this channel.

    Returns a channel-type object by default, or some user-defined subclass if desired.
    """
    if g is None:
        gpar = Par('g')
    elif isinstance(g, str):
        gpar = Par(g, 'g')
    else:
        gpar = Par(repr(g), 'g')
    if vrev is None:
        vrevpar = Par('vrev')
    elif isinstance(vrev, str):
        try:
            val = float(vrev)
        except ValueError:
            # allow for ModelSpec reference to or function of other variables
            vrevpar = QuantSpec('vrev', vrev)
        else:
            vrevpar = Par(vrev, 'vrev')
    elif isinstance(vrev, float) or isinstance(vrev, int):
        vrevpar = Par(repr(vrev), 'vrev')
    else:
        # allow for ModelSpec function of other variables
        vrevpar = vrev
    if hasattr(vrevpar, 'name'):
        declare_list = [gpar, vrevpar]
    else:
        declare_list = [gpar]
    thischannel = subclass(name)
    v_pot = Var(voltage)   # this is to provide the local name only
    if nonlocal_variables is None:
        nonlocal_variables = []
    # deal with gating variable, if present
    if s is None:
        condQ = gpar
        assert arate==brate==None
        assert spow==1, "power should be unused when gating variable is None"
        # override
        spow=0
    else:
        if arate==brate==None:
            assert isinstant == False
            gatevar = Par(s)
        else:
            if isinstance(arate, (Quantity, QuantSpec)):
                aterm = Var(arate(), name='a'+s)
            elif isinstance(arate, str):
                aterm = Var(arate, name='a'+s)
            else:
                raise TypeError("arate argument must be a string or QuantSpec")
            if isinstance(brate, (Quantity, QuantSpec)):
                bterm = Var(brate(), name='b'+s)
            elif isinstance(brate, str):
                bterm = Var(brate, name='b'+s)
            else:
                raise TypeError("brate argument must be a string or QuantSpec")
            # temporary declaration of symbol s (the argument string) as a Var
            s_ = Var(s)
            if isinstant:
                if noauxs:
                    gatevar = Var( aterm()/(aterm()+bterm()), name=s,
                                 domain=[0,1], specType='ExpFuncSpec')
                else:
                    taus_term = Var('0', name='tau'+s)
                    sinf_term = Var( aterm/(aterm+bterm), name=s+'inf')
                    taus_fn = Fun( '0', [voltage]+nonlocal_variables,
                                   'dssrt_fn_tau'+s )
                    sinf_fn = Fun( aterm()/(aterm()+bterm()),
                                   [voltage]+nonlocal_variables,
                                   'dssrt_fn_'+s+'inf' )
                    gatevar = Var( aterm()/(aterm()+bterm()), name=s,
                                 domain=[0,1], specType='ExpFuncSpec')
            else:
                if noauxs:
                    gatevar = Var( aterm()*(1-s_)-bterm()*s_, name=s,
                                 domain=[0,1], specType='RHSfuncSpec')
                else:
                    taus_term = Var( 1/(aterm+bterm), name='tau'+s)
                    sinf_term = Var( aterm*taus_term, name=s+'inf')
                    taus_fn = Fun( 1/(aterm()+bterm()), [voltage]+nonlocal_variables,
                                   'dssrt_fn_tau'+s )
                    sinf_fn = Fun( aterm()/(aterm()+bterm()),
                                   [voltage]+nonlocal_variables,
                                   'dssrt_fn_'+s+'inf' )
                    gatevar = Var( aterm()*(1-s_)-bterm()*s_, name=s,
                                 domain=[0,1], specType='RHSfuncSpec')
            if not noauxs:
                declare_list.extend([taus_term,sinf_term,aterm,bterm,
                                     taus_fn, sinf_fn])
        declare_list.append(gatevar)
        condQ = gpar*makePowerSpec(gatevar,spow)

    # deal with second gating variable, if present
    if s2 is not None:
        assert s is not None, "Must use first gating variable name first!"
        assert spow2 != 0, "Do not use second gating variable with spow2 = 0!"
        if isinstance(arate2, (Quantity, QuantSpec)):
            aterm2 = Var(arate2(), name='a'+s2)
        elif isinstance(arate2, str):
            aterm2 = Var(arate2, name='a'+s2)
        else:
            raise TypeError("arate2 argument must be a string, Quantity or QuantSpec")
        if isinstance(brate2, (Quantity, QuantSpec)):
            bterm2 = Var(brate2(), name='b'+s2)
        elif isinstance(brate2, str):
            bterm2 = Var(brate2, name='b'+s2)
        else:
            raise TypeError("brate2 argument must be a string, Quantity or QuantSpec")
        s2_ = Var(s2)
        if isinstant2:
            if noauxs:
                gatevar2 = Var( aterm2()/(aterm2()+bterm2()), name=s2,
                              domain=[0,1], specType='ExpFuncSpec')
            else:
                taus2_term = Var( '0', name='tau'+s2)
                sinf2_term = Var( aterm2()/(aterm2()+bterm2()), name=s2+'inf')
                taus2_fn = Fun( '0', [voltage]+nonlocal_variables,
                                'dssrt_fn_tau'+s2 )
                sinf2_fn = Fun( aterm2()/(aterm2()+bterm2()),
                                [voltage]+nonlocal_variables,
                                'dssrt_fn_'+s2+'inf' )
                gatevar2 = Var( aterm2()/(aterm2()+bterm2()), name=s2,
                              domain=[0,1], specType='ExpFuncSpec')
        else:
            if noauxs:
                gatevar2 = Var(aterm2()*(1-s2_)-bterm2()*s2_, name=s2,
                               domain=[0,1], specType='RHSfuncSpec')
            else:
                taus2_term = Var( 1/(aterm2()+bterm2()), name='tau'+s2)
                sinf2_term = Var( aterm2()*taus2_term(), name=s2+'inf')
                taus2_fn = Fun( 1/(aterm2()+bterm2()), [voltage]+nonlocal_variables,
                                'dssrt_fn_tau'+s2 )
                sinf2_fn = Fun( aterm2()/(aterm2()+bterm2()),
                                [voltage]+nonlocal_variables,
                                'dssrt_fn_'+s2+'inf' )
                gatevar2 = Var(aterm2()*(1-s2_)-bterm2()*s2_, name=s2,
                               domain=[0,1], specType='RHSfuncSpec')
        condQ = condQ * makePowerSpec(gatevar2, spow2)
        declare_list.append(gatevar2)
        if not noauxs:
            declare_list.extend([taus2_term, sinf2_term, aterm2, bterm2,
                                 taus2_fn, sinf2_fn])

    if noauxs:
        I = Var(condQ*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
        declare_list.append(I)
    else:
        cond = Var(condQ, 'cond', [0,Inf], specType='ExpFuncSpec')
        shunt = Var(cond*vrevpar, 'shunt', specType='ExpFuncSpec')
        I = Var(cond*(v_pot-vrevpar), 'I', specType='ExpFuncSpec')
        if s is None:
            arg_list = []
        else:
            arg_list = [s]
        if s2 is not None:
            arg_list.append(s2)
        # vrev may be a Par, a singleton reference to another parameter already defined,
        # which counts the same; or else a compound reference (function of parameters)
        if not isinstance(vrevpar, Par) and vrevpar.freeSymbols != [str(vrevpar)]:
            if parlist is not None:
                avoid_pars = [str(q) for q in parlist if isinstance(q, Par)]
            else:
                avoid_pars = []
            arg_list.extend(remain(vrevpar.freeSymbols, avoid_pars))
        condQ_var_free = intersect(condQ.freeSymbols, nonlocal_variables)
        cond_var_free = intersect(cond.freeSymbols, nonlocal_variables)
        cond_fn = Fun(condQ(), arg_list+condQ_var_free, 'dssrt_fn_cond')
        shunt_fn = Fun(cond()*vrevpar, arg_list+cond_var_free, 'dssrt_fn_shunt')
        I_fn = Fun(cond()*(v_pot-vrevpar), arg_list+[voltage]+cond_var_free,
                   'dssrt_fn_I')
        declare_list.extend([I,cond,shunt, cond_fn, shunt_fn, I_fn])
        if gamma1 is None:
            gamma1 = {}
        if gamma2 is None:
            gamma2 = {}
        thischannel.gamma1 = gamma1
        thischannel.gamma2 = gamma2
    if parlist is not None:
        declare_list.extend(parlist)
    thischannel.add(declare_list)
    return thischannel


def makeSoma(name, voltage=voltage, channelList=None, C=None, noauxs=True,
             subclass=soma, channelclass=channel):
    """Build a soma type of "compartment" from a list of channels and a membrane
    capacitance C, using the local voltage name.

    Specify a sub-class for the channel to target specific types to summate,
    in case the user wants to have runtime control over which are "active" in the
    dynamics (mainly related to the DSSRT toolbox).

    Returns a soma-type object by default, or some user-defined subclass if desired.
    """
    channel_cls_name = className(channelclass)
    v = Var(QuantSpec(str(voltage), '-for(%s,I,+)/C'%channel_cls_name, 'RHSfuncSpec'),
            domain=[-150,100])
    if C is None:
        C = Par('C')
    elif isinstance(C, str):
        gpar = Par(C, 'C')
    else:
        C = Par(repr(C), 'C')

    asoma = soma.__new__(subclass)
    asoma.__init__(name)
    declare_list = copy(channelList)
    if noauxs:
        declare_list.extend([v,C])
    else:
        tauv = Var(QuantSpec('tauv', '1/(C*for(%s,cond,+))'%channel_cls_name))
        vinf = Var(QuantSpec('vinf', 'for(%s,shunt,+)'%channel_cls_name + \
                                       '/for(%s,cond,+)'%channel_cls_name))
        # for each cond in channel, append signature of dssrt_fn_cond
        arg_list_c = []
        arg_list_s = []
        c_sum = []
        s_sum = []
        dssrt_inputs = {}
        vstr = str(voltage)
        for c in channelList:
            def process_targ_name(inp):
                if inp == vstr:
                    return inp
                else:
                    return c.name+'.'+inp
            for targ, inputs in c.gamma1.items():
                proc_targ = process_targ_name(targ)
                if proc_targ not in dssrt_inputs:
                    dssrt_inputs[proc_targ] = args(gamma1=[], gamma2=[])
                dssrt_inputs[proc_targ].gamma1.extend([process_targ_name(i) \
                                                                     for i in inputs])
            for targ, inputs in c.gamma2.items():
                proc_targ = process_targ_name(targ)
                if proc_targ not in dssrt_inputs:
                    dssrt_inputs[proc_targ] = args(gamma1=[], gamma2=[])
                dssrt_inputs[proc_targ].gamma2.extend([process_targ_name(i) \
                                                                     for i in inputs])
            cond = c['dssrt_fn_cond']
            shunt = c['dssrt_fn_shunt']
            cond_sig = cond.signature
            shunt_sig = shunt.signature
            arg_list_c.extend(cond_sig)
            arg_list_s.extend(shunt_sig)
            c_sum.append( c.name + '.' + cond.name+'('+','.join(cond_sig)+')' )
            s_sum.append( c.name + '.' + shunt.name+'('+','.join(shunt_sig)+')' )
        arg_list1 = makeSeqUnique(arg_list_c)
        arg_list1.sort()
        arg_list2 = makeSeqUnique(arg_list_c+arg_list_s)
        arg_list2.sort()
        tauv_fn = Fun('1/(C*(%s))'%'+'.join(c_sum), arg_list1,
                      'dssrt_fn_tau'+vstr)
        vinf_fn = Fun('(%s)/(%s)'%('+'.join(s_sum), '+'.join(c_sum)),
                      arg_list2, 'dssrt_fn_'+vstr+'inf')
        declare_list.extend([v,C,vinf,tauv,tauv_fn,vinf_fn])
        asoma.dssrt_inputs = dssrt_inputs
    asoma.add(declare_list)
    return asoma



def makeDendrite(name, voltage=voltage, channelList=None, C=None, noauxs=True,
             subclass=dendr_compartment, channelclass=channel):
    """Build a dendrite type of "compartment" from a list of channels and a membrane
    capacitance C, using the local voltage name.

    Specify a sub-class for the channel to target specific types to summate,
    in case the user wants to have runtime control over which are "active" in the
    dynamics (mainly related to the DSSRT toolbox).

    Returns a dendrite-type object by default, or some user-defined subclass if desired.
    """
    channel_cls_name = className(channelclass)
    v = Var(QuantSpec(str(voltage), '-for(%s,I,+)/C'%channel_cls_name, 'RHSfuncSpec'),
            domain=[-150,100])
    if C is None:
        C = Par('C')
    elif isinstance(C, str):
        gpar = Par(C, 'C')
    else:
        C = Par(repr(C), 'C')

    acomp = dendr_compartment.__new__(subclass)
    acomp.__init__(name)
    declare_list = copy(channelList)
    if noauxs:
        declare_list.extend([v,C])
    else:
        tauv = Var(QuantSpec('tauv', '1/(C*for(%s,cond,+))'%channel_cls_name))
        vinf = Var(QuantSpec('vinf', 'for(%s,shunt,+)'%channel_cls_name + \
                                       '/for(%s,cond,+)'%channel_cls_name))
        declare_list.extend([v,C,vinf,tauv])
    acomp.add(declare_list)
    return acomp


def makeNeurite(name, voltage=voltage, channelList=None, C=None, noauxs=True,
             subclass=dendr_compartment, channelclass=channel):
    """Build a neurite type of "compartment" from a list of channels and a membrane
    capacitance C, using the local voltage name.

    Specify a sub-class for the channel to target specific types to summate,
    in case the user wants to have runtime control over which are "active" in the
    dynamics (mainly related to the DSSRT toolbox).

    Returns a neurite-type object by default, or some user-defined subclass if desired.
    """
    channel_cls_name = className(channelclass)
    v = Var(QuantSpec(str(voltage), '-for(%s,I,+)/C'%channel_cls_name, 'RHSfuncSpec'),
            domain=[-150,100])
    if C is None:
        C = Par('C')
    elif isinstance(C, str):
        gpar = Par(C, 'C')
    else:
        C = Par(repr(C), 'C')

    acomp = neurite_compartment.__new__(subclass)
    acomp.__init__(name)
    declare_list = copy(channelList)
    if noauxs:
        declare_list.extend([v,C])
    else:
        tauv = Var(QuantSpec('tauv', '1/(C*for(%s,cond,+))'%channel_cls_name))
        vinf = Var(QuantSpec('vinf', 'for(%s,shunt,+)'%channel_cls_name + \
                                       '/for(%s,cond,+)'%channel_cls_name))
        declare_list.extend([v,C,vinf,tauv])
    acomp.add(declare_list)
    return acomp


def makePointNeuron(name, voltage=voltage, channelList=None, synapseList=None,
                    C=None, noauxs=True):
    """Factory function for single compartment neurons ("point neurons").

    Returns a neuron type object from a list of ion channels
    and synapses (if any are defined). The ion channels will internally be built into
    a soma compartment type on the fly.
    """
    asoma = makeSoma('soma', voltage, channelList, C=C, noauxs=noauxs)
    n = neuron(name)
    n.add(asoma)
    if synapseList is not None:
        n.add(synapseList)
    return n


def makePointNeuronNetwork(name, componentList):
    """Factory function returning a pnnetwork type object from a list of compatible
    components (somas and synapses).
    """
    net = pnnetwork(name)
    net.add(componentList)
    return net


def makeNeuronNetwork(name, neuronList):
    """Factory function returning a network type object from a list of compatible
    components (neurons). The neurons can be single or multi-compartmental.

    Currently untested!
    """
    net = network(name)
    net.add(neuronList)
    return net


def makeSynapse(name, gatevar, precompartment, typestr, threshfun=None,
                alpha=None, beta=None, targetchannel=None,
                evalopt=True, noauxs=True):
    """Make a chemical synapse channel object.
    """
    if targetchannel is None:
        raise TypeError("Must provide name of synaptic channel object in "
                        "target cell's compartment")
    if typestr == 'exc':
        if alpha is None:
            alpha = 10.
        if beta is None:
            beta = 0.5
        syn = exc_synapse(name)
    elif typestr == 'inh':
        if alpha is None:
            alpha = 1.
        if beta is None:
            beta = 0.1
        syn = inh_synapse(name)
    elif typestr == "":
        syn = synapse(name)
    else:
        raise ValueError("Invalid type of synapse specified")
    s_ = Var(gatevar)
    if alpha is None:
        a = Par('alpha')
    elif isinstance(alpha, str):
        gpar = Par(alpha, 'alpha')
    else:
        a = Par(repr(alpha), 'alpha')
    if beta is None:
        b = Par('beta')
    elif isinstance(beta, str):
        gpar = Par(beta, 'beta')
    else:
        b = Par(repr(beta), 'beta')
    if threshfun is None:
        f = Fun('0.5+0.5*tanh(v/4.)', ['v'], 'thresh')
    else:
        assert isinstance(threshfun, tuple), \
               "threshfun must be pair (vname, funbody)"
        if isinstance(threshfun[1], QuantSpec):
            funbody = threshfun[1].specStr
        elif isinstance(threshfun[1], str):
            funbody = threshfun[1]
        else:
            raise TypeError("threshold function must be a string or a "
                            "QuantSpec")
        if threshfun[0] not in funbody:
            print("Warning: voltage name %s does not appear in function body!"%threshfun[0])
        f = Fun(funbody, [threshfun[0]], 'thresh')
    assert len(f.signature) == 1, \
           'threshold function must be a function of a single argument (voltage)'
    if evalopt:
        alpha_ = a*f.eval(**{f.signature[0]: precompartment})
    else:
        alpha_ = a*f(precompartment)
        syn.add(f)
    s = Var(alpha_*(1-s_)-b*s_, name=gatevar,
            specType='RHSfuncSpec', domain=[0,1])
    syn.add([s,a,b])
    syn.addConnxnTarget(targetchannel)
    if not noauxs:
        sinf = Var( alpha_/(alpha_+b), name=gatevar+'inf')
        taus = Var( 1/(alpha_+b), name='tau'+gatevar)
        sinf_fn = Fun( alpha_/(alpha_+b), [precompartment], 'dssrt_fn_'+gatevar+'inf')
        taus_fn = Fun( 1/(alpha_+b), [precompartment], 'dssrt_fn_tau'+gatevar)
        syn.add([sinf,taus,sinf_fn,taus_fn])
    return syn


def makeAdaptingSynapse(name, gatevar, adaptvar, precompartment, typestr,
                adapt_typestr,
                threshfun=None, alpha=None, beta=None,
                threshfun_d=None, alpha_d=None, beta_d=None,
                targetchannel=None, evalopt=True, noauxs=True):
    """Make an adapting chemical synapse channel object.
    """
    if targetchannel is None:
        raise TypeError("Must provide name of synaptic channel object in "
                        "target cell's compartment")
    if typestr == 'exc':
        if alpha is None:
            alpha = 10.
        if beta is None:
            beta = 0.5
        syn = exc_synapse(name)
    elif typestr == 'inh':
        if alpha is None:
            alpha = 1.
        if beta is None:
            beta = 0.1
        syn = inh_synapse(name)
    elif typestr == "":
        syn = synapse(name)
    else:
        raise ValueError("Invalid type of synapse specified")
    # synaptic variable
    s_ = Var(gatevar)
    if alpha is None:
        a = Par('alpha')
    elif isinstance(alpha, str):
        a = Par(alpha, 'alpha')
    else:
        a = Par(repr(alpha), 'alpha')
    if beta is None:
        b = Par('beta')
    elif isinstance(beta, str):
        b = Par(beta, 'beta')
    else:
        b = Par(repr(beta), 'beta')
    if threshfun is None:
        f = Fun('0.5+0.5*tanh(v/4.)', ['v'], 'thresh')
    else:
        assert isinstance(threshfun, tuple), \
               "threshfun must be pair (vname, funbody)"
        if isinstance(threshfun[1], QuantSpec):
            funbody = threshfun[1].specStr
        elif isinstance(threshfun[1], str):
            funbody = threshfun[1]
        else:
            raise TypeError("threshold function must be a string or a "
                            "QuantSpec")
        assert threshfun[0] in funbody, \
               "voltage name %s does not appear in function body!"%threshfun[0]
        f = Fun(funbody, [threshfun[0]], 'thresh')
    assert len(f.signature) == 1, \
           'threshold function must be a function of a single argument (voltage)'
    if evalopt:
        alpha_ = a*f.eval(**{f.signature[0]: precompartment})
    else:
        alpha_ = a*f(precompartment)
        syn.add(f)
    # adaptive variable
    if adapt_typestr == 'f':
        at = ''
    elif adapt_typestr == 'd':
        at = '-'
    else:
        raise ValueError("Invalid type for adapting synapse: use 'f' or 'd'")
    d_ = Var(adaptvar)
    if alpha_d is None:
        a_d = Par('alpha_d')
    elif isinstance(alpha_d, str):
        a_d = Par(alpha_d, 'alpha_d')
    else:
        a_d = Par(repr(alpha_d), 'alpha_d')
    if beta_d is None:
        b_d = Par('beta_d')
    elif isinstance(beta_d, str):
        b_d = Par(beta_d, 'beta_d')
    else:
        b_d = Par(repr(beta_d), 'beta_d')
    if threshfun_d is None:
        f_d = Fun('0.5+0.5*tanh(%sv/4.)'%atype, ['v'], 'thresh')
    else:
        assert isinstance(threshfun_d, tuple), \
               "threshfun must be pair (vname, funbody)"
        if isinstance(threshfun_d[1], QuantSpec):
            funbody_d = threshfun_d[1].specStr
        elif isinstance(threshfun_d[1], str):
            funbody_d = threshfun_d[1]
        else:
            raise TypeError("threshold function must be a string or a "
                            "QuantSpec")
        assert threshfun_d[0] in funbody_d, \
               "voltage name %s does not appear in function body!"%threshfun[0]
        f_d = Fun(funbody_d, [threshfun_d[0]], 'thresh')
    assert len(f_d.signature) == 1, \
           'threshold function must be a function of a single argument (voltage)'
    if evalopt:
        alpha_d_ = a_d*f_d.eval(**{f_d.signature[0]: precompartment})
    else:
        alpha_d_ = a_d*f_d(precompartment)
        syn.add(f_d)
    d = Var(alpha_d_*(1-d_)-b_d*d_, name=adaptvar,
            specType='RHSfuncSpec', domain=[0,1])
    s = Var(alpha_*(d_-s_)-b*s_, name=gatevar,
            specType='RHSfuncSpec', domain=[0,1])
    syn.add([s,a,b,d,a_d,b_d])
    syn.addConnxnTarget(targetchannel)
    if not noauxs:
        sinf = Var( alpha_*d_/(alpha_+b), name=gatevar+'inf')
        taus = Var( 1/(alpha_+b), name='tau'+gatevar)
        dinf = Var( alpha_d_/(alpha_d_+b_d), name=adaptvar+'inf')
        taud = Var( 1/(alpha_d_+b_d), name='tau'+adaptvar)
        sinf_fn = Fun( )
        taus_fn = Fun( )
        syn.add([sinf,taus,dinf,taud,sinf_fn,taus_fn])
    return syn

# -----------------------------------------------------------------------------

## Helper functions

def makePowerSpec(v, p):
    if isinstance(p, int):
        if p < 0:
            raise ValueError("power should be > 0")
        elif p==0:
            return "1"
        elif p > 4:
            return Pow(v, p)
        else:
            res = v
            for i in range(p-1):
                res = res * v
            return res
    else:
        return Pow(v, p)


# the following functions will be removed if the calling order for Quantities
# is changed (as it ought to be some day).

def makePar(parname, val=None):
    if val is None:
        return Par(parname)
    elif isinstance(val, str):
        gpar = Par(val, parname)
    else:
        return Par(repr(val), parname)


def makeFun(funname, sig, defn):
    return Fun(defn, sig, funname)
