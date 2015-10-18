"""
NineML interface code for neural network modeling.

This toolbox assumes you have NineML python bindings installed.

R. Clewley, 2012
"""
from __future__ import division, absolute_import

from PyDSTool import *
from PyDSTool.common import _seq_types, _num_types

import numpy as np
from scipy import linspace, isfinite, sign, alltrue, sometrue

import copy, os, sys

allODEgens = findGenSubClasses('ODEsystem')

# ----------------------------------------------------------------------------

_functions = ['get_nineml_model']

_classes = ['NineMLModel']

_features = []

__all__ = _functions + _classes + _features

# ----------------------------------------------------------------------------

try:
    import nineml.abstraction as al
except ImportError:
    raise ImportError("NineML python API is needed for this toolbox to work")


class NineMLModel(LeafComponent):
    compatibleGens=allODEgens + ['ExplicitFnGen']
    targetLangs=targetLangs


def get_nineml_model(c, model_name, target='Vode', extra_args=None,
                     alg_args=None, max_t=np.Inf):
    """component c is a NineML abstraction layer component, and
    assumes you have imported a NineML model through the Python API or built it directly.

    This function only works for 'flat' models, where there is no hierarchy of hybrid models.
    """
    assert c.is_flat(), "Import currently only works for 'flat' models"

    if alg_args is None:
        alg_args = {}

    # lists
    pars = [p.name for p in c.parameters]
    vars = list(c.state_variable_names)
    fns = list(c.alias_names)

    # Turn aliases into function defs:
    # multiple passes until all dependencies resolved
    done = False
    dependencies = {}
    sigs = {}

    while not done:
        done = all([str(a.lhs) in sigs for a in c.aliases])
        for a in c.aliases:
            deps = list(str(s) for s in a.rhs_symbols)
            resolved = True
            fnlist = []
            sig = []
            for d in deps:
                if d in fns:
                    # do we have complete info for this d yet?
                    resolved = resolved and d in sigs
                    fnlist.append(d)
                elif d in vars:
                    sig.append(d)
            dependencies[str(a.lhs)] = fnlist
            if resolved:
                for f in fnlist:
                    sig.extend(sigs[f])
                sigs[str(a.lhs)] = makeSeqUnique(list(sig))

    # Quantity types
    declare_fns = []

    for a in c.aliases:
        sig = sigs[str(a.lhs)]
        fnspec = QuantSpec(str(a.lhs), a.rhs_str)
        fnspec.mapNames({'heaviside': 'heav'})
        # add arguments to the function calls
        for f in dependencies[str(a.lhs)]:
            fi_list = [i for i in range(len(fnspec.parser.tokenized)) if fnspec.parser.tokenized[i] == f]
            offset = 0
            for fi in fi_list:
                arg_spec = QuantSpec('args', '(' + ','.join(sigs[f]) + ')')
                new_defstr = ''.join(fnspec[:fi+1+offset]) + str(arg_spec) + ''.join(fnspec[fi+1+offset:])
                fnspec = QuantSpec(str(a.lhs), new_defstr)
                offset += len(arg_spec.parser.tokenized)
        declare_fns.append(Fun(str(fnspec), sig, name=str(a.lhs)))

    declare_pars = [Par(p) for p in pars]

    targetGen = target + '_ODEsystem'
    if target in ['Vode', 'Euler']:
        targetlang = 'python'
    else:
        targetlang = 'c'

    reg_MCs = {}
    reg_epmaps = {}
    reg_info_list = []
    reg_models = {}

    all_reg_names = list(c.regime_names)
    num_regs = c.num_regimes
    if num_regs > 1:
        is_hybrid = True
    else:
        is_hybrid = False
        # This is the default value, updated to True if there are any
        # state assignments for events (even for only one regime)
        # r is a singleton regime
        r = c.regimes.next()
        for e in r.on_conditions:
            for s in e.state_assignments:
                is_hybrid = True
                break

    if is_hybrid:
        vars.append('regime_')

    reg_ix_to_name = {}
    for reg_ix, r in enumerate(c.regimes):
        reg_ix_to_name[reg_ix] = r.name
    reg_name_to_ix = invertMap(reg_ix_to_name)

    for reg_ix, r in enumerate(c.regimes):
        declare_vars = []
        new_vars = get_regime_model(r, fns, sigs)
        if len(new_vars) != c.num_state_variables:
            if len(new_vars) == 0:
                reg_type = 'ExplicitFn'
                for vname in c.state_variable_names:
                    new_vars.append( Var('initcond(%s)'%vname, name=vname, specType='ExpFuncSpec') )
                if is_hybrid:
                    new_vars.append( Var('%i'%reg_ix, name='regime_', specType='ExpFuncSpec',
                                         domain=(int, Discrete, reg_ix)) )
            else:
                # have to make ODEs with 0 for their RHS
                reg_type = targetGen
                for vname in remain(list(c.state_variable_names), [v.name for v in new_vars]):
                    new_vars.append( Var('0', name=vname, specType='RHSfuncSpec') )
                if is_hybrid:
                    new_vars.append( Var('0', name='regime_', specType='RHSfuncSpec',
                                         domain=(int, Discrete, reg_ix)) )
        else:
            reg_type = targetGen
            if is_hybrid:
                new_vars.append( Var('0', name='regime_', specType='RHSfuncSpec',
                                     domain=(int, Discrete, reg_ix)) )
        declare_vars.extend(new_vars)
        declare_list = declare_pars + declare_fns + declare_vars

        reg_spec = NineMLModel(name=r.name)
        reg_spec.add(declare_list)

        if extra_args is not None:
            # filter out the objects with names matching
            # those missing in this spec
            free = reg_spec.freeSymbols
            extra_names = intersect([ea.name for ea in extra_args], free)
            for name in extra_names:
                reg_spec.add([ea for ea in extra_args \
                              if (ea.name in extra_names and \
                                  ea.name not in reg_spec._registry)])

        reg_spec.flattenSpec(ignoreInputs=True)

        # build events for this regime
        # event mappings for this regime
        epmaps = {}
        events = []
        for e in r.on_events:
            raise NotImplementedError("Non-transition events not yet implemented")

        for e in r.on_conditions:
            defq = QuantSpec('rhs', e.trigger.rhs_str)
            toks = defq.parser.tokenized
            if '=' in toks:
                ix = toks.index('=')
                dirn = 0
            elif '>' in toks:
                ix = toks.index('>')
                dirn = 1
            elif '>=' in toks:
                ix = toks.index('>=')
                dirn = 1
            elif '<' in toks:
                ix = toks.index('<')
                dirn = -1
            elif '<=' in toks:
                ix = toks.index('<=')
                dirn = -1
            else:
                raise ValueError("Event type not implemented!")
            new_defstr = ''.join(toks[:ix]) + '-' + '(' + ''.join(toks[ix+1:]) + ')'
            evnames = [oe.port_name for oe in e.output_events]
            if evnames == []:
                # no outputs, must create an event name based on LHS of trigger condition
                if "".join(toks[:ix]) == 't':
                    evnames = ['time_condition']
                else:
                    # !!! HACKY here, not sure what to do to create a good event name
                    evnames = ["".join([tok for tok in toks[:ix] if tok in vars])+'_event']
            if r.name != e.source_regime.name:
                # event doesn't belong with this regime
                continue
            else:
                reg_target = e.target_regime.name
            for evname in evnames:
                ev_args = {'name': evname,
                       'eventtol': 1e-3,
                       'eventdelay': 1e-3,
                       'starttime': 0,
                       'term': is_hybrid
                        }
                events.append(Events.makeZeroCrossEvent(new_defstr,
                                dirn, ev_args, targetlang=targetlang,
                                flatspec=reg_spec.flatSpec))
                edict = {}
                for s in e.state_assignments:
                    edict[str(s.lhs)] = s.rhs_str
                if is_hybrid:
                    edict['regime_'] = str(reg_name_to_ix[reg_target])
                if len(edict) == 0:
                    epmaps[evname] = (evname, reg_target)
                else:
                    epmaps[evname] = (evname, (reg_target,
                                               EvMapping(edict,
                                                         infodict=dict(pars=pars,
                                                                     vars=vars))))
            reg_epmaps[r.name] = epmaps

        if reg_type == 'ExplicitFn':
            targetGenClassName = 'ExplicitFnGen'
        else:
            targetGenClassName = targetGen
        gen_dict = {'modelspec': reg_spec,
                    'target': targetGenClassName,
                    'algparams': alg_args}

        modelC = ModelConstructor(r.name,
                              generatorspecs={reg_spec.name:
                                              gen_dict},
                              indepvar=('t',[0,max_t]))
        if is_hybrid:
            modelC.icvalues = {reg_spec.name: {'regime_': reg_ix}}

        if events != []:
            modelC.addEvents(reg_spec.name, events)
        reg_model = modelC.getModel()
        reg_MCs[r.name] = modelC
        reg_models[r.name] = reg_model
        reg_MI = intModelInterface(reg_model)

        reg_info_list.append(makeModelInfoEntry(reg_MI, all_reg_names,
                                      epmaps.values()))

    if is_hybrid:
        # build MIs, map events, etc.
        the_model = Model.HybridModel({'name': model_name,
                            'modelInfo': makeModelInfo(reg_info_list)})
    else:
        the_model = reg_models.values()[0]
    return the_model



def get_regime_model(r, fns, sigs):
    # Quantity types
    declare_vars = []

    # only one regime assumed for now
    for dx in r.time_derivatives:
        atoms = dx.lhs_atoms
        assert 't' in atoms
        assert len(atoms) == 2
        for a in atoms:
            if a != 't':
                vname = a
        varspec = QuantSpec('D_'+vname, dx.rhs_str)
        fns_used = intersect(varspec.freeSymbols, fns)
        for f in fns_used:
            fi = varspec.parser.tokenized.index(f)
            new_defstr = ''.join(varspec[:fi+1]) + '(' + ','.join(sigs[f]) + ')' + ''.join(varspec[fi+1:])
            varspec = QuantSpec('D_'+vname, new_defstr)
        # no domains known
        declare_vars.append(Var(str(varspec), name=vname, specType='RHSfuncSpec'))

    return declare_vars


##################################################
#  CODE NOTES FOR FUTURE DEVELOPMENT
##################################################
