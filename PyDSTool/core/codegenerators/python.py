#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from PyDSTool.common import invertMap, intersect, remain, concatStrDict, makeUniqueFn
from PyDSTool.parseUtils import _indentstr, convertPowers, makeParList, parseMatrixStrToDictStr, count_sep
from PyDSTool.Symbolic import QuantSpec
from PyDSTool.utils import compareList, info

from .base import _processReused


class PythonCodeGenerator(object):

    def generate_aux(self, fspec, pytarget=False):
        if pytarget:
            assert fspec.targetlang == 'python', \
                'Wrong target language for this call'
        auxnames = fspec._auxfnspecs.keys()
        # User aux fn interface
        uafi = {}
        # protectednames = auxnames + fspec._protected_mathnames + \
        #                  fspec._protected_randomnames + \
        #                  fspec._protected_scipynames + \
        #                  fspec._protected_specialfns + \
        #                  ['abs', 'and', 'or', 'not', 'True', 'False']
        # Deal with built-in auxiliary functions (don't make their names unique)
        # In this version, the textual code here doesn't get executed. Only
        # the function names in the second position of the tuple are needed.
        # Later, the text will probably be removed.
        auxfns = {}
        auxfns['globalindepvar'] = \
            ("def _auxfn_globalindepvar(ds, parsinps, t):\n"
             + _indentstr
             + "return ds.globalt0 + t", '_auxfn_globalindepvar')
        auxfns['initcond'] = \
            ("def _auxfn_initcond(ds, parsinps, varname):\n"
             + _indentstr
             + "return ds.initialconditions[varname]", '_auxfn_initcond')
        auxfns['heav'] = \
            ("def _auxfn_heav(ds, parsinps, x):\n" + _indentstr
             + "if x>0:\n" + 2 * _indentstr
             + "return 1\n" + _indentstr + "else:\n"
             + 2 * _indentstr + "return 0", '_auxfn_heav')
        auxfns['if'] = \
            ("def _auxfn_if(ds, parsinps, c, e1, e2):\n"
             + _indentstr + "if c:\n" + 2 * _indentstr
             + "return e1\n" + _indentstr
             + "else:\n" + 2 * _indentstr + "return e2", '_auxfn_if')
        auxfns['getindex'] = \
            ("def _auxfn_getindex(ds, parsinps, varname):\n"
             + _indentstr
             + "return ds._var_namemap[varname]", '_auxfn_getindex')
        auxfns['getbound'] = \
            ("def _auxfn_getbound(ds, parsinps, name, bd):\n"
             + _indentstr + "try:\n"
             + 2 * _indentstr + "return ds.xdomain[name][bd]\n"
             + _indentstr + "except KeyError:\n" + 2 * _indentstr
             + "try:\n" + 3 * _indentstr
             + "return ds.pdomain[name][bd]\n" + 2 * _indentstr
             + "except KeyError, e:\n" + 3 * _indentstr
             + "print 'Invalid var / par name %s'%name,\n"
             + 3 * _indentstr + "print 'or bounds not well defined:'\n"
             + 3 * _indentstr + "print ds.xdomain, ds.pdomain\n"
             + 3 * _indentstr + "raise (RuntimeError, e)",
             '_auxfn_getbound')
        # the internal functions may be used by user-defined functions,
        # so need them to be accessible to processTokens when parsing
        fspec._pyauxfns = auxfns
        # add the user-defined function names for cross-referencing checks
        # (without their definitions)
        for auxname in auxnames:
            fspec._pyauxfns[auxname] = None
        # don't process the built-in functions -> unique fns because
        # they are global definitions existing throughout the
        # namespace
        fspec._protected_auxnames.extend(['Jacobian', 'Jacobian_pars'])
        # protected names are the names that must not be used for
        # user-specified auxiliary fn arguments
        protectednames = fspec.pars + fspec.inputs \
            + ['abs', 'pow', 'and', 'or', 'not', 'True', 'False'] \
            + fspec._protected_auxnames + auxnames \
            + fspec._protected_scipynames + fspec._protected_specialfns \
            + fspec._protected_macronames + fspec._protected_mathnames \
            + fspec._protected_randomnames + fspec._protected_reusenames
        # checks for user-defined auxiliary fns
        # name map for fixing inter-auxfn references
        auxfn_namemap = {}
        specials_base = fspec.pars + fspec._protected_auxnames \
            + ['abs', 'pow', 'and', 'or', 'not', 'True', 'False'] \
            + auxnames + fspec._protected_scipynames \
            + fspec._protected_specialfns \
            + fspec._protected_macronames + fspec._protected_mathnames \
            + fspec._protected_randomnames + fspec._protected_reusenames
        for auxname in auxnames:
            auxinfo = fspec._auxfnspecs[auxname]
            try:
                if len(auxinfo) != 2:
                    raise ValueError('auxinfo tuple must be of length 2')
            except TypeError:
                raise TypeError('fnspecs argument must contain pairs')
            # auxinfo[0] = tuple or list of parameter names
            # auxinfo[1] = string containing body of function definition
            assert isinstance(auxinfo[0], list), ('aux function arguments '
                                                  'must be given as a list')
            assert isinstance(auxinfo[1], str), ('aux function specification '
                                                 'must be a string '
                                                 'of the function code')
            # Process Jacobian functions, etc., specially, if present
            if auxname == 'Jacobian':
                if not compareList(auxinfo[0], ['t'] + fspec.vars):
                    print ['t'] + fspec.vars
                    print "Auxinfo =", auxinfo[0]
                    raise ValueError(
                        "Invalid argument list given in Jacobian.")
                auxparlist = ["t", "x", "parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t", "x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = fspec.vars
                specvars.sort()
                specdict = {}.fromkeys(specvars)
                if len(specvars) == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in Jacobian for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in Jacobian for 1D system"
                    specdict[specvars[0]] = auxstr
                else:
                    specdict = parseMatrixStrToDictStr(auxstr, specvars)
                reusestr, body_processed_dict = self._processReusedPy(
                    fspec, specvars,
                    specdict,
                    specials=specials + specials_base)
                body_processed = self._specStrParse(fspec, specvars,
                                                    body_processed_dict, 'xjac',
                                                    specials=specials + specials_base)
                auxstr_py = self._generate_fun(fspec, '_auxfn_Jac',
                                               reusestr + body_processed,
                                               'xjac', specvars)
                # check Jacobian
                m = n = len(specvars)
                specdict_check = {}.fromkeys(specvars)
                for specname in specvars:
                    temp = body_processed_dict[specname]
                    specdict_check[specname] = \
                        count_sep(temp.replace("[", "").replace("]", "")) + 1
                body_processed = ""
                for row in range(m):
                    if specdict_check[specvars[row]] != n:
                        print "Row %i: " % m, specdict[specvars[row]]
                        print "Found length %i" % specdict_check[specvars[row]]
                        raise ValueError("Jacobian should be %sx%s" % (m, n))
            elif auxname == 'Jacobian_pars':
                if not compareList(auxinfo[0], ['t'] + fspec.vars):
                    print ['t'] + fspec.vars
                    print "Auxinfo =", auxinfo[0]
                    raise ValueError(
                        "Invalid argument list given in Jacobian.")
                auxparlist = ["t", "x", "parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t", "x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = fspec.vars
                specvars.sort()
                specdict = {}.fromkeys(fspec.vars)
                if len(specvars) == len(fspec.vars) == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in Jacobian for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in Jacobian for 1D system"
                    specdict[specvars[0]] = auxstr
                else:
                    specdict = parseMatrixStrToDictStr(auxstr, fspec.vars)
                reusestr, body_processed_dict = self._processReusedPy(
                    fspec, fspec.vars,
                    specdict,
                    specials=specials + specials_base)
                body_processed = self._specStrParse(fspec, fspec.vars,
                                                    body_processed_dict, 'pjac',
                                                    specials=specials + specials_base)
                auxstr_py = self._generate_fun(fspec, '_auxfn_Jac_p',
                                               reusestr + body_processed,
                                               'pjac', fspec.vars)
                # check Jacobian
                n = len(specvars)
                m = len(fspec.vars)
                specdict_check = {}.fromkeys(fspec.vars)
                for specname in fspec.vars:
                    temp = body_processed_dict[specname]
                    specdict_check[specname] = \
                        count_sep(temp.replace("[", "").replace("]", "")) + 1
                body_processed = ""
                for row in range(m):
                    try:
                        if specdict_check[fspec.vars[row]] != n:
                            print "Row %i: " % m, specdict[fspec.vars[row]]
                            print "Found length %i" % specdict_check[fspec.vars[row]]
                            raise ValueError(
                                "Jacobian w.r.t. pars should be %sx%s" % (m, n))
                    except IndexError:
                        print "\nFound:\n"
                        info(specdict)
                        raise ValueError(
                            "Jacobian w.r.t. pars should be %sx%s" % (m, n))
            elif auxname == 'massMatrix':
                if not compareList(auxinfo[0], ['t'] + fspec.vars):
                    print ['t'] + fspec.vars
                    print "Auxinfo =", auxinfo[0]
                    raise ValueError(
                        "Invalid argument list given in Mass Matrix.")
                auxparlist = ["t", "x", "parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t", "x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = fspec.vars
                specvars.sort()
                specdict = {}.fromkeys(specvars)
                if len(specvars) == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in mass matrix for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in mass matrix for 1D system"
                    specdict[specvars.values()[0]] = auxstr
                else:
                    specdict = parseMatrixStrToDictStr(auxstr, specvars)
                reusestr, body_processed_dict = self._processReusedPy(
                    fspec, specvars,
                    specdict,
                    specials=specials + specials_base)
                body_processed = self._specStrParse(fspec, specvars,
                                                    body_processed_dict, 'xmat',
                                                    specials=specials + specials_base)
                auxstr_py = self._generate_fun(fspec, '_auxfn_massMatrix',
                                               reusestr + body_processed,
                                               'xmat', specvars)
                # check matrix
                m = n = len(specvars)
                specdict_check = {}.fromkeys(specvars)
                for specname in specvars:
                    specdict_check[specname] = 1 + \
                        count_sep(
                            body_processed_dict[specname].replace("[", "").replace("]", ""))
                body_processed = ""
                for row in range(m):
                    if specdict_check[specvars[row]] != n:
                        print "Row %i: " % m, specdict[specvars[row]]
                        print "Found length %i" % specdict_check[specvars[row]]
                        raise ValueError(
                            "Mass matrix should be %sx%s" % (m, n))
            else:
                user_parstr = makeParList(auxinfo[0])
                # `parsinps` is always added to allow reference to own
                # parameters
                if user_parstr == '':
                    # no arguments, user calls as fn()
                    auxparstr = 'parsinps'
                else:
                    auxparstr = 'parsinps, ' + user_parstr
                auxstr_py = 'def _auxfn_' + auxname + '(ds, ' + auxparstr \
                            + '):\n'
                auxparlist = auxparstr.replace(" ", "").split(",")
                badparnames = intersect(auxparlist,
                                        remain(protectednames, auxnames))
                if badparnames != []:
                    print "Bad parameter names in auxiliary function", \
                        auxname, ":", badparnames
                    # print auxinfo[0]
                    # print auxparlist
                    raise ValueError("Cannot use protected names (including"
                                     " globally visible system parameters for auxiliary "
                                     "function arguments")
                # special symbols to allow in parsing function body
                specials = auxparlist
                specials.remove('parsinps')
                illegalterms = remain(fspec.vars + fspec.auxvars, specials)
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                reusestr, body_processed_dict = self._processReusedPy(
                    fspec, [auxname],
                    {auxname: auxstr},
                    specials=specials + specials_base,
                    dovars=False,
                    illegal=illegalterms)
                body_processed = self._specStrParse(fspec, [auxname],
                                                    body_processed_dict,
                                                    specials=specials +
                                                    specials_base,
                                                    dovars=False,
                                                    noreturndefs=True,
                                                    illegal=illegalterms)
                auxstr_py += reusestr + _indentstr + 'return ' \
                    + body_processed
            # syntax validation done in makeUniqueFn
            try:
                auxfns[auxname] = makeUniqueFn(auxstr_py)
                # Note: this automatically updates fspec._pyauxfns too
            except Exception:
                print 'Error in supplied auxiliary spec dictionary code'
                raise
            auxfn_namemap['ds.' + auxname] = 'ds.' + auxfns[auxname][1]
            # prepare user-interface wrapper function (not method)
            if specials == [''] or specials == []:
                fn_args = ''
            else:
                fn_args = ',' + ','.join(specials)
            fn_elts = ['def ', auxname, '(self', fn_args,
                       ',__parsinps__=None):\n\t', 'if __parsinps__ is None:\n\t\t',
                       '__parsinps__=self.map_ixs(self.genref)\n\t',
                       'return self.genref.', auxfns[auxname][1],
                       '(__parsinps__', fn_args, ')\n']
            uafi[auxname] = ''.join(fn_elts)
        # resolve inter-auxiliary function references
        for auxname, auxspec in auxfns.iteritems():
            dummyQ = QuantSpec('dummy', auxspec[0], preserveSpace=True,
                               treatMultiRefs=False)
            dummyQ.mapNames(auxfn_namemap)
            auxfns[auxname] = (dummyQ(), auxspec[1])
        if pytarget:
            fspec.auxfns = auxfns
        # keep _pyauxfns handy for users to access python versions of functions
        # from python, even using non-python target languages
        #
        # Changes to auxfns was already changing fspec._pyauxfns so the following line
        # is not needed
        # fspec._pyauxfns.update(auxfns)  # same thing if pytarget==True
        fspec._user_auxfn_interface = uafi
        fspec._protected_auxnames.extend(auxnames)

    def generate_spec(self, fspec):
        assert fspec.targetlang == 'python', ('Wrong target language for this'
                                              ' call')
        assert fspec.varspecs != {}, 'varspecs attribute must be defined'
        specnames_unsorted = fspec.varspecs.keys()
        _vbfs_inv = invertMap(fspec._varsbyforspec)
        # Process state variable specifications
        if len(_vbfs_inv) > 0:
            specname_vars = []
            specname_auxvars = []
            for varname in fspec.vars:
                # check if varname belongs to a for macro grouping in
                # fspec.varspecs
                if varname not in specname_vars:
                    specname_vars.append(varname)
            for varname in fspec.auxvars:
                # check if varname belongs to a for macro grouping in
                # fspec.varspecs
                if varname not in specname_auxvars:
                    specname_auxvars.append(varname)
        else:
            specname_vars = intersect(fspec.vars, specnames_unsorted)
            specname_auxvars = intersect(fspec.auxvars, specnames_unsorted)
        specname_vars.sort()
        for vn, vs in fspec.varspecs.items():
            if any([pt in vs for pt in ('^', '**')]):
                fspec.varspecs[vn] = convertPowers(vs, 'pow')
        fspec.vars.sort()
        reusestr, specupdated = self._processReusedPy(fspec, specname_vars,
                                                      fspec.varspecs)
        fspec.varspecs.update(specupdated)
        temp = self._specStrParse(fspec, specname_vars, fspec.varspecs, 'xnew')
        specstr_py = self._generate_fun(
            fspec, '_specfn', reusestr + temp, 'xnew',
            specname_vars, docodeinserts=True)
        # Process auxiliary variable specifications
        specname_auxvars.sort()
        assert fspec.auxvars == specname_auxvars, \
            ('Mismatch between declared auxiliary'
             ' variable names and varspecs keys')
        reusestraux, specupdated = self._processReusedPy(
            fspec, specname_auxvars,
            fspec.varspecs)
        fspec.varspecs.update(specupdated)
        tempaux = self._specStrParse(
            fspec, specname_auxvars, fspec.varspecs, 'auxvals')
        auxspecstr_py = self._generate_fun(
            fspec, '_auxspecfn', reusestraux + tempaux,
            'auxvals', specname_auxvars,
            docodeinserts=True)
        try:
            spec_info = makeUniqueFn(specstr_py)
        except SyntaxError:
            print "Syntax error in specification:\n", specstr_py
            raise
        try:
            auxspec_info = makeUniqueFn(auxspecstr_py)
        except SyntaxError:
            print "Syntax error in auxiliary spec:\n", auxspecstr_py
            raise
        fspec.spec = spec_info
        fspec.auxspec = auxspec_info

    def _generate_fun(self, fspec, name, specstr, resname, specnames,
                      docodeinserts=False):
        # Set up function header
        retstr = 'def ' + name + \
            '(ds, t, x, parsinps):\n'  # print t, x, parsinps\n'
        # add arbitrary code inserts, if present and option is switched on
        # (only used for vector field definitions)
        lstart = len(fspec.codeinserts['start'])
        lend = len(fspec.codeinserts['end'])
        if docodeinserts:
            if lstart > 0:
                start_code = self._specStrParse(fspec, ['inserts'],
                                                {'inserts': fspec.codeinserts[
                                                    'start']}, '',
                                                noreturndefs=True, ignoreothers=True,
                                                doing_inserts=True)
            else:
                start_code = ''
            if lend > 0:
                end_code = self._specStrParse(fspec, ['inserts'],
                                              {'inserts': fspec.codeinserts[
                                                  'end']}, '',
                                              noreturndefs=True, ignoreothers=True,
                                              doing_inserts=True)
            else:
                end_code = ''
        else:
            start_code = end_code = ''
        retstr += start_code + specstr + end_code
        # Add the return line to the function
        if len(specnames) == 1:
            retstr += _indentstr + 'return array([' + resname + '0])\n'
        else:
            retstr += _indentstr + 'return array([' \
                + makeParList(range(len(specnames)), resname) + '])\n'
        return retstr

    def _specStrParse(self, fspec, specnames, specdict, resname='', specials=[],
        dovars=True, dopars=True, doinps=True,
        noreturndefs=False, forexternal=False, illegal=[],
            ignoreothers=False, doing_inserts=False):
        # use 'noreturndefs' switch if calling this function just to "parse"
        # a spec string for other purposes, e.g. for using in an event setup
        # or an individual auxiliary function spec
        assert isinstance(specnames, list), "specnames must be a list"
        if noreturndefs or forexternal:
            assert len(specnames) == 1, ("can only pass a single specname for "
                                         "'forexternal' or 'noreturndefs' options")
        allnames = fspec.vars + fspec.pars + fspec.inputs + fspec.auxvars \
            + ['abs', 'and', 'or', 'not', 'True', 'False'] \
            + fspec._protected_auxnames \
            + fspec._protected_scipynames + fspec._protected_specialfns \
            + fspec._protected_macronames + fspec._protected_mathnames \
            + fspec._protected_randomnames + fspec._protected_reusenames
        allnames = remain(allnames, illegal)
        if dovars:
            if forexternal:
                var_arrayixstr = dict(zip(fspec.vars,
                                          ["'" + v + "'" for v in fspec.vars]))
                aux_arrayixstr = dict(zip(fspec.auxvars,
                                          ["'" + v + "'" for v in fspec.auxvars]))
            else:
                var_arrayixstr = dict(zip(fspec.vars, map(lambda i: str(i),
                                                          range(len(fspec.vars)))))
                aux_arrayixstr = dict(zip(fspec.auxvars, map(lambda i: str(i),
                                                             range(len(fspec.auxvars)))))
        else:
            var_arrayixstr = {}
            aux_arrayixstr = {}
        # ODE solvers typically don't recognize external inputs
        # so they have to be lumped in with the parameters
        # argument `parsinps` holds the combined pars and inputs
        if dopars:
            if forexternal:
                if doinps:
                    # parsinps_names is pars and inputs, each sorted
                    # *individually*
                    parsinps_names = fspec.pars + fspec.inputs
                else:
                    parsinps_names = fspec.pars
                # for external calls we want parname -> 'parname'
                parsinps_arrayixstr = dict(zip(parsinps_names,
                                               ["'" + pn + "'" for pn in parsinps_names]))
            else:
                if doinps:
                    # parsinps_names is pars and inputs, each sorted
                    # *individually*
                    parsinps_names = fspec.pars + fspec.inputs
                else:
                    parsinps_names = fspec.pars
                parsinps_arrayixstr = dict(zip(parsinps_names,
                                               map(lambda i: str(i),
                                                   range(len(parsinps_names)))))
        else:
            parsinps_names = []
            parsinps_arrayixstr = {}
        specialtokens = remain(allnames, specials) + ['(', 't'] \
            + remain(specials, ['t'])
        specstr_lang = ''
        specname_count = 0
        for specname in specnames:
            specstr = specdict[specname]
            assert type(
                specstr) == str, "Specification for %s was not a string" % specname
            if not noreturndefs:
                specstr_lang += _indentstr + \
                    resname + str(specname_count) + ' = '
            specname_count += 1
            specstr_lang += fspec.processTokens(allnames, specialtokens,
                                                specstr, var_arrayixstr,
                                                aux_arrayixstr, parsinps_names,
                                                parsinps_arrayixstr, specname, ignoreothers,
                                                doing_inserts)
            if not noreturndefs or not forexternal:
                specstr_lang += '\n'  # prepare for next line
        return specstr_lang

    def _processReusedPy(self, fspec, specnames, specdict, specials=[],
                         dovars=True, dopars=True, doinps=True, illegal=[]):
        """Process reused subexpression terms for Python code."""

        reused, specupdated, new_protected, order = _processReused(specnames,
                                                                   specdict,
                                                                   fspec.reuseterms,
                                                                   _indentstr)
        fspec._protected_reusenames = new_protected
        # symbols to parse are at indices 2 and 4 of 'reused' dictionary
        reusedParsed = self._parseReusedTermsPy(fspec, reused, [2, 4],
                                                specials=specials, dovars=dovars,
                                                dopars=dopars, doinps=doinps,
                                                illegal=illegal)
        reusedefs = {}.fromkeys(new_protected)
        for _, deflist in reusedParsed.iteritems():
            for d in deflist:
                reusedefs[d[2]] = d
        return (concatStrDict(reusedefs, intersect(order, reusedefs.keys())),
                specupdated)

    def _parseReusedTermsPy(self, fspec, d, symbol_ixs, specials=[],
                            dovars=True, dopars=True, doinps=True, illegal=[]):
        """Process dictionary of reused term definitions (in spec syntax)."""
        # ... to parse special symbols to actual Python.
        # expect symbols to be processed at d list's entries given in
        # symbol_ixs.
        allnames = fspec.vars + fspec.pars + fspec.inputs + fspec.auxvars \
            + ['abs'] + fspec._protected_auxnames \
            + fspec._protected_scipynames + fspec._protected_specialfns \
            + fspec._protected_macronames + fspec._protected_mathnames \
            + fspec._protected_randomnames + fspec._protected_reusenames
        allnames = remain(allnames, illegal)
        if dovars:
            var_arrayixstr = dict(zip(fspec.vars, map(lambda i: str(i),
                                                      range(len(fspec.vars)))))
            aux_arrayixstr = dict(zip(fspec.auxvars, map(lambda i: str(i),
                                                         range(len(fspec.auxvars)))))
        else:
            var_arrayixstr = {}
            aux_arrayixstr = {}
        if dopars:
            if doinps:
                # parsinps_names is pars and inputs, each sorted
                # *individually*
                parsinps_names = fspec.pars + fspec.inputs
            else:
                parsinps_names = fspec.pars
            parsinps_arrayixstr = dict(zip(parsinps_names,
                                           map(lambda i: str(i),
                                               range(len(parsinps_names)))))
        else:
            parsinps_names = []
            parsinps_arrayixstr = {}
        specialtokens = remain(allnames, specials) + ['(', 't'] + specials
        for specname, itemlist in d.iteritems():
            listix = -1
            for strlist in itemlist:
                listix += 1
                if strlist == []:
                    continue
                if len(strlist) < max(symbol_ixs):
                    raise ValueError("Symbol indices out of range in "
                                     "call to _parseReusedTermsPy")
                for ix in symbol_ixs:
                    symbol = strlist[ix]
                    parsedsymbol = fspec.processTokens(allnames,
                                                       specialtokens, symbol,
                                                       var_arrayixstr, aux_arrayixstr,
                                                       parsinps_names, parsinps_arrayixstr,
                                                       specname)
                    # must strip possible trailing whitespace!
                    d[specname][listix][ix] = parsedsymbol.strip()
        return d
