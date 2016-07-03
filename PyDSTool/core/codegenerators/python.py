#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from PyDSTool.common import invertMap, intersect, remain, concatStrDict, makeUniqueFn
from PyDSTool.parseUtils import _indentstr, convertPowers, makeParList, parseMatrixStrToDictStr, count_sep
from PyDSTool.Symbolic import QuantSpec
from PyDSTool.utils import compareList, info

from .base import _processReused, CodeGenerator
import six

PYTHON_FUNCTION_TEMPLATE = """
def {name}(ds, t, x, parsinps):
{start}
{spec}
{end}
    return array([{result}])
"""


class Python(CodeGenerator):

    def __init__(self, *args, **kwargs):
        super(Python, self).__init__(*args, **kwargs)
        self._fn_template = PYTHON_FUNCTION_TEMPLATE

    def generate_aux(self):
        auxnames = list(self.fspec._auxfnspecs.keys())
        # User aux fn interface
        uafi = {}
        # protectednames = auxnames + self.fspec._protected_mathnames + \
        #                  self.fspec._protected_randomnames + \
        #                  self.fspec._protected_scipynames + \
        #                  self.fspec._protected_specialfns + \
        #                  ['abs', 'and', 'or', 'not', 'True', 'False']
        auxfns = self.builtin_aux
        # the internal functions may be used by user-defined functions,
        # so need them to be accessible to processTokens when parsing
        self.fspec._pyauxfns = auxfns
        # add the user-defined function names for cross-referencing checks
        # (without their definitions)
        for auxname in auxnames:
            self.fspec._pyauxfns[auxname] = None
        # don't process the built-in functions -> unique fns because
        # they are global definitions existing throughout the
        # namespace
        self.fspec._protected_auxnames.extend(['Jacobian', 'Jacobian_pars'])
        # protected names are the names that must not be used for
        # user-specified auxiliary fn arguments
        protectednames = self.fspec.pars + self.fspec.inputs \
            + ['abs', 'pow', 'and', 'or', 'not', 'True', 'False'] \
            + self.fspec._protected_auxnames + auxnames \
            + self.fspec._protected_numpynames \
            + self.fspec._protected_scipynames \
            + self.fspec._protected_specialfns \
            + self.fspec._protected_macronames \
            + self.fspec._protected_mathnames \
            + self.fspec._protected_randomnames \
            + self.fspec._protected_reusenames
        # checks for user-defined auxiliary fns
        # name map for fixing inter-auxfn references
        auxfn_namemap = {}
        specials_base = self.fspec.pars + self.fspec._protected_auxnames \
            + ['abs', 'pow', 'and', 'or', 'not', 'True', 'False'] \
            + auxnames + self.fspec._protected_scipynames \
            + self.fspec._protected_numpynames + self.fspec._protected_specialfns \
            + self.fspec._protected_macronames + self.fspec._protected_mathnames \
            + self.fspec._protected_randomnames + self.fspec._protected_reusenames
        for auxname in auxnames:
            auxinfo = self.fspec._auxfnspecs[auxname]
            try:
                if len(auxinfo) != 2:
                    raise ValueError('auxinfo tuple must be of length 2')
            except TypeError:
                raise TypeError('fnspecs argument must contain pairs')
            # auxinfo[0] = tuple or list of parameter names
            # auxinfo[1] = string containing body of function definition
            assert isinstance(auxinfo[0], list), ('aux function arguments '
                                                  'must be given as a list')
            assert isinstance(auxinfo[1], six.string_types), \
                   ('aux function specification '
                    'must be a string of the function code')
            # Process Jacobian functions, etc., specially, if present
            if auxname == 'Jacobian':
                if not compareList(auxinfo[0], ['t'] + self.fspec.vars):
                    print(['t'] + self.fspec.vars)
                    print("Auxinfo =" + str(auxinfo[0]))
                    raise ValueError(
                        "Invalid argument list given in Jacobian.")
                auxparlist = ["t", "x", "parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t", "x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = self.fspec.vars
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
                    specvars,
                    specdict,
                    specials=specials + specials_base)
                body_processed = self._specStrParse(specvars,
                                                    body_processed_dict, 'xjac',
                                                    specials=specials + specials_base)
                auxstr_py = self._generate_fun('_auxfn_Jac',
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
                        print("Row %i: " % m + specdict[specvars[row]])
                        print("Found length %i" % specdict_check[specvars[row]])
                        raise ValueError("Jacobian should be %sx%s" % (m, n))
            elif auxname == 'Jacobian_pars':
                if not compareList(auxinfo[0], ['t'] + self.fspec.pars):
                    print(['t'] + self.fspec.pars)
                    print("Auxinfo =" + str(auxinfo[0]))
                    raise ValueError(
                        "Invalid argument list given in Jacobian.")
                auxparlist = ["t", "x", "parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t", "x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = self.fspec.pars
                specvars.sort()
                specdict = {}.fromkeys(self.fspec.vars)
                if len(specvars) == len(self.fspec.vars) == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in Jacobian for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in Jacobian for 1D system"
                    specdict[specvars[0]] = auxstr
                else:
                    specdict = parseMatrixStrToDictStr(auxstr, self.fspec.vars)
                reusestr, body_processed_dict = self._processReusedPy(
                    self.fspec.vars,
                    specdict,
                    specials=specials + specials_base)
                body_processed = self._specStrParse(self.fspec.vars,
                                                    body_processed_dict, 'pjac',
                                                    specials=specials + specials_base)
                auxstr_py = self._generate_fun('_auxfn_Jac_p',
                                               reusestr + body_processed,
                                               'pjac', self.fspec.vars)
                # check Jacobian
                n = len(specvars)
                m = len(self.fspec.vars)
                specdict_check = {}.fromkeys(self.fspec.vars)
                for specname in self.fspec.vars:
                    temp = body_processed_dict[specname]
                    specdict_check[specname] = \
                        count_sep(temp.replace("[", "").replace("]", "")) + 1
                body_processed = ""
                for row in range(m):
                    try:
                        if specdict_check[self.fspec.vars[row]] != n:
                            print("Row %i: " % m + specdict[self.fspec.vars[row]])
                            print("Found length %i" % specdict_check[self.fspec.vars[row]])
                            raise ValueError(
                                "Jacobian w.r.t. pars should be %sx%s" % (m, n))
                    except IndexError:
                        print("\nFound:\n")
                        info(specdict)
                        raise ValueError(
                            "Jacobian w.r.t. pars should be %sx%s" % (m, n))
            elif auxname == 'massMatrix':
                if not compareList(auxinfo[0], ['t'] + self.fspec.vars):
                    print(['t'] + self.fspec.vars)
                    print("Auxinfo =" + str(auxinfo[0]))
                    raise ValueError(
                        "Invalid argument list given in Mass Matrix.")
                auxparlist = ["t", "x", "parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t", "x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = self.fspec.vars
                specvars.sort()
                specdict = {}.fromkeys(specvars)
                if len(specvars) == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in mass matrix for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in mass matrix for 1D system"
                    specdict[list(specvars.values())[0]] = auxstr
                else:
                    specdict = parseMatrixStrToDictStr(auxstr, specvars)
                reusestr, body_processed_dict = self._processReusedPy(
                    specvars,
                    specdict,
                    specials=specials + specials_base)
                body_processed = self._specStrParse(specvars,
                                                    body_processed_dict, 'xmat',
                                                    specials=specials + specials_base)
                auxstr_py = self._generate_fun('_auxfn_massMatrix',
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
                        print("Row %i: " % m + specdict[specvars[row]])
                        print("Found length %i" % specdict_check[specvars[row]])
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
                    print("Bad parameter names in auxiliary function %s: %r" % (auxname, badparnames))
                    # print(str(auxinfo[0]))
                    # print(str(auxparlist))
                    raise ValueError("Cannot use protected names (including"
                                     " globally visible system parameters for auxiliary "
                                     "function arguments")
                # special symbols to allow in parsing function body
                specials = auxparlist
                specials.remove('parsinps')
                illegalterms = remain(self.fspec.vars + self.fspec.auxvars, specials)
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                reusestr, body_processed_dict = self._processReusedPy(
                    [auxname],
                    {auxname: auxstr},
                    specials=specials + specials_base,
                    dovars=False,
                    illegal=illegalterms)
                body_processed = self._specStrParse([auxname],
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
                # Note: this automatically updates self.fspec._pyauxfns too
            except Exception:
                print('Error in supplied auxiliary spec dictionary code')
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
        for auxname, auxspec in auxfns.items():
            dummyQ = QuantSpec('dummy', auxspec[0], preserveSpace=True,
                               treatMultiRefs=False)
            dummyQ.mapNames(auxfn_namemap)
            auxfns[auxname] = (dummyQ(), auxspec[1])
        self.fspec._user_auxfn_interface = uafi
        self.fspec._protected_auxnames.extend(auxnames)
        return auxfns

    @property
    def builtin_aux(self):
        if not hasattr(self, '_builtin_aux'):
            self._builtin_aux = self.__generate_builtin_aux()

        return self._builtin_aux

    def __generate_builtin_aux(self):
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
             + "except KeyError as e:\n" + 3 * _indentstr
             + "print('Invalid var / par name %s'%name)\n"
             + 3 * _indentstr + "print('or bounds not well defined:')\n"
             + 3 * _indentstr + "print('%r %r' % (ds.xdomain, ds.pdomain))\n"
             + 3 * _indentstr + "raise RuntimeError(e)",
             '_auxfn_getbound')

        return auxfns

    def generate_spec(self):
        assert self.fspec.targetlang == 'python', ('Wrong target language for this'
                                              ' call')
        assert self.fspec.varspecs != {}, 'varspecs attribute must be defined'
        specnames_unsorted = list(self.fspec.varspecs.keys())
        _vbfs_inv = invertMap(self.fspec._varsbyforspec)
        # Process state variable specifications
        if len(_vbfs_inv) > 0:
            specname_vars = []
            specname_auxvars = []
            for varname in self.fspec.vars:
                # check if varname belongs to a for macro grouping in
                # self.fspec.varspecs
                if varname not in specname_vars:
                    specname_vars.append(varname)
            for varname in self.fspec.auxvars:
                # check if varname belongs to a for macro grouping in
                # self.fspec.varspecs
                if varname not in specname_auxvars:
                    specname_auxvars.append(varname)
        else:
            specname_vars = intersect(self.fspec.vars, specnames_unsorted)
            specname_auxvars = intersect(self.fspec.auxvars, specnames_unsorted)
        specname_vars.sort()
        for vn, vs in list(self.fspec.varspecs.items()):
            if any([pt in vs for pt in ('^', '**')]):
                self.fspec.varspecs[vn] = convertPowers(vs, 'pow')
        self.fspec.vars.sort()
        reusestr, specupdated = self._processReusedPy(specname_vars,
                                                      self.fspec.varspecs)
        self.fspec.varspecs.update(specupdated)
        temp = self._specStrParse(specname_vars, self.fspec.varspecs, 'xnew')
        specstr_py = self._generate_fun(
            '_specfn', reusestr + temp, 'xnew',
            specname_vars, docodeinserts=True)
        # Process auxiliary variable specifications
        specname_auxvars.sort()
        assert self.fspec.auxvars == specname_auxvars, \
            ('Mismatch between declared auxiliary'
             ' variable names and varspecs keys')
        reusestraux, specupdated = self._processReusedPy(
            specname_auxvars,
            self.fspec.varspecs)
        self.fspec.varspecs.update(specupdated)
        tempaux = self._specStrParse(
            specname_auxvars, self.fspec.varspecs, 'auxvals')
        auxspecstr_py = self._generate_fun(
            '_auxspecfn', reusestraux + tempaux,
            'auxvals', specname_auxvars,
            docodeinserts=True)
        try:
            spec_info = makeUniqueFn(specstr_py)
        except SyntaxError:
            print("Syntax error in specification:\n" + specstr_py)
            raise
        try:
            auxspec_info = makeUniqueFn(auxspecstr_py)
        except SyntaxError:
            print("Syntax error in auxiliary spec:\n" + auxspecstr_py)
            raise
        self.fspec.spec = spec_info
        self.fspec.auxspec = auxspec_info

    def _generate_fun(self, name, specstr, resname, specnames, docodeinserts=False):
        """Generate string with Python code for function `name`"""

        fdef = self._fn_template.format(
            name=name,
            start=self._format_user_code(self.opts['start']) if docodeinserts else '',
            spec=specstr,
            end=self._format_user_code(self.opts['end']) if docodeinserts else '',
            result=makeParList(range(len(specnames)), resname)
        )
        return '\n'.join([s for s in fdef.split('\n') if s]) + '\n'

    def _specStrParse(self, specnames, specdict, resname='', specials=[],
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
        allnames = self.fspec.vars + self.fspec.pars + self.fspec.inputs + self.fspec.auxvars \
            + ['abs', 'and', 'or', 'not', 'True', 'False'] \
            + self.fspec._protected_auxnames \
            + self.fspec._protected_scipynames \
            + self.fspec._protected_numpynames \
            + self.fspec._protected_specialfns \
            + self.fspec._protected_macronames \
            + self.fspec._protected_mathnames \
            + self.fspec._protected_randomnames \
            + self.fspec._protected_symbolicnames \
            + self.fspec._protected_reusenames
        allnames = remain(allnames, illegal)
        if dovars:
            if forexternal:
                var_arrayixstr = dict(zip(self.fspec.vars,
                                          ["'" + v + "'" for v in self.fspec.vars]))
                aux_arrayixstr = dict(zip(self.fspec.auxvars,
                                          ["'" + v + "'" for v in self.fspec.auxvars]))
            else:
                var_arrayixstr = dict([(v,str(i)) for i, v in enumerate(self.fspec.vars)])
                aux_arrayixstr = dict([(v,str(i)) for i, v in enumerate(self.fspec.auxvars)])
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
                    parsinps_names = self.fspec.pars + self.fspec.inputs
                else:
                    parsinps_names = self.fspec.pars
                # for external calls we want parname -> 'parname'
                parsinps_arrayixstr = dict(zip(parsinps_names,
                                               ["'" + pn + "'" for pn in parsinps_names]))
            else:
                if doinps:
                    # parsinps_names is pars and inputs, each sorted
                    # *individually*
                    parsinps_names = self.fspec.pars + self.fspec.inputs
                else:
                    parsinps_names = self.fspec.pars
                parsinps_arrayixstr = dict([(p,str(i)) for i, p in enumerate(parsinps_names)])
        else:
            parsinps_names = []
            parsinps_arrayixstr = {}
        specialtokens = remain(allnames, specials) + ['(', 't'] \
            + remain(specials, ['t'])
        specstr_lang = ''
        specname_count = 0
        for specname in specnames:
            specstr = specdict[specname]
            assert isinstance(specstr, six.string_types), \
                   "Specification for %s was not a string" % specname
            if not noreturndefs:
                specstr_lang += _indentstr + \
                    resname + str(specname_count) + ' = '
            specname_count += 1
            specstr_lang += self.fspec.processTokens(allnames, specialtokens,
                                                specstr, var_arrayixstr,
                                                aux_arrayixstr, parsinps_names,
                                                parsinps_arrayixstr, specname, ignoreothers,
                                                doing_inserts)
            if not noreturndefs or not forexternal:
                specstr_lang += '\n'  # prepare for next line
        return specstr_lang

    def _processReusedPy(self, specnames, specdict, specials=[],
                         dovars=True, dopars=True, doinps=True, illegal=[]):
        """Process reused subexpression terms for Python code."""

        reused, specupdated, new_protected, order = _processReused(specnames,
                                                                   specdict,
                                                                   self.fspec.reuseterms,
                                                                   _indentstr)
        self.fspec._protected_reusenames = new_protected
        # symbols to parse are at indices 2 and 4 of 'reused' dictionary
        reusedParsed = self._parseReusedTermsPy(reused, [2, 4],
                                                specials=specials, dovars=dovars,
                                                dopars=dopars, doinps=doinps,
                                                illegal=illegal)
        reusedefs = {}.fromkeys(new_protected)
        for _, deflist in reusedParsed.items():
            for d in deflist:
                reusedefs[d[2]] = d
        return (concatStrDict(reusedefs, intersect(order, reusedefs.keys())),
                specupdated)

    def _parseReusedTermsPy(self, d, symbol_ixs, specials=[],
                            dovars=True, dopars=True, doinps=True, illegal=[]):
        """Process dictionary of reused term definitions (in spec syntax)."""
        # ... to parse special symbols to actual Python.
        # expect symbols to be processed at d list's entries given in
        # symbol_ixs.
        allnames = self.fspec.vars + self.fspec.pars + self.fspec.inputs + self.fspec.auxvars \
            + ['abs'] + self.fspec._protected_auxnames \
            + self.fspec._protected_scipynames \
            + self.fspec._protected_numpynames \
            + self.fspec._protected_specialfns \
            + self.fspec._protected_macronames \
            + self.fspec._protected_mathnames \
            + self.fspec._protected_randomnames \
            + self.fspec._protected_reusenames
        allnames = remain(allnames, illegal)
        if dovars:
            var_arrayixstr = dict(zip(self.fspec.vars, map(lambda i: str(i),
                                                      range(len(self.fspec.vars)))))
            aux_arrayixstr = dict(zip(self.fspec.auxvars, map(lambda i: str(i),
                                                         range(len(self.fspec.auxvars)))))
        else:
            var_arrayixstr = {}
            aux_arrayixstr = {}
        if dopars:
            if doinps:
                # parsinps_names is pars and inputs, each sorted
                # *individually*
                parsinps_names = self.fspec.pars + self.fspec.inputs
            else:
                parsinps_names = self.fspec.pars
            parsinps_arrayixstr = dict(zip(parsinps_names,
                                           map(lambda i: str(i),
                                               range(len(parsinps_names)))))
        else:
            parsinps_names = []
            parsinps_arrayixstr = {}
        specialtokens = remain(allnames, specials) + ['(', 't'] + specials
        for specname, itemlist in d.items():
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
                    parsedsymbol = self.fspec.processTokens(allnames,
                                                       specialtokens, symbol,
                                                       var_arrayixstr, aux_arrayixstr,
                                                       parsinps_names, parsinps_arrayixstr,
                                                       specname)
                    # must strip possible trailing whitespace!
                    d[specname][listix][ix] = parsedsymbol.strip()
        return d

    def _format_user_code(self, code):
        if not code:
            return ''

        code_ = self._specStrParse(
            ['inserts'],
            {'inserts': code},
            '',
            noreturndefs=True,
            ignoreothers=True,
            doing_inserts=True).strip()
        return self._format_code(_indentstr + code_)
