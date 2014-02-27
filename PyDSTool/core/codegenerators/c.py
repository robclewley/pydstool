#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from PyDSTool.common import invertMap, intersect, concatStrDict
from PyDSTool.parseUtils import convertPowers, parseMatrixStrToDictStr, addArgToCalls, wrapArgInCall, splitargs, findEndBrace
from PyDSTool.Symbolic import QuantSpec
from PyDSTool.utils import compareList, info

from .base import _processReused


class CCodeGenerator(object):

    def generate_aux(self, fspec):
        auxnames = fspec._auxfnspecs.keys()
        auxfns = {}
        # parameter and variable definitions
        # sorted version of var and par names sorted version of par
        # names (vars not #define'd in aux functions unless Jacobian)
        vnames = fspec.vars
        pnames = fspec.pars
        vnames.sort()
        pnames.sort()
        for auxname in auxnames:
            assert auxname not in ['auxvars', 'vfieldfunc'], \
                ("auxiliary function name '" + auxname + "' clashes with internal"
                 " names")
        # must add parameter argument so that we can name
        # parameters inside the functions! this would either
        # require all calls to include this argument (yuk!) or
        # else we add these extra parameters automatically to
        # every call found in the .c code (as is done currently.
        # this is still an untidy solution, but there you go...)
        for auxname in auxnames:
            auxspec = fspec._auxfnspecs[auxname]
            assert len(auxspec) == 2, 'auxspec tuple must be of length 2'
            if not isinstance(auxspec[0], list):
                print "Found type ", type(auxspec[0])
                print "Containing: ", auxspec[0]
                raise TypeError('aux function arguments '
                                'must be given as a list')
            if not isinstance(auxspec[1], str):
                print "Found type ", type(auxspec[1])
                print "Containing: ", auxspec[1]
                raise TypeError('aux function specification '
                                'must be a string of the function code')
            # Process Jacobian functions specially, if present
            if auxname == 'Jacobian':
                sig = "void jacobian("
                if not compareList(auxspec[0], ['t'] + fspec.vars):
                    print ['t'] + fspec.vars
                    print "Auxspec =", auxspec[0]
                    raise ValueError(
                        "Invalid argument list given in Jacobian.")
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxspec[1], 'pow')
                else:
                    auxstr = auxspec[1]
                parlist = "unsigned n_, unsigned np_, double t, double *Y_,"
                ismat = True
                sig += parlist + \
                    " double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
                specvars = fspec.vars
                specvars.sort()
                n = len(specvars)
                m = n
                specdict_temp = {}.fromkeys(specvars)
                if m == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in Jacobian for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in Jacobian for 1D system"
                    specdict_temp[specvars[0]] = auxstr
                else:
                    specdict_temp = parseMatrixStrToDictStr(auxstr, specvars)
                reusestr, body_processed_dict = self._processReusedC(
                    fspec, specvars,
                    specdict_temp)
                specdict = {}.fromkeys(specvars)
                for specname in specvars:
                    temp = body_processed_dict[specname]
                    specdict[specname] = splitargs(
                        temp.replace("[", "").replace("]", ""))
                body_processed = ""
                # C integrators expect column-major matrices
                for col in range(n):
                    for row in range(m):
                        try:
                            body_processed += "f_[" + str(col) + "][" + str(row) \
                                + "] = " + specdict[specvars[row]][col] + ";\n"
                        except IndexError:
                            raise ValueError(
                                "Jacobian should be %sx%s" % (m, n))
                body_processed += "\n"
                auxspec_processedDict = {auxname: body_processed}
            elif auxname == 'Jacobian_pars':
                sig = "void jacobianParam("
                if not compareList(auxspec[0], ['t'] + fspec.vars):
                    print ['t'] + fspec.vars
                    print "Auxspec =", auxspec[0]
                    raise ValueError(
                        "Invalid argument list given in Jacobian.")
                parlist = "unsigned n_, unsigned np_, double t, double *Y_,"
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxspec[1], 'pow')
                else:
                    auxstr = auxspec[1]
                ismat = True
                # specials = ["t","Y_","n_","np_","wkn_","wk_"]
                sig += parlist + \
                    " double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
                specvars = fspec.vars
                specvars.sort()
                n = len(specvars)
                if n == 0:
                    raise ValueError("Cannot have a Jacobian w.r.t. pars"
                                     " because no pars are defined")
                m = len(fspec.vars)
                specdict_temp = {}.fromkeys(fspec.vars)
                if m == n == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in Jacobian for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in Jacobian for 1D system"
                    specdict_temp[fspec.vars.values()[0]] = auxstr
                else:
                    specdict_temp = parseMatrixStrToDictStr(
                        auxstr, fspec.vars, m)
                reusestr, body_processed_dict = self._processReusedC(
                    fspec, fspec.vars,
                    specdict_temp)
                specdict = {}.fromkeys(fspec.vars)
                for specname in fspec.vars:
                    temp = body_processed_dict[specname]
                    specdict[specname] = splitargs(
                        temp.replace("[", "").replace("]", ""))
                body_processed = ""
                # C integrators expect column-major matrices
                for col in range(n):
                    for row in range(m):
                        try:
                            body_processed += "f_[" + str(col) + "][" + str(row) \
                                + "] = " + specdict[
                                    fspec.vars[row]][col] + ";\n"
                        except (IndexError, KeyError):
                            print "\nFound matrix:\n"
                            info(specdict)
                            raise ValueError(
                                "Jacobian should be %sx%s" % (m, n))
                body_processed += "\n"
                auxspec_processedDict = {auxname: body_processed}
            elif auxname == 'massMatrix':
                sig = "void massMatrix("
                if not compareList(auxspec[0], ['t'] + fspec.vars):
                    raise ValueError(
                        "Invalid argument list given in Mass Matrix.")
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxspec[1], 'pow')
                else:
                    auxstr = auxspec[1]
                parlist = "unsigned n_, unsigned np_,"
                ismat = True
                # specials = ["n_","np_","wkn_","wk_"]
                sig += parlist + \
                    " double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
                specvars = fspec.vars
                specvars.sort()
                n = len(specvars)
                m = n
                specdict_temp = {}.fromkeys(specvars)
                if m == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in mass matrix for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in mass matrix for 1D system"
                    specdict_temp[specvars.values()[0]] = auxstr
                else:
                    specdict_temp = parseMatrixStrToDictStr(
                        auxstr, specvars, m)
                reusestr, body_processed_dict = self._processReusedC(
                    fspec, specvars,
                    specdict_temp)
                specdict = {}.fromkeys(specvars)
                for specname in specvars:
                    temp = body_processed_dict[
                        specname].replace("[", "").replace("]", "")
                    specdict[specname] = splitargs(temp)
                body_processed = ""
                # C integrators expect column-major matrices
                for col in range(n):
                    for row in range(m):
                        try:
                            body_processed += "f_[" + str(col) + "][" + str(row) \
                                + "] = " + specdict[specvars[row]][col] + ";\n"
                        except KeyError:
                            raise ValueError(
                                "Mass matrix should be %sx%s" % (m, n))
                body_processed += "\n"
                auxspec_processedDict = {auxname: body_processed}
            else:
                ismat = False
                sig = "double " + auxname + "("
                parlist = ""
                namemap = {}
                for parname in auxspec[0]:
                    if parname == '':
                        continue
                    parlist += "double " + "__" + parname + "__, "
                    namemap[parname] = '__' + parname + '__'
                sig += parlist + "double *p_, double *wk_, double *xv_)"
                auxstr = auxspec[1]
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                prep_auxstr = self._processSpecialC(auxstr)
                prep_auxstr_quant = QuantSpec('prep_q',
                                              prep_auxstr.replace(
                                                  ' ', '').replace('\n', ''),
                                              treatMultiRefs=False, preserveSpace=True)
                # have to do name map now in case function's formal arguments
                # coincide with state variable names, which may get tied up
                # in reused terms and not properly matched to the formal args.
                prep_auxstr_quant.mapNames(namemap)
                auxspec = (auxspec[0], prep_auxstr_quant())
                reusestr, auxspec_processedDict = self._processReusedC(
                    fspec, [auxname],
                    {auxname: auxspec[1]})
                # addition of parameter done in Generator code
                # dummyQ = QuantSpec('dummy', auxspec_processedDict[auxname])
                # auxspec_processed = ""
                # add pars argument to inter-aux fn call
                # auxfn_found = False   # then expect a left brace next
                # for tok in dummyQ:
                #     if auxfn_found:
                # expect left brace in this tok
                #         if tok == '(':
                #             auxspec_processed += tok + 'p_, '
                #             auxfn_found = False
                #         else:
                #             raise ValueError("Problem parsing inter-auxiliary"
                #                              " function call")
                #     elif tok in fspec.auxfns and tok not in \
                #             ['Jacobian', 'Jacobian_pars']:
                #         auxfn_found = True
                #         auxspec_processed += tok
                #     else:
                #         auxspec_processed += tok
                # body_processed = "return "+auxspec_processed + ";\n\n"
            # add underscore to local names, to avoid clash with global
            # '#define' names
            dummyQ = QuantSpec('dummy', auxspec_processedDict[auxname],
                               treatMultiRefs=False, preserveSpace=True)
            body_processed = "return " * (not ismat) + dummyQ() + ";\n\n"
            # auxspecstr = sig + " {\n\n" + pardefines + vardefines*ismat \
            auxspecstr = sig + " {\n\n" \
                + "\n" + (len(reusestr) > 0) * "/* reused term definitions */\n" \
                + reusestr + (len(reusestr) > 0) * "\n" + body_processed \
                + "}"
               # + parundefines + varundefines*ismat + "}"
            # sig as second entry, whereas Python-coded specifications
            # have the fn name there
            auxfns[auxname] = (auxspecstr, sig)
        # Don't apply #define's for built-in functions
        auxfns['heav'] = ("int heav(double x_, double *p_, double *wk_, double *xv_) {\n"
                                +
                                "  if (x_>0.0) {return 1;} else {return 0;}\n}",
                                "int heav(double x_, double *p_, double *wk_, double *xv_)")
        auxfns['__rhs_if'] = ("double __rhs_if(int cond_, double e1_, "
                                    +
                                    "double e2_, double *p_, double *wk_, double *xv_) {\n"
                                    +
                                    "  if (cond_) {return e1_;} else {return e2_;};\n}",
                                    "double __rhs_if(int cond_, double e1_, double e2_, double *p_, double *wk_, double *xv_)")
        auxfns['__maxof2'] = ("double __maxof2(double e1_, double e2_, double *p_, double *wk_, double *xv_) {\n"
                                    +
                                    "if (e1_ > e2_) {return e1_;} else {return e2_;};\n}",
                                    "double __maxof2(double e1_, double e2_, double *p_, double *wk_, double *xv_)")
        auxfns['__minof2'] = ("double __minof2(double e1_, double e2_, double *p_, double *wk_, double *xv_) {\n"
                                    +
                                    "if (e1_ < e2_) {return e1_;} else {return e2_;};\n}",
                                    "double __minof2(double e1_, double e2_, double *p_, double *wk_, double *xv_)")
        auxfns['__maxof3'] = ("double __maxof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_) {\n"
                                    +
                                    "double temp_;\nif (e1_ > e2_) {temp_ = e1_;} else {temp_ = e2_;};\n"
                                    +
                                    "if (e3_ > temp_) {return e3_;} else {return temp_;};\n}",
                                    "double __maxof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_)")
        auxfns['__minof3'] = ("double __minof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_) {\n"
                                    +
                                    "double temp_;\nif (e1_ < e2_) {temp_ = e1_;} else {temp_ = e2_;};\n"
                                    +
                                    "if (e3_ < temp_) {return e3_;} else {return temp_;};\n}",
                                    "double __minof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_)")
        auxfns['__maxof4'] = ("double __maxof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_) {\n"
                                    +
                                    "double temp_;\nif (e1_ > e2_) {temp_ = e1_;} else {temp_ = e2_;};\n"
                                    +
                                    "if (e3_ > temp_) {temp_ = e3_;};\nif (e4_ > temp_) {return e4_;} else {return temp_;};\n}",
                                    "double __maxof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_)")
        auxfns['__minof4'] = ("double __minof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_) {\n"
                                    +
                                    "double temp_;\nif (e1_ < e2_) {temp_ = e1_;} else {temp_ = e2_;};\n"
                                    +
                                    "if (e3_ < temp_) {temp_ = e3_;};\nif (e4_ < temp_) {return e4_;} else {return temp_;};\n}",
                                    "double __minof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_)")
        # temporary placeholders for these built-ins...
        cases_ic = ""
        cases_index = ""
        for i in xrange(len(fspec.vars)):
            if i == 0:
                command = 'if'
            else:
                command = 'else if'
            vname = fspec.vars[i]
            cases_ic += "  " + command + " (strcmp(varname, " + '"' + vname + '"'\
                + ")==0)\n\treturn gICs[" + str(i) + "];\n"
            cases_index += "  " + command + " (strcmp(name, " + '"' + vname + '"'\
                + ")==0)\n\treturn " + str(i) + ";\n"
        # add remaining par names for getindex
        for i in xrange(len(fspec.pars)):
            pname = fspec.pars[i]
            cases_index += "  else if" + " (strcmp(name, " + '"' + pname + '"'\
                           + ")==0)\n\treturn " + str(
                               i + len(fspec.vars)) + ";\n"
        cases_ic += """  else {\n\tfprintf(stderr, "Invalid variable name %s for """ \
            + """initcond call\\n", varname);\n\treturn 0.0/0.0;\n\t}\n"""
        cases_index += """  else {\n\tfprintf(stderr, "Invalid name %s for """ \
            + """getindex call\\n", name);\n\treturn 0.0/0.0;\n\t}\n"""
        auxfns['initcond'] = ("double initcond(char *varname, double *p_, double *wk_, double *xv_) {\n"
                                    + "\n" + cases_ic + "}",
                                    'double initcond(char *varname, double *p_, double *wk_, double *xv_)')
        auxfns['getindex'] = ("int getindex(char *name, double *p_, double *wk_, double *xv_) {\n"
                                    + "\n" + cases_index + "}",
                                    'int getindex(char *name, double *p_, double *wk_, double *xv_)')
        auxfns['globalindepvar'] = ("double globalindepvar(double t, double *p_, double *wk_, double *xv_)"
                                          + " {\n  return globalt0+t;\n}",
                                          'double globalindepvar(double t, double *p_, double *wk_, double *xv_)')
        auxfns['getbound'] = \
            ("double getbound(char *name, int which_bd, double *p_, double *wk_, double *xv_) {\n"
             + "  return gBds[which_bd][getindex(name)];\n}",
             'double getbound(char *name, int which_bd, double *p_, double *wk_, double *xv_)')

        return auxfns

    def generate_spec(self, fspec):
        assert fspec.targetlang == 'c', ('Wrong target language for this'
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
        # sorted version of var and par names
        pnames = fspec.pars
        inames = fspec.inputs
        pnames.sort()
        inames.sort()
        pardefines = ""
        vardefines = ""
        inpdefines = ""
        parundefines = ""
        varundefines = ""
        inpundefines = ""
        # produce vector field specification
        assert fspec.vars == specname_vars, ('Mismatch between declared '
                                             ' variable names and varspecs keys')
        valid_depTargNames = fspec.inputs + fspec.vars + fspec.auxvars
        for specname, specstr in fspec.varspecs.iteritems():
            assert type(
                specstr) == str, "Specification for %s was not a string" % specname
            if any([pt in specstr for pt in ('^', '**')]):
                specstr = convertPowers(specstr, 'pow')
            specQS = QuantSpec('__spectemp__', specstr)
            for s in specQS:
                if s in valid_depTargNames and (specname, s) not in \
                        fspec.dependencies:  # and specname != s:
                    fspec.dependencies.append((specname, s))
        # pre-process reused sub-expression dictionary to adapt for
        # known calling sequence in C
        reusestr, specupdated = self._processReusedC(fspec, specname_vars,
                                                     fspec.varspecs)
        fspec.varspecs.update(specupdated)
        specstr_C = self._generate_fun(
            fspec, 'vfieldfunc', reusestr, specname_vars,
            pardefines, vardefines, inpdefines,
            parundefines, varundefines, inpundefines,
            True)
        fspec.spec = specstr_C
        # produce auxiliary variables specification
        specname_auxvars.sort()
        assert fspec.auxvars == specname_auxvars, \
            ('Mismatch between declared auxiliary'
             ' variable names and varspecs keys')
        if fspec.auxvars != []:
            reusestraux, specupdated = self._processReusedC(
                fspec, specname_auxvars,
                fspec.varspecs)
            fspec.varspecs.update(specupdated)
        if fspec.auxvars == []:
            auxspecstr_C = self._generate_fun(fspec, 'auxvars', '',
                                              specname_auxvars,
                                              '', '', '',
                                              '', '', '', False)
        else:
            auxspecstr_C = self._generate_fun(fspec, 'auxvars', reusestraux,
                                              specname_auxvars, pardefines,
                                              vardefines, inpdefines, parundefines,
                                              varundefines, inpundefines,
                                              False)
        fspec.auxspec = auxspecstr_C

    def _generate_fun(self, fspec, funcname, reusestr, specnames, pardefines,
                      vardefines, inpdefines, parundefines, varundefines,
                      inpundefines, docodeinserts):
        sig = "void " + funcname + "(unsigned n_, unsigned np_, double t, double *Y_, " \
              + \
            "double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
        # specstr = sig + "{\n\n" + pardefines + vardefines + "\n"
        specstr = sig + "{" + pardefines + vardefines + inpundefines + "\n"
        if docodeinserts and fspec.codeinserts['start'] != '':
            specstr += '/* Verbose code insert -- begin */\n' \
                + fspec.codeinserts['start'] \
                + '/* Verbose code insert -- end */\n\n'
        specstr += (len(reusestr) > 0) * "/* reused term definitions */\n" \
            + reusestr + "\n"
        auxdefs_parsed = {}
        # add function body
        for i in xrange(len(specnames)):
            xname = specnames[i]
            fbody = fspec.varspecs[xname]
            fbody_parsed = self._processSpecialC(fbody)
            if fspec.auxfns:
                fbody_parsed = addArgToCalls(fbody_parsed,
                                             fspec.auxfns.keys(),
                                             "p_, wk_, xv_")
                if 'initcond' in fspec.auxfns:
                    # convert 'initcond(x)' to 'initcond("x")' for
                    # compatibility with C syntax
                    fbody_parsed = wrapArgInCall(fbody_parsed,
                                                 'initcond', '"')
            specstr += "f_[" + str(i) + "] = " + fbody_parsed + ";\n"
            auxdefs_parsed[xname] = fbody_parsed
        if docodeinserts and fspec.codeinserts['end'] != '':
            specstr += '\n/* Verbose code insert -- begin */\n' \
                + fspec.codeinserts['end'] \
                + '/* Verbose code insert -- end */\n'
        specstr += "\n" + parundefines + varundefines + inpundefines + "}\n\n"
        fspec._auxdefs_parsed = auxdefs_parsed
        return (specstr, funcname)

    def _processReusedC(self, fspec, specnames, specdict):
        """Process reused subexpression terms for C code."""

        if fspec.auxfns:
            def addParToCall(s):
                return addArgToCalls(self._processSpecialC(s),
                                     fspec.auxfns.keys(), "p_, wk_, xv_")
            parseFunc = addParToCall
        else:
            parseFunc = self._processSpecialC
        reused, specupdated, new_protected, order = _processReused(specnames,
                                                                   specdict,
                                                                   fspec.reuseterms,
                                                                   '', 'double', ';',
                                                                   parseFunc)
        fspec._protected_reusenames = new_protected
        reusedefs = {}.fromkeys(new_protected)
        for _, deflist in reused.iteritems():
            for d in deflist:
                reusedefs[d[2]] = d
        return (concatStrDict(reusedefs, intersect(order, reusedefs.keys())),
                specupdated)

    def _processSpecialC(self, specStr):
        """Pre-process 'if' statements and names of 'abs' and 'sign' functions,
        as well as logical operators.
        """
        qspec = QuantSpec('spec', specStr, treatMultiRefs=False)
        qspec.mapNames({'abs': 'fabs', 'sign': 'signum', 'mod': 'fmod',
                        'and': '&&', 'or': '||', 'not': '!',
                        'True': 1, 'False': 0,
                        'max': '__maxof', 'min': '__minof'})
        qtoks = qspec.parser.tokenized
        # default value
        new_specStr = str(qspec)
        if 'if' in qtoks:
            new_specStr = ""
            num_ifs = qtoks.count('if')
            if_ix = -1
            ix_continue = 0
            for _ in range(num_ifs):
                if_ix = qtoks[if_ix + 1:].index('if') + if_ix + 1
                new_specStr += "".join(qtoks[ix_continue:if_ix]) + "__rhs_if("
                rbrace_ix = findEndBrace(qtoks[if_ix + 1:]) + if_ix + 1
                ix_continue = rbrace_ix + 1
                new_specStr += "".join(qtoks[if_ix + 2:ix_continue])
            new_specStr += "".join(qtoks[ix_continue:])
            qspec = QuantSpec('spec', new_specStr)
            qtoks = qspec.parser.tokenized
        if '__minof' in qtoks:
            new_specStr = ""
            num = qtoks.count('__minof')
            n_ix = -1
            ix_continue = 0
            for _ in range(num):
                n_ix = qtoks[n_ix + 1:].index('__minof') + n_ix + 1
                new_specStr += "".join(qtoks[ix_continue:n_ix])
                rbrace_ix = findEndBrace(qtoks[n_ix + 1:]) + n_ix + 1
                ix_continue = rbrace_ix + 1
                #assert qtoks[n_ix+2] == '[', "Error in min() syntax"
                #assert qtoks[rbrace_ix-1] == ']', "Error in min() syntax"
                #new_specStr += "".join(qtoks[n_ix+3:rbrace_ix-1]) + ")"
                num_args = qtoks[n_ix + 2:ix_continue].count(',') + 1
                if num_args > 4:
                    raise NotImplementedError(
                        "Max of more than 4 arguments not currently supported in C")
                new_specStr += '__minof%s(' % str(num_args)
                new_specStr += "".join(
                    [q for q in qtoks[n_ix + 2:ix_continue] if q not in ('[', ']')])
            new_specStr += "".join(qtoks[ix_continue:])
            qspec = QuantSpec('spec', new_specStr)
            qtoks = qspec.parser.tokenized
        if '__maxof' in qtoks:
            new_specStr = ""
            num = qtoks.count('__maxof')
            n_ix = -1
            ix_continue = 0
            for _ in range(num):
                n_ix = qtoks[n_ix + 1:].index('__maxof') + n_ix + 1
                new_specStr += "".join(qtoks[ix_continue:n_ix])
                rbrace_ix = findEndBrace(qtoks[n_ix + 1:]) + n_ix + 1
                ix_continue = rbrace_ix + 1
                #assert qtoks[n_ix+2] == '[', "Error in max() syntax"
                #assert qtoks[rbrace_ix-1] == ']', "Error in max() syntax"
                #new_specStr += "".join(qtoks[n_ix+3:rbrace_ix-1]) + ")"
                num_args = qtoks[n_ix + 2:ix_continue].count(',') + 1
                if num_args > 4:
                    raise NotImplementedError(
                        "Min of more than 4 arguments not currently supported in C")
                new_specStr += '__maxof%s(' % str(num_args)
                new_specStr += "".join(
                    [q for q in qtoks[n_ix + 2:ix_continue] if q not in ('[', ']')])
            new_specStr += "".join(qtoks[ix_continue:])
            qspec = QuantSpec('spec', new_specStr)
            qtoks = qspec.parser.tokenized
        return new_specStr
