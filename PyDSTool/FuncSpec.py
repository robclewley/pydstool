"""Functional specification classes.

   Robert Clewley, August 2005.

This module aids in building internal representations of ODEs, etc.,
particularly for the benefit of Automatic Differentiation
and for manipulation of abstraction digraphs.
"""

# PyDSTool imports
from __future__ import division, absolute_import, print_function
from .utils import *
from .common import *
from .parseUtils import *
from .errors import *
from .utils import info as utils_info
from .Symbolic import QuantSpec, allmathnames_symbolic

# Other imports
from copy import copy, deepcopy
from numpy import any
import six

import PyDSTool.core.codegenerators as CG

__all__ = ['RHSfuncSpec', 'ImpFuncSpec', 'ExpFuncSpec', 'FuncSpec',
           'getSpecFromFile', 'resolveClashingAuxFnPars', 'makePartialJac']

# XXX: this method is used elsewhere (Events.py)
_processReused = CG._processReused

# ---------------------------------------------------------------
class FuncSpec(object):
    """Functional specification of dynamics: abstract class.

    NOTES ON BUILT-IN AUX FUNCTIONS (WITH SYNTAX AS USED IN SPEC STRING):

    globalindepvar(t) -> global independent variable (time) reference

    initcond(varname) -> initial condition of that variable in this DS

    heav(x) = 1 if x > 0, 0 otherwise

    getindex(varname) -> index of varname in internal representation of
     variables as array

    getbound(name, which_bd) -> value of user-defined bound on the named
     variable or parameter, either the lower (which_bd=0) or higher
     (which_bd=1)

    if(condition, expr1, expr2) -> if condition as a function of state,
     parameters and time is true, then evaluate <expr1>, else evaluate
     <expr2>.

    MACRO `for` SYNTAX:

     for(i, ilo, ihi, expr_in_i) -> list of expressions where each
      occurrence of `[i]` is replaced with the appropriate integer.
      The letter i can be replaced with any other single character.

    MACRO `sum` SYNTAX:

     sum(i, ilo, ihi, expr_in_i) -> an expression that sums
      over the expression replacing any occurrence of `[i]` with
      the appropriate integer.
    """
    def __init__(self, kw_):
        # All math package names are reserved
        self._protected_mathnames = protected_mathnames
        self._protected_randomnames = protected_randomnames
        self._protected_scipynames = protected_scipynames
        self._protected_numpynames = protected_numpynames
        self._protected_specialfns = protected_specialfns
        self._protected_builtins = protected_builtins
        self._protected_symbolicnames = allmathnames_symbolic
        # We add internal default auxiliary function names for use by
        # functional specifications.
        self._builtin_auxnames = builtin_auxnames
        self._protected_macronames = protected_macronames
        self._protected_auxnames = copy(self._builtin_auxnames)
        self._protected_reusenames = []   # for reusable sub-expression terms
        needKeys = ['name', 'vars']
        optionalKeys = ['pars', 'inputs', 'varspecs', 'spec', '_for_macro_info',
                   'targetlang', 'fnspecs', 'auxvars', 'reuseterms',
                   'codeinsert_start', 'codeinsert_end', 'ignorespecial']
        self._initargs = deepcopy(kw_)

        # Do not destruct input arg
        kw = deepcopy(kw_)

        self.__validate_input(kw, needKeys + optionalKeys)

        # spec name
        self.name = kw.pop('name', 'untitled')
        # declare name lists: variables, aux variables, parameters, inputs
        for name in ['vars', 'pars', 'inputs', 'auxvars']:
            ns = kw.pop(name, [])
            setattr(self, name, [ns] if isinstance(ns, six.string_types)
                    else sorted(ns))

        self.targetlang = kw.pop('targetlang', 'python')
        if self.targetlang == 'c':
            self._defstr = "#define"
            self._undefstr = "#undef"
        else:
            self._defstr = ""
            self._undefstr = ""
        if 'ignorespecial' in kw:
            self._ignorespecial = kw['ignorespecial']
        else:
            self._ignorespecial = []

        codegen_opts = dict((k, kw.pop(k, '')) for k in ['codeinsert_start', 'codeinsert_end'])
        self.codegen = CG.getCodeGenerator(self, **codegen_opts)
        # ------------------------------------------
        # reusable terms in function specs
        self.reuseterms = kw.pop('reuseterms', {})

        # auxfns dict of functionality for auxiliary functions (in
        # either python or C). for instance, these are used for global
        # time reference, access of regular variables to initial
        # conditions, and user-defined quantities.
        self.auxfns = {}
        if 'fnspecs' in kw:
            self._auxfnspecs = deepcopy(kw['fnspecs'])
        else:
            self._auxfnspecs = {}
        # spec dict of functionality, as a string for each var
        # (in either python or C, or just for python?)
        if '_for_macro_info' in kw:
            self._varsbyforspec = kw['_for_macro_info'].varsbyforspec
        else:
            self._varsbyforspec = {}
        if 'varspecs' in kw:
            numaux = len(self.auxvars)
            if '_for_macro_info' in kw:
                if kw['_for_macro_info'].numfors > 0:
                    num_varspecs = numaux + len(self.vars) - kw['_for_macro_info'].totforvars + \
                                   kw['_for_macro_info'].numfors
                else:
                    num_varspecs = numaux + len(self.vars)
            else:
                num_varspecs = numaux + len(self.vars)
            if len(kw['varspecs']) != len(self._varsbyforspec) and \
               len(kw['varspecs']) != num_varspecs:
                print("# state variables: %d" % len(self.vars))
                print("# auxiliary variables: %d" % numaux)
                print("# of variable specs: %d" % len(kw['varspecs']))
                raise ValueError('Incorrect size of varspecs')
            self.varspecs = deepcopy(kw['varspecs'])
        else:
            self.varspecs = {}
        self.codeinserts = {'start': '', 'end': ''}
        # spec dict of functionality, as python functions,
        # or the paths/names of C dynamic linked library files
        # can be user-defined or generated from generateSpec
        if 'spec' in kw:
            assert isinstance(kw['spec'], tuple), ("'spec' must be a pair:"
                                    " (spec body, spec name)")
            assert len(kw['spec'])==2, ("'spec' must be a pair:"
                                    " (spec body, spec name)")
            self.spec = deepcopy(kw['spec'])
            # auxspec not used for explicitly-given specs. it's only for
            # auto-generated python auxiliary variable specs (as py functions)
            self.auxspec = {}
            if 'dependencies' in kw:
                self._dependencies = kw['dependencies']
            else:
                raise PyDSTool_KeyError("Dependencies must be provided "
                         "explicitly when using 'spec' form of initialization")
        else:
            self.spec = {}
            self.auxspec = {}
        self.defined = False  # initial value
        self.validateDef(self.vars, self.pars, self.inputs, self.auxvars, list(self._auxfnspecs.keys()))
        # ... exception if not valid
        # pre-process specification string for built-in macros (like `for`,
        # i.e. that are not also auxiliary functions, like the in-line `if`)
        self.doPreMacros()
        # !!!
        # want to create _pyauxfns but not C versions until after main spec
        # !!!
        self.generateAuxFns()
        if self.spec == {}:
            assert self.varspecs != {}, \
                   'No functional specification provided!'
            self.generateSpec()
            # exception if the following is not successful
            self.validateDependencies(self.dependencies)
        #self.generateAuxFns()
        # self.validateSpecs()
        # algparams is only used by ImplicitFnGen to pass extra info to Variable
        self.algparams = {}
        self.defined = True

    def __validate_input(self, kw, valid_keys):
        """Global input dictionary validation"""
        invalid = set(kw.keys()) - set(valid_keys)
        if invalid:
            raise PyDSTool_KeyError(
                'Invalid keys %r passed in argument dict' % list(invalid))

        if 'vars' not in kw:
            raise PyDSTool_KeyError(
                "Require a variables specification key -- 'vars'")

        spec_keys = ['varspecs', 'spec']
        if all(k not in kw for k in spec_keys):
            raise PyDSTool_KeyError(
                "Require a functional specification key -- 'spec' or 'varspecs'")

        if all(k in kw for k in spec_keys):
            raise PyDSTool_KeyError(
                "Cannot provide both 'spec' and 'varspecs' keys")

    @property
    def targetlang(self):
        return self._targetlang

    @targetlang.setter
    def targetlang(self, value):
        try:
            value = value.lower()
            if value not in targetLangs:
                raise ValueError('Invalid specification for targetlang')
        except AttributeError:
            raise TypeError("Expected string type for target language")

        self._targetlang = value

    @property
    def dependencies(self):
        if not hasattr(self, '_dependencies'):
            deps = set()
            valid_targets = self.inputs + self.vars
            for name, spec in self.varspecs.items():
                specQ = QuantSpec('__spectemp__', spec)
                [deps.add((name, s)) for s in specQ if s in valid_targets]

            self._dependencies = sorted(deps)

        return self._dependencies

    @property
    def reuseterms(self):
        return self._reuseterms

    @reuseterms.setter
    def reuseterms(self, terms):
        if not isinstance(terms, dict):
            raise ValueError('reuseterms must be a dictionary of strings ->'
                               ' replacement strings')
        self._reuseterms = dict(
            (t, rt) for t, rt in terms.items()
            if self.__term_valid(t) and self.__repterm_valid(rt)
        )

    def __term_valid(self, term):
        if isNumericToken(term):
            # don't replace numeric terms (sometimes these are
            # generated automatically by Constructors when resolving
            # explicit variable inter-dependencies)
            return False

        if term[0] in '+/*':
            print("Error in term:%s" % term)
            raise ValueError('terms to be substituted must not begin '
                                'with arithmetic operators')
        if term[0] == '-':
            term = '(' + term + ')'
        if term[-1] in '+-/*':
            print("Error in term:%s" % term)
            raise ValueError('terms to be substituted must not end with '
                                'arithmetic operators')
        for s in term:
            if self.targetlang == 'python':
                if s in r'[]{}~@#$%&\|?^': # <>! now OK, e.g. for "if" statements
                    print("Error in term:%s" % term)
                    raise ValueError('terms to be substituted must be '
                        'alphanumeric or contain arithmetic operators '
                        '+ - / *')
            else:
                if s in r'[]{}~!@#$%&\|?><': # removed ^ from this list
                    print("Error in term:%s" % term)
                    raise ValueError('terms to be substituted must be alphanumeric or contain arithmetic operators + - / *')
        return True

    def __repterm_valid(self, repterm):
        if repterm[0] in num_chars:
            print("Error in replacement term:%s" % repterm)
            raise ValueError('replacement terms must not begin with numbers')
        for s in repterm:
            if s in r'+-/*.()[]{}~!@#$%^&\|?><,':
                print("Error in replacement term:%s" % repterm)
                raise ValueError('replacement terms must be alphanumeric')

        return True

    def __hash__(self):
        """Unique identifier for this specification."""
        deflist = [self.name, self.targetlang]
        # lists
        for l in [self.pars, self.vars, self.auxvars, self.inputs,
                  self.spec, self.auxspec]:
            deflist.append(tuple(l))
        # dicts
        for d in [self.auxfns, self.codeinserts]:
            deflist.append(tuple(sortedDictItems(d, byvalue=False)))
        return hash(tuple(deflist))

    def recreate(self, targetlang):
        if targetlang == self.targetlang:
            # print "Returning a deep copy of self"
            return deepcopy(self)
        fs = FuncSpec.__new__(self.__class__)
        new_args = deepcopy(self._initargs)
        if self.codeinserts['start'] != '':
            del new_args['codeinsert_start']
            print("Warning: code insert (start) ignored for new target")
        if self.codeinserts['end'] != '':
            del new_args['codeinsert_end']
            print("Warning: code insert (end) ignored for new target")
        new_args['targetlang'] = targetlang
        fs.__init__(new_args)
        return fs

    def __call__(self):
        # info is defined in utils.py
        utils_info(self.__dict__, "FuncSpec " + self.name)


    # def info(self, verbose=1):
    #     if verbose > 0:
    #         # info is defined in utils.py
    #         utils_info(self.__dict__, "FuncSpec " + self.name,
    #              recurseDepthLimit=1+verbose)
    #     else:
    #         print self.__repr__()


    # # This function doesn't work -- it generates:
    # #    global name 'self' is not defined
    # # in the _specfn call
    # def validateSpecs(self):
    #     # dummy values for internal values possibly needed by auxiliary fns
    #     self.globalt0 = 0
    #     self.initialconditions = {}.fromkeys(self.vars, 0)
    #     lenparsinps = len(self.pars)+len(self.inputs)
    #     pi_vals = zeros(lenparsinps, float64)
    #     _specfn(1, self.initialconditions.values(), pi_vals)

    def validateDef(self, vars, pars, inputs, auxvars, auxfns):
        """Validate definition of the functional specification."""
        # verify that vars, pars, and inputs are non-overlapping lists
        assert not intersect(vars, pars), 'variable and param names overlap'
        assert not intersect(vars, inputs), 'variable and input names overlap'
        assert not intersect(pars, inputs), 'param and input names overlap'
        assert not intersect(vars, auxfns), ('variable and auxiliary function '
                                             'names overlap')
        assert not intersect(pars, auxfns), ('param and auxiliary function '
                                             'names overlap')
        assert not intersect(inputs, auxfns), ('input and auxiliary function '
                                               'names overlap')
        assert not intersect(vars, auxvars), ('variable and auxiliary variable '
                                             'names overlap')
        assert not intersect(pars, auxvars), ('param and auxiliary variable '
                                             'names overlap')
        assert not intersect(inputs, auxvars), ('input and auxiliary variable '
                                               'names overlap')
        # verify uniqueness of all names
        assert isUniqueSeq(vars), 'variable names are repeated'
        assert isUniqueSeq(pars), 'parameter names are repeated'
        assert isUniqueSeq(inputs), 'input names are repeated'
        if auxvars != []:
            assert isUniqueSeq(auxvars), 'auxiliary variable names are repeated'
        if auxfns != []:
            assert isUniqueSeq(auxfns), 'auxiliary function names are repeated'
        allnames = vars+pars+inputs+auxvars
        allprotectednames = self._protected_mathnames + \
                            self._protected_scipynames + \
                            self._protected_numpynames + \
                            self._protected_specialfns + \
                            self._protected_randomnames + \
                            self._protected_auxnames + \
                            ['abs', 'min', 'max', 'and', 'or', 'not',
                             'True', 'False']
        # other checks
        first_char_check = [alphabet_chars_RE.match(n[0]) \
                                     is not None for n in allnames]
        if not all(first_char_check):
            print("Offending names:%r" % [n for i, n in enumerate(allnames) \
                                       if not first_char_check[i]])
            raise ValueError('Variable, parameter, and input names must not '
                         'begin with non-alphabetic chars')
        protected_overlap = intersect(allnames, allprotectednames)
        if protected_overlap != []:
            print("Overlapping names:%r" % protected_overlap)
            raise ValueError('Variable, parameter, and input names must not '
                         'overlap with protected math / aux function names')
        ## Not yet implemented ?
        # verify that targetlang is consistent with spec contents?
        # verify that spec is consistent with specstring (if not empty)?

    def validateDependencies(self, dependencies):
        """Validate the stored dependency pairs for self-consistency."""
        # dependencies is a list of unique ordered pairs (i,o)
        # where (i,o) means 'variable i directly depends on variable o'
        # (o can include inputs)
        assert isinstance(dependencies, list), ('dependencies must be a list '
                                                'of unique ordered pairs')
        # Verify all names in dependencies are in self.vars
        # and that (i,o) pairs are unique in dependencies
        all_vars = self.vars+self.auxvars
        for d in dependencies:
            try:
                i, o = d
            except:
                raise ValueError('dependencies must be ordered pairs')
            firstpos = dependencies.index(d)
            assert d not in dependencies[firstpos+1:], \
                   'dependency pairs must be unique'
            assert i in all_vars, 'unknown variable name %s in dependencies'%i
            assert o in self.vars or o in self.inputs, \
                   'unknown variable name %s in dependencies'%o
        # No need to verify that dependencies are consistent with spec,
        # if spec was generated automatically

    def generateAuxFns(self):
        if self.targetlang != 'python':
            # Always makes a set of python versions of the functions for future
            # use by user at python level
            # FIXME: hack to generate _pyauxfns
            # FIXME: as a side effect this creates '_user_auxfns_interface' field
            CG.getCodeGenerator(self, 'python').generate_aux()
        if self.targetlang != 'matlab':
            self.auxfns = self.codegen.generate_aux()
        else:
            for name, spec in self._auxfnspecs.items():
                self.__validate_aux_spec(name, spec)
                if name in ['Jacobian', 'Jacobian_pars', 'massMatrix']:
                    code, signature = self.codegen.generate_special(name, spec)
                else:
                    code, signature = self.codegen.generate_auxfun(name, spec)
                self.auxfns[name] = (code, signature)
                self._protected_auxnames.append(name)

    def __validate_aux_spec(self, name, spec):
        assert name not in ['auxvars', 'vfield'], \
            ("auxiliary function name '" + name + "' clashes with internal"
                " names")
        assert len(spec) == 2, 'auxspec tuple must be of length 2'
        if not isinstance(spec[0], list):
            raise TypeError('aux function arguments must be given as a list')
        if not isinstance(spec[1], six.string_types):
            raise TypeError('aux function specification must be a string of the function code')

    def generateSpec(self):
        """Automatically generate callable target-language functions from
        the user-defined specification strings."""
        if self.targetlang != 'matlab':
            self.codegen.generate_spec()
        else:
            assert self.varspecs != {}, 'varspecs attribute must be defined'
            assert set(self.vars) - set(self.varspecs.keys()) == set([]), 'Mismatch between declared variable names and varspecs keys'
            for name, spec in self.varspecs.items():
                assert isinstance(spec, six.string_types), \
                       "Specification for %s was not a string" % name
            self.spec = self.codegen.generate_spec(self.vars, self.varspecs)

    def generate_user_module(self, eventstruct, **kwargs):
        return self.codegen.generate_user_module(self, eventstruct, **kwargs)

    def doPreMacros(self):
        """Pre-process any macro spec definitions (e.g. `for` loops)."""

        assert self.varspecs != {}, 'varspecs attribute must be defined'
        specnames_unsorted = list(self.varspecs.keys())
        _vbfs_inv = invertMap(self._varsbyforspec)
        # Process state variable specifications
        if len(_vbfs_inv) > 0:
            specname_vars = []
            specname_auxvars = []
            for varname in self.vars:
                # check if varname belongs to a for macro grouping in self.varspecs
                specname = _vbfs_inv[varname]
                if specname not in specname_vars:
                    specname_vars.append(specname)
            for varname in self.auxvars:
                # check if varname belongs to a for macro grouping in self.varspecs
                specname = _vbfs_inv[varname]
                if specname not in specname_auxvars:
                    specname_auxvars.append(specname)
        else:
            specname_vars = intersect(self.vars, specnames_unsorted)
            specname_auxvars = intersect(self.auxvars, specnames_unsorted)
        specname_vars.sort()
        specname_auxvars.sort()
        specnames = specname_vars + specname_auxvars  # sorted *individually*
        specnames_temp = copy(specnames)
        for specname in specnames_temp:
            leftbrack_ix = specname.find('[')
            rightbrack_ix = specname.find(']')
            test_sum = leftbrack_ix + rightbrack_ix
            if test_sum > 0:
                # both brackets found -- we expect a `for` macro in specstr
                assert rightbrack_ix - leftbrack_ix == 2, ('Misuse of square '
                                 'brackets in spec definition. Expected single'
                                 ' character between left and right brackets.')
                # if remain(self._varsbyforspec[specname], self.vars) == []:
                #     foundvar = True
                # else:
                #     foundvar = False  # auxiliary variable instead
                rootstr = specname[:leftbrack_ix]
                istr = specname[leftbrack_ix+1]
                specstr = self.varspecs[specname]
                assert specstr[:4] == 'for(', ('Expected `for` macro when '
                                'square brackets used in name definition')
                # read contents of braces
                arginfo = readArgs(specstr[3:])
                if not arginfo[0]:
                    raise ValueError('Error finding '
                            'arguments applicable to `for` '
                            'macro')
                arglist = arginfo[1]
                assert len(arglist) == 4, ('Wrong number of arguments passed '
                                           'to `for` macro. Expected 4')
                istr = arglist[0]
                allnames = self.vars + self.pars + self.inputs + self.auxvars \
                           + self._protected_mathnames \
                           + self._protected_randomnames \
                           + self._protected_auxnames \
                           + self._protected_scipynames \
                           + self._protected_numpynames \
                           + self._protected_specialfns \
                           + self._protected_macronames \
                           + self._protected_builtins \
                           + ['True', 'False']
                assert istr not in allnames, ('loop index in `for` macro '
                                              'must not be a reserved name')
                assert alphabet_chars_RE.match(istr[0]) is not None, \
                       ('loop index symbol in `for` macro must start with '
                        'a letter')
                for ichar in istr:
                    assert name_chars_RE.match(ichar) is not None, \
                                         ('loop index symbol in `for` macro '
                                                'must be alphanumeric')
                ilo = int(arglist[1])
                ihi = int(arglist[2])
                # NOTE: rootstr + '['+istr+'] = ' + arglist[3]
                expr = arglist[3]
                # add macro text
                varspecs = self._macroFor(rootstr, istr, ilo, ihi, expr)
                specnames_gen = list(varspecs.keys())
                # now we update the dictionary of specnames with the
                # processed, expanded versions
                specnames.remove(specname)
                # if foundvar:
                #     assert rootstr+'['+istr+']' in self.varspecs, ('Mismatch '
                #                                  'between declared variables '
                #                                'and loop index in `for` macro')
                #     #self.vars.remove(specname)
                # else:
                #     assert rootstr+'['+istr+']' in self.varspecs, ('Mismatch '
                #                                  'between declared variables '
                #                                'and loop index in `for` macro')
                #     self.auxvars.remove(specname)
                del(self.varspecs[specname])
                for sname in specnames_gen:
                    self.varspecs[sname] = varspecs[sname]
                    specnames.append(sname)
                    # if foundvar:
                    #     self.vars.append(sname)
                    # else:
                    #     self.auxvars.append(sname)
            elif test_sum == -2:
                pass
                # no brackets found. regular definition line. take no action.
            else:
                raise AssertionError('Misuse of square brackets in spec '
                                       'definition. Expected single'
                                 ' character between left and right brackets.')

    def _macroFor(self, rootstr, istr, ilo, ihi, expr_in_i):
        """Internal utility function to build multiple instances of expression
        'expr_in_i' where integer i has been substituted for values from ilo to ihi.
        Returns dictionary keyed by rootstr+str(i) for each i.
        """
        # already tested for the same number of [ and ] occurrences
        retdict = {}
        q = QuantSpec('__temp__', expr_in_i)
        eval_pieces = {}
        for ix, tok in enumerate(q):
            if tok[0] == '[':
                eval_str = tok[1:-1]
                if istr in eval_str:
                    eval_pieces[ix] = eval_str
                # otherwise may be a different, embedded temp index for another
                # sum, etc., so don't touch it
        keys = list(eval_pieces.keys())
        keys.sort()
        ranges = remove_indices_from_range(keys, len(q.parser.tokenized)-1)
        # By virtue of this syntax, the first [] cannot be before some other text
        pieces = []
        eval_ixs = []
        for ri, r in enumerate(ranges):
            if len(r) == 1:
                pieces.append(q[r[0]])
            else:
                # len(r) == 2
                pieces.append(''.join(q[r[0]:r[1]]))
            if ri+1 == len(ranges):
                # last one - check if there's an eval piece placeholder to append at the end
                if len(keys) > 0 and keys[-1] == r[-1]:
                    pieces.append('')
                    eval_ixs.append(len(pieces)-1)
                # else do nothing
            else:
                # in-between pieces, so append a placeholder for an eval piece
                pieces.append('')
                eval_ixs.append(len(pieces)-1)
        for i in range(ilo, ihi+1):
            for k, ei in zip(keys, eval_ixs):
                s = eval_pieces[k].replace(istr, str(i))
                try:
                    pieces[ei] = str(int(eval(s)))
                except NameError:
                    # maybe recursive 'sum' syntax, so a different index letter
                    pieces[ei] = s
            retdict[rootstr+str(i)] = ''.join(pieces)+'\n'
        return retdict

    def _macroSum(self, istr, ilo, ihi, expr_in_i):
        def_dict = self._macroFor('', istr, int(ilo), int(ihi), expr_in_i)
        retstr = '(' + "+".join([term.strip() for term in def_dict.values()]) + ')'
        return retstr

    def processTokens(self, allnames, specialtokens, specstr,
                        var_arrayixstr, aux_arrayixstr, parsinps_names,
                        parsinps_arrayixstr, specname, ignoreothers=False,
                        doing_inserts=False):
        # This function is an earlier version of parseUtils.py's
        # parse method of a parserObject.
        # This function should be replaced with an adapted version
        # of parserObject that can handle auxiliary function call
        # parsing and the inbuilt macros. This is some of the worst-organized
        # code I ever wrote, early in my experience with Python. My apologies...
        returnstr = ''
        if specstr[-1] != ')':
            # temporary hack because strings not ending in ) lose their last
            # character!
            specstr += ' '
        scount = 0
        speclen = len(specstr)
        valid_depnames = self.vars+self.auxvars
        s = ''
        ignore_list = ['', ' ', '\n'] + allnames
        foundtoken = False
        # initial value for special treatment of the 'initcond' built-in
        # auxiliary function's argument
        strname_arg_imminent = False
        auxfn_args_imminent = False
        while scount < speclen:
            stemp = specstr[scount]
            scount += 1
            if name_chars_RE.match(stemp) is None:
                # found a non-alphanumeric char
                # so just add to returnstr with accumulated s characters
                # (these will have been deleted if s contained a target
                # name)
                if not ignoreothers and s not in ignore_list:
                    # adding allnames catches var names etc. that are valid
                    # in auxiliary functions but are not special tokens
                    # and must be left alone
                    print("Error in specification `%s` with token `%s` :\n" % (specname, specstr))
                    raise ValueError('Undeclared or illegal token `'+s+'` in'
                                       ' spec string `'+specname+'`')
                if stemp == '^' and self.targetlang == 'python':
                    raise ValueError('Character `^` is not allowed. '
                                       'Please use the pow() call')
                if stemp == '(':
                    returnstr += s
                    s = stemp
                else:
                    returnstr += s
                    if len(returnstr)>1 and stemp == returnstr[-1] == "*":
                        # check for ** case
                        raise ValueError('Operator ** is not allowed. '
                                   'Please use the pow() call')
                    returnstr += stemp
                    s = ''
                    continue
            else:
                if s == '' and stemp not in num_chars:
                    s += stemp
                elif s != '':
                    s += stemp
                else:
                    returnstr += stemp
                    continue
            if s in specialtokens + self._ignorespecial:
                if s != '(':
                    if scount < speclen - 1:
                        if name_chars_RE.match(specstr[scount]) is None:
                            foundtoken = True
                        else:
                            if s in ['e','E'] and \
                               name_chars_RE.match(specstr[scount]).group() \
                                       in num_chars+['-']:
                                # not expecting an arithmetic symbol or space
                                # ... we *are* expecting a numeric
                                foundtoken = True
                    else:
                        foundtoken = True
                else:
                    foundtoken = True
                if foundtoken:
                    if s == '(':
                        if auxfn_args_imminent:
                            returnstr += s+'parsinps, '
                            auxfn_args_imminent = False
                        else:
                            returnstr += s
                    elif s == 'abs':
                        returnstr += s
                    elif s in var_arrayixstr and \
                         (len(returnstr)==0 or len(returnstr)>0 and \
                          returnstr[-1] not in ["'", '"']):
                        if strname_arg_imminent:
                            returnstr += "'"+s+"'"
                            strname_arg_imminent = False
                        else:
                            if specname in valid_depnames \
                               and (specname, s) not in self.dependencies:
                                self.dependencies.append((specname,s))
                            returnstr += 'x['+var_arrayixstr[s]+']'
                    elif s in aux_arrayixstr:
                        if strname_arg_imminent:
                            returnstr += "'"+s+"'"
                            strname_arg_imminent = False
                        else:
                            print("Spec name:%s" % specname)
                            print("Spec string:%s" % specstr)
                            print("Problem symbol:%s" % s)
                            raise NameError('auxiliary variables cannot '
                                         'appear on any right-hand side '
                                         'except their initial value')
                    elif s in parsinps_arrayixstr and \
                         (len(returnstr)==0 or len(returnstr)>0 and \
                          returnstr[-1] not in ["'", '"']):
                        if s in self.inputs:
                            if specname in valid_depnames and \
                                   (specname, s) not in self.dependencies:
                                self.dependencies.append((specname,s))
                        if strname_arg_imminent:
                            returnstr += "'"+s+"'"
                            strname_arg_imminent = False
                        else:
                            returnstr += 'parsinps[' + \
                                      parsinps_arrayixstr[s] + ']'
                    elif s in self._protected_mathnames:
                        if s in ['e','E']:
                            # special case where e is either = exp(0)
                            # as a constant or it's an exponent in 1e-4
                            if len(returnstr)>0:
                                if returnstr[-1] not in num_chars+['.']:
                                    returnstr += 'math.'+s.lower()
                                else:
                                    returnstr += s
                            else:
                                returnstr += 'math.'+s.lower()
                        else:
                            returnstr += 'math.'+s
                    elif s in self._protected_randomnames:
                        if len(returnstr) > 0:
                            if returnstr[-1] == '.':
                                # not a standalone name (e.g. "sample" may be a method call
                                # in an embedded system)
                                returnstr += s
                            else:
                                returnstr += 'random.'+s
                        else:
                            returnstr += 'random.'+s
                    elif s in self._protected_scipynames:
                        if len(returnstr) > 0:
                            if returnstr[-1] == '.':
                                # not a standalone name (e.g. may be a method call in an
                                # embedded system)
                                returnstr += s
                            else:
                                returnstr += 'scipy.'+s
                        else:
                            returnstr += 'scipy.'+s
                    elif s in self._protected_numpynames:
                        if len(returnstr) > 0:
                            if returnstr[-1] == '.':
                                # not a standalone name (e.g. may be a method call in an
                                # embedded system)
                                returnstr += s
                            else:
                                returnstr += 'numpy.'+s
                        else:
                            returnstr += 'numpy.'+s
                    elif s in self._protected_specialfns:
                        if self.targetlang != 'python':
                            print("Function %s is currently not supported "%s +
                                "outside of python target language definitions")
                            raise ValueError("Invalid special function for "
                                             "non-python target definition")
                        # replace the underscore in the name with a dot
                        # to access scipy.special
                        returnstr += 'scipy.'+s.replace('_','.')
                    elif s in self._protected_macronames:
                        if doing_inserts:
                            # Code inserts don't use macro versions of "if", "for", etc.
                            # They are interpreted as regular python
                            returnstr += s
                        else:
                            if specname in self._pyauxfns:
                                # remove vars, auxs, inputs
                                to_remove = self.vars + self.auxvars + self.inputs
                                filtfunc = lambda n: n not in to_remove
                                specialtokens_temp = list(filter(filtfunc,
                                                        specialtokens+self._ignorespecial))
                            else:
                                specialtokens_temp = specialtokens+self._ignorespecial
                            if s == 'if':
                                # hack for special 'if' case
                                # read contents of braces
                                endargbrace = findEndBrace(specstr[scount:]) \
                                                 + scount + 1
                                argstr = specstr[scount:endargbrace]
                                procstr = self.processTokens(allnames,
                                                specialtokens_temp, argstr,
                                                var_arrayixstr,
                                                aux_arrayixstr, parsinps_names,
                                                parsinps_arrayixstr, specname)
                                arginfo = readArgs(procstr)
                                if not arginfo[0]:
                                    raise ValueError('Error finding '
                                            'arguments applicable to `if` '
                                            'macro')
                                # advance pointer in specstr according to
                                # how many tokens/characters were read in for
                                # the argument list
                                scount += len(argstr) # not arginfo[2]
                                arglist = arginfo[1]
                                assert len(arglist) == 3, ('Wrong number of'
                                                ' arguments passed to `if`'
                                                ' macro. Expected 3')
                                returnstr += 'ds.' + self._pyauxfns[s][1] + \
                                             '(parsinps, '+procstr[1:]
                            elif s == 'for':
                                raise ValueError('Macro '+s+' cannot '
                                        'be used here')
                            elif s == 'sum':
                                endargbrace = findEndBrace(specstr[scount:]) \
                                                 + scount + 1
                                argstr = specstr[scount:endargbrace]
                                arginfo = readArgs(argstr)
                                if not arginfo[0]:
                                    raise ValueError('Error finding '
                                            'arguments applicable to `sum` '
                                            'macro')
                                arglist = arginfo[1]
                                assert len(arglist) == 4, ('Wrong number of'
                                                ' arguments passed to `sum`'
                                                ' macro. Expected 4')
                                # advance pointer in specstr according to
                                # how many tokens/characters were read in for
                                # the argument list
                                scount += len(argstr)
                                # recursively process main argument
                                returnstr += self.processTokens(allnames,
                                                specialtokens_temp,
                                                self._macroSum(*arglist), var_arrayixstr,
                                                aux_arrayixstr, parsinps_names,
                                                parsinps_arrayixstr, specname)
                            else:
                                # max and min just pass through
                                returnstr += s
                    elif s in self._protected_auxnames:
                        if s in ['initcond', 'getbound']:
                            # must prepare parser for upcoming variable
                            # name in argument that must only be
                            # converted to its index in x[]
                            strname_arg_imminent = True
                        # add internal prefix (to avoid method name clashes
                        # in DS objects, for instance) unless built-in function
                        returnstr += 'ds.' + self._pyauxfns[s][1]
                        auxfn_args_imminent = True
                    elif s in self._pyauxfns:
                        # treat inter-aux function dependencies:
                        # any protected auxnames will already have been
                        # processed because this is placed after that check.
                        # don't reference self._pyauxfns[s] because it doesn't
                        # contain the processed definition of the function.
                        if s in ['initcond', 'getbound']:
                            # must prepare parser for upcoming variable
                            # name in argument that must only be
                            # converted to its index in x[]
                            strname_arg_imminent = True
                        # add internal prefix (to avoid method name clashes
                        # in DS objects, for instance) unless built-in function
                        returnstr += 'ds.' + s
                        auxfn_args_imminent = True
                    elif s in self._protected_reusenames:
                        returnstr += s
                    else:
                        # s is e.g. a declared argument to an aux fn but
                        # only want to ensure it is present. no action to take.
                        returnstr += s
                    # reset for next iteration
                    s = ''
                    foundtoken = False
        # end of scount while loop
        return returnstr

    def _infostr(self, verbose=1):
        if verbose == 0:
            outputStr = "FuncSpec " + self.name
        else:
            outputStr = '*********** FuncSpec:  '+self.name + ' ***********'
            outputStr += '\nTarget lang:  '+ self.targetlang
            outputStr += '\nVariables:  '
            for v in self.vars:
                outputStr += v+'  '
            outputStr += '\nParameters:  '
            if len(self.pars):
                for p in self.pars:
                    outputStr += p+'  '
            else:
                outputStr += '[]'
            outputStr += '\nExternal inputs:  '
            if len(self.inputs):
                for i in self.inputs:
                    outputStr += i+'  '
            else:
                outputStr += '[]'
        if verbose == 2:
            outputStr += "\nSpecification functions (in target language):"
            outputStr += "\n  (ignore any arguments `ds` and `parsinps`," \
                       + "\n   which are for internal use only)\n"
            if self.spec == {}:
                outputStr += "\n None\n"
            else:
                outputStr += "\n  "+self.spec[0]+"\n"
            if len(self.auxvars) and self.auxspec != {}:
                outputStr += " "+self.auxspec[0]
            if self._protected_auxnames != []:
                outputStr += '\n\nUser-defined auxiliary variables:  '
                for v in self.auxvars:
                    outputStr += v+'  '
                outputStr += '\n\nUser-defined auxiliary functions (in target ' + \
                             'language):'
                for auxname, auxdef in sortedDictItems(self.auxfns):
                    # verbose option shows up builtin auxiliary func definitions
                    if auxname not in self._builtin_auxnames or verbose > 0:
                        outputStr += '\n  ' + auxdef[0] + '\n'
            outputStr += "\n\nDependencies in specification functions - pair (i, o)"\
                    " means i depends on o:\n  " + str(self.dependencies)
        return outputStr

    def info(self, verbose=0):
        print(self._infostr(verbose))

    def __repr__(self):
        return self._infostr(verbose=0)

    __str__ = __repr__


    # XXX: methods are to be removed
    # These methods don't belongs to this class, but are used by clients
    # elsewhere (Events.py)
    def _specStrParse(self, specnames, specdict, resname='', specials=[],
                      dovars=True, dopars=True, doinps=True,
                      noreturndefs=False, forexternal=False, illegal=[],
                      ignoreothers=False, doing_inserts=False):
        return CG.getCodeGenerator(self, 'python')._specStrParse(specnames, specdict, resname, specials,
                        dovars, dopars, doinps,
                        noreturndefs, forexternal, illegal,
                        ignoreothers, doing_inserts)

    def _parseReusedTermsPy(self, d, symbol_ixs, specials=[],
                        dovars=True, dopars=True, doinps=True, illegal=[]):
        return CG.getCodeGenerator(self, 'python')._parseReusedTermsPy(d, symbol_ixs, specials,
                        dovars, dopars, doinps, illegal)

    def _processSpecialC(self, specStr):
        return CG.getCodeGenerator(self, 'c')._processSpecialC(specStr)

# Sub-classes of FuncSpec
class RHSfuncSpec(FuncSpec):
    """Right-hand side definition for vars defined."""

    def __init__(self, kw):
        FuncSpec.__init__(self, kw)


class ExpFuncSpec(FuncSpec):
    """Explicit definition of vars defined."""

    def __init__(self, kw):
        assert 'codeinsert_start' not in kw, ('code inserts invalid for '
                                            'explicit function specification')
        assert 'codeinsert_end' not in kw, ('code inserts invalid for '
                                            'explicit function specification')
        FuncSpec.__init__(self, kw)


class ImpFuncSpec(FuncSpec):
    """Assumes this will be set to equal zero when solving for vars defined."""

    # funcspec will possibly be the same for several variables
    # so it's repeated, but must be checked so that only solved
    # once for all relevant variables
    def __init__(self, kw):
        assert 'codeinsert_start' not in kw, ('code inserts invalid for '
                                            'implicit function specification')
        assert 'codeinsert_end' not in kw, ('code inserts invalid for '
                                            'implicit function specification')
        FuncSpec.__init__(self, kw)


# ----------------------------------------------
## Public exported functions
# ----------------------------------------------
def makePartialJac(spec_pair, varnames, select=None):
    """Use this when parameters have been added to a modified Generator which
    might clash with aux fn argument names. (E.g., used by find_nullclines).

    'select' option (list of varnames) selects those entries from the Jac of the varnames,
       e.g. for constructing Jacobian w.r.t. 'parameters' using a parameter formerly
       a variable (e.g. for find_nullclines).
    """
    fargs, fspec = spec_pair
    J = QuantSpec('J', fspec)
    # find positions of actual varnames in f argument list
    # then extract terms from the Jacobian matrix, simplifying to a scalar if 1D
    dim = len(varnames)
    if J.dim == dim:
        # nothing to do
        return (fargs, fspec)
    assert J.dim > dim, "Cannot add variable names to system while using its old Jacobian aux function"
    assert remain(varnames, fargs) == [], "Invalid variable names to resolve Jacobian aux function"
    assert fargs[0] == 't'
    # -1 adjusts for 't' being the first argument
    vixs = [fargs.index(v)-1 for v in varnames]
    vixs.sort()
    if select is None:
        select = varnames
        sixs = vixs
    else:
        sixs = [fargs.index(v)-1 for v in select]
    if dim == 1:
        fspec = str(J.fromvector(vixs[0]).fromvector(sixs[0]))
    else:
        terms = []
        for i in vixs:
            Ji = J.fromvector(i)
            subterms = []
            for j in sixs:
                subterms.append( str(Ji.fromvector(j)) )
            terms.append( "[" + ",".join(subterms) + "]" )
        fspec = "[" + ",".join(terms) + "]"
    # retain order of arguments
    fargs_new = ['t'] + [fargs[ix+1] for ix in vixs]
    return (fargs_new, fspec)


def resolveClashingAuxFnPars(fnspecs, varspecs, parnames):
    """Use this when parameters have been added to a modified Generator which
    might clash with aux fn argument names. (E.g., used by find_nullclines).
    Will remove arguments that are now considered parameters by the system,
    in both the function definitions and their use in specs for the variables.
    """
    changed_fns = []
    new_fnspecs = {}
    for fname, (fargs, fspec) in fnspecs.items():
        common_names = intersect(fargs, parnames)
        if fname in parnames:
            print("Problem with function definition %s" % fname)
            raise ValueError("Unrecoverable clash between parameter names and aux fn name")
        if common_names == []:
            new_fnspecs[fname] = (fargs, fspec)
        else:
            changed_fns.append(fname)
            new_fnspecs[fname] = (remain(fargs, parnames), fspec)

    new_varspecs = {}
    for vname, vspec in varspecs.items():
        q = QuantSpec('__temp__', vspec)
        # only update use of functions both changed and used in the varspecs
        used_fns = intersect(q.parser.tokenized, changed_fns)
        for f in used_fns:
            ix = q.parser.tokenized.index(f)
            # identify arg list for this fn call
            rest = ''.join(q.parser.tokenized[ix+1:])
            end_ix = findEndBrace(rest)
            # get string of this arg list
            argstr = rest[:end_ix+1]
            # split
            success, args_list, arglen = readArgs(argstr)
            assert success, "Parsing arguments failed"
            new_args_list = []
            # remove parnames
            for arg in args_list:
                qarg = QuantSpec('a', arg)
                # if parameter appears in a compound expression in the argument,
                # then we don't know how to process it, so issue warning [was: raise exception]
                if len(qarg.parser.tokenized) > 1:
                    if any([p in qarg for p in parnames]):
                        # do not put raw parameter name arguments into new arg list
                        #raise ValueError("Cannot process argument to aux fn %s"%f)
                        print("Warning: some auxiliary function parameters clash in function %s" %f)
                    new_args_list.append(arg)
                elif arg not in parnames:
                    # do not put raw parameter name arguments into new arg list
                    new_args_list.append(arg)
            new_argstr = ','.join(new_args_list)
            # update vspec and q for next f
            vspec = ''.join(q[:ix+1]) + '(' + new_argstr + ')' + rest[end_ix+1:]
            q = QuantSpec('__temp__', vspec)
        new_varspecs[vname] = vspec
    return new_fnspecs, new_varspecs



def getSpecFromFile(specfilename):
    """Read text specs from a file"""
    try:
        f = open(specfilename, 'r')
        s = f.read()
    except IOError as e:
        print('File error: %s' % str(e))
        raise
    f.close()
    return s
