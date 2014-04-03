"""Functional specification classes.

   Robert Clewley, August 2005.

This module aids in building internal representations of ODEs, etc.,
particularly for the benefit of Automatic Differentiation
and for manipulation of abstraction digraphs.
"""

# PyDSTool imports
from __future__ import division, absolute_import
from .utils import *
from .common import *
from .parseUtils import *
from .errors import *
from .utils import info as utils_info
from .Symbolic import QuantSpec

# Other imports
from copy import copy, deepcopy
import math, random, numpy, scipy, scipy.special
from numpy import any

__all__ = ['RHSfuncSpec', 'ImpFuncSpec', 'ExpFuncSpec', 'FuncSpec',
           'getSpecFromFile', 'resolveClashingAuxFnPars', 'makePartialJac']

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

    def __init__(self, kw):
        # All math package names are reserved
        self._protected_mathnames = protected_mathnames
        self._protected_randomnames = protected_randomnames
        self._protected_scipynames = protected_scipynames
        self._protected_specialfns = protected_specialfns
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
        self._initargs = deepcopy(kw)
        # PROCESS NECESSARY KEYS -------------------
        try:
            # spec name
            if 'name' in kw:
                self.name = kw['name']
            else:
                self.name = 'untitled'
            # declare variables (name list)
            if isinstance(kw['vars'], list):
                vars = kw['vars'][:]  # take copy
            else:
                assert isinstance(kw['vars'], str), 'Invalid variable name'
                vars = [kw['vars']]
        except KeyError:
            raise PyDSTool_KeyError('Necessary keys missing from argument dict')
        foundKeys = len(needKeys)
        # PROCESS OPTIONAL KEYS --------------------
        # declare pars (name list)
        if 'pars' in kw:
            if isinstance(kw['pars'], list):
                pars = kw['pars'][:]  # take copy
            else:
                assert isinstance(kw['pars'], str), 'Invalid parameter name'
                pars = [kw['pars']]
            foundKeys += 1
        else:
            pars = []
        # declare external inputs (name list)
        if 'inputs' in kw:
            if isinstance(kw['inputs'], list):
                inputs = kw['inputs'][:]   # take copy
            else:
                assert isinstance(kw['inputs'], str), 'Invalid input name'
                inputs = [kw['inputs']]
            foundKeys += 1
        else:
            inputs = []
        if 'targetlang' in kw:
            try:
                tlang = kw['targetlang'].lower()
            except AttributeError:
                raise TypeError("Expected string type for target language")
            if tlang not in targetLangs:
                raise ValueError('Invalid specification for targetlang')
            self.targetlang = tlang
            foundKeys += 1
        else:
            self.targetlang = 'python'  # default
        if self.targetlang == 'c':
            self._defstr = "#define"
            self._undefstr = "#undef"
        else:
            self._defstr = ""
            self._undefstr = ""
        if 'ignorespecial' in kw:
            self._ignorespecial = kw['ignorespecial']
            foundKeys += 1
        else:
            self._ignorespecial = []
        # ------------------------------------------
        # reusable terms in function specs
        if 'reuseterms' in kw:
            if isinstance(kw['reuseterms'], dict):
                self.reuseterms = deepcopy(kw['reuseterms'])
            else:
                raise ValueError('reuseterms must be a dictionary of strings ->'
                                   ' replacement strings')
            ignore_list = []
            for term, repterm in self.reuseterms.iteritems():
                assert isinstance(term, str), \
                       "terms in 'reuseterms' dictionary must be strings"
                # if term[0] in num_chars+['.']:
                #     raise ValueError('terms must not be numerical values')
                if isNumericToken(term):
                    # don't replace numeric terms (sometimes these are
                    # generated automatically by Constructors when resolving
                    # explicit variable inter-dependencies)
                    ignore_list.append(term)
                # not sure about the next check any more...
                # what is the point of not allowing subs terms to begin with op?
                if term[0] in '+/*':
                    print "Error in term:", term
                    raise ValueError('terms to be substituted must not begin '
                                     'with arithmetic operators')
                if term[0] == '-':
                    term = '(' + term + ')'
                if term[-1] in '+-/*':
                    print "Error in term:", term
                    raise ValueError('terms to be substituted must not end with '
                                     'arithmetic operators')
                for s in term:
                    if self.targetlang == 'python':
                        if s in '[]{}~@#$%&\|?^': # <>! now OK, e.g. for "if" statements
                            print "Error in term:", term
                            raise ValueError('terms to be substituted must be '
                                'alphanumeric or contain arithmetic operators '
                                '+ - / *')
                    else:
                        if s in '[]{}~!@#$%&\|?><': # removed ^ from this list
                            print "Error in term:", term
                            raise ValueError('terms to be substituted must be alphanumeric or contain arithmetic operators + - / *')
                if repterm[0] in num_chars:
                    print "Error in replacement term:", repterm
                    raise ValueError('replacement terms must not begin with numbers')
                for s in repterm:
                    if s in '+-/*.()[]{}~!@#$%^&\|?><,':
                        print "Error in replacement term:", repterm
                        raise ValueError('replacement terms must be alphanumeric')
            for t in ignore_list:
                del self.reuseterms[t]
            foundKeys += 1
        else:
            self.reuseterms = {}
        # auxiliary variables declaration
        if 'auxvars' in kw:
            if isinstance(kw['auxvars'], list):
                auxvars = kw['auxvars'][:]   # take copy
            else:
                assert isinstance(kw['auxvars'], str), 'Invalid variable name'
                auxvars = [kw['auxvars']]
            foundKeys += 1
        else:
            auxvars = []
        # auxfns dict of functionality for auxiliary functions (in
        # either python or C). for instance, these are used for global
        # time reference, access of regular variables to initial
        # conditions, and user-defined quantities.
        self.auxfns = {}
        if 'fnspecs' in kw:
            self._auxfnspecs = deepcopy(kw['fnspecs'])
            foundKeys += 1
        else:
            self._auxfnspecs = {}
        # spec dict of functionality, as a string for each var
        # (in either python or C, or just for python?)
        assert 'varspecs' in kw or 'spec' in kw, ("Require a functional "
                                "specification key -- 'spec' or 'varspecs'")
        if '_for_macro_info' in kw:
            foundKeys += 1
            self._varsbyforspec = kw['_for_macro_info'].varsbyforspec
        else:
            self._varsbyforspec = {}
        if 'varspecs' in kw:
            if auxvars == []:
                numaux = 0
            else:
                numaux = len(auxvars)
            if '_for_macro_info' in kw:
                if kw['_for_macro_info'].numfors > 0:
                    num_varspecs = numaux + len(vars) - kw['_for_macro_info'].totforvars + \
                                   kw['_for_macro_info'].numfors
                else:
                    num_varspecs = numaux + len(vars)
            else:
                num_varspecs = numaux + len(vars)
            if len(kw['varspecs']) != len(self._varsbyforspec) and \
               len(kw['varspecs']) != num_varspecs:
                print "# state variables: ", len(vars)
                print "# auxiliary variables: ", numaux
                print "# of variable specs: ", len(kw['varspecs'])
                raise ValueError('Incorrect size of varspecs')
            self.varspecs = deepcopy(kw['varspecs'])
            foundKeys += 1  # for varspecs
        else:
            self.varspecs = {}
        self.codeinserts = {'start': '', 'end': ''}
        if 'codeinsert_start' in kw:
            codestr = kw['codeinsert_start']
            assert isinstance(codestr, str), 'code insert must be a string'
            if self.targetlang == 'python':
                # check initial indentation (as early predictor of whether
                # indentation has been done properly)
                if codestr[:4] != _indentstr:
                    codestr = _indentstr+codestr
            # additional spacing in function spec
            if codestr[-1] != '\n':
                addnl = '\n'
            else:
                addnl = ''
            self.codeinserts['start'] = codestr+addnl
            foundKeys += 1
        if 'codeinsert_end' in kw:
            codestr = kw['codeinsert_end']
            assert isinstance(codestr, str), 'code insert must be a string'
            if self.targetlang == 'python':
                # check initial indentation (as early predictor of whether
                # indentation has been done properly)
                assert codestr[:4] == "    ", ("First line of inserted "
                                        "python code at start of spec was "
                                               "wrongly indented")
            # additional spacing in function spec
            if codestr[-1] != '\n':
                addnl = '\n'
            else:
                addnl = ''
            self.codeinserts['end'] = codestr+addnl
            foundKeys += 1
        # spec dict of functionality, as python functions,
        # or the paths/names of C dynamic linked library files
        # can be user-defined or generated from generateSpec
        if 'spec' in kw:
            if 'varspecs' in kw:
                raise PyDSTool_KeyError, \
                      "Cannot provide both 'spec' and 'varspecs' keys"
            assert isinstance(kw['spec'], tuple), ("'spec' must be a pair:"
                                    " (spec body, spec name)")
            assert len(kw['spec'])==2, ("'spec' must be a pair:"
                                    " (spec body, spec name)")
            self.spec = deepcopy(kw['spec'])
            # auxspec not used for explicitly-given specs. it's only for
            # auto-generated python auxiliary variable specs (as py functions)
            self.auxspec = {}
            if 'dependencies' in kw:
                self.dependencies = kw['dependencies']
            else:
                raise PyDSTool_KeyError("Dependencies must be provided "
                         "explicitly when using 'spec' form of initialization")
            foundKeys += 2
        else:
            self.spec = {}
            self.auxspec = {}
            self.dependencies = []
        if len(kw) > foundKeys:
            raise PyDSTool_KeyError('Invalid keys passed in argument dict')
        self.defined = False  # initial value
        self.validateDef(vars, pars, inputs, auxvars, self._auxfnspecs.keys())
        # ... exception if not valid
        # Fine to do the following if we get this far:
        # sort for final order that will be used for determining array indices
        vars.sort()
        pars.sort()
        inputs.sort()
        auxvars.sort()
        self.vars = vars
        self.pars = pars
        self.inputs = inputs
        self.auxvars = auxvars
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
            print "Warning: code insert (start) ignored for new target"
        if self.codeinserts['end'] != '':
            del new_args['codeinsert_end']
            print "Warning: code insert (end) ignored for new target"
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
                            self._protected_specialfns + \
                            self._protected_randomnames + \
                            self._protected_auxnames + \
                            ['abs', 'min', 'max', 'and', 'or', 'not',
                             'True', 'False']
        # other checks
        first_char_check = [alphabet_chars_RE.match(n[0]) \
                                     is not None for n in allnames]
        if not all(first_char_check):
            print "Offending names:", [n for i, n in enumerate(allnames) \
                                       if not first_char_check[i]]
            raise ValueError('Variable, parameter, and input names must not '
                         'begin with non-alphabetic chars')
        protected_overlap = intersect(allnames, allprotectednames)
        if protected_overlap != []:
            print "Overlapping names:", protected_overlap
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
        # Always makes a set of python versions of the functions for future
        # use by user at python level
        if self.targetlang == 'python':
            self._genAuxFnPy(pytarget=True)
        elif self.targetlang == 'c':
            self._genAuxFnC()
            self._genAuxFnPy()
        elif self.targetlang == 'matlab':
            self._genAuxFnMatlab()
            self._genAuxFnPy()
        elif self.targetlang == 'dstool':
            raise NotImplementedError
        elif self.targetlang == 'xpp':
            raise NotImplementedError
        else:
            raise ValueError('targetlang attribute must be in '+str(targetLangs))


    def generateSpec(self):
        """Automatically generate callable target-language functions from
        the user-defined specification strings."""
        if self.targetlang == 'python':
            self._genSpecPy()
        elif self.targetlang == 'c':
            self._genSpecC()
        elif self.targetlang == 'matlab':
            self._genSpecMatlab()
        elif self.targetlang == 'odetools':
            raise NotImplementedError
        elif self.targetlang == 'xpp':
            raise NotImplementedError
        else:
            raise ValueError('targetlang attribute must be in '+str(targetLangs))


    def doPreMacros(self):
        """Pre-process any macro spec definitions (e.g. `for` loops)."""

        assert self.varspecs != {}, 'varspecs attribute must be defined'
        specnames_unsorted = self.varspecs.keys()
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
                           + self._protected_specialfns \
                           + self._protected_macronames \
                           + ['abs', 'and', 'or', 'not', 'True', 'False']
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
                specnames_gen = varspecs.keys()
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
        avoid_toks = []
        for ix, tok in enumerate(q):
            if tok[0] == '[':
                eval_str = tok[1:-1]
                if istr in eval_str:
                    eval_pieces[ix] = eval_str
                # otherwise may be a different, embedded temp index for another
                # sum, etc., so don't touch it
        keys = eval_pieces.keys()
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

    # ----------------- Python specifications ----------------

    def _genAuxFnPy(self, pytarget=False):
        if pytarget:
            assert self.targetlang == 'python', \
               'Wrong target language for this call'
        auxnames = self._auxfnspecs.keys()
        # User aux fn interface
        uafi = {}
        # protectednames = auxnames + self._protected_mathnames + \
        #                  self._protected_randomnames + \
        #                  self._protected_scipynames + \
        #                  self._protected_specialfns + \
        #                  ['abs', 'and', 'or', 'not', 'True', 'False']
        # Deal with built-in auxiliary functions (don't make their names unique)
        # In this version, the textual code here doesn't get executed. Only
        # the function names in the second position of the tuple are needed.
        # Later, the text will probably be removed.
        auxfns = {}
        auxfns['globalindepvar'] = \
                   ("def _auxfn_globalindepvar(ds, parsinps, t):\n" \
                    + _indentstr \
                    + "return ds.globalt0 + t", '_auxfn_globalindepvar')
        auxfns['initcond'] = \
                   ("def _auxfn_initcond(ds, parsinps, varname):\n" \
                    + _indentstr \
                    + "return ds.initialconditions[varname]",'_auxfn_initcond')
        auxfns['heav'] = \
                   ("def _auxfn_heav(ds, parsinps, x):\n" + _indentstr \
                      + "if x>0:\n" + 2*_indentstr \
                      + "return 1\n" + _indentstr + "else:\n" \
                      + 2*_indentstr + "return 0", '_auxfn_heav')
        auxfns['if'] = \
                   ("def _auxfn_if(ds, parsinps, c, e1, e2):\n" \
                    + _indentstr + "if c:\n" + 2*_indentstr \
                    + "return e1\n" + _indentstr \
                    + "else:\n" + 2*_indentstr + "return e2", '_auxfn_if')
        auxfns['getindex'] = \
                   ("def _auxfn_getindex(ds, parsinps, varname):\n" \
                    + _indentstr \
                    + "return ds._var_namemap[varname]", '_auxfn_getindex')
        auxfns['getbound'] = \
                   ("def _auxfn_getbound(ds, parsinps, name, bd):\n" \
                    + _indentstr + "try:\n" \
                    + 2*_indentstr + "return ds.xdomain[name][bd]\n" \
                    + _indentstr + "except KeyError:\n" + 2*_indentstr \
                    + "try:\n" + 3*_indentstr \
                    + "return ds.pdomain[name][bd]\n" + 2*_indentstr \
                    + "except KeyError, e:\n" + 3*_indentstr \
                    + "print 'Invalid var / par name %s'%name,\n" \
                    + 3*_indentstr + "print 'or bounds not well defined:'\n" \
                    + 3*_indentstr + "print ds.xdomain, ds.pdomain\n" \
                    + 3*_indentstr + "raise (RuntimeError, e)",
                    '_auxfn_getbound')
        # the internal functions may be used by user-defined functions,
        # so need them to be accessible to __processTokens when parsing
        self._pyauxfns = auxfns
        # add the user-defined function names for cross-referencing checks
        # (without their definitions)
        for auxname in auxnames:
            self._pyauxfns[auxname] = None
        # don't process the built-in functions -> unique fns because
        # they are global definitions existing throughout the
        # namespace
        self._protected_auxnames.extend(['Jacobian','Jacobian_pars'])
        # protected names are the names that must not be used for
        # user-specified auxiliary fn arguments
        protectednames = self.pars + self.inputs \
                   + ['abs', 'pow', 'and', 'or', 'not', 'True', 'False'] \
                   + self._protected_auxnames + auxnames \
                   + self._protected_scipynames + self._protected_specialfns \
                   + self._protected_macronames + self._protected_mathnames \
                   + self._protected_randomnames + self._protected_reusenames
        ### checks for user-defined auxiliary fns
        # name map for fixing inter-auxfn references
        auxfn_namemap = {}
        specials_base = self.pars + self._protected_auxnames \
                   + ['abs', 'pow', 'and', 'or', 'not', 'True', 'False'] \
                   + auxnames + self._protected_scipynames \
                   + self._protected_specialfns \
                   + self._protected_macronames + self._protected_mathnames \
                   + self._protected_randomnames + self._protected_reusenames
        for auxname in auxnames:
            auxinfo = self._auxfnspecs[auxname]
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
                if not compareList(auxinfo[0],['t']+self.vars):
                    print ['t']+self.vars
                    print "Auxinfo =", auxinfo[0]
                    raise ValueError("Invalid argument list given in Jacobian.")
                auxparlist = ["t","x","parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t","x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = self.vars
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
                reusestr, body_processed_dict = self._processReusedPy(specvars,
                                               specdict,
                                               specials=specials+specials_base)
                body_processed = self._specStrParse(specvars,
                                          body_processed_dict, 'xjac',
                                          specials=specials+specials_base)
                auxstr_py = self._genSpecFnPy('_auxfn_Jac',
                                               reusestr+body_processed,
                                               'xjac', specvars)
                # check Jacobian
                m = n = len(specvars)
                specdict_check = {}.fromkeys(specvars)
                for specname in specvars:
                    temp = body_processed_dict[specname]
                    specdict_check[specname] = \
                            count_sep(temp.replace("[","").replace("]",""))+1
                body_processed = ""
                for row in range(m):
                    if specdict_check[specvars[row]] != n:
                        print "Row %i: "%m, specdict[specvars[row]]
                        print "Found length %i"%specdict_check[specvars[row]]
                        raise ValueError("Jacobian should be %sx%s"%(m,n))
            elif auxname == 'Jacobian_pars':
                if not compareList(auxinfo[0],['t']+self.pars):
                    print ['t']+self.pars
                    print "Auxinfo =", auxinfo[0]
                    raise ValueError("Invalid argument list given in Jacobian.")
                auxparlist = ["t","x","parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t","x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = self.pars
                specvars.sort()
                specdict = {}.fromkeys(self.vars)
                if len(specvars) == len(self.vars) == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in Jacobian for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in Jacobian for 1D system"
                    specdict[specvars[0]] = auxstr
                else:
                    specdict = parseMatrixStrToDictStr(auxstr, self.vars)
                reusestr, body_processed_dict = self._processReusedPy(self.vars,
                                               specdict,
                                               specials=specials+specials_base)
                body_processed = self._specStrParse(self.vars,
                                          body_processed_dict, 'pjac',
                                          specials=specials+specials_base)
                auxstr_py = self._genSpecFnPy('_auxfn_Jac_p',
                                               reusestr+body_processed,
                                               'pjac', self.vars)
                # check Jacobian
                n = len(specvars)
                m = len(self.vars)
                specdict_check = {}.fromkeys(self.vars)
                for specname in self.vars:
                    temp = body_processed_dict[specname]
                    specdict_check[specname] = \
                            count_sep(temp.replace("[","").replace("]",""))+1
                body_processed = ""
                for row in range(m):
                    try:
                        if specdict_check[self.vars[row]] != n:
                            print "Row %i: "%m, specdict[self.vars[row]]
                            print "Found length %i"%specdict_check[self.vars[row]]
                            raise ValueError("Jacobian w.r.t. pars should be %sx%s"%(m,n))
                    except IndexError:
                        print "\nFound:\n"
                        info(specdict)
                        raise ValueError("Jacobian w.r.t. pars should be %sx%s"%(m,n))
            elif auxname == 'massMatrix':
                if not compareList(auxinfo[0],['t']+self.vars):
                    print ['t']+self.vars
                    print "Auxinfo =", auxinfo[0]
                    raise ValueError("Invalid argument list given in Mass Matrix.")
                auxparlist = ["t","x","parsinps"]
                # special symbols to allow in parsing function body
                specials = ["t","x"]
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                specvars = self.vars
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
                reusestr, body_processed_dict = self._processReusedPy(specvars,
                                               specdict,
                                               specials=specials+specials_base)
                body_processed = self._specStrParse(specvars,
                                          body_processed_dict, 'xmat',
                                          specials=specials+specials_base)
                auxstr_py = self._genSpecFnPy('_auxfn_massMatrix',
                                               reusestr+body_processed,
                                               'xmat', specvars)
                # check matrix
                m = n = len(specvars)
                specdict_check = {}.fromkeys(specvars)
                for specname in specvars:
                    specdict_check[specname] = 1 + \
                        count_sep(body_processed_dict[specname].replace("[","").replace("]",""))
                body_processed = ""
                for row in range(m):
                    if specdict_check[specvars[row]] != n:
                        print "Row %i: "%m, specdict[specvars[row]]
                        print "Found length %i"%specdict_check[specvars[row]]
                        raise ValueError("Mass matrix should be %sx%s"%(m,n))
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
                            +'):\n'
                auxparlist = auxparstr.replace(" ","").split(",")
                badparnames = intersect(auxparlist,
                                        remain(protectednames,auxnames))
                if badparnames != []:
                    print "Bad parameter names in auxiliary function", \
                            auxname, ":", badparnames
                    #print auxinfo[0]
                    #print auxparlist
                    raise ValueError("Cannot use protected names (including" \
                        " globally visible system parameters for auxiliary " \
                                  "function arguments")
                # special symbols to allow in parsing function body
                specials = auxparlist
                specials.remove('parsinps')
                illegalterms = remain(self.vars + self.auxvars, specials)
                auxstr = auxinfo[1]
                if any([pt in auxstr for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                reusestr, body_processed_dict = self._processReusedPy([auxname],
                                               {auxname:auxstr},
                                               specials=specials+specials_base,
                                               dovars=False,
                                               illegal=illegalterms)
                body_processed = self._specStrParse([auxname],
                                          body_processed_dict,
                                          specials=specials+specials_base,
                                          dovars=False,
                                          noreturndefs=True,
                                          illegal=illegalterms)
                auxstr_py += reusestr + _indentstr + 'return ' \
                          + body_processed
            # syntax validation done in makeUniqueFn
            try:
                auxfns[auxname] = makeUniqueFn(auxstr_py)
                # Note: this automatically updates self._pyauxfns too
            except:
                print 'Error in supplied auxiliary spec dictionary code'
                raise
            auxfn_namemap['ds.'+auxname] = 'ds.'+auxfns[auxname][1]
            # prepare user-interface wrapper function (not method)
            if specials == [''] or specials == []:
                fn_args = ''
            else:
                fn_args = ','+','.join(specials)
            fn_elts = ['def ', auxname, '(self', fn_args,
                       ',__parsinps__=None):\n\t', 'if __parsinps__ is None:\n\t\t',
                       '__parsinps__=self.map_ixs(self.genref)\n\t',
                       'return self.genref.', auxfns[auxname][1],
                       '(__parsinps__', fn_args, ')\n']
            uafi[auxname] =  ''.join(fn_elts)
        # resolve inter-auxiliary function references
        for auxname, auxspec in auxfns.iteritems():
            dummyQ = QuantSpec('dummy', auxspec[0], preserveSpace=True,
                               treatMultiRefs=False)
            dummyQ.mapNames(auxfn_namemap)
            auxfns[auxname] = (dummyQ(), auxspec[1])
        if pytarget:
            self.auxfns = auxfns
        # keep _pyauxfns handy for users to access python versions of functions
        # from python, even using non-python target languages
        #
        # Changes to auxfns was already changing self._pyauxfns so the following line
        # is not needed
        #self._pyauxfns.update(auxfns)  # same thing if pytarget==True
        self._user_auxfn_interface = uafi
        self._protected_auxnames.extend(auxnames)



    def _genSpecFnPy(self, name, specstr, resname, specnames,
                     docodeinserts=False):
        # Set up function header
        retstr = 'def '+name+'(ds, t, x, parsinps):\n' #    print t, x, parsinps\n'
        # add arbitrary code inserts, if present and option is switched on
        # (only used for vector field definitions)
        lstart = len(self.codeinserts['start'])
        lend = len(self.codeinserts['end'])
        if docodeinserts:
            if lstart>0:
                start_code = self._specStrParse(['inserts'],
                               {'inserts':self.codeinserts['start']}, '',
                                noreturndefs=True, ignoreothers=True,
                                doing_inserts=True)
            else:
                start_code = ''
            if lend > 0:
                end_code = self._specStrParse(['inserts'],
                               {'inserts':self.codeinserts['end']}, '',
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


    def _genSpecPy(self):
        assert self.targetlang == 'python', ('Wrong target language for this'
                                             ' call')
        assert self.varspecs != {}, 'varspecs attribute must be defined'
        specnames_unsorted = self.varspecs.keys()
        _vbfs_inv = invertMap(self._varsbyforspec)
        # Process state variable specifications
        if len(_vbfs_inv) > 0:
            specname_vars = []
            specname_auxvars = []
            for varname in self.vars:
                # check if varname belongs to a for macro grouping in self.varspecs
                specname = _vbfs_inv[varname]
                if varname not in specname_vars:
                    specname_vars.append(varname)
            for varname in self.auxvars:
                # check if varname belongs to a for macro grouping in self.varspecs
                specname = _vbfs_inv[varname]
                if varname not in specname_auxvars:
                    specname_auxvars.append(varname)
        else:
            specname_vars = intersect(self.vars, specnames_unsorted)
            specname_auxvars = intersect(self.auxvars, specnames_unsorted)
        specname_vars.sort()
        for vn, vs in self.varspecs.items():
            if any([pt in vs for pt in ('^', '**')]):
                self.varspecs[vn] = convertPowers(vs, 'pow')
        self.vars.sort()
        reusestr, specupdated = self._processReusedPy(specname_vars,
                                                      self.varspecs)
        self.varspecs.update(specupdated)
        temp = self._specStrParse(specname_vars, self.varspecs, 'xnew')
        specstr_py = self._genSpecFnPy('_specfn', reusestr+temp, 'xnew',
                                       specname_vars, docodeinserts=True)
        # Process auxiliary variable specifications
        specname_auxvars.sort()
        assert self.auxvars == specname_auxvars, \
                   ('Mismatch between declared auxiliary'
                    ' variable names and varspecs keys')
        reusestraux, specupdated = self._processReusedPy(specname_auxvars,
                                                         self.varspecs)
        self.varspecs.update(specupdated)
        tempaux = self._specStrParse(specname_auxvars, self.varspecs, 'auxvals')
        auxspecstr_py = self._genSpecFnPy('_auxspecfn', reusestraux+tempaux,
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
        self.spec = spec_info
        self.auxspec = auxspec_info


    def _processReusedPy(self, specnames, specdict, specials=[],
                        dovars=True, dopars=True, doinps=True, illegal=[]):
        """Process reused subexpression terms for Python code."""

        reused, specupdated, new_protected, order = _processReused(specnames,
                                                        specdict,
                                                        self.reuseterms,
                                                        _indentstr)
        self._protected_reusenames = new_protected
        # symbols to parse are at indices 2 and 4 of 'reused' dictionary
        reusedParsed = self._parseReusedTermsPy(reused, [2,4],
                                        specials=specials, dovars=dovars,
                                        dopars=dopars, doinps=doinps,
                                                  illegal=illegal)
        reusedefs = {}.fromkeys(new_protected)
        for vname, deflist in reusedParsed.iteritems():
            for d in deflist:
                reusedefs[d[2]] = d
        return (concatStrDict(reusedefs, intersect(order,reusedefs.keys())),
                       specupdated)


    def _parseReusedTermsPy(self, d, symbol_ixs, specials=[],
                        dovars=True, dopars=True, doinps=True, illegal=[]):
        """Process dictionary of reused term definitions (in spec syntax)."""
        # ... to parse special symbols to actual Python.
        # expect symbols to be processed at d list's entries given in
        # symbol_ixs.
        allnames = self.vars + self.pars + self.inputs + self.auxvars \
                   + ['abs'] + self._protected_auxnames \
                   + self._protected_scipynames + self._protected_specialfns \
                   + self._protected_macronames + self._protected_mathnames \
                   + self._protected_randomnames + self._protected_reusenames
        allnames = remain(allnames, illegal)
        if dovars:
            var_arrayixstr = dict(zip(self.vars, map(lambda i: str(i), \
                                     range(len(self.vars))) ))
            aux_arrayixstr = dict(zip(self.auxvars, map(lambda i: str(i), \
                                     range(len(self.auxvars))) ))
        else:
            var_arrayixstr = {}
            aux_arrayixstr = {}
        if dopars:
            if doinps:
                # parsinps_names is pars and inputs, each sorted
                # *individually*
                parsinps_names = self.pars+self.inputs
            else:
                parsinps_names = self.pars
            parsinps_arrayixstr = dict(zip(parsinps_names,
                                        map(lambda i: str(i), \
                                        range(len(parsinps_names))) ))
        else:
            parsinps_names = []
            parsinps_arrayixstr = {}
        specialtokens = remain(allnames,specials) + ['(', 't'] + specials
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
                    parsedsymbol = self.__processTokens(allnames,
                                    specialtokens, symbol,
                                    var_arrayixstr, aux_arrayixstr,
                                    parsinps_names, parsinps_arrayixstr,
                                    specname)
                    # must strip possible trailing whitespace!
                    d[specname][listix][ix] = parsedsymbol.strip()
        return d


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
        allnames = self.vars + self.pars + self.inputs + self.auxvars \
                   + ['abs', 'and', 'or', 'not', 'True', 'False'] \
                   + self._protected_auxnames \
                   + self._protected_scipynames + self._protected_specialfns \
                   + self._protected_macronames + self._protected_mathnames \
                   + self._protected_randomnames + self._protected_reusenames
        allnames = remain(allnames, illegal)
        if dovars:
            if forexternal:
                var_arrayixstr = dict(zip(self.vars,
                                          ["'"+v+"'" for v in self.vars]))
                aux_arrayixstr = dict(zip(self.auxvars,
                                          ["'"+v+"'" for v in self.auxvars]))
            else:
                var_arrayixstr = dict(zip(self.vars, map(lambda i: str(i), \
                                         range(len(self.vars))) ))
                aux_arrayixstr = dict(zip(self.auxvars, map(lambda i: str(i),\
                                         range(len(self.auxvars))) ))
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
                    parsinps_names = self.pars+self.inputs
                else:
                    parsinps_names = self.pars
                # for external calls we want parname -> 'parname'
                parsinps_arrayixstr = dict(zip(parsinps_names,
                                       ["'"+pn+"'" for pn in parsinps_names]))
            else:
                if doinps:
                    # parsinps_names is pars and inputs, each sorted
                    # *individually*
                    parsinps_names = self.pars+self.inputs
                else:
                    parsinps_names = self.pars
                parsinps_arrayixstr = dict(zip(parsinps_names,
                                            map(lambda i: str(i), \
                                            range(len(parsinps_names))) ))
        else:
            parsinps_names = []
            parsinps_arrayixstr = {}
        specialtokens = remain(allnames,specials) + ['(', 't'] \
                        + remain(specials,['t'])
        specstr_lang = ''
        specname_count = 0
        for specname in specnames:
            specstr = specdict[specname]
            assert type(specstr)==str, "Specification for %s was not a string"%specname
            if not noreturndefs:
                specstr_lang += _indentstr + resname+str(specname_count)+' = '
            specname_count += 1
            specstr_lang += self.__processTokens(allnames, specialtokens,
                                    specstr, var_arrayixstr,
                                    aux_arrayixstr, parsinps_names,
                                    parsinps_arrayixstr, specname, ignoreothers,
                                    doing_inserts)
            if not noreturndefs or not forexternal:
                specstr_lang += '\n'  # prepare for next line
        return specstr_lang


    def __processTokens(self, allnames, specialtokens, specstr,
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
                    print "Error in specification `" + specname + \
                          "` with token `"+s+"` :\n", specstr
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
                            print "Spec name:", specname
                            print "Spec string:", specstr
                            print "Problem symbol:", s
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
                    elif s in self._protected_specialfns:
                        if self.targetlang != 'python':
                            print "Function %s is currently not supported "%s, \
                                "outside of python target language definitions"
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
                                specialtokens_temp = filter(filtfunc,
                                                        specialtokens+self._ignorespecial)
                            else:
                                specialtokens_temp = specialtokens+self._ignorespecial
                            if s == 'if':
                                # hack for special 'if' case
                                # read contents of braces
                                endargbrace = findEndBrace(specstr[scount:]) \
                                                 + scount + 1
                                argstr = specstr[scount:endargbrace]
                                procstr = self.__processTokens(allnames,
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
                                returnstr += self.__processTokens(allnames,
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


    # --------------------- C code specifications -----------------------

    def _processReusedC(self, specnames, specdict):
        """Process reused subexpression terms for C code."""

        if self.auxfns:
            def addParToCall(s):
                return addArgToCalls(self._processSpecialC(s),
                                      self.auxfns.keys(), "p_, wk_, xv_")
            parseFunc = addParToCall
        else:
            parseFunc = self._processSpecialC
        reused, specupdated, new_protected, order = _processReused(specnames,
                                                          specdict,
                                                          self.reuseterms,
                                                          '', 'double', ';',
                                                          parseFunc)
        self._protected_reusenames = new_protected
        reusedefs = {}.fromkeys(new_protected)
        for vname, deflist in reused.iteritems():
            for d in deflist:
                reusedefs[d[2]] = d
        return (concatStrDict(reusedefs, intersect(order, reusedefs.keys())),
                       specupdated)


    def _genAuxFnC(self):
        auxnames = self._auxfnspecs.keys()
        # parameter and variable definitions
        # sorted version of var and par names sorted version of par
        # names (vars not #define'd in aux functions unless Jacobian)
        vnames = self.vars
        pnames = self.pars
        vnames.sort()
        pnames.sort()
        for auxname in auxnames:
            assert auxname not in ['auxvars', 'vfieldfunc'], \
               ("auxiliary function name '" +auxname+ "' clashes with internal"
                " names")
        # must add parameter argument so that we can name
        # parameters inside the functions! this would either
        # require all calls to include this argument (yuk!) or
        # else we add these extra parameters automatically to
        # every call found in the .c code (as is done currently.
        # this is still an untidy solution, but there you go...)
        for auxname in auxnames:
            auxspec = self._auxfnspecs[auxname]
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
                if not compareList(auxspec[0],['t']+self.vars):
                    print ['t']+self.vars
                    print "Auxspec =", auxspec[0]
                    raise ValueError("Invalid argument list given in Jacobian.")
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxspec[1], 'pow')
                else:
                    auxstr = auxspec[1]
                parlist = "unsigned n_, unsigned np_, double t, double *Y_,"
                ismat = True
                sig += parlist + " double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
                specvars = self.vars
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
                reusestr, body_processed_dict = self._processReusedC(specvars,
                                                   specdict_temp)
                specdict = {}.fromkeys(specvars)
                for specname in specvars:
                    temp = body_processed_dict[specname]
                    specdict[specname] = splitargs(temp.replace("[","").replace("]",""))
                body_processed = ""
                # C integrators expect column-major matrices
                for col in range(n):
                    for row in range(m):
                        try:
                            body_processed += "f_[" + str(col) + "][" + str(row) \
                            + "] = " + specdict[specvars[row]][col] + ";\n"
                        except IndexError:
                            raise ValueError("Jacobian should be %sx%s"%(m,n))
                body_processed += "\n"
                auxspec_processedDict = {auxname: body_processed}
            elif auxname == 'Jacobian_pars':
                sig = "void jacobianParam("
                if not compareList(auxspec[0],['t']+self.pars):
                    print ['t']+self.pars
                    print "Auxspec =", auxspec[0]
                    raise ValueError("Invalid argument list given in Jacobian.")
                parlist = "unsigned n_, unsigned np_, double t, double *Y_,"
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxspec[1], 'pow')
                else:
                    auxstr = auxspec[1]
                ismat = True
                # specials = ["t","Y_","n_","np_","wkn_","wk_"]
                sig += parlist + " double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
                specvars = self.pars
                specvars.sort()
                n = len(specvars)
                if n == 0:
                    raise ValueError("Cannot have a Jacobian w.r.t. pars"
                                     " because no pars are defined")
                m = len(self.vars)
                specdict_temp = {}.fromkeys(self.vars)
                if m == n == 1:
                    assert '[' not in auxstr, \
                           "'[' character invalid in Jacobian for 1D system"
                    assert ']' not in auxstr, \
                           "']' character invalid in Jacobian for 1D system"
                    specdict_temp[self.vars.values()[0]] = auxstr
                else:
                    specdict_temp = parseMatrixStrToDictStr(auxstr, self.vars, m)
                reusestr, body_processed_dict = self._processReusedC(self.vars,
                                                   specdict_temp)
                specdict = {}.fromkeys(self.vars)
                for specname in self.vars:
                    temp = body_processed_dict[specname]
                    specdict[specname] = splitargs(temp.replace("[","").replace("]",""))
                body_processed = ""
                # C integrators expect column-major matrices
                for col in range(n):
                    for row in range(m):
                        try:
                            body_processed += "f_[" + str(col) + "][" + str(row) \
                            + "] = " + specdict[self.vars[row]][col] + ";\n"
                        except (IndexError, KeyError):
                            print n, specvars
                            print "\nFound matrix:\n"
                            info(specdict)
                            raise ValueError("Jacobian should be %sx%s"%(m,n))
                body_processed += "\n"
                auxspec_processedDict = {auxname: body_processed}
            elif auxname == 'massMatrix':
                sig = "void massMatrix("
                if not compareList(auxspec[0],['t']+self.vars):
                    raise ValueError("Invalid argument list given in Mass Matrix.")
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxspec[1], 'pow')
                else:
                    auxstr = auxspec[1]
                parlist = "unsigned n_, unsigned np_,"
                ismat = True
                # specials = ["n_","np_","wkn_","wk_"]
                sig += parlist + " double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
                specvars = self.vars
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
                    specdict_temp = parseMatrixStrToDictStr(auxstr, specvars, m)
                reusestr, body_processed_dict = self._processReusedC(specvars,
                                                   specdict_temp)
                specdict = {}.fromkeys(specvars)
                for specname in specvars:
                    temp = body_processed_dict[specname].replace("[","").replace("]","")
                    specdict[specname] = splitargs(temp)
                body_processed = ""
                # C integrators expect column-major matrices
                for col in range(n):
                    for row in range(m):
                        try:
                            body_processed += "f_[" + str(col) + "][" + str(row) \
                            + "] = " + specdict[specvars[row]][col] + ";\n"
                        except KeyError:
                            raise ValueError("Mass matrix should be %sx%s"%(m,n))
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
                    namemap[parname] = '__'+parname+'__'
                sig += parlist + "double *p_, double *wk_, double *xv_)"
                auxstr = auxspec[1]
                if any([pt in auxspec[1] for pt in ('^', '**')]):
                    auxstr = convertPowers(auxstr, 'pow')
                prep_auxstr = self._processSpecialC(auxstr)
                prep_auxstr_quant = QuantSpec('prep_q',
                                  prep_auxstr.replace(' ','').replace('\n',''),
                                  treatMultiRefs=False, preserveSpace=True)
                # have to do name map now in case function's formal arguments
                # coincide with state variable names, which may get tied up
                # in reused terms and not properly matched to the formal args.
                prep_auxstr_quant.mapNames(namemap)
                auxspec = (auxspec[0], prep_auxstr_quant())
                reusestr, auxspec_processedDict = self._processReusedC([auxname],
                                                     {auxname:auxspec[1]})
                # addition of parameter done in Generator code
                # dummyQ = QuantSpec('dummy', auxspec_processedDict[auxname])
                # auxspec_processed = ""
                # # add pars argument to inter-aux fn call
                # auxfn_found = False   # then expect a left brace next
                # for tok in dummyQ:
                #     if auxfn_found:
                #         # expect left brace in this tok
                #         if tok == '(':
                #             auxspec_processed += tok + 'p_, '
                #             auxfn_found = False
                #         else:
                #             raise ValueError("Problem parsing inter-auxiliary"
                #                              " function call")
                #     elif tok in self.auxfns and tok not in \
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
            body_processed = "return "*(not ismat) + dummyQ() + ";\n\n"
            # auxspecstr = sig + " {\n\n" + pardefines + vardefines*ismat \
            auxspecstr = sig + " {\n\n" \
                + "\n" + (len(reusestr)>0)*"/* reused term definitions */\n" \
                + reusestr + (len(reusestr)>0)*"\n" + body_processed \
                + "}"
               # + parundefines + varundefines*ismat + "}"
            # sig as second entry, whereas Python-coded specifications
            # have the fn name there
            self.auxfns[auxname] = (auxspecstr, sig)
        # Don't apply #define's for built-in functions
        self.auxfns['heav'] = ("int heav(double x_, double *p_, double *wk_, double *xv_) {\n" \
                             + "  if (x_>0.0) {return 1;} else {return 0;}\n}",
                  "int heav(double x_, double *p_, double *wk_, double *xv_)")
        self.auxfns['__rhs_if'] = ("double __rhs_if(int cond_, double e1_, " \
                        + "double e2_, double *p_, double *wk_, double *xv_) {\n" \
                        + "  if (cond_) {return e1_;} else {return e2_;};\n}",
              "double __rhs_if(int cond_, double e1_, double e2_, double *p_, double *wk_, double *xv_)")
        self.auxfns['__maxof2'] = ("double __maxof2(double e1_, double e2_, double *p_, double *wk_, double *xv_) {\n" \
                                + "if (e1_ > e2_) {return e1_;} else {return e2_;};\n}",
                "double __maxof2(double e1_, double e2_, double *p_, double *wk_, double *xv_)")
        self.auxfns['__minof2'] = ("double __minof2(double e1_, double e2_, double *p_, double *wk_, double *xv_) {\n" \
                                + "if (e1_ < e2_) {return e1_;} else {return e2_;};\n}",
                "double __minof2(double e1_, double e2_, double *p_, double *wk_, double *xv_)")
        self.auxfns['__maxof3'] = ("double __maxof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_) {\n" \
                               + "double temp_;\nif (e1_ > e2_) {temp_ = e1_;} else {temp_ = e2_;};\n" \
                               + "if (e3_ > temp_) {return e3_;} else {return temp_;};\n}",
                "double __maxof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_)")
        self.auxfns['__minof3'] = ("double __minof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_) {\n" \
                               + "double temp_;\nif (e1_ < e2_) {temp_ = e1_;} else {temp_ = e2_;};\n" \
                               + "if (e3_ < temp_) {return e3_;} else {return temp_;};\n}",
                "double __minof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_)")
        self.auxfns['__maxof4'] = ("double __maxof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_) {\n" \
                               + "double temp_;\nif (e1_ > e2_) {temp_ = e1_;} else {temp_ = e2_;};\n" \
                               + "if (e3_ > temp_) {temp_ = e3_;};\nif (e4_ > temp_) {return e4_;} else {return temp_;};\n}",
                "double __maxof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_)")
        self.auxfns['__minof4'] = ("double __minof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_) {\n" \
                               + "double temp_;\nif (e1_ < e2_) {temp_ = e1_;} else {temp_ = e2_;};\n" \
                               + "if (e3_ < temp_) {temp_ = e3_;};\nif (e4_ < temp_) {return e4_;} else {return temp_;};\n}",
                "double __minof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_)")
        # temporary placeholders for these built-ins...
        cases_ic = ""
        cases_index = ""
        for i in xrange(len(self.vars)):
            if i == 0:
                command = 'if'
            else:
                command = 'else if'
            vname = self.vars[i]
            cases_ic += "  " + command + " (strcmp(varname, " + '"' + vname + '"'\
                     + ")==0)\n\treturn gICs[" + str(i) + "];\n"
            cases_index += "  " + command + " (strcmp(name, " + '"' + vname + '"'\
                     + ")==0)\n\treturn " + str(i) + ";\n"
        # add remaining par names for getindex
        for i in xrange(len(self.pars)):
            pname = self.pars[i]
            cases_index += "  else if" + " (strcmp(name, " + '"' + pname + '"'\
                           +")==0)\n\treturn " + str(i+len(self.vars)) + ";\n"
        cases_ic += """  else {\n\tfprintf(stderr, "Invalid variable name %s for """ \
                 + """initcond call\\n", varname);\n\treturn 0.0/0.0;\n\t}\n"""
        cases_index += """  else {\n\tfprintf(stderr, "Invalid name %s for """ \
                 + """getindex call\\n", name);\n\treturn 0.0/0.0;\n\t}\n"""
        self.auxfns['initcond'] = ("double initcond(char *varname, double *p_, double *wk_, double *xv_) {\n" \
                                   + "\n" + cases_ic + "}",
                                   'double initcond(char *varname, double *p_, double *wk_, double *xv_)')
        self.auxfns['getindex'] = ("int getindex(char *name, double *p_, double *wk_, double *xv_) {\n" \
                                   + "\n" + cases_index + "}",
                                   'int getindex(char *name, double *p_, double *wk_, double *xv_)')
        self.auxfns['globalindepvar'] = ("double globalindepvar(double t, double *p_, double *wk_, double *xv_)" \
                                          + " {\n  return globalt0+t;\n}",
                                         'double globalindepvar(double t, double *p_, double *wk_, double *xv_)')
        self.auxfns['getbound'] = \
                    ("double getbound(char *name, int which_bd, double *p_, double *wk_, double *xv_) {\n" \
                     + "  return gBds[which_bd][getindex(name)];\n}",
                 'double getbound(char *name, int which_bd, double *p_, double *wk_, double *xv_)')


    def _genSpecC(self):
        assert self.targetlang == 'c', ('Wrong target language for this'
                                             ' call')
        assert self.varspecs != {}, 'varspecs attribute must be defined'
        specnames_unsorted = self.varspecs.keys()
        _vbfs_inv = invertMap(self._varsbyforspec)
        # Process state variable specifications
        if len(_vbfs_inv) > 0:
            specname_vars = []
            specname_auxvars = []
            for varname in self.vars:
                # check if varname belongs to a for macro grouping in self.varspecs
                specname = _vbfs_inv[varname]
                if varname not in specname_vars:
                    specname_vars.append(varname)
            for varname in self.auxvars:
                # check if varname belongs to a for macro grouping in self.varspecs
                specname = _vbfs_inv[varname]
                if varname not in specname_auxvars:
                    specname_auxvars.append(varname)
        else:
            specname_vars = intersect(self.vars, specnames_unsorted)
            specname_auxvars = intersect(self.auxvars, specnames_unsorted)
        specname_vars.sort()
        # sorted version of var and par names
        vnames = specname_vars
        pnames = self.pars
        inames = self.inputs
        pnames.sort()
        inames.sort()
        pardefines = ""
        vardefines = ""
        inpdefines = ""
        parundefines = ""
        varundefines = ""
        inpundefines = ""
        # produce vector field specification
        assert self.vars == specname_vars, ('Mismatch between declared '
                                        ' variable names and varspecs keys')
        valid_depTargNames = self.inputs+self.vars+self.auxvars
        for specname, specstr in self.varspecs.iteritems():
            assert type(specstr)==str, "Specification for %s was not a string"%specname
            if any([pt in specstr for pt in ('^', '**')]):
                specstr = convertPowers(specstr, 'pow')
            specQS = QuantSpec('__spectemp__',  specstr)
            for s in specQS:
                if s in valid_depTargNames and (specname, s) not in \
                       self.dependencies: # and specname != s:
                    self.dependencies.append((specname, s))
        # pre-process reused sub-expression dictionary to adapt for
        # known calling sequence in C
        reusestr, specupdated = self._processReusedC(specname_vars,
                                                     self.varspecs)
        self.varspecs.update(specupdated)
        specstr_C = self._genSpecFnC('vfieldfunc', reusestr, specname_vars,
                                       pardefines, vardefines, inpdefines,
                                       parundefines, varundefines, inpundefines,
                                       True)
        self.spec = specstr_C
        # produce auxiliary variables specification
        specname_auxvars.sort()
        assert self.auxvars == specname_auxvars, \
                   ('Mismatch between declared auxiliary'
                    ' variable names and varspecs keys')
        if self.auxvars != []:
            reusestraux, specupdated = self._processReusedC(specname_auxvars,
                                                        self.varspecs)
            self.varspecs.update(specupdated)
        if self.auxvars == []:
            auxspecstr_C = self._genSpecFnC('auxvars', '',
                                        specname_auxvars,
                                        '', '', '',
                                        '', '', '', False)
        else:
            auxspecstr_C = self._genSpecFnC('auxvars', reusestraux,
                                        specname_auxvars, pardefines,
                                        vardefines, inpdefines, parundefines,
                                        varundefines, inpundefines,
                                        False)
        self.auxspec = auxspecstr_C


    def _genSpecFnC(self, funcname, reusestr, specnames, pardefines,
                    vardefines, inpdefines, parundefines, varundefines,
                    inpundefines, docodeinserts):
        sig = "void " + funcname + "(unsigned n_, unsigned np_, double t, double *Y_, " \
              + "double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_)"
        # specstr = sig + "{\n\n" + pardefines + vardefines + "\n"
        specstr = sig + "{" + pardefines + vardefines + inpundefines + "\n"
        if docodeinserts and self.codeinserts['start'] != '':
            specstr += '/* Verbose code insert -- begin */\n' \
                            + self.codeinserts['start'] \
                            + '/* Verbose code insert -- end */\n\n'
        specstr += (len(reusestr)>0)*"/* reused term definitions */\n" \
                   + reusestr + "\n"
        auxdefs_parsed = {}
        # add function body
        for i in xrange(len(specnames)):
            xname = specnames[i]
            fbody = self.varspecs[xname]
            fbody_parsed = self._processSpecialC(fbody)
            if self.auxfns:
                fbody_parsed = addArgToCalls(fbody_parsed,
                                            self.auxfns.keys(),
                                            "p_, wk_, xv_")
                if 'initcond' in self.auxfns:
                    # convert 'initcond(x)' to 'initcond("x")' for
                    # compatibility with C syntax
                    fbody_parsed = wrapArgInCall(fbody_parsed,
                                    'initcond', '"')
            specstr += "f_[" + str(i) + "] = " + fbody_parsed + ";\n"
            auxdefs_parsed[xname] = fbody_parsed
        if docodeinserts and self.codeinserts['end'] != '':
            specstr += '\n/* Verbose code insert -- begin */\n' \
                    + self.codeinserts['end'] \
                    + '/* Verbose code insert -- end */\n'
        specstr += "\n" + parundefines + varundefines + inpundefines + "}\n\n"
        self._auxdefs_parsed = auxdefs_parsed
        return (specstr, funcname)


    def _doPreMacrosC(self):
        # Pre-processor macros are presently not available for C-code
        # specifications
        pass


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
            for ifstmt in range(num_ifs):
                if_ix = qtoks[if_ix+1:].index('if')+if_ix+1
                new_specStr += "".join(qtoks[ix_continue:if_ix]) + "__rhs_if("
                rbrace_ix = findEndBrace(qtoks[if_ix+1:])+if_ix+1
                ix_continue = rbrace_ix+1
                new_specStr += "".join(qtoks[if_ix+2:ix_continue])
            new_specStr += "".join(qtoks[ix_continue:])
            qspec = QuantSpec('spec', new_specStr)
            qtoks = qspec.parser.tokenized
        if '__minof' in qtoks:
            new_specStr = ""
            num = qtoks.count('__minof')
            n_ix = -1
            ix_continue = 0
            for stmt in range(num):
                n_ix = qtoks[n_ix+1:].index('__minof')+n_ix+1
                new_specStr += "".join(qtoks[ix_continue:n_ix])
                rbrace_ix = findEndBrace(qtoks[n_ix+1:])+n_ix+1
                ix_continue = rbrace_ix+1
                #assert qtoks[n_ix+2] == '[', "Error in min() syntax"
                #assert qtoks[rbrace_ix-1] == ']', "Error in min() syntax"
                #new_specStr += "".join(qtoks[n_ix+3:rbrace_ix-1]) + ")"
                num_args = qtoks[n_ix+2:ix_continue].count(',') + 1
                if num_args > 4:
                    raise NotImplementedError("Max of more than 4 arguments not currently supported in C")
                new_specStr += '__minof%s(' % str(num_args)
                new_specStr += "".join([q for q in qtoks[n_ix+2:ix_continue] if q not in ('[',']')])
            new_specStr += "".join(qtoks[ix_continue:])
            qspec = QuantSpec('spec', new_specStr)
            qtoks = qspec.parser.tokenized
        if '__maxof' in qtoks:
            new_specStr = ""
            num = qtoks.count('__maxof')
            n_ix = -1
            ix_continue = 0
            for stmt in range(num):
                n_ix = qtoks[n_ix+1:].index('__maxof')+n_ix+1
                new_specStr += "".join(qtoks[ix_continue:n_ix])
                rbrace_ix = findEndBrace(qtoks[n_ix+1:])+n_ix+1
                ix_continue = rbrace_ix+1
                #assert qtoks[n_ix+2] == '[', "Error in max() syntax"
                #assert qtoks[rbrace_ix-1] == ']', "Error in max() syntax"
                #new_specStr += "".join(qtoks[n_ix+3:rbrace_ix-1]) + ")"
                num_args = qtoks[n_ix+2:ix_continue].count(',') + 1
                if num_args > 4:
                    raise NotImplementedError("Min of more than 4 arguments not currently supported in C")
                new_specStr += '__maxof%s(' % str(num_args)
                new_specStr += "".join([q for q in qtoks[n_ix+2:ix_continue] if q not in ('[',']')])
            new_specStr += "".join(qtoks[ix_continue:])
            qspec = QuantSpec('spec', new_specStr)
            qtoks = qspec.parser.tokenized
        return new_specStr

    # ------------ Matlab code specifications -----------------------

    def _genAuxFnMatlab(self):
        auxnames = self.auxfns.keys()
        # parameter and variable definitions

        # sorted version of var and par names sorted version of par
        # names (vars not #define'd in aux functions unless Jacobian)
        vnames = self.vars
        pnames = self.pars
        vnames.sort()
        pnames.sort()

        for auxname in auxnames:
            assert auxname not in ['auxvars', 'vfield'], \
               ("auxiliary function name '" +auxname+ "' clashes with internal"
                " names")
        # must add parameter argument so that we can name
        # pars inside the functions! this would either
        # require all calls to include this argument (yuk!) or
        # else we add these extra pars automatically to
        # every call found in the .c code (as is done currently.
        # this is still an untidy solution, but there you go...)
        for auxname, auxspec in self._auxfnspecs.iteritems():
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
            # assert auxspec[1].find('^') == -1, ('carat character ^ is not '
            #                                 'permitted in function definitions'
            #                                 '-- use pow(x,p) syntax instead')
            # Process Jacobian functions specially, if present
            if auxname == 'Jacobian':
                raise NotImplementedError
            elif auxname == 'Jacobian_pars':
                raise NotImplementedError
            elif auxname == 'massMatrix':
                raise NotImplementedError
            else:
                ismat = False
                topstr = "function y_ = " + auxname + "("
                commentstr = "% Auxilliary function " + auxname + " for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"
                parlist = ""
                namemap = {}
                for parname in auxspec[0]:
                    parlist += parname + "__, "
                    namemap[parname] = parname+'__'
                topstr += parlist + " p_)\n"
                sig = topstr + commentstr
                pardefines = self._prepareMatlabPDefines(pnames)
                auxstr = auxspec[1]
                if any([pt in auxstr for pt in ('pow', '**')]):
                    auxstr = convertPowers(auxstr, '^')
                reusestr, auxspec_processedDict = self._processReusedMatlab([auxname],
                        {auxname:auxstr.replace(' ','').replace('\n','')})
                # addition of parameter done in Generator code

            dummyQ = QuantSpec('dummy', auxspec_processedDict[auxname],
                               treatMultiRefs=False, preserveSpace=True)
            if not ismat:
                dummyQ.mapNames(namemap)
            body_processed = "y_ = "*(not ismat) + dummyQ() + ";\n\n"
            # auxspecstr = sig + " {\n\n" + pardefines + vardefines*ismat \
            auxspecstr = sig + pardefines + " \n\n" \
                + "\n" + (len(reusestr)>0)*"% reused term definitions \n" \
                + reusestr + (len(reusestr)>0)*"\n" + body_processed
            # sig as second entry, whereas Python-coded specifications
            # have the fn name there
            self.auxfns[auxname] = (auxspecstr, sig)
        self._protected_auxnames.extend(auxnames)
        # Don't apply #define's for built-in functions


    def _genSpecMatlab(self):
        assert self.targetlang == 'matlab', ('Wrong target language for this'
                                             ' call')
        assert self.varspecs != {}, 'varspecs attribute must be defined'
        specnames_unsorted = self.varspecs.keys()
        specname_vars = intersect(self.vars, specnames_unsorted)
        specname_vars.sort()
        # parameter and variable definitions
        # sorted version of var and par names
        vnames = specname_vars
        pnames = self.pars
        pnames.sort()
        pardefines = self._prepareMatlabPDefines(pnames)
        vardefines = self._prepareMatlabVDefines(vnames)
        # produce vector field specification
        assert self.vars == specname_vars, ('Mismatch between declared '
                                        ' variable names and varspecs keys')
        valid_depTargNames = self.inputs+self.vars+self.auxvars
        for specname, specstr in self.varspecs.iteritems():
            assert type(specstr)==str, "Specification for %s was not a string"%specname
            if any([pt in specstr for pt in ('pow', '**')]):
                specstr = convertPowers(specstr, '^')
            specQS = QuantSpec('__spectemp__',  specstr)
            for s in specQS:
                if s in valid_depTargNames and (specname, s) not in \
                       self.dependencies: # and specname != s:
                    self.dependencies.append((specname, s))
        # pre-process reused sub-expression dictionary to adapt for
        # known calling sequence in Matlab
        reusestr, specupdated = self._processReusedMatlab(specname_vars,
                                                     self.varspecs)
        self.varspecs.update(specupdated)
        specstr_Matlab = self._genSpecFnMatlab('vfield', reusestr, specname_vars,
                                          pardefines, vardefines, True)
        self.spec = specstr_Matlab
        # do not produce auxiliary variables specification


    def _genSpecFnMatlab(self, funcname, reusestr, specnames, pardefines,
                         vardefines, docodeinserts):
        topstr = "function [vf_, y_] = " + funcname + "(vf_, t_, x_, p_)\n"
        commentstr = "% Vector field definition for model " + self.name + "\n% Generated by PyDSTool for ADMC++ target\n\n"

        specstr = topstr + commentstr + pardefines + vardefines + "\n"
        if docodeinserts and self.codeinserts['start'] != '':
            specstr += '% Verbose code insert -- begin \n' \
                            + self.codeinserts['start'] \
                            + '% Verbose code insert -- end \n\n'
        specstr += (len(reusestr)>0)*"% reused term definitions \n" \
                   + reusestr + "\n"
        # add function body
        for i in xrange(len(specnames)):
            xname = specnames[i]
            fbody = self.varspecs[xname]
            fbody_parsed = self._processIfMatlab(fbody)
            if self.auxfns:
                fbody_parsed = addArgToCalls(fbody_parsed,
                                            self.auxfns.keys(),
                                            "p_")
               # if 'initcond' in self.auxfns:
                    # convert 'initcond(x)' to 'initcond("x")' for
                    # compatibility with C syntax
                #    fbody_parsed = wrapArgInCall(fbody_parsed,
                 #                   'initcond', '"')
            specstr += "y_(" + str(i+1) + ") = " + fbody_parsed + ";\n"
        if docodeinserts and self.codeinserts['end'] != '':
            specstr += '\n% Verbose code insert -- begin \n' \
                    + self.codeinserts['end'] \
                    + '% Verbose code insert -- end \n'
        specstr += "\n\n"
        return (specstr, funcname)


    def _processReusedMatlab(self, specnames, specdict):
        """Process reused subexpression terms for Matlab code."""

        if self.auxfns:
            def addParToCall(s):
                return addArgToCalls(s, self.auxfns.keys(), "p_")
            parseFunc = addParToCall
        else:
            parseFunc = idfn
        reused, specupdated, new_protected, order = _processReused(specnames,
                                                          specdict,
                                                          self.reuseterms,
                                                          '', '', ';',
                                                          parseFunc)
        self._protected_reusenames = new_protected
        reusedefs = {}.fromkeys(new_protected)
        for vname, deflist in reused.iteritems():
            for d in deflist:
                reusedefs[d[2]] = d
        return (concatStrDict(reusedefs, intersect(order, reusedefs.keys())),
                       specupdated)
    # NEED TO CHECK WHETHER THIS IS NECESSARY AND WORKS
    # IF STATEMENTS LOOK DIFFERENT IN MATLAB
    def _processIfMatlab(self, specStr):
        qspec = QuantSpec('spec', specStr)
        qtoks = qspec[:]
        if 'if' in qtoks:
            raise NotImplementedError
        else:
            new_specStr = specStr
        return new_specStr


    def _prepareMatlabPDefines(self, pnames):
        pardefines = ""
        for i in xrange(len(pnames)):
            p = pnames[i]
            pardefines += "\t" + p + " = p_(" + str(i+1) + ");\n"

        alldefines = "\n% Parameter definitions\n\n" + pardefines
        return alldefines


    def _prepareMatlabVDefines(self, vnames):
        vardefines = ""
        for i in xrange(len(vnames)):
            v = vnames[i]
            vardefines += "\t" + v + " = x_(" + str(i+1) + ");\n"
        alldefines = "\n% Variable definitions\n\n" + vardefines
        return alldefines


    # ------------ Other utilities -----------------------

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
                for auxname in self.auxfns:
                    # verbose option shows up builtin auxiliary func definitions
                    if auxname not in self._builtin_auxnames or verbose>0:
                        outputStr += '\n  '+self.auxfns[auxname][0]+'\n'
            outputStr += "\n\nDependencies in specification functions - pair (i, o)"\
                    " means i depends on o:\n  " + str(self.dependencies)
        return outputStr

    def info(self, verbose=0):
        print self._infostr(verbose)

    def __repr__(self):
        return self._infostr(verbose=0)

    __str__ = __repr__



# -----------------------------------

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



def _processReused(specnames, specdict, reuseterms, indentstr='',
                    typestr='', endstatementchar='', parseFunc=idfn):
    """Process substitutions of reused terms."""

    seenrepterms = []  # for new protected names (global to all spec names)
    reused = {}.fromkeys(specnames)
    reuseterms_inv = invertMap(reuseterms)
    # establish order for reusable terms, in case of inter-dependencies
    are_dependent = []
    deps = {}
    for origterm, rterm in reuseterms.iteritems():
        for ot, rt in reuseterms.iteritems():
            if proper_match(origterm, rt):
                if rterm not in are_dependent:
                    are_dependent.append(rterm)
                try:
                    deps[rterm].append(rt)
                except KeyError:
                    # new list
                    deps[rterm] = [rt]
    order = remain(reuseterms.values(), are_dependent) + are_dependent
    for specname in specnames:
        reused[specname] = []
        specstr = specdict[specname]
        repeatkeys = []
        for origterm, repterm in reuseterms.iteritems():
            # only add definitions if string found
            if proper_match(specstr, origterm):
                specstr = specstr.replace(origterm, repterm)
                if repterm not in seenrepterms:
                    reused[specname].append([indentstr,
                                        typestr+' '*(len(typestr)>0),
                                        repterm, " = ",
                                        parseFunc(origterm),
                                        endstatementchar, "\n"])
                    seenrepterms.append(repterm)
            else:
                # look for this term on second pass
                repeatkeys.append(origterm)
        if len(seenrepterms) > 0:
            # don't bother with a second pass if specstr has not changed
            for origterm in repeatkeys:
                # second pass
                repterm = reuseterms[origterm]
                if proper_match(specstr, origterm):
                    specstr = specstr.replace(origterm, repterm)
                    if repterm not in seenrepterms:
                        seenrepterms.append(repterm)
                        reused[specname].append([indentstr,
                                                 typestr+' '*(len(typestr)>0),
                                                 repterm, " = ",
                                                 parseFunc(origterm),
                                                 endstatementchar, "\n"])
        # if replacement terms have already been used in the specifications
        # and there are no occurrences of the terms meant to be replaced then
        # just log the definitions that will be needed without replacing
        # any strings.
        if reused[specname] == [] and len(reuseterms) > 0:
            for origterm, repterm in reuseterms.iteritems():
                # add definition if *replacement* string found in specs
                if proper_match(specstr, repterm) and repterm not in seenrepterms:
                    reused[specname].append([indentstr,
                                        typestr+' '*(len(typestr)>0),
                                        repterm, " = ",
                                        parseFunc(origterm),
                                        endstatementchar, "\n"])
                    seenrepterms.append(repterm)
        specdict[specname] = specstr
        # add any dependencies for repeated terms to those that will get
        # defined when functions are instantiated
        add_reps = []
        for r in seenrepterms:
            if r in are_dependent:
                for repterm in deps[r]:
                    if repterm not in seenrepterms:
                        reused[specname].append([indentstr,
                                        typestr+' '*(len(typestr)>0),
                                        repterm, " = ",
                                        parseFunc(reuseterms_inv[repterm]),
                                        endstatementchar, "\n"])
                        seenrepterms.append(repterm)
    # reuseterms may be automatically provided for a range of definitions
    # that may or may not contain instances, and it's too inefficient to
    # check in advance, so we'll not cause an error here if none show up.
    # if len(seenrepterms) == 0 and len(reuseterms) > 0:
    #     print "Reuse terms expected:", reuseterms
    #     info(specdict)
    #     raise RuntimeError("Declared reusable term definitions did not match"
    #                        " any occurrences in the specifications")
    return (reused, specdict, seenrepterms, order)




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
    for fname, (fargs, fspec) in fnspecs.iteritems():
        common_names = intersect(fargs, parnames)
        if fname in parnames:
            print "Problem with function definition", fname
            raise ValueError("Unrecoverable clash between parameter names and aux fn name")
        if common_names == []:
            new_fnspecs[fname] = (fargs, fspec)
        else:
            changed_fns.append(fname)
            new_fnspecs[fname] = (remain(fargs, parnames), fspec)

    new_varspecs = {}
    for vname, vspec in varspecs.iteritems():
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
                        print "Warning: some auxiliary function parameters clash in function %s" %f
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
    except IOError, e:
        print 'File error:', str(e)
        raise
    f.close()
    return s



