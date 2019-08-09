"""
    Parser utilities.

    Robert Clewley, September 2005.

    Includes AST code by Pearu Peterson and Ryan Gutenkunst,
      modified by R. Clewley.
"""

# IMPORTS
from __future__ import division, absolute_import, print_function
from .errors import *
from .common import *
import re
import math, random
from numpy import alltrue, sometrue
import numpy as np
from copy import copy, deepcopy
import parser, symbol, token

# --------------------------------------------------------------------------
# GLOBAL BEHAVIOUR CONTROLS -- leave both as True for use with PyDSTool.
# switch to change output from using 'x**y' (DO_POW=False)
# to 'pow(x,y)' (DO_POW=TRUE)
DO_POW=True
# switch to activate treatment of integer constants as decimals in expressions
# and simplifications of expressions.
DO_DEC=True

# ------------------------------------------------------------------------
### Protected names

protected_numpynames = ['arcsin', 'arccos', 'arctan', 'arctan2',
                           'arccosh', 'arcsinh', 'arctanh']
# scipy special functions
scipy_specialfns = ['airy', 'airye', 'ai_zeros', 'bi_zeros', 'ellipj',
            'ellipk', 'ellipkinc', 'ellipe', 'ellipeinc', 'jn',
            'jv', 'jve', 'yn', 'yv', 'yve', 'kn', 'kv', 'kve',
            'iv', 'ive', 'hankel1', 'hankel1e', 'hankel2',
            'hankel2e', 'lmbda', 'jnjnp_zeros', 'jnyn_zeros',
            'jn_zeros', 'jnp_zeros', 'yn_zeros', 'ynp_zeros',
            'y0_zeros', 'y1_zeros', 'y1p_zeros', 'j0', 'j1',
            'y0', 'y1', 'i0', 'i0e', 'i1', 'i1e', 'k0', 'k0e',
            'k1', 'k1e', 'itj0y0', 'it2j0y0', 'iti0k0', 'it2i0k0',
            'besselpoly', 'jvp', 'yvp', 'kvp', 'ivp', 'h1vp',
            'h2vp', 'spherical_jn', 'spherical_yn', 'spherical_in',
            'spherical_kn', 'riccati_jn', 'riccati_yn',
            'struve', 'modstruve', 'itstruve0', 'it2struve0',
            'itmodstruve0', 'bdtr', 'bdtrc', 'bdtri', 'btdtr',
            'btdtri', 'fdtr', 'fdtrc', 'fdtri', 'gdtr', 'gdtrc',
            'gdtria', 'nbdtr', 'nbdtrc', 'nbdtri', 'pdtr', 'pdtrc',
            'pdtri', 'stdtr', 'stdtridf', 'stdtrit', 'chdtr', 'chdtrc',
            'chdtri', 'ndtr', 'ndtri', 'smirnov', 'smirnovi',
            'kolmogorov', 'kolmogi', 'tklmbda', 'gamma',
            'gammaln', 'gammainc', 'gammaincinv', 'gammaincc',
            'gammainccinv', 'beta', 'betaln', 'betainc',
            'betaincinv', 'psi',
            'digamma', 'rgamma', 'polygamma', 'erf', 'erfc',
            'erfinv', 'erfcinv', 'erf_zeros', 'fresnel',
            'fresnel_zeros', 'fresnelc_zeros', 'fresnels_zeros',
            'modfresnelp', 'modfresnelm', 'lpn', 'lqn', 'lpmn',
            'lqmn', 'lpmv', 'sph_harm', 'legendre', 'chebyt',
            'chebyu', 'chebyc', 'chebys', 'jacobi', 'laguerre',
            'genlaguerre', 'hermite', 'hermitenorm', 'gegenbauer',
            'sh_legendre', 'sh_chebyt', 'sh_chebyu', 'sh_jacobi',
            'hyp2f1', 'hyp1f1', 'hyperu', 'hyp0f1', 'hyp2f0',
            'hyp1f2', 'hyp3f0', 'pbdv', 'pbvv', 'pbwa', 'pbdv_seq',
            'pbvv_seq', 'pbdn_seq', 'mathieu_a', 'mathieu_b',
            'mathieu_even_coef', 'mathieu_odd_coef', 'mathieu_cem',
            'mathieu_sem', 'mathieu_modcem1', 'mathieu_modcem2',
            'mathieu_modsem1', 'mathieu_modsem2', 'pro_ang1',
            'pro_rad1', 'pro_rad2', 'obl_ang1', 'obl_rad1',
            'obl_rad2', 'pro_cv', 'obl_cv', 'pro_cv_seq',
            'obl_cv_seq', 'pro_ang1_cv', 'pro_rad1_cv',
            'pro_rad2_cv', 'obl_ang1_cv', 'obl_rad1_cv',
            'obl_rad2_cv', 'kelvin', 'kelvin_zeros', 'ber',
            'bei', 'berp', 'beip', 'ker', 'kei', 'kerp', 'keip',
            'ber_zeros', 'bei_zeros', 'berp_zeros', 'beip_zeros',
            'ker_zeros', 'kei_zeros', 'kerp_zeros', 'keip_zeros',
            'expn', 'exp1', 'expi', 'wofz', 'dawsn', 'shichi',
            'sici', 'spence', 'zeta', 'zetac', 'cbrt', 'exp10',
            'exp2', 'radian', 'cosdg', 'sindg', 'tandg', 'cotdg',
            'log1p', 'expm1', 'cosm1', 'round']
protected_scipynames = ['sign', 'mod']
protected_specialfns = ['special_'+s for s in scipy_specialfns]
protected_mathnames = [s for s in dir(math) if not s.startswith('__')]
protected_randomnames = [s for s in dir(random) if not s.startswith('_')] # yes, just single _
protected_builtins = ['abs', 'pow', 'min', 'max', 'sum', 'and', 'not', 'or']
# We add internal default auxiliary function names for use by
# functional specifications.
builtin_auxnames = ['globalindepvar', 'initcond', 'heav', 'if',
                    'getindex', 'getbound']

protected_macronames = ['for', 'if', 'max', 'min', 'sum']

reserved_keywords = ['and', 'not', 'or', 'del', 'for', 'if', 'is', 'raise',
                'assert', 'elif', 'from', 'lambda', 'return', 'break', 'else',
                'global', 'try', 'class', 'except', 'while',
                'continue', 'exec', 'import', 'pass', 'yield', 'def',
                'finally', 'in', 'print', 'as', 'None']

convert_power_reserved_keywords = ['del', 'for', 'if', 'is', 'raise',
                'assert', 'elif', 'from', 'lambda', 'return', 'break', 'else',
                'global', 'try', 'class', 'except', 'while',
                'continue', 'exec', 'import', 'pass', 'yield', 'def',
                'finally', 'in', 'print', 'as', 'None']

# 'abs' is defined in python core, so doesn't appear in math
protected_allnames = protected_mathnames + protected_scipynames \
                    + protected_numpynames \
                    + protected_specialfns \
                    + protected_randomnames \
                    + protected_numpynames \
                    + builtin_auxnames + protected_macronames \
                    + protected_builtins

# signature lengths for builtin auxiliary functions and macros, for
# use by ModelSpec in eval() method (to create correct-signatured temporary
# functions).
builtinFnSigInfo = {'globalindepvar': 1, 'initcond': 1, 'heav': 1, 'getindex': 1,
                    'if': 3, 'for': 4, 'getbound': 2, 'max': 1, 'min': 1}

# ------------------------------------------------------------------------

# EXPORTS
_functions = ['readArgs', 'findEndBrace', 'makeParList', 'joinStrs',
              'parseMatrixStrToDictStr', 'joinAsStrs', 'replaceSep',
              'wrapArgInCall', 'addArgToCalls', 'findNumTailPos',
              'isToken', 'isNameToken', 'isNumericToken', 'count_sep',
              'isHierarchicalName', 'replaceSepInv', 'replaceSepListInv',
              'replaceSepList', 'convertPowers', 'mapNames',
              'replaceCallsWithDummies', 'isIntegerToken', 'proper_match',
              'remove_indices_from_range']

_objects = ['protected_auxnamesDB', 'protected_allnames', 'protected_macronames',
            'protected_mathnames', 'protected_randomnames', 'builtin_auxnames',
            'protected_scipynames', 'protected_numpynames', 'protected_builtins',
            'protected_specialfns', 'builtinFnSigInfo', 'scipy_specialfns']

_classes = ['symbolMapClass', 'parserObject', 'auxfnDBclass']

_constants = ['name_chars_RE', 'num_chars', 'ZEROS', 'ONES', 'NAMESEP',
              '_indentstr', 'alphabet_chars_RE', 'alphanumeric_chars_RE']

_symbfuncs = ['simplify', 'simplify_str', 'ensurebare', 'ensureparen',
             'trysimple', 'ensuredecimalconst', 'doneg', 'dosub', 'doadd',
             'dodiv', 'domul', 'dopower', 'splitastLR', 'ast2string', 'string2ast',
             'sym2name', 'ast2shortlist', 'splitargs', 'mapPowStr',
             'toPowSyntax', 'ensureparen_div']

_symbconsts = ['syms']

__all__ = _functions + _classes + _objects + _constants + _symbfuncs + _symbconsts

#-----------------------------------------------------------------------------

## constants for parsing
name_chars_RE = re.compile(r'\w')
alphanumeric_chars_RE = re.compile('[a-zA-Z0-9]')   # without the '_'
alphabet_chars_RE = re.compile('[a-zA-Z]')
num_chars = [str(i) for i in range(10)]

if DO_POW:
    POW_STR = 'pow(%s,%s)'
else:
    POW_STR = '%s**%s'

if DO_DEC:
    ZEROS = ['0','0.0','0.','(0)','(0.0)','(0.)']
    ONES = ['1','1.0','1.','(1)','(1.0)','(1.)']
    TENS = ['10', '10.0', '10.','(10)','(10.0)','(10.)']
else:
    ZEROS = ['0']
    ONES = ['1']
    TENS = ['10']

# separator for compound, hierarchical names
NAMESEP = '.'

# for python indentation in automatically generated code, use 4 spaces
_indentstr = "    "


# ----------------------------------------------------------------------------
# This section: code by Pearu Peterson, adapted by Ryan Gutenkunst
#   and Robert Clewley.

syms=token.tok_name
for s in symbol.sym_name.keys():
    syms[s]=symbol.sym_name[s]


def mapPowStr(t, p='**'):
    """Input an expression of the form 'pow(x,y)'. Outputs an expression of the
    form x**y, x^y, or pow(x,y) and where x and y have also been processed to the
    target power syntax.
    Written by R. Clewley"""
    ll = splitargs(ast2string(t[2])[1:-1])
    if p=='**':
        lpart = dopower(ensureparen(ast2string(toDoubleStarSyntax(string2ast(ll[0]))),1),
                       ensureparen(ast2string(toDoubleStarSyntax(string2ast(ll[1]))),1),
                       '%s**%s')
        if len(t) > 3:
            res = ensureparen(ast2string(toDoubleStarSyntax(['power',string2ast('__LPART__')]+t[3:])))
            return res.replace('__LPART__', lpart)
        else:
            return ensureparen(lpart,1)
    elif p=='^':
        lpart = dopower(ensureparen(ast2string(toCircumflexSyntax(string2ast(ll[0]))),1),
                       ensureparen(ast2string(toCircumflexSyntax(string2ast(ll[1]))),1),
                       '%s^%s')
        if len(t) > 3:
            res = ensureparen(ast2string(toCircumflexSyntax(['power',string2ast('__LPART__')]+t[3:])),1)
            return res.replace('__LPART__', lpart)
        else:
            return ensureparen(lpart,1)
    elif p=='pow':
        lpart = dopower(ensurebare(ast2string(toPowSyntax(string2ast(ll[0])))),
                       ensurebare(ast2string(toPowSyntax(string2ast(ll[1])))),
                       'pow(%s,%s)')
        if len(t) > 3:
            res = ensurebare(ast2string(toPowSyntax(['power',string2ast('__LPART__')]+t[3:])))
            return res.replace('__LPART__', lpart)
        else:
            return ensureparen(lpart,1)
    else:
        raise ValueError("Invalid power operator")


def toCircumflexSyntax(t):
    # R. Clewley
    if isinstance(t[0], str):
        if t[0] in ['power', 'atom_expr']:
            if t[2][0] == 'DOUBLESTAR':
                return string2ast(ensureparen(dopower(ast2string(toCircumflexSyntax(t[1])),
                                    ast2string(toCircumflexSyntax(t[3])),
                         '%s^%s'),1))
            if t[1] == ['NAME', 'pow']:
                return string2ast(ensureparen(mapPowStr(t,'^'),1))
    o = []
    for i in t:
        if isinstance(i,list):
            if type(i[0]) == str and i[0].islower():
                o.append(toCircumflexSyntax(i))
            else:
                o.append(i)
        else:
            o.append(i)
    return o


def toDoubleStarSyntax(t):
    # R. Clewley
    if isinstance(t[0], str):
        if t[0] == 'xor_expr' and t[2][0]=='CIRCUMFLEX':
            # ^ syntax has complex binding rules in python parser's AST!
            # trick - easy to convert to ** first. then, using a bit of a hack
            # convert to string and back to AST so that proper AST for **
            # is formed.
            tc = copy(t)
            tc[0] = 'power'
            tc[2] = ['DOUBLESTAR', '**']
            return toDoubleStarSyntax(string2ast(ast2string(tc)))   # yes, i mean this
        if t[0] in ['power', 'atom_expr'] and t[1] == ['NAME', 'pow']:
            return string2ast(ensureparen(mapPowStr(t,'**'),1))
    o = []
    for i in t:
        if isinstance(i,list):
            if type(i[0]) == str and i[0].islower():
                o.append(toDoubleStarSyntax(i))
            else:
                o.append(i)
        else:
            o.append(i)
    return o


def toPowSyntax(t):
    # R. Clewley
    if isinstance(t[0],str):
        if t[0] in ['power', 'atom_expr']:
            try:
                if t[2][0]=='DOUBLESTAR':
                    try:
                        return string2ast(dopower(ensurebare(ast2string(toPowSyntax(t[1]))),
                                            ensurebare(ast2string(toPowSyntax(t[3]))),
                                 'pow(%s,%s)'))
                    except IndexError:
                        # there's a pow statement already here, not a **
                        # so ignore
                        return t
                elif t[1][1] == 'pow':
                    return string2ast(ensureparen(mapPowStr(t,'pow'),1))
                elif len(t)>3 and t[3][0]=='DOUBLESTAR':
                    try:
                        return string2ast(dopower(ensurebare(ast2string(toPowSyntax(t[1:3]))),
                                            ensurebare(ast2string(toPowSyntax(t[4]))),
                                 'pow(%s,%s)'))
                    except IndexError:
                        # there's a pow statement already here, not a **
                        # so ignore
                        return t
            except:
                print(t)
                print(ast2string(t))
                raise
        elif t[0] == 'xor_expr' and t[2][0]=='CIRCUMFLEX':
            # ^ syntax has complex binding rules in python parser's AST!
            # trick - easy to convert to ** first. then, using a bit of a hack
            # convert to string and back to AST so that proper AST for **
            # is formed.
            tc = copy(t)
            tc[0] = 'power'
            tc[2] = ['DOUBLESTAR', '**']
            return toPowSyntax(string2ast(ast2string(tc)))   # yes, i mean this
    o = []
    for i in t:
        if isinstance(i,list):
            if type(i[0]) == str and i[0].islower():
                o.append(toPowSyntax(i))
            else:
                o.append(i)
        else:
            o.append(i)
    return o


def temp_macro_names(s):
    t = s
    for m in convert_power_reserved_keywords:
        t = t.replace(m, '__'+m+'__')
    return t

def temp_macro_names_inv(s):
    t = s
    for m in convert_power_reserved_keywords:
        t = t.replace('__'+m+'__', m)
    return t


def convertPowers(s, target="pow"):
    """convertPowers takes a string argument and maps all occurrences
    of power sub-expressions into the chosen target syntax. That option
    is one of "**", "^", or "pow"."""
    # temp_macro_names switches python reserved keywords to adapted names
    # that won't make the python parser used in string2ast raise a syntax error
    # ... but only invoke this relatively expensive function in the unlikely
    # event a clash occurs.
    # R. Clewley
    if target=="**":
        try:
            return ast2string(toDoubleStarSyntax(string2ast(s)))
        except SyntaxError:
            s = temp_macro_names(s)
            return temp_macro_names_inv(ast2string(toDoubleStarSyntax(string2ast(s))))
    elif target=="^":
        try:
            return ast2string(ensureints(toCircumflexSyntax(string2ast(s))))
        except SyntaxError:
            s = temp_macro_names(s)
            return temp_macro_names_inv(ast2string(ensureints(toCircumflexSyntax(string2ast(s)))))
    elif target=="pow":
        try:
            return ast2string(toPowSyntax(string2ast(s)))
        except SyntaxError:
            s = temp_macro_names(s)
            return temp_macro_names_inv(ast2string(toPowSyntax(string2ast(s))))
    else:
        raise ValueError("Invalid target syntax")


def ensureints(t):
    """Ensure that any floating point constants appearing in t that are
    round numbers get converted to integer representations."""
    if type(t)==str:
        return ast2string(ensureints(string2ast(t)))
    o = []
    for i in t:
        if type(i) == list:
            if type(i[0]) == str and i[0].islower():
                o.append(ensureints(i))
            elif i[0]=='NUMBER':
                # CAPS for constants
                o.append(string2ast(trysimple(ast2string(i))))
            else:
                o.append(i)
        else:
            o.append(i)
    return o


def splitargs(da, lbraces=['('], rbraces=[')']):
    """Function to split string-delimited arguments in a string without
    being fooled by those that occur in function calls.
    Written by Pearu Peterson. Adapted by Rob Clewley to accept different
    braces."""
    if alltrue([da.find(lbrace)<0 for lbrace in lbraces]):
        return da.split(',')
    ll=[];o='';ii=0
    for i in da:
        if i==',' and ii==0:
            ll.append(o)
            o=''
        else:
            if i in lbraces: ii=ii+1
            if i in rbraces: ii=ii-1
            o=o+i
    ll.append(o)
    return ll

def ast2shortlist(t):
    if type(t) is parser.STType: return ast2shortlist(t.tolist())
    if not isinstance(t, list): return t
    if t[1] == '': return None
    if not isinstance(t[1], list): return t
    if len(t) == 2 and isinstance(t[1], list):
        return ast2shortlist(t[1])
    o=[]
    for tt in map(ast2shortlist, t[1:]):
        if tt is not None:
            o.append(tt)
    if len(o)==1: return o[0]
    return [t[0]]+o

def sym2name(t):
    if type(t) is parser.STType: return sym2name(t.tolist())
    if not isinstance(t, list): return t
    return [syms[t[0]]]+list(map(sym2name,t[1:]))

def string2ast(t):
    return sym2name(ast2shortlist(parser.expr(t)))

def ast2string(t):
    #if isinstance(t, str): return t
    if type(t) is parser.STType: return ast2string(t.tolist())
    if not isinstance(t, list): return None
    if not isinstance(t[1], list): return t[1]
    o=''
    for tt in map(ast2string,t):
        if isinstance(tt, str):
            o=o+tt
    return o

def splitastLR(t):
    lft=t[1]
    rt=t[3:]
    if len(rt)>1:
        rt=[t[0]]+rt
    else:
        rt=rt[0]
    return lft,rt

def dopower(l,r,pow_str=POW_STR):
    if r in ZEROS: return '1'
    if l in ZEROS: return '0'
    if l in ONES: return '1'
    if r in ONES: return l
    if pow_str=='%s**%s':
        return trysimple('%s**%s'%(ensureparen(l),ensureparen(r)))
    elif pow_str == '%s^%s':
        return trysimple('%s^%s'%(ensuredecimalconst(ensureparen(l)),ensuredecimalconst(ensureparen(r))))
    elif pow_str == 'pow(%s,%s)':
        return trysimple('pow(%s,%s)'%(l,r)) #trysimple('pow(%s,%s)'%(ensurebare(l),ensurebare(r)))
    else:
        raise ValueError("Invalid target power syntax")

def domul(l,r):
    if l in ZEROS or r in ZEROS: return '0'
    if l in ONES: return r
    if r in ONES: return l
    if l in ['-'+o for o in ONES]: return doneg(r)
    if r in ['-'+o for o in ONES]: return doneg(l)
    lft = string2ast(l)
    rt = string2ast(r)
    lft_neg = lft[0] == 'factor' and lft[1][0]=='MINUS'
    rt_neg = rt[0] == 'factor' and rt[1][0]=='MINUS'
    if lft_neg:
        new_l = l[1:]
    else:
        new_l = l
    if rt_neg:
        new_r = r[1:]
    else:
        new_r = r
    if lft_neg and rt_neg or not (lft_neg or rt_neg):
        return trysimple('%s*%s'%(ensureparen(new_l,ismul=1),
                                  ensureparen(new_r,ismul=1)))
    else:
        return trysimple('-%s*%s'%(ensureparen(new_l,ismul=1),
                                   ensureparen(new_r,ismul=1)))

def dodiv(l,r):
    if r in ZEROS: raise ValueError("Division by zero in expression")
    if l in ZEROS: return '0'
    if r in ONES: return l
##    if l in ['-'+o for o in ONES]: return doneg(dodiv('1',r))
    if r in ['-'+o for o in ONES]: return doneg(l)
    if r==l: return '1'
    lft = string2ast(l)
    rt = string2ast(r)
    lft_neg = lft[0] == 'factor' and lft[1][0]=='MINUS'
    rt_neg = rt[0] == 'factor' and rt[1][0]=='MINUS'
    if lft_neg:
        new_l = l[1:]
    else:
        new_l = l
    if rt_neg:
        new_r = r[1:]
    else:
        new_r = r
    if lft_neg and rt_neg or not (lft_neg or rt_neg):
        return trysimple('%s/%s'%(ensureparen(ensuredecimalconst(new_l),ismul=1,do_decimal=DO_DEC),
                               ensureparen(ensuredecimalconst(new_r),1,do_decimal=DO_DEC)),
                         do_decimal=DO_DEC)
    else:
        return trysimple('-%s/%s'%(ensureparen(ensuredecimalconst(new_l),ismul=1,do_decimal=DO_DEC),
                               ensureparen(ensuredecimalconst(new_r),1,do_decimal=DO_DEC)),
                         do_decimal=DO_DEC)

def doadd(l,r):
    if l in ZEROS and r in ZEROS: return '0'
    if l in ZEROS: return r
    if r in ZEROS: return l
    if l==r: return trysimple(domul('2',l))
    if r[0]=='-': return trysimple('%s%s'%(l,r))
    return trysimple('%s+%s'%(l,r))

def dosub(l,r):
    if l in ZEROS and r in ZEROS: return '0'
    if l in ZEROS: return doneg(r)
    if r in ZEROS: return l
    if l==r: return '0'
    if r[0]=='-': return ensureparen(trysimple('%s+%s'%(l,doneg(r))))
    return trysimple('%s-%s'%(l,r))

def doneg(l):
    if l in ZEROS: return '0'
##    if l[0]=='-': return l[1:]
    t=string2ast(l)
##    print "doneg called with %s"%l, "\n", t, "\n"
    if t[0]=='atom' and t[1][0]=='LPAR' and t[-1][0]=='RPAR':
        return ensureparen(doneg(ast2string(t[2])))
    if t[0]=='arith_expr':
        # Propagate -ve sign into the sum
        o=doneg(ast2string(t[1]))
        aexpr = t[2:]
        for i in range(0,len(aexpr),2):
            if aexpr[i][0]=='PLUS':
                o = dosub(o, ast2string(aexpr[i+1]))
            else:
                o = doadd(o, ast2string(aexpr[i+1]))
        return o
    if t[0]=='term':
        # Propagate -ve sign onto the first term
        tc = copy(t)
        if t[1][0]=='factor' and t[1][1][0]=='MINUS':
            tc[1] = t[1][2]
        else:
            tc[1] = string2ast(doneg(ast2string(t[1])))
        return ast2string(tc)
    if t[0]=='factor' and t[1][0] == 'MINUS':
        return ast2string(t[2])
    return trysimple('-%s'%l)

def ensuredecimalconst(t):
    return trysimple(t,do_decimal=DO_DEC)

def trysimple(t,do_decimal=False):
    try:
        t_e = eval(t, {}, {})
        add_point = do_decimal and t_e != 0 and DO_DEC
        if type(t_e) == int and add_point:
            t = repr(t_e)+".0"
        elif type(t_e) == float and int(t_e)==t_e:
            # explicitly use repr here in case t string was e.g. '2e7'
            # which evals to a float, but the string itself represents an int
            if add_point:
                t = repr(t_e)
            else:
                t = repr(int(t_e))
        else:
            t = repr(t_e)
    except:
        pass
    return t

def ensureparen_div(tt):
    if tt[0] == 'term' and tt[2][0] == 'SLASH' and len(tt[3:])>1:
        return ['term',
                string2ast(ensureparen(ast2string(tt[1])+'/'+ast2string(tt[3]),
                                       1))] + tt[4:]
    else:
        return tt

def ensureparen(t,flag=0,ismul=0,do_decimal=False):
    t=trysimple(t, do_decimal)
    tt=string2ast(t)
    if t[0]=='-':
        if tt[0] == 'factor': # or tt[0] == 'term' and ismul:
            # single number doesn't need braces
            return t
        else:
##        print "0: ", t, "->", '(%s)'%t
            return '(%s)'%t
    if tt[0]=='arith_expr':
##        print "1: ", t, "->", '(%s)'%t
        return '(%s)'%t
    if flag>0:
        if tt[0] == 'term':
            return '(%s)'%t
        elif tt[0] == 'power':
            if tt[1] == ['NAME', 'pow']:
                return t
            else:
                # ** case
                for x in tt[1:]:
                    if x[0] == 'arith_expr':
                        return '(%s)'%t
        elif tt[0] == 'xor_expr':    # added xor_expr for ^ powers
            if len(tt)>3:
                for x in tt[1:]:
                    if x[0] == 'arith_expr':
                        return '(%s)'%t
            else:
                return t
##        if t[0]=='(' and t[-1]==')' and t[1:-1].find('(') < t[1:-1].find(')'):
##            # e.g. (1+x) doesn't need another set of braces
##            print "2: ", t, "->", t
##            return t
##        else:
##            # e.g. (1+x)-(3-y) does need braces
##            print "3: ", t, "->", '(%s)'%t
##            return '(%s)'%t
    return t

def ensurebare(t):
    """Ensure no braces in string expression (where possible).
    Written by Robert Clewley"""
    t=trysimple(t)
    try:
        if t[0]=='(' and t[-1]==')':
            if t[1:-1].find('(') < t[1:-1].find(')'):
                return t[1:-1]
            else:
                # false positive, e.g. (1+x)-(3-y)
                return t
        else:
            return t
    except IndexError:
        return t

def collect_numbers(t):
    """Re-arrange 'term' or 'arith_expr' expressions to combine numbers.
    Numbers go first unless in a sum the numeric term is negative and not all
    the remaining terms are negative."""
    # number may be prefixed by a 'factor'
##    print "collect called with %s"%ast2string(t), "\n", t, "\n"
    if t[0] == 'arith_expr':
        args = [simplify(a) for a in t[1::2]]
        numargs = len(args)
        if args[0][0] == 'factor' and args[0][1][0] == 'MINUS':
            ops = [-1]
            # remove negative sign from term as we've recorded the sign for the sum
            args[0] = args[0][2]
        else:
            ops = [1]
        for op_t in t[2::2]:
            if op_t[0]=='PLUS':
                ops.append(1)
            else:
                ops.append(-1)
        num_ixs = []
        oth_ixs = []
        for i, a in enumerate(args):
            if a[0]=='NUMBER':
                num_ixs.append(i)
            else:
                oth_ixs.append(i)
        res_num = '0'
        # enter numbers first
        for nix in num_ixs:
            if ops[nix] > 0:
                res_num=doadd(res_num,ast2string(simplify(args[nix])))
            else:
                res_num=dosub(res_num,ast2string(simplify(args[nix])))
        # follow by other terms
        res_oth = '0'
        for oix in oth_ixs:
            if ops[oix] > 0:
                res_oth=doadd(res_oth,ast2string(simplify(args[oix])))
            else:
                res_oth=dosub(res_oth,ast2string(simplify(args[oix])))
        if res_num[0] == '-' and res_oth[0] != '-':
            # switch order
            return string2ast(doadd(res_oth,res_num))
        else:
            return string2ast(doadd(res_num,res_oth))
    elif t[0] == 'term':
        args = [simplify(a) for a in t[1::2]]
        numargs = len(args)
        ops = [1]
        for op_t in t[2::2]:
            if op_t[0]=='STAR':
                ops.append(1)
            else:
                ops.append(-1)
        num_ixs = []
        oth_ixs = []
        for i, a in enumerate(args):
            if a[0]=='NUMBER':
                num_ixs.append(i)
            else:
                oth_ixs.append(i)
        res_numerator = '1'
        res_denominator = '1'
        # enter numbers first
        for nix in num_ixs:
            if ops[nix] > 0:
                res_numerator=domul(res_numerator,ast2string(simplify(args[nix])))
            else:
                res_denominator=domul(res_denominator,ast2string(simplify(args[nix])))
        # follow by other terms
        for oix in oth_ixs:
            if ops[oix] > 0:
                res_numerator=domul(res_numerator,ast2string(simplify(args[oix])))
            else:
                res_denominator=domul(res_denominator,ast2string(simplify(args[oix])))
        return string2ast(dodiv(res_numerator,res_denominator))
    else:
        return t

def simplify_str(s):
    return s
##    """String output version of simplify"""
##    t=string2ast(s)
##    if isinstance(t, list) and len(t) == 1:
##        return ast2string(simplify(t[0]))
##    if t[0]=='NUMBER':
##        if 'e' in s:
##            epos=s.find('e')
##            man=s[:epos]
##            ex=s[epos:]
##        else:
##            ex=''
##            man=s
##        if man[-1] == '.':
##            return man+'0'+ex
##        else:
##            return s
##    if t[0]=='NAME':
##        return s
##    if t[0]=='factor':
##        return doneg(ensureparen(ast2string(simplify(t[2:][0]))))
##    if t[0]=='arith_expr':
##        return ast2string(collect_numbers(t))
##    if t[0]=='term':
##        return ast2string(collect_numbers(ensureparen_div(t)))
##    if t[0]=='power': # covers math functions like sin, cos, and log10
##        if t[2][0]=='trailer':
##            if len(t)>3:
##                term1 = simplify(t[:3])
##                if term1[0] in ['NUMBER', 'NAME']: #'power'
##                    formatstr='%s'
##                else:
##                    formatstr='(%s)'
##                return ast2string([t[0],string2ast(formatstr%ast2string(term1)),
##                                  simplify(t[3:])])
##            elif len(t)==3 and t[1][1]=='pow':
##                # 'pow' syntax case
##                terms = t[2][2][1::2]
##                return dopower(ast2string(simplify(terms[0])),
##                               ast2string(simplify(terms[1])))
##            # else ignore and carry on
##        ts=[];o=[];
##        for i in t[1:]:
##            if i[0]=='DOUBLESTAR':
##                if len(o)==1: o=o[0]
##                ts.append(o);
##                o=[]
##            else: o.append(simplify(i))
##        if len(o)==1: o=o[0]
##        ts.append(o)
##        if t[2][0]=='DOUBLESTAR':
##            st,lft,rt=map(ast2string,[t,ts[0],ts[1]])
##            return dopower(simplify_str(lft),simplify_str(rt))
##        if t[2][0]=='trailer':
##            return ast2string(simplify(ts[0]))
##    if t[0] in ['arglist','testlist']:
##        o=[]
##        for i in t[1::2]:
##            o.append(ast2string(simplify(i)))
##        return ','.join(o)
##    if t[0]=='atom':
####        if t[1][0]=='LPAR' and t[-1][0]=='RPAR':
####            return ast2string(simplify(t[2:-1]))
####        else:
##        return ensureparen(ast2string(simplify(t[2:-1])))
##    if t[1][0]=='trailer': # t=[[NAME,f],[trailer,[(],[ll],[)]]]
##        # just simplify arguments to functions
##        return ast2string([['NAME',t[0][1]], ['trailer',['LPAR','('],
##                                              simplify(t[1][2]),['RPAR',')']]])
##    return s


def simplify(t):
    return t
##    """Attempt to simplify symbolic expression string.
##    Adapted by R. Clewley from original DiffStr() code by Pearu Peterson and Ryan Gutenkunst.
##
##    Essentially the same format as DiffStr() except we just move down the
##    syntax tree calling appropriate 'do' functions on the parts to
##    simplify them."""
##    if isinstance(t, list) and len(t) == 1:
##        return simplify(t[0])
##    if t[0]=='NUMBER':
##        if 'e' in t[1]:
##            epos=t[1].find('e')
##            man=t[1][:epos]
##            ex=t[1][epos:]
##        else:
##            man=t[1]
##            ex=''
##        if man[-1] == '.':
##            return ['NUMBER', man+'0'+ex]
##        else:
##            return t
##    if t[0]=='NAME':
##        return t
##    if t[0]=='factor':
##        return string2ast(doneg(ensureparen(ast2string(simplify(t[2:][0])))))
##    if t[0]=='arith_expr':
##        return collect_numbers(t)
##    if t[0]=='term':
##        return collect_numbers(ensureparen_div(t))
##    if t[0]=='xor_expr' and t[2]=='CIRCUMFLEX': # covers alternative power syntax
##        alt = copy(t)
##        alt[0] = 'power'; alt[2]=='DOUBLESTAR'
##        return toCircumflexSyntax(simplify(alt))
##    if t[0]=='power': # covers math functions like sin, cos and log10
##        if t[2][0]=='trailer':
##            if len(t)>3:
##                term1 = simplify(t[:3])
##                if term1[0] in ['NUMBER', 'NAME']: # 'power'
##                    formatstr='%s'
##                else:
##                    formatstr='(%s)'
##                return [t[0],string2ast(formatstr%ast2string(term1)),simplify(t[3:])]
##            elif len(t)==3 and t[1][1]=='pow':
##                # 'pow' syntax case
##                terms = t[2][2][1::2]
##                return string2ast(dopower(ast2string(simplify(terms[0])),
##                                          ast2string(simplify(terms[1]))))
##            # else ignore and carry on
##        ts=[];o=[];
##        for i in t[1:]:
##            if i[0]=='DOUBLESTAR':
##                if len(o)==1: o=o[0]
##                ts.append(o);
##                o=[]
##            else: o.append(simplify(i))
##        if len(o)==1: o=o[0]
##        ts.append(o)
##        if t[2][0]=='DOUBLESTAR':
##            st,lft,rt=map(ast2string,[t,ts[0],ts[1]])
##            return string2ast(dopower(simplify_str(lft),simplify_str(rt)))
##        if t[2][0]=='trailer':
##            return simplify(ts[0])
##    if t[0] in ['arglist','testlist']:
##        o=[]
##        for i in t[1::2]:
##            o.append(ast2string(simplify(i)))
##        return string2ast(','.join(o))
##    if t[0]=='atom':
####        if t[1][0]=='LPAR' and t[-1][0]=='RPAR':
####            return simplify(t[2:-1])
####        else:
##        return string2ast(ensureparen(ast2string(simplify(t[2:-1]))))
##    if t[1][0]=='trailer': # t=[[NAME,f],[trailer,[(],[ll],[)]]]
##        # just simplify arguments to functions
##        return [['NAME',t[0][1]], ['trailer',['LPAR','('],
##                                   simplify(t[1][2]),['RPAR',')']]]
##    return t



# ----------------------------------------------------------------------------

class symbolMapClass(object):
    """Abstract class for hassle-free symbol re-mappings."""
    def __init__(self, symbolMap=None):
        if isinstance(symbolMap, symbolMapClass):
            self.lookupDict = copy(symbolMap.lookupDict)
        elif symbolMap is None:
            self.lookupDict = {}
        else:
            self.lookupDict = copy(symbolMap)

    def __call__(self, arg):
        #print("symbolMapClass %s, with arg %s"%(self.lookupDict.items()[0], str(arg)))
        if isinstance(arg, str):
            if arg in self.lookupDict:
                return self.lookupDict[arg]
            else:
                try:
                    po = parserObject(arg, False)
                except:
                    # cannot do anything to it!
                    return arg
                else:
                    if len(po.tokenized) <= 1:
                        # don't recurse, we have a single token or whitespace/CR/LF
                        return arg
                    else:
                        return "".join(mapNames(self,po.tokenized))
        elif hasattr(arg, 'mapNames'):
            # Quantity or QuantSpec
            res = copy(arg)
            res.mapNames(self)
            return res
        elif hasattr(arg, 'coordnames'):
            # treat as point or pointset -- just transform the coordnames
            # ensure return type is the same
            res = copy(arg)
            try:
                res.mapNames(self)
            except AttributeError:
                raise TypeError("symbolMapClass does not know how to "
                                "process this type of argument")
            return res
        elif hasattr(arg, 'items'):
            # ensure return type is the same
            try:
                res = copy(arg)
            except TypeError:
                # not copyable, so no need to worry
                res = arg
            try:
                for k, v in arg.items():
                    new_k = self.__call__(k)
                    new_v = self.__call__(v)
                    res[new_k] = new_v
                    # delete unprocessed entry in res, from copy of arg
                    if k != new_k:
                        del res[k]
            except TypeError:
                # probably not a mutable type - we know what to do with a tuple
                if isinstance(arg, tuple):
                    return tuple([self.__getitem__(v) for v in arg])
                else:
                    return arg
            except:
                raise TypeError("symbolMapClass does not know how to "
                                "process this type of argument")
            return res
        else:
            # assume arg is iterable and mutable (list, array, etc.)
            # ensure return type is the same
            try:
                res = copy(arg)
            except TypeError:
                # not copyable, so no need to worry
                res = arg
            try:
                for i, v in enumerate(arg):
                    # overwrite unprocessed entry in res, from copy of arg
                    res[i] = self(v)
            except TypeError:
                # probably not a mutable type - we know what to do with a tuple
                if isinstance(arg, tuple):
                    return tuple([self(v) for v in arg])
                else:
                    try:
                        return self.__getitem__(res)
                    except:
                        raise TypeError("symbolMapClass does not know how to "
                         "process this type of argument (%s)"%str(type(arg)))
            except:
                try:
                    return self.__getitem__(res)
                except:
                    raise TypeError("symbolMapClass does not know how to "
                         "process this type of argument (%s)"%str(type(arg)))
            else:
                return res

    __hash__ = None

    def __eq__(self, other):
        try:
            return self.lookupDict == other.lookupDict
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        return symbolMapClass(deepcopy(self.lookupDict))

    def __setitem__(self, symbol, mappedsymbol):
        self.lookupDict[symbol] = mappedsymbol

    def __getitem__(self, symbol):
        try:
            return self.lookupDict[symbol]
        except (KeyError, TypeError):
            return symbol

    def __delitem__(self, symbol):
        del self.lookupDict[symbol]

    def __contains__(self, symbol):
        return self.lookupDict.__contains__(symbol)

    def keys(self):
        return list(self.lookupDict.keys())

    def values(self):
        return list(self.lookupDict.values())

    def items(self):
        return list(self.lookupDict.items())

    def iterkeys(self):
        return iter(self.lookupDict.keys())

    def itervalues(self):
        return iter(self.lookupDict.values())

    def iteritems(self):
        return iter(self.lookupDict.items())

    def inverse(self):
        return symbolMapClass(dict([(v, k) for k, v in self.lookupDict.items()]))

    def update(self, amap):
        try:
            # perhaps passed a symbolMapClass object
            self.lookupDict.update(amap.lookupDict)
        except AttributeError:
            # was passed a dict
            self.lookupDict.update(amap)

    def reorder(self):
        """Return numpy array of indices that can be used to re-order
        a list of values that have been sorted by this symbol map object,
        so that the list becomes ordered according to the alphabetical
        order of the map's keys.
        """
        # sorted by keys
        keys, vals = sortedDictLists(self.lookupDict, byvalue=False)
        return np.argsort(vals)


    def __len__(self):
        return len(self.lookupDict)

    def copy(self):
        return symbolMapClass(self)

    def __repr__(self):
        return "Symbol mapping"

    __str__ = __repr__

    def has_key(self, k):
        return k in self.lookupDict


class auxfnDBclass(object):
    """Auxiliary function database, for use by parsers."""
    def __init__(self):
        self.auxnames = {}

    def addAuxFn(self, auxfnName, parserObj):
        if parserObj not in self.auxnames:
            self.auxnames[parserObj] = auxfnName
        else:
            raise ValueError("Parser object " + parserObj.name + " already "
                             "exists in auxiliary function database")

    def __repr__(self):
        return "ModelSpec internal helper class: auxfnDBclass object"

    __str__ = __repr__

    def __call__(self, parserObj=None):
        if parserObj is None:
            # return all auxiliary functions known
            return list(self.auxnames.values())
        else:
            try:
                return [self.auxnames[parserObj]]
            except KeyError:
                return []

    def removeAuxFn(self, auxfnName):
        flagdelete = None
        for k, v in self.auxnames.items():
            if v == auxfnName:
                flagdelete = k
                break
        if flagdelete is not None:
            del self.auxnames[k]

    def clear(self, parserObj):
        if parserObj in self.auxnames:
            del self.auxnames[parserObj]

    def clearall(self):
        self.auxnames = {}


# only need one of these per session
global protected_auxnamesDB
protected_auxnamesDB = auxfnDBclass()


class parserObject(object):
    """Alphanumeric symbol (pseudo-)parser for mathematical expressions.

    An AST is not properly implemented -- rather, we tokenize,
    identify free symbols, and apply a small number of syntactic rule checks.
    The target language parser is relied upon for full syntax checking.
    """

    def __init__(self, specStr, includeProtected=True,
                 treatMultiRefs=False, ignoreTokens=[],
                 preserveSpace=False):
        # freeSymbols does not include protected names, and so forth.
        # freeSymbols only contains unrecognized alphanumeric symbols.
        self.usedSymbols = []
        self.freeSymbols = []
        self.preserveSpace = preserveSpace
        # token by token list of the specStr
        self.tokenized = []
        if type(specStr) is str:
            self.specStr = specStr
        else:
            print("Found type %s: %r" % (type(specStr), specStr))
            raise TypeError("specStr must be a string")
        self.treatMultiRefs = treatMultiRefs
        # record init options in case want to reset
        self.ignoreTokens = copy(ignoreTokens)
        self.includeProtected = includeProtected
        # process specStr with empty symbol map to create
        # self.freeSymbols and self.usedSymbols
        self.parse(ignoreTokens, None, includeProtected, reset=True)


    def isCompound(self, ops=['+', '-', '*', '/']):
        """Function to verify whether an expression is 'compound',
        in the sense that it has an operator at the root of its syntax
        parse tree (i.e. not inside braces)."""
        result = False
        nested = 0
        if len(self.tokenized) > 2:
            stage = 0
            for s in self.tokenized:
                if stage == 0:
                    if s in self.usedSymbols and nested == 0:
                        stage = 1
                    elif s == ')':
                        stage = 1
                        nested = max([0, nested-1])
                    elif s == '(':
                        nested += 1
                elif stage == 1:
                    if s in ops and nested == 0:
                        stage = 2
                    elif s == '(':
                        nested += 1
                    elif s == ')':
                        nested = max([0, nested-1])
                    elif nested == 0:
                        stage = 0
                elif stage == 2:
                    if s in self.usedSymbols and nested == 0:
                        stage = 3
                    elif s == '(':
                        stage = 3
                        nested += 1
                    elif s == ')':
                        nested = max([0, nested-1])
                    elif nested == 0:
                        stage = 0
                if stage == 3:
                    result = True
                    break
        return result

    def __call__(self, specialtoks=None, symbolMap=None, includeProtected=True):
        if specialtoks is None:
            if self.ignoreTokens is not None:
                specialtoks = self.ignoreTokens
            else:
                specialtoks = []
        if self.tokenized == []:
            return self.parse(specialtoks, symbolMap, includeProtected)
        else:
            if symbolMap is None:
                return "".join(self.tokenized)
            else:
                return "".join(symbolMap(self.tokenized))

    def find(self, token):
        """Find all occurrences of the given token in the expression, returning a list
        of indices (empty if not present).
        """
        if self.tokenized == []:
            self.parse([])
        return [i for i, t in enumerate(self.tokenized) if t == token]

    def parse(self, specialtoks, symbolMap=None, includeProtected=True,
              reset=False):
        if reset:
            self.usedSymbols = []
            self.freeSymbols = []
        if symbolMap is None:
            # dummy identity function
            symbolMap = lambda x: x
        specialtokens = specialtoks + ['('] + self.usedSymbols
        if includeProtected:
            specialtokens.extend(protected_allnames)
            protected_auxnames = protected_auxnamesDB(self)
        else:
            protected_auxnames = []
        if self.treatMultiRefs:
            specialtokens.append('[')
        dohierarchical = '.' not in specialtokens
        allnames = specialtokens + protected_auxnames
        specstr = self.specStr   # eases notation
        returnstr = ""
        if specstr == "":
            # Hack for empty specs to pass without losing their last characters
            specstr = " "
        elif specstr[-1] != ')':
            # temporary hack because strings not ending in ) lose their last
            # character!
            # Problem could be with line "if scount < speclen - 1:" below
            # ... should it be <= ?
            specstr += " "
        scount = 0
        speclen = len(specstr)
        # temp holders for used and free symbols
        used = copy(self.usedSymbols)
        free = copy(self.freeSymbols)
        tokenized = []
        # current token being built is: s
        # current character being processed is: stemp
        s = ''
        foundtoken = False
        while scount < speclen:
            stemp = specstr[scount]
            ## DEBUGGING PRINT STATEMENTS
##            print "\n*********************************************"
##            print specstr[:scount]
##            print stemp
##            print s
##            print tokenized
            scount += 1
            if name_chars_RE.match(stemp) is None:
                # then stemp is a non-alphanumeric char
                if s not in ['', ' ', '\n', '\t']:
                    # checking allnames catches var names etc. that are valid
                    # in auxiliary functions but are not special tokens
                    # and must be left alone
                    if s in allnames:
                        snew = symbolMap(s)
                        tokenized.append(snew)
                        if snew not in used:
                            used.append(snew)
                        returnstr += snew
                    else:
                        # isdecimal cases:
                        #  s = number followed by a '.'
                        #  s = number with 'e' followed by +/-
                        #      (just a number dealt with elsewhere)
                        isnumtok = isNumericToken(s)
                        issimpledec = stemp == '.' and isnumtok
                        isexpdec = s[-1] in ['e','E'] and s[0] not in ['e','E']\
                             and stemp in ['-','+'] and isnumtok
                        isdecimal = issimpledec or isexpdec
                        ishierarchicalname = stemp == '.' and isNameToken(s)
                        if isdecimal or (ishierarchicalname and dohierarchical):
                            # continue building token
                            s += stemp
                            continue
                        else:
                            # We have found a complete token
                            snew = symbolMap(s)
                            if s[0] not in num_chars + ['+','-'] \
                                    and snew not in free:
                                free.append(snew)
                            tokenized.append(snew)
                            if snew not in used:
                                used.append(snew)
                            returnstr += snew
                if stemp in ['+', '-']:
                    # may be start of a unary number
                    try:
                        next_stemp = specstr[scount]
                    except IndexError:
                        tokenized.append(stemp)
                        returnstr += stemp
                        s = ''
                        continue
                    if (tokenized==[] or tokenized[-1]=='(') and \
                       next_stemp in num_chars + ['.']:
                        # continue to build token
                        s += stemp
                        continue
                    elif len(tokenized)>0 and tokenized[-1] == '+':
                        # process double sign
                        if stemp == '-':
                            tokenized[-1] = '-'
                            returnstr = returnstr[:-1] + '-'
                        # else do nothing -- keep just one '+'
                        s = ''
                        continue
                    elif len(tokenized)>0 and tokenized[-1] == '-':
                        # process double sign
                        if stemp == '-':
                            tokenized[-1] = '+'
                            returnstr = returnstr[:-1] + '+'
                        # else do nothing -- keep just one '-'
                        s = ''
                        continue
                    else:
                        tokenized.append(stemp)
                        returnstr += stemp
                        s = ''
                        continue
                elif stemp in ['`', '!', '@', '#', '$', '{',
                               '}', "\\"]:
                    if stemp in specialtokens:
                        tokenized.append(stemp)
                        returnstr += stemp
                        s = ''
                        continue
                    else:
##                        if stemp == '^':
##                            raise ValueError('Symbol ^ is not allowed. '
##                                             'Please use the pow() call')
##                        else:
                        print("Problem with string '%s'"%specstr)
                        raise ValueError('Symbol %s is illegal. '%stemp)
                elif stemp == '[':
                    # self.treatMultiRefs == False and '[' in specialtokens
                    # means it was probably in the ignoreToken list in __init__
                    if self.treatMultiRefs and len(tokenized)>0 \
                            and (tokenized[-1].isalnum() or \
                                 ('[' in specialtokens and not ( \
                                  isVectorClause(specstr[scount-1:]) or \
                                  len(tokenized)>1 and tokenized[-2] in \
                                  ('max', 'min', 'max_', 'min_')))):
                        # then this is probably an actual multiRef
                        s = '['
                    elif '[' in specialtokens:
                        returnstr += '['
                        # s already to tokenized in this case
                        tokenized.append('[')   # was just '['
                        s = ''
                        continue
                    else:
                        raise ValueError("Syntax error: Square braces not to "
                                         "be used outside of multiple Quantity"
                                         " definitions and references")
                # only use next clause if want to process function call
                # arguments specially
#                elif stemp == '(':
#                    returnstr += s
#                    s = stemp
                else:
                    if stemp == "*":
                        if len(returnstr)>1 and returnstr[-1] == "*":
                            # check for ** case
                            if tokenized[-1] == '*':
                                tokenized[-1] = '**'
                            else:
                                tokenized.append('**')
                            s = ''
                            returnstr += stemp
                            continue   # avoids returnstr += stemp below
##                            if "**" in specialtokens:
##                                if tokenized[-1] == '*':
##                                    tokenized[-1] = '**'
##                                else:
##                                    tokenized.append('**')
##                                s = ''
##                                returnstr += "**"
##                                continue   # avoids returnstr += stemp below
##                            else:
##                                raise ValueError('Operator ** is not allowed. '
##                                       'Please use the pow() call')
                        else:
                            # just a single *
                            tokenized.append('*')
                    elif stemp == "=":
                        if len(returnstr)>1:
                             # check for >= and <= cases
                            if returnstr[-1] == ">":
                                if tokenized[-1] == '>':
                                    tokenized[-1] = '>='
                                else:
                                    tokenized.append('>=')
                                s = ''
                                returnstr += stemp
                                continue   # avoids returnstr += stemp below
                            elif returnstr[-1] == "<":
                                if tokenized[-1] == '<':
                                    tokenized[-1] = '<='
                                else:
                                    tokenized.append('<=')
                                s = ''
                                returnstr += stemp
                                continue   # avoids returnstr += stemp below
                            else:
                                tokenized.append('=')
                        else:
                            tokenized.append('=')
                    elif stemp in [" ","\t","\n"]:
                        if self.preserveSpace: tokenized.append(stemp)
                    else:
                        tokenized.append(stemp)
                    s = ''
                    returnstr += stemp
                    continue
            else:
                s += stemp
            if s in specialtokens:
                # only use next clause if want to process function call
                # arguments specially
#                if s == '(':
#                    foundtoken = False  # so that don't enter next if statement
#                    tokenized.append(s)
#                    returnstr += s
#                    s = ''
                if s == '[' and self.treatMultiRefs and len(tokenized)>0 \
                        and (tokenized[-1].isalnum() or \
                             ('[' in specialtokens and not isVectorClause(specstr[scount-1:]))):
                    # then this is probably an actual multiRef ...
                    # will treat as multiRef if there's an alphanumeric
                    # token directly preceding '[', e.g. z[i,0,1]
                    # or if commas are not inside,
                    # otherwise would catch vector usage, e.g. [[x,y],[a,b]]
                    foundtoken = False    # don't go into next clause
                    # copy the square-bracketed clause to the output
                    # verbatim, and add whole thing as a token.
                    try:
                        rbpos = specstr[scount:].index(']')
                    except ValueError:
                        raise ValueError("Mismatch [ and ] in spec")
                    # expr includes both brackets
                    expr = specstr[scount-1:scount+rbpos+1]
                    # find index name in this expression. this should
                    # be the only free symbol, otherwise syntax error.
                    temp = parserObject(expr[1:-1],
                                        includeProtected=False)
                    if len(temp.freeSymbols) == 1:
                        free.extend(temp.freeSymbols)
                        # not sure whether should add to usedSymbols
                        # for consistency with freeSymbols, or leave
                        # it out for consistency with tokenized
#                                used.append(temp.freeSymbols)
                    else:
                        raise ValueError("Invalid index clause in "
                                         "multiple quantity reference -- "
                                         "multiple index names used in [...]")
                    # start next parsing iteration after the ']'
                    scount += rbpos+1
                    # treat whole expression as one symbol
                    returnstr += expr
                    tokenized.append(expr)
                    used.append(expr)
                    s = ''
                else:
                    if scount < speclen - 1:
                        if name_chars_RE.match(specstr[scount]) is None:
                            foundtoken = True
                        else:
                            if s[-1] in ['e','E'] and s[0] not in ['e','E'] and \
                               name_chars_RE.match(specstr[scount]).group() \
                                       in num_chars+['-','+']:
                                # not expecting an arithmetic symbol or space
                                # ... we *are* expecting a numeric
                                foundtoken = True
                    else:
                        foundtoken = True
                if foundtoken:
                    if includeProtected:
                        if s == 'for':
                            # check next char is '('
                            if specstr[scount] != '(':
                                print("Next char found:%s" % specstr[scount])
                                raise ValueError("Invalid 'for' macro syntax")
                            # find next ')' (for statement should contain no
                            # braces itself)
                            try:
                                rbpos = specstr[scount:].index(')')
                            except ValueError:
                                raise ValueError("Mismatch ( and ) in 'for' "
                                                 "macro")
                            # expr includes both brackets
                            expr = specstr[scount:scount+rbpos+1]
                            # find index name in this expression. this should
                            # be the only free symbol, otherwise syntax error.
                            temp = parserObject(expr, includeProtected=False)
                            macrotests = [len(temp.tokenized) == 7,
                               temp.tokenized[2] == temp.tokenized[4] == ',',
                               temp.tokenized[5] in ['+','*']]
                            if not alltrue(macrotests):
                                print("specstr was: %s" % specstr)
                                print("tokens: %r" % temp.tokenized)
                                print("test results: %r" % macrotests)
                                raise ValueError("Invalid sub-clause in "
                                                 "'for' macro")
                            # start next parsing iteration after the ')'
                            scount += rbpos+1
                            # keep contents of braces as one symbol
                            returnstr += s+expr
                            tokenized.extend([s,expr])
                            if s not in used:
                                used.append(s)
                            if expr not in used:
                                used.append(expr)
                        elif s == 'abs':
                            snew = symbolMap(s)
                            returnstr += snew
                            tokenized.append(snew)
                            if snew not in used:
                                used.append(snew)
                        elif s in protected_scipynames + protected_numpynames \
                                 + protected_specialfns:
                            snew = symbolMap(s)
                            returnstr += snew
                            tokenized.append(snew)
                            if snew not in used:
                                used.append(snew)
                        elif s in protected_mathnames:
                            if s in ['e','E']:
                                # special case where e is either = exp(0)
                                # as a constant or it's an exponent in 1.0e-4
                                if len(returnstr)>0:
                                    if returnstr[-1] not in num_chars+['.']:
                                        snew = symbolMap(s.lower())
                                        returnstr += snew
                                        tokenized.append(snew)
                                        if snew not in used:
                                            used.append(snew)
                                    else:
                                        returnstr += s
                                else:
                                    snew = symbolMap(s.lower())
                                    returnstr += snew
                                    tokenized.append(snew)
                                    if snew not in used:
                                        used.append(snew)
                            else:
                                snew = symbolMap(s)
                                returnstr += snew
                                tokenized.append(snew)
                                if snew not in used:
                                    used.append(snew)
                        elif s in protected_randomnames:
                            snew = symbolMap(s)
                            returnstr += snew
                            tokenized.append(snew)
                            if snew not in used:
                                used.append(snew)
                        elif s in protected_auxnames:
                            snew = symbolMap(s)
                            returnstr += snew
                            tokenized.append(snew)
                        else:
                            # s is e.g. a declared argument to an aux fn but
                            # only want to ensure it is present. no action to
                            # take.
                            snew = symbolMap(s)
                            tokenized.append(snew)
                            returnstr += snew
                    else:
                        # not includeProtected names case, so just map
                        # symbol
                        snew = symbolMap(s)
                        tokenized.append(snew)
                        returnstr += snew
                    # reset for next iteration
                    s = ''
                    foundtoken = False
        # end of scount while loop
        if reset:
            # hack to remove any string literals
            actual_free = [sym for sym in free if sym in tokenized]
            for sym in [sym for sym in free if sym not in actual_free]:
                is_literal = False
                for tok in tokenized:
                    # does symbol appear in quotes inside token?
                    # if so, then it's a literal and not a free symbol
                    if ('"' in tok or "'" in tok) and sym in tok:
                        start_ix = tok.index(sym)
                        end_ix = start_ix + len(sym) - 1
                        if len(tok) > end_ix and start_ix > 0:
                            # then symbol is embedded inside the string
                            doub = tok[start_ix-1] == tok[end_ix+1] == '"'
                            sing = tok[start_ix-1] == tok[end_ix+1] == "'"
                            is_literal = doub or sing
                if not is_literal:
                    actual_free.append(sym)
            self.usedSymbols = used
            self.freeSymbols = actual_free
            self.specStr = returnstr
            self.tokenized = tokenized
        # strip extraneous whitespace
        return returnstr.strip()


#-----------------------------------------------------------------------------


def isToken(s, treatMultiRefs=False):
    # token must be alphanumeric string (with no punctuation, operators, etc.)
    if not isinstance(s, str):
        return False
    try:
        temp = parserObject(s, includeProtected=False,
                            treatMultiRefs=treatMultiRefs)
    except ValueError:
        return False
    if treatMultiRefs and s.find('[')>0:
        lenval = 2
    else:
        lenval = 1
    return not temp.isCompound() and len(temp.usedSymbols) == lenval \
            and len(temp.freeSymbols) == lenval \
            and len(temp.tokenized) == lenval


def isNameToken(s, treatMultiRefs=False):
    return isToken(s, treatMultiRefs=treatMultiRefs) \
        and s[0] not in num_chars and not (len(s)==1 and s=="_")

def isIntegerToken(arg):
    return alltrue([t in '0123456789' for t in arg])

def isNumericToken(arg):
    # supports unary + / - at front, and checks for usage of exponentials
    # (using 'E' or 'e')
    try:
        s = arg.lower()
    except AttributeError:
        return False
    try:
        if s[0] in ['+','-']:
            s_rest = s[1:]
        else:
            s_rest = s
    except IndexError:
        return False
    pts = s.count('.')
    exps = s.count('e')
    pm = s_rest.count('+') + s_rest.count('-')
    if pts > 1 or exps > 1 or pm > 1:
        return False
    if exps == 1:
        exp_pos = s.find('e')
        pre_exp = s[:exp_pos]
        # must be numbers before and after the 'e'
        if not sometrue([n in num_chars for n in pre_exp]):
            return False
        if s[-1]=='e':
            # no chars after 'e'!
            return False
        if not sometrue([n in num_chars for n in s[exp_pos:]]):
            return False
        # check that any additional +/- occurs directly after 'e'
        if pm == 1:
            pm_pos = max([s_rest.find('+'), s_rest.find('-')])
            if s_rest[pm_pos-1] != 'e':
                return False
            e_rest = s_rest[pm_pos+1:]   # safe due to previous check
        else:
            e_rest = s[exp_pos+1:]
        # only remaining chars in s after e and possible +/- are numbers
        if '.' in e_rest:
            return False
    # cannot use additional +/- if not using exponent
    if pm == 1 and exps == 0:
        return False
    return alltrue([n in num_chars + ['.', 'e', '+', '-'] for n in s_rest])


def findNumTailPos(s):
    """Find position of numeric tail in alphanumeric string.

    e.g. findNumTailPos('abc678') = 3"""
    try:
        l = len(s)
        if l > 1:
            if s[-1] not in num_chars or s[0] in num_chars:
                raise ValueError("Argument must be an alphanumeric string "
                              "starting with a letter and ending in a number")
            for i in range(1, l+1):
                if s[-i] not in num_chars:
                    return l-i+1
        else:
            raise ValueError("Argument must be alphanumeric string starting "
                             "with a letter and ending in a number")
    except TypeError:
        raise ValueError("Argument must be alphanumeric string starting "
                             "with a letter and ending in a number")


def isHierarchicalName(s, sep=NAMESEP, treatMultiRefs=False):
    s_split = s.split(sep)
    return len(s_split) > 1 and alltrue([isNameToken(t, treatMultiRefs) \
                                         for t in s_split])

def isVectorClause(s):
    brace = findEndBrace(s, '[',']')
    if s[0] == '[' and isinstance(brace, int):
        return ',' in s[1:brace]


def replaceSepList(speclist):
    return [replaceSep(spec) for spec in speclist]

def replaceSepListInv(speclist):
    return [replaceSepInv(spec) for spec in speclist]

def replaceSepInv(spec):
    """Invert default name separator replacement."""
    return replaceSep(spec, "_", NAMESEP)


def replaceSep(spec, sourcesep=NAMESEP, targetsep="_"):
    """Replace hierarchy separator character with another and return spec
    string. e.g. "." -> "_"
    Only replaces the character between name tokens, not between numbers."""

    try:
        assert all([alphanumeric_chars_RE.match(char) is None \
                    for char in sourcesep])
    except AssertionError:
        raise ValueError("Source separator must be non-alphanumeric")
    try:
        assert all([alphanumeric_chars_RE.match(char) is None \
                    for char in targetsep])
    except AssertionError:
        raise ValueError("Target separator must be non-alphanumeric")
    if isinstance(spec, str):
        return replaceSepStr(spec, sourcesep, targetsep)
    else:
        # safe way to get string definition from either Variable or QuantSpec
        return replaceSepStr(str(spec()), sourcesep, targetsep)


def mapNames(themap, target):
    """Map names in <target> argument using the symbolMapClass
    object <themap>, returning a renamed version of the target.
    N.B. Only maps the keys of a dictionary type"""
    try:
        themap.lookupDict
    except AttributeError:
        t = repr(type(themap))
        raise TypeError("Map argument must be of type symbolMapClass, not %s"%t)
    if hasattr(target, 'mapNames'):
        ct = copy(target)
        ct.mapNames(themap)  # these methods work in place
        return ct
    elif isinstance(target, list):
        return themap(target)
    elif isinstance(target, tuple):
        return tuple(themap(target))
    elif hasattr(target, 'items'):
        o = {}
        for k, v in target.items():
            o[themap(k)] = v
        return o
    elif isinstance(target, str):
        return themap(target)
    elif target is None:
        return None
    else:
        raise TypeError("Invalid target type %s"%repr(type(target)))


## internal functions for replaceSep
def replaceSepQSpec(spec, sourcesep, targetsep):
    # currently unused because many QuantSpec's with hierarchical names
    # are not properly tokenized! They may have the tokenized form
    # ['a_parent','.','a_child', <etc.>] rather than
    # ['a_parent.a_child', <etc.>]
    outstr = ""
    # search for pattern <nameToken><sourcechar><nameToken>
    try:
        for t in spec[:]:
            if isHierarchicalName(t, sourcesep):
                outstr += t.replace(sourcesep, targetsep)
            else:
                # just return original token
                outstr += t
    except AttributeError:
        raise ValueError("Invalid QuantSpec passed to replaceSep()")
    return outstr


def replaceSepStr(spec, sourcesep, targetsep):
    # spec is a string (e.g. 'leaf.v'), and p.tokenized contains the
    # spec split by the separator (e.g. ['leaf', '.', 'v']) if the separator
    # is not "_", otherwise it will be retained as part of the name
    outstr = ""
    treatMultiRefs = '[' in spec and ']' in spec
    p = parserObject(spec, treatMultiRefs=treatMultiRefs,
                     includeProtected=False)
    # search for pattern <nameToken><sourcechar><nameToken>
    # state 0: not in pattern
    # state 1: found nameToken
    # state 2: found nameToken then sourcechar
    # state 3: found complete pattern
    state = 0
    for t in p.tokenized:
        if isNameToken(t):
            if sourcesep in t:
                # in case sourcesep == '_'
                tsplit = t.split(sourcesep)
                if alltrue([isNameToken(ts) for ts in tsplit]):
                    outstr += targetsep.join(tsplit)
                else:
                    outstr += t
            else:
                if state == 0:
                    state = 1
                    outstr += t
                elif state == 1:
                    state = 0
                    outstr += t
                else:
                    # state == 2
                    state = 1   # in case another separator follows
                    outstr += targetsep + t
            continue
        elif t == sourcesep:
            if state == 1:
                state = 2
                # delay output until checked next token
            else:
                # not part of pattern, so reset
                state = 0
                outstr += t
        else:
            state = 0
            outstr += t
    return outstr


def joinStrs(strlist):
    """Join a list of strings into a single string (in order)."""

    return ''.join(strlist)


def joinAsStrs(objlist,sep=""):
    """Join a list of objects in their string representational form."""

    retstr = ''
    for o in objlist:
        if type(o) is str:
            retstr += o + sep
        else:
            retstr += str(o) + sep
    avoidend = len(sep)
    if avoidend > 0:
        return retstr[:-avoidend]
    else:
        return retstr


def count_sep(specstr, sep=','):
    """Count number of specified separators (default = ',') in given string,
    avoiding occurrences of the separator inside nested braces"""

    num_seps = 0
    brace_depth = 0
    for s in specstr:
        if s == sep and brace_depth == 0:
            num_seps += 1
        elif s == '(':
            brace_depth += 1
        elif s == ')':
            brace_depth -= 1
    return num_seps


def parseMatrixStrToDictStr(specstr, specvars, m=0):
    """Convert string representation of m-by-n matrix into a single
    string, assuming a nested comma-delimited list representation in
    the input, and outputting a dictionary of the sub-lists, indexed
    by the ordered list of names specvars (specified as an
    argument)."""

    specdict = {}
    # matrix is n by m
    n = len(specvars)
    if n == 0:
        raise ValueError("parseMatrixStrToDictStr: specvars was empty")
    if m == 0:
        # assume square matrix
        m = len(specvars)
    # strip leading and trailing whitespace
    spectemp1 = specstr.strip()
    assert spectemp1[0] == '[' and spectemp1[-1] == ']', \
        ("Matrix must be supplied as a Python matrix, using [ and ] syntax")
    # strip first [ and last ] and then all whitespace and \n
    spectemp2 = spectemp1[1:-1].replace(' ','').replace('\n','')
    splitdone = False
    entrycount = 0
    startpos = 0
    try:
        while not splitdone:
            nextrbrace = findEndBrace(spectemp2[startpos:],
                                       '[', ']') + startpos
            if nextrbrace is None:
                raise ValueError("Mismatched braces before end of string")
            specdict[specvars[entrycount]] = \
                                spectemp2[startpos:nextrbrace+1]
            entrycount += 1
            if entrycount < n:
                nextcomma = spectemp2.find(',', nextrbrace)
                if nextcomma > 0:
                    nextlbrace = spectemp2.find('[', nextcomma)
                    if nextlbrace > 0:
                        startpos = nextlbrace
                    else:
                        raise ValueError("Not enough comma-delimited entries")
                else:
                    raise ValueError("Not enough comma-delimited entries")
            else:
                splitdone = True
    except:
        print("Error in matrix specification")
        raise
    return specdict


def readArgs(argstr, lbchar='(', rbchar=')'):
    """Parse arguments out of string beginning and ending with braces
    (default: round brace).

    Returns a triple: [success_boolean, list of arguments, number of args]"""
    bracetest = argstr[0] == lbchar and argstr[-1] == rbchar
    rest = argstr[1:-1].replace(" ","")
    pieces = []
    while True:
        if '(' in rest:
            lix = rest.index('(')
            rix = findEndBrace(rest[lix:]) + lix
            new = rest[:lix].split(",")
            if len(pieces) > 0:
                pieces[-1] = pieces[-1] + new[0]
                pieces.extend(new[1:])
            else:
                pieces.extend(new)
            if len(pieces) > 0:
                pieces[-1] = pieces[-1] + rest[lix:rix+1]
            else:
                pieces.append(rest[lix:rix+1])
            rest = rest[rix+1:]
        else:
            new = rest.split(",")
            if len(pieces) > 0:
                pieces[-1] = pieces[-1] + new[0]
                pieces.extend(new[1:])
            else:
                pieces.extend(new)
            # quit while loop
            break
    return [bracetest, pieces, len(argstr)]


def findEndBrace(s, lbchar='(', rbchar=')'):
    """Find position in string (or list of strings), s, at which final matching
    brace occurs (if at all). If not found, returns None.

    s[0] must be the left brace character. Default left and right braces are
    '(' and ')'. Change them with the optional second and third arguments.
    """
    pos = 0
    assert s[0] == lbchar, 'string argument must begin with left brace'
    stemp = s
    leftbrace_count = 0
    notDone = True
    while len(stemp) > 0 and notDone:
        # for compatibility with s being a list, use index method
        try:
            left_pos = stemp.index(lbchar)
        except ValueError:
            left_pos = -1
        try:
            right_pos = stemp.index(rbchar)
        except ValueError:
            right_pos = -1
        if left_pos >= 0:
            if left_pos < right_pos:
                if left_pos >= 0:
                    leftbrace_count += 1
                    pos += left_pos+1
                    stemp = s[pos:]
                else:
                    # no left braces found. next brace is right.
                    if leftbrace_count > 0:
                        leftbrace_count -= 1
                        pos += right_pos+1
                        stemp = s[pos:]
            else:
                # right brace found first
                leftbrace_count -= 1
                pos += right_pos+1
                stemp = s[pos:]
        else:
            if right_pos >= 0:
                # right brace found first
                leftbrace_count -= 1
                pos += right_pos+1
                stemp = s[pos:]
            else:
                # neither were found (both == -1)
                raise ValueError('End of string found before closing brace')
        if leftbrace_count == 0:
            notDone = False
            # adjust for
            pos -= 1
    if leftbrace_count == 0:
        return pos
    else:
        return None


def makeParList(objlist, prefix=''):
    """wrap objlist into a comma separated string of str(objects)"""
    parlist = ', '.join([prefix + str(i) for i in objlist])
    return parlist


def wrapArgInCall(source, callfn, wrapL, wrapR=None, argnums=[0],
                  notFirst=False):
    """Add delimiters to single argument in function call."""
    done = False
    output = ""
    currpos = 0
    first_occurrence = True
    if wrapR is None:
        # if no specific wrapR is specified, just use wrapL
        # e.g. for symmetric delimiters such as quotes
        wrapR = wrapL
    assert isinstance(wrapL, str) and isinstance(wrapR, str), \
           "Supplied delimiters must be strings"
    while not done:
        # find callfn in source
        findposlist = [source[currpos:].find(callfn+'(')]
        try:
            findpos = min([x for x in findposlist if x >= 0]) + currpos
        except ValueError:
            done = True
        if not done:
            # find start and end braces
            startbrace = source[findpos:].find('(')+findpos
            endbrace = findEndBrace(source[startbrace:])+startbrace
            output += source[currpos:startbrace+1]
            # if more than one argument present, apply wrapping to specified
            # arguments
            currpos = startbrace+1
            numargs = source[startbrace+1:endbrace].count(',')+1
            if max(argnums) >= numargs:
                raise ValueError("Specified argument number out of range")
            if numargs > 1:
                for argix in range(numargs-1):
                    nextcomma = source[currpos:endbrace].find(',')
                    argstr = source[currpos:currpos + nextcomma]
                    # get rid of leading or tailing whitespace
                    argstr = argstr.strip()
                    if argix in argnums:
                        if first_occurrence and notFirst:
                            output += source[currpos:currpos + nextcomma + 1]
                        else:
                            output += wrapL + argstr + wrapR + ','
                        first_occurrence = False
                    else:
                        output += source[currpos:currpos + nextcomma + 1]
                    currpos += nextcomma + 1
                if numargs-1 in argnums:
                    if first_occurrence and notFirst:
                        # just include last argument as it was if this is
                        # the first occurrence
                        output += source[currpos:endbrace+1]
                    else:
                        # last argument needs to wrapped too
                        argstr = source[currpos:endbrace]
                        # get rid of leading or tailing whitespace
                        argstr = argstr.strip()
                        output += wrapL + argstr + wrapR + ')'
                    first_occurrence = False
                else:
                    # just include last argument as it was
                    output += source[currpos:endbrace+1]
            else:
                if argnums[0] != 0:
                    raise ValueError("Specified argument number out of range")
                if first_occurrence and notFirst:
                    output += source[currpos:endbrace+1]
                else:
                    argstr = source[currpos:endbrace]
                    # get rid of leading or tailing whitespace
                    argstr = argstr.strip()
                    output += wrapL + argstr + wrapR + ')'
                first_occurrence = False
            currpos = endbrace+1
        else:
            output += source[currpos:]
    return output


##def replaceCallsWithDummies(source, callfns, used_dummies=None, notFirst=False):
##    """Replace all function calls in source with dummy names,
##    for the functions listed in callfns. Returns a pair (new_source, d)
##    where d is a dict mapping the dummy names used to the function calls."""
##    # This function used to work on lists of callfns directly, but I can't
##    # see why it stopped working. So I just added this recursing part at
##    # the front to reduce the problem to a singleton function name each time.
##    print "\nEntered with ", source, callfns, used_dummies
##    if used_dummies is None:
##        used_dummies = 0
##    if isinstance(callfns, list):
##        if len(callfns) > 1:
##            res = source
##            dummies = {}
##            #remaining_fns = callfns[:]
##            for f in callfns:
##                #remaining_fns.remove(f)
##                new_res, d = replaceCallsWithDummies(res, [f], used_dummies, notFirst)
##                res = new_res
##                if d != {}:
##                    dummies.update(d)
##                    used_dummies = max( (used_dummies, max(d.keys())) )
##            if dummies != {}: # and remaining_fns != []:
##                new_dummies = dummies.copy()
##                for k, v in dummies.items():
##                    new_v, new_d = replaceCallsWithDummies(v, callfns,
##                                                used_dummies, notFirst=True)
##                    new_dummies[k] = new_v
##                    if new_d != {}:
##                        new_dummies.update(new_d)
##                        used_dummies = max( (used_dummies, max(new_dummies.keys())) )
##                dummies.update(new_dummies)
##            return res, dummies
##    else:
##        raise TypeError("Invalid list of function names")
##    done = False
##    dummies = {}
##    output = ""
##    currpos = 0
##    doneFirst = False
##    while not done:
##        # find any callfns in source (now just always a singleton)
##        findposlist_candidates = [source[currpos:].find(fname+'(') for fname in callfns]
##        findposlist = []
##        for candidate_pos in findposlist_candidates:
##            # remove any that are actually only full funcname matches to longer strings
##            # that happen to have that funcname at the end: e.g. functions ['if','f']
##            # and find a match 'f(' in the string 'if('
##            if currpos+candidate_pos-1 >= 0:
##                if not isNameToken(source[currpos+candidate_pos-1]):
##                    findposlist.append(candidate_pos)
##            else:
##                # no earlier character in source, so must be OK
##                findposlist.append(candidate_pos)
##        if len(findposlist) > 0 and notFirst:
##            findposlist = findposlist[1:]
##        try:
##            findpos = min(filter(lambda x:x>=0, findposlist))+currpos
##        except ValueError:
##            done = True
##        if not done:
##            # find start and end braces
##            startbrace = source[findpos:].find('(')+findpos
##            endbrace = findEndBrace(source[startbrace:])+startbrace
##            sub_source = source[startbrace+1:endbrace]
##            embedded_calls = [sub_source.find(fname+'(') for fname in callfns]
##            try:
##                subpositions = filter(lambda x:x>0, embedded_calls)
##                if subpositions == []:
##                    filtered_sub_source = sub_source
##                    new_d = {}
##                else:
##                    filtered_sub_source, new_d = \
##                            replaceCallsWithDummies(sub_source,
##                                                    callfns, used_dummies)
##            except ValueError:
##                pass
##            else:
##                if new_d != {}:
##                    dummies.update(new_d)
##                    used_dummies = max( (used_dummies, max(dummies.keys())) )
##            used_dummies += 1
##            dummies[used_dummies] = source[findpos:startbrace+1] + \
##                                      filtered_sub_source + ')'
##            output += source[currpos:findpos] + '__dummy%i__' % used_dummies
##            currpos = endbrace+1
##        else:
##            output += source[currpos:]
##    return output, dummies


def replaceCallsWithDummies(source, callfns, used_dummies=None, notFirst=False):
    """Replace all function calls in source with dummy names,
    for the functions listed in callfns. Returns a pair (new_source, d)
    where d is a dict mapping the dummy names used to the function calls.
    """
    # This function used to work on lists of callfns directly, but I can't
    # see why it stopped working. So I just added this recursing part at
    # the front to reduce the problem to a singleton function name each time.
    if used_dummies is None:
        used_dummies = 0
    done = False
    dummies = {}
    output = ""
    currpos = 0
    doneFirst = False
    while not done:
        # find any callfns in source (now just always a singleton)
        findposlist_candidates = [source[currpos:].find(fname+'(') for fname in callfns]
        findposlist = []
        for candidate_pos in findposlist_candidates:
            # remove any that are actually only full funcname matches to longer strings
            # that happen to have that funcname at the end: e.g. functions ['if','f']
            # and find a match 'f(' in the string 'if('
            if currpos+candidate_pos-1 >= 0:
                if not isNameToken(source[currpos+candidate_pos-1]):
                    findposlist.append(candidate_pos)
            else:
                # no earlier character in source, so must be OK
                findposlist.append(candidate_pos)
        findposlist = [ix for ix in findposlist if ix >= 0]
        findposlist.sort()
        if not doneFirst and notFirst and len(findposlist) > 0:
            findposlist = findposlist[1:]
            doneFirst = True
        try:
            findpos = findposlist[0]+currpos
        except IndexError:
            done = True
        if not done:
            # find start and end braces
            startbrace = source[findpos:].find('(')+findpos
            endbrace = findEndBrace(source[startbrace:])+startbrace
            sub_source = source[startbrace+1:endbrace]
            embedded_calls = [sub_source.find(fname+'(') for fname in callfns]
            try:
                subpositions = [x for x in embedded_calls if x > 0]
                if subpositions == []:
                    filtered_sub_source = sub_source
                    new_d = {}
                else:
                    filtered_sub_source, new_d = \
                            replaceCallsWithDummies(sub_source,
                                                    callfns, used_dummies)
            except ValueError:
                pass
            else:
                if new_d != {}:
                    dummies.update(new_d)
                    used_dummies = max( (used_dummies, max(dummies.keys())) )
            used_dummies += 1
            dummies[used_dummies] = source[findpos:startbrace+1] + \
                                      filtered_sub_source + ')'
            output += source[currpos:findpos] + '__dummy%i__' % used_dummies
            currpos = endbrace+1
        else:
            output += source[currpos:]
    if dummies != {}:
        new_dummies = dummies.copy()
        for k, v in dummies.items():
            new_v, new_d = replaceCallsWithDummies(v, callfns,
                                        used_dummies, notFirst=True)
            new_dummies[k] = new_v
            if new_d != {}:
                new_dummies.update(new_d)
                used_dummies = max( (used_dummies, max(new_dummies.keys())) )
        dummies.update(new_dummies)
    return output, dummies


def addArgToCalls(source, callfns, arg, notFirst=''):
    """Add an argument to calls in source, to the functions listed in callfns.
    """
    # This function used to work on lists of callfns directly, but I can't
    # see why it stopped working. So I just added this recursing part at
    # the front to reduce the problem to a singleton function name each time.
    if isinstance(callfns, list):
        if len(callfns) > 1:
            res = source
            for f in callfns:
                res = addArgToCalls(res, [f], arg, notFirst)
            return res
    else:
        raise TypeError("Invalid list of function names")
    done = False
    output = ""
    currpos = 0
    while not done:
        # find any callfns in source (now just always a singleton)
        findposlist_candidates = [source[currpos:].find(fname+'(') for fname in callfns]
        findposlist = []
        for candidate_pos in findposlist_candidates:
            # remove any that are actually only full funcname matches to longer strings
            # that happen to have that funcname at the end: e.g. functions ['if','f']
            # and find a match 'f(' in the string 'if('
            if currpos+candidate_pos-1 >= 0:
                if not isNameToken(source[currpos+candidate_pos-1]):
                    findposlist.append(candidate_pos)
                    # remove so that findpos except clause doesn't get confused
                    findposlist_candidates.remove(candidate_pos)
            else:
                # no earlier character in source, so must be OK
                findposlist.append(candidate_pos)
                # remove so that findpos except clause doesn't get confused
                findposlist_candidates.remove(candidate_pos)
        try:
            findpos = min([x for x in findposlist if x >= 0]) + currpos
        except ValueError:
            # findposlist is empty
            done = True
            # commented this stuff out from before this function only ever
            # dealt with a singleton callfns list - probably the source of
            # the original bug!
##            if currpos < len(source) and len(findposlist_candidates) > 0 and \
##                       findposlist_candidates[0] >= 0:
##                currpos += findposlist_candidates[0] + 2
##                output += source[:findposlist_candidates[0]+2]
##                continue
##            else:
##                done = True
        if not done:
            # find start and end braces
            startbrace = source[findpos:].find('(')+findpos
            endbrace = findEndBrace(source[startbrace:])+startbrace
            sub_source = source[startbrace+1:endbrace]
            embedded_calls = [sub_source.find(fname+'(') for fname in callfns]
            try:
                subpositions = [x for x in embedded_calls if x > 0]
                if subpositions == []:
                    filtered_sub_source = sub_source
                else:
                    filtered_sub_source = addArgToCalls(sub_source,callfns,arg,notFirst)
            except ValueError:
                pass
            # add unchanged part to output
            # insert arg before end brace
            if currpos==0 and callfns == [notFirst]:
                notFirst = ''
                addStr = ''
            else:
                if filtered_sub_source == '':
                    addStr = arg
                else:
                    addStr = ', ' + arg
            output += source[currpos:startbrace+1] + filtered_sub_source \
                    + addStr + ')'
            currpos = endbrace+1
        else:
            output += source[currpos:]
    return output


def proper_match(specstr, term):
    """Determine whether string argument 'term' appears one or more times in
    string argument 'specstr' as a proper symbol (not just as part of a longer
    symbol string or a number).
    """
    ix = 0
    term_len = len(term)
    while ix < len(specstr) and term_len > 0:
        found_ix = specstr[ix:].find(term)
        pos = found_ix + ix
        if found_ix > -1:
            try:
                if specstr[pos + term_len] not in [')', '+', '-', '/', '*', ' ',
                                          ']', ',', '<', '>', '=', '&', '^']:
                    # then term continues with additional name characters:
                    # no match after all
                    ix = pos + term_len
                    continue
            except IndexError:
                # no other chars remaining, so doesn't matter
                pass
            if isNumericToken(term):
                # have to be careful that we don't replace part
                # of another number, e.g. origterm == '0'
                # with repterm == 'abc', and the numeric literal '130'
                # appears in specstr. The danger is that we'd end up
                # with new specstr = '13abc'
                if specstr[pos-1] in num_chars + ['.', 'e'] or \
                   specstr[pos + term_len] in num_chars + ['.', 'e']:
                    ix = pos + term_len
                    continue
            return True
        else:
            break
    return False


def remove_indices_from_range(ixs, max_ix):
    """From the indices 0:max_ix+1, remove the individual
    index values in ixs.
    Returns the remaining ranges of indices and singletons.
    """
    ranges = []
    i0 = 0
    for ix in ixs:
        i1 = ix - 1
        if i1 < i0:
            i0 = ix + 1
        elif i1 == i0:
            ranges.append([i0])
            i0 = ix + 1
        else:
            ranges.append([i0,i1+1])
            i0 = ix + 1
    if i0 < max_ix:
        ranges.append([i0, max_ix+1])
    elif i0 == max_ix:
        ranges.append([i0])
    return ranges
