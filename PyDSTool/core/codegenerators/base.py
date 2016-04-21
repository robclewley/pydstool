#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from PyDSTool.common import idfn, invertMap, remain
from PyDSTool.parseUtils import proper_match, convertPowers


class CodeGenerator(object):

    def __init__(self, fspec, **kwargs):
        self.fspec = fspec
        self.opts = self._set_options(**kwargs)

    def _set_options(self, **kwargs):

        try:
            opts = {
                'start': kwargs.pop('codeinsert_start', '').strip(),
                'end': kwargs.pop('codeinsert_end', '').strip(),
                'define': kwargs.pop('define', ''),
                'power_sign': kwargs.pop('power_sign', '**'),
            }
        except AttributeError:
            raise ValueError('code insert must be a string')

        if list(kwargs.keys()):
            raise KeyError("CodeGenerator: keywords %r unsupported" % list(kwargs.keys()))

        return opts

    def _format_code(self, code, before='', after='', sep='\n'):
        return sep.join([s for s in [before, code, after] if s]) if code else ''

    def generate_aux(self):
        raise NotImplementedError

    def generate_spec(self):
        raise NotImplementedError

    def generate_special(self, name, spec):
        raise NotImplementedError
    def defineMany(self, names, listid, start=0):
        return ''.join([self.define(n, listid, i + start) for i, n in enumerate(names)])

    def define(self, name, listid, index):
        return self.opts['define'].format(name, listid, index)

    def _normalize_spec(self, spec):
        if any([pt in spec for pt in ('pow', '**', '^')]):
            spec = convertPowers(spec, self.opts['power_sign'])
        spec = self._process_builtins(spec)
        return spec

    def _process_builtins(self, spec):
        return spec

def _processReused(specnames, specdict, reuseterms, indentstr='',
                   typestr='', endstatementchar='', parseFunc=idfn):
    """Process substitutions of reused terms."""

    seenrepterms = []  # for new protected names (global to all spec names)
    reused = {}.fromkeys(specnames)
    reuseterms_inv = invertMap(reuseterms)
    # establish order for reusable terms, in case of inter-dependencies
    are_dependent = []
    deps = {}
    for origterm, rterm in reuseterms.items():
        for _, rt in reuseterms.items():
            if proper_match(origterm, rt):
                if rterm not in are_dependent:
                    are_dependent.append(rterm)
                try:
                    deps[rterm].append(rt)
                except KeyError:
                    # new list
                    deps[rterm] = [rt]
    order = remain(sorted(reuseterms.values()), are_dependent) + are_dependent
    for specname in specnames:
        reused[specname] = []
        specstr = specdict[specname]
        repeatkeys = []
        for origterm, repterm in reuseterms.items():
            # only add definitions if string found
            if proper_match(specstr, origterm):
                specstr = specstr.replace(origterm, repterm)
                if repterm not in seenrepterms:
                    reused[specname].append([indentstr,
                                             typestr + ' ' *
                                             (len(typestr) > 0),
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
                                                 typestr + ' ' *
                                                 (len(typestr) > 0),
                                                 repterm, " = ",
                                                 parseFunc(origterm),
                                                 endstatementchar, "\n"])
        # if replacement terms have already been used in the specifications
        # and there are no occurrences of the terms meant to be replaced then
        # just log the definitions that will be needed without replacing
        # any strings.
        if reused[specname] == [] and len(reuseterms) > 0:
            for origterm, repterm in reuseterms.items():
                # add definition if *replacement* string found in specs
                if proper_match(specstr, repterm) and repterm not in seenrepterms:
                    reused[specname].append([indentstr,
                                             typestr + ' ' *
                                             (len(typestr) > 0),
                                             repterm, " = ",
                                             parseFunc(origterm),
                                             endstatementchar, "\n"])
                    seenrepterms.append(repterm)
        specdict[specname] = specstr
        # add any dependencies for repeated terms to those that will get
        # defined when functions are instantiated
        for r in seenrepterms:
            if r in are_dependent:
                for repterm in deps[r]:
                    if repterm not in seenrepterms:
                        reused[specname].append([indentstr,
                                                 typestr + ' ' *
                                                 (len(typestr) > 0),
                                                 repterm, " = ",
                                                 parseFunc(
                                                     reuseterms_inv[repterm]),
                                                 endstatementchar, "\n"])
                        seenrepterms.append(repterm)
    # reuseterms may be automatically provided for a range of definitions
    # that may or may not contain instances, and it's too inefficient to
    # check in advance, so we'll not cause an error here if none show up.
    # if len(seenrepterms) == 0 and len(reuseterms) > 0:
    #     print("Reuse terms expected:%r" % reuseterms)
    #     info(specdict)
    #     raise RuntimeError("Declared reusable term definitions did not match"
    #                        " any occurrences in the specifications")
    return (reused, specdict, seenrepterms, order)
