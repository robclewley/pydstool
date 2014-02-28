#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from . import base
from . import python
from . import c
from . import matlab


def getCodeGenerator(fspec, lang_=None):
    lang = lang_ or fspec.targetlang
    if lang == 'python':
        return python.Python(fspec)
    elif lang == 'c':
        return c.C(fspec)
    elif lang == 'matlab':
        return matlab.Matlab(fspec)
    else:
        return base.CodeGenerator(fspec)


_processReused = base._processReused
