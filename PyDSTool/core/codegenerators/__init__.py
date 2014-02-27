#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from . import base
from . import python
from . import c
from . import matlab

def getCodeGenerator(lang):
    if lang == 'python':
        return python.PythonCodeGenerator()
    elif lang == 'c':
        return c.CCodeGenerator()
    elif lang == 'matlab':
        return matlab.MatlabCodeGenerator()
    else:
        return base.CodeGenerator()


_processReused = base._processReused
