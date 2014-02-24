#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import pytest
from PyDSTool.Generator.DDEsystem import DDEsystem


def test_ddesystem_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        DDEsystem({'name': 'dummy'})
