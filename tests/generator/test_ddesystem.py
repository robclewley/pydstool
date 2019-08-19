#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest
from PyDSTool.Generator.DDEsystem import DDEsystem


def test_ddesystem_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        DDEsystem({'name': 'dummy'})
