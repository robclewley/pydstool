#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyDSTool.Toolbox import event_driven_simulator as eds


def test_smoke():
    assert not eds.__doc__
