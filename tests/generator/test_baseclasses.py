#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import pytest

from PyDSTool import (
    PyDSTool_KeyError,
)
from PyDSTool.Generator.baseclasses import (
    ixmap,
)


def test_ixmap_raises_exception(mocker):
    gen = mocker.MagicMock()
    gen.pars = gen.inputs = {}
    m = ixmap(gen)
    with pytest.raises(PyDSTool_KeyError):
        print(m[0])
