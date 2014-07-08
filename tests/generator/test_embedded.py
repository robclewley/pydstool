#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import mock
import pytest

from PyDSTool import (
    PyDSTool_KeyError,
)
from PyDSTool.Generator import (
    EmbeddedSysGen,
)


def test_mandatory_specfn_key():
    model = mock.MagicMock()
    model.query.return_value = {}
    with pytest.raises(PyDSTool_KeyError):
        EmbeddedSysGen({'name': 'test', 'system': model})
