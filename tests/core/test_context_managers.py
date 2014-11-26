# -*- coding: utf-8 -*-

import sys
from tempfile import mkstemp

import pytest

from PyDSTool.core.context_managers import (
    RedirectStderr,
    RedirectStdout,
    RedirectNoOp
)


@pytest.fixture
def tmpfile():
    _, fname = mkstemp()
    return fname


def test_stdout_redirecting(tmpfile):
    msg = 'Hello!'
    with RedirectStdout(tmpfile):
        print(msg)

    with open(tmpfile) as f:
        assert f.read() == msg + '\n'


def test_stderr_redirecting(tmpfile):
    msg = 'Hello!'
    with RedirectStderr(tmpfile):
        sys.stderr.write(msg)

    with open(tmpfile) as f:
        assert f.read() == msg


def test_redirecting_both_streams(tmpfile):
    msg = 'Hello!'
    errmsg = 'Warning!'
    with RedirectStdout(tmpfile), RedirectStderr(tmpfile, mode='a'):
        print(msg)
        sys.stderr.write(errmsg)

    with open(tmpfile) as f:
        lines = f.readlines()
        assert msg + '\n' in lines
        assert errmsg in lines


def test_no_redirecting(capsys):
    msg = 'Hello!'
    errmsg = 'Error!'
    with RedirectNoOp():
        print(msg)
        sys.stderr.write(errmsg)

    out, err = capsys.readouterr()
    assert out == msg + '\n'
    assert err == errmsg
