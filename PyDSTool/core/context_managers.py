# -*- coding: utf-8 -*-

"""Context managers implemented for (mostly) internal use"""

import contextlib
import functools
from io import UnsupportedOperation
import os
import sys


__all__ = ['RedirectStdout', 'RedirectStderr']


@contextlib.contextmanager
def _stdchannel_redirected(stdchannel, dest_filename, mode='w'):
    """
    A context manager to temporarily redirect stdout or stderr

    Originally by Marc Abramowitz, 2013
    (http://marc-abramowitz.com/archives/2013/07/19/python-context-manager-for-redirected-stdout-and-stderr/)
    """

    oldstdchannel = None
    dest_file = None
    try:
        if stdchannel is None:
            yield iter([None])
        else:
            oldstdchannel = os.dup(stdchannel.fileno())
            dest_file = open(dest_filename, mode)
            os.dup2(dest_file.fileno(), stdchannel.fileno())
            yield
    except (UnsupportedOperation, AttributeError):
        yield iter([None])
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


RedirectStdout = functools.partial(_stdchannel_redirected, sys.stdout)
RedirectStderr = functools.partial(_stdchannel_redirected, sys.stderr)
RedirectNoOp = functools.partial(_stdchannel_redirected, None, '')
