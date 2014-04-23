#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import os


def clean_files(vf_names):
    for v in vf_names:
        for f in _generate_filenames(v):
            os.remove(f) if os.path.exists(f) else None


def _generate_filenames(vf_name):

    parts = (
        ('%s_', '_vf.py'),
        ('%s_', '_vf.pyc'),
        ('_%s_', '_vf.so'),
        ('_%s_', '_vf.cpython-33m.so'),
        ('_%s_', '_vf.cpython-34m.so'),
    )
    for g in ['radau5', 'dop853']:
        for prefix, suffix in parts:
            yield ''.join([prefix % g, vf_name, suffix])
