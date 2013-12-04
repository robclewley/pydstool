#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def clean_files(files):
    for f in files:
        os.remove(f) if os.path.exists(f) else None
