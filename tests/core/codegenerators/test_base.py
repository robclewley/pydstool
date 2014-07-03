#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyDSTool.core.codegenerators.base import CodeGenerator


def test_define():
    m = CodeGenerator(None, define='{0}: {1}({2})\n')
    assert 'Q: p(1)\n' == m.define('Q', 'p', 1)


def test_define_many_for_empty_list():
    m = CodeGenerator(None)
    assert '' == m.defineMany([], 'v', 1)


def test_defineMany_for_nonempty_list():
    m = CodeGenerator(None, define='{0}->{1}_[{2}]\n')
    assert [
        'x->x_[1]',
        'y->x_[2]',
        '',
    ] == m.defineMany(['x', 'y'], 'x', 1).split('\n')
