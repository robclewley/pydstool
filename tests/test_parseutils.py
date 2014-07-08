#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyDSTool.parseUtils import (
    addArgToCalls,
    proper_match,
    wrapArgInCall,
)


def test_wrapArgInCall():
    """Test: wrapping delimiters around call arguments"""

    fs = 'initcond(v,p)'
    assert 'initcond("v",p)' == wrapArgInCall(fs, 'initcond', '"')
    assert 'initcond(v,"p")' == wrapArgInCall(fs, 'initcond', '"', argnums=[1])
    assert 'initcond([v],[p])' == wrapArgInCall(fs, 'initcond', '[', ']', [0, 1])


def test_combo_wrapArgInCall_and_addArgToCalls():
    """Test combo of addArgToCalls and wrapArgInCall with embedded calls"""

    fs = "1/zeta(y_rel(y,initcond(y)+initcond(z)),z-initcond(x))+zeta(0.)"
    fs_p = addArgToCalls(fs, ['zeta', 'y_rel', 'initcond', 'nothing'], "p")
    assert '1/zeta(y_rel(y,initcond(y, p)+initcond(z, p), p),z-initcond(x, p), p)+zeta(0., p)' == fs_p
    assert '1/zeta(y_rel(y,initcond("y", p)+initcond("z", p), p),z-initcond("x", p), p)+zeta(0., p)' == wrapArgInCall(fs_p, 'initcond', '"')


def test_proper_match():
    s = '1 +abc13 + abc'
    assert proper_match(s, 'abc')
    assert not proper_match(s[:10], 'abc')
