#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import numpy

from PyDSTool.common import (
    isUniqueSeq,
    makeSeqUnique,
    invertMap,
)


def test_isUniqueSeq_returns_true_for_empty_list():
    assert isUniqueSeq([])


def test_isUniqueSeq_returns_true_for_list_with_unique_items():
    assert isUniqueSeq(['1', '2', '3'])


def test_isUniqueSeq_returns_false_for_list_with_duplicated_items():
    assert not isUniqueSeq(['1', '2', '1'])


def test_makeSeqUnique_returns_list_with_unique_items():
    assert isUniqueSeq(makeSeqUnique(['1', '2', '1']))


def test_makeSeqUnique_returns_array_when_asarray_set():
    assert isinstance(makeSeqUnique([1, 2, 1], asarray=True), numpy.ndarray)


def test_makeSeqUnique_returns_array_with_unique_items():
    assert isUniqueSeq(makeSeqUnique([1, 2, 1], asarray=True))


def test_makeSeqUnique_preserves_order():
    assert makeSeqUnique('AAAABBBCCDAABBB') == ['A', 'B', 'C', 'D']


def test_invertMap_for_dict():
    assert {'x': 'x', 'y': 'y'} == invertMap({'x': ['x'], 'y': ['y']})
