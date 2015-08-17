#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import numpy

from PyDSTool.common import (
    concatStrDict,
    filteredDict,
    invertMap,
    isUniqueSeq,
    makeSeqUnique,
)

from PyDSTool.Points import Point


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


def test_filteredDict_works_as_expected():
    d = {'x': 1, 'y': 2, 'z': 3}
    assert d == filteredDict(d, d.keys())
    assert {'x': 1} == filteredDict(d, ['x'])
    assert {'x': 1, 'y': 2} == filteredDict(d, ['x', 'y'])


def test_filteredDict_negative_works_as_expected():
    d = {'x': 1, 'y': 2, 'z': 3}
    assert {} == filteredDict(d, d.keys(), neg=True)
    assert {'y': 2, 'z': 3} == filteredDict(d, ['x'], neg=True)
    assert {'z': 3} == filteredDict(d, ['x', 'y'], neg=True)


def test_filteredDict_works_as_expected_when_key_is_absent():
    d = {'x': 1, 'y': 2, 'z': 3}
    assert {} == filteredDict(d, ['w'])
    assert d == filteredDict(d, ['w'], neg=True)


def test_filteredDict_works_as_expected_for_Point():
    p = Point({'x': 1, 'y': 2, 'z': 3})
    assert {} == filteredDict(p, ['w'])
    assert {'x': 1, 'y': 2} == filteredDict(p, ['x', 'y'])
    assert {'z': 3} == filteredDict(p, ['x', 'y'], neg=True)


def test_concatStrDict_for_empty_dict():
    assert '' == concatStrDict({})


def test_concatStrDict_without_order():
    d = {'a': ['a'] * 3, 'b': ['b'] * 2, 'c': ['c\n']}
    s = concatStrDict(d)
    assert 'aaa' in s
    assert 'bb' in s
    assert 'c\n' in s


def test_concatStrDict_with_order():
    d = {'a': ['a'] * 3, 'b': ['b'] * 2, 'c': ['c\n']}
    order = ['a', 'b', 'c']
    assert 'aaabbc\n' == concatStrDict(d, order)
    assert 'c\nbbaaa' == concatStrDict(d, reversed(order))
