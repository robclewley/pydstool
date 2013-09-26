"""
Tests on Pointset labelling class: PointInfo
 Robert Clewley, November 2005
"""

from PyDSTool.Points import PointInfo
from PyDSTool.parseUtils import symbolMapClass

import pytest


@pytest.fixture()
def pi():
    p = PointInfo()
    p[3] = ('a', {'bif': 'sn'})
    p[5] = ('a', {'bif': 'h'})
    p['a'] = (7, {'bif': 'h'})
    p[1] = ('b', {'bif': 'c'})
    return p


def test_access_by_label(pi):
    assert pi['a'].keys() == [3, 5, 7]
    assert pi['b'].keys() == [1]
    assert pi[3].keys() == ['a']


def test_sorting_by_index(pi):
    assert [s[0] for s in pi.sortByIndex()] == [1, 3, 5, 7]
    assert pi.getIndices() == [1, 3, 5, 7]


def test_updating(pi):
    pi.update(3, 'c', {'foo': 'bar'})
    pi.update(3, 'h')
    assert pi[3].keys() == ['a', 'h', 'c']


def test_access_out_of_range(pi):
    with pytest.raises(IndexError):
        pi[-1]


def test_add_index_out_of_range(pi):
    with pytest.raises(IndexError):
        pi[-1] = "wrong"


def test_nonexistent_index(pi):
    """For an index that does not exist, get an empty label back"""
    assert pi[50] == {}


def test_removing_nonexistent_index(pi):
    with pytest.raises(KeyError):
        pi.remove('a', 3, 10)


def test_removing_index(pi):
    # remove all indices associated with label 'a'
    pi.remove('a')
    assert len(pi) == 1, "p ended up the wrong size!"


def test_mapping(pi):
    sm = symbolMapClass({'a': 'A'})
    pi[0] = ['a', ('k', {'bif': 'H', 'otherinfo': 'hello'})]
    pi.update(3, 'c', {'foo': 'bar'})
    pi.update(3, 'h')
    # Mapping 'a' labels to 'A' using mapNames(<symbolMapClass instance>) method of PointInfo
    pi.mapNames(sm)
    assert pi.getLabels() == ['A', 'b', 'c', 'h', 'k']
