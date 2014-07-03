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


def test_point_info_creating(pi):

    # default
    p1 = PointInfo()
    assert p1.getIndices() == []
    assert p1.getLabels() == []

    # from another PointInfo
    p2 = PointInfo(pi)
    assert p2.getIndices() == pi.getIndices()
    assert p2.getLabels() == pi.getLabels()

    # from dict with string as values
    p3 = PointInfo({
        1: 'a',
        2: 'b',
    })
    assert p3.getIndices() == [1, 2]
    assert p3.getLabels() == ['a', 'b']

    # from dict with dict as values
    p4 = PointInfo({
        1: {'a': 0},
        2: {'b': 1}
    })
    assert p4.getIndices() == [1, 2]
    assert p4.getLabels() == ['a', 'b']

    # index must be int value
    with pytest.raises(TypeError):
        PointInfo({
            1.0: 'a',
            2.0: 'b',
        })

    # creating from array of tuples is illegal
    with pytest.raises(TypeError):
        PointInfo([(1.0, 'a'), (2.0, 'b')])


def test_access(pi):

    # by label
    assert list(pi['a'].keys()) == [3, 5, 7]
    assert list(pi['b'].keys()) == [1]

    # by index
    assert list(pi[3].keys()) == ['a']

    # by list
    assert pi[[1, 3]] == PointInfo({
        1: {'b': {'bif': 'c'}},
        3: {'a': {'bif': 'sn'}}
    })
    assert pi[['a', 'b']] == PointInfo({
        1: {'b': {'bif': 'c'}},
        3: {'a': {'bif': 'sn'}},
        5: {'a': {'bif': 'h'}},
        7: {'a': {'bif': 'h'}},
    })

    # by mixed 'int' and 'str' list is illegal
    with pytest.raises(TypeError):
        pi[['a', 1]]

    # by tuple is illegal
    with pytest.raises(TypeError):
        pi[(3, 'a')]

    # by list which is not 'all ints' or 'all strings' is illegal
    with pytest.raises(TypeError):
        assert pi[[1.0, 2.0]]


def test_access_empty_pointinfo():
    assert PointInfo()[1] == {}
    assert PointInfo()[[1, 3]] == PointInfo()
    assert PointInfo()[slice(3)] == PointInfo()
    assert PointInfo()[['a', 'b']] == PointInfo()


def test_sorting(pi):
    # by index
    assert [s[0] for s in pi.sortByIndex()] == [1, 3, 5, 7]
    assert pi.getIndices() == [1, 3, 5, 7]

    # by label
    assert [s[0] for s in pi.sortByLabel()] == ['a', 'b']
    assert pi.getLabels() == ['a', 'b']


def test_updating(pi):
    pi.update(3, 'c', {'foo': 'bar'})
    pi.update(3, 'h')
    assert all(k in pi[3].keys() for k in ['a', 'h', 'c'])


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
