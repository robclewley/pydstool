"""
Tests on Pointset labelling class: PointInfo
 Robert Clewley, November 2005
"""

from PyDSTool import *
from PyDSTool.parseUtils import symbolMapClass

print "Tests on Pointset labelling class: PointInfo"

sm = symbolMapClass({'a': 'A'})

p = PointInfo()
p[3] = ('a', {'bif': 'sn'})
p[5] = ('a', {'bif': 'h'})
p['a'] = (5, {'bif': 'h'})
p[1] = ('b', {'bif': 'c'})

assert p['a'].keys() == [3,5]
sorted = p.sortByIndex()
assert [s[0] for s in sorted] == [1,3,5]
assert p.getIndices() == [1,3,5]

p.update(3, 'c', {'foo': 'bar'})
p.update(3, 'h')

assert p[3].keys() == ['a','h','c']

try:
    p[-1]
except IndexError:
    # "Successfully could not access an index out of range
    pass
else:
    raise AssertionError

try:
    p[-1] = "wrong"
except IndexError:
    # Successfully could not add an index out of range
    pass
else:
    raise AssertionError

# For an index that does not exist, get an empty label back
assert p[50] == {}

try:
    # index 10 not present!
    p.remove('a', 3, 10)
except KeyError:
    # Successfully could not remove index label for non-existent entry
    pass
else:
    raise AssertionError

del p['b']
try:
    del p['not_there']
except KeyError:
    # Successfully could not delete index label for non-existent entry
    pass
else:
    raise AssertionError

# remove all indices associated with label 'a'
p.remove('a')
assert len(p) == 1, "p ended up the wrong size!"

p[0] = ['a', ('k', {'bif': 'H', 'otherinfo': 'hello'})]
# Mapping 'a' labels to 'A' using mapNames(<symbolMapClass instance>) method of PointInfo
p.mapNames(sm)
assert p.getLabels() == ['A', 'c', 'h', 'k']
