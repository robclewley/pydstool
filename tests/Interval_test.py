from PyDSTool import Interval, uncertain, contained, notcontained, PyDSTool_UncertainValueError, PyDSTool_TypeError
from numpy import array, Inf

print '---------- Test for Interval.py ----------'
a=Interval('a', float, [-1,1], abseps=1e-5)
b=Interval('b', float, [-1.,1.])
c=Interval('c', float, [-3.,-2.])
assert b.contains(a) == uncertain

assert -2 < a
assert a > -2
assert a < 1 + 2*a._abseps
assert not (a < 1 + 0.5*a._abseps)
assert 1 + 2*a._abseps > a
assert not (1 + 0.5*a._abseps > a)
assert c < b
assert c < a
assert a > c
assert ([-5, 0, -1] < a) == [True, False, False]
assert (a> array([-5, -4, -1])) == [True, True, False]

print "m=Interval('test1', float, (0,1))"
m=Interval('test1', float, (0,1))
print 'm =>   ', m
print 'm() =>   ', m()
print 'm.info() =>  ',
m.info()
print '0.1 in m =>   ', 0.1 in m
print
print "n=Interval('test2', float, (0,0.4))"
n=Interval('test2', float, (0,0.4))
try:
    print 'n in m =>  ', n in m, '\n'
    raise RuntimeError("Failed test")
except PyDSTool_UncertainValueError, e:
    print 'UncertainValueError: ', e
print "s=Interval('a_singleton', float, 0.4)"
s=Interval('a_singleton', float, 0.4)
print 's.get() =>  ', s.get()
print "s in m =>  ", s in m, '  (works for singleton interval s)\n'
r=Interval('another_singleton', float, 0.0)
print "r=Interval('another_singleton', float, 0.0)"
b=Interval('b', int, 0)
print "b=Interval('b', int, 0)"
try:
    print "b in m =>  ", b in m
    raise RuntimeError("Failed test")
except PyDSTool_UncertainValueError, e:
    print 'UncertainValueError: ', e
i=Interval('i', int, (0,1))
print "i=Interval('i', int, (0,1))"
print "b in i =>  ", b in i, " (true because we're comparing integer intervals)"
ii=i+3
assert ii[0] == i[0]+3
assert ii[1] == i[1]+3
iii=4*i
assert iii[0] == 4*i[0]
assert iii[1] == 4*i[1]
iiii=2-i
assert iiii[0] == 2-i[1]
assert iiii[1] == 2-i[0]
iiiii_1=1/i
assert iiiii_1[0] == 1
assert iiiii_1[1] == Inf
iiiii_2=-1/i
assert iiiii_2[0] == -Inf
assert iiiii_2[1] == -1
assert i.contains(1) is contained   # because discrete-valued interval

print "\nUse the explicit `contains` method to avoid exceptions, and instead"
print "   get a NumIntMembership type returned..."
print "m.contains(i) =>  ", m.contains(i)
print "m.contains(0.4) => ", m.contains(0.4)

j = Interval('test3', float, (0,0.999999999))
print "j = Interval('test3', float, (0,0.999999999))"
print "p = m.contains(j)"
p = m.contains(j)
print "p is uncertain => ", p is uncertain

print "\nBut don't try to compare NumIntMembership objects to booleans..."
print "q = m.contains(0.9)"
q = m.contains(0.9)
assert q is contained
assert not(q is True)
print "q is True => ", q is True, " (false because q is not a boolean type)"
print "... but can use in a statement such as 'if m.contains(0.9): ...etc.'"

print "\nElementary `interval logic` can be performed when checking endpoints"
print "   for interval containment."
print "contained and notcontained => ", contained and notcontained
print "contained and uncertain => ", contained and uncertain
print "notcontained and notcontained => ", notcontained and notcontained

print "\nm.uniformSample(0.09, strict=False, avoidendpoints=True) => ", \
      m.uniformSample(0.09, strict=False, avoidendpoints=True)

print "\nm.uniformSample(0.09, strict=False) => ", \
      m.uniformSample(0.09, strict=False)

print "i2=Interval('i2', int, (0,10))"
i2=Interval('i2', int, (0,10))
print "\ni2.uniformSample(2, strict=False, avoidendpoints=True) => ", \
      i2.uniformSample(2, strict=False, avoidendpoints=True)

print "i3=Interval('i3', float, (0.,0.4))"
i3=Interval('i3', float, (0.,0.4))
print "\ni3.uniformSample(0.36, strict=False) => ", \
      i3.uniformSample(0.36, strict=False)

print "\ni3.uniformSample(0.36, strict=False, avoidendpoints=True) => ", \
      i3.uniformSample(0.36, strict=False, avoidendpoints=True)

print "\ni3.uniformSample(0.36, strict=True) => ", \
      i3.uniformSample(0.36, strict=True)


print "\nInfinite intervals"
print "inf1 = Interval('inf1', float, [0,Inf])"
inf1 = Interval('inf1', float, [0,Inf], abseps=0)
print "0 in inf1 => ", 0 in inf1
print "inf1.contains(inf1) => ", inf1.contains(inf1)
print "inf2 = Interval('inf2', float, [-Inf,Inf])"
inf2 = Interval('inf2', float, [-Inf,Inf])
print "inf2.contains(inf2) => ", inf2.contains(inf2)
print "inf2.contains(inf1) => ", inf2.contains(inf1)
print "inf3 = Interval('inf3', float, [-Inf,0])"
inf3 = Interval('inf3', float, [-Inf,0])
print "inf3.contains(inf2) => ", inf3.contains(inf2)
inf_int = Interval('inf3', int, [-Inf,0])
print "inf_int = Interval('inf3', int, [-Inf,0])"
print "inf_int.contains(inf3) => "
try:
    inf_int.contains(inf3)
except PyDSTool_TypeError, e:
    print " ",e
print "inf_int.contains(-Inf) => ", inf_int.contains(-Inf)
assert inf_int.contains(-Inf)
i4 = Interval('i4', int, [-5,5])
print "i4 = Interval('i4', int, [-5,5])"
print "inf_int.intersect(i4) => ", inf_int.intersect(i4).get()
print "Intersection should fail on mixed-type intervals ..."
try:
    result1 = inf3.intersect(i4).get()
except:
    result1 = " >> FAILURE"
try:
    result2 = j.intersect(i2).get()
except:
    result2 = " >> FAILURE"
print "inf3.intersect(i4) => ", result1
assert result1 == " >> FAILURE"
print "j.intersect(i2) => ", result2
assert result2 == " >> FAILURE"
i5 = Interval('i5', int, [4,4])
assert i5.issingleton
i5._abseps = 0
assert 4 in i5
assert 4.0 in i5
i5._abseps = 1e-5
assert 4 in i5
assert 4.0 in i5

print "Tests passed"