"""
Tests and demonstration of Point, Pointset, and their label classes
"""
from PyDSTool import *

# POINTS
print "\n****** Point class test ******\n"
print "x uses Python float type:"
xstr = """x = Point({'coorddict': {'x0': [1.123456789], 'x1': [-0.4],
               'x2': [4000]},
           'coordtype': float})"""
print xstr
exec(xstr)
# float is equivalent to float64
print "x => ", repr(x)
print "x.toarray() = ", x.toarray()
print "\nprint x => ", x
print "x.dimension => ", x.dimension, ", x.coordnames => ", x.coordnames
print "x.coordtype => ", x.coordtype
print "x.coordtype => ", x.coordtype
print "x('x1') = ", x('x1')
print "x(['x1','x0']) = ", x(['x1','x0'])
print "x([0,1]) = ", x([0,1])
print "\nChanging x entries is done by x[index] = value:"
print "x[1] = -0.45"
x[1] = -0.45
print "\nThe index can also be a name, a list of names, or even a dictionary:"
print "x[['x0', 'x1']] = [4.11103, -0.56])"
x[['x0', 'x1']] = [4.11103, -0.56]
print "\ny is a 1D point (with integer type):"
# can also specify as array([4])
ystr = """y = Point({'y': 4})"""
print ystr
exec(ystr)
print "print y => ", y
print "y(0) = ", y(0)
print "type(y(0)) => ", type(y(0))
print "y([0]) = ", y([0])
print "y.toarray() = ", y.toarray()
assert comparePointCoords(x,(x+0)*1,fussy=True)



### POINTSETS

print "\n\n****** Pointset test ******\n"
print "v is a 'singleton' pointset, to make sure this doesn't break the interface"
vstr = """v = Pointset({'coorddict': {'x0': 0.2, 'x1': -1.2},
             'indepvardict': {'t': 0.01},
             'coordtype': float64,
             'indepvartype': float64
              })"""
print vstr
exec(vstr)
print "print v =>", v
print "\nprint v(0.01) => ", v(0.01)
print "and v(0.01) is a Point object\n"
print "print v(0.01, 0) => ", v(0.01, 0)
print "and v(0.01, 0) is a float\n"
print "print v(0.01, 'x0') => ", v(0.01, 'x0')

print "\nk tests deprecated syntax for single-point pointset"
kstr = """k = Pointset({'coordarray': array(0.1),
              'coordnames': 'k0',
              'indepvarname': 't',
              'indepvararray': array(0.0)})"""
print kstr
exec(kstr)
assert k.dimension == 1
print "print k.toarray() => ", k.toarray()
print "print k['t'] => ", k['t']
print "print k(0.0) => ", k(0.0)
print "print k => ", k

print "\nu tests non-parameterized pointset"
ustr = """u = Pointset({'coordarray': array([10., 20., 30., 40.])})"""
exec(ustr)
print ustr
print "u.toarray() => ", u.toarray()
print "isparameterized(u) => ", isparameterized(u)
print "print u => ", u

print "\nw tests alternative declaration syntax, and other forms of calling"
wstr = """wp = Pointset({'coordarray': array([[4.456, 2.34634, 7.3431, 5.443],
                              [-10.0336, -5.2235, -3.23221, -0.01],
                              [3e5, 3.1e5, 3.3e5, 2.8e5]], float64),
              'coordnames': ['x0', 'x1', 'x2'],
              'indepvarname': 't',
              'indepvararray': array([0.0, 1.0, 2.0, 3.0], float64)})"""
print wstr
exec(wstr)
assert type(wp.coordarray)==type(array([1,2],float64))
print "wp.dimension => ", wp.dimension
print "print wp(0.0) => ", wp(0.0)
print "type(wp(0.0)) => ", type(wp(0.0))
print "print wp(1.0)(0) => ", wp(1.0)(0)
print "print wp(2.0, 'x1') => ", wp(2.0, 'x1')
print "\nprint wp(2.0, ['x2', 'x1']) => ", wp(2.0, ['x2', 'x1'])
print "type(wp(2.0, ['x1', 'x2'])) => ", type(wp(2.0, ['x1', 'x2']))
print "print wp[['x1','x0']] => ", wp[['x1','x0']]
print "\nwp.info(1) =>", wp.info(1)
print "wp(1.0).info(1) =>", wp(1.0).info(1)
print "wp['t'] => ", wp['t']
print "\nCall several 't' values at once (explicit values only -- no ellipses):"
print "wp([1., 2.]) => ", wp([1., 2.])
print "\nExtract a coordinate (only by name) as a regular array:"
w_x0 = wp['x0']
print "w_x0 = wp['x0']  => ", w_x0

print "\nExtract a point of w as a regular array:"
w_at_1 = wp(1.).toarray()
print "w_at_1 = wp(1.).toarray()  => ", w_at_1

print "\nMany forms to access individual values or sub-arrays:"
print "wp(1., 'x1') => ", wp(1., 'x1')
print "wp(1.)('x1') => ", wp(1.)('x1')
print "wp(1., 1)) => ", wp(1., 1)
print "wp([1.,3.], 1) => ", wp([1.,3.], 1)
print "wp([1.,3.])('x1') => ", wp([1.,3.])['x1']
print "wp(1.)([0,1]) => ", wp(1.)([0,1])
print "but ... wp([1.])(1., [0,1]) => ", wp([1.])(1., [0,1])
print "... because wp([1.]) is a Pointset and wp(1.) is a Point"
print "This is why wp(1.).toarray() shows a different array shape to wp([1.]).toarray():"
print "wp(1.).toarray().shape => ", wp(1.).toarray().shape
print "wp([1.]).toarray().shape => ", wp([1.]).toarray().shape

print "\nChange a point in w using wp[indepvar_value] = point:"
print "Old value at t=1.0:  wp(1.0) =>", wp(1.0)
print "wp[1] = x"
wp[1] = x
print "w has now been updated for the meshpoint at t=1.0  =>"
print "New value at t=1.0:  wp(1.0) => ", wp(1.0)
assert type(wp.coordarray)==type(array([1,2],float64))

print "\nWe can test equality between arrays, as usual:"
print "w_at_1 != wp(1.).toarray() => ", w_at_1 != wp(1.).toarray()
print "We can also compare with a Pointset object:"
print "wp(1.) != w_at_1 => ", wp(1.) != w_at_1
print "But we can't put an array on the left-hand side if a Point or " \
      "Pointset is on the right."

print "\nTo demonstrate appending a Point and Pointset to another Pointset:"
vw = Pointset({'coorddict': {'x0': [0.1, 0.15], 'x1': [100., 102], 'x2': [0.2, 0.1]},
             'indepvardict': {'t': [4.5, 5.0]},
             'coordtype': float64,
             'indepvartype': float64,
             'labels': {1: 'c'}
              })
print "vw.labels -->", vw.labels
print "wp.append(vw)"
wp.append(vw)
print "wp.labels -->", wp.labels
assert type(wp.coordarray)==type(array([1,2],float64))
wp.append(Point({'coorddict': {'t': 6.5, 'x0': 2, 'x1': -300, 'x2': -0.9997}}))
assert type(wp.coordarray)==type(array([1,2],float64))
print "\nwp.toarray() -->\n", wp.toarray()
print "\nwp(4.5) -->\n", wp(4.5)
print "\nwp[[3,6]] -->", wp[[3,6]]
print "\nwp[3:5] -->", wp[3:5]
print "\nwp[2:] -->", wp[2:]
try:
    # index out of range
    wp[10:]
except ValueError:
    pass
print "\nwp[wp.findIndex(4.5)] -->\n", wp[wp.findIndex(4.5)]
print "\nwp.labels -->", wp.labels
print "\nLabels test:"
wp.labels[3] = ('a', {'bif':'SN'})
print "wp.labels[3] -->", wp.labels[3]
wp_part = wp[3:5]
print "wp_part.labels -->", wp_part.labels
assert wp_part.labels[0] == wp.labels[3]
wpt = wp(3.)
assert wpt.labels == {'a': {'bif':'SN'}}
wp_ins = Pointset({'coorddict': {'x0': [-2.1, -4., -5., -4.5], 'x1': [50., 51., 52., 54.], 'x2': [0.01, 0.02, 0.4, 0.9]},
             'indepvardict': {'t': [1.5, 5.2, 9., 10.]},
             'coordtype': float64,
             'indepvartype': float64,
             'labels': {2: 'b', 3: {'a': {'bif':'H'}}}
              })
print "\nwp_ins object created to insert into wp:"
print wp_ins
wp.insert(wp_ins)
print "\nwp.insert(wp_ins) -->\n", wp

print "\nTo demonstrate building a Pointset from a list of Point objects:"
codestr = """
pointlist = []
for t in wp['t']:
    pointlist.append(wp(t))
w_reconstructed = pointsToPointset(pointlist, 't', wp['t'])
"""
print codestr
exec(codestr)
print "\nAnd to demonstrate that this yields an identical object:"
print "w_reconstructed == w  => ", w_reconstructed == wp

try:
    w_double = w_reconstructed.append(w_reconstructed)
    raise RuntimeError("Internal error with Pointset class!")
except ValueError:
    print "(ensure that any independent variable values to append are well-ordered)"

print "\nTest of non-parameterized use of pointsToPointset:"
wnp = pointsToPointset(pointlist)
print "(Adding two additional labels to wnp)"
wnp.labels[0]=('b', {})
wnp.addlabel(4, 'c', {'bif': 'H'})  # preferred syntax
print wnp
print "\nwnp[:] -->\n", wnp[:]
print "-- OK!"

print "\nCan iterate over points and pointsets:"
print "for p in wnp.bylabel('a'):\n  print p\n"
for p in wnp.bylabel('a'):
    print p


wp2 = Pointset({'coorddict': {'x0': [-4.5, 2, 3], 'x1': [54, 62, 64], 'x2': [0.9, 0.8, 0.2]},
             'indepvardict': {'t': [10, 11, 12]},
             'coordtype': float64,
             'indepvartype': float64,
             'labels': {0: {'a_different': {'bif':'H'}},
                        2: 'd'}
              })
wp.append(wp2, skipMatchingIndepvar=True)
assert len(wp) == 13
assert wp.bylabel('b')['t'][0] == 9.0
assert all(wp.bylabel('a')['t'] == array([3., 10.]))
assert wp.bylabel('d')['t'][0] == 12.0
assert all(wp.bylabel('a_different')['t'] == array([10.]))
z = wp[-5:]
assert z.labels.getIndices() == [1,2,4]

# -----------------------------

print "\n"
print "x (point) and wp, wnp (param'd and non-param'd pointsets) are available in the global namespace,", \
      "to play with interactively now that this script has run."

