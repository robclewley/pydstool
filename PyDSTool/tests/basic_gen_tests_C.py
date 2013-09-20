from PyDSTool import *

print "Basic generator tests for C-based vector fields..."


fnspecs = {'testif': (['x'], 'if(x<0.0,0.0,x)'),
          'testmin': (['x', 'y'], 'min(x,y)'),
          'testmax': (['x', 'y'], 'max(x,y)'),
          'testmin2': (['x', 'y'], '1/(2+min(1+(x*3),y))+y'),
          'indexfunc': (['x'], 'pi*x')
          }


DSargs = args(name='test',
              pars={'p0': 0, 'p1': 1, 'p2': 2},
              varspecs={'z[j]': 'for(j, 0, 1, 2*z[j+1] + p[j])',
                   'z2': '-z0 + p2 + 1'},
              fnspecs=fnspecs
              )
tmm = Generator.Dopri_ODEsystem(DSargs)

# test user interface to aux functions and different combinations of embedded
# macros

assert tmm.auxfns.testif(1.0) == 1.0
assert tmm.auxfns.testmin(1.0, 2.0) == 1.0
assert tmm.auxfns.testmax(1.0, 2.0) == 2.0
assert tmm.auxfns.testmin2(1.0, 2.0) == 2.25
assert tmm.Rhs(0, {'z0': 0.5, 'z1': 0.2, 'z2': 2.1})[1] == 5.2

DSargs2 = args(name='test2',
              pars={'p0': 0, 'p1': 1, 'p2': 2},
              varspecs={'y[i]': 'for(i, 0, 1, y[i]+if(y[i+1]<2, 2+p[i]+getbound("y2",0), indexfunc([i])+y[i+1]) - 3)',
                        'y2': '0'
                        },
              xdomain={'y2': [-10,10]},
              fnspecs=fnspecs,
              ics={'y0': 0, 'y1': 0, 'y2': 0.1}
              )
tm2 = Generator.Dopri_ODEsystem(DSargs2)
tm2.set(tdata=[0,10])
traj = tm2.compute('test')
assert allclose(tm2.Rhs(0, {'y0':0, 'y1': 0.3, 'y2': 5}), array([-11. ,  2.3+pi,  0. ]))

print "  ...passed"
