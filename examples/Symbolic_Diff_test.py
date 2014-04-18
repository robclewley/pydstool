"""Tests on symbolic differentiation using ModelSpec Quantity objects.

    Robert Clewley, September 2005.
"""
from __future__ import print_function

from PyDSTool import *
from numpy.linalg import norm
from numpy.testing.utils import assert_approx_equal

# testing direct string -> string diff with complex pattern of negative signs
assert DiffStr('x-(4*x*y)/(1+x*x)','x') == \
       '1-4*(y)/(1+x*x)+(4*x*y)*2*x*pow((1+x*x),-2)'

print("Showing the variety of ways that symbolic Diff() can be used:")

f1 = '[-3*x**2+2*(x+y),-y/2]'
f2 = ['-3*x**2+2*(x+y)','-y/2']
x=Var('x')
y=Var('y')
f3 = [-3*Pow(x,2)+2*(x+y),-y/2]
f4 = ['-3*x**2.+2*(x+y)','-y/2.']
xx = QuantSpec('dummy','x')

print("The following strings should all be identical")
print(Diff(f1,'x'))
print(Diff(f2,'x'))
print(Diff(f3,'x'))
print(Diff(f1,x))
print(Diff(f2,x))
print(Diff(f3,x))
print(Diff(f3,xx))
print("\n")
print(Diff(f1, ['x','y']))
print(Diff(f1, [x,y]))
print(Diff(f1, [xx,y]))
print(Diff(f2, ['x','y']))
print(Diff(f2, [x,y]))
print(Diff(f3, ['x','y']))
print(Diff(f3, [x,y]))

print("--------------------------\n")

print("Now some more complex tests...")
t=Var('t')
s=Var('s')

assert str(Diff(Pow((t*5),2),t)) != '0'

p=Par('3.','p')
f = Fun(QuantSpec('f', str(2.0+s-10*(t**2)+Exp(p))), ['s','t'])
f_0=Fun(QuantSpec('f_0', str(Diff(f(s,t),t))), ['t'])
print(2*f.eval(s=3,t=t))
print(Diff('-10*Pow(t,2)','t'))
print(Diff(2*f.eval(s=3,t=t), t))
print(Diff(3+t*f.eval(s=3,t=t),t))
print(Diff(3+t*f(s,t),t).eval(s=3,t=1,p=p))
print(Diff(3+t*f(s,t),t).eval(s=3,t=1,p=p()))
assert Diff(str(f(s,t)),'t') == Diff(f(s,t),t)
q1=Diff(f(s,t),t)
q2=Diff(str(f(s,t)),t)
assert q1 == q2
q1.difference(q2)

print("\n")
print(Diff(f(t,s),t))
print(Diff(2*f(3,t*5), t))
assert str(Diff(2*f(3,t*5), t)) != str(0)
assert f(s,t) != f(t,s)

print(f(s,t).eval())
q=f(s,t)
print(q.eval())

print(Diff('g(s)',s))
print(Diff('g(s)',s).eval())
dg_dt=Fun(QuantSpec('g_0', '2-Sin(t/2)'),['t'])
assert str(Diff('g(t)',t).eval()) != 'g_0(t)'
print("\n\n")
print(Diff('g(s)',s))
print(Diff('g(s)',s).eval())


g=Fun('',[t],'g') # declare empty function
assert str(g(t)) == 'g(t)'
print(Diff(g(s),s).eval())

assert eval(str(Diff('pow(1,2)*t','t'))) == 1
assert eval(str(Diff(Pow(1,2)*t,t))) == 1
assert str(Diff(Sin(Pow(t,1)),t)) == 'Cos(t)'

q=QuantSpec('q','-0+3+pow(g(x)*h(y,x),1)*1')
print(Diff(q,'x'))
assert str(Diff(q,'x')) == '(g_0(x)*h(y,x)+g(x)*h_1(y,x))'
# BROKEN in this version (need to move to SymPy)
#print Diff(q,'x').eval()
#assert str(Diff(q,'x').eval()) == '(2-Sin(x/2))*h(y,x)+g(x)*h_1(y,x)'

p0=Var('p0')
p1=Var('p1')

pv=Var([p0,p1], 'p')
print(pv())
print(pv.eval())

u=Var('Pi/(2*Sin(Pi*t/2))','u')
assert u.eval(t=1).tonumeric() == pi/2

# Old tests for symbolic strings only (but via the Diff function)
x='x'

test1 = ['x','a*x','a','a(x)','a(x)*x','a(x)*b(x)',
            'a+x','a(x)+x','a(x)+b(x)+c(x)','a(x)*b(x)*c(x)',
            'a(x)*(b(x)+c(x))','(a(x)+b(x))*c(x)',
            'a(x)/b(x)','(a(x))**b(x)',
            'x**n','x**5','x**-5','sin(x)','cos(x)','exp(x)',
            'ln(x)','log(x)','log10(x)','asin(x)','sinh(x)']
sols1 = ['1', 'a', '0', 'a_0(x)', 'a_0(x)*x+a(x)',
         'a_0(x)*b(x)+a(x)*b_0(x)', '1', 'a_0(x)+1',
         'a_0(x)+b_0(x)+c_0(x)',
         'a_0(x)*b(x)*c(x)+a(x)*(b_0(x)*c(x)+b(x)*c_0(x))',
         'a_0(x)*(b(x)+c(x))+a(x)*(b_0(x)+c_0(x))',
         '(a_0(x)+b_0(x))*c(x)+(a(x)+b(x))*c_0(x)',
         'a_0(x)/b(x)-a(x)*b_0(x)*Pow(b(x),-2)',
##         '(Log(a(x))*b_0(x)+a_0(x)*b(x)/a(x))*Pow(a(x),b(x))',
         '(Log(a(x))*b_0(x)+a_0(x)*b(x)/(a(x)))*Pow(a(x),b(x))',
         'n*Pow(x,n-1)', '5*Pow(x,4)', '-5*Pow(x,-6)',
         'Cos(x)', '-Sin(x)', 'Exp(x)',
##         '1/x', '1/x', 'Log(10)/x', '1/Sqrt(1-Pow(x,2))',
         '1.0/x', '1.0/x', 'Log(10)/x', '1.0/Sqrt(1-Pow(x,2))',
         'Cosh(x)']
for i, s in enumerate(test1):
    dstr=str(Diff(s,x))
    if dstr != sols1[i]:
        raise 'Test %d: Diff(%s,%s) = %s but should be %s'%(i,s,x,dstr,sols1[i])

str1='x**-1'
sols2 = ['-Pow(x,-2)', '2*Pow(x,-3)']
for i in range(2):
    dstr=str(Diff(str1,'x'))
    if dstr != sols2[i]:
        raise 'Diff(%s,%s) = %s but should be %s'%(str1,'x',dstr,sols2[i])
    str1=dstr

str2='pow(x,-1.5)'
dstr2=str(Diff(str2,'x'))
if dstr2 != '-1.5*Pow(x,-2.5)':
    raise "Diff(%s,%s)=%s but should be %s"%(str2,'x',dstr2, '-1.5*Pow(x,-2.5)')

#------------------
### CACHE TEST (unused)
# loadDiffs('temp')
### Do some tests here
# print __Diff_saved
# saveDiffs('temp')
#------------------


print("\nSymbolic vector tests")

q0=Var(p0+3,'q0')
q1=Var(Diff(1+Sin(Pow(p0,3)+q0),p0),'q1')

qv=Var([q0,q1], 'q')
print(qv())
print(qv.eval())

v=Var('v')
w=Var('w')
f=Var([-3*Pow((2*v+1),3)+2*(w+v),-w/2], 'f')

df = Diff(f, [v,w])
print(df)
dfe = df.eval(v=3,w=10).tonumeric()
print(dfe)
assert isinstance(dfe, ndarray)
assert isinstance(df.fromvector(), list)

y0=Var('y0')
y1=Var('y1')
y2=Var('y2')
t=Var('t')

ydot0=Fun(-0.04*y0 + 1e4*y1*y2, [y0, y1, y2], 'ydot0')
ydot2=Fun(3e7*y1*y1, [y0, y1, y2], 'ydot2')
ydot1=Fun(-ydot0(y0,y1,y2)-ydot2(y0,y1,y2), [y0, y1, y2], 'ydot1')

F = Fun([ydot0(y0,y1,y2),ydot1(y0,y1,y2),ydot2(y0,y1,y2)], [y0,y1,y2], 'F')
assert F.dim == 3
DF = Diff(F,[y0,y1,y2])
DF0,DF1,DF2 = DF.fromvector()
assert_approx_equal(DF0.fromvector()[0].tonumeric(), -0.04)
# str(Diff(F,[y0,y1,y2])) should be (to within numerical rounding errors):
#'[[-0.04,10000*y2,10000*y1],[0.040000000000000001,(-10000*y2)-30000000*2*y1,-10000*y1],[0,30000000*2*y1,0]]')

jac=Fun(Diff(F,[y0,y1,y2]), [t, y0, y1, y2], 'Jacobian')
assert jac(t, 0.1,y0+1,0.5).eval(y0=0) == jac(t, 0.1,1+y0,0.5).eval(y0=0)
assert jac(t, 0.1,y0,0.5) == jac(t, 0.1,0+y0,0.5)

x=Var('x')
y=Var('y')

f1 = Fun([-3*x**3+2*(x+y),-y/2], [x,y], 'f1')
f2 = ['-3*x**3+2*(x+y)','-y/2']
f3 = [-3*x**3.+2*(x+y),-y/2.]
print("\n\nVector-valued function f(x,y) =", f1)
print("The function string can be passed to Diff in various ways...")
print(str(f1))
print(str(f2))
print(str(f3))
print("\nThe following outputs are for Diff(f,'x') for each of these forms")
print("They should all be the same (except for some may contain decimal points)")
f4 = [-3*Pow((2*x+1),3)+2*(x+y),-y/2]
xx = QuantSpec('dummy','x')
f5=Var([-3*Pow((2*x+1),3)+2*(x+y),-y/2], 'f5')

assert Diff(f1,x) == Diff(f1,'x')
print(Diff(f1,x))
print(Diff(f3,x))
print(Diff(f3,xx))
print(Diff(f4,x))
print(Diff(f4,xx))
print("\nExamples of Jacobian Diff(f, [x,y])...")
assert Diff(f1, [x,y]) == Diff(f1, ['x','y']) == Diff(f1(x,y), [x,y])
print(Diff(f2, ['x','y']))
print(Diff(f3, ['x','y']))
print(Diff(f1, [xx,y]))
print(Diff(f1, [xx,'y']))
print(Diff(f2, [x,y]))
print(Diff(f3, [x,y]), "\n")
print(Diff(f4, [x,y]))
df5 = Diff(f5, [x,y])
print(df5)
print(df5.eval(x=3,y=10).tonumeric())
print(df5.eval(x=3,y=10).fromvector(0))
print(df5.fromvector(0))
assert isinstance(df5.fromvector(), list)
a = df5.fromvector(0).eval(x=3,y=10).tonumeric()
b = df5.eval(x=3,y=10).tonumeric()[0]
assert a[0]==b[0] and a[1]==b[1]


# --
print("\nExamples of differentiation using nested functions")
print(" - this functionality is built in to Symbolic.prepJacobian")
func_ma_spec = (['p', 'v'], '0.32*(v+54)/(1-exp(-p*(v+54)/4))')
ma = Fun(func_ma_spec[1], func_ma_spec[0], 'ma')
ma_1 = Fun(Diff(ma, 'v'), ['v'], 'ma_1')
func_mb_spec = (['v'], '0.28*(v+27)/(exp((v+27)/5)-1)')
mb = Fun(func_mb_spec[1], func_ma_spec[0], 'mb')
mb_0 = Fun(Diff(mb, 'v'), ['v'], 'mb_0')
func_ma_1_spec = (['p', 'v'], str(ma_1.spec))
func_mb_0_spec = (['v'], str(mb_0.spec))

# artificial example to introduce time dependence
rhs_m = 'exp(-2*t)*(100-v) + ma(1, v)*(1-m)-mb(v)*m'
jac_part = Diff(rhs_m, ['v', 'm'])

f = expr2fun(jac_part, ma_1=func_ma_1_spec, mb_0=func_mb_0_spec,
             ma=func_ma_spec, mb=func_mb_spec)
assert remain(f._args, ['t','v','m']) == []
assert abs(norm(f(0, -50, 0.2)) - 8.565615) < 1e-6
# alternative usage syntax
assert abs(norm(f(**{'t': 0, 'v': -50, 'm': 0.2})) - 8.565615) < 1e-6
