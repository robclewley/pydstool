"""
Test object deletion, and corresponding deletion of dynamically created
 class methods that are no longer referenced.
"""

from PyDSTool import *

print("Test object deletion and deletion of dynamically created class methods")

var1 = Variable(Pointset(coordarray = array(list(range(10)), float)*0.1,
                        indepvararray = array(list(range(10)), float)*0.5
                        ), name='v1')

del var1
try:
    print(var1)
except NameError:
    print("   ...passed")
else:
    raise AssertionError("Variable deletion unsuccessful")
