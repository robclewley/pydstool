## Exceptions

__all__ = ['PyDSTool_Error', 'PyDSTool_BoundsError', 'PyDSTool_KeyError',
           'PyDSTool_UncertainValueError', 'PyDSTool_TypeError',
           'PyDSTool_ExistError', 'PyDSTool_AttributeError',
           'PyDSTool_ValueError', 'PyDSTool_UndefinedError',
           'PyDSTool_InitError', 'PyDSTool_ClearError',
           'PyDSTool_ContError']


class PyDSTool_Error(Exception):
    def __init__(self, value=None):
        self.value = value
        self.code = None
    def __str__(self):
        return repr(self.value)
    def __repr__(self):
        return repr(self.value)

class PyDSTool_UncertainValueError(PyDSTool_Error):
    def __init__(self, value, varval=None):
        if varval is None:
            valstr = ''
        else:
            valstr = ' at variable = '+str(varval)
        self.varval = varval
        PyDSTool_Error.__init__(self, value+valstr)

class PyDSTool_BoundsError(PyDSTool_Error):
    pass

class PyDSTool_KeyError(PyDSTool_Error):
    pass

class PyDSTool_ValueError(PyDSTool_Error):
    pass

class PyDSTool_TypeError(PyDSTool_Error):
    pass

class PyDSTool_ExistError(PyDSTool_Error):
    pass

class PyDSTool_UndefinedError(PyDSTool_Error):
    pass

class PyDSTool_AttributeError(PyDSTool_Error):
    pass

class PyDSTool_InitError(PyDSTool_Error):
    pass

class PyDSTool_ClearError(PyDSTool_Error):
    pass

class PyDSTool_ContError(PyDSTool_Error):
    pass