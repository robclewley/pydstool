%module auto

%{
    #define SWIG_FILE_WITH_INIT
    #include "automod.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%include typemaps.i
%include carrays.i

%typemap(in) int * {
	int i, n;
	if((!PyList_Check($input)) && ($input != Py_None)) {
		PyErr_SetString(PyExc_ValueError,"Expected a list or None as input");
		return NULL;
	}
    if($input == Py_None) {
        $1 = NULL;
    } else {
        n = PyList_Size($input);
        $1 = (int *) malloc(n*sizeof(int));
        for( i = 0; i < n; i++ ) {
            PyObject *o = PyList_GetItem($input,i);
            if(PyNumber_Check(o)) {
                $1[i] = PyInt_AsLong(o);
            }
            else {
                PyErr_SetString(PyExc_ValueError,"List elements must be numbers.");
                return NULL;
            }
        }
    }
}

%typemap(in) double * {
	int i, n;
	if((!PyList_Check($input)) && ($input != Py_None)) {
		PyErr_SetString(PyExc_ValueError,"Expected a list or None as input");
		return NULL;
	}
    if($input == Py_None) {
        $1 = NULL;
    } else {
        n = PyList_Size($input);
        $1 = (double *) malloc(n*sizeof(double));
        for( i = 0; i < n; i++ ) {
            PyObject *o = PyList_GetItem($input,i);
            if(PyNumber_Check(o)) {
                $1[i] = PyFloat_AsDouble(o);
            }
            else {
                PyErr_SetString(PyExc_ValueError,"List elements must be numbers.");
                return NULL;
            }
        }
    }
}

%typemap(out) int {
	$result = Py_BuildValue("(i)", $1);
}

%typemap(freearg) double* {
	if($1) free($1);
}

%typemap(freearg) int* {
    if($1) free($1);
}

%apply (int* INPLACE_ARRAY1, int DIM1) {(int* A, int nd1)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* A, int nd1, int nd2)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* A, int nd1)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* A, int nd1, int nd2)}
%apply (double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(double* A, int nd1, int nd2, int nd3)}

%include "automod.h"