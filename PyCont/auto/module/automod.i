%module auto

%include typemaps.i
%include carrays.i

%array_functions(double,doubleArray)
%array_functions(int, intArray)

%{
#include <numpy/libnumarray.h>
%}

%init %{
import_libnumarray();
%}

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

%typemap(freearg) double* {
	if($1) free($1);
}

%typemap(freearg) int* {
    if($1) free($1);
}

extern PyObject* Compute(void);

extern PyObject* Initialize(void);

extern PyObject* SetData(int ips, int ilp, int isw, int isp, int sjac, int sflow, int nsm, int nmx, int ndim,
                         int ntst, int ncol, int iad, double epsl, double epsu, double epss, int itmx,
                         int itnw, double ds, double dsmin, double dsmax, int npr, int iid,
                         int nicp, int *icp, int nuzr, int *iuz, double *vuz);

extern PyObject* SetInitPoint(double *u, int npar, int *ipar, double *par, int *icp, int nups,
                              double *ups, double *udotps, double *rldot, int adaptcycle);

extern PyObject* Reset(void);

extern PyObject* ClearParams(void);

extern PyObject* ClearSolution(void);

extern PyObject* ClearSpecialPoints(void);

extern PyObject* ClearAll(void);
