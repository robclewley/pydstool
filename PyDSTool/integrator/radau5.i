%module radau5

%include typemaps.i
%include carrays.i

%array_functions(double,doubleArray)
%array_functions(int, intArray)

%{
#include "interface.h"
#include "radau5mod.h"
%}

%typemap(in) double * {
	int i, n;
	if(!PyList_Check($input)) {
		PyErr_SetString(PyExc_ValueError,"Expected a list as input");
		return NULL;
	}
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

%typemap(in) int * {
	int i, n;
	if(!PyList_Check($input)) {
		PyErr_SetString(PyExc_ValueError,"Expected a list as input");
		return NULL;
	}
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





%typemap(freearg) double* {
	if($1) free($1);
}



extern PyObject* Integrate(double *ic, double t, double hinit, double hmax,
		    double safety,
		    double jacRecompute, double newtonStop,
		    double stepChangeLB, double stepChangeUB,
		    double stepSizeLB, double stepSizeUB,
		    int hessenberg,  int maxNewton,
		    int newtonStart, int index1dim, int index2dim,
		    int index3dim, int stepSizeStrategy,
		    int DAEstructureM1, int DAEstructureM2,
		    int useJac, int useMass, int verbose,
		    int calcAux, int calcSpecTimes);

extern PyObject* InitBasic(int PhaseDim, int ParamDim, int nAux, int nEvents, int nExtInputs,
		    int HasJac, int HasJacP, int HasMass, int extraSize);

extern PyObject* CleanUp( void );

extern PyObject* InitInteg(int Maxpts, double *atol, double *rtol );

extern PyObject* ClearInteg( void );

extern PyObject* InitEvents( int Maxevtpts, int *EventActive, int *EventDir, int *EventTerm,
		      double *EventInterval, double *EventDelay, double *EventTol,
		      int *Maxbisect, double EventNearCoef);

extern PyObject* ClearEvents( void );

extern PyObject* InitExtInputs( int nExtInputs, int *extInputLens, double *extInputVals,
			 double *extInputTimes);

extern PyObject* ClearExtInputs( void );

extern PyObject* SetRunParameters(double *ic, double *pars, double gt0, double t0,
			   	double tend, int refine, int specTimeLen, double *specTimes,
				double *upperBounds, double *lowerBounds );

extern PyObject* ClearParams( void );

extern PyObject* Reset( void );

extern PyObject* SetContParameters(double tend, double *pars, double *upperBounds, double *lowerBounds);

extern PyObject* Vfield(double t, double *x, double *p);

extern PyObject* Jacobian(double t, double *x, double *p);

extern PyObject* JacobianP(double t, double *x, double *p);

extern PyObject* AuxFunc(double t, double *x, double *p);

extern PyObject* MassMatrix(double t, double *x, double *p);
