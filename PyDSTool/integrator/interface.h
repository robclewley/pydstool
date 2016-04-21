#ifndef __INTERFACE__
#define __INTERFACE__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h> 
#include "memory.h"
#include <string.h>
#include <assert.h>
#include <Python.h>

#include <numpy/arrayobject.h>
#include "integration.h"
#include "vfield.h"


PyObject* InitBasic(int PhaseDim, int ParamDim, int nAux, int nEvents, int nExtInputs, 	     
		    int HasJac, int HasJacP, int HasMass, int extraSize);

PyObject* CleanUp( void );  

PyObject* InitInteg(int Maxpts, double *atol, double *rtol );

PyObject* ClearInteg( void );

PyObject* InitEvents( int Maxevtpts, int *EventActive, int *EventDir, int *EventTerm,
		      double *EventInterval, double *EventDelay, double *EventTol,
		      int *Maxbisect, double EventNearCoef);

PyObject* ClearEvents( void );

PyObject* InitExtInputs( int nExtInputs, int *extInputLens, double *extInputVals, 
			 double *extInputTimes);

PyObject* ClearExtInputs( void );

PyObject* SetRunParameters(double *ic, double *pars, double gt0, double t0, 
			   double tend, int refine, int specTimeLen, double *specTimes, 
			   double *upperBounds, double *lowerBounds);

PyObject* ClearParams( void );

PyObject* Reset( void );

PyObject* SetContParameters(double tend, double *pars, double *upperBounds, double *lowerBounds);

PyObject* Vfield(double t, double *x, double *p);

PyObject* Jacobian(double t, double *x, double *p);

PyObject* JacobianP(double t, double *x, double *p);

PyObject* AuxFunc(double t, double *x, double *p);

PyObject* MassMatrix(double t, double *x, double *p);

#if (PY_VERSION_HEX < 0x03000000)
void _init_numpy(void);
#else
int _init_numpy(void);
#endif


#endif
