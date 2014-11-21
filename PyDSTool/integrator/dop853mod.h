#ifndef __DOP853MOD__
#define __DOP853MOD__

#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <stdlib.h> 
#include "memory.h"
#include <string.h>
#include <assert.h>
#include "Python.h"
#include <numpy/arrayobject.h>
#include "dop853.h"
#include "vfield.h"
#include "events.h"
#include "integration.h"
#include "eventFinding.h"

/* Call solout with dense output on every iteration */
#define DENSE_OUTPUT_CALL 2

/* Vector error tolerances = True */
#define VECTOR_ERR_TOL 1

/* Default value for machine epsilon; defined in float.h */
#define UROUND DBL_EPSILON

/* Stiffness test (disabled) */
#define TEST_STIFF -1

void dopri_adjust_h(double t, double* h);

void dopri_solout(long nr, double xold, double x, double* y, 
		  unsigned n, int* irtrn);

double contsolfun(unsigned k, double x);

void refine(int n, double xold, double x);

void vfield(unsigned n, double x, double *y, double *p, double *f);

void vfieldjac(int *n, double *t, double *x, double *df, int *ldf, 
	       double *rpar, int *ipar);

void vfieldmas(int *n, double *am, int *lmas, double *rpar, int *ipar, double *t, double *x);

PyObject* Integrate(double *ic, double t, double hinit, 
		    double hmax, double safety, 
		    double fac1, double fac2, double beta, 
		    int verbose, int calcAux, int calcSpecTimes, 
		    int checkBounds, int boundCheckMaxSteps, double *magBound);

#endif
