#ifndef __RADAU5MOD__
#define __RADAU5MOD__

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
#include "radau5.h"
#include "vfield.h"
#include "events.h"
#include "integration.h"
#include "eventFinding.h"
#include "interface.h"

/* Call solout with dense output on every iteration */
#define DENSE_OUTPUT_CALL 1

/* Vector error tolerances = True */
#define VECTOR_ERR_TOL 1

/* Default value for machine epsilon; defined in float.h */
#define UROUND DBL_EPSILON

/* Stiffness test (disabled) */
#define TEST_STIFF -1

void radau_adjust_h(double* t, double* h);

void radau_solout(int *nr, double *xold, double *x, double *y, double *cont,
		  int *lrc, int *n, double *rpar, int *ipar, int *irtrn);

double contsolfun(unsigned k, double x);

void refine(int n, double xold, double x);

void vfield(int *n, double *t, double *x, double *f, double *rpar, int *ipar);

void vfieldjac(int *n, double *t, double *x, double *df, int *ldf, 
	       double *rpar, int *ipar);

void vfieldmas(int *n, double *am, int *lmas, double *rpar, int *ipar, double *t, double *x);

PyObject* Integrate(double *ic, double t, double hinit, double hmax, 
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


#endif
