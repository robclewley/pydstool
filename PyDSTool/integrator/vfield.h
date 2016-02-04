#ifndef __VFIELD__
#define __VFIELD__

void vfieldfunc(unsigned n, unsigned np, double t, double *Y, double *p, double *f,
		unsigned wkn, double *wk, unsigned xvn, double *xv);

void jacobian(unsigned n, unsigned np, double t, double *Y, double *p, double **f,
	      unsigned wkn, double *wk, unsigned xvn, double *xv);

void jacobianParam(unsigned n, unsigned np, double t, double *Y, double *p, double **f,
		   unsigned wkn, double *wk, unsigned xvn, double *xv);

void auxvars(unsigned n, unsigned np, double t, double *Y, double *p, double *g, 
	     unsigned wkn, double *wk, unsigned xvn, double *xv);

void massMatrix(unsigned n, unsigned np, double t, double *Y, double *p, double **f,
		unsigned wkn, double *wk, unsigned xvn, double *xv);

#endif

