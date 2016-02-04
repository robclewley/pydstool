#ifndef __EVENTS__
#define __EVENTS__

typedef double (*EvFunType)(unsigned n, double x, double *y, double *p, 
			    unsigned wkn, double *wk, unsigned xvn, double *xv);

typedef double (*ContSolFunType)(unsigned k, double x);

int CompareEvents(const void *element1, const void *element2); 

void assignEvents( EvFunType *events );

#endif


