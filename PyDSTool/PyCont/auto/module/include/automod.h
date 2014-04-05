#ifndef __AUTOMOD__
#define __AUTOMOD__

#include "Python.h"
#include "auto_f2c.h"
#include "auto_c.h"
#include <numpy/arrayobject.h>

int Initialize(void);

int SetData(int ips, int ilp, int isw, int isp, int sjac, int sflow, int nsm, int nmx, int ndim, 
                  int ntst, int ncol, int iad, double epsl, double epsu, double epss, int itmx,
                  int itnw, double ds, double dsmin, double dsmax, int npr, int iid,
                  int nicp, int *icp, int nuzr, int *iuz, double *vuz);
                      
int SetInitPoint(double *u, int npar, int *ipar, double *par, int *icp, int nups,
                       double *ups, double *udotps, double *rldot, int adaptcycle);

int Compute(void);

int ClearAll(void);

int getSolutionNum(void);

void getSolutionVar(double *A, int nd1, int nd2, int nd3);

void getSolutionPar(double *A, int nd1, int nd2);

void getFloquetMultipliers(double *A, int nd1, int nd2, int nd3);

void getJacobians(double *A, int nd1, int nd2, int nd3);

void getNumIters(int *A, int nd1, int nd2);

int getSpecPtNum(void);

void getSpecPtDims(int i, int *A, int nd1);

void getSpecPtFlags(int i, int *A, int nd1);

void getSpecPt_ups(int i, double *A, int nd1, int nd2);

void getSpecPt_udotps(int i, double *A, int nd1, int nd2);

void getSpecPt_rldot(int i, double *A, int nd1);

void getSpecPt_flow1(int i, double *A, int nd1, int nd2, int nd3);

void getSpecPt_flow2(int i, double *A, int nd1, int nd2, int nd3);

/* int Reset(void);

int ClearParams(void);

int ClearSolution(void);

int ClearSpecialPoints(void); */

#endif
