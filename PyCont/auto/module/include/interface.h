#ifndef __INTERFACE__
#define __INTERFACE__

#include "auto_f2c.h"
#include "auto_c.h"
#include <Python.h>
#include <numpy/libnumarray.h>

PyObject* Initialize(void);

PyObject* SetData(int ips, int ilp, int isw, int isp, int sjac, int sflow, int nsm, int nmx, int ndim, 
                  int ntst, int ncol, int iad, double epsl, double epsu, double epss, int itmx,
                  int itnw, double ds, double dsmin, double dsmax, int npr, int iid,
                  int nicp, int *icp, int nuzr, int *iuz, double *vuz);
                      
PyObject* SetInitPoint(double *u, int npar, int *ipar, double *par, int *icp, int nups,
                       double *ups, double *udotps, double *rldot, int adaptcycle);

PyObject* Reset(void);

PyObject* ClearParams(void);

PyObject* ClearSolution(void);

PyObject* ClearSpecialPoints(void);

PyObject* ClearAll(void);

#endif