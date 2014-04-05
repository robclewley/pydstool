#ifndef __INTERFACE__
#define __INTERFACE__

#include "auto_f2c.h"
#include "auto_c.h"
#include <Python.h>
#include "autovfield.h"
#include <numpy/arrayobject.h>

#define SUCCESS 1
#define FAILURE 0

// Auto routines
int func(integer ndim, const doublereal *u, const integer *icp,
         const doublereal *par, integer ijac,
         doublereal *f, doublereal *dfdu, doublereal *dfdp);
int stpnt(integer ndim, doublereal t, doublereal *u, doublereal *par);
int bcnd(integer ndim, const doublereal *par, const integer *icp,
         integer nbc, const doublereal *u0, const doublereal *u1, integer ijac,
         doublereal *fb, doublereal *dbc);
int icnd(integer ndim, const doublereal *par, const integer *icp,
         integer nint, const doublereal *u, const doublereal *uold,
         const doublereal *udot, const doublereal *upold, integer ijac,
         doublereal *fi, doublereal *dint);
int fopt(integer ndim, const doublereal *u, const integer *icp,
         const doublereal *par, integer ijac,
         doublereal *fs, doublereal *dfdu, doublereal *dfdp);
int pvls(integer ndim, const void *u, doublereal *par);

#endif