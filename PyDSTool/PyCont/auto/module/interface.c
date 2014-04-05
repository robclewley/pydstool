#include "interface.h"

AutoData *gIData = NULL;

/**************************************
 **************************************
 *    REQUIRED AUTO FUNCTIONS
 **************************************
 **************************************/

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int func(integer ndim, const doublereal *u, const integer *icp,
         const doublereal *par, integer ijac,
         doublereal *f, doublereal *dfdu, doublereal *dfdp) {

  /* Jacobian and Parameter Jacobian loops clearly slow things down here. */
  
  integer i;
  doublereal **jac;
  doublereal **jacp;
  
  vfieldfunc(ndim, 0, 0, u, par, f, 0, NULL, 0, NULL);
  
  // Jacobian
  jac = (doublereal **)MALLOC(ndim*sizeof(doublereal *));
  jac[0] = dfdu;
  for (i=1; i<ndim; i++) jac[i] = jac[i-1] + ndim;
  jacobian(ndim, 0, 0, u, par, jac, 0, NULL, 0, NULL);
  FREE(jac);
  
  // Parameter Jacobian
  jacp = (doublereal **)MALLOC(gIData->npar*sizeof(doublereal *));
  jacp[0] = dfdp;
  for (i=1; i<gIData->npar; i++) {
      if (i != 10)
          jacp[i] = jacp[i-1] + ndim;
      else
          jacp[i] = jacp[i-1] + 41*ndim;    // Index = 50
  }
  jacobianParam(ndim, 0, 0, u, par, jacp, 0, NULL, 0, NULL);
  FREE(jacp);
  
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int stpnt(integer ndim, doublereal t, doublereal *u, doublereal *par) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int bcnd(integer ndim, const doublereal *par, const integer *icp,
         integer nbc, const doublereal *u0, const doublereal *u1, integer ijac,
         doublereal *fb, doublereal *dbc) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int icnd(integer ndim, const doublereal *par, const integer *icp,
         integer nint, const doublereal *u, const doublereal *uold,
         const doublereal *udot, const doublereal *upold, integer ijac,
         doublereal *fi, doublereal *dint) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int fopt(integer ndim, const doublereal *u, const integer *icp,
         const doublereal *par, integer ijac,
         doublereal *fs, doublereal *dfdu, doublereal *dfdp) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int pvls(integer ndim, const void *u, doublereal *par) {
    u = (doublereal *)u;
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

