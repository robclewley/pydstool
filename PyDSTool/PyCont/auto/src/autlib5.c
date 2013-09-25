/* autlib5.f -- translated by f2c (version 19970805).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "auto_f2c.h"
#include "auto_c.h"

/* All of these global structures correspond to common
   blocks in the original code.  They are ONLY used within
   the Homcont code.
*/
struct {
  integer itwist, istart, iequib, nfixed, npsi, nunstab, nstab, nrev;
} blhom_1;

struct {
  integer *ipsi, *ifixed, *irev;
} blhmp_1 = {NULL,NULL,NULL};

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*        Subroutines for Homoclinic Bifurcation Analysis                  */
/*       (A. R. Champneys, Yu. A. Kuznetsov, B. Sandstede,                 */
/*        B. E. Oldeman)                                                   */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */

/*     ---------- ---- */
/* Subroutine */ int 
fnho(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp)
{
  /* System generated locals */
  integer dfdu_dim1, dfdp_dim1;

  /* Local variables */

  doublereal rtmp;
  integer i, j;
  doublereal ep;
  integer ndm = iap->ndm;
  doublereal umx;
  /* set up local scratch arrays.  We do them as statics
     so we only have to allocate them once.  These routines
     are called many times and the allocation time
     is pohibitive is we allocate and deallocate at
     every call. */
#ifdef STATIC_ALLOC
  static doublereal *dfu=NULL;
  static doublereal *uu1=NULL,*uu2=NULL,*ff1=NULL,*ff2=NULL;
  if(dfu==NULL)
    dfu = (doublereal *)MALLOC(sizeof(doublereal)*ndim*ndim);
  if(uu==NULL)
    uu = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  if(ff1==NULL)
    ff1 = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  if(ff2==NULL)
    ff2 = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
#else
  doublereal *dfu, *uu, *ff1, *ff2;
  dfu = (doublereal *)MALLOC(sizeof(doublereal)*ndim*ndim);
#endif

/* Generates the equations for homoclinic bifurcation analysis */

/* Generate the function. */

  ffho(iap, rap, ndim, u, uold, icp, par, f, ndm, dfu);

  if (ijac == 0) {
#ifndef STATIC_ALLOC
    FREE(dfu);
#endif      
    return 0;
  }

#ifndef STATIC_ALLOC
  uu = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  ff1 = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  ff2 = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
#endif

  /* Generate the Jacobian. */
  dfdu_dim1 = dfdp_dim1 = ndim;

  umx = 0.;
  for (i = 0; i < ndim; ++i) {
    if (fabs(u[i]) > umx) {
      umx = fabs(u[i]);
    }
  }

  rtmp = HMACH;
  ep = rtmp * (umx + 1);

  for (i = 0; i < ndim; ++i) {
    uu[i] = u[i];
  }
  for (i = 0; i < ndim; ++i) {
    uu[i] = u[i] - ep;
    ffho(iap, rap, ndim, uu, uold, icp, par, ff1, ndm, dfu);
    uu[i] = u[i] + ep;
    ffho(iap, rap, ndim, uu, uold, icp, par, ff2, ndm, dfu);
    uu[i] = u[i];
    for (j = 0; j < ndim; ++j) {
      ARRAY2D(dfdu, j, i) = (ff2[j] - ff1[j]) / (ep * 2);
    }
  }

  for (i = 0; i < iap->nfpr; ++i) {
    par[icp[i]] += ep;
    ffho(iap, rap, ndim, u, uold, icp, par, ff1, ndm, dfu);
    for (j = 0; j < ndim; ++j) {
      ARRAY2D(dfdp, j, icp[i]) = (ff1[j] - f[j]) / ep;
    }
    par[icp[i]] -= ep;
  }
#ifndef STATIC_ALLOC
  FREE(dfu);
  FREE(uu);
  FREE(ff1);
  FREE(ff2);
#endif

  return 0;
} /* fnho_ */


/*     ---------- ---- */
/* Subroutine */ int 
ffho(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu)
{
  /* System generated locals */
  integer dfdu_dim1;

  /* Local variables */

  integer i, j;
  doublereal dum1;

  dfdu_dim1 = ndm;
    
  if (blhom_1.istart >= 0) {
    if (blhom_1.itwist == 0) {
      /*        *Evaluate the R.-H. sides */
      funi(iap, rap, ndm, u, uold, icp, par, 0, f, NULL, NULL);
    } else if (blhom_1.itwist == 1) {
      /*        *Adjoint variational equations for normal vector */
      funi(iap, rap, ndm, u, uold, icp, par, 1, f, dfdu, NULL);
      /*        *Set F = - (Df)^T u */
      for (j = 0; j < ndm; ++j) {
        dum1 = 0.;
        for (i = 0; i < ndm; ++i) {
          dum1 += ARRAY2D(dfdu, i, j) * u[ndm + i];
        }
        f[ndm + j] = -dum1;
      }
      /*        *Set F =  F + PAR(10) * f */
      for (j = 0; j < ndm; ++j) {
        f[ndm + j] += par[9] * f[j];
      }
    }
  } else {
    /*        Homoclinic branch switching */
    for (j = 0; j < ndim; j += ndm) {
      funi(iap, rap, ndm, &u[j], &uold[j], icp, par, 0, &f[j], NULL, NULL);
    }      
  }

  /* Scale by truncation interval T=PAR(11) */

  if (blhom_1.istart >= 0) {
    for (i = 0; i < ndim; ++i) {
      f[i] = par[10] * f[i];
    }
  } else {
    for (i = 0; i < ndm; ++i) {
      f[i] = par[9] * f[i];
      for (j = 1; j < ndim/ndm - 1; ++j) {
        f[i + ndm * j] = par[(j << 1) + 18] * f[i + ndm * j];
      }
      f[i + ndim - ndm] = par[10] * f[i + ndim - ndm];
    }
  }

  return 0;
} /* ffho_ */


/*     ---------- ---- */
/* Subroutine */ int 
bcho(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nbc, const doublereal *u0, const doublereal *u1, doublereal *f, integer ijac, doublereal *dbc)
{
  /* System generated locals */
  integer dbc_dim1;

  /* Local variables */

  integer nfpr;
  doublereal rtmp;
  integer i, j;
  doublereal ep, *ff1, *ff2, *uu, umx;
  integer nbc0;

/* Generates the boundary conditions for homoclinic bifurcation analysis */

  dbc_dim1 = nbc;
  
  nbc0 = iap->nbc0;
  nfpr = iap->nfpr;

/* Generate the function. */

  fbho(iap, ndim, par, icp, nbc, nbc0, u0, u1, f);

  if (ijac == 0) {
    return 0;
  }

  ff1=(doublereal *)MALLOC(sizeof(doublereal)*(iap->nbc));
  ff2=(doublereal *)MALLOC(sizeof(doublereal)*(iap->nbc));
  uu=(doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
		     
  /* Derivatives with respect to U0. */

  umx = 0.;
  for (i = 0; i < ndim; ++i) {
    if (fabs(u0[i]) > umx) {
      umx = fabs(u0[i]);
    }
  }
  rtmp = HMACH;
  ep = rtmp * (umx + 1);
  for (i = 0; i < ndim; ++i) {
    uu[i] = u0[i];
  }
  for (i = 0; i < ndim; ++i) {
    uu[i] = u0[i] - ep;
    fbho(iap, ndim, par, icp, nbc, nbc0, uu, u1, ff1);
    uu[i] = u0[i] + ep;
    fbho(iap, ndim, par, icp, nbc, nbc0, uu, u1, ff2);
    uu[i] = u0[i];
    for (j = 0; j < nbc; ++j) {
      ARRAY2D(dbc, j, i) = (ff2[j] - ff1[j]) / (ep * 2);
    }
  }

  /* Derivatives with respect to U1. */

  umx = 0.;
  for (i = 0; i < ndim; ++i) {
    if (fabs(u1[i]) > umx) {
      umx = fabs(u1[i]);
    }
  }
  rtmp = HMACH;
  ep = rtmp * (umx + 1);
  for (i = 0; i < ndim; ++i) {
    uu[i] = u1[i];
  }
  for (i = 0; i < ndim; ++i) {
    uu[i] = u1[i] - ep;
    fbho(iap, ndim, par, icp, nbc, nbc0, u0, uu, ff1);
    uu[i] = u1[i] + ep;
    fbho(iap, ndim, par, icp, nbc, nbc0, u0, uu, ff2);
    uu[i] = u1[i];
    for (j = 0; j < nbc; ++j) {
      ARRAY2D(dbc, j, (ndim + i)) = (ff2[j] - ff1[j]) / (ep * 2);
    }
  }

  for (i = 0; i < nfpr; ++i) {
    par[icp[i]] += ep;
    fbho(iap, ndim, par, icp, nbc, nbc0, u0, u1, ff2);
    for (j = 0; j < nbc; ++j) {
      ARRAY2D(dbc, j, (ndim * 2) + icp[i]) = (ff2[j] - f[j]) / ep;
    }
    par[icp[i]] -= ep;
  }
  FREE(ff1);
  FREE(ff2);
  FREE(uu);

  return 0;
} /* bcho_ */

static doublereal fbho_ddot(const doublereal *a, const doublereal *b, integer ndim)
{
  integer i;
  doublereal result = 0.0;

  for (i = 0; i < ndim; i++)
    result += a[i] * b[i];
  return result;
}

static doublereal fbho_ddotsub(const doublereal *a, const doublereal *b,
                               const doublereal *c, integer ndim)
{
  integer i;
  doublereal result = 0.0;

  for (i = 0; i < ndim; i++)
    result += a[i] * (b[i] - c[i]);
  return result;
}

static int fbho_regular(const iap_type *iap, integer ndim, doublereal *par,
                        const integer *icp, const doublereal *u0,
                        const doublereal *u1, doublereal *fb,
                        const doublereal *xequib1, const doublereal *xequib2)
{
    /* Local variables */

  integer i, k;
  integer jb = 0;
  integer ndm;

  doublereal **bound;
  doublereal **vr[2] = {NULL, NULL}, **vt[2] = {NULL, NULL};
  static doublereal *umax = NULL;

    /* I am not 100% sure if this is supposed to be iap->ndm or iap->ndim,
       but it appears from looking at the code that it should be iap->ndm.
       Also, note that I have replaced the occurances of N X in the algorithms
       with (iap->ndm), so if you change it to iap->ndim here you will
       need to make the similiar changes in the algorithms.
       Finally, the routines called from here prjctn and eigho
       also depend on these arrays, and more importantly the algorithm,
       having N X.  So, they all need to be changed at once.
    */

  ndm = iap->ndm;
  bound   = DMATRIX(ndm, ndm);
  
  /*        *Projection boundary conditions for the homoclinic orbit */
  /*        *NSTAB boundary conditions at t=0 */
  prjctn(bound, xequib1, icp, par, -1, 1, 1, ndm);
  for (i = 0; i < blhom_1.nstab; ++i) {
    fb[jb++] = fbho_ddotsub(bound[i], u0, xequib1, ndm);
    /*         write(9,*) 'fb',jb,fb(jb) */
  }
  /*        *NUNSTAB boundary conditions at t=1 */
  if (blhom_1.nrev == 0) {
    prjctn(bound, xequib2, icp, par, 1, 2, 1, ndm);
    for (i = ndm - blhom_1.nunstab; i < ndm; ++i) {
      if (blhom_1.istart >= 0) {
        fb[jb] = fbho_ddotsub(bound[i], u1, xequib2, ndm);
      } else {
        fb[jb] = fbho_ddotsub(bound[i], &u1[ndim - ndm], xequib2, ndm);
        if (blhom_1.itwist == 0) {
          /*                     allow jump at end. */
          fb[jb] += par[21];
        }
      }
      ++jb;
    }
  } else {
    /*         *NUNSTAB symmetric boundary conditions at t=1 if NREV=1*/
    for (i = 0; i < ndm; ++i) {
      if (blhmp_1.irev[i] > 0) {
        fb[jb++] = u1[i];
      }
    }
  }
  if (blhom_1.nfixed > 0 || blhom_1.iequib == 2) {
    doublereal *ri[2], *rr[2];
    ri[0] = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
    rr[0] = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
    if (blhom_1.iequib < 0) {
      ri[1] = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
      rr[1] = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
    }
    /*        *NFIXED extra boundary conditions for the fixed conditions  */
    if (blhom_1.nfixed > 0) {
      vr[0] = DMATRIX(ndm, ndm);
      if (blhom_1.iequib < 0) {
        vr[1] = DMATRIX(ndm, ndm);
      }
      eigho(1, 2, rr[0], ri[0], vr[0], xequib1, icp, par, ndm);
      if (blhom_1.iequib < 0) {
        eigho(1, 2, rr[1], ri[1], vr[1], xequib2, icp, par, ndm);
      }
      for (i = 0; i < blhom_1.nfixed; ++i) {
        if (blhmp_1.ifixed[i] > 10 && vt[0] == NULL) {
          vt[0] = DMATRIX(ndm, ndm);
          eigho(1, 1, rr[0], ri[0], vt[0], xequib1, icp, par, ndm);
          if (blhom_1.iequib < 0) {
            vt[1] = DMATRIX(ndm, ndm);
            eigho(1, 1, rr[1], ri[1], vt[1], xequib2, icp, par, ndm);
          }
        }
        fb[jb] = psiho(iap, blhmp_1.ifixed[i], rr, ri, vr, vt, icp, par,
                       u0, u1);
        ++jb;
      }
    }
    /*       *extra boundary condition in the case of a saddle-node homoclinic*/
    if (blhom_1.iequib == 2) {
      if (vt[0] == NULL) {
        vt[0] = DMATRIX(ndm, ndm);
        eigho(1, 1, rr[0], ri[0], vt[0], xequib1, icp, par, ndm);
      }
      fb[jb++] = rr[0][blhom_1.nstab];
    }
    FREE(ri[0]);
    FREE(rr[0]);
    if (blhom_1.iequib < 0) {
      FREE(ri[1]);
      FREE(rr[1]);
    }
  }
  /*        *NDM initial conditions for the equilibrium if IEQUIB=1,2,-2   */
  if (blhom_1.iequib != 0 && blhom_1.iequib != -1) {
    func(ndm, xequib1, icp, par, 0, &fb[jb], NULL, NULL);
    jb += ndm;
    /*        *NDM extra initial conditions for the equilibrium if IEQUIB=-2 */
    if (blhom_1.iequib == -2) {
      func(ndm, xequib2, icp, par, 0, &fb[jb], NULL, NULL);
      jb += ndm;
    }
  }
  /*        *boundary conditions for normal vector */
  if (blhom_1.itwist == 1 && blhom_1.istart >= 0) {
    /*           *-orthogonal to the unstable directions of A  at t=0 */
    prjctn(bound, xequib1, icp, par, 1, 1, 2, ndm);
    for (i = ndm - blhom_1.nunstab; i < ndm; ++i) {
      fb[jb++] = fbho_ddot(&u0[ndm], bound[i], ndm);
    }
    /*           *-orthogonal to the stable directions of A  at t=1 */
    prjctn(bound, xequib2, icp, par, -1, 2, 2, ndm);
    for (i = 0; i < blhom_1.nstab; ++i) {
      fb[jb++] = fbho_ddot(&u1[ndm], bound[i], ndm);
    }
  } else if (blhom_1.istart < 0) {
    /*      Branch switching to n-homoclinic orbits. */
    /*         More boundary conditions: continuity+gaps */
    for (k = 0; k < ndim/ndm-1; ++k) {
      for (i = 0; i < ndm; ++i) {
        fb[jb] = u0[ndm * (k + 1) + i] - u1[ndm * k + i];
        if (blhom_1.itwist == 1) {
          /*     Lin(-Sandstede): PAR(20,22,...) contain the gap sizes, */
          /*     PAR(NPARX-2*NDM+1...NPARX-NDM) contains the adjoint unit */
          /*     vector at the gaps. */
          fb[jb] -= par[(k << 1) + 19] * par[NPARX - (ndm << 1) + i];
        }
        ++jb;
      }
    }
    /*     Poincare sections: <x-x_0,\dot x_0>=0 		       */
    /*     PAR(NPARX-NDM+1...NPARX) contain the derivatives of the     */
    /*     point x_0 in the original 				       */
    /*     homoclinic orbit that is furthest from the equilibrium.     */
    /*	   x_0=umax is initialized at each run to an end point, and so */
    /*	   is always in the Poincare section			       */
    if (umax == NULL) {
      umax = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
      for (i = 0; i < ndm; ++i) {
        umax[i] = u1[i];
      }
    }
    for (k = 0; k < ndim/ndm-1; ++k) {
      fb[jb++] += fbho_ddotsub(&par[NPARX - ndm], &u1[k * ndm], umax, ndm);
    }
  }
  FREE_DMATRIX(bound);
  FREE_DMATRIX(vr[0]);
  FREE_DMATRIX(vt[0]);
  if (vr[1] != NULL) 
    FREE_DMATRIX(vr[1]);
  if (vt[1] != NULL) 
    FREE_DMATRIX(vt[1]);
  return jb;
}

static int fbho_homotopy(const iap_type *iap, integer ndm, doublereal *par,
                         const integer *icp, const doublereal *u0,
                         const doublereal *u1, doublereal *fb,
                         const doublereal *xequib1, const doublereal *xequib2)
{
  integer i, j;

  integer jb = 0;
  integer ip;
  integer kp;

  doublereal *ri = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
  doublereal *rr = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
  doublereal **vr, **vt;
  
  ip = 11;
  if (blhom_1.iequib >= 0) {
    ip += ndm;
  } else {
    ip += ndm << 1;
  }
  kp = ip;
  /*        *Explicit boundary conditions for homoclinic orbit at t=0 */
  vr = DMATRIX(ndm, ndm);
  eigho(1, 2, rr, ri, vr, xequib1, icp, par, ndm);
  jb = ndm;
  if (blhom_1.nunstab > 1) {
    fb[jb] = 0.;
    kp = ip + blhom_1.nunstab;
    for (j = 0; j < blhom_1.nunstab; ++j) {
      for (i = 0; i < ndm; ++i) {
        fb[i] += u0[i] - xequib1[i] - par[ip + j + 1] * 
	  vr[ndm - blhom_1.nunstab + j][i];
      }
      /* Computing 2nd power */
      fb[jb] += par[ip + j + 1] * par[ip + j + 1];
    }
    fb[jb++] -= par[ip];
  } else {
    kp = ip + 1;
    for (i = 0; i < ndm; ++i) {
      fb[i] = u0[i] - xequib1[i] - par[ip] * par[ip + 1] * 
	vr[ndm - blhom_1.nunstab][i];
    }
  }
  FREE_DMATRIX(vr);
  /*        *Projection boundary conditions for the homoclinic orbit at t=1 */
  vt = DMATRIX(ndm, ndm);
  eigho(1, 1, rr, ri, vt, xequib2, icp, par, ndm);
  FREE(rr);
  FREE(ri);
  for (i = ndm - blhom_1.nunstab; i < ndm; ++i) {
    ++kp;
    fb[jb++] = fbho_ddotsub(vt[i], u1, xequib2, ndm) - par[kp];
  }
  FREE_DMATRIX(vt);
  /*        *NDM initial conditions for the equilibrium if IEQUIB=1,2,-2 */
  if (blhom_1.iequib != 0 && blhom_1.iequib != -1) {
    func(ndm, xequib1, icp, par, 0, &fb[jb], NULL, NULL);
    jb += ndm;
    /*        *NDM extra initial conditions for the equilibrium if IEQUIB=-2 */
    if (blhom_1.iequib == -2) {
      func(ndm, xequib2, icp, par, 0, &fb[jb], NULL, NULL);
      jb += ndm;
    }
  }
  return jb;
}

/*     ---------- ---- */
/* Subroutine */ int 
fbho(const iap_type *iap, integer ndim, doublereal *par, const integer *icp, integer nbc, integer nbc0, const doublereal *u0, const doublereal *u1, doublereal *fb)
{
  integer i, jb, nbcn;
  integer ndm = iap->ndm;
  doublereal *xequib1, *xequib2;
  
  /* Generates the boundary conditions for homoclinic orbits. */

  /*     *Initialization */
  for (i = 0; i < nbc; ++i) {
    fb[i] = 0.;
  }

  if (blhom_1.iequib == 0 || blhom_1.iequib == -1) {
    pvls(ndm, u0, par);
  }
  /*              write(9,*) 'Xequib:' */
  xequib1 = &par[11];
    /*              write(9,*) I,XEQUIB1(I) */
  if (blhom_1.iequib >= 0) {
    xequib2 = &par[11];
  } else {
    xequib2 = &par[ndm + 11];
  }

  /*     **Regular Continuation** */
  if (blhom_1.istart != 3) {
    jb = fbho_regular(iap, ndim, par, icp, u0, u1, fb, xequib1, xequib2);
  } else {
    /*     **Starting Solutions using Homotopy** */
    jb = fbho_homotopy(iap, ndm, par, icp, u0, u1, fb, xequib1, xequib2);
  }
  /*      write(9,*) NBCN,NBC */
  /* *user defined extra boundary conditions */
  nbcn = nbc - jb;
  if (nbcn > 0) {
    bcnd(ndim, par, icp, nbcn, u0, u1, 0, &fb[jb], NULL);
  } else if (nbcn < 0) {
    printf("Evil BUG!: Negative number of boundary conditions left\n");
    exit(1);
  }
  return 0;
} /* fbho_ */


/*     ---------- ---- */
/* Subroutine */ int 
icho(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *f, integer ijac, doublereal *dint)
{
  /* System generated locals */
  integer dint_dim1;

  /* Local variables */

  integer nfpr;
  doublereal rtmp;
  integer i, j;
  doublereal ep, *ff1, *ff2, *uu, umx;
  integer nnt0;

/* Generates integral conditions for homoclinic bifurcation analysis */

  dint_dim1 = nint;
  
  nnt0 = iap->nnt0;
  nfpr = iap->nfpr;

/* Generate the function. */

  fiho(iap, rap, ndim, par, icp, nint, nnt0, u, uold, udot, upold, f);

  if (ijac == 0) {
    return 0;
  }

  /* Generate the Jacobian. */

  ff1 = (doublereal *)MALLOC(sizeof(doublereal)*(iap->nint));
  ff2 = (doublereal *)MALLOC(sizeof(doublereal)*(iap->nint));
  uu = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));

  umx = 0.;
  for (i = 0; i < ndim; ++i) {
    if (fabs(u[i]) > umx) {
      umx = fabs(u[i]);
    }
  }

  rtmp = HMACH;
  ep = rtmp * (umx + 1);

  for (i = 0; i < ndim; ++i) {
    uu[i] = u[i];
  }
  for (i = 0; i < ndim; ++i) {
    uu[i] = u[i] - ep;
    fiho(iap, rap, ndim, par, icp, nint, nnt0, uu, uold, udot, upold, ff1);
    uu[i] = u[i] + ep;
    fiho(iap, rap, ndim, par, icp, nint, nnt0, uu, uold, udot, upold, ff2);
    uu[i] = u[i];
    for (j = 0; j < nint; ++j) {
      ARRAY2D(dint, j, i) = (ff2[j] - ff1[j]) / (ep * 2);
    }
  }

  for (i = 0; i < nfpr; ++i) {
    par[icp[i]] += ep;
    fiho(iap, rap, ndim, par, icp, nint, nnt0, u, uold, udot, upold, ff1);
    for (j = 0; j < nint; ++j) {
      ARRAY2D(dint, j, ndim + icp[i]) = (ff1[j] - f[j]) / ep;
    }
    par[icp[i]] -= ep;
  }

  FREE(ff1);
  FREE(ff2);
  FREE(uu);
  return 0;
} /* icho_ */


/*     ---------- ---- */
/* Subroutine */ int 
fiho(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, integer nnt0, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *fi)
{
  /* Local variables */

  integer i, jb;
  integer ndm;
  doublereal dum;

  /* Generates the integral conditions for homoclinic orbits. */

  ndm = iap->ndm;
  jb = 0;

/* Integral phase condition for homoclinic orbit */

  if (blhom_1.nrev == 0 && blhom_1.istart >= 0) {
    dum = 0.;
    for (i = 0; i < ndm; ++i) {
      dum += upold[i] * (u[i] - uold[i]);
    }
    fi[jb++] = fbho_ddotsub(upold, u, uold, ndm);

    /* Integral phase condition for adjoint equation */

    if (blhom_1.nrev == 0 && blhom_1.itwist == 1 && blhom_1.istart >= 0) {
      fi[jb++] = fbho_ddotsub(&uold[ndm], &u[ndm], &uold[ndm], ndm);
    }
  }

  /* User-defined integral constraints */

  if (jb < nint) {
    icnd(ndm, par, icp, nint, u, uold, udot, upold, 0, &fi[jb], NULL);
  }
  return 0;
} /* fiho_ */


/*     ---------- ---- */
/* Subroutine */ int 
inho(iap_type *iap, integer *icp, doublereal *par)
{

    /* Local variables */
  integer ndim, nint, nuzr, i, nfree, icorr, nbc, ndm, irs, isw;

  /* Allocate memory for global structures. */
  FREE(blhmp_1.ipsi);
  FREE(blhmp_1.ifixed);
  FREE(blhmp_1.irev);

  blhmp_1.ipsi   = (integer *)MALLOC(sizeof(integer)*NPARX);
  blhmp_1.ifixed = (integer *)MALLOC(sizeof(integer)*NPARX);
  blhmp_1.irev   = (integer *)MALLOC(sizeof(integer)*(iap->ndm));

/* Reads from fort.11 specific constants for homoclinic continuation. */
/* Sets up re-defined constants in IAP. */
/* Sets other constants in the following common blocks. */


/* set various constants */

    /* Parameter adjustments */
    /*--par;*/
    /*--icp;*/
    

    
  ndim = iap->ndim;
  irs = iap->irs;
  isw = iap->isw;
  nbc = iap->nbc;
  nint = iap->nint;
  nuzr = iap->nuzr;
  ndm = ndim;

  fp12 = fopen("fort.12","r");
  fscanf(fp12,"%ld %ld %ld %ld %ld",&blhom_1.nunstab,&blhom_1.nstab,
	 &blhom_1.iequib,&blhom_1.itwist,&blhom_1.istart);
  /*go to the end of the line*/
  while(fgetc(fp12)!='\n');
    
  /* updated reading in of constants for reversible equations */
  /* replaces location in datafile of compzero */

  fscanf(fp12,"%ld",&blhom_1.nrev);
  /*go to the end of the line*/
  while(fgetc(fp12)!='\n');
  if (blhom_1.nrev > 0) {
    for (i = 0; i < ndm; ++i) {
      fscanf(fp12,"%ld",&blhmp_1.irev[i]);
    }
    /*go to the end of the line*/
    while(fgetc(fp12)!='\n');
  }
    
  fscanf(fp12,"%ld",&blhom_1.nfixed);
  /*go to the end of the line*/
  while(fgetc(fp12)!='\n');
  if (blhom_1.nfixed > 0) {
    for (i = 0; i < blhom_1.nfixed; ++i) {
      fscanf(fp12,"%ld",&blhmp_1.ifixed[i]);
    }
    /*go to the end of the line*/
    while(fgetc(fp12)!='\n');
  }
  fscanf(fp12,"%ld",&blhom_1.npsi);
  /*go to the end of the line*/
  while(fgetc(fp12)!='\n');
  if (blhom_1.npsi > 0) {
    for (i = 0; i < blhom_1.npsi; ++i) {
      fscanf(fp12,"%ld",&blhmp_1.ipsi[i]);
    }
    /*go to the end of the line*/
    while(fgetc(fp12)!='\n');
  }
  fclose(fp12);
  nfree = blhom_1.nfixed + 2 - blhom_1.nrev + nint + nbc;

  if (blhom_1.istart < 0) {
  /*        n-homoclinic branch switching */
    nfree += -blhom_1.istart - 1;
    ndim = ndm*(-blhom_1.istart+1);

  /* Free parameter (artificial parameter for psi) */
  /* nondegeneracy parameter of the adjoint */

  } else if (blhom_1.itwist == 1) {
    icp[nfree] = 9;
    ++nfree;
    par[9] = 0.;
    ndim = ndm * 2;
  }

  /* Extra free parameters for equilibrium if iequib=1,2,-2 */

  if (blhom_1.iequib != 0 && blhom_1.iequib != -1) {
    for (i = 0; i < ndm; ++i) {
      icp[nfree + i] = i + 11;
    }
  }

  if (blhom_1.iequib == -2) {
    for (i = 0; i < ndm; ++i) {
      icp[nfree + ndm + i] = ndm + 11 + i;
    }
  }

  if (blhom_1.istart != 3) {
    /*     *regular continuation */
    if (blhom_1.istart >= 0) {
      nint += blhom_1.itwist + 1 - blhom_1.nrev;
    }
    if (isw == 2) {
      icorr = 2;
    } else {
      icorr = 1;
    }
    nbc += blhom_1.nstab + blhom_1.nunstab + ndim - ndm +
      blhom_1.iequib * ndm + nfree - nint - icorr;
    if (blhom_1.iequib == 2) {
      nbc -= (ndm - 1);
    }
    if (blhom_1.iequib < 0) {
      nbc -= (blhom_1.iequib * 3 + 2) * ndm;
    }
  } else {
    /*     *starting solutions using homotopy */
    if (blhom_1.nunstab == 1) {
      nbc = ndm * (blhom_1.iequib + 1) + 1;
    } else {
      nbc = ndm * (blhom_1.iequib + 1) + blhom_1.nunstab + 1;
    }
    if (blhom_1.iequib == 2) {
      fprintf(fp9,"WARNING: IEQUIB=2 NOT ALLOWED WITH ISTART=3\n");	
    }
    if (blhom_1.iequib < 0) {
      nbc -= ndm * (blhom_1.iequib * 3 + 2);
    }
    nint = 0;
  }

  /* write new constants into IAP */

  iap->ndim = ndim;
  iap->nbc = nbc;
  iap->nint = nint;
  iap->nuzr = nuzr;
  iap->ndm = ndm;

  return 0;
} /* inho_ */


/*     ---------- ------ */
/* Subroutine */ int intpho(iap_type *iap, rap_type *rap, integer ndm,
        integer ncolrs, doublereal tm, 
        doublereal dtm, integer ndx, doublereal **ups, doublereal **udotps, 
        doublereal t, doublereal dt, integer n, integer ndim, integer j, 
        integer j1)
{
  /* Local variables */
  doublereal d, z;
  integer i, k, l;
  integer k1, l1;
  integer ncp1 = ncolrs + 1;
  doublereal *w = (doublereal *)MALLOC(sizeof(doublereal) * ncp1);
  doublereal *x = (doublereal *)MALLOC(sizeof(doublereal) * ncp1);
      
/*     Finds interpolant (TM(.) , UPS(.), UDOTPS(.) ) on the new mesh */
/*     at times TM,TM+DTM using the old mesh at times T,T+DT. */

/*     Used by TRANHO to initiate branch switching to n-homoclinic orbits. */

  d = dtm / ncolrs;
  for (l = 0; l < ncp1; ++l) {
    x[l] = tm + l * d;
  }
  for (i = 0; i < ncolrs; ++i) {
    z = t + dt * i / ncolrs;
    intwts(iap, rap, &ncp1, &z, x, w);
    k1 = i * ndim + n;
    for (k = 0; k < ndm; ++k) {
      ups[j1][k1 + k] = w[ncolrs] * ups[j + 1][n + k];
      udotps[j1][k1 + k] = w[ncolrs] * udotps[j + 1][n + k];
      for (l = 0; l < ncolrs; ++l) {
        l1 = k + l * ndim + n;
        ups[j1][k1 + k] += w[l] * ups[j][l1];
        udotps[j1][k1 + k] += w[l] * udotps[j][l1];
      }
    }
  }

  FREE(w);
  FREE(x);
  
  return 0;
} /* intpho_ */

/*     ---------- ------ */
/* Subroutine */ int tranho(iap_type *iap, rap_type *rap,
        integer *ntsr, integer ncolrs, integer ndm, 
        integer ndim, doublereal *tm, doublereal *dtm, integer ndx, 
        doublereal **ups, doublereal **udotps, const integer *icp, doublereal *par)

/*     Transform the data representation of the homoclinic orbit into */
/*     an object suitable for homoclinic branch switching: */

/*     dim|1...............NDM|NDM+1......NDIM-NDM|NDIM-NDM+1......NDIM| */
/*        |                   |                   |                    | */
/*     t=0|start of hom. orbit|maximum from equil.| maximum from equil.| */
/*        |       :           |       :           |       :            | */
/*        |       :           |end of hom. orbit  |       :            | */
/*        |       :           |start of hom. orbit|       :            | */
/*        |       :           |        :          |       :            | */
/*     t=1|maximum from equil.|maximum from equil.| end of hom. orbit  | */

/*     Called by PREHO */
{
  /* Local variables */
  integer jmax;
  doublereal upsi, a[3], b[3];
  integer i, j, k, l;
  doublereal t[3], dnorm, tmmax;
  integer i2, j2[3], k2;
  doublereal tt[3];
  doublereal upsmax;
  doublereal dum1, dum2;
  doublereal *ttm = (doublereal *)MALLOC(sizeof(doublereal) * (*ntsr << 2));

/* Find maximum from the equilibrium */

  upsmax = 0.;
  jmax = 0;
  for (j = 0; j <= *ntsr; ++j) {
    upsi = 0.;
    for (i = 0; i < ndm; ++i) {
      upsi += (ups[j][i] - par[i + 11]) *
              (ups[j][i] - par[i + 11]);
    }
    if (upsi > upsmax) {
      upsmax = upsi;
      jmax = j;
    }
  }
  tmmax = tm[jmax];
  
  func(ndm, ups[jmax], icp, par, 0, &par[NPARX-ndm], &dum1, &dum2);

/*     PAR(NPARX-NDM+1...NPARX) contains the point furthest from */
/*     the equilibrium. */
/*     PAR(10)=the time for the unstable manifold tail. */
/*     PAR(11)=the time for the stable manifold tail. */
/*     PAR(20,22,...) contain the gap sizes. */
/*     PAR(21,23,...) contain the times between Poincare sections */

  par[9] = par[10] * tmmax;
  par[19] = 0.;
  for (k = 1; k < ndim/ndm-1; ++k) {
    par[(k << 1) + 18] = par[10];
    par[(k << 1) + 19] = 0.;
  }
  par[10] *= 1. - tmmax;

/*     Remember adjoint at maximum for applying Lin's method */
/*     PAR(NPARX-2*NDM+1...NPARX-NDM) will contain the adjoint unit */
/*     vector at the gaps. */

  if (blhom_1.itwist == 1) {
    dnorm = 0.;
    for (i = 0; i < ndm; ++i) {
      par[NPARX - (ndm << 1) + i] = ups[jmax][ndm + i];
      dnorm += ups[jmax][ndm + i] * ups[jmax][ndm + i];
    }
    dnorm = sqrt(dnorm);
    for (i = 0; i < ndm; ++i) {
      par[NPARX - (ndm << 1) + i] /= dnorm;
    }
  }

/*     Prepare the new NDIM*NCOLRS dimensional UPS matrix */
/*     Move everything to the end in "middle part format" */
/*     so that we can subsequently overwrite the beginning. */

  for (l = (*ntsr << 1) - 1; l >= *ntsr - 1; --l) {
    j = l - ((*ntsr << 1) - 1) + jmax;
    if (j < 0) {
      j += *ntsr;
    }
    ttm[l] = tm[j] - tmmax;
    if (ttm[l] < 0.) {
      ttm[l] += 1.;
    }
    for (k = 0; k < ncolrs * ndim; k += ndim) {
      for (i = k; i < k + ndm; ++i) {
        ups[l][i + ndm] = ups[j][i];
        udotps[l][i + ndm] = udotps[j][i];
        ups[l][i] = ups[j][i];
        udotps[l][i] = udotps[j][i];
        if (l < (*ntsr << 1) - jmax + 1) {
          ups[l + jmax - 1][i + ndim - ndm] = ups[j][i];
          udotps[l + jmax - 1][i + ndim - ndm] = udotps[j][i];
        }
      }
    }
  }
  ttm[(*ntsr << 1) - 1] = 1.;

/*     create matching mesh */
/*     merge TM(1..JMAX)/TMMAX, TM(JMAX..NTSR)-TMMAX, */
/*           TM(1..JMAX)+1D0-TMMAX, */
/*           (TM(JMAX..NTSR)-TMMAX)/(1D0-TMMAX) */

  j2[0] = (*ntsr << 1) - jmax;
  j2[1] = *ntsr;
  j2[2] = *ntsr;
  a[0] = tmmax - 1.;
  a[1] = 0.;
  a[2] = 0.;
  b[0] = tmmax;
  b[1] = 1.;
  b[2] = 1. - tmmax;
  *ntsr = (*ntsr << 1) - 2;
  for (i = 0; i < 3; ++i) {
    t[i] = (ttm[j2[i]] + a[i]) / b[i];
    tt[i] = (ttm[j2[i] - 1] + a[i]) / b[i];
  }
  for (j = 1; j <= *ntsr; ++j) {
    tm[j] = t[0];
    i2 = 0;
    for (i = 1; i < 3; ++i) {
      if (t[i] < tm[j]) {
        tm[j] = t[i];
        i2 = i;
      }
    }

    dtm[j - 1] = tm[j] - tm[j - 1];
    
/*     copy first part to temp arrays upst */
/*     Replace UPS and UDOTPS by its interpolant on the new mesh : */

    intpho(iap, rap, ndm, ncolrs, tt[0], t[0] - tt[0], ndx, ups, udotps, tm[j - 1],
           dtm[j - 1], 0, ndim, j2[0] - 1, j - 1);
    /*     Remesh middle part : */

    intpho(iap, rap, ndm, ncolrs, tt[1], t[1] - tt[1], ndx, ups, udotps, tm[j - 1],
           dtm[j - 1], ndm, ndim, j2[1] - 1, j - 1);

/*     Remesh last part : */
    
    intpho(iap, rap, ndm, ncolrs, tt[2], t[2] - tt[2], ndx, ups, udotps, tm[j - 1],
           dtm[j - 1], ndim - ndm, ndim, j2[2] + jmax - 2, j - 1);

/*     Copy middle parts, this applies only for 1->n switching */
/*     where n>=3 and NDIM=(n+1)*NDM: (NDIM/NDM)-3 times. */

    for (k2 = ndm; k2 < ndim - ndm * 2; k2 += ndm) {
      for (k = ndm; k <= (ncolrs - 1) * ndim + ndm; k += ndim) {
        for (i = k; i < k + ndm; ++i) {
          ups[j - 1][i + k2] = ups[j - 1][i];
          udotps[j - 1][i + k2] = udotps[j - 1][i];
        }
      }
    }
    ++j2[i2];
    tt[i2] = t[i2];
    t[i2] = (ttm[j2[i2]] + a[i2]) / b[i2];
  }

/*     Adjust end points */

  for (i = 0; i < ndm; ++i) {
    for (k2 = i; k2 < ndim; k2 += ndm) {
      ups[*ntsr][k2] = ups[*ntsr + 1][i + ndm];
      udotps[*ntsr][k2] = udotps[*ntsr + 1][i + ndm];
    }
    ups[*ntsr][i + ndim - ndm] = ups[0][i];
    udotps[*ntsr][i + ndim - ndm] = udotps[0][i];
  }
  
  FREE(ttm);
  return 0;
} /* tranho_ */


/*     ---------- ------ */
/* Subroutine */ int cpbkho(integer *ntsr, integer ncolrs, integer *nar, 
        integer ndm, doublereal *tm, doublereal *dtm, integer ndx, 
        doublereal **ups, doublereal **udotps, doublereal *par)

/*     Copy the homoclinic orbit back from the special representation */
/*     This is called from PREHO in order to perform normal continuation */
/*     again once the branch switching is complete. */
{
  /* Local variables */
  integer ndim;
  doublereal time;
  integer i, j, k, l, m;
  doublereal tbase;
  integer ncopy;
  
  ndim = ndm * (blhom_1.itwist + 1);
  ncopy = *nar / ndm;
  j = 0;
  time = par[9] + par[10];
  for (k = 1; k < ncopy; ++k) {
    time += par[(k << 1) + 18];
  }
  tbase = time - par[10];
  tm[*ntsr * ncopy] = 1.;
  for (k = ncopy - 1; k >= 0; --k) {
    for (j = *ntsr - 1; j >= 0; --j) {
      i = j + *ntsr * k;
      for (l = 0; l < ncolrs; ++l) {
        for (m = 0; m < ndm; ++m) {
          ups[i][l * ndim + m] = ups[j][l * *nar + k * ndm + m];
          udotps[i][l * ndim + m] = udotps[j][l * *nar + k * ndm + m];
        }
      }
      if (k == 0) {
        tm[i] = tm[j] * par[9] / time;
      } else if (k == ncopy - 1) {
        tm[i] = (tbase + tm[j] * par[10]) / time;
      } else {
        tm[i] = (tbase + tm[j] * par[(k << 1) + 18]) / time;
      }
      dtm[i] = tm[i + 1] - tm[i];
    }
    if (k == 1) {
      tbase -= par[9];
    } else {
      tbase -= par[(k << 1) + 16];
    }
  }
  *ntsr *= ncopy;

/* Last equal to first */

  for (k = 1; k <= ndim; ++k) {
    ups[*ntsr][k] = ups[0][k];
    udotps[*ntsr][k] = udotps[0][k];
  }
  par[9] = 0.;
  par[10] = time;
  *nar = ndm;
  return 0;
} /* cpbkho_ */

/*     ---------- ----- */
/* Subroutine */ int 
preho(iap_type *iap, rap_type *rap, doublereal *par, const integer *icp,
      integer ndx, integer *ntsr, integer *nar, integer ncolrs, doublereal **ups,
      doublereal **udotps, doublereal *tm, doublereal *dtm)
{
  /* Local variables */
  integer jmin, ndim, ndm;
  doublereal upsi;
  integer i, j, k;
  doublereal tmmin;
  integer k1, k2, ii;
  doublereal upsmin, dum1, dum2;
  integer ist;


  

/* Preprocesses (perturbs) restart data to enable */
/* initial computation of the adjoint variable */


    /* Parameter adjustments */
    /*--tm;*/
    /*--par;*/
  
  ndim = iap->ndim;
  ndm = iap->ndm;  

  /* Shift phase if necessary if continuing from */
  /* a periodic orbit into a homoclinic one */

  if (blhom_1.istart == 4) {

    /* Try to find an approximate value for the equilibrium if it's not
       explicitely given. This is just the point where the speed is minimal.
       We hope that Newton's method will do the rest. */
    if (blhom_1.iequib > 0) {
      doublereal *ui = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
      doublereal *f = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
      upsmin = 1e20;
      jmin = 0;
      for (j = 0; j <= *ntsr; ++j) {
        for (i = 0; i < ndm; ++i) {
          ui[i] = ups[j][i];
        }
        func(ndm, ui, icp, par, 0, f, &dum1, &dum2);
        upsi = 0.;
        for (i = 0; i < ndm; ++i) {
          upsi += f[i] * f[i];
        }
        if (upsi < upsmin) {
          jmin = j;
          upsmin = upsi;
        }
      }
      for (i = 0; i < ndm; ++i) {
        par[11+i] = ups[jmin][i];
      }
      FREE(ui);
      FREE(f);
    }

    /* First find smallest value in norm */

    upsmin = 1e20;
    jmin = 0;
    for (j = 0; j < *ntsr + 1; ++j) {
      upsi = 0.;
      for (i = 0; i < ndm; ++i) {
	upsi += (ups[j][i] - par[i + 11]) * (ups[j][i] - par[i + 11]);
      }
      if (upsi <= upsmin) {
	upsmin = upsi;
	jmin = j;
      }
    }
    tmmin = tm[jmin];

    /* And then do the actual shift */

    if (jmin != 0) {
      ist = -1;
      j = *ntsr;
      for (ii = 0; ii < *ntsr; ++ii) {
	if (j == *ntsr) {
	  ++ist;
	  tm[j] = tm[ist];
	  for (k = 0; k < ncolrs * ndim; ++k) {
	    ups[j][k] = ups[ist][k];
	    udotps[j][k] = udotps[ist][k];
	  }
	  j = ist;
	}
	i = j;
	j = j + jmin;
	if (j >= *ntsr) {
	  j -= *ntsr;
	}
	if (j == ist) {
	  j = *ntsr;
	}
	tm[i] = tm[j] - tmmin;
	if (tm[i] < 0.) {
	  tm[i] += 1.;
	}
	for (k = 0; k < ncolrs * ndim; ++k) {
	  ups[i][k] = ups[j][k];
	  udotps[i][k] = udotps[j][k];
	}
      }

      /* Last equal to first */

      tm[*ntsr] = 1.;
      for (k = 0; k < ncolrs * ndim; ++k) {
	ups[*ntsr][k] = ups[0][k];
	udotps[*ntsr][k] = udotps[0][k];
      }

    }
  }

  /* If ISTART<0 we perform homoclinic branch switching and need */
  /* to change the representation of the homoclinic orbit in UPS and */
  /* UDOTPS. */

  if (blhom_1.istart < 0 && *nar < ndim && *nar < ndm * 3) {
    tranho(iap, rap, ntsr, ncolrs, ndm, ndim, tm, dtm, ndx, ups, udotps, icp, par);
  } else if (blhom_1.istart < 0 && *nar < ndim && *nar >= ndm * 3) {
/* Copy forelast part */
    for (j = 0; j <= *ntsr; ++j) {
      for (k = 0; k < ndim * ncolrs; k += ndim) {
        for (i = ndim - 1; i >= *nar - ndm; --i) {
          ups[j][k + i] =
              ups[j][k + i - ndim + *nar];
          udotps[j][k + i] =
              udotps[j][k + i - ndim + *nar];
        }
      }
    }
    for (i = 1; i <= (ndim - *nar) / ndm; ++i) {
      par[(*nar / ndm + (i << 1)) + 15] = par[(*nar << 1) / ndm + 15];
      par[(*nar / ndm + (i << 1)) + 14] = par[(*nar << 1) / ndm + 14];
    }
    par[(*nar << 1) / ndm + 15] = (ups[0][*nar - ndm] -
        ups[*ntsr][*nar - (ndm << 1)]) /
          par[NPARX - (ndm << 1)];
  } else if (*nar > ndm << 1 && blhom_1.istart >= 0) {
    /*        Use the usual representation again for normal continuation. */
      cpbkho(ntsr, ncolrs, nar, ndm, tm, dtm, ndx, ups, udotps, par);
  }

  /* Preprocesses (perturbs) restart data to enable */
  /* initial computation of the adjoint variable */
  
  if (*nar != ndim && blhom_1.itwist == 1 && blhom_1.istart >= 0) {
    for (j = 0; j < *ntsr; ++j) {
      for (i = 0; i < ncolrs; ++i) {
	k1 = i * ndim;
	k2 = (i + 1) * ndim - 1;
	for (k = k1 + *nar; k <= k2; ++k) {
	  ups[j][k] = .1;
	}
      }
    }
    for (k = *nar; k < ndim; ++k) {
      ups[*ntsr][k] = .1;
    }
  }

  return 0;
} /* preho_ */


/*     ---------- ------ */
/* Subroutine */ int 
stpnho(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu)
{
  /* Local variables */
  integer ndim, ncol, nfpr, ntst, ncol1, i, j, k;
  doublereal t, *u;
  integer k1, k2;

  doublereal dt;
  integer lab, ibr;

  u = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  /* Generates a starting point for the continuation of a branch of */
  /* of solutions to general boundary value problems by calling the user */
  /* supplied subroutine STPNT where an analytical solution is given. */

  /* Local */

    /* Parameter adjustments */
    /*--par;*/
    /*--icp;*/
    /*--rlcur;*/
    /*--rldot;*/
    /*--tm;*/
    /*--dtm;*/
  ndim = iap->ndim;
  ntst = iap->ntst;
  ncol = iap->ncol;
  nfpr = iap->nfpr;

/* Generate the (initially uniform) mesh. */

  msh(iap, rap, tm);
  dt = 1. / (ntst * ncol);

  for (j = 0; j < ntst + 1; ++j) {
    if (j == ntst) {
      ncol1 = 1;
    } else {
      ncol1 = ncol;
    }
    for (i = 0; i < ncol1; ++i) {
      t = tm[j] + i * dt;
      k1 = i * ndim;
      k2 = (i + 1) * ndim - 1;
      stpho(iap, icp, u, par, t);
      for (k = k1; k <= k2; ++k) {
	ups[j][k] = u[k - k1];
      }
    }
  }

  *ntsr = ntst;
  *ncolrs = ncol;
  ibr = 1;
  iap->ibr = ibr;
  lab = 0;
  iap->lab = lab;

  for (i = 0; i < nfpr; ++i) {
    rlcur[i] = par[icp[i]];
  }

  *nodir = 1;
  FREE(u);
  return 0;
} /* stpnho_ */


/*     ---------- ----- */
/* Subroutine */ int 
stpho(iap_type *iap, integer *icp, doublereal *u, doublereal *par, doublereal t)
{
    /* Local variables */

  integer i, j;

  integer ip;
  integer kp;
  integer ndm;

  doublereal *ri;
  doublereal *rr, **vr, **vt, *xequib;

  ndm = iap->ndm;

  /* Generates a starting point for homoclinic continuation */
  /* If ISTART=2 it calls STPNHO. */
  /* If ISTART=3 it sets up the homotopy method. */

  /* Initialize parameters */

  stpnt(ndm, t, u, par);

  /* Initialize solution and additional parameters */

  if (blhom_1.istart != 3)
    return 0;

  /* ----------------------------------------------------------------------- */
  /* case 1: */
  /* Obsolete option */
  /* case 2: */
  /*     *Regular continuation (explicit solution in STHO) */
  /* break; */

  /* ----------------------------------------------------------------------- */
  /* case 3: */
  /*     *Starting solutions using homotopy */

  ri      = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
  rr      = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
  vr      = DMATRIX(ndm, ndm);
  vt      = DMATRIX(ndm, ndm);

  pvls(ndm, u, par);
  xequib = &par[11];
  eigho(1, 1, rr, ri, vt, xequib, icp, par, ndm);
  eigho(1, 2, rr, ri, vr, xequib, icp, par, ndm);

  /* Set up artificial parameters at the left-hand end point of orbit */

  ip = 11;
  if (blhom_1.iequib >= 0) {
    ip += ndm;
  } else {
    ip += ndm * 2;
  }
  kp = ip;

/* Parameters xi 1=1, xi i=0, i=2,NSTAB */

  par[ip + 1] = 1.;
  if (blhom_1.nunstab > 1) {
    for (i = 1; i < blhom_1.nunstab; ++i) {
      par[ip + i + 1] = 0.;
    }
  }
  ip += blhom_1.nunstab;

/*Starting guess for homoclinic orbit in real principal unstable direction
*/

  for (i = 0; i < ndm; ++i) {
    u[i] = xequib[i] + vr[blhom_1.nstab][i] * par[kp] * par[kp + 1] *
      exp(rr[blhom_1.nstab] * t * par[10]);
  }
  for (i = 0; i < ndm; ++i) {
    fprintf(fp9,"stpho %20.10f\n",u[i]);	
  }
  fprintf(fp9,"\n");	

/* Artificial parameters at the right-hand end point of the orbit */
/* omega_i=<x(1)-x_o,w_i^*> */

  for (i = 0; i < blhom_1.nunstab; ++i) {
    par[ip + i + 1] = 0.;
    for (j = 0; j < ndm; ++j) {
      par[ip + i + 1] += vr[blhom_1.nstab][j] * par[kp] * par[kp + 1] *
        exp(rr[blhom_1.nstab] * par[10]) * vt[blhom_1.nstab + i][j];
    }
  }
  ip += blhom_1.nunstab;
  FREE(ri);
  FREE(rr);
  FREE_DMATRIX(vr);
  FREE_DMATRIX(vt);

  return 0;
  /* -----------------------------------------------------------------------
   */
} /* stpho_ */


/*     ---------- ------ */
/* Subroutine */ int 
pvlsho(iap_type *iap, rap_type *rap, integer *icp, doublereal *dtm, integer *ndxloc, doublereal **ups, integer *ndim, doublereal **p0, doublereal **p1, doublereal *par)
{
  /* Local variables */
  integer i, j;
  struct {
    doublereal *rr[2], *ri[2], **v[2], **vt[2];
    integer ineig;
  } bleig;

  doublereal orient;

  integer iid, ndm;

    /* Parameter adjustments */
    /*--icp;*/
    /*--dtm;*/
    /*--par;*/
  for (i = 0; i < (blhom_1.iequib < 0 ? 2 : 1); i++)
  {
    bleig.rr[i]     = (doublereal *)MALLOC(sizeof(doublereal)*(*ndim));
    bleig.ri[i]     = (doublereal *)MALLOC(sizeof(doublereal)*(*ndim));
    bleig.v[i]      = DMATRIX(*ndim, *ndim);
    bleig.vt[i]     = DMATRIX(*ndim, *ndim);
  }
  
  iid = iap->iid;
  ndm = iap->ndm;

  pvlsbv(iap, rap, icp, dtm, ndxloc, ups, ndim, 
	 p0, p1, par);

  /*      *Compute eigenvalues */
  bleig.ineig = 0;
  eigho(1, 2, bleig.rr[0], bleig.ri[0], bleig.v[0], &par[11], icp, par, ndm);
  if (blhom_1.iequib < 0) {
    eigho(1, 2, bleig.rr[1], bleig.ri[1], bleig.v[1], &par[11 + ndm], icp, par, ndm);
  }
  if (iid >= 3) {
    fprintf(fp9,"EIGENVALUES\n");
    for (j = 0; j < ndm; ++j) {
      fprintf(fp9," (%12.7f %12.7f)\n",bleig.rr[0][j],bleig.ri[0][j]);
    }
    if (blhom_1.iequib < 0)
    {
      fprintf(fp9,"EIGENVALUES of RHS equilibrium\n");
      for (j = 0; j < ndm; ++j) {
        fprintf(fp9," (%12.7f %12.7f)\n",bleig.rr[1][j],bleig.ri[1][j]);	
      }
    }
  }
  if (blhom_1.itwist == 1 && blhom_1.istart >= 0) {
    eigho(1, 1, bleig.rr[0], bleig.ri[0], bleig.vt[0], 
	  &par[11], icp, par, ndm);
    if (blhom_1.iequib < 0) {
      eigho(1, 1, bleig.rr[1], bleig.ri[1], bleig.vt[1],
            &par[11 + ndm], icp, par, ndm);
    }
    bleig.ineig = 1;
    orient = psiho(iap, 0, bleig.rr, bleig.ri, bleig.v, bleig.vt, icp, par,
                   ups[0], ups[iap->ntst]);
    if (iid >= 3) {
      if (orient < 0.) {
	fprintf(fp9," Non-orientable, (%20.10f)\n",orient);	
      } else {
	fprintf(fp9," Orientable (%20.10f)\n",orient);	
      }
    }
  }

  for (i = 0; i < blhom_1.npsi; ++i) {
    if (blhmp_1.ipsi[i] > 10 && bleig.ineig == 0) {
      eigho(1, 1, bleig.rr[0], bleig.ri[0], bleig.vt[0], &par[11], icp, par, ndm);
      if (blhom_1.iequib < 0) {
        eigho(1, 1, bleig.rr[1], bleig.ri[1], bleig.vt[1], &par[11 + ndm], icp, par, ndm);
      }
      bleig.ineig = 1;
    }
    par[blhmp_1.ipsi[i] + 19] =
      psiho(iap, blhmp_1.ipsi[i], bleig.rr, bleig.ri,
            bleig.v, bleig.vt, icp, par, ups[0], ups[iap->ntst]);
    if (iid >= 3) {
      fprintf(fp9," PSI(%2ld)=%20.10f\n",blhmp_1.ipsi[i],par[blhmp_1.ipsi[i] + 19]);	

    }
  }

  for (i = 0; i < (blhom_1.iequib < 0 ? 2 : 1); i++)
  {
    FREE(bleig.rr[i]);
    FREE(bleig.ri[i]);
    FREE_DMATRIX(bleig.v[i]);
    FREE_DMATRIX(bleig.vt[i]);
  }
  return 0;
} /* pvlsho_ */


/*     -------- ------- -------- ----- */
doublereal 
psiho(const iap_type *iap, integer is, doublereal **rr, doublereal **ri, doublereal ***v, doublereal ***vt, const integer *icp, doublereal *par, const doublereal *pu0, const doublereal *pu1)
{
  /* System generated locals */
  doublereal ret_val;

    /* Local variables */

  integer i, j;
  doublereal *f0, *f1, droot, s1, s2, f0norm, f1norm, u0norm, u1norm;
  integer ndm;

  f0 = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndm));    
  f1 = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndm));

/* The conditions for degenerate homoclinic orbits are given by PSI(IS)=0.*/
/* RR and RI contain the real and imaginary parts of eigenvalues which are */
/* ordered with respect to their real parts (smallest first). */
/* The (generalised) real eigenvectors are stored as the ROWS of V. */
/* The (generalised) real left eigenvectors are in the ROWS of VT. */
/* In the block ENDPTS are stored the co-ordinates of the left (PU0) */
/* and right (PU1) endpoints of the solution (+  vector if that is computed)*/


/* Local */


    /* Parameter adjustments */
    /*--par;*/
    /*--icp;*/
    /*--ri;*/
    /*--rr;*/
    
  ndm = iap->ndm;

  func(ndm, pu0, icp, par, 0, f0, NULL, NULL);
  func(ndm, pu1, icp, par, 0, f1, NULL, NULL);

  ret_val = 0.;

  switch (is) {

/*  Compute orientation */

  case 0:
    s1 = 0.;
    s2 = 0.;
    f0norm = 0.;
    f1norm = 0.;
    u0norm = 0.;
    u1norm = 0.;
    for (j = 0; j < ndm; ++j) {
      s1 += f1[j] * pu0[ndm + j];
      s2 += f0[j] * pu1[ndm + j];
      /* Computing 2nd power */
      f0norm += f0[j] * f0[j];
      /* Computing 2nd power */
      f1norm += f1[j] * f1[j];
      /* Computing 2nd power */
      u0norm += pu0[j + ndm] * pu0[j + ndm];
      /* Computing 2nd power */
      u1norm += pu1[j + ndm] * pu1[j + ndm];
    }
    droot = sqrt(f0norm * f1norm * u0norm * u1norm);
    if (droot != 0.) {
      ret_val = -s1 * s2 / droot;
    } else {
      ret_val = 0.;
    }
    break;

  /* Resonant eigenvalues (neutral saddle) */

 case 1:
  ret_val = rr[0][blhom_1.nstab - 1] + rr[0][blhom_1.nstab] +
      ri[0][blhom_1.nstab - 1] + ri[0][blhom_1.nstab];
  break;

/* Double real leading eigenvalues (stable) */
/*   (saddle, saddle-focus transition) */

 case 2:
  if (fabs(ri[0][blhom_1.nstab - 1]) > HMACHHO) {
    /* Computing 2nd power */
    doublereal tmp= ri[0][blhom_1.nstab - 1] - ri[0][blhom_1.nstab - 2];
    ret_val = -(tmp * tmp);
  } else {
    /* Computing 2nd power */
    doublereal tmp = rr[0][blhom_1.nstab - 1] - rr[0][blhom_1.nstab - 2];
    ret_val = tmp * tmp;
  }
  break;

/* Double real positive eigenvalues (unstable) */
/*   (saddle, saddle-focus transition) */

 case 3:
  if (fabs(ri[0][blhom_1.nstab]) > HMACHHO) {
    /* Computing 2nd power */
    doublereal tmp = ri[0][blhom_1.nstab] - ri[0][blhom_1.nstab + 1];
    ret_val = -(tmp * tmp);
  } else {
    /* Computing 2nd power */
    doublereal tmp = rr[0][blhom_1.nstab] - rr[0][blhom_1.nstab + 1];
    ret_val = tmp * tmp;
  }
  break;

/* Neutral saddle, saddle-focus or bi-focus (includes 1, above, also) */

 case 4:
  ret_val = rr[0][blhom_1.nstab - 1] + rr[0][blhom_1.nstab];
  break;

  /* Neutrally-divergent saddle-focus (stable eigenvalues complex) */

 case 5:
  ret_val = rr[0][blhom_1.nstab - 1] + rr[0][blhom_1.nstab] + rr[0][blhom_1.nstab - 2];
  break;

/* Neutrally-divergent saddle-focus (unstable eigenvalues complex) */

 case 6:
  ret_val = rr[0][blhom_1.nstab - 1] + rr[0][blhom_1.nstab] + rr[0][blhom_1.nstab + 1];
  break;

/* Three leading eigenvalues (stable) */

 case 7:
  {
    double vnorm1 = 0.0;
    double vnorm2 = 0.0;
    for (i = 0; i < ndm; i++) {
      vnorm1 += fabs(v[0][blhom_1.nstab - 1][i]);
      vnorm2 += fabs(v[0][blhom_1.nstab - 3][i]);
    }
    if (vnorm1 > vnorm2)
      ret_val = rr[0][blhom_1.nstab - 1] - rr[0][blhom_1.nstab - 3];
    else
      ret_val = rr[0][blhom_1.nstab - 3] - rr[0][blhom_1.nstab - 1];
  }
  break;

  /* Three leading eigenvalues (unstable) */

 case 8:
  {
    double vnorm1 = 0.0;
    double vnorm2 = 0.0;
    for (i = 0; i < ndm; i++) {
      vnorm1 += fabs(v[0][blhom_1.nstab][i]);
      vnorm2 += fabs(v[0][blhom_1.nstab + 2][i]);
    }
    if (vnorm1 > vnorm2)
      ret_val = rr[0][blhom_1.nstab] - rr[0][blhom_1.nstab + 2];
    else
      ret_val = rr[0][blhom_1.nstab + 2] - rr[0][blhom_1.nstab];
  }
  break;

  /* Local bifurcation (zero eigenvalue or Hopf): NSTAB decreases */
  /*  (nb. the problem becomes ill-posed after a zero of 9 or 10) */

 case 9:
  ret_val = rr[0][blhom_1.nstab - 1];
  break;

/* Local bifurcation (zero eigenvalue or Hopf): NSTAB increases */

 case 10:
  ret_val = rr[0][blhom_1.nstab];
  break;

  /* Orbit flip (with respect to leading stable direction) */
  /*     e.g. 1D unstable manifold */

 case 11:
  for (j = 0; j < ndm; ++j) {
    ret_val += f1[j] * vt[0][blhom_1.nstab - 1][j];
  }
  ret_val *= exp(-par[10] * rr[0][blhom_1.nstab - 1] / 2.);
  break;

  /* Orbit flip (with respect to leading unstable direction) */
  /*     e.g. 1D stable manifold */

 case 12:
  for (j = 0; j < ndm; ++j) {
    ret_val += f0[j] * vt[0][blhom_1.nstab][j];
  }
  ret_val *= exp(par[10] * rr[0][blhom_1.nstab] / 2.);
  break;

  /* Inclination flip (critically twisted) with respect to stable manifold 
*/
/*   e.g. 1D unstable manifold */

 case 13:
  for (i = 0; i < ndm; ++i) {
    ret_val += pu0[ndm + i] * v[0][blhom_1.nstab - 1][i];
  }
  ret_val *= exp(-par[10] * rr[0][blhom_1.nstab - 1] / 2.);
  break;

  /* Inclination flip (critically twisted) with respect to unstable manifold
 */
/*   e.g. 1D stable manifold */

 case 14:
  for (i = 0; i < ndm; ++i) {
    ret_val += pu1[ndm + i] * v[0][blhom_1.nstab][i];
  }
  ret_val *= exp(par[10] * rr[0][blhom_1.nstab] / 2.);
  break;

  /* Non-central homoclinic to saddle-node (in stable manifold) */

 case 15:
  for (i = 0; i < ndm; ++i) {
    ret_val += (par[i + 11] - pu1[i]) * v[0][blhom_1.nstab][i];
  }
  break;

/* Non-central homoclinic to saddle-node (in unstable manifold) */

 case 16:
  for (i = 0; i < ndm; ++i) {
    ret_val += (par[i + 11] - pu0[i]) * v[0][blhom_1.nstab][i];
  }
  break;
  }

  FREE(f0);
  FREE(f1);
  return ret_val;

} /* psiho_ */


/*     ---------- ----- */
/* Subroutine */ int 
eigho(integer isign, integer itrans, doublereal *rr, doublereal *ri, doublereal **vret, const doublereal *xequib, const integer *icp, doublereal *par, integer ndm)
{
  /* System generated locals */
  integer dfdu_dim1, dfdp_dim1;

  /* Local variables */

  doublereal *dfdp, *dfdu;
  doublereal **zz;

  integer i, j, k, ifail;
  doublereal vdot, tmp;

  doublereal *f;

  doublereal **vi, **vr, *fv1;
  integer *iv1;
  static doublereal **vrprev[2] = {NULL, NULL};

  dfdp = (doublereal *)MALLOC(sizeof(doublereal)*ndm*NPARX);
  dfdu = (doublereal *)MALLOC(sizeof(doublereal)*ndm*ndm);
  zz   = DMATRIX(ndm, ndm);

  f     = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
  vi    = DMATRIX(ndm, ndm);
  vr    = DMATRIX(ndm, ndm);
  fv1   = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
  iv1   = (integer *)MALLOC(sizeof(integer)*ndm);

  /* Uses EISPACK routine RG to calculate the eigenvalues/eigenvectors */
  /* of the linearization matrix a (obtained from DFHO) and orders them */
  /* according to their real parts. Simple continuity with respect */
  /* previous call with same value of ITRANS. */

  /* 	input variables */
  /* 		ISIGN  = 1 => left-hand endpoint */
  /*       	       = 2 => right-hand endpoint */
  /*               ITRANS = 1 use transpose of A */
  /*                      = 2 otherwise */

/*       output variables */
/* 		RR,RI real and imaginary parts of eigenvalues, ordered w.r.t */
/* 	           real parts (largest first) */
/* 	        VRET the rows of which are real parts of corresponding */
/*                  eigenvectors */



/* Local */


    /* Parameter adjustments */
    /*--rr;*/
    /*--ri;*/
    /*--xequib;*/
    /*--icp;*/
    /*--par;*/
  dfdp_dim1 = ndm;
  dfdu_dim1 = ndm;
    
  ifail = 0;

  func(ndm, xequib, icp, par, 1, f, dfdu, dfdp);

  if (itrans == 1) {
    for (i = 0; i < ndm; ++i) {
      for (j = 0; j < i; ++j) {
        tmp = ARRAY2D(dfdu, i, j);
	ARRAY2D(dfdu, i, j) = ARRAY2D(dfdu, j, i);
        ARRAY2D(dfdu, j, i) = tmp;
      }
    }
  }

  /* EISPACK call for eigenvalues and eigenvectors */
  rg(ndm, ndm, dfdu, rr, ri, 1, *zz, iv1, fv1, &ifail);

  if (ifail != 0) {
    fprintf(fp9,"EISPACK EIGENVALUE ROUTINE FAILED !\n");	
  }

  for (j = 0; j < ndm; ++j) {
    if (ri[j] > 0.) {
      for (i = 0; i < ndm; ++i) {
	vr[i][j] = zz[j][i];
	vi[i][j] = zz[j + 1][i];
      }
    } else if (ri[j] < 0.) {
      for (i = 0; i < ndm; ++i) {
	vr[i][j] = zz[j - 1][i];
	vi[i][j] = -zz[j][i];
      }
    } else {
      for (i = 0; i < ndm; ++i) {
	vr[i][j] = zz[j][i];
	vi[i][j] = 0.;
      }
    }
  }
  /*Order the eigenvectors/values according size of real part of eigenvalue.
*/
/*     (smallest first) */

  for (i = 0; i < ndm - 1; ++i) {
    for (j = i + 1; j < ndm; ++j) {
      if (rr[i] > rr[j]) {
	tmp = rr[i];
	rr[i] = rr[j];
	rr[j] = tmp;
	tmp = ri[i];
	ri[i] = ri[j];
	ri[j] = tmp;
	for (k = 0; k < ndm; ++k) {
	  tmp = vr[k][i];
	  vr[k][i] = vr[k][j];
	  vr[k][j] = tmp;
	  tmp = vi[k][i];
	  vi[k][i] = vi[k][j];
	  vi[k][j] = tmp;
	}
      }
    }
  }

  /* Choose sign of real part of eigenvectors to be */
  /* commensurate with that of the corresponding eigenvector */
  /* from the previous call with the same value of ISIGN */

  if (vrprev[itrans - 1] == NULL) {
    vrprev[itrans - 1] = DMATRIX(ndm, ndm);      
    for (j = 0; j < ndm; ++j) {
      for (i = 0; i < ndm; ++i) {
	vrprev[itrans - 1][i][j] = vr[i][j];
      }
    }
  }
  for (i = 0; i < ndm; ++i) {
    vdot = 0.;
    for (j = 0; j < ndm; ++j) {
      vdot += vr[j][i] * vrprev[itrans - 1][j][i];
    }
    if (vdot < 0.) {
      for (j = 0; j < ndm; ++j) {
	vr[j][i] = -vr[j][i];
	/*               VI(J,I)=-VI(J,I) */
      }
    }
    for (j = 0; j < ndm; ++j) {
      vrprev[itrans - 1][j][i] = vr[j][i];
    }
  }

  /* Send back the transpose of the matrix of real parts of eigenvectors */
  for (i = 0; i < ndm; ++i) {
    for (j = 0; j < ndm; ++j) {
      vret[i][j] = vr[j][i];
    }
  }

  FREE(f    );
  FREE_DMATRIX(vi);
  FREE_DMATRIX(vr);
  FREE(fv1  );
  FREE(iv1  );
  FREE(dfdp);
  FREE(dfdu);
  FREE_DMATRIX(zz);

  return 0;
} /* eigho_ */


/*     ---------- ------ */
/* Subroutine */ int 
prjctn(doublereal **bound, const doublereal *xequib, const integer *icp, doublereal *par, integer imfd, integer is, integer itrans, integer ndm)
{
  /* System generated locals */
  integer dfdu_dim1 = ndm;

    /* Local variables */
  integer i, j, k;
  integer mcond, k1, k2, m0;

  doublereal *dfdu;
  
  doublereal det, eps;

  doublereal *fdum;
  doublereal **cnow;
  doublereal *a;
  doublereal *v;
    
  static doublereal **cprevs[2][2] = {{NULL, NULL}, {NULL, NULL}};
  doublereal **cprev = cprevs[is - 1][itrans - 1];
  
  /* Compute NUNSTAB (or NSTAB) projection boundary condition functions */
  /*onto to the UNSTABLE (or STABLE) manifold of the appropriate equilibrium
   */

/*    IMFD   = -1 stable eigenspace */
/*           =  1 unstable eigenspace */
/*    ITRANS =  1 use transpose of A */
/*           =  2 otherwise */
/*    IS     =  I (1 or 2) implies use the ith equilibrium in XEQUIB */

/* Use the normalization in Beyn 1990 (4.4) to ensure continuity */
/* w.r.t parameters. */
/* For the purposes of this routine the "previous point on the */
/* branch" is at the values of PAR at which the routine was last */
/* called with the same values of IS and ITRANS. */
  fdum   = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
  dfdu = (doublereal *)MALLOC(sizeof(doublereal)*ndm*ndm);
  func(ndm, xequib, icp, par, 1, fdum, dfdu, NULL);
  FREE(fdum);

  /* Compute transpose of A if ITRANS=1 */
  a = (doublereal *)MALLOC(sizeof(doublereal)*ndm*ndm);
  if (itrans == 1) {
    for (i = 0; i < ndm; ++i) {
      for (j = 0; j < ndm; ++j) {
	a[i + j * ndm] = ARRAY2D(dfdu,j,i);
      }
    }
  } else {
    for (i = 0; i < ndm; ++i) {
      for (j = 0; j < ndm; ++j) {
        a[i + j * ndm] = ARRAY2D(dfdu,i,j);
      }
    }
  }
  FREE(dfdu);

  v = (doublereal *)MALLOC(sizeof(doublereal)*ndm*ndm);
  /* Compute basis V to put A in upper Hessenberg form */
  {
    /* This is here since I don't want to change the calling sequence of the
       BLAS routines. */
    integer tmp = 1;
    doublereal *ort = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
    orthes(&ndm, &ndm, &tmp, &ndm, a, ort);
    ortran(&ndm, &ndm, &tmp, &ndm, a, ort, v);
    FREE(ort);
  }

  /* Force A to be upper Hessenberg */
  if (ndm > 2) {
    for (i = 2; i < ndm; ++i) {
      for (j = 0; j < i - 1; ++j) {
	a[i + j * ndm] = 0.;
      }
    }
  }

  /* Computes basis to put A in "Quasi Upper-Triangular form" */
  /* with the positive (negative) eigenvalues first if IMFD =-1 (=1) */
  eps = HMACHHO;
  {
    /* This is here since I don't want to change the calling sequence of the
       BLAS routines. */
    integer tmp = 1;
    integer *type__ = (integer *)MALLOC(sizeof(integer)*ndm);
    doublereal *ei = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
    doublereal *er = (doublereal *)MALLOC(sizeof(doublereal)*ndm);
    hqr3lc(a, v, &ndm, &tmp, &ndm, &eps, er, ei, type__, &ndm, &ndm, &imfd);
    FREE(type__);
    FREE(ei);
    FREE(er);
  }
  FREE(a);
  
  /* Put the basis in the appropriate part of the matrix CNOW */
  if (imfd == 1) {
    k1 = ndm - blhom_1.nunstab + 1;
    k2 = ndm;
  } else {
    k1 = 1;
    k2 = blhom_1.nstab;
  }
  mcond = k2 - k1 + 1;
  m0 = k1 - 1;

  cnow = DMATRIX(ndm, ndm);
  for (i = k1 - 1; i < k2; ++i) {
    for (j = 0; j < ndm; ++j) {
      cnow[i][j] = v[j + (i - k1 + 1) * ndm];
    }
  }
  FREE(v);

  /* the prjctn_ function uses this array to test if this is
       the first time the prjctn_ function has been called.
       Accordingly, it was initialized to NULLs above. */
    
  /* Set previous matrix to be the present one if this is the first call */

    /* Note by Randy Paffenroth:  There is a slight problem here
       in that this array is used before it is assigned to,
       hence its value is, in general, undefined.  It has
       worked because the just happened to be filled
       with zeros, even though this is not guaranteed.
       Note by Bart Oldeman - it's a static now, and initialized by NULLs */
  if (cprev == NULL) {
    cprev = DMATRIX(ndm, ndm);
    cprevs[is - 1][itrans - 1] = cprev;
    for (i = 0; i < ndm; ++i) {
      for (j = 0; j < ndm; ++j) {
	cprev[i][j] = 0.;
      }
    }
    for (i = k1 - 1; i < k2; ++i) {
      for (j = 0; j < ndm; ++j) {
	cprev[i][j] = cnow[i][j];
	bound[i][j] = cnow[i][j];
      }
    }
  } else {
    doublereal **dum1 = DMATRIX(ndm, ndm);
    doublereal **dum2 = DMATRIX(ndm, ndm);
    doublereal **d = DMATRIX(ndm, ndm);

    /* Calculate the (transpose of the) BEYN matrix D and hence BOUND */
    for (i = 0; i < mcond; ++i) {
      for (j = 0; j < mcond; ++j) {
        dum1[i][j] = 0.;
        dum2[i][j] = 0.;
        for (k = 0; k < ndm; ++k) {
          dum1[i][j] += cprev[i + m0][k] * cnow[j + m0][k];
          dum2[i][j] += cprev[i + m0][k] * cprev[j + m0][k];
        }
      }
    }

    if (mcond > 0) {
      ge(mcond, ndm, *dum1, mcond, ndm, *d, ndm, *dum2, &det);
    }
    FREE_DMATRIX(dum1);
    FREE_DMATRIX(dum2);
    
    for (i = 0; i < mcond; ++i) {
      for (j = 0; j < ndm; ++j) {
        bound[i + m0][j] = 0.;
        for (k = 0; k < mcond; ++k) {
          bound[i + m0][j] += d[k][i] * cnow[k + m0][j];
        }
      }
    }
    FREE_DMATRIX(d);
    
    for (i = k1 - 1; i < k2; ++i) {
      for (j = 0; j < ndm; ++j) {
        cprev[i][j] = bound[i][j];
      }
    }
  }
  FREE_DMATRIX(cnow);
  return 0;
} /* prjctn_ */
