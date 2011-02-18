/* autlib4.f -- translated by f2c (version 19970805).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "auto_f2c.h"
#include "auto_c.h"


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*    Floquet Multiplier Computation (Tom Fairgrieve, U. of Toronto) */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  References: */
/*  T. F. Fairgrieve, PhD Thesis, University of Toronto, 1994. */
/*  T. F. Fairgrieve, A. D. Jepson, O.K. Floquet multipliers, */
/*  SIAM J. Numer. Anal. 28. No. 5, 1991, 1446-1462. */

/*  Please inform Tom Fairgrieve (tff@na.utoronto.ca) of any */
/*  modifications to or errors in these routines. */
/*  Mailing Address: T.F. Fairgrieve, Department of Computer Science, */
/*  University of Toronto, Toronto, Ontario, CANADA  M5S 1A4C */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  Routines included in this file: */

/*  subroutine flowkm : new routine to compute floquet multipliers */
/*  subroutine dhhpr  : compute a Householder matrix */
/*  subroutine dhhap  : appy a Householder matrix */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  Required library routines (included in the file eispack.f) : */

/* subroutine qzhes  : QZ reduction to Hessenberg form             (EISPACK)*/
/* subroutine qzit   : QZ reduction to quasi-upper triangular form (EISPACK)*/
/* subroutine qzval  : QZ calculation of eigenvalues               (EISPACK)*/
/* function   epslon : machine constant routine                    (EISPACK)*/
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* function   dnrm2  : compute l2-norm of a vector                 (BLAS-1)*/
/* function   ddot   : dot product of two vectors                  (BLAS-1)*/
/* subroutine dscal  : scale a vector by a constant                (BLAS-1)*/
/* function   idamax : find index of element with max abs value    (BLAS-1)*/
/* subroutine daxpy  : constant times a vector plus a vector       (BLAS-1)*/
/* subroutine drot   : apply a plane rotation                      (BLAS-1)*/
/* subroutine dswap  : swap two vectors                            (BLAS-1)*/
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* subroutine dgemc  : matrix-matrix copy */
/* subroutine xerbla : BLAS error handling routine                 (BLAS-2)*/
/* function   lsame  : compare character strings                   (BLAS-2)*/
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* subroutine dgemm  : matrix-matrix multiply                      (BLAS-3)*/
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  subroutines ezsvd, ndrotg, ndsvd, prse, sig22, sigmin, sndrtg : */
/*                      Demmel-Kahan svd routines */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */

/* Subroutine */ int 
flowkm(integer ndim, doublereal **c0, doublereal **c1, integer iid, doublecomplex *ev)
{
    

  /* System generated locals */
  integer rwork_dim1;

  /* Local variables */
  doublereal beta, *svde, *svds, svdu[1], *svdv;


  integer i, j;

  doublereal *v, *x;

  logical infev;

  doublereal const__;

  integer ndimm1;
  doublereal nrmc0x, nrmc1x, *qzalfi, *qzbeta;
  integer svdinf;
  doublereal *qzalfr;
  integer qzierr;
  doublereal *svdwrk, qzz[1], *rwork;

  rwork = (doublereal *)MALLOC(sizeof(doublereal)*ndim*ndim);
  svde = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  svds = (doublereal *)MALLOC(sizeof(doublereal)*(ndim+1));
  svdv = (doublereal *)MALLOC(sizeof(doublereal)*ndim*ndim);
  v = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  x = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  qzalfi = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  qzbeta = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  qzalfr = (doublereal *)MALLOC(sizeof(doublereal)*ndim);
  svdwrk = (doublereal *)MALLOC(sizeof(doublereal)*ndim);

  /*  Subroutine to compute Floquet multipliers via the "deflated circuit */
  /*  pencil" method. This routine is called by the AUTO routine FNSPBV */

  /*  storage for SVD computations */

  /*  compute right singular vectors only */

  /*  storage for generalized eigenvalue computations */

  /*      LOGICAL           QZMATZ */
  /*  don't want to accumulate the transforms --- vectors not needed */

  /*  BLAS routines */


  /*  routines from EISPACK */


  /*  own routines */


  /*  Jim Demmel's svd routine  (demmel@nyu.edu) */


  /*  builtin F77 functions */

  /* xx   DOUBLE COMPLEX    DCMPLX */

  /*  Make sure that you have enough local storage. */

  /* Parameter adjustments */
  /*--ev;*/
  rwork_dim1 = ndim;

  /* Change sign of P1 so that we get the sign of the multipliers right. */

  for (j = 0; j < ndim; ++j) {
    for (i = 0; i < ndim; ++i) {
      c1[j][i] = -c1[j][i];
    }
  }

  /*  Print the undeflated circuit pencil (C0, C1). */

  if (iid > 4) {
    fprintf(fp9," Undeflated circuit pencil (C0, C1) \n");	

    fprintf(fp9,"   C0 : \n");	

    for (i = 0; i < ndim; ++i) {
      for (j = 0; j < ndim; ++j) {
	fprintf(fp9," %23.16f",c0[j][i]);	
      }
      fprintf(fp9,"\n");	

    }
    fprintf(fp9,"   C1 : \n");	

    for (i = 0; i < ndim; ++i) {
      for (j = 0; j < ndim; ++j) {
	fprintf(fp9," %23.16f",c1[j][i]);
      }
      fprintf(fp9,"\n");	

    }
  }

  /*  PART I: */
  /*  ======= */

  /*  Deflate the Floquet multiplier at +1.0 so that the deflated */
  /*  circuit pencil is not defective at periodic branch turning points. */

  /* The matrix (C0 - C1) should be (nearly) singular.  Find an approximatio
     n*/
  /*  to the right null vector (call it X).  This will be our approximation 
   */
  /*  to the eigenvector corresponding to the fixed multiplier at +1.0. */

  /*  There are many ways to get this approximation.  We could use */
  /*    1) p'(0) = f(p(0)) */
  /*    2) AUTO'86 routine NLVC applied to C0-C1 */
  /*    3) the right singular vector corresponding to the smallest */
  /*       singular value of C0-C1 */

  /*  I've chosen option 3) because it should introduce as little roundoff 
   */
  /* error as possible.  Although it is more expensive, this is insignifican
     t*/
  /* relative to the rest of the AUTO computations. Also, the SVD does give 
     a*/
  /*  version of the Householder matrix which we would have to compute */
  /* anyways.  But note that it gives V = ( X perp | X ) and not (X | Xperp)
     ,*/
  /* which the Householder routine would give.  This will permute the deflat
     ed*/
  /*  circuit pencil, so that the part to be deflated is in the last column,
   */
  /*  not it the first column, as was shown in the paper. */

  for (j = 0; j < ndim; ++j) {
    for (i = 0; i < ndim; ++i) {
      ARRAY2D(rwork, i, j) = c0[j][i] - c1[j][i];
    }
  }
  {
    /* This is here since I don't want to change the calling sequence of the
       BLAS routines. */
    integer tmp = 1;
    doublereal tmp_tol = 1.0E-16;
    ezsvd(rwork, &ndim, &ndim, &ndim, svds, svde, svdu, &tmp, 
	  svdv, &ndim, svdwrk, &tmp, &svdinf, &tmp_tol);
  }
  if (svdinf != 0) {
    fprintf(fp9," NOTE : Warning from subroutine FLOWKM SVD routine returned SVDINF = %4ld        Floquet multiplier calculations may be wrong\n",svdinf);	

  }

  /*  Apply a Householder matrix (call it H1) based on the null vector */
  /*  to (C0, C1) from the right.  H1 = SVDV = ( Xperp | X ), where X */
  /*  is the null vector. */

  {          
    /* This is here since I don't want to change the calling sequence of the
       BLAS routines. */
    doublereal tmp1 = 1.0;
    doublereal tmp0 = 0.0;
    logical tmp_false = FALSE_;

    dgemm("n", "n", &ndim, &ndim, &ndim, &tmp1, *c0, &ndim, svdv, 
	  &ndim, &tmp0, rwork, &ndim, 1L, 1L);
    dgemc(&ndim, &ndim, rwork, &ndim, *c0, &ndim, &tmp_false);
    dgemm("n", "n", &ndim, &ndim, &ndim, &tmp1, *c1, &ndim, svdv, 
	  &ndim, &tmp0, rwork, &ndim, 1L, 1L);
    dgemc(&ndim, &ndim, rwork, &ndim, *c1, &ndim, &tmp_false);
  }
  /*  Apply a Householder matrix (call it H2) based on */
  /*  (C0*X/||C0*X|| + C1*X/||C1*X||) / 2 */
  /*  to (C0*H1, C1*H1) from the left. */

  {
    /* This is here since I don't want to change the calling sequence of the
       BLAS routines. */
    integer tmp = 1;
    nrmc0x = dnrm2(&ndim, &c0[ndim - 1][0], &tmp);
    nrmc1x = dnrm2(&ndim, &c1[ndim - 1][0], &tmp);
  }
  for (i = 0; i < ndim; ++i) {
    x[i] = (c0[ndim - 1][i] / nrmc0x + c1[ndim - 1][i] / nrmc1x) / 2.;
  }
  dhhpr(1, ndim, ndim, x, 1, &beta, v);
  dhhap(1, ndim, ndim, ndim, &beta, v, LEFT, c0, ndim);
  dhhap(1, ndim, ndim, ndim, &beta, v, LEFT, c1, ndim);

  /* Rescale so that (H2^T)*C0*(H1)(1,NDIM) ~= (H2^T)*C1*(H1)(1,NDIM) ~= 1.0
   */

  /* Computing MAX */
  const__ = max(fabs(c0[ndim - 1][0]),fabs(c1[ndim - 1][0]));
  for (j = 0; j < ndim; ++j) {
    for (i = 0; i < ndim; ++i) {
      c0[j][i] /= const__;
      c1[j][i] /= const__;
    }
  }

  /*  Finished the deflation process! Print the deflated circuit pencil. */

  if (iid > 4) {
    fprintf(fp9," Deflated cicuit pencil (H2^T)*(C0, C1)*(H1) \n");	

    fprintf(fp9,"   (H2^T)*C0*(H1) : \n");	

    for (i = 0; i < ndim; ++i) {
      for (j = 0; j < ndim; ++j) {
	fprintf(fp9," %23.16f",c0[j][i]);
      }
      fprintf(fp9,"\n");	
    }
    fprintf(fp9,"   (H2^T)*C1*(H1) : \n");	

    for (i = 0; i < ndim; ++i) {
      for (j = 0; j < ndim; ++j) {
	fprintf(fp9," %23.16f",c1[j][i]);
      }
      fprintf(fp9,"\n");	

    }
  }

  /*  At this point we have */

  /*     (C0Bar, C1Bar) */
  /* ::= (H2^T)*(C0, C1)*(H1). */

  /*     (( B0^T     | Beta0  )  ( B1^T     | Beta1  ))  1 */
  /*   = (( ----------------- ), ( ----------------- )) */
  /*     (( C0BarDef | Delta0 )  ( C1BarDef | Delta1 )) NDIM-1 */

  /*         NDIM-1      1          NDIM-1      1 */

  /*  and approximations to the Floquet multipliers are */
  /*  (Beta0/Beta1) union the eigenvalues of the deflated pencil */
  /*  (C0BarDef, C1BarDef). */

  /*  PART II: */
  /*  ======== */

  /*  Compute the eigenvalues of the deflated circuit pencil */
  /*  (C0BarDef, C1BarDef) */
  /*  by using the QZ routines from EISPACK. */

  ndimm1 = ndim - 1;

  /*  reduce the generalized eigenvalue problem to a simpler form */
  /*   (C0BarDef,C1BarDef) = (upper hessenberg, upper triangular) */


  qzhes(ndim, ndimm1, &c0[0][1], &c1[0][1], FALSE_ , qzz);

  /*  now reduce to an even simpler form */
  /*   (C0BarDef,C1BarDef) = (quasi-upper triangular, upper triangular) */

  qzit(ndim, ndimm1, &c0[0][1], &c1[0][1], QZEPS1, FALSE_ , qzz, &qzierr);
  if (qzierr != 0) {
    fprintf(fp9," NOTE : Warning from subroutine FLOWKM : QZ routine returned QZIERR = %4ld        Floquet multiplier calculations may be wrong \n",qzierr);	

  }

  /*  compute the generalized eigenvalues */

  qzval(ndim, ndimm1, &c0[0][1], &c1[0][1], qzalfr, qzalfi, 
	qzbeta, FALSE_, qzz);

  /*  Pack the eigenvalues into complex form. */
  ev[0].r = c0[ndim - 1][0] / c1[ndim - 1][0];
  ev[0].i = 0.;
  infev = FALSE_;
  for (j = 0; j < ndimm1; ++j) {
    if (qzbeta[j] != 0.) {
      ev[j + 1].r = qzalfr[j] / qzbeta[j];
      ev[j + 1].i = qzalfi[j] / qzbeta[j];
    } else {
      ev[j + 1].r = 1e30, ev[j + 1].i = 1e30;
      infev = TRUE_;
    }
  }
  if (infev) {
    fprintf(fp9," NOTE : Warning from subroutine FLOWKM : Infinite Floquet multiplier represented by CMPLX( 1.0D+30, 1.0D+30 )\n");	

  }

  FREE(svde); 
  FREE(svds); 
  FREE(svdv); 
  FREE(v); 
  FREE(x); 
  FREE(qzalfi); 
  FREE(qzbeta); 
  FREE(qzalfr); 
  FREE(svdwrk);
  FREE(rwork);

  return 0;

} /* flowkm_ */


/* ************************** */
/* *  Householder routines  * */
/* ************************** */


/*  Subroutines for performing Householder plane rotations. */

/*  DHHPR: for computing Householder transformations and */
/*  DHHAP: for applying them. */

/*  Ref: Golub and van Loan, Matrix Calcualtions, */
/*       First Edition, Pages 38-43 */

/* Subroutine */ int 
dhhpr(integer k, integer j, integer n, doublereal *x, integer incx, doublereal *beta, doublereal *v)
{

  /* Local variables */
  static integer iend, jmkp1;

  static integer i, l;
  static doublereal m, alpha;

  static integer istart;

    


  /*     IMPLICIT UNDEFINED (A-Z,a-z) */
  /*     .. Scalar Arguments .. */
  /*     .. Array Arguments .. */
  /*     .. */

  /*  Purpose */
  /*  ======= */

  /*  DHHPR  computes a Householder Plane Rotation (G&vL Alg. 3.3-1) */
  /*  defined by v and beta. */
  /*  (I - beta v vt) * x is such that x_i = 0 for i=k+1 to j. */

  /*  Parameters */
  /*  ========== */

  /*  K      - INTEGER. */
  /*           On entry, K specifies that the K+1st entry of X */
  /*           be the first to be zeroed. */
  /*           K must be at least one. */
  /*           Unchanged on exit. */

  /*  J      - INTEGER. */
  /*           On entry, J specifies the last entry of X to be zeroed. */
  /*           J must be >= K and <= N. */
  /*           Unchanged on exit. */

  /*  N      - INTEGER. */
  /*           On entry, N specifies the (logical) length of X. */
  /*           Unchanged on exit. */

  /*  X      - DOUBLE PRECISION array of DIMENSION at least */
  /*           ( 1 + ( N - 1 )*abs( INCX ) ). */
  /*           On entry, X specifies the vector to be (partially) zeroed. */
  /*           Unchanged on exit. */

  /*  INCX   - INTEGER. */
  /*           On entry, INCX specifies the increment for the elements of */
  /*           X. INCX must be > zero. If X represents part of a matrix, */
  /*           then use INCX = 1 if a column vector is being zeroed and */
  /*           INCX = NDIM if a row vector is being zeroed. */
  /*           Unchanged on exit. */

  /*  BETA   - DOUBLE PRECISION. */
  /*           BETA specifies the scalar beta. (see pg. 40 of G and v.L.) */

  /*  V      - DOUBLE PRECISION array of DIMENSION at least n. */
  /*           Is updated to be the appropriate Householder vector for */
  /*           the given problem. (Note: space for the implicit zeroes is */
  /*          assumed to be present. Will save on time for index translation
	      .)*/

  /*  -- Written by Tom Fairgrieve, */
  /*                Department of Computer Science, */
  /*                University of Toronto, */
  /*                Toronto, Ontario CANADA  M5S 1A4 */

  /*     .. Local Scalars .. */
  /*     .. External Functions from BLAS .. */
  /*     .. External Subroutines from BLAS .. */
  /*     .. Intrinsic Functions .. */

  /*     .. Executable Statements .. */

  /*  Test the input parameters. */

  /* Parameter adjustments */
  /*--v;*/
  /*--x;*/

    
  if (k < 1 || k > j) {
    fprintf(fp9,"Domain error for K in DHHPR\n");	
    exit(0);
  }
  if (j > n) {
    fprintf(fp9,"Domain error for J in DHHPR\n");	
    exit(0);
  }
  if (incx < 1) {
    fprintf(fp9,"Domain error for INCX in DHHPR\n");	
    exit(0);
  }

  /*  Number of potential non-zero elements in V. */

  jmkp1 = j - k + 1;

  /*  Find M := max{ |x_k|, ... , |x_j| } */

  m = fabs(x[-1 + idamax(&jmkp1, &x[-1 + k], &incx)]);

  /*  alpha := 0 */
  /*  For i = k to j */
  /*      v_i = x_i / m */
  /*      alpha := alpha + v_i^2    (i.e. alpha = vtv) */
  /*  End For */
  /*  alpha :=  sqrt( alpha ) */

  /*  Copy X(K)/M, ... , X(J)/M to V(K), ... , V(J) */

  if (incx == 1) {
    for (i = k - 1; i < j; ++i) {
      v[i] = x[i] / m;
    }
  } else {
    iend = jmkp1 * incx;
    istart = (k - 1) * incx + 1;
    l = k;
    for (i = istart; incx < 0 ? i >= iend : i <= iend; i += incx) 
      {
	v[-1 + l] = x[-1 + i] / m;
	++l;
      }
  }

  /*  Compute alpha */
  {
    /* This is here since I don't want to change the calling sequence of the
       BLAS routines. */
    integer tmp = 1;
    alpha = dnrm2(&jmkp1, &v[-1 + k], &tmp);
  }
  /*  beta := 1/(alpha(alpha + |V_k|)) */

  *beta = 1. / (alpha * (alpha + fabs(v[-1 + k])));

  /*  v_k := v_k + sign(v_k)*alpha */

  v[-1 + k] += d_sign(1.0, v[-1 + k]) * alpha;

  /*  Done ! */

  return 0;

  /*     End of DHHPR. */

} /* dhhpr_ */

/* Subroutine */ int 
dhhap(integer k, integer j, integer n, integer q, doublereal *beta, doublereal *v, integer job, doublereal **a, integer lda)
{
    /* Local variables */

  static integer jmkp1;
  static doublereal s;
  static integer col, row;

    


  /*     IMPLICIT LOGICAL (A-Z) */
  /*     .. Scalar Arguments .. */
  /*     .. Array Arguments .. */
  /*     .. */

  /*  Purpose */
  /*  ======= */

  /*  DHHAP applies a Householder Plane Rotation defined by v and beta */
  /*  to the matrix A.  If JOB = 1 then A := (I - beta*v*vt)A and if */
  /*  JOB = 2 then A := A(I - beta*v*vt). (See Golub and van Loan */
  /*  Alg. 3.3-2.) */

  /*  Parameters */
  /*  ========== */

  /*  K      - INTEGER. */
  /*           On entry, K specifies that the V(K) may be the first */
  /*           non-zero entry of V. */
  /*           K must be at least one. */
  /*           Unchanged on exit. */

/*  J      - INTEGER. */
/*           On entry, J specifies the last non-zero entry of V. */
/*           J must be >= K and <= N. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the row dimension of A. */
/*           Unchanged on exit. */

/*  Q      - INTEGER. */
/*           On entry, Q specifies the column dimension of A. */
/*           Unchanged on exit. */

/*  BETA   - DOUBLE PRECISION. */
/*           BETA specifies the scalar beta. (see pg. 40 of G and v.L.) */
/*           Unchanged on exit. */

/*  V      - DOUBLE PRECISION array of DIMENSION at least n. */
/*           Householder vector v. */
/*           Unchanged on exit. */

/*  JOB    - INTEGER. */
/*          On entry, JOB specifies the order of the Householder applicati
on.*/
/*           If JOB = 1 then A := (I - beta*v*vt)A and if JOB = 2 then */
/*           A := A(I - beta*v*vt) */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( LDA, Q ). */
/*           On entry, A specifies the matrix to be transformed. */
/*           On exit, A specifies the transformed matrix. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the declared leading dimension of A. 
*/
/*           Unchanged on exit. */

/*  -- Written by Tom Fairgrieve, */
/*                Department of Computer Science, */
/*                University of Toronto, */
/*                Toronto, Ontario CANADA  M5S 1A4 */

/*     .. Local Scalars .. */
/*     .. External Functions from BLAS .. */

/*     .. Executable Statements .. */

/*  Test the input parameters. */

    /* Parameter adjustments */
    /*--v;*/
  if (job != 1 && job != 2) {
    fprintf(fp9,"Domain error for JOB in DHHAP\n");	
    exit(0);
  }
  if (k < 1 || k > j) {
    fprintf(fp9,"Domain error for K in DHHAP\n");	
    exit(0);
  }
  if (job == 1) {
    if (j > n) {
      fprintf(fp9,"Domain error for J in DHHAP\n");	
      exit(0);
    }
  } else {
    if (j > q) {
      fprintf(fp9,"Domain error for J in DHHAP\n");	
      exit(0);
    }
  }

  /*  Minimum {row,col} dimension of update. */

  jmkp1 = j - k + 1;

  /*  If (JOB = 1) then */
  /*      For p = 1, ... , q */
  /*          s := beta*(v_k*a_k,p + ... + v_j*a_j,p) */
  /*          For i = k, ..., j */
  /*              a_i,p := a_i,p - s*v_i */
  /*          End For */
  /*      End For */
  /*  Else % JOB=2 */
  /*      For p = 1, ... , n */
  /*          s := beta*(v_k*a_p,k + ... + v_j*a_p,j) */
  /*          For i = k, ..., j */
  /*              a_p,i := a_p,i - s*v_i */
  /*          End For */
  /*      End For */
  /*  End If */

  if (job == 1) {
    for (col = 0; col < q; ++col) {
      {
	/* This is here since I don't want to change the calling sequence of the
	   BLAS routines. */
	integer tmp = 1;
	s = *beta * ddot(&jmkp1, &v[-1 + k], &tmp, &a[col][-1 + k], &tmp);
      }
      for (row = k - 1; row < j; ++row) {
	a[col][row] -= s * v[row];
      }
    }
  } else {
    for (row = 0; row < n; ++row) {
      {
	/* This is here since I don't want to change the calling sequence of the
	   BLAS routines. */
	integer tmp = 1;
	s = *beta * ddot(&jmkp1, &v[-1 + k], &tmp, &a[k - 1][row], &lda);
      }
      for (col = k - 1; col < j; ++col) {
	a[col][row] -= s * v[col];
      }
    }
  }
    
  /*  Done ! */

  return 0;

  /*     End of DHHAP. */

} /* dhhap_ */













