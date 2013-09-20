/* eispack.f -- translated by f2c (version 19970805).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "auto_f2c.h"
#include "math.h"
#include "auto_c.h"

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*          Eigenvalue solver from EISPACK */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
rg(integer nm, integer n, doublereal *a, doublereal *wr, doublereal *wi, integer matz, doublereal *z__, integer *iv1, doublereal *fv1, integer *ierr)
{
    /* System generated locals */
    integer a_dim1, a_offset, z_dim1, z_offset;

    /* Local variables */

    static integer is1, is2;




/*     THIS SUBROUTINE CALLS THE RECOMMENDED SEQUENCE OF */
/*     SUBROUTINES FROM THE EIGENSYSTEM SUBROUTINE PACKAGE (EISPACK) */
/*     TO FIND THE EIGENVALUES AND EIGENVECTORS (IF DESIRED) */
/*     OF A REAL GENERAL MATRIX. */

/*     ON INPUT */

/*        NM  MUST BE SET TO THE ROW DIMENSION OF THE TWO-DIMENSIONAL */
/*        DIMENSION STATEMENT. */

/*        N  IS THE ORDER OF THE MATRIX  A. */

/*        A  CONTAINS THE REAL GENERAL MATRIX. */

/*        MATZ  IS AN INTEGER VARIABLE SET EQUAL TO ZERO IF */
/*        ONLY EIGENVALUES ARE DESIRED.  OTHERWISE IT IS SET TO */
/*        ANY NON-ZERO INTEGER FOR BOTH EIGENVALUES AND EIGENVECTORS. */

/*     ON OUTPUT */

/*        WR  AND  WI  CONTAIN THE REAL AND IMAGINARY PARTS, */
/*        RESPECTIVELY, OF THE EIGENVALUES.  COMPLEX CONJUGATE */
/*        PAIRS OF EIGENVALUES APPEAR CONSECUTIVELY WITH THE */
/*        EIGENVALUE HAVING THE POSITIVE IMAGINARY PART FIRST. */

/*        Z  CONTAINS THE REAL AND IMAGINARY PARTS OF THE EIGENVECTORS */
/*        IF MATZ IS NOT ZERO.  IF THE J-TH EIGENVALUE IS REAL, THE */
/*        J-TH COLUMN OF  Z  CONTAINS ITS EIGENVECTOR.  IF THE J-TH */
/*        EIGENVALUE IS COMPLEX WITH POSITIVE IMAGINARY PART, THE */
/*        J-TH AND (J+1)-TH COLUMNS OF  Z  CONTAIN THE REAL AND */
/*        IMAGINARY PARTS OF ITS EIGENVECTOR.  THE CONJUGATE OF THIS */
/*        VECTOR IS THE EIGENVECTOR FOR THE CONJUGATE EIGENVALUE. */

/*        IERR  IS AN INTEGER OUTPUT VARIABLE SET EQUAL TO AN ERROR */
/*           COMPLETION CODE DESCRIBED IN THE DOCUMENTATION FOR HQR */
/*           AND HQR2.  THE NORMAL COMPLETION CODE IS ZERO. */

/*        IV1  AND  FV1  ARE TEMPORARY STORAGE ARRAYS. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    --fv1;
    --iv1;
    z_dim1 = nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;
    --wi;
    --wr;
    a_dim1 = nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    /* Function Body */
    if (n <= nm) {
	goto L10;
    }
    *ierr = n * 10;
    goto L50;

L10:
    balanc(&nm, &n, &a[a_offset], &is1, &is2, &fv1[1]);
    elmhes(&nm, &n, &is1, &is2, &a[a_offset], &iv1[1]);
    if (matz != 0) {
	goto L20;
    }
/*     .......... FIND EIGENVALUES ONLY .......... */
    hqr(&nm, &n, &is1, &is2, &a[a_offset], &wr[1], &wi[1], ierr);
    goto L50;
/*     .......... FIND BOTH EIGENVALUES AND EIGENVECTORS .......... */
L20:
    eltran(&nm, &n, &is1, &is2, &a[a_offset], &iv1[1], &z__[z_offset]);
    hqr2(&nm, &n, &is1, &is2, &a[a_offset], &wr[1], &wi[1], &z__[z_offset], 
	    ierr);
    if (*ierr != 0) {
	goto L50;
    }
    balbak(&nm, &n, &is1, &is2, &fv1[1], &n, &z__[z_offset]);
L50:
    return 0;
} /* rg_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
hqr(integer *nm, integer *n, integer *low, integer *igh, doublereal *h__, doublereal *wr, doublereal *wi, integer *ierr)
{
    /* System generated locals */
    integer h_dim1, h_offset, i__1, i__2, i__3;
    doublereal d__1, d__2;

    /* Local variables */
    static doublereal norm;
    static integer i__, j, k, l, m;
    static doublereal p, q, r__, s, t, w, x, y;
    static integer na, en, ll, mm;
    static doublereal zz;
    static logical notlas;
    static integer mp2, itn, its, enm2;
    static doublereal tst1, tst2;



/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE HQR, */
/*     NUM. MATH. 14, 219-231(1970) BY MARTIN, PETERS, AND WILKINSON. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 359-371(1971). */

/*     THIS SUBROUTINE FINDS THE EIGENVALUES OF A REAL */
/*     UPPER HESSENBERG MATRIX BY THE QR METHOD. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRIX. */

/*        LOW AND IGH ARE INTEGERS DETERMINED BY THE BALANCING */
/*          SUBROUTINE  BALANC.  IF  BALANC  HAS NOT BEEN USED, */
/*          SET LOW=1, IGH=N. */

/*        H CONTAINS THE UPPER HESSENBERG MATRIX.  INFORMATION ABOUT */
/*          THE TRANSFORMATIONS USED IN THE REDUCTION TO HESSENBERG */
/*          FORM BY  ELMHES  OR  ORTHES, IF PERFORMED, IS STORED */
/*          IN THE REMAINING TRIANGLE UNDER THE HESSENBERG MATRIX. */

/*     ON OUTPUT */

/*        H HAS BEEN DESTROYED.  THEREFORE, IT MUST BE SAVED */
/*          BEFORE CALLING  HQR  IF SUBSEQUENT CALCULATION AND */
/*          BACK TRANSFORMATION OF EIGENVECTORS IS TO BE PERFORMED. */

/*        WR AND WI CONTAIN THE REAL AND IMAGINARY PARTS, */
/*          RESPECTIVELY, OF THE EIGENVALUES.  THE EIGENVALUES */
/*          ARE UNORDERED EXCEPT THAT COMPLEX CONJUGATE PAIRS */
/*          OF VALUES APPEAR CONSECUTIVELY WITH THE EIGENVALUE */
/*          HAVING THE POSITIVE IMAGINARY PART FIRST.  IF AN */
/*          ERROR EXIT IS MADE, THE EIGENVALUES SHOULD BE CORRECT */
/*          FOR INDICES IERR+1,...,N. */

/*        IERR IS SET TO */
/*          ZERO       FOR NORMAL RETURN, */
/*          J          IF THE LIMIT OF 30*N ITERATIONS IS EXHAUSTED */
/*                     WHILE THE J-TH EIGENVALUE IS BEING SOUGHT. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    --wi;
    --wr;
    h_dim1 = *nm;
    h_offset = h_dim1 + 1;
    h__ -= h_offset;

    /* Function Body */
    *ierr = 0;
    norm = 0.;
    k = 1;
/*     .......... STORE ROOTS ISOLATED BY BALANC */
/*                AND COMPUTE MATRIX NORM .......... */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {

	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
/* L40: */
	    norm += (d__1 = h__[i__ + j * h_dim1], fabs(d__1));
	}

	k = i__;
	if (i__ >= *low && i__ <= *igh) {
	    goto L50;
	}
	wr[i__] = h__[i__ + i__ * h_dim1];
	wi[i__] = 0.;
L50:
	;
    }

    en = *igh;
    t = 0.;
    itn = *n * 30;
/*     .......... SEARCH FOR NEXT EIGENVALUES .......... */
L60:
    if (en < *low) {
	goto L1001;
    }
    its = 0;
    na = en - 1;
    enm2 = na - 1;
/*     .......... LOOK FOR SINGLE SMALL SUB-DIAGONAL ELEMENT */
/*                FOR L=EN STEP -1 UNTIL LOW DO -- .......... */
L70:
    i__1 = en;
    for (ll = *low; ll <= i__1; ++ll) {
	l = en + *low - ll;
	if (l == *low) {
	    goto L100;
	}
	s = (d__1 = h__[l - 1 + (l - 1) * h_dim1], fabs(d__1)) + (d__2 = h__[l 
		+ l * h_dim1], fabs(d__2));
	if (s == 0.) {
	    s = norm;
	}
	tst1 = s;
	tst2 = tst1 + (d__1 = h__[l + (l - 1) * h_dim1], fabs(d__1));
	if (tst2 == tst1) {
	    goto L100;
	}
/* L80: */
    }
/*     .......... FORM SHIFT .......... */
L100:
    x = h__[en + en * h_dim1];
    if (l == en) {
	goto L270;
    }
    y = h__[na + na * h_dim1];
    w = h__[en + na * h_dim1] * h__[na + en * h_dim1];
    if (l == na) {
	goto L280;
    }
    if (itn == 0) {
	goto L1000;
    }
    if (its != 10 && its != 20) {
	goto L130;
    }
/*     .......... FORM EXCEPTIONAL SHIFT .......... */
    t += x;

    i__1 = en;
    for (i__ = *low; i__ <= i__1; ++i__) {
/* L120: */
	h__[i__ + i__ * h_dim1] -= x;
    }

    s = (d__1 = h__[en + na * h_dim1], fabs(d__1)) + (d__2 = h__[na + enm2 * 
	    h_dim1], fabs(d__2));
    x = s * .75;
    y = x;
    w = s * -.4375 * s;
L130:
    ++its;
    --itn;
/*     .......... LOOK FOR TWO CONSECUTIVE SMALL */
/*                SUB-DIAGONAL ELEMENTS. */
/*                FOR M=EN-2 STEP -1 UNTIL L DO -- .......... */
    i__1 = enm2;
    for (mm = l; mm <= i__1; ++mm) {
	m = enm2 + l - mm;
	zz = h__[m + m * h_dim1];
	r__ = x - zz;
	s = y - zz;
	p = (r__ * s - w) / h__[m + 1 + m * h_dim1] + h__[m + (m + 1) * 
		h_dim1];
	q = h__[m + 1 + (m + 1) * h_dim1] - zz - r__ - s;
	r__ = h__[m + 2 + (m + 1) * h_dim1];
	s = fabs(p) + fabs(q) + fabs(r__);
	p /= s;
	q /= s;
	r__ /= s;
	if (m == l) {
	    goto L150;
	}
	tst1 = fabs(p) * ((d__1 = h__[m - 1 + (m - 1) * h_dim1], fabs(d__1)) + 
		fabs(zz) + (d__2 = h__[m + 1 + (m + 1) * h_dim1], fabs(d__2)));
	tst2 = tst1 + (d__1 = h__[m + (m - 1) * h_dim1], fabs(d__1)) * (fabs(q) 
		+ fabs(r__));
	if (tst2 == tst1) {
	    goto L150;
	}
/* L140: */
    }

L150:
    mp2 = m + 2;

    i__1 = en;
    for (i__ = mp2; i__ <= i__1; ++i__) {
	h__[i__ + (i__ - 2) * h_dim1] = 0.;
	if (i__ == mp2) {
	    goto L160;
	}
	h__[i__ + (i__ - 3) * h_dim1] = 0.;
L160:
	;
    }
/*     .......... DOUBLE QR STEP INVOLVING ROWS L TO EN AND */
/*                COLUMNS M TO EN .......... */
    i__1 = na;
    for (k = m; k <= i__1; ++k) {
	notlas = k != na;
	if (k == m) {
	    goto L170;
	}
	p = h__[k + (k - 1) * h_dim1];
	q = h__[k + 1 + (k - 1) * h_dim1];
	r__ = 0.;
	if (notlas) {
	    r__ = h__[k + 2 + (k - 1) * h_dim1];
	}
	x = fabs(p) + fabs(q) + fabs(r__);
	if (x == 0.) {
	    goto L260;
	}
	p /= x;
	q /= x;
	r__ /= x;
L170:
	d__1 = sqrt(p * p + q * q + r__ * r__);
	s = d_sign(d__1, p);
	if (k == m) {
	    goto L180;
	}
	h__[k + (k - 1) * h_dim1] = -s * x;
	goto L190;
L180:
	if (l != m) {
	    h__[k + (k - 1) * h_dim1] = -h__[k + (k - 1) * h_dim1];
	}
L190:
	p += s;
	x = p / s;
	y = q / s;
	zz = r__ / s;
	q /= p;
	r__ /= p;
	if (notlas) {
	    goto L225;
	}
/*     .......... ROW MODIFICATION .......... */
	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
	    p = h__[k + j * h_dim1] + q * h__[k + 1 + j * h_dim1];
	    h__[k + j * h_dim1] -= p * x;
	    h__[k + 1 + j * h_dim1] -= p * y;
/* L200: */
	}

/* Computing MIN */
	i__2 = en, i__3 = k + 3;
	j = min(i__2,i__3);
/*     .......... COLUMN MODIFICATION .......... */
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    p = x * h__[i__ + k * h_dim1] + y * h__[i__ + (k + 1) * h_dim1];
	    h__[i__ + k * h_dim1] -= p;
	    h__[i__ + (k + 1) * h_dim1] -= p * q;
/* L210: */
	}
	goto L255;
L225:
/*     .......... ROW MODIFICATION .......... */
	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
	    p = h__[k + j * h_dim1] + q * h__[k + 1 + j * h_dim1] + r__ * h__[
		    k + 2 + j * h_dim1];
	    h__[k + j * h_dim1] -= p * x;
	    h__[k + 1 + j * h_dim1] -= p * y;
	    h__[k + 2 + j * h_dim1] -= p * zz;
/* L230: */
	}

/* Computing MIN */
	i__2 = en, i__3 = k + 3;
	j = min(i__2,i__3);
/*     .......... COLUMN MODIFICATION .......... */
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    p = x * h__[i__ + k * h_dim1] + y * h__[i__ + (k + 1) * h_dim1] + 
		    zz * h__[i__ + (k + 2) * h_dim1];
	    h__[i__ + k * h_dim1] -= p;
	    h__[i__ + (k + 1) * h_dim1] -= p * q;
	    h__[i__ + (k + 2) * h_dim1] -= p * r__;
/* L240: */
	}
L255:

L260:
	;
    }

    goto L70;
/*     .......... ONE ROOT FOUND .......... */
L270:
    wr[en] = x + t;
    wi[en] = 0.;
    en = na;
    goto L60;
/*     .......... TWO ROOTS FOUND .......... */
L280:
    p = (y - x) / 2.;
    q = p * p + w;
    zz = sqrt((fabs(q)));
    x += t;
    if (q < 0.) {
	goto L320;
    }
/*     .......... REAL PAIR .......... */
    zz = p + d_sign(zz, p);
    wr[na] = x + zz;
    wr[en] = wr[na];
    if (zz != 0.) {
	wr[en] = x - w / zz;
    }
    wi[na] = 0.;
    wi[en] = 0.;
    goto L330;
/*     .......... COMPLEX PAIR .......... */
L320:
    wr[na] = x + p;
    wr[en] = x + p;
    wi[na] = zz;
    wi[en] = -zz;
L330:
    en = enm2;
    goto L60;
/*     .......... SET ERROR -- ALL EIGENVALUES HAVE NOT */
/*                CONVERGED AFTER 30*N ITERATIONS .......... */
L1000:
    *ierr = en;
L1001:
    return 0;
} /* hqr_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
hqr2(integer *nm, integer *n, integer *low, integer *igh, doublereal *h__, doublereal *wr, doublereal *wi, doublereal *z__, integer *ierr)
{
    /* System generated locals */
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3;
    doublereal d__1, d__2, d__3, d__4;

    /* Local variables */

    static doublereal norm;
    static integer i__, j, k, l, m;
    static doublereal p, q, r__, s, t, w, x, y;
    static integer na, ii, en, jj;
    static doublereal ra, sa;
    static integer ll, mm, nn;
    static doublereal vi, vr, zz;
    static logical notlas;
    static integer mp2, itn, its, enm2;
    static doublereal tst1, tst2;



/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE HQR2, */
/*     NUM. MATH. 16, 181-204(1970) BY PETERS AND WILKINSON. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 372-395(1971). */

/*     THIS SUBROUTINE FINDS THE EIGENVALUES AND EIGENVECTORS */
/*     OF A REAL UPPER HESSENBERG MATRIX BY THE QR METHOD.  THE */
/*     EIGENVECTORS OF A REAL GENERAL MATRIX CAN ALSO BE FOUND */
/*     IF  ELMHES  AND  ELTRAN  OR  ORTHES  AND  ORTRAN  HAVE */
/*     BEEN USED TO REDUCE THIS GENERAL MATRIX TO HESSENBERG FORM */
/*     AND TO ACCUMULATE THE SIMILARITY TRANSFORMATIONS. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRIX. */

/*        LOW AND IGH ARE INTEGERS DETERMINED BY THE BALANCING */
/*          SUBROUTINE  BALANC.  IF  BALANC  HAS NOT BEEN USED, */
/*          SET LOW=1, IGH=N. */

/*        H CONTAINS THE UPPER HESSENBERG MATRIX. */

/*        Z CONTAINS THE TRANSFORMATION MATRIX PRODUCED BY  ELTRAN */
/*          AFTER THE REDUCTION BY  ELMHES, OR BY  ORTRAN  AFTER THE */
/*          REDUCTION BY  ORTHES, IF PERFORMED.  IF THE EIGENVECTORS */
/*          OF THE HESSENBERG MATRIX ARE DESIRED, Z MUST CONTAIN THE */
/*          IDENTITY MATRIX. */

/*     ON OUTPUT */

/*        H HAS BEEN DESTROYED. */

/*        WR AND WI CONTAIN THE REAL AND IMAGINARY PARTS, */
/*          RESPECTIVELY, OF THE EIGENVALUES.  THE EIGENVALUES */
/*          ARE UNORDERED EXCEPT THAT COMPLEX CONJUGATE PAIRS */
/*          OF VALUES APPEAR CONSECUTIVELY WITH THE EIGENVALUE */
/*          HAVING THE POSITIVE IMAGINARY PART FIRST.  IF AN */
/*          ERROR EXIT IS MADE, THE EIGENVALUES SHOULD BE CORRECT */
/*          FOR INDICES IERR+1,...,N. */

/*        Z CONTAINS THE REAL AND IMAGINARY PARTS OF THE EIGENVECTORS. */
/*          IF THE I-TH EIGENVALUE IS REAL, THE I-TH COLUMN OF Z */
/*          CONTAINS ITS EIGENVECTOR.  IF THE I-TH EIGENVALUE IS COMPLEX 
*/
/*          WITH POSITIVE IMAGINARY PART, THE I-TH AND (I+1)-TH */
/*          COLUMNS OF Z CONTAIN THE REAL AND IMAGINARY PARTS OF ITS */
/*          EIGENVECTOR.  THE EIGENVECTORS ARE UNNORMALIZED.  IF AN */
/*          ERROR EXIT IS MADE, NONE OF THE EIGENVECTORS HAS BEEN FOUND. 
*/

/*        IERR IS SET TO */
/*          ZERO       FOR NORMAL RETURN, */
/*          J          IF THE LIMIT OF 30*N ITERATIONS IS EXHAUSTED */
/*                     WHILE THE J-TH EIGENVALUE IS BEING SOUGHT. */

/*     CALLS CDIV FOR COMPLEX DIVISION. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    z_dim1 = *nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;
    --wi;
    --wr;
    h_dim1 = *nm;
    h_offset = h_dim1 + 1;
    h__ -= h_offset;

    /* Function Body */
    *ierr = 0;
    norm = 0.;
    k = 1;
/*     .......... STORE ROOTS ISOLATED BY BALANC */
/*                AND COMPUTE MATRIX NORM .......... */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {

	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
/* L40: */
	    norm += (d__1 = h__[i__ + j * h_dim1], fabs(d__1));
	}

	k = i__;
	if (i__ >= *low && i__ <= *igh) {
	    goto L50;
	}
	wr[i__] = h__[i__ + i__ * h_dim1];
	wi[i__] = 0.;
L50:
	;
    }

    en = *igh;
    t = 0.;
    itn = *n * 30;
/*     .......... SEARCH FOR NEXT EIGENVALUES .......... */
L60:
    if (en < *low) {
	goto L340;
    }
    its = 0;
    na = en - 1;
    enm2 = na - 1;
/*     .......... LOOK FOR SINGLE SMALL SUB-DIAGONAL ELEMENT */
/*                FOR L=EN STEP -1 UNTIL LOW DO -- .......... */
L70:
    i__1 = en;
    for (ll = *low; ll <= i__1; ++ll) {
	l = en + *low - ll;
	if (l == *low) {
	    goto L100;
	}
	s = (d__1 = h__[l - 1 + (l - 1) * h_dim1], fabs(d__1)) + (d__2 = h__[l 
		+ l * h_dim1], fabs(d__2));
	if (s == 0.) {
	    s = norm;
	}
	tst1 = s;
	tst2 = tst1 + (d__1 = h__[l + (l - 1) * h_dim1], fabs(d__1));
	if (tst2 == tst1) {
	    goto L100;
	}
/* L80: */
    }
/*     .......... FORM SHIFT .......... */
L100:
    x = h__[en + en * h_dim1];
    if (l == en) {
	goto L270;
    }
    y = h__[na + na * h_dim1];
    w = h__[en + na * h_dim1] * h__[na + en * h_dim1];
    if (l == na) {
	goto L280;
    }
    if (itn == 0) {
	goto L1000;
    }
    if (its != 10 && its != 20) {
	goto L130;
    }
/*     .......... FORM EXCEPTIONAL SHIFT .......... */
    t += x;

    i__1 = en;
    for (i__ = *low; i__ <= i__1; ++i__) {
/* L120: */
	h__[i__ + i__ * h_dim1] -= x;
    }

    s = (d__1 = h__[en + na * h_dim1], fabs(d__1)) + (d__2 = h__[na + enm2 * 
	    h_dim1], fabs(d__2));
    x = s * .75;
    y = x;
    w = s * -.4375 * s;
L130:
    ++its;
    --itn;
/*     .......... LOOK FOR TWO CONSECUTIVE SMALL */
/*                SUB-DIAGONAL ELEMENTS. */
/*                FOR M=EN-2 STEP -1 UNTIL L DO -- .......... */
    i__1 = enm2;
    for (mm = l; mm <= i__1; ++mm) {
	m = enm2 + l - mm;
	zz = h__[m + m * h_dim1];
	r__ = x - zz;
	s = y - zz;
	p = (r__ * s - w) / h__[m + 1 + m * h_dim1] + h__[m + (m + 1) * 
		h_dim1];
	q = h__[m + 1 + (m + 1) * h_dim1] - zz - r__ - s;
	r__ = h__[m + 2 + (m + 1) * h_dim1];
	s = fabs(p) + fabs(q) + fabs(r__);
	p /= s;
	q /= s;
	r__ /= s;
	if (m == l) {
	    goto L150;
	}
	tst1 = fabs(p) * ((d__1 = h__[m - 1 + (m - 1) * h_dim1], fabs(d__1)) + 
		fabs(zz) + (d__2 = h__[m + 1 + (m + 1) * h_dim1], fabs(d__2)));
	tst2 = tst1 + (d__1 = h__[m + (m - 1) * h_dim1], fabs(d__1)) * (fabs(q) 
		+ fabs(r__));
	if (tst2 == tst1) {
	    goto L150;
	}
/* L140: */
    }

L150:
    mp2 = m + 2;

    i__1 = en;
    for (i__ = mp2; i__ <= i__1; ++i__) {
	h__[i__ + (i__ - 2) * h_dim1] = 0.;
	if (i__ == mp2) {
	    goto L160;
	}
	h__[i__ + (i__ - 3) * h_dim1] = 0.;
L160:
	;
    }
/*     .......... DOUBLE QR STEP INVOLVING ROWS L TO EN AND */
/*                COLUMNS M TO EN .......... */
    i__1 = na;
    for (k = m; k <= i__1; ++k) {
	notlas = k != na;
	if (k == m) {
	    goto L170;
	}
	p = h__[k + (k - 1) * h_dim1];
	q = h__[k + 1 + (k - 1) * h_dim1];
	r__ = 0.;
	if (notlas) {
	    r__ = h__[k + 2 + (k - 1) * h_dim1];
	}
	x = fabs(p) + fabs(q) + fabs(r__);
	if (x == 0.) {
	    goto L260;
	}
	p /= x;
	q /= x;
	r__ /= x;
L170:
	d__1 = sqrt(p * p + q * q + r__ * r__);
	s = d_sign(d__1, p);
	if (k == m) {
	    goto L180;
	}
	h__[k + (k - 1) * h_dim1] = -s * x;
	goto L190;
L180:
	if (l != m) {
	    h__[k + (k - 1) * h_dim1] = -h__[k + (k - 1) * h_dim1];
	}
L190:
	p += s;
	x = p / s;
	y = q / s;
	zz = r__ / s;
	q /= p;
	r__ /= p;
	if (notlas) {
	    goto L225;
	}
/*     .......... ROW MODIFICATION .......... */
	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
	    p = h__[k + j * h_dim1] + q * h__[k + 1 + j * h_dim1];
	    h__[k + j * h_dim1] -= p * x;
	    h__[k + 1 + j * h_dim1] -= p * y;
/* L200: */
	}

/* Computing MIN */
	i__2 = en, i__3 = k + 3;
	j = min(i__2,i__3);
/*     .......... COLUMN MODIFICATION .......... */
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    p = x * h__[i__ + k * h_dim1] + y * h__[i__ + (k + 1) * h_dim1];
	    h__[i__ + k * h_dim1] -= p;
	    h__[i__ + (k + 1) * h_dim1] -= p * q;
/* L210: */
	}
/*     .......... ACCUMULATE TRANSFORMATIONS .......... */
	i__2 = *igh;
	for (i__ = *low; i__ <= i__2; ++i__) {
	    p = x * z__[i__ + k * z_dim1] + y * z__[i__ + (k + 1) * z_dim1];
	    z__[i__ + k * z_dim1] -= p;
	    z__[i__ + (k + 1) * z_dim1] -= p * q;
/* L220: */
	}
	goto L255;
L225:
/*     .......... ROW MODIFICATION .......... */
	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
	    p = h__[k + j * h_dim1] + q * h__[k + 1 + j * h_dim1] + r__ * h__[
		    k + 2 + j * h_dim1];
	    h__[k + j * h_dim1] -= p * x;
	    h__[k + 1 + j * h_dim1] -= p * y;
	    h__[k + 2 + j * h_dim1] -= p * zz;
/* L230: */
	}

/* Computing MIN */
	i__2 = en, i__3 = k + 3;
	j = min(i__2,i__3);
/*     .......... COLUMN MODIFICATION .......... */
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    p = x * h__[i__ + k * h_dim1] + y * h__[i__ + (k + 1) * h_dim1] + 
		    zz * h__[i__ + (k + 2) * h_dim1];
	    h__[i__ + k * h_dim1] -= p;
	    h__[i__ + (k + 1) * h_dim1] -= p * q;
	    h__[i__ + (k + 2) * h_dim1] -= p * r__;
/* L240: */
	}
/*     .......... ACCUMULATE TRANSFORMATIONS .......... */
	i__2 = *igh;
	for (i__ = *low; i__ <= i__2; ++i__) {
	    p = x * z__[i__ + k * z_dim1] + y * z__[i__ + (k + 1) * z_dim1] + 
		    zz * z__[i__ + (k + 2) * z_dim1];
	    z__[i__ + k * z_dim1] -= p;
	    z__[i__ + (k + 1) * z_dim1] -= p * q;
	    z__[i__ + (k + 2) * z_dim1] -= p * r__;
/* L250: */
	}
L255:

L260:
	;
    }

    goto L70;
/*     .......... ONE ROOT FOUND .......... */
L270:
    h__[en + en * h_dim1] = x + t;
    wr[en] = h__[en + en * h_dim1];
    wi[en] = 0.;
    en = na;
    goto L60;
/*     .......... TWO ROOTS FOUND .......... */
L280:
    p = (y - x) / 2.;
    q = p * p + w;
    zz = sqrt((fabs(q)));
    h__[en + en * h_dim1] = x + t;
    x = h__[en + en * h_dim1];
    h__[na + na * h_dim1] = y + t;
    if (q < 0.) {
	goto L320;
    }
/*     .......... REAL PAIR .......... */
    zz = p + d_sign(zz, p);
    wr[na] = x + zz;
    wr[en] = wr[na];
    if (zz != 0.) {
	wr[en] = x - w / zz;
    }
    wi[na] = 0.;
    wi[en] = 0.;
    x = h__[en + na * h_dim1];
    s = fabs(x) + fabs(zz);
    p = x / s;
    q = zz / s;
    r__ = sqrt(p * p + q * q);
    p /= r__;
    q /= r__;
/*     .......... ROW MODIFICATION .......... */
    i__1 = *n;
    for (j = na; j <= i__1; ++j) {
	zz = h__[na + j * h_dim1];
	h__[na + j * h_dim1] = q * zz + p * h__[en + j * h_dim1];
	h__[en + j * h_dim1] = q * h__[en + j * h_dim1] - p * zz;
/* L290: */
    }
/*     .......... COLUMN MODIFICATION .......... */
    i__1 = en;
    for (i__ = 1; i__ <= i__1; ++i__) {
	zz = h__[i__ + na * h_dim1];
	h__[i__ + na * h_dim1] = q * zz + p * h__[i__ + en * h_dim1];
	h__[i__ + en * h_dim1] = q * h__[i__ + en * h_dim1] - p * zz;
/* L300: */
    }
/*     .......... ACCUMULATE TRANSFORMATIONS .......... */
    i__1 = *igh;
    for (i__ = *low; i__ <= i__1; ++i__) {
	zz = z__[i__ + na * z_dim1];
	z__[i__ + na * z_dim1] = q * zz + p * z__[i__ + en * z_dim1];
	z__[i__ + en * z_dim1] = q * z__[i__ + en * z_dim1] - p * zz;
/* L310: */
    }

    goto L330;
/*     .......... COMPLEX PAIR .......... */
L320:
    wr[na] = x + p;
    wr[en] = x + p;
    wi[na] = zz;
    wi[en] = -zz;
L330:
    en = enm2;
    goto L60;
/*     .......... ALL ROOTS FOUND.  BACKSUBSTITUTE TO FIND */
/*                VECTORS OF UPPER TRIANGULAR FORM .......... */
L340:
    if (norm == 0.) {
	goto L1001;
    }
/*     .......... FOR EN=N STEP -1 UNTIL 1 DO -- .......... */
    i__1 = *n;
    for (nn = 1; nn <= i__1; ++nn) {
	en = *n + 1 - nn;
	p = wr[en];
	q = wi[en];
	na = en - 1;
	if (q < 0.) {
	    goto L710;
	} else if (q == 0) {
	    goto L600;
	} else {
	    goto L800;
	}
/*     .......... REAL VECTOR .......... */
L600:
	m = en;
	h__[en + en * h_dim1] = 1.;
	if (na == 0) {
	    goto L800;
	}
/*     .......... FOR I=EN-1 STEP -1 UNTIL 1 DO -- .......... */
	i__2 = na;
	for (ii = 1; ii <= i__2; ++ii) {
	    i__ = en - ii;
	    w = h__[i__ + i__ * h_dim1] - p;
	    r__ = 0.;

	    i__3 = en;
	    for (j = m; j <= i__3; ++j) {
/* L610: */
		r__ += h__[i__ + j * h_dim1] * h__[j + en * h_dim1];
	    }

	    if (wi[i__] >= 0.) {
		goto L630;
	    }
	    zz = w;
	    s = r__;
	    goto L700;
L630:
	    m = i__;
	    if (wi[i__] != 0.) {
		goto L640;
	    }
	    t = w;
	    if (t != 0.) {
		goto L635;
	    }
	    tst1 = norm;
	    t = tst1;
L632:
	    t *= .01;
	    tst2 = norm + t;
	    if (tst2 > tst1) {
		goto L632;
	    }
L635:
	    h__[i__ + en * h_dim1] = -r__ / t;
	    goto L680;
/*     .......... SOLVE REAL EQUATIONS .......... */
L640:
	    x = h__[i__ + (i__ + 1) * h_dim1];
	    y = h__[i__ + 1 + i__ * h_dim1];
	    q = (wr[i__] - p) * (wr[i__] - p) + wi[i__] * wi[i__];
	    t = (x * s - zz * r__) / q;
	    h__[i__ + en * h_dim1] = t;
	    if (fabs(x) <= fabs(zz)) {
		goto L650;
	    }
	    h__[i__ + 1 + en * h_dim1] = (-r__ - w * t) / x;
	    goto L680;
L650:
	    h__[i__ + 1 + en * h_dim1] = (-s - y * t) / zz;

/*     .......... OVERFLOW CONTROL .......... */
L680:
	    t = (d__1 = h__[i__ + en * h_dim1], fabs(d__1));
	    if (t == 0.) {
		goto L700;
	    }
	    tst1 = t;
	    tst2 = tst1 + 1. / tst1;
	    if (tst2 > tst1) {
		goto L700;
	    }
	    i__3 = en;
	    for (j = i__; j <= i__3; ++j) {
		h__[j + en * h_dim1] /= t;
/* L690: */
	    }

L700:
	    ;
	}
/*     .......... END REAL VECTOR .......... */
	goto L800;
/*     .......... COMPLEX VECTOR .......... */
L710:
	m = na;
/*     .......... LAST VECTOR COMPONENT CHOSEN IMAGINARY SO THAT */
/*                EIGENVECTOR MATRIX IS TRIANGULAR .......... */
	if ((d__1 = h__[en + na * h_dim1], fabs(d__1)) <= (d__2 = h__[na + en *
		 h_dim1], fabs(d__2))) {
	    goto L720;
	}
	h__[na + na * h_dim1] = q / h__[en + na * h_dim1];
	h__[na + en * h_dim1] = -(h__[en + en * h_dim1] - p) / h__[en + na * 
		h_dim1];
	goto L730;
L720:
	d__1 = -h__[na + en * h_dim1];
	d__2 = h__[na + na * h_dim1] - p;
        {
            doublereal c_b81 = 0.;
            cdiv(&c_b81, &d__1, &d__2, &q, &h__[na + na * h_dim1], &h__[na + en *
                                                                       h_dim1]);
        }
L730:
	h__[en + na * h_dim1] = 0.;
	h__[en + en * h_dim1] = 1.;
	enm2 = na - 1;
	if (enm2 == 0) {
	    goto L800;
	}
/*     .......... FOR I=EN-2 STEP -1 UNTIL 1 DO -- .......... */
	i__2 = enm2;
	for (ii = 1; ii <= i__2; ++ii) {
	    i__ = na - ii;
	    w = h__[i__ + i__ * h_dim1] - p;
	    ra = 0.;
	    sa = 0.;

	    i__3 = en;
	    for (j = m; j <= i__3; ++j) {
		ra += h__[i__ + j * h_dim1] * h__[j + na * h_dim1];
		sa += h__[i__ + j * h_dim1] * h__[j + en * h_dim1];
/* L760: */
	    }

	    if (wi[i__] >= 0.) {
		goto L770;
	    }
	    zz = w;
	    r__ = ra;
	    s = sa;
	    goto L795;
L770:
	    m = i__;
	    if (wi[i__] != 0.) {
		goto L780;
	    }
	    d__1 = -ra;
	    d__2 = -sa;
	    cdiv(&d__1, &d__2, &w, &q, &h__[i__ + na * h_dim1], &h__[i__ + 
		    en * h_dim1]);
	    goto L790;
/*     .......... SOLVE COMPLEX EQUATIONS .......... */
L780:
	    x = h__[i__ + (i__ + 1) * h_dim1];
	    y = h__[i__ + 1 + i__ * h_dim1];
	    vr = (wr[i__] - p) * (wr[i__] - p) + wi[i__] * wi[i__] - q * q;
	    vi = (wr[i__] - p) * 2. * q;
	    if (vr != 0. || vi != 0.) {
		goto L784;
	    }
	    tst1 = norm * (fabs(w) + fabs(q) + fabs(x) + fabs(y) + fabs(zz));
	    vr = tst1;
L783:
	    vr *= .01;
	    tst2 = tst1 + vr;
	    if (tst2 > tst1) {
		goto L783;
	    }
L784:
	    d__1 = x * r__ - zz * ra + q * sa;
	    d__2 = x * s - zz * sa - q * ra;
	    cdiv(&d__1, &d__2, &vr, &vi, &h__[i__ + na * h_dim1], &h__[i__ + 
		    en * h_dim1]);
	    if (fabs(x) <= fabs(zz) + fabs(q)) {
		goto L785;
	    }
	    h__[i__ + 1 + na * h_dim1] = (-ra - w * h__[i__ + na * h_dim1] + 
		    q * h__[i__ + en * h_dim1]) / x;
	    h__[i__ + 1 + en * h_dim1] = (-sa - w * h__[i__ + en * h_dim1] - 
		    q * h__[i__ + na * h_dim1]) / x;
	    goto L790;
L785:
	    d__1 = -r__ - y * h__[i__ + na * h_dim1];
	    d__2 = -s - y * h__[i__ + en * h_dim1];
	    cdiv(&d__1, &d__2, &zz, &q, &h__[i__ + 1 + na * h_dim1], &h__[
		    i__ + 1 + en * h_dim1]);

/*     .......... OVERFLOW CONTROL .......... */
L790:
/* Computing MAX */
	    d__3 = (d__1 = h__[i__ + na * h_dim1], fabs(d__1)), d__4 = (d__2 = 
		    h__[i__ + en * h_dim1], fabs(d__2));
	    t = max(d__3,d__4);
	    if (t == 0.) {
		goto L795;
	    }
	    tst1 = t;
	    tst2 = tst1 + 1. / tst1;
	    if (tst2 > tst1) {
		goto L795;
	    }
	    i__3 = en;
	    for (j = i__; j <= i__3; ++j) {
		h__[j + na * h_dim1] /= t;
		h__[j + en * h_dim1] /= t;
/* L792: */
	    }

L795:
	    ;
	}
/*     .......... END COMPLEX VECTOR .......... */
L800:
	;
    }
/*     .......... END BACK SUBSTITUTION. */
/*                VECTORS OF ISOLATED ROOTS .......... */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ >= *low && i__ <= *igh) {
	    goto L840;
	}

	i__2 = *n;
	for (j = i__; j <= i__2; ++j) {
/* L820: */
	    z__[i__ + j * z_dim1] = h__[i__ + j * h_dim1];
	}

L840:
	;
    }
/*     .......... MULTIPLY BY TRANSFORMATION MATRIX TO GIVE */
/*                VECTORS OF ORIGINAL FULL MATRIX. */
/*                FOR J=N STEP -1 UNTIL LOW DO -- .......... */
    i__1 = *n;
    for (jj = *low; jj <= i__1; ++jj) {
	j = *n + *low - jj;
	m = min(j,*igh);

	i__2 = *igh;
	for (i__ = *low; i__ <= i__2; ++i__) {
	    zz = 0.;

	    i__3 = m;
	    for (k = *low; k <= i__3; ++k) {
/* L860: */
		zz += z__[i__ + k * z_dim1] * h__[k + j * h_dim1];
	    }

	    z__[i__ + j * z_dim1] = zz;
/* L880: */
	}
    }

    goto L1001;
/*     .......... SET ERROR -- ALL EIGENVALUES HAVE NOT */
/*                CONVERGED AFTER 30*N ITERATIONS .......... */
L1000:
    *ierr = en;
L1001:
    return 0;
} /* hqr2_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
cdiv(doublereal *ar, doublereal *ai, doublereal *br, doublereal *bi, doublereal *cr, doublereal *ci)
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Local variables */
    static doublereal s, ais, bis, ars, brs;


/*     COMPLEX DIVISION, (CR,CI) = (AR,AI)/(BR,BI) */

    s = fabs(*br) + fabs(*bi);
    ars = *ar / s;
    ais = *ai / s;
    brs = *br / s;
    bis = *bi / s;
/* Computing 2nd power */
    d__1 = brs;
/* Computing 2nd power */
    d__2 = bis;
    s = d__1 * d__1 + d__2 * d__2;
    *cr = (ars * brs + ais * bis) / s;
    *ci = (ais * brs - ars * bis) / s;
    return 0;
} /* cdiv_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
balanc(integer *nm, integer *n, doublereal *a, integer *low, integer *igh, doublereal *scale)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    doublereal d__1;

    /* Local variables */
    static integer iexc;
    static doublereal c__, f, g;
    static integer i__, j, k, l, m;
    static doublereal r__, s, radix, b2;
    static integer jj;
    static logical noconv;



/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE BALANCE, */
/*     NUM. MATH. 13, 293-304(1969) BY PARLETT AND REINSCH. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 315-326(1971). */

/*     THIS SUBROUTINE BALANCES A REAL MATRIX AND ISOLATES */
/*     EIGENVALUES WHENEVER POSSIBLE. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRIX. */

/*        A CONTAINS THE INPUT MATRIX TO BE BALANCED. */

/*     ON OUTPUT */

/*        A CONTAINS THE BALANCED MATRIX. */

/*        LOW AND IGH ARE TWO INTEGERS SUCH THAT A(I,J) */
/*          IS EQUAL TO ZERO IF */
/*           (1) I IS GREATER THAN J AND */
/*           (2) J=1,...,LOW-1 OR I=IGH+1,...,N. */

/*        SCALE CONTAINS INFORMATION DETERMINING THE */
/*           PERMUTATIONS AND SCALING FACTORS USED. */

/*     SUPPOSE THAT THE PRINCIPAL SUBMATRIX IN ROWS LOW THROUGH IGH */
/*     HAS BEEN BALANCED, THAT P(J) DENOTES THE INDEX INTERCHANGED */
/*     WITH J DURING THE PERMUTATION STEP, AND THAT THE ELEMENTS */
/*     OF THE DIAGONAL MATRIX USED ARE DENOTED BY D(I,J).  THEN */
/*        SCALE(J) = P(J),    FOR J = 1,...,LOW-1 */
/*                 = D(J,J),      J = LOW,...,IGH */
/*                 = P(J)         J = IGH+1,...,N. */
/*     THE ORDER IN WHICH THE INTERCHANGES ARE MADE IS N TO IGH+1, */
/*     THEN 1 TO LOW-1. */

/*     NOTE THAT 1 IS RETURNED FOR IGH IF IGH IS ZERO FORMALLY. */

/*     THE ALGOL PROCEDURE EXC CONTAINED IN BALANCE APPEARS IN */
/*     BALANC  IN LINE.  (NOTE THAT THE ALGOL ROLES OF IDENTIFIERS */
/*     K,L HAVE BEEN REVERSED.) */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    --scale;
    a_dim1 = *nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    /* Function Body */
    radix = 16.;

    b2 = radix * radix;
    k = 1;
    l = *n;
    goto L100;
/*     .......... IN-LINE PROCEDURE FOR ROW AND */
/*                COLUMN EXCHANGE .......... */
L20:
    scale[m] = (doublereal) j;
    if (j == m) {
	goto L50;
    }

    i__1 = l;
    for (i__ = 1; i__ <= i__1; ++i__) {
	f = a[i__ + j * a_dim1];
	a[i__ + j * a_dim1] = a[i__ + m * a_dim1];
	a[i__ + m * a_dim1] = f;
/* L30: */
    }

    i__1 = *n;
    for (i__ = k; i__ <= i__1; ++i__) {
	f = a[j + i__ * a_dim1];
	a[j + i__ * a_dim1] = a[m + i__ * a_dim1];
	a[m + i__ * a_dim1] = f;
/* L40: */
    }

L50:
    switch ((int)iexc) {
	case 1:  goto L80;
	case 2:  goto L130;
    }
/*     .......... SEARCH FOR ROWS ISOLATING AN EIGENVALUE */
/*                AND PUSH THEM DOWN .......... */
L80:
    if (l == 1) {
	goto L280;
    }
    --l;
/*     .......... FOR J=L STEP -1 UNTIL 1 DO -- .......... */
L100:
    i__1 = l;
    for (jj = 1; jj <= i__1; ++jj) {
	j = l + 1 - jj;

	i__2 = l;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (i__ == j) {
		goto L110;
	    }
	    if (a[j + i__ * a_dim1] != 0.) {
		goto L120;
	    }
L110:
	    ;
	}

	m = l;
	iexc = 1;
	goto L20;
L120:
	;
    }

    goto L140;
/*     .......... SEARCH FOR COLUMNS ISOLATING AN EIGENVALUE */
/*                AND PUSH THEM LEFT .......... */
L130:
    ++k;

L140:
    i__1 = l;
    for (j = k; j <= i__1; ++j) {

	i__2 = l;
	for (i__ = k; i__ <= i__2; ++i__) {
	    if (i__ == j) {
		goto L150;
	    }
	    if (a[i__ + j * a_dim1] != 0.) {
		goto L170;
	    }
L150:
	    ;
	}

	m = k;
	iexc = 2;
	goto L20;
L170:
	;
    }
/*     .......... NOW BALANCE THE SUBMATRIX IN ROWS K TO L .......... */
    i__1 = l;
    for (i__ = k; i__ <= i__1; ++i__) {
/* L180: */
	scale[i__] = 1.;
    }
/*     .......... ITERATIVE LOOP FOR NORM REDUCTION .......... */
L190:
    noconv = FALSE_;

    i__1 = l;
    for (i__ = k; i__ <= i__1; ++i__) {
	c__ = 0.;
	r__ = 0.;

	i__2 = l;
	for (j = k; j <= i__2; ++j) {
	    if (j == i__) {
		goto L200;
	    }
	    c__ += (d__1 = a[j + i__ * a_dim1], fabs(d__1));
	    r__ += (d__1 = a[i__ + j * a_dim1], fabs(d__1));
L200:
	    ;
	}
/*     .......... GUARD AGAINST ZERO C OR R DUE TO UNDERFLOW .........
. */
	if (c__ == 0. || r__ == 0.) {
	    goto L270;
	}
	g = r__ / radix;
	f = 1.;
	s = c__ + r__;
L210:
	if (c__ >= g) {
	    goto L220;
	}
	f *= radix;
	c__ *= b2;
	goto L210;
L220:
	g = r__ * radix;
L230:
	if (c__ < g) {
	    goto L240;
	}
	f /= radix;
	c__ /= b2;
	goto L230;
/*     .......... NOW BALANCE .......... */
L240:
	if ((c__ + r__) / f >= s * .95) {
	    goto L270;
	}
	g = 1. / f;
	scale[i__] *= f;
	noconv = TRUE_;

	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
/* L250: */
	    a[i__ + j * a_dim1] *= g;
	}

	i__2 = l;
	for (j = 1; j <= i__2; ++j) {
/* L260: */
	    a[j + i__ * a_dim1] *= f;
	}

L270:
	;
    }

    if (noconv) {
	goto L190;
    }

L280:
    *low = k;
    *igh = l;
    return 0;
} /* balanc_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
balbak(integer *nm, integer *n, integer *low, integer *igh, doublereal *scale, integer *m, doublereal *z__)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, k;
    static doublereal s;
    static integer ii;



/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE BALBAK, */
/*     NUM. MATH. 13, 293-304(1969) BY PARLETT AND REINSCH. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 315-326(1971). */

/*     THIS SUBROUTINE FORMS THE EIGENVECTORS OF A REAL GENERAL */
/*     MATRIX BY BACK TRANSFORMING THOSE OF THE CORRESPONDING */
/*     BALANCED MATRIX DETERMINED BY  BALANC. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRIX. */

/*        LOW AND IGH ARE INTEGERS DETERMINED BY  BALANC. */

/*        SCALE CONTAINS INFORMATION DETERMINING THE PERMUTATIONS */
/*          AND SCALING FACTORS USED BY  BALANC. */

/*        M IS THE NUMBER OF COLUMNS OF Z TO BE BACK TRANSFORMED. */

/*        Z CONTAINS THE REAL AND IMAGINARY PARTS OF THE EIGEN- */
/*          VECTORS TO BE BACK TRANSFORMED IN ITS FIRST M COLUMNS. */

/*     ON OUTPUT */

/*        Z CONTAINS THE REAL AND IMAGINARY PARTS OF THE */
/*          TRANSFORMED EIGENVECTORS IN ITS FIRST M COLUMNS. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    --scale;
    z_dim1 = *nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;

    /* Function Body */
    if (*m == 0) {
	goto L200;
    }
    if (*igh == *low) {
	goto L120;
    }

    i__1 = *igh;
    for (i__ = *low; i__ <= i__1; ++i__) {
	s = scale[i__];
/*     .......... LEFT HAND EIGENVECTORS ARE BACK TRANSFORMED */
/*                IF THE FOREGOING STATEMENT IS REPLACED BY */
/*                S=1.0D0/SCALE(I). .......... */
	i__2 = *m;
	for (j = 1; j <= i__2; ++j) {
/* L100: */
	    z__[i__ + j * z_dim1] *= s;
	}

/* L110: */
    }
/*     ......... FOR I=LOW-1 STEP -1 UNTIL 1, */
/*               IGH+1 STEP 1 UNTIL N DO -- .......... */
L120:
    i__1 = *n;
    for (ii = 1; ii <= i__1; ++ii) {
	i__ = ii;
	if (i__ >= *low && i__ <= *igh) {
	    goto L140;
	}
	if (i__ < *low) {
	    i__ = *low - ii;
	}
	k = (integer) scale[i__];
	if (k == i__) {
	    goto L140;
	}

	i__2 = *m;
	for (j = 1; j <= i__2; ++j) {
	    s = z__[i__ + j * z_dim1];
	    z__[i__ + j * z_dim1] = z__[k + j * z_dim1];
	    z__[k + j * z_dim1] = s;
/* L130: */
	}

L140:
	;
    }

L200:
    return 0;
} /* balbak_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
elmhes(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, integer *int__)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;

    /* Local variables */
    static integer i__, j, m;
    static doublereal x, y;
    static integer la, mm1, kp1, mp1;



/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE ELMHES, */
/*     NUM. MATH. 12, 349-368(1968) BY MARTIN AND WILKINSON. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 339-358(1971). */

/*     GIVEN A REAL GENERAL MATRIX, THIS SUBROUTINE */
/*     REDUCES A SUBMATRIX SITUATED IN ROWS AND COLUMNS */
/*     LOW THROUGH IGH TO UPPER HESSENBERG FORM BY */
/*     STABILIZED ELEMENTARY SIMILARITY TRANSFORMATIONS. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRIX. */

/*        LOW AND IGH ARE INTEGERS DETERMINED BY THE BALANCING */
/*          SUBROUTINE  BALANC.  IF  BALANC  HAS NOT BEEN USED, */
/*          SET LOW=1, IGH=N. */

/*        A CONTAINS THE INPUT MATRIX. */

/*     ON OUTPUT */

/*        A CONTAINS THE HESSENBERG MATRIX.  THE MULTIPLIERS */
/*          WHICH WERE USED IN THE REDUCTION ARE STORED IN THE */
/*          REMAINING TRIANGLE UNDER THE HESSENBERG MATRIX. */

/*        INT CONTAINS INFORMATION ON THE ROWS AND COLUMNS */
/*          INTERCHANGED IN THE REDUCTION. */
/*          ONLY ELEMENTS LOW THROUGH IGH ARE USED. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    a_dim1 = *nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;
    --int__;

    /* Function Body */
    la = *igh - 1;
    kp1 = *low + 1;
    if (la < kp1) {
	goto L200;
    }

    i__1 = la;
    for (m = kp1; m <= i__1; ++m) {
	mm1 = m - 1;
	x = 0.;
	i__ = m;

	i__2 = *igh;
	for (j = m; j <= i__2; ++j) {
	    if ((d__1 = a[j + mm1 * a_dim1], fabs(d__1)) <= fabs(x)) {
		goto L100;
	    }
	    x = a[j + mm1 * a_dim1];
	    i__ = j;
L100:
	    ;
	}

	int__[m] = i__;
	if (i__ == m) {
	    goto L130;
	}
/*     .......... INTERCHANGE ROWS AND COLUMNS OF A .......... */
	i__2 = *n;
	for (j = mm1; j <= i__2; ++j) {
	    y = a[i__ + j * a_dim1];
	    a[i__ + j * a_dim1] = a[m + j * a_dim1];
	    a[m + j * a_dim1] = y;
/* L110: */
	}

	i__2 = *igh;
	for (j = 1; j <= i__2; ++j) {
	    y = a[j + i__ * a_dim1];
	    a[j + i__ * a_dim1] = a[j + m * a_dim1];
	    a[j + m * a_dim1] = y;
/* L120: */
	}
/*     .......... END INTERCHANGE .......... */
L130:
	if (x == 0.) {
	    goto L180;
	}
	mp1 = m + 1;

	i__2 = *igh;
	for (i__ = mp1; i__ <= i__2; ++i__) {
	    y = a[i__ + mm1 * a_dim1];
	    if (y == 0.) {
		goto L160;
	    }
	    y /= x;
	    a[i__ + mm1 * a_dim1] = y;

	    i__3 = *n;
	    for (j = m; j <= i__3; ++j) {
/* L140: */
		a[i__ + j * a_dim1] -= y * a[m + j * a_dim1];
	    }

	    i__3 = *igh;
	    for (j = 1; j <= i__3; ++j) {
/* L150: */
		a[j + m * a_dim1] += y * a[j + i__ * a_dim1];
	    }

L160:
	    ;
	}

L180:
	;
    }

L200:
    return 0;
} /* elmhes_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
eltran(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, integer *int__, doublereal *z__)
{
    /* System generated locals */
    integer a_dim1, a_offset, z_dim1, z_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, kl, mm, mp, mp1;



/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE ELMTRANS, 
*/
/*     NUM. MATH. 16, 181-204(1970) BY PETERS AND WILKINSON. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 372-395(1971). */

/*     THIS SUBROUTINE ACCUMULATES THE STABILIZED ELEMENTARY */
/*     SIMILARITY TRANSFORMATIONS USED IN THE REDUCTION OF A */
/*     REAL GENERAL MATRIX TO UPPER HESSENBERG FORM BY  ELMHES. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRIX. */

/*        LOW AND IGH ARE INTEGERS DETERMINED BY THE BALANCING */
/*          SUBROUTINE  BALANC.  IF  BALANC  HAS NOT BEEN USED, */
/*          SET LOW=1, IGH=N. */

/*        A CONTAINS THE MULTIPLIERS WHICH WERE USED IN THE */
/*          REDUCTION BY  ELMHES  IN ITS LOWER TRIANGLE */
/*          BELOW THE SUBDIAGONAL. */

/*        INT CONTAINS INFORMATION ON THE ROWS AND COLUMNS */
/*          INTERCHANGED IN THE REDUCTION BY  ELMHES. */
/*          ONLY ELEMENTS LOW THROUGH IGH ARE USED. */

/*     ON OUTPUT */

/*        Z CONTAINS THE TRANSFORMATION MATRIX PRODUCED IN THE */
/*          REDUCTION BY  ELMHES. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

/*     .......... INITIALIZE Z TO IDENTITY MATRIX .......... */
    /* Parameter adjustments */
    z_dim1 = *nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;
    --int__;
    a_dim1 = *nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    /* Function Body */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {

	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* L60: */
	    z__[i__ + j * z_dim1] = 0.;
	}

	z__[j + j * z_dim1] = 1.;
/* L80: */
    }

    kl = *igh - *low - 1;
    if (kl < 1) {
	goto L200;
    }
/*     .......... FOR MP=IGH-1 STEP -1 UNTIL LOW+1 DO -- .......... */
    i__1 = kl;
    for (mm = 1; mm <= i__1; ++mm) {
	mp = *igh - mm;
	mp1 = mp + 1;

	i__2 = *igh;
	for (i__ = mp1; i__ <= i__2; ++i__) {
/* L100: */
	    z__[i__ + mp * z_dim1] = a[i__ + (mp - 1) * a_dim1];
	}

	i__ = int__[mp];
	if (i__ == mp) {
	    goto L140;
	}

	i__2 = *igh;
	for (j = mp; j <= i__2; ++j) {
	    z__[mp + j * z_dim1] = z__[i__ + j * z_dim1];
	    z__[i__ + j * z_dim1] = 0.;
/* L130: */
	}

	z__[i__ + mp * z_dim1] = 1.;
L140:
	;
    }

L200:
    return 0;
} /* eltran_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  EISPACK routines needed in the computation of Floquet multipliers */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
qzhes(integer nm, integer n, doublereal *a, doublereal *b, logical matz, doublereal *z__)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, z_dim1, z_offset, i__1, i__2, 
	    i__3;
    doublereal d__1, d__2;

    /* Local variables */
    static integer i__, j, k, l;
    static doublereal r__, s, t;
    static integer l1;
    static doublereal u1, u2, v1, v2;
    static integer lb, nk1, nm1, nm2;
    static doublereal rho;



/*     THIS SUBROUTINE IS THE FIRST STEP OF THE QZ ALGORITHM */
/*     FOR SOLVING GENERALIZED MATRIX EIGENVALUE PROBLEMS, */
/*     SIAM J. NUMER. ANAL. 10, 241-256(1973) BY MOLER AND STEWART. */

/*     THIS SUBROUTINE ACCEPTS A PAIR OF REAL GENERAL MATRICES AND */
/*     REDUCES ONE OF THEM TO UPPER HESSENBERG FORM AND THE OTHER */
/*     TO UPPER TRIANGULAR FORM USING ORTHOGONAL TRANSFORMATIONS. */
/*     IT IS USUALLY FOLLOWED BY  QZIT,  QZVAL  AND, POSSIBLY,  QZVEC. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRICES. */

/*        A CONTAINS A REAL GENERAL MATRIX. */

/*        B CONTAINS A REAL GENERAL MATRIX. */

/*        MATZ SHOULD BE SET TO .TRUE. IF THE RIGHT HAND TRANSFORMATIONS 
*/
/*          ARE TO BE ACCUMULATED FOR LATER USE IN COMPUTING */
/*          EIGENVECTORS, AND TO .FALSE. OTHERWISE. */

/*     ON OUTPUT */

/*        A HAS BEEN REDUCED TO UPPER HESSENBERG FORM.  THE ELEMENTS */
/*          BELOW THE FIRST SUBDIAGONAL HAVE BEEN SET TO ZERO. */

/*        B HAS BEEN REDUCED TO UPPER TRIANGULAR FORM.  THE ELEMENTS */
/*          BELOW THE MAIN DIAGONAL HAVE BEEN SET TO ZERO. */

/*        Z CONTAINS THE PRODUCT OF THE RIGHT HAND TRANSFORMATIONS IF */
/*          MATZ HAS BEEN SET TO .TRUE.  OTHERWISE, Z IS NOT REFERENCED. 
*/

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

/*     .......... INITIALIZE Z .......... */
    /* Parameter adjustments */
    z_dim1 = nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;
    b_dim1 = nm;
    b_offset = b_dim1 + 1;
    b -= b_offset;
    a_dim1 = nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    /* Function Body */
    if (! (matz)) {
	goto L10;
    }

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {

	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    z__[i__ + j * z_dim1] = 0.;
/* L2: */
	}

	z__[j + j * z_dim1] = 1.;
/* L3: */
    }
/*     .......... REDUCE B TO UPPER TRIANGULAR FORM .......... */
L10:
    if (n <= 1) {
	goto L170;
    }
    nm1 = n - 1;

    i__1 = nm1;
    for (l = 1; l <= i__1; ++l) {
	l1 = l + 1;
	s = 0.;

	i__2 = n;
	for (i__ = l1; i__ <= i__2; ++i__) {
	    s += (d__1 = b[i__ + l * b_dim1], fabs(d__1));
/* L20: */
	}

	if (s == 0.) {
	    goto L100;
	}
	s += (d__1 = b[l + l * b_dim1], fabs(d__1));
	r__ = 0.;

	i__2 = n;
	for (i__ = l; i__ <= i__2; ++i__) {
	    b[i__ + l * b_dim1] /= s;
/* Computing 2nd power */
	    d__1 = b[i__ + l * b_dim1];
	    r__ += d__1 * d__1;
/* L25: */
	}

	d__1 = sqrt(r__);
	r__ = d_sign(d__1, b[l + l * b_dim1]);
	b[l + l * b_dim1] += r__;
	rho = r__ * b[l + l * b_dim1];

	i__2 = n;
	for (j = l1; j <= i__2; ++j) {
	    t = 0.;

	    i__3 = n;
	    for (i__ = l; i__ <= i__3; ++i__) {
		t += b[i__ + l * b_dim1] * b[i__ + j * b_dim1];
/* L30: */
	    }

	    t = -t / rho;

	    i__3 = n;
	    for (i__ = l; i__ <= i__3; ++i__) {
		b[i__ + j * b_dim1] += t * b[i__ + l * b_dim1];
/* L40: */
	    }

/* L50: */
	}

	i__2 = n;
	for (j = 1; j <= i__2; ++j) {
	    t = 0.;

	    i__3 = n;
	    for (i__ = l; i__ <= i__3; ++i__) {
		t += b[i__ + l * b_dim1] * a[i__ + j * a_dim1];
/* L60: */
	    }

	    t = -t / rho;

	    i__3 = n;
	    for (i__ = l; i__ <= i__3; ++i__) {
		a[i__ + j * a_dim1] += t * b[i__ + l * b_dim1];
/* L70: */
	    }

/* L80: */
	}

	b[l + l * b_dim1] = -s * r__;

	i__2 = n;
	for (i__ = l1; i__ <= i__2; ++i__) {
	    b[i__ + l * b_dim1] = 0.;
/* L90: */
	}

L100:
	;
    }
/*     .......... REDUCE A TO UPPER HESSENBERG FORM, WHILE */
/*                KEEPING B TRIANGULAR .......... */
    if (n == 2) {
	goto L170;
    }
    nm2 = n - 2;

    i__1 = nm2;
    for (k = 1; k <= i__1; ++k) {
	nk1 = nm1 - k;
/*     .......... FOR L=N-1 STEP -1 UNTIL K+1 DO -- .......... */
	i__2 = nk1;
	for (lb = 1; lb <= i__2; ++lb) {
	    l = n - lb;
	    l1 = l + 1;
/*     .......... ZERO A(L+1,K) .......... */
	    s = (d__1 = a[l + k * a_dim1], fabs(d__1)) + (d__2 = a[l1 + k * 
		    a_dim1], fabs(d__2));
	    if (s == 0.) {
		goto L150;
	    }
	    u1 = a[l + k * a_dim1] / s;
	    u2 = a[l1 + k * a_dim1] / s;
	    d__1 = sqrt(u1 * u1 + u2 * u2);
	    r__ = d_sign(d__1, u1);
	    v1 = -(u1 + r__) / r__;
	    v2 = -u2 / r__;
	    u2 = v2 / v1;

	    i__3 = n;
	    for (j = k; j <= i__3; ++j) {
		t = a[l + j * a_dim1] + u2 * a[l1 + j * a_dim1];
		a[l + j * a_dim1] += t * v1;
		a[l1 + j * a_dim1] += t * v2;
/* L110: */
	    }

	    a[l1 + k * a_dim1] = 0.;

	    i__3 = n;
	    for (j = l; j <= i__3; ++j) {
		t = b[l + j * b_dim1] + u2 * b[l1 + j * b_dim1];
		b[l + j * b_dim1] += t * v1;
		b[l1 + j * b_dim1] += t * v2;
/* L120: */
	    }
/*     .......... ZERO B(L+1,L) .......... */
	    s = (d__1 = b[l1 + l1 * b_dim1], fabs(d__1)) + (d__2 = b[l1 + l * 
		    b_dim1], fabs(d__2));
	    if (s == 0.) {
		goto L150;
	    }
	    u1 = b[l1 + l1 * b_dim1] / s;
	    u2 = b[l1 + l * b_dim1] / s;
	    d__1 = sqrt(u1 * u1 + u2 * u2);
	    r__ = d_sign(d__1, u1);
	    v1 = -(u1 + r__) / r__;
	    v2 = -u2 / r__;
	    u2 = v2 / v1;

	    i__3 = l1;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		t = b[i__ + l1 * b_dim1] + u2 * b[i__ + l * b_dim1];
		b[i__ + l1 * b_dim1] += t * v1;
		b[i__ + l * b_dim1] += t * v2;
/* L130: */
	    }

	    b[l1 + l * b_dim1] = 0.;

	    i__3 = n;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		t = a[i__ + l1 * a_dim1] + u2 * a[i__ + l * a_dim1];
		a[i__ + l1 * a_dim1] += t * v1;
		a[i__ + l * a_dim1] += t * v2;
/* L140: */
	    }

	    if (! (matz)) {
		goto L150;
	    }

	    i__3 = n;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		t = z__[i__ + l1 * z_dim1] + u2 * z__[i__ + l * z_dim1];
		z__[i__ + l1 * z_dim1] += t * v1;
		z__[i__ + l * z_dim1] += t * v2;
/* L145: */
	    }

L150:
	    ;
	}

/* L160: */
    }

L170:
    return 0;
} /* qzhes_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
qzit(integer nm, integer n, doublereal *a, doublereal *b, doublereal eps1, logical matz, doublereal *z__, integer *ierr)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, z_dim1, z_offset, i__1, i__2, 
	    i__3;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    static doublereal epsa, epsb;
    static integer i__, j, k, l;
    static doublereal r__, s, t, anorm, bnorm;
    static integer enorn;
    static doublereal a1, a2, a3;
    static integer k1, k2, l1;
    static doublereal u1, u2, u3, v1, v2, v3, a11, a12, a21, a22, a33, a34, 
	    a43, a44, b11, b12, b22, b33;
    static integer na, ld;
    static doublereal b34, b44;
    static integer en;
    static doublereal ep;
    static integer ll;
    static doublereal sh;

    static logical notlas;
    static integer km1, lm1;
    static doublereal ani, bni;
    static integer ish, itn, its, enm2, lor1;



/*     THIS SUBROUTINE IS THE SECOND STEP OF THE QZ ALGORITHM */
/*     FOR SOLVING GENERALIZED MATRIX EIGENVALUE PROBLEMS, */
/*     SIAM J. NUMER. ANAL. 10, 241-256(1973) BY MOLER AND STEWART, */
/*     AS MODIFIED IN TECHNICAL NOTE NASA TN D-7305(1973) BY WARD. */

/*     THIS SUBROUTINE ACCEPTS A PAIR OF REAL MATRICES, ONE OF THEM */
/*     IN UPPER HESSENBERG FORM AND THE OTHER IN UPPER TRIANGULAR FORM. */
/*     IT REDUCES THE HESSENBERG MATRIX TO QUASI-TRIANGULAR FORM USING */
/*     ORTHOGONAL TRANSFORMATIONS WHILE MAINTAINING THE TRIANGULAR FORM */
/*     OF THE OTHER MATRIX.  IT IS USUALLY PRECEDED BY  QZHES  AND */
/*     FOLLOWED BY  QZVAL  AND, POSSIBLY,  QZVEC. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRICES. */

/*        A CONTAINS A REAL UPPER HESSENBERG MATRIX. */

/*        B CONTAINS A REAL UPPER TRIANGULAR MATRIX. */

/*        EPS1 IS A TOLERANCE USED TO DETERMINE NEGLIGIBLE ELEMENTS. */
/*          EPS1 = 0.0 (OR NEGATIVE) MAY BE INPUT, IN WHICH CASE AN */
/*          ELEMENT WILL BE NEGLECTED ONLY IF IT IS LESS THAN ROUNDOFF */
/*          ERROR TIMES THE NORM OF ITS MATRIX.  IF THE INPUT EPS1 IS */
/*          POSITIVE, THEN AN ELEMENT WILL BE CONSIDERED NEGLIGIBLE */
/*          IF IT IS LESS THAN EPS1 TIMES THE NORM OF ITS MATRIX.  A */
/*          POSITIVE VALUE OF EPS1 MAY RESULT IN FASTER EXECUTION, */
/*          BUT LESS ACCURATE RESULTS. */

/*        MATZ SHOULD BE SET TO .TRUE. IF THE RIGHT HAND TRANSFORMATIONS 
*/
/*          ARE TO BE ACCUMULATED FOR LATER USE IN COMPUTING */
/*          EIGENVECTORS, AND TO .FALSE. OTHERWISE. */

/*        Z CONTAINS, IF MATZ HAS BEEN SET TO .TRUE., THE */
/*          TRANSFORMATION MATRIX PRODUCED IN THE REDUCTION */
/*          BY  QZHES, IF PERFORMED, OR ELSE THE IDENTITY MATRIX. */
/*          IF MATZ HAS BEEN SET TO .FALSE., Z IS NOT REFERENCED. */

/*     ON OUTPUT */

/*        A HAS BEEN REDUCED TO QUASI-TRIANGULAR FORM.  THE ELEMENTS */
/*          BELOW THE FIRST SUBDIAGONAL ARE STILL ZERO AND NO TWO */
/*          CONSECUTIVE SUBDIAGONAL ELEMENTS ARE NONZERO. */

/*        B IS STILL IN UPPER TRIANGULAR FORM, ALTHOUGH ITS ELEMENTS */
/*          HAVE BEEN ALTERED.  THE LOCATION B(N,1) IS USED TO STORE */
/*          EPS1 TIMES THE NORM OF B FOR LATER USE BY  QZVAL  AND  QZVEC. 
*/

/*        Z CONTAINS THE PRODUCT OF THE RIGHT HAND TRANSFORMATIONS */
/*          (FOR BOTH STEPS) IF MATZ HAS BEEN SET TO .TRUE.. */

/*        IERR IS SET TO */
/*          ZERO       FOR NORMAL RETURN, */
/*          J          IF THE LIMIT OF 30*N ITERATIONS IS EXHAUSTED */
/*                     WHILE THE J-TH EIGENVALUE IS BEING SOUGHT. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    z_dim1 = nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;
    b_dim1 = nm;
    b_offset = b_dim1 + 1;
    b -= b_offset;
    a_dim1 = nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    /* Function Body */
    *ierr = 0;
/*     .......... COMPUTE EPSA,EPSB .......... */
    anorm = 0.;
    bnorm = 0.;

    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ani = 0.;
	if (i__ != 1) {
	    ani = (d__1 = a[i__ + (i__ - 1) * a_dim1], fabs(d__1));
	}
	bni = 0.;

	i__2 = n;
	for (j = i__; j <= i__2; ++j) {
	    ani += (d__1 = a[i__ + j * a_dim1], fabs(d__1));
	    bni += (d__1 = b[i__ + j * b_dim1], fabs(d__1));
/* L20: */
	}

	if (ani > anorm) {
	    anorm = ani;
	}
	if (bni > bnorm) {
	    bnorm = bni;
	}
/* L30: */
    }

    if (anorm == 0.) {
	anorm = 1.;
    }
    if (bnorm == 0.) {
	bnorm = 1.;
    }
    ep = eps1;
    if (ep > 0.) {
	goto L50;
    }
/*     .......... USE ROUNDOFF LEVEL IF EPS1 IS ZERO .......... */
    ep = epslon(1.0);
L50:
    epsa = ep * anorm;
    epsb = ep * bnorm;
/*     .......... REDUCE A TO QUASI-TRIANGULAR FORM, WHILE */
/*                KEEPING B TRIANGULAR .......... */
    lor1 = 1;
    enorn = n;
    en = n;
    itn = n * 30;
/*     .......... BEGIN QZ STEP .......... */
L60:
    if (en <= 2) {
	goto L1001;
    }
    if (! (matz)) {
	enorn = en;
    }
    its = 0;
    na = en - 1;
    enm2 = na - 1;
L70:
    ish = 2;
/*     .......... CHECK FOR CONVERGENCE OR REDUCIBILITY. */
/*                FOR L=EN STEP -1 UNTIL 1 DO -- .......... */
    i__1 = en;
    for (ll = 1; ll <= i__1; ++ll) {
	lm1 = en - ll;
	l = lm1 + 1;
	if (l == 1) {
	    goto L95;
	}
	if ((d__1 = a[l + lm1 * a_dim1], fabs(d__1)) <= epsa) {
	    goto L90;
	}
/* L80: */
    }

L90:
    a[l + lm1 * a_dim1] = 0.;
    if (l < na) {
	goto L95;
    }
/*     .......... 1-BY-1 OR 2-BY-2 BLOCK ISOLATED .......... */
    en = lm1;
    goto L60;
/*     .......... CHECK FOR SMALL TOP OF B .......... */
L95:
    ld = l;
L100:
    l1 = l + 1;
    b11 = b[l + l * b_dim1];
    if (fabs(b11) > epsb) {
	goto L120;
    }
    b[l + l * b_dim1] = 0.;
    s = (d__1 = a[l + l * a_dim1], fabs(d__1)) + (d__2 = a[l1 + l * a_dim1], 
	    fabs(d__2));
    u1 = a[l + l * a_dim1] / s;
    u2 = a[l1 + l * a_dim1] / s;
    d__1 = sqrt(u1 * u1 + u2 * u2);
    r__ = d_sign(d__1, u1);
    v1 = -(u1 + r__) / r__;
    v2 = -u2 / r__;
    u2 = v2 / v1;

    i__1 = enorn;
    for (j = l; j <= i__1; ++j) {
	t = a[l + j * a_dim1] + u2 * a[l1 + j * a_dim1];
	a[l + j * a_dim1] += t * v1;
	a[l1 + j * a_dim1] += t * v2;
	t = b[l + j * b_dim1] + u2 * b[l1 + j * b_dim1];
	b[l + j * b_dim1] += t * v1;
	b[l1 + j * b_dim1] += t * v2;
/* L110: */
    }

    if (l != 1) {
	a[l + lm1 * a_dim1] = -a[l + lm1 * a_dim1];
    }
    lm1 = l;
    l = l1;
    goto L90;
L120:
    a11 = a[l + l * a_dim1] / b11;
    a21 = a[l1 + l * a_dim1] / b11;
    if (ish == 1) {
	goto L140;
    }
/*     .......... ITERATION STRATEGY .......... */
    if (itn == 0) {
	goto L1000;
    }
    if (its == 10) {
	goto L155;
    }
/*     .......... DETERMINE TYPE OF SHIFT .......... */
    b22 = b[l1 + l1 * b_dim1];
    if (fabs(b22) < epsb) {
	b22 = epsb;
    }
    b33 = b[na + na * b_dim1];
    if (fabs(b33) < epsb) {
	b33 = epsb;
    }
    b44 = b[en + en * b_dim1];
    if (fabs(b44) < epsb) {
	b44 = epsb;
    }
    a33 = a[na + na * a_dim1] / b33;
    a34 = a[na + en * a_dim1] / b44;
    a43 = a[en + na * a_dim1] / b33;
    a44 = a[en + en * a_dim1] / b44;
    b34 = b[na + en * b_dim1] / b44;
    t = (a43 * b34 - a33 - a44) * .5;
    r__ = t * t + a34 * a43 - a33 * a44;
    if (r__ < 0.) {
	goto L150;
    }
/*     .......... DETERMINE SINGLE SHIFT ZEROTH COLUMN OF A .......... */
    ish = 1;
    r__ = sqrt(r__);
    sh = -t + r__;
    s = -t - r__;
    if ((d__1 = s - a44, fabs(d__1)) < (d__2 = sh - a44, fabs(d__2))) {
	sh = s;
    }
/*     .......... LOOK FOR TWO CONSECUTIVE SMALL */
/*                SUB-DIAGONAL ELEMENTS OF A. */
/*                FOR L=EN-2 STEP -1 UNTIL LD DO -- .......... */
    i__1 = enm2;
    for (ll = ld; ll <= i__1; ++ll) {
	l = enm2 + ld - ll;
	if (l == ld) {
	    goto L140;
	}
	lm1 = l - 1;
	l1 = l + 1;
	t = a[l + l * a_dim1];
	if ((d__1 = b[l + l * b_dim1], fabs(d__1)) > epsb) {
	    t -= sh * b[l + l * b_dim1];
	}
	if ((d__1 = a[l + lm1 * a_dim1], fabs(d__1)) <= (d__2 = t / a[l1 + l * 
		a_dim1], fabs(d__2)) * epsa) {
	    goto L100;
	}
/* L130: */
    }

L140:
    a1 = a11 - sh;
    a2 = a21;
    if (l != ld) {
	a[l + lm1 * a_dim1] = -a[l + lm1 * a_dim1];
    }
    goto L160;
/*     .......... DETERMINE DOUBLE SHIFT ZEROTH COLUMN OF A .......... */
L150:
    a12 = a[l + l1 * a_dim1] / b22;
    a22 = a[l1 + l1 * a_dim1] / b22;
    b12 = b[l + l1 * b_dim1] / b22;
    a1 = ((a33 - a11) * (a44 - a11) - a34 * a43 + a43 * b34 * a11) / a21 + 
	    a12 - a11 * b12;
    a2 = a22 - a11 - a21 * b12 - (a33 - a11) - (a44 - a11) + a43 * b34;
    a3 = a[l1 + 1 + l1 * a_dim1] / b22;
    goto L160;
/*     .......... AD HOC SHIFT .......... */
L155:
    a1 = 0.;
    a2 = 1.;
    a3 = 1.1605;
L160:
    ++its;
    --itn;
    if (! (matz)) {
	lor1 = ld;
    }
/*     .......... MAIN LOOP .......... */
    i__1 = na;
    for (k = l; k <= i__1; ++k) {
	notlas = k != na && ish == 2;
	k1 = k + 1;
	k2 = k + 2;
/* Computing MAX */
	i__2 = k - 1;
	km1 = max(i__2,l);
/* Computing MIN */
	i__2 = en, i__3 = k1 + ish;
	ll = min(i__2,i__3);
	if (notlas) {
	    goto L190;
	}
/*     .......... ZERO A(K+1,K-1) .......... */
	if (k == l) {
	    goto L170;
	}
	a1 = a[k + km1 * a_dim1];
	a2 = a[k1 + km1 * a_dim1];
L170:
	s = fabs(a1) + fabs(a2);
	if (s == 0.) {
	    goto L70;
	}
	u1 = a1 / s;
	u2 = a2 / s;
	d__1 = sqrt(u1 * u1 + u2 * u2);
	r__ = d_sign(d__1, u1);
	v1 = -(u1 + r__) / r__;
	v2 = -u2 / r__;
	u2 = v2 / v1;

	i__2 = enorn;
	for (j = km1; j <= i__2; ++j) {
	    t = a[k + j * a_dim1] + u2 * a[k1 + j * a_dim1];
	    a[k + j * a_dim1] += t * v1;
	    a[k1 + j * a_dim1] += t * v2;
	    t = b[k + j * b_dim1] + u2 * b[k1 + j * b_dim1];
	    b[k + j * b_dim1] += t * v1;
	    b[k1 + j * b_dim1] += t * v2;
/* L180: */
	}

	if (k != l) {
	    a[k1 + km1 * a_dim1] = 0.;
	}
	goto L240;
/*     .......... ZERO A(K+1,K-1) AND A(K+2,K-1) .......... */
L190:
	if (k == l) {
	    goto L200;
	}
	a1 = a[k + km1 * a_dim1];
	a2 = a[k1 + km1 * a_dim1];
	a3 = a[k2 + km1 * a_dim1];
L200:
	s = fabs(a1) + fabs(a2) + fabs(a3);
	if (s == 0.) {
	    goto L260;
	}
	u1 = a1 / s;
	u2 = a2 / s;
	u3 = a3 / s;
	d__1 = sqrt(u1 * u1 + u2 * u2 + u3 * u3);
	r__ = d_sign(d__1, u1);
	v1 = -(u1 + r__) / r__;
	v2 = -u2 / r__;
	v3 = -u3 / r__;
	u2 = v2 / v1;
	u3 = v3 / v1;

	i__2 = enorn;
	for (j = km1; j <= i__2; ++j) {
	    t = a[k + j * a_dim1] + u2 * a[k1 + j * a_dim1] + u3 * a[k2 + j * 
		    a_dim1];
	    a[k + j * a_dim1] += t * v1;
	    a[k1 + j * a_dim1] += t * v2;
	    a[k2 + j * a_dim1] += t * v3;
	    t = b[k + j * b_dim1] + u2 * b[k1 + j * b_dim1] + u3 * b[k2 + j * 
		    b_dim1];
	    b[k + j * b_dim1] += t * v1;
	    b[k1 + j * b_dim1] += t * v2;
	    b[k2 + j * b_dim1] += t * v3;
/* L210: */
	}

	if (k == l) {
	    goto L220;
	}
	a[k1 + km1 * a_dim1] = 0.;
	a[k2 + km1 * a_dim1] = 0.;
/*     .......... ZERO B(K+2,K+1) AND B(K+2,K) .......... */
L220:
	s = (d__1 = b[k2 + k2 * b_dim1], fabs(d__1)) + (d__2 = b[k2 + k1 * 
		b_dim1], fabs(d__2)) + (d__3 = b[k2 + k * b_dim1], fabs(d__3));
	if (s == 0.) {
	    goto L240;
	}
	u1 = b[k2 + k2 * b_dim1] / s;
	u2 = b[k2 + k1 * b_dim1] / s;
	u3 = b[k2 + k * b_dim1] / s;
	d__1 = sqrt(u1 * u1 + u2 * u2 + u3 * u3);
	r__ = d_sign(d__1, u1);
	v1 = -(u1 + r__) / r__;
	v2 = -u2 / r__;
	v3 = -u3 / r__;
	u2 = v2 / v1;
	u3 = v3 / v1;

	i__2 = ll;
	for (i__ = lor1; i__ <= i__2; ++i__) {
	    t = a[i__ + k2 * a_dim1] + u2 * a[i__ + k1 * a_dim1] + u3 * a[i__ 
		    + k * a_dim1];
	    a[i__ + k2 * a_dim1] += t * v1;
	    a[i__ + k1 * a_dim1] += t * v2;
	    a[i__ + k * a_dim1] += t * v3;
	    t = b[i__ + k2 * b_dim1] + u2 * b[i__ + k1 * b_dim1] + u3 * b[i__ 
		    + k * b_dim1];
	    b[i__ + k2 * b_dim1] += t * v1;
	    b[i__ + k1 * b_dim1] += t * v2;
	    b[i__ + k * b_dim1] += t * v3;
/* L230: */
	}

	b[k2 + k * b_dim1] = 0.;
	b[k2 + k1 * b_dim1] = 0.;
	if (! (matz)) {
	    goto L240;
	}

	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    t = z__[i__ + k2 * z_dim1] + u2 * z__[i__ + k1 * z_dim1] + u3 * 
		    z__[i__ + k * z_dim1];
	    z__[i__ + k2 * z_dim1] += t * v1;
	    z__[i__ + k1 * z_dim1] += t * v2;
	    z__[i__ + k * z_dim1] += t * v3;
/* L235: */
	}
/*     .......... ZERO B(K+1,K) .......... */
L240:
	s = (d__1 = b[k1 + k1 * b_dim1], fabs(d__1)) + (d__2 = b[k1 + k * 
		b_dim1], fabs(d__2));
	if (s == 0.) {
	    goto L260;
	}
	u1 = b[k1 + k1 * b_dim1] / s;
	u2 = b[k1 + k * b_dim1] / s;
	d__1 = sqrt(u1 * u1 + u2 * u2);
	r__ = d_sign(d__1, u1);
	v1 = -(u1 + r__) / r__;
	v2 = -u2 / r__;
	u2 = v2 / v1;

	i__2 = ll;
	for (i__ = lor1; i__ <= i__2; ++i__) {
	    t = a[i__ + k1 * a_dim1] + u2 * a[i__ + k * a_dim1];
	    a[i__ + k1 * a_dim1] += t * v1;
	    a[i__ + k * a_dim1] += t * v2;
	    t = b[i__ + k1 * b_dim1] + u2 * b[i__ + k * b_dim1];
	    b[i__ + k1 * b_dim1] += t * v1;
	    b[i__ + k * b_dim1] += t * v2;
/* L250: */
	}

	b[k1 + k * b_dim1] = 0.;
	if (! (matz)) {
	    goto L260;
	}

	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    t = z__[i__ + k1 * z_dim1] + u2 * z__[i__ + k * z_dim1];
	    z__[i__ + k1 * z_dim1] += t * v1;
	    z__[i__ + k * z_dim1] += t * v2;
/* L255: */
	}

L260:
	;
    }
/*     .......... END QZ STEP .......... */
    goto L70;
/*     .......... SET ERROR -- ALL EIGENVALUES HAVE NOT */
/*                CONVERGED AFTER 30*N ITERATIONS .......... */
L1000:
    *ierr = en;
/*     .......... SAVE EPSB FOR USE BY QZVAL AND QZVEC .......... */
L1001:
    if (n > 1) {
	b[n + b_dim1] = epsb;
    }
    return 0;
} /* qzit_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int
qzval(integer nm, integer n, doublereal *a, doublereal *b, doublereal *alfr, doublereal *alfi, doublereal *beta, logical matz, doublereal *z__)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, z_dim1, z_offset, i__1, i__2;
    doublereal d__1, d__2, d__3, d__4;

    /* Local variables */
    static doublereal epsb, c__, d__, e;
    static integer i__, j;
    static doublereal r__, s, t, a1, a2, u1, u2, v1, v2, a11, a12, a21, a22, 
	    b11, b12, b22, di, ei;
    static integer na;
    static doublereal an, bn;
    static integer en;
    static doublereal cq, dr;
    static integer nn;
    static doublereal cz, ti, tr, a1i, a2i, a11i, a12i, a22i, a11r, a12r, 
	    a22r, sqi, ssi;
    static integer isw;
    static doublereal sqr, szi, ssr, szr;



/*     THIS SUBROUTINE IS THE THIRD STEP OF THE QZ ALGORITHM */
/*     FOR SOLVING GENERALIZED MATRIX EIGENVALUE PROBLEMS, */
/*     SIAM J. NUMER. ANAL. 10, 241-256(1973) BY MOLER AND STEWART. */

/*     THIS SUBROUTINE ACCEPTS A PAIR OF REAL MATRICES, ONE OF THEM */
/*     IN QUASI-TRIANGULAR FORM AND THE OTHER IN UPPER TRIANGULAR FORM. */
/*     IT REDUCES THE QUASI-TRIANGULAR MATRIX FURTHER, SO THAT ANY */
/*     REMAINING 2-BY-2 BLOCKS CORRESPOND TO PAIRS OF COMPLEX */
/*     EIGENVALUES, AND RETURNS QUANTITIES WHOSE RATIOS GIVE THE */
/*     GENERALIZED EIGENVALUES.  IT IS USUALLY PRECEDED BY  QZHES */
/*     AND  QZIT  AND MAY BE FOLLOWED BY  QZVEC. */

/*     ON INPUT */

/*        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL */
/*          DIMENSION STATEMENT. */

/*        N IS THE ORDER OF THE MATRICES. */

/*        A CONTAINS A REAL UPPER QUASI-TRIANGULAR MATRIX. */

/*        B CONTAINS A REAL UPPER TRIANGULAR MATRIX.  IN ADDITION, */
/*          LOCATION B(N,1) CONTAINS THE TOLERANCE QUANTITY (EPSB) */
/*          COMPUTED AND SAVED IN  QZIT. */

/*        MATZ SHOULD BE SET TO .TRUE. IF THE RIGHT HAND TRANSFORMATIONS 
*/
/*          ARE TO BE ACCUMULATED FOR LATER USE IN COMPUTING */
/*          EIGENVECTORS, AND TO .FALSE. OTHERWISE. */

/*        Z CONTAINS, IF MATZ HAS BEEN SET TO .TRUE., THE */
/*          TRANSFORMATION MATRIX PRODUCED IN THE REDUCTIONS BY QZHES */
/*          AND QZIT, IF PERFORMED, OR ELSE THE IDENTITY MATRIX. */
/*          IF MATZ HAS BEEN SET TO .FALSE., Z IS NOT REFERENCED. */

/*     ON OUTPUT */

/*        A HAS BEEN REDUCED FURTHER TO A QUASI-TRIANGULAR MATRIX */
/*          IN WHICH ALL NONZERO SUBDIAGONAL ELEMENTS CORRESPOND TO */
/*          PAIRS OF COMPLEX EIGENVALUES. */

/*        B IS STILL IN UPPER TRIANGULAR FORM, ALTHOUGH ITS ELEMENTS */
/*          HAVE BEEN ALTERED.  B(N,1) IS UNALTERED. */

/*        ALFR AND ALFI CONTAIN THE REAL AND IMAGINARY PARTS OF THE */
/*          DIAGONAL ELEMENTS OF THE TRIANGULAR MATRIX THAT WOULD BE */
/*          OBTAINED IF A WERE REDUCED COMPLETELY TO TRIANGULAR FORM */
/*          BY UNITARY TRANSFORMATIONS.  NON-ZERO VALUES OF ALFI OCCUR */
/*          IN PAIRS, THE FIRST MEMBER POSITIVE AND THE SECOND NEGATIVE. 
*/

/*        BETA CONTAINS THE DIAGONAL ELEMENTS OF THE CORRESPONDING B, */
/*          NORMALIZED TO BE REAL AND NON-NEGATIVE.  THE GENERALIZED */
/*          EIGENVALUES ARE THEN THE RATIOS ((ALFR+I*ALFI)/BETA). */

/*        Z CONTAINS THE PRODUCT OF THE RIGHT HAND TRANSFORMATIONS */
/*          (FOR ALL THREE STEPS) IF MATZ HAS BEEN SET TO .TRUE. */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    z_dim1 = nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;
    --beta;
    --alfi;
    --alfr;
    b_dim1 = nm;
    b_offset = b_dim1 + 1;
    b -= b_offset;
    a_dim1 = nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    /* Function Body */
    epsb = b[n + b_dim1];
    isw = 1;
/*     .......... FIND EIGENVALUES OF QUASI-TRIANGULAR MATRICES. */
/*                FOR EN=N STEP -1 UNTIL 1 DO -- .......... */
    i__1 = n;
    for (nn = 1; nn <= i__1; ++nn) {
	en = n + 1 - nn;
	na = en - 1;
	if (isw == 2) {
	    goto L505;
	}
	if (en == 1) {
	    goto L410;
	}
	if (a[en + na * a_dim1] != 0.) {
	    goto L420;
	}
/*     .......... 1-BY-1 BLOCK, ONE REAL ROOT .......... */
L410:
	alfr[en] = a[en + en * a_dim1];
	if (b[en + en * b_dim1] < 0.) {
	    alfr[en] = -alfr[en];
	}
	beta[en] = (d__1 = b[en + en * b_dim1], fabs(d__1));
	alfi[en] = 0.;
	goto L510;
/*     .......... 2-BY-2 BLOCK .......... */
L420:
	if ((d__1 = b[na + na * b_dim1], fabs(d__1)) <= epsb) {
	    goto L455;
	}
	if ((d__1 = b[en + en * b_dim1], fabs(d__1)) > epsb) {
	    goto L430;
	}
	a1 = a[en + en * a_dim1];
	a2 = a[en + na * a_dim1];
	bn = 0.;
	goto L435;
L430:
	an = (d__1 = a[na + na * a_dim1], fabs(d__1)) + (d__2 = a[na + en * 
		a_dim1], fabs(d__2)) + (d__3 = a[en + na * a_dim1], fabs(d__3)) 
		+ (d__4 = a[en + en * a_dim1], fabs(d__4));
	bn = (d__1 = b[na + na * b_dim1], fabs(d__1)) + (d__2 = b[na + en * 
		b_dim1], fabs(d__2)) + (d__3 = b[en + en * b_dim1], fabs(d__3));
	a11 = a[na + na * a_dim1] / an;
	a12 = a[na + en * a_dim1] / an;
	a21 = a[en + na * a_dim1] / an;
	a22 = a[en + en * a_dim1] / an;
	b11 = b[na + na * b_dim1] / bn;
	b12 = b[na + en * b_dim1] / bn;
	b22 = b[en + en * b_dim1] / bn;
	e = a11 / b11;
	ei = a22 / b22;
	s = a21 / (b11 * b22);
	t = (a22 - e * b22) / b22;
	if (fabs(e) <= fabs(ei)) {
	    goto L431;
	}
	e = ei;
	t = (a11 - e * b11) / b11;
L431:
	c__ = (t - s * b12) * .5;
	d__ = c__ * c__ + s * (a12 - e * b12);
	if (d__ < 0.) {
	    goto L480;
	}
/*     .......... TWO REAL ROOTS. */
/*                ZERO BOTH A(EN,NA) AND B(EN,NA) .......... */
	d__1 = sqrt(d__);
	e += c__ + d_sign(d__1, c__);
	a11 -= e * b11;
	a12 -= e * b12;
	a22 -= e * b22;
	if (fabs(a11) + fabs(a12) < fabs(a21) + fabs(a22)) {
	    goto L432;
	}
	a1 = a12;
	a2 = a11;
	goto L435;
L432:
	a1 = a22;
	a2 = a21;
/*     .......... CHOOSE AND APPLY REAL Z .......... */
L435:
	s = fabs(a1) + fabs(a2);
	u1 = a1 / s;
	u2 = a2 / s;
	d__1 = sqrt(u1 * u1 + u2 * u2);
	r__ = d_sign(d__1, u1);
	v1 = -(u1 + r__) / r__;
	v2 = -u2 / r__;
	u2 = v2 / v1;

	i__2 = en;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    t = a[i__ + en * a_dim1] + u2 * a[i__ + na * a_dim1];
	    a[i__ + en * a_dim1] += t * v1;
	    a[i__ + na * a_dim1] += t * v2;
	    t = b[i__ + en * b_dim1] + u2 * b[i__ + na * b_dim1];
	    b[i__ + en * b_dim1] += t * v1;
	    b[i__ + na * b_dim1] += t * v2;
/* L440: */
	}

	if (! (matz)) {
	    goto L450;
	}

	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    t = z__[i__ + en * z_dim1] + u2 * z__[i__ + na * z_dim1];
	    z__[i__ + en * z_dim1] += t * v1;
	    z__[i__ + na * z_dim1] += t * v2;
/* L445: */
	}

L450:
	if (bn == 0.) {
	    goto L475;
	}
	if (an < fabs(e) * bn) {
	    goto L455;
	}
	a1 = b[na + na * b_dim1];
	a2 = b[en + na * b_dim1];
	goto L460;
L455:
	a1 = a[na + na * a_dim1];
	a2 = a[en + na * a_dim1];
/*     .......... CHOOSE AND APPLY REAL Q .......... */
L460:
	s = fabs(a1) + fabs(a2);
	if (s == 0.) {
	    goto L475;
	}
	u1 = a1 / s;
	u2 = a2 / s;
	d__1 = sqrt(u1 * u1 + u2 * u2);
	r__ = d_sign(d__1, u1);
	v1 = -(u1 + r__) / r__;
	v2 = -u2 / r__;
	u2 = v2 / v1;

	i__2 = n;
	for (j = na; j <= i__2; ++j) {
	    t = a[na + j * a_dim1] + u2 * a[en + j * a_dim1];
	    a[na + j * a_dim1] += t * v1;
	    a[en + j * a_dim1] += t * v2;
	    t = b[na + j * b_dim1] + u2 * b[en + j * b_dim1];
	    b[na + j * b_dim1] += t * v1;
	    b[en + j * b_dim1] += t * v2;
/* L470: */
	}

L475:
	a[en + na * a_dim1] = 0.;
	b[en + na * b_dim1] = 0.;
	alfr[na] = a[na + na * a_dim1];
	alfr[en] = a[en + en * a_dim1];
	if (b[na + na * b_dim1] < 0.) {
	    alfr[na] = -alfr[na];
	}
	if (b[en + en * b_dim1] < 0.) {
	    alfr[en] = -alfr[en];
	}
	beta[na] = (d__1 = b[na + na * b_dim1], fabs(d__1));
	beta[en] = (d__1 = b[en + en * b_dim1], fabs(d__1));
	alfi[en] = 0.;
	alfi[na] = 0.;
	goto L505;
/*     .......... TWO COMPLEX ROOTS .......... */
L480:
	e += c__;
	ei = sqrt(-d__);
	a11r = a11 - e * b11;
	a11i = ei * b11;
	a12r = a12 - e * b12;
	a12i = ei * b12;
	a22r = a22 - e * b22;
	a22i = ei * b22;
	if (fabs(a11r) + fabs(a11i) + fabs(a12r) + fabs(a12i) < fabs(a21) + fabs(
		a22r) + fabs(a22i)) {
	    goto L482;
	}
	a1 = a12r;
	a1i = a12i;
	a2 = -a11r;
	a2i = -a11i;
	goto L485;
L482:
	a1 = a22r;
	a1i = a22i;
	a2 = -a21;
	a2i = 0.;
/*     .......... CHOOSE COMPLEX Z .......... */
L485:
	cz = sqrt(a1 * a1 + a1i * a1i);
	if (cz == 0.) {
	    goto L487;
	}
	szr = (a1 * a2 + a1i * a2i) / cz;
	szi = (a1 * a2i - a1i * a2) / cz;
	r__ = sqrt(cz * cz + szr * szr + szi * szi);
	cz /= r__;
	szr /= r__;
	szi /= r__;
	goto L490;
L487:
	szr = 1.;
	szi = 0.;
L490:
	if (an < (fabs(e) + ei) * bn) {
	    goto L492;
	}
	a1 = cz * b11 + szr * b12;
	a1i = szi * b12;
	a2 = szr * b22;
	a2i = szi * b22;
	goto L495;
L492:
	a1 = cz * a11 + szr * a12;
	a1i = szi * a12;
	a2 = cz * a21 + szr * a22;
	a2i = szi * a22;
/*     .......... CHOOSE COMPLEX Q .......... */
L495:
	cq = sqrt(a1 * a1 + a1i * a1i);
	if (cq == 0.) {
	    goto L497;
	}
	sqr = (a1 * a2 + a1i * a2i) / cq;
	sqi = (a1 * a2i - a1i * a2) / cq;
	r__ = sqrt(cq * cq + sqr * sqr + sqi * sqi);
	cq /= r__;
	sqr /= r__;
	sqi /= r__;
	goto L500;
L497:
	sqr = 1.;
	sqi = 0.;
/*     .......... COMPUTE DIAGONAL ELEMENTS THAT WOULD RESULT */
/*                IF TRANSFORMATIONS WERE APPLIED .......... */
L500:
	ssr = sqr * szr + sqi * szi;
	ssi = sqr * szi - sqi * szr;
	i__ = 1;
	tr = cq * cz * a11 + cq * szr * a12 + sqr * cz * a21 + ssr * a22;
	ti = cq * szi * a12 - sqi * cz * a21 + ssi * a22;
	dr = cq * cz * b11 + cq * szr * b12 + ssr * b22;
	di = cq * szi * b12 + ssi * b22;
	goto L503;
L502:
	i__ = 2;
	tr = ssr * a11 - sqr * cz * a12 - cq * szr * a21 + cq * cz * a22;
	ti = -ssi * a11 - sqi * cz * a12 + cq * szi * a21;
	dr = ssr * b11 - sqr * cz * b12 + cq * cz * b22;
	di = -ssi * b11 - sqi * cz * b12;
L503:
	t = ti * dr - tr * di;
	j = na;
	if (t < 0.) {
	    j = en;
	}
	r__ = sqrt(dr * dr + di * di);
	beta[j] = bn * r__;
	alfr[j] = an * (tr * dr + ti * di) / r__;
	alfi[j] = an * t / r__;
	if (i__ == 1) {
	    goto L502;
	}
L505:
	isw = 3 - isw;
L510:
	;
    }
    b[n + b_dim1] = epsb;

    return 0;
} /* qzval_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
doublereal 
epslon(doublereal x)
{
    /* System generated locals */
    doublereal ret_val, d__1;

    /* Local variables */
    static doublereal a, b, c__, eps;


/*     ESTIMATE UNIT ROUNDOFF IN QUANTITIES OF SIZE X. */


/*     THIS PROGRAM SHOULD FUNCTION PROPERLY ON ALL SYSTEMS */
/*     SATISFYING THE FOLLOWING TWO ASSUMPTIONS, */
/*        1.  THE BASE USED IN REPRESENTING FLOATING POINT */
/*            NUMBERS IS NOT A POWER OF THREE. */
/*        2.  THE QUANTITY  A  IN STATEMENT 10 IS REPRESENTED TO */
/*            THE ACCURACY USED IN FLOATING POINT VARIABLES */
/*            THAT ARE STORED IN MEMORY. */
/*     THE STATEMENT NUMBER 10 AND THE GO TO 10 ARE INTENDED TO */
/*     FORCE OPTIMIZING COMPILERS TO GENERATE CODE SATISFYING */
/*     ASSUMPTION 2. */
/*     UNDER THESE ASSUMPTIONS, IT SHOULD BE TRUE THAT, */
/*            A  IS NOT EXACTLY EQUAL TO FOUR-THIRDS, */
/*            B  HAS A ZERO FOR ITS LAST BIT OR DIGIT, */
/*            C  IS NOT EXACTLY EQUAL TO ONE, */
/*            EPS  MEASURES THE SEPARATION OF 1.0 FROM */
/*                 THE NEXT LARGER FLOATING POINT NUMBER. */
/*     THE DEVELOPERS OF EISPACK WOULD APPRECIATE BEING INFORMED */
/*     ABOUT ANY SYSTEMS WHERE THESE ASSUMPTIONS DO NOT HOLD. */

/*     THIS VERSION DATED 4/6/83. */

    a = 1.3333333333333333;
L10:
    b = a - 1.;
    c__ = b + b + b;
    eps = (d__1 = c__ - 1., fabs(d__1));
    if (eps == 0.) {
	goto L10;
    }
    ret_val = eps * fabs(x);
    return ret_val;
} /* epslon_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  BLAS-1 routines needed in the computation of Floquet multipliers */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
doublereal 
dnrm2(integer *n, doublereal *dx, integer *incx)
{
    /* Initialized data */

#define zero 0.
#define one  1.
#define cutlo 8.232e-11
#define cuthi 1.304e19

    /* Format strings */

    /* System generated locals */
    integer i__1, i__2;
    doublereal ret_val, d__1;

    /* Local variables */
    static doublereal xmax;
    static integer next, i__, j, nn;
    static doublereal hitest, sum;

    /* Parameter adjustments */
    --dx;

    /* Function Body */

/*     EUCLIDEAN NORM OF THE N-VECTOR STORED IN DX() WITH STORAGE */
/*     INCREMENT INCX . */
/*     IF    N .LE. 0 RETURN WITH RESULT = 0. */
/*     IF N .GE. 1 THEN INCX MUST BE .GE. 1 */

/*           C.L.LAWSON, 1978 JAN 08 */

/*     FOUR PHASE METHOD     USING TWO BUILT-IN CONSTANTS THAT ARE */
/*     HOPEFULLY APPLICABLE TO ALL MACHINES. */
/*         CUTLO = MAXIMUM OF  DSQRT(U/EPS)  OVER ALL KNOWN MACHINES. */
/*         CUTHI = MINIMUM OF  DSQRT(V)      OVER ALL KNOWN MACHINES. */
/*     WHERE */
/*         EPS = SMALLEST NO. SUCH THAT EPS + 1. .GT. 1. */
/*         U   = SMALLEST POSITIVE NO.   (UNDERFLOW LIMIT) */
/*         V   = LARGEST  NO.            (OVERFLOW  LIMIT) */

/*     BRIEF OUTLINE OF ALGORITHM.. */

/*     PHASE 1    SCANS ZERO COMPONENTS. */
/*     MOVE TO PHASE 2 WHEN A COMPONENT IS NONZERO AND .LE. CUTLO */
/*     MOVE TO PHASE 3 WHEN A COMPONENT IS .GT. CUTLO */
/*     MOVE TO PHASE 4 WHEN A COMPONENT IS .GE. CUTHI/M */
/*     WHERE M = N FOR X() REAL AND M = 2*N FOR COMPLEX. */

/*     VALUES FOR CUTLO AND CUTHI.. */
/*     DOCUMENT THE LIMITING VALUES ARE AS FOLLOWS.. */
/*     CUTLO, S.P.   U/EPS = 2**(-102) FOR  HONEYWELL.  CLOSE SECONDS ARE 
*/
/*                   UNIVAC AND DEC AT 2**(-103) */
/*                   THUS CUTLO = 2**(-51) = 4.44089E-16 */
/*     CUTHI, S.P.   V = 2**127 FOR UNIVAC, HONEYWELL, AND DEC. */
/*                   THUS CUTHI = 2**(63.5) = 1.30438E19 */
/*     CUTLO, D.P.   U/EPS = 2**(-67) FOR HONEYWELL AND DEC. */
/*                   THUS CUTLO = 2**(-33.5) = 8.23181D-11 */
/*     CUTHI, D.P.   SAME AS S.P.  CUTHI = 1.30438D19 */
/*     DATA CUTLO, CUTHI / 8.232D-11,  1.304D19 / */
/*     DATA CUTLO, CUTHI / 4.441E-16,  1.304E19 / */

    if (*n > 0) {
	goto L10;
    }
    ret_val = zero;
    goto L300;

L10:
    next = 0;
    sum = zero;
    nn = *n * *incx;
/*                                                 BEGIN MAIN LOOP */
    i__ = 1;
L20:
    switch ((int)next) {
	case 0: goto L30;
	case 1: goto L50;
	case 2: goto L70;
	case 3: goto L110;
    }
L30:
    if ((d__1 = dx[i__], fabs(d__1)) > cutlo) {
	goto L85;
    }
    next = 1;
    xmax = zero;

/*                        PHASE 1.  SUM IS ZERO */

L50:
    if (dx[i__] == zero) {
	goto L200;
    }
    if ((d__1 = dx[i__], fabs(d__1)) > cutlo) {
	goto L85;
    }

/*                                PREPARE FOR PHASE 2. */
    next = 2;
    goto L105;

/*                                PREPARE FOR PHASE 4. */

L100:
    i__ = j;
    next = 3;
    sum = sum / dx[i__] / dx[i__];
L105:
    xmax = (d__1 = dx[i__], fabs(d__1));
    goto L115;

/*                   PHASE 2.  SUM IS SMALL. */
/*                             SCALE TO AVOID DESTRUCTIVE UNDERFLOW. */

L70:
    if ((d__1 = dx[i__], fabs(d__1)) > cutlo) {
	goto L75;
    }

/*                     COMMON CODE FOR PHASES 2 AND 4. */
/*                     IN PHASE 4 SUM IS LARGE.  SCALE TO AVOID OVERFLOW. 
*/

L110:
    if ((d__1 = dx[i__], fabs(d__1)) <= xmax) {
	goto L115;
    }
/* Computing 2nd power */
    d__1 = xmax / dx[i__];
    sum = one + sum * (d__1 * d__1);
    xmax = (d__1 = dx[i__], fabs(d__1));
    goto L200;

L115:
/* Computing 2nd power */
    d__1 = dx[i__] / xmax;
    sum += d__1 * d__1;
    goto L200;


/*                  PREPARE FOR PHASE 3. */

L75:
    sum = sum * xmax * xmax;


/*     FOR REAL OR D.P. SET HITEST = CUTHI/N */
/*     FOR COMPLEX      SET HITEST = CUTHI/(2*N) */

L85:
    hitest = cuthi / (real) (*n);

/*                   PHASE 3.  SUM IS MID-RANGE.  NO SCALING. */

    i__1 = nn;
    i__2 = *incx;
    for (j = i__; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
	if ((d__1 = dx[j], fabs(d__1)) >= hitest) {
	    goto L100;
	}
/* L95: */
/* Computing 2nd power */
	d__1 = dx[j];
	sum += d__1 * d__1;
    }
    ret_val = sqrt(sum);
    goto L300;

L200:
    i__ += *incx;
    if (i__ <= nn) {
	goto L20;
    }

/*              END OF MAIN LOOP. */

/*              COMPUTE SQUARE ROOT AND ADJUST FOR SCALING. */

    ret_val = xmax * sqrt(sum);
L300:
    return ret_val;
} /* dnrm2_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
doublereal 
ddot(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;
    doublereal ret_val;

    /* Local variables */
    static integer i__, m;
    static doublereal dtemp;
    static integer ix, iy, mp1;


/*     FORMS THE DOT PRODUCT OF TWO VECTORS. */
/*     USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE. */
/*     JACK DONGARRA, LINPACK, 3/11/78. */


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    ret_val = 0.;
    dtemp = 0.;
    if (*n <= 0) {
	return ret_val;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*        CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS */
/*          NOT EQUAL TO 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp += dx[ix] * dy[iy];
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    ret_val = dtemp;
    return ret_val;

/*        CODE FOR BOTH INCREMENTS EQUAL TO 1 */


/*        CLEAN-UP LOOP */

L20:
    m = *n % 5;
    if (m == 0) {
	goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp += dx[i__] * dy[i__];
/* L30: */
    }
    if (*n < 5) {
	goto L60;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 5) {
	dtemp = dtemp + dx[i__] * dy[i__] + dx[i__ + 1] * dy[i__ + 1] + dx[
		i__ + 2] * dy[i__ + 2] + dx[i__ + 3] * dy[i__ + 3] + dx[i__ + 
		4] * dy[i__ + 4];
/* L50: */
    }
L60:
    ret_val = dtemp;
    return ret_val;
} /* ddot_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
dscal(integer *n, doublereal *da, doublereal *dx, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__, m, nincx, mp1;


/*     SCALES A VECTOR BY A CONSTANT. */
/*     USES UNROLLED LOOPS FOR INCREMENT EQUAL TO ONE. */
/*     JACK DONGARRA, LINPACK, 3/11/78. */


    /* Parameter adjustments */
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        CODE FOR INCREMENT NOT EQUAL TO 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	dx[i__] = *da * dx[i__];
/* L10: */
    }
    return 0;

/*        CODE FOR INCREMENT EQUAL TO 1 */


/*        CLEAN-UP LOOP */

L20:
    m = *n % 5;
    if (m == 0) {
	goto L40;
    }
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	dx[i__] = *da * dx[i__];
/* L30: */
    }
    if (*n < 5) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__2 = *n;
    for (i__ = mp1; i__ <= i__2; i__ += 5) {
	dx[i__] = *da * dx[i__];
	dx[i__ + 1] = *da * dx[i__ + 1];
	dx[i__ + 2] = *da * dx[i__ + 2];
	dx[i__ + 3] = *da * dx[i__ + 3];
	dx[i__ + 4] = *da * dx[i__ + 4];
/* L50: */
    }
    return 0;
} /* dscal_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
integer 
idamax(integer *n, doublereal *dx, integer *incx)
{
    /* System generated locals */
    integer ret_val, i__1;
    doublereal d__1;

    /* Local variables */
    static doublereal dmax__;
    static integer i__, ix;


/*     FINDS THE INDEX OF ELEMENT HAVING MAX. ABSOLUTE VALUE. */
/*     JACK DONGARRA, LINPACK, 3/11/78. */


    /* Parameter adjustments */
    --dx;

    /* Function Body */
    ret_val = 0;
    if (*n < 1) {
	return ret_val;
    }
    ret_val = 1;
    if (*n == 1) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        CODE FOR INCREMENT NOT EQUAL TO 1 */

    ix = 1;
    dmax__ = fabs(dx[1]);
    ix += *incx;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if ((d__1 = dx[ix], fabs(d__1)) <= dmax__) {
	    goto L5;
	}
	ret_val = i__;
	dmax__ = (d__1 = dx[ix], fabs(d__1));
L5:
	ix += *incx;
/* L10: */
    }
    return ret_val;

/*        CODE FOR INCREMENT EQUAL TO 1 */

L20:
    dmax__ = fabs(dx[1]);
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if ((d__1 = dx[i__], fabs(d__1)) <= dmax__) {
	    goto L30;
	}
	ret_val = i__;
	dmax__ = (d__1 = dx[i__], fabs(d__1));
L30:
	;
    }
    return ret_val;
} /* idamax_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
daxpy(integer *n, doublereal *da, doublereal *dx, integer *incx, doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;


/*     CONSTANT TIMES A VECTOR PLUS A VECTOR. */
/*     USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE. */
/*     JACK DONGARRA, LINPACK, 3/11/78. */


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*da == 0.) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*        CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS */
/*          NOT EQUAL TO 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dy[iy] += *da * dx[ix];
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*        CODE FOR BOTH INCREMENTS EQUAL TO 1 */


/*        CLEAN-UP LOOP */

L20:
    m = *n % 4;
    if (m == 0) {
	goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dy[i__] += *da * dx[i__];
/* L30: */
    }
    if (*n < 4) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 4) {
	dy[i__] += *da * dx[i__];
	dy[i__ + 1] += *da * dx[i__ + 1];
	dy[i__ + 2] += *da * dx[i__ + 2];
	dy[i__ + 3] += *da * dx[i__ + 3];
/* L50: */
    }
    return 0;
} /* daxpy_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
drot(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy, doublereal *c__, doublereal *s)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;
    static doublereal dtemp;
    static integer ix, iy;


/*     APPLIES A PLANE ROTATION. */
/*     JACK DONGARRA, LINPACK, 3/11/78. */


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*       CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS NOT EQUAL */
/*         TO 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = *c__ * dx[ix] + *s * dy[iy];
	dy[iy] = *c__ * dy[iy] - *s * dx[ix];
	dx[ix] = dtemp;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       CODE FOR BOTH INCREMENTS EQUAL TO 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = *c__ * dx[i__] + *s * dy[i__];
	dy[i__] = *c__ * dy[i__] - *s * dx[i__];
	dx[i__] = dtemp;
/* L30: */
    }
    return 0;
} /* drot_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
dswap(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m;
    static doublereal dtemp;
    static integer ix, iy, mp1;


/*     INTERCHANGES TWO VECTORS. */
/*     USES UNROLLED LOOPS FOR INCREMENTS EQUAL ONE. */
/*     JACK DONGARRA, LINPACK, 3/11/78. */


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*       CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS NOT EQUAL */
/*         TO 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = dx[ix];
	dx[ix] = dy[iy];
	dy[iy] = dtemp;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       CODE FOR BOTH INCREMENTS EQUAL TO 1 */


/*       CLEAN-UP LOOP */

L20:
    m = *n % 3;
    if (m == 0) {
	goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = dx[i__];
	dx[i__] = dy[i__];
	dy[i__] = dtemp;
/* L30: */
    }
    if (*n < 3) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 3) {
	dtemp = dx[i__];
	dx[i__] = dy[i__];
	dy[i__] = dtemp;
	dtemp = dx[i__ + 1];
	dx[i__ + 1] = dy[i__ + 1];
	dy[i__ + 1] = dtemp;
	dtemp = dx[i__ + 2];
	dx[i__ + 2] = dy[i__ + 2];
	dy[i__ + 2] = dtemp;
/* L50: */
    }
    return 0;
} /* dswap_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
dgemc(integer *m, integer *n, doublereal *a, integer *lda, doublereal *b, integer *ldb, logical *trans)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, i1, i2;

    /* Local variables */
    integer i, j, mm, mmp1;



/*  This subroutine copies a double precision real */
/*  M by N matrix stored in A to double precision real B. */
/*  If TRANS is true, B is assigned A transpose. */



    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = a_dim1 + 1;
    a -= a_offset;
    b_dim1 = *ldb;

    /* Function Body */
    if (*trans) {
	i1 = *n;
	for (j = 1; j <= i1; ++j) {

/*         USES UNROLLED LOOPS */
/*         from JACK DONGARRA, LINPACK, 3/11/78. */

	    mm = *m % 7;
	    if (mm == 0) {
		goto L80;
	    }
	    i2 = mm;
	    for (i = 1; i <= i2; ++i) {
		b[j-1 + (i-1) * b_dim1] = a[i + j * a_dim1];
/* L70: */
	    }
	    if (*m < 7) {
		goto L99;
	    }
L80:
	    mmp1 = mm + 1;
	    i2 = *m;
	    for (i = mmp1; i <= i2; i += 7) {
		b[j-1 + (i-1) * b_dim1] = a[i + j * a_dim1];
		b[j-1 + (i-1 + 1) * b_dim1] = a[i + 1 + j * a_dim1];
		b[j-1 + (i-1 + 2) * b_dim1] = a[i + 2 + j * a_dim1];
		b[j-1 + (i-1 + 3) * b_dim1] = a[i + 3 + j * a_dim1];
		b[j-1 + (i-1 + 4) * b_dim1] = a[i + 4 + j * a_dim1];
		b[j-1 + (i-1 + 5) * b_dim1] = a[i + 5 + j * a_dim1];
		b[j-1 + (i-1 + 6) * b_dim1] = a[i + 6 + j * a_dim1];
/* L90: */
	    }
L99:
/* L100: */
	    ;
	}
    } else {
	i1 = *n;
	for (j = 1; j <= i1; ++j) {

/*         USES UNROLLED LOOPS */
/*         from JACK DONGARRA, LINPACK, 3/11/78. */

	    mm = *m % 7;
	    if (mm == 0) {
		goto L180;
	    }
	    i2 = mm;
	    for (i = 1; i <= i2; ++i) {
		b[(j-1) * b_dim1 + i-1] = a[i + j * a_dim1];
/* L170: */
	    }
	    if (*m < 7) {
		goto L199;
	    }
L180:
	    mmp1 = mm + 1;
	    i2 = *m;
	    for (i = mmp1; i <= i2; i += 7) {
		b[i-1 + (j-1) * b_dim1] = a[i + j * a_dim1];
		b[i-1 + 1 + (j-1) * b_dim1] = a[i + 1 + j * a_dim1];
		b[i-1 + 2 + (j-1) * b_dim1] = a[i + 2 + j * a_dim1];
		b[i-1 + 3 + (j-1) * b_dim1] = a[i + 3 + j * a_dim1];
		b[i-1 + 4 + (j-1) * b_dim1] = a[i + 4 + j * a_dim1];
		b[i-1 + 5 + (j-1) * b_dim1] = a[i + 5 + j * a_dim1];
		b[i-1 + 6 + (j-1) * b_dim1] = a[i + 6 + j * a_dim1];
/* L190: */
	    }
L199:
/* L200: */
	    ;
	}
    }
    return 0;
} /* dgemc_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  BLAS-2 routines needed in the computation of Floquet multipliers */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
xerbla(char *srname, integer *info, integer srname_len)
{
    /* Format strings */

    /* Builtin functions */

    /* Fortran I/O blocks */


/*     ..    Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  XERBLA  is an error handler for the Level 2 BLAS routines. */

/*  It is called by the Level 2 BLAS routines if an input parameter is */
/*  invalid. */

/*  Installers should consider modifying the STOP statement in order to */
/*  call system-specific exception-handling facilities. */

/*  Parameters */
/*  ========== */

/*  SRNAME - CHARACTER*6. */
/*           On entry, SRNAME specifies the name of the routine which */
/*           called XERBLA. */

/*  INFO   - INTEGER. */
/*           On entry, INFO specifies the position of the invalid */
/*           parameter in the parameter-list of the calling routine. */


/*  Auxiliary routine for Level 2 Blas. */

/*  Written on 20-July-1986. */

/*     .. Executable Statements .. */

    printf("On entry to %c%c%c%c%c%c parameter number %ld had an illegal value\n",
	   srname[0],srname[1],srname[2],srname[3],srname[4],srname[5],(*info));
    exit(0);
    return 0;

/*     End of XERBLA. */

} /* xerbla_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
logical 
lsame(char *ca, char *cb, integer ca_len, integer cb_len)
{
    /* System generated locals */
    logical ret_val;

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  LSAME  tests if CA is the same letter as CB regardless of case. */
/*  CB is assumed to be an upper case letter. LSAME returns .TRUE. if */
/*  CA is either the same as CB or the equivalent lower case letter. */

/*  N.B. This version of the routine is only correct for ASCII code. */
/*       Installers must modify the routine for other character-codes. */

/*       For EBCDIC systems the constant IOFF must be changed to -64. */
/*       For CDC systems using 6-12 bit representations, the system- */
/*       specific code in comments must be activated. */

/*  Parameters */
/*  ========== */

/*  CA     - CHARACTER*1 */
/*  CB     - CHARACTER*1 */
/*           On entry, CA and CB specify characters to be compared. */
/*           Unchanged on exit. */


/*  Auxiliary routine for Level 2 Blas. */

/*  -- Written on 20-July-1986 */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, Nag Central Office. */

/*     .. Parameters .. */
/*     .. Intrinsic Functions .. */
/*     .. Executable Statements .. */

/*     Test if the characters are equal */

    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;

/*     Now test for equivalence */

    if (! ret_val) {
	ret_val = *(unsigned char *)ca - 32 == *(unsigned char *)cb;
    }

    return ret_val;

/*  The following comments contain code for CDC systems using 6-12 bit */
/*  representations. */

/*     .. Parameters .. */
/*     INTEGER                ICIRFX */
/*     .. Scalar Arguments .. */
/*     CHARACTER*1            CB */
/*     .. Array Arguments .. */
/*     CHARACTER*1            CA(*) */
/*     .. Local Scalars .. */
/*     INTEGER                IVAL */
/*     .. Intrinsic Functions .. */
/*     INTRINSIC              ICHAR, CHAR */
/*     .. Executable Statements .. */

/*     See if the first character in string CA equals string CB. */

/*     LSAME = CA(1) .EQ. CB .AND. CA(1) .NE. CHAR(ICIRFX) */

/*     IF (LSAME) RETURN */

/*     The characters are not identical. Now check them for equivalence. 
*/
/*     Look for the 'escape' character, circumflex, followed by the */
/*     letter. */

/*     IVAL = ICHAR(CA(2)) */
/*     IF (IVAL.GE.ICHAR('A') .AND. IVAL.LE.ICHAR('Z')) THEN */
/*        LSAME = CA(1) .EQ. CHAR(ICIRFX) .AND. CA(2) .EQ. CB */
/*     END IF */

/*     RETURN */

/*     End of LSAME. */

} /* lsame_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*  BLAS-3 routines needed in the computation of Floquet multipliers */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
dgemm(char *transa, char *transb, integer *m, integer *n, integer *k, doublereal *alpha, doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *beta, doublereal *c__, integer *ldc, integer transa_len, integer transb_len)
{
    /* System generated locals */
    integer a_dim1, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer info;
    static logical nota, notb;
    static doublereal temp;
    static integer i__, j, l, ncola;

    static integer nrowa, nrowb;


/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGEMM  performs one of the matrix-matrix operations */

/*     C := alpha*op( A )*op( B ) + beta*C, */

/*  where  op( X ) is one of */

/*     op( X ) = X   or   op( X ) = X', */

/*  alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
*/
/*  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix. 
*/

/*  Parameters */
/*  ========== */

/*  TRANSA - CHARACTER*1. */
/*           On entry, TRANSA specifies the form of op( A ) to be used in 
*/
/*           the matrix multiplication as follows: */

/*              TRANSA = 'N' or 'n',  op( A ) = A. */

/*              TRANSA = 'T' or 't',  op( A ) = A'. */

/*              TRANSA = 'C' or 'c',  op( A ) = A'. */

/*           Unchanged on exit. */

/*  TRANSB - CHARACTER*1. */
/*           On entry, TRANSB specifies the form of op( B ) to be used in 
*/
/*           the matrix multiplication as follows: */

/*              TRANSB = 'N' or 'n',  op( B ) = B. */

/*              TRANSB = 'T' or 't',  op( B ) = B'. */

/*              TRANSB = 'C' or 'c',  op( B ) = B'. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry,  M  specifies  the number  of rows  of the  matrix 
*/
/*           op( A )  and of the  matrix  C.  M  must  be at least  zero. 
*/
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry,  N  specifies the number  of columns of the matrix 
*/
/*           op( B ) and the number of columns of the matrix C. N must be 
*/
/*           at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry,  K  specifies  the number of columns of the matrix 
*/
/*           op( A ) and the number of rows of the matrix op( B ). K must 
*/
/*           be at least  zero. */
/*           Unchanged on exit. */

/*  ALPHA  - DOUBLE PRECISION. */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is 
*/
/*           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise. */
/*           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k 
*/
/*           part of the array  A  must contain the matrix  A,  otherwise 
*/
/*           the leading  k by m  part of the array  A  must contain  the 
*/
/*           matrix A. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared 
*/
/*           in the calling (sub) program. When  TRANSA = 'N' or 'n' then 
*/
/*           LDA must be at least  max( 1, m ), otherwise  LDA must be at 
*/
/*           least  max( 1, k ). */
/*           Unchanged on exit. */

/*  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is 
*/
/*           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise. */
/*           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n 
*/
/*           part of the array  B  must contain the matrix  B,  otherwise 
*/
/*           the leading  n by k  part of the array  B  must contain  the 
*/
/*           matrix B. */
/*           Unchanged on exit. */

/*  LDB    - INTEGER. */
/*           On entry, LDB specifies the first dimension of B as declared 
*/
/*           in the calling (sub) program. When  TRANSB = 'N' or 'n' then 
*/
/*           LDB must be at least  max( 1, k ), otherwise  LDB must be at 
*/
/*           least  max( 1, n ). */
/*           Unchanged on exit. */

/*  BETA   - DOUBLE PRECISION. */
/*           On entry,  BETA  specifies the scalar  beta.  When  BETA  is 
*/
/*           supplied as zero then C need not be set on input. */
/*           Unchanged on exit. */

/*  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ). */
/*           Before entry, the leading  m by n  part of the array  C must 
*/
/*           contain the matrix  C,  except when  beta  is zero, in which 
*/
/*           case C need not be set on entry. */
/*           On exit, the array  C  is overwritten by the  m by n  matrix 
*/
/*           ( alpha*op( A )*op( B ) + beta*C ). */

/*  LDC    - INTEGER. */
/*           On entry, LDC specifies the first dimension of C as declared 
*/
/*           in  the  calling  (sub)  program.   LDC  must  be  at  least 
*/
/*           max( 1, m ). */
/*           Unchanged on exit. */


/*  Level 3 Blas routine. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */


/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Local Scalars .. */
/*     .. Parameters .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not 
*/
/*     transposed and set  NROWA, NCOLA and  NROWB  as the number of rows 
*/
/*     and  columns of  A  and the  number of  rows  of  B  respectively. 
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    b_dim1 = *ldb;
    b_offset = b_dim1 + 1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = c_dim1 + 1;
    c__ -= c_offset;

    /* Function Body */
    nota = lsame(transa, "N", 1L, 1L);
    notb = lsame(transb, "N", 1L, 1L);
    if (nota) {
	nrowa = *m;
	ncola = *k;
    } else {
	nrowa = *k;
	ncola = *m;
    }
    if (notb) {
	nrowb = *k;
    } else {
	nrowb = *n;
    }

/*     Test the input parameters. */

    info = 0;
    if (! nota && ! lsame(transa, "C", 1L, 1L) && ! lsame(transa, "T", 1L, 
	    1L)) {
	info = 1;
    } else if (! notb && ! lsame(transb, "C", 1L, 1L) && ! lsame(transb, 
	    "T", 1L, 1L)) {
	info = 2;
    } else if (*m < 0) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < max(1,nrowa)) {
	info = 8;
    } else if (*ldb < max(1,nrowb)) {
	info = 10;
    } else if (*ldc < max(1,*m)) {
	info = 13;
    }
    if (info != 0) {
	xerbla("DGEMM ", &info, 6L);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || ((*alpha == 0. || *k == 0) && *beta == 1.)) {
	return 0;
    }

/*     And if  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (*beta == 0.) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = 0.;
/* L10: */
		}
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L30: */
		}
/* L40: */
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (notb) {
	if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L50: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L60: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (b[l + j * b_dim1] != 0.) {
			temp = *alpha * b[l + j * b_dim1];
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__-1 + (l-1) * a_dim1];
/* L70: */
			}
		    }
/* L80: */
		}
/* L90: */
	    }
	} else {

/*           Form  C := alpha*A'*B + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l-1 + (i__-1) * a_dim1] * b[l + j * b_dim1];
/* L100: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L110: */
		}
/* L120: */
	    }
	}
    } else {
	if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L130: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L140: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (b[j + l * b_dim1] != 0.) {
			temp = *alpha * b[j + l * b_dim1];
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__-1 + (l-1) * a_dim1];
/* L150: */
			}
		    }
/* L160: */
		}
/* L170: */
	    }
	} else {

/*           Form  C := alpha*A'*B' + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l-1 + (i__-1) * a_dim1] * b[j + l * b_dim1];
/* L180: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L190: */
		}
/* L200: */
	    }
	}
    }

    return 0;

/*     End of DGEMM . */

} /* dgemm_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Demmel-Kahan SVD routines needed for computing the Floquet multipliers */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
ezsvd(doublereal *x, integer *ldx, integer *n, integer *p, doublereal *s, doublereal *e, doublereal *u, integer *ldu, doublereal *v, integer *ldv, doublereal *work, integer *job, integer *info, doublereal *tol)
{
    /* System generated locals */
    integer x_dim1, x_offset, u_dim1, u_offset, v_dim1, v_offset;

    /* Local variables */
    static integer idbg, skip, iidir, ifull;

    static integer kount, kount1, kount2, limshf;
    static doublereal maxsin;
    static integer maxitr;



/*     new svd by J. Demmel, W. Kahan */
/*     finds singular values of bidiagonal matrices with guaranteed high 
*/
/*     relative precision */

/*     easy to use version of ndsvd ("hard to use" version, below) */
/*     with defaults for some ndsvd parameters */

/*     all parameters same as linpack dsvdc except for tol: */

/*     tol  = if positive, desired relative precision in singular values 
*/
/*            if negative, desired absolute precision in singular values 
*/
/*               (expressed as abs(tol) * sigma-max) */
/*            (in both cases, abs(tol) should be less than 1 and */
/*             greater than macheps) */

/*        I have tested this software on a SUN 3 in double precision */
/*        IEEE arithmetic with macheps about 2.2e-16 and tol=1e-14; */
/*        In general I recommend tol 10-100 times larger than macheps. */

/*        On the average it appears to be as fast or faster than dsvdc. */
/*        I have seen it go 3.5 times faster and 2 times slower at the */
/*        extremes. */

/*     defaults for ndsvd parameters (see ndsvd for more description of */
/*     these parameters) are: */

/*     set to no debug output */
    /* Parameter adjustments */
    x_dim1 = *ldx;
    x_offset = x_dim1 + 1;
    x -= x_offset;
    --s;
    --e;
    u_dim1 = *ldu;
    u_offset = u_dim1 + 1;
    u -= u_offset;
    v_dim1 = *ldv;
    v_offset = v_dim1 + 1;
    v -= v_offset;
    --work;

    /* Function Body */
    idbg = 0;
/*     use zero-shift normally */
    ifull = 0;
/*     use normal bidiagonalization code */
    skip = 0;
/*     choose chase direction normally */
    iidir = 0;
/*     maximum 30 QR sweeps per singular value */
    maxitr = 30;

    ndsvd(&x[x_offset], ldx, n, p, &s[1], &e[1], &u[u_offset], ldu, &v[
	    v_offset], ldv, &work[1], job, info, &maxitr, tol, &idbg, &ifull, 
	    &kount, &kount1, &kount2, &skip, &limshf, &maxsin, &iidir);
    return 0;
} /* ezsvd_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
ndrotg(doublereal *f, doublereal *g, doublereal *cs, doublereal *sn)
{

    /* Local variables */
    static doublereal t, tt;

/*     faster version of drotg, except g unchanged on return */
/*     cs, sn returned so that -sn*f+cs*g = 0 */
/*     and returned f = cs*f + sn*g */

/*     if g=0, then cs=1 and sn=0 (in case svd adds extra zero row */
/*         to bidiagonal, this makes sure last row rotation is trivial) */

/*     if f=0 and g.ne.0, then cs=0 and sn=1 without floating point work 
*/
/*         (in case s(i)=0 in svd so that bidiagonal deflates, this */
/*          computes rotation without any floating point operations) */

    if (*f == 0.) {
	if (*g == 0.) {
/*         this case needed in case extra zero row added in svd, s
o */
/*         bottom rotation always trivial */
	    *cs = 1.;
	    *sn = 0.;
	} else {
/*         this case needed for s(i)=0 in svd to compute rotation 
*/
/*         cheaply */
	    *cs = 0.;
	    *sn = 1.;
	    *f = *g;
	}
    } else {
	if (fabs(*f) > fabs(*g)) {
	    t = *g / *f;
	    tt = sqrt(t * t + 1.);
	    *cs = 1. / tt;
	    *sn = t * *cs;
	    *f *= tt;
	} else {
	    t = *f / *g;
	    tt = sqrt(t * t + 1.);
	    *sn = 1. / tt;
	    *cs = t * *sn;
	    *f = *g * tt;
	}
    }
    return 0;
} /* ndrotg_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
ndsvd(doublereal *x, integer *ldx, integer *n, integer *p, doublereal *s, doublereal *e, doublereal *u, integer *ldu, doublereal *v, integer *ldv, doublereal *work, integer *job, integer *info, integer *maxitr, doublereal *tol, integer *idbg, integer *ifull, integer *kount, integer *kount1, integer *kount2, integer *skip, integer *limshf, doublereal *maxsin, integer *iidir)
{
    /* System generated locals */
    integer x_dim1, x_offset, u_dim1, u_offset, v_dim1, v_offset, i__1, i__2, 
	    i__3;
    doublereal d__1, d__2, d__3, d__4;

    /* Local variables */
    static doublereal abse;

    static integer idir;
    static doublereal abss;

    static integer oldm, jobu;
    static doublereal cosl;
    static integer iter;
    static doublereal temp, smin, smax, cosr, sinl, sinr;

    static doublereal test;

    static integer nctp1, nrtp1;
    static doublereal f, g;
    static integer i__, j, k, l, m;
    static doublereal t;

    static doublereal oldcs;
    static integer oldll, iisub;
    static doublereal shift, oldsn, sigmn;
    static integer minnp, maxit;
    static doublereal sminl;

    static doublereal sigmx;
    static logical wantu, wantv;
    static doublereal gg, lambda;
    static integer oldacc;
    static doublereal cs;
    static integer ll, mm;
    static doublereal sm;
    static integer lu;
    static doublereal sn, mu;

    static doublereal thresh;

    static integer lm1, lp1, lll, nct, ncu;
    static doublereal sll;
    static integer nrt;
    static doublereal emm1, smm1;

    /* Fortran I/O blocks */



/*     LINPACK SVD modified by: */
/*     James Demmel                      W. Kahan */
/*     Courant Institute                 Computer Science Dept. */
/*     demmel@acf8.nyu.edu               U.C. Berkeley */

/*     modified version designed to guarantee relative accuracy of */
/*     all singular values of intermediate bidiagonal form */

/*    extra input/output parameters in addition to those from LINPACK SVD:
*/

/*     extra input paramters: */

/*     tol  = if positive, desired relative precision in singular values 
*/
/*            if negative, desired absolute precision in singular values 
*/
/*               (expressed as abs(tol) * sigma-max) */
/*            (abs(tol) should be less than 1 and greater than macheps) */

/*     idbg = 0 for no debug output (normal setting) */
/*          = 1 convergence, shift decisions (written to standard output) 
*/
/*          = 2 for above plus before, after qr */

/*    ifull= 0 if decision to use zero-shift set normally (normal setting)
*/
/*          = 1 if always set to nonzero-shift */
/*          = 2 if always set to zero-shift */

/*     skip =-1 means standard code but do all work of bidiagonalization 
*/
/*              (even if input bidiagonal) */
/*            0 means standard code (normal setting) */
/*            1 means assume x is bidiagonal, and skip bidiagonalization 
*/
/*              entirely */
/*          (skip used for timing tests) */

/*     iidir = 0 if idir (chase direction) chosen normally */
/*             1 if idir=1 (chase top to bottom) always */
/*             2 if idir=2 (chase bottom to top) always */

/*     extra output parameters: */

/*     kount =number of qr sweeps taken */

/*     kount1=number of passes through inner loop of full qr */

/*     kount2=number of passes through inner loop of zero-shift qr */

/*     limshf = number of times the shift was greater than its threshold 
*/
/*              (nct*smin) and had to be decreased */

/*     maxsin = maximum sin in inner loop of zero-shift */


/*     new version designed to be robust with respect to over/underflow */
/*     have fast inner loop when shift is zero, */
/*     guarantee relative accuracy of all singular values */

/*     dsvdc is a subroutine to reduce a double precision nxp matrix x */
/*     by orthogonal transformations u and v to diagonal form.  the */
/*     diagonal elements s(i) are the singular values of x.  the */
/*     columns of u are the corresponding left singular vectors, */
/*     and the columns of v the right singular vectors. */

/*     on entry */

/*         x         double precision(ldx,p), where ldx.ge.n. */
/*                   x contains the matrix whose singular value */
/*                   decomposition is to be computed.  x is */
/*                   destroyed by dsvdc. */

/*         ldx       integer. */
/*                   ldx is the leading dimension of the array x. */

/*         n         integer. */
/*                   n is the number of rows of the matrix x. */

/*         p         integer. */
/*                   p is the number of columns of the matrix x. */

/*         ldu       integer. */
/*                   ldu is the leading dimension of the array u. */
/*                   (see below). */

/*         ldv       integer. */
/*                   ldv is the leading dimension of the array v. */
/*                   (see below). */

/*         work      double precision(n). */
/*                   work is a scratch array. */

/*         job       integer. */
/*                   job controls the computation of the singular */
/*                   vectors.  it has the decimal expansion ab */
/*                   with the following meaning */

/*                        a.eq.0    do not compute the left singular */
/*                                  vectors. */
/*                        a.eq.1    return the n left singular vectors */
/*                                  in u. */
/*                        a.ge.2    return the first min(n,p) singular */
/*                                  vectors in u. */

/*     blas daxpy,ddot,dscal,dswap,dnrm2,drotg */
/*     fortran dabs,dmax1,max0,min0,mod,dsqrt,dsign */
/*     prse,ndrotg,sig22,sndrtg,sigmin */

/*     internal variables */


/*     new variables */
/*     double precision sg1,sg2 */


/*     set the maximum number of iterations. */

/*     maxit = 30 */
    /* Parameter adjustments */
    x_dim1 = *ldx;
    x_offset = x_dim1 + 1;
    x -= x_offset;
    --s;
    --e;
    u_dim1 = *ldu;
    u_offset = u_dim1 + 1;
    u -= u_offset;
    v_dim1 = *ldv;
    v_offset = v_dim1 + 1;
    v -= v_offset;
    --work;

    /* Function Body */
    *kount = 0;
    *kount1 = 0;
    *kount2 = 0;
    *limshf = 0;
    *maxsin = (double)0.;

/*     determine what is to be computed. */

    wantu = FALSE_;
    wantv = FALSE_;
    jobu = *job % 100 / 10;
    ncu = *n;
    if (jobu > 1) {
	ncu = min(*n,*p);
    }
    if (jobu != 0) {
	wantu = TRUE_;
    }
    if (*job % 10 != 0) {
	wantv = TRUE_;
    }

/*     reduce x to bidiagonal form, storing the diagonal elements */
/*     in s and the super-diagonal elements in e. */

    *info = 0;
/* Computing MIN */
    i__1 = *n - 1;
    nct = min(i__1,*p);
/* Computing MAX */
/* Computing MIN */
    i__3 = *p - 2;
    i__1 = 0, i__2 = min(i__3,*n);
    nrt = max(i__1,i__2);
    lu = max(nct,nrt);
    if (*skip <= 0) {
	if (lu < 1) {
	    goto L170;
	}
	i__1 = lu;
	for (l = 1; l <= i__1; ++l) {
            integer c__1 = 1;
	    lp1 = l + 1;
	    if (l > nct) {
		goto L20;
	    }

/*           compute the transformation for the l-th column and */
/*           place the l-th diagonal in s(l). */

	    i__2 = *n - l + 1;
	    s[l] = dnrm2(&i__2, &x[l + l * x_dim1], &c__1);
	    if (s[l] == 0. && *skip == 0) {
		goto L10;
	    }
	    if (x[l + l * x_dim1] != 0.) {
		s[l] = d_sign(s[l], x[l + l * x_dim1]);
	    }
	    i__2 = *n - l + 1;
	    d__1 = 1. / s[l];
	    dscal(&i__2, &d__1, &x[l + l * x_dim1], &c__1);
	    x[l + l * x_dim1] += 1.;
L10:
	    s[l] = -s[l];
L20:
	    if (*p < lp1) {
		goto L50;
	    }
	    i__2 = *p;
	    for (j = lp1; j <= i__2; ++j) {
                integer c__1 = 1;
		if (l > nct) {
		    goto L30;
		}
		if (s[l] == 0. && *skip == 0) {
		    goto L30;
		}

/*              apply the transformation. */

		i__3 = *n - l + 1;
		t = -ddot(&i__3, &x[l + l * x_dim1], &c__1, &x[l + j * 
			x_dim1], &c__1) / x[l + l * x_dim1];
		i__3 = *n - l + 1;
		daxpy(&i__3, &t, &x[l + l * x_dim1], &c__1, &x[l + j * 
			x_dim1], &c__1);
L30:

/*           place the l-th row of x into  e for the */
/*           subsequent calculation of the row transformation.
 */

		e[j] = x[l + j * x_dim1];
/* L40: */
	    }
L50:
	    if (! wantu || l > nct) {
		goto L70;
	    }

/*           place the transformation in u for subsequent back */
/*           multiplication. */

	    i__2 = *n;
	    for (i__ = l; i__ <= i__2; ++i__) {
		u[i__ + l * u_dim1] = x[i__ + l * x_dim1];
/* L60: */
	    }
L70:
	    if (l > nrt) {
		goto L150;
	    }

/*           compute the l-th row transformation and place the */
/*           l-th super-diagonal in e(l). */

	    i__2 = *p - l;
	    e[l] = dnrm2(&i__2, &e[lp1], &c__1);
	    if (e[l] == 0. && *skip == 0) {
		goto L80;
	    }
	    if (e[lp1] != 0.) {
		e[l] = d_sign(e[l], e[lp1]);
	    }
	    i__2 = *p - l;
	    d__1 = 1. / e[l];
	    dscal(&i__2, &d__1, &e[lp1], &c__1);
	    e[lp1] += 1.;
L80:
	    e[l] = -e[l];
	    if (lp1 > *n || (e[l] == 0. && *skip == 0)) {
		goto L120;
	    }

/*              apply the transformation. */

	    i__2 = *n;
	    for (i__ = lp1; i__ <= i__2; ++i__) {
		work[i__] = 0.;
/* L90: */
	    }
	    i__2 = *p;
	    for (j = lp1; j <= i__2; ++j) {
		i__3 = *n - l;
		daxpy(&i__3, &e[j], &x[lp1 + j * x_dim1], &c__1, &work[lp1], 
			&c__1);
/* L100: */
	    }
	    i__2 = *p;
	    for (j = lp1; j <= i__2; ++j) {
		i__3 = *n - l;
		d__1 = -e[j] / e[lp1];
		daxpy(&i__3, &d__1, &work[lp1], &c__1, &x[lp1 + j * x_dim1], 
			&c__1);
/* L110: */
	    }
L120:
	    if (! wantv) {
		goto L140;
	    }

/*              place the transformation in v for subsequent */
/*              back multiplication. */

	    i__2 = *p;
	    for (i__ = lp1; i__ <= i__2; ++i__) {
		v[i__ + l * v_dim1] = e[i__];
/* L130: */
	    }
L140:
L150:
/* L160: */
	    ;
	}
L170:
	;
    }

/*     set up the final bidiagonal matrix or order m. */

/* Computing MIN */
    i__1 = *p, i__2 = *n + 1;
    m = min(i__1,i__2);
    nctp1 = nct + 1;
    nrtp1 = nrt + 1;
    if (*skip <= 0) {
	if (nct < *p) {
	    s[nctp1] = x[nctp1 + nctp1 * x_dim1];
	}
	if (*n < m) {
	    s[m] = 0.;
	}
	if (nrtp1 < m) {
	    e[nrtp1] = x[nrtp1 + m * x_dim1];
	}
	e[m] = 0.;

/*     if required, generate u. */

	if (! wantu) {
	    goto L300;
	}
	if (ncu < nctp1) {
	    goto L200;
	}
	i__1 = ncu;
	for (j = nctp1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		u[i__ + j * u_dim1] = 0.;
/* L180: */
	    }
	    u[j + j * u_dim1] = 1.;
/* L190: */
	}
L200:
	if (nct < 1) {
	    goto L290;
	}
	i__1 = nct;
	for (ll = 1; ll <= i__1; ++ll) {
	    l = nct - ll + 1;
	    if (s[l] == 0.) {
		goto L250;
	    }
	    lp1 = l + 1;
	    if (ncu < lp1) {
		goto L220;
	    }
	    i__2 = ncu;
	    for (j = lp1; j <= i__2; ++j) {
                integer c__1 = 1;
		i__3 = *n - l + 1;
		t = -ddot(&i__3, &u[l + l * u_dim1], &c__1, &u[l + j * 
			u_dim1], &c__1) / u[l + l * u_dim1];
		i__3 = *n - l + 1;
		daxpy(&i__3, &t, &u[l + l * u_dim1], &c__1, &u[l + j * 
			u_dim1], &c__1);
/* L210: */
	    }
L220:
	    i__2 = *n - l + 1;
            {
                doublereal c_b367 = -1.;
                integer c__1 = 1;
                dscal(&i__2, &c_b367, &u[l + l * u_dim1], &c__1);
            }
	    u[l + l * u_dim1] += 1.;
	    lm1 = l - 1;
	    if (lm1 < 1) {
		goto L240;
	    }
	    i__2 = lm1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		u[i__ + l * u_dim1] = 0.;
/* L230: */
	    }
L240:
	    goto L270;
L250:
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		u[i__ + l * u_dim1] = 0.;
/* L260: */
	    }
	    u[l + l * u_dim1] = 1.;
L270:
/* L280: */
	    ;
	}
L290:
L300:

/*     if it is required, generate v. */

	if (! wantv) {
	    goto L350;
	}
	i__1 = *p;
	for (ll = 1; ll <= i__1; ++ll) {
	    l = *p - ll + 1;
	    lp1 = l + 1;
	    if (l > nrt) {
		goto L320;
	    }
	    if (e[l] == 0.) {
		goto L320;
	    }
	    i__2 = *p;
	    for (j = lp1; j <= i__2; ++j) {
                integer c__1 = 1;
		i__3 = *p - l;
		t = -ddot(&i__3, &v[lp1 + l * v_dim1], &c__1, &v[lp1 + j * 
			v_dim1], &c__1) / v[lp1 + l * v_dim1];
		i__3 = *p - l;
		daxpy(&i__3, &t, &v[lp1 + l * v_dim1], &c__1, &v[lp1 + j * 
			v_dim1], &c__1);
/* L310: */
	    }
L320:
	    i__2 = *p;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		v[i__ + l * v_dim1] = 0.;
/* L330: */
	    }
	    v[l + l * v_dim1] = 1.;
/* L340: */
	}
L350:
	;
    }


    if (*skip == 1) {
/*       set up s,e,u,v assuming x bidiagonal on input */
	minnp = min(*n,*p);
	i__1 = minnp;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    s[i__] = x[i__ + i__ * x_dim1];
	    if (i__ < *p) {
		e[i__] = x[i__ + (i__ + 1) * x_dim1];
	    }
/* L351: */
	}
	if (*n < *p) {
	    s[*n + 1] = (double)0.;
	}
	e[m] = 0.;
	if (wantu) {
	    i__1 = ncu;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    u[i__ + j * u_dim1] = (double)0.;
/* L353: */
		}
		u[j + j * u_dim1] = 1.;
/* L352: */
	    }
	}
	if (wantv) {
	    i__1 = *p;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *p;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    v[i__ + j * v_dim1] = (double)0.;
/* L355: */
		}
		v[j + j * v_dim1] = 1.;
/* L354: */
	    }
	}
    }

/*     main iteration loop for the singular values. */

/*     convert maxit to bound on total number of passes through */
/*     inner loops of qr iteration (half number of rotations) */
    maxit = *maxitr * m * m / 2;
    iter = 0;
    oldll = -1;
    oldm = -1;
    oldacc = -1;
    if (*tol > (double)0.) {
/*       relative accuracy desired */
	thresh = 0.;
    } else {
/*       absolute accuracy desired */
	smax = (d__1 = s[m], fabs(d__1));
	i__1 = m - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    d__3 = smax, d__4 = (d__1 = s[i__], fabs(d__1)), d__3 = max(d__3,
		    d__4), d__4 = (d__2 = e[i__], fabs(d__2));
	    smax = max(d__3,d__4);
/* L1111: */
	}
	thresh = fabs(*tol) * smax;
    }
    mm = m;

/*     begin loop */
L999:
    if (*idbg > 0) {
      integer c__1 = 1;
      printf("top of loop\n");
      printf("oldll,oldm,oldacc,m,iter,maxit,ifull,thresh=%ld,%ld,%ld,%ld,%ld,%ld,%ld,%f\n",
	     oldll,oldm,oldacc,oldacc,iter,maxit,(*ifull),thresh);
	prse(&c__1, &mm, n, p, &s[1], &e[1]);
    }

/*     check for being done */
    if (m == 1) {
	goto L998;
    }

/*     check number of iterations */
    if (iter >= maxit) {
	goto L997;
    }

/*     compute minimum s(i) and max of all s(i),e(i) */
    if (*tol <= (double)0. && (d__1 = s[m], fabs(d__1)) <= thresh) {
	s[m] = (double)0.;
    }
    smax = (d__1 = s[m], fabs(d__1));
    smin = smax;

/*     reset convergence threshold if starting new part of matrix */
    if (m <= oldll && *tol > (double)0.) {
	thresh = 0.;
    }
    if (*idbg > 0) {
            printf("thresh=%f\n",thresh);
    }
    i__1 = m;
    for (lll = 1; lll <= i__1; ++lll) {
	ll = m - lll;
	if (ll == 0) {
	    goto L1003;
	}
	if (*tol <= (double)0. && (d__1 = s[ll], fabs(d__1)) <= thresh) {
	    s[ll] = (double)0.;
	}
	if ((d__1 = e[ll], fabs(d__1)) <= thresh) {
	    goto L1002;
	}
	abss = (d__1 = s[ll], fabs(d__1));
	abse = (d__1 = e[ll], fabs(d__1));
	smin = min(smin,abss);
/* Computing MAX */
	d__1 = max(smax,abss);
	smax = max(d__1,abse);
/* L1001: */
    }
L1002:
    e[ll] = 0.;

/*     matrix splits since e(ll)=0 */
    if (ll == m - 1) {
/*       convergence of bottom singular values */
	--m;
	if (*idbg > 0) {
	        printf("convergence\n");
	}
	goto L999;
    }
L1003:
    ++ll;
/*     e(ll) ... e(m-1) are nonzero */
    if (*idbg > 0) {
      printf("work on block ll,m=%ld,%ld\n",ll,m);	
      printf("smin=%f\n",smin);

      printf("smax=%f\n",smax);
    }

/*     2 by 2 block - handle specially to guarantee convergence */
    if (ll == m - 1) {
/*       after one step */
	++(*kount1);
/*       shift = sigmin(s(m-1),e(m-1),s(m)) */
/*       rotate, setting e(m-1)=0 and s(m)=+-shift */
/*       if (s(ll).eq.0.0d0) then */
/*         f = 0.0d0 */
/*       else */
/*         f = (abs(s(ll)) - shift)*(dsign(1.0d0,s(ll))+shift/s(ll)) 
*/
/*       endif */
/*       g=e(ll) */
/*       call ndrotg(f,g,cs,sn) */
/*       sg1=dsign(1.0d0,s(m)) */
/*       sg2=dsign(1.0d0,cs) */
/*       f = cs*s(ll) + sn*e(ll) */
/*       g = sn*s(m) */
/*       if (idbg.gt.0) then */
/*         abss = cs*s(m) */
/*         abse = -sn*s(ll) + cs*e(ll) */
/*       endif */
/*       if (wantv) call drot(p,v(1,ll),1,v(1,m),1,cs,sn) */
/*       call ndrotg(f,g,cs,sn) */
/*       s(ll)=f */
/*       if (wantu.and.ll.lt.n) call drot(n,u(1,ll),1,u(1,m),1,cs,sn) 
*/
/*       e(ll) = 0.0d0 */
/*       s(m) = shift * dsign(1.0d0,cs) * sg1 * sg2 */
/*       if (idbg.gt.0) then */
/*         print *,'2 by 2 block' */
/*         print *,'shift=',shift */
/*         print *,'check shift=',-sn*abse+cs*abss */
/*         print *,'check zero=',cs*abse+sn*abss */
/*       endif */
	sig22(&s[m - 1], &e[m - 1], &s[m], &sigmn, &sigmx, &sinr, &cosr, 
	       &sinl, &cosl);
	s[m - 1] = sigmx;
	e[m - 1] = (double)0.;
	s[m] = sigmn;
	if (wantv) {
            integer c__1 = 1;
	    drot(p, &v[ll * v_dim1 + 1], &c__1, &v[m * v_dim1 + 1], &c__1, &
		    cosr, &sinr);
	}
/*       if wantu and ll.eq.n, then rotation trivial */
	if (wantu && ll < *n) {
            integer c__1 = 1;
	    drot(n, &u[ll * u_dim1 + 1], &c__1, &u[m * u_dim1 + 1], &c__1, &
		    cosl, &sinl);
	}
	goto L999;
    }

/*     choose shift direction if new submatrix */
/*     if (ll.ne.oldll .or. m.ne.oldm) then */
/*     choose shift direction if working on entirely new submatrix */
    if (ll > oldm || m < oldll) {
	if (((d__1 = s[ll], fabs(d__1)) >= (d__2 = s[m], fabs(d__2)) && *iidir == 0) || *iidir == 1) {
/*         chase bulge from top (big end) to bottom (small end) */
/*         if m=n+1, chase from top to bottom even if s(ll)=0 */
	    idir = 1;
	} else {
/*         chase bulge from bottom (big end) to top (small end) */
	    idir = 2;
	}
    }
    if (*idbg > 0) {
            printf("idir=%ld\n",idir);
    }

/*     compute lower bound on smallest singular value */
/*     if old lower bound still good, do not recompute it */
/*     if (ll.ne.oldll .or. m.ne.oldm .or. oldacc.ne.1) then */
/*       compute lower bound */
/*       sminl = smin */
/*       oldacc = 1 */
/*       if (sminl.gt.0.0d0) then */
/*         if (idir.eq.1) then */
/*           do 1004 lll=ll,m-1 */
/*             abse = abs(e(lll)) */
/*             abss = abs(s(lll)) */
/*             if (abss.lt.abse) then */
/*               sminl = sminl * (abss/abse) */
/*               oldacc = -1 */
/*             endif */
/* L1004: */
/*         else */
/*           do 1005 lll=ll,m-1 */
/*             abse = abs(e(lll)) */
/*             abss = abs(s(lll+1)) */
/*             if (abss.lt.abse) then */
/*               sminl = sminl * (abss/abse) */
/*               oldacc = -1 */
/*             endif */
/* L1005: */
/*         endif */
/*       endif */
/*       oldll = ll */
/*       oldm = m */
/*       sminl is lower bound on smallest singular value */
/*       within a factor of sqrt(m*(m+1)/2) */
/*       if oldacc = 1 as well, sminl is also upper bound */
/*       note that smin is always an upper bound */

/*       compute convergence threshold */
/*       thresh = tol*sminl */
/*     endif */
/*     if (idbg.gt.0) then */
/*       print *,'oldll,oldm,oldacc=',oldll,oldm,oldacc */
/*       print *,'sminl=',sminl */
/*       print *,'thresh=',thresh */
/*     endif */

/*     test again for convergence using new thresh */
/*     iconv = 0 */
/*     do 1014 lll=ll,m-1 */
/*       if (dabs(e(lll)).le.thresh) then */
/*         e(lll) = 0.0d0 */
/*         iconv = 1 */
/*       endif */
/* L1014: */
/*     if (iconv.eq.1) goto 999 */

/*     Kahan's convergence test */
    sminl = (double)0.;
    if (*tol > (double)0.) {
	if (idir == 1) {
/*         forward direction */
/*         apply test on bottom 2 by 2 only */
	    if ((d__1 = e[m - 1], fabs(d__1)) <= *tol * (d__2 = s[m], fabs(d__2)
		    )) {
/*           convergence of bottom element */
		e[m - 1] = 0.;
		goto L999;
	    }
/*         apply test in forward direction */
	    mu = (d__1 = s[ll], fabs(d__1));
	    sminl = mu;
	    i__1 = m - 1;
	    for (lll = ll; lll <= i__1; ++lll) {
		if ((d__1 = e[lll], fabs(d__1)) <= *tol * mu) {
/*             test for negligibility satisfied */
		    if (*idbg >= 1) {
		            printf("knew: e(lll),mu=%f,%f\n",e[lll],mu);
		    }
		    e[lll] = 0.;
		    goto L999;
		} else {
		    mu = (d__1 = s[lll + 1], fabs(d__1)) * (mu / (mu + (d__2 = 
			    e[lll], fabs(d__2))));
		}
		sminl = min(sminl,mu);
/* L3330: */
	    }
	} else {
/*         idir=2,  backwards direction */
/*         apply test on top 2 by 2 only */
	    if ((d__1 = e[ll], fabs(d__1)) <= *tol * (d__2 = s[ll], fabs(d__2)))
		     {
/*           convergence of top element */
		e[ll] = 0.;
		goto L999;
	    }
/*         apply test in backward direction */
	    lambda = (d__1 = s[m], fabs(d__1));
	    sminl = lambda;
	    i__1 = ll;
	    for (lll = m - 1; lll >= i__1; --lll) {
		if ((d__1 = e[lll], fabs(d__1)) <= *tol * lambda) {
/*             test for negligibility satisfied */
		    if (*idbg >= 1) {
		            printf("knew: e(lll),lambda=%f,%f\n",e[lll],lambda);
		    }
		    e[lll] = 0.;
		    goto L999;
		} else {
		    lambda = (d__1 = s[lll], fabs(d__1)) * (lambda / (lambda + 
			    (d__2 = e[lll], fabs(d__2))));
		}
		sminl = min(sminl,lambda);
/* L3331: */
	    }
	}
	thresh = *tol * sminl;
/*       thresh = 0 */
    }
    oldll = ll;
    oldm = m;

/*     test for zero shift */
    test = nct * *tol * (sminl / smax) + 1.;
    if ((test == 1. && *ifull != 1 && *tol > (double)0.) || *ifull == 2) {
/*       do a zero shift so that roundoff does not contaminate */
/*       smallest singular value */
	shift = 0.;
	if (*idbg > 0) {
	        printf("sminl test for shift is zero\n");
	}
    } else {

/*       compute shift from 2 by 2 block at end of matrix */
	if (idir == 1) {
	    smm1 = s[m - 1];
	    emm1 = e[m - 1];
	    sm = s[m];
	    sll = s[ll];
	} else {
	    smm1 = s[ll + 1];
	    emm1 = e[ll];
	    sm = s[ll];
	    sll = s[m];
	}
	if (*idbg > 0) {
	        printf("smm1,emm1,sm=%f,%f,%f\n",smm1,emm1,sm);
	}
	shift = sigmin(&smm1, &emm1, &sm);
	if (*idbg > 0) {
	  printf("sigma-min of 2 by 2 corner=%f\n",shift);
	}
	if (*tol > (double)0.) {
	    if (shift > nct * smin) {
		++(*limshf);
		shift = nct * smin;
		if (*idbg > 0) {
		        printf("shift limited\n");
		}
	    }
	    if (*idbg > 0) {
	            printf("shift=%f\n",shift);
	    }
	    temp = shift / sll;
	    if (*idbg > 0) {
	            printf("temp=%f\n",temp);
	    }
/* Computing 2nd power */
	    d__1 = temp;
	    test = 1. - d__1 * d__1;
/*         test to see if shift negligible */
	    if (*ifull != 1 && test == 1.) {
		shift = 0.;
	    }
	} else {
/*         if shift much larger than s(ll), first rotation could b
e 0, */
/*         leading to infinite loop; avoid by doing 0 shift in thi
s case */
	    if (shift > (d__1 = s[ll], fabs(d__1))) {
		test = s[ll] / shift;
		if (test + 1. == 1.) {
		    ++(*limshf);
		    if (*ifull != 1) {
			shift = (double)0.;
		    }
		    if (*idbg > 0 && *ifull != 1) {
		            printf("shift limited\n");
		    }
		}
	    }
	    test = smax + smin;
	    if (test == smax && *ifull != 1) {
		shift = (double)0.;
	    }
	}
	if (*idbg > 0) {
	        printf("test,shift=%f,%f\n",test,shift);
	}
    }

/*     increment iteration counter */
    iter = iter + m - ll;
    ++(*kount);
    if (*idbg > 1) {
      printf("s,e before qr\n");
	prse(&ll, &m, n, p, &s[1], &e[1]);
    }

/*     if shift = 0, do simplified qr iteration */
    if (shift == 0.) {
	*kount2 = *kount2 + m - ll;

/*       if idir=1, chase bulge from top to bottom */
	if (idir == 1) {
	    if (*idbg > 2) {
	            printf("qr with zero shift, top to bottom\n");
	    }
	    oldcs = 1.;
	    f = s[ll];
	    g = e[ll];
	    i__1 = m - 1;
	    for (k = ll; k <= i__1; ++k) {
/*           if (idbg.gt.2) print *,'qr inner loop, k=',k */
/*           if (idbg.gt.3) print *,'f,g=',f,g */
		ndrotg(&f, &g, &cs, &sn);
/* Computing MAX */
		d__1 = *maxsin, d__2 = fabs(sn);
		*maxsin = max(d__1,d__2);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
		if (wantv) {
                    integer c__1 = 1;
		    drot(p, &v[k * v_dim1 + 1], &c__1, &v[(k + 1) * v_dim1 + 
			    1], &c__1, &cs, &sn);
		}
		if (k != ll) {
		    e[k - 1] = oldsn * f;
		}
/*           if (k.ne.ll .and. idbg.gt.3) print *,'e(k-1)=',e(
k-1) */
		f = oldcs * f;
/*           if (idbg.gt.3) print *,'f=',f */
		temp = s[k + 1];
/*           if (idbg.gt.3) print *,'temp=',temp */
		g = temp * sn;
/*           if (idbg.gt.3) print *,'g=',g */
		gg = temp * cs;
/*           if (idbg.gt.3) print *,'gg=',gg */
		ndrotg(&f, &g, &cs, &sn);
/* Computing MAX */
		d__1 = *maxsin, d__2 = fabs(sn);
		*maxsin = max(d__1,d__2);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
/*           if wantu and k.eq.n, then s(k+1)=0 so g=0 so cs=1
 and sn=0 */
		if (wantu && k < *n) {
                    integer c__1 = 1;
		    drot(n, &u[k * u_dim1 + 1], &c__1, &u[(k + 1) * u_dim1 + 
			    1], &c__1, &cs, &sn);
		}
		s[k] = f;
/*           if (idbg.gt.3) print *,'s(k)=',s(k) */
		f = gg;
/*           if (idbg.gt.3) print *,'f=',f */
		g = e[k + 1];
/*           if (idbg.gt.3) print *,'g=',g */
		oldcs = cs;
/*           if (idbg.gt.3) print *,'oldcs=',oldcs */
		oldsn = sn;
/*           if (idbg.gt.3) print *,'oldsn=',oldsn */
/*           if (idbg.gt.2) call prse(ll,m,n,p,s,e) */
/* L1006: */
	    }
	    e[m - 1] = gg * sn;
/*         if (idbg.gt.3) print *,'e(m-1)=',e(m-1) */
	    s[m] = gg * cs;
/*         if (idbg.gt.3) print *,'s(m)=',s(m) */

/*         test convergence */
	    if (*idbg > 0) {
	            printf("convergence decision for zero shift top to bottom\n");
		          printf("e(m-1), threshold=%f,%f\n",e[m - 1],thresh);
		if ((d__1 = e[m - 1], fabs(d__1)) <= thresh) {
		        printf("***converged***\n");
		}
	    }
	    if ((d__1 = e[m - 1], fabs(d__1)) <= thresh) {
		e[m - 1] = 0.;
	    }
	} else {
/*       (idir=2, so chase bulge from bottom to top) */
	    if (*idbg > 2) {
      printf("qr with zero shift, bottom to top\n");
	    }
	    oldcs = 1.;
	    f = s[m];
	    g = e[m - 1];
	    i__1 = ll + 1;
	    for (k = m; k >= i__1; --k) {
/*           if (idbg.gt.2) print *,'qr inner loop, k=',k */
/*           if (idbg.gt.3) print *,'f,g=',f,g */
		ndrotg(&f, &g, &cs, &sn);
/* Computing MAX */
		d__1 = *maxsin, d__2 = fabs(sn);
		*maxsin = max(d__1,d__2);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
/*           if m=n+1, always chase from top to bottom so no t
est for */
/*           k.lt.n necessary */
		if (wantu) {
                    integer c__1 = 1;
		    d__1 = -sn;
		    drot(n, &u[(k - 1) * u_dim1 + 1], &c__1, &u[k * u_dim1 + 
			    1], &c__1, &cs, &d__1);
		}
		if (k != m) {
		    e[k] = oldsn * f;
		}
/*           if (k.ne.m .and. idbg.gt.3) print *,'e(k)=',e(k) 
*/
		f = oldcs * f;
/*           if (idbg.gt.3) print *,'f=',f */
		temp = s[k - 1];
/*           if (idbg.gt.3) print *,'temp=',temp */
		g = sn * temp;
/*           if (idbg.gt.3) print *,'g=',g */
		gg = cs * temp;
/*           if (idbg.gt.3) print *,'gg=',gg */
		ndrotg(&f, &g, &cs, &sn);
/* Computing MAX */
		d__1 = *maxsin, d__2 = fabs(sn);
		*maxsin = max(d__1,d__2);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
		if (wantv) {
                    integer c__1 = 1;
		    d__1 = -sn;
		    drot(p, &v[(k - 1) * v_dim1 + 1], &c__1, &v[k * v_dim1 + 
			    1], &c__1, &cs, &d__1);
		}
		s[k] = f;
/*           if (idbg.gt.3) print *,'s(k)=',s(k) */
		f = gg;
/*           if (idbg.gt.3) print *,'f=',f */
		if (k != ll + 1) {
		    g = e[k - 2];
		}
/*           if (k.ne.ll+1 .and. idbg.gt.3) print *,'g=',g */
		oldcs = cs;
/*           if (idbg.gt.3) print *,'oldcs=',oldcs */
		oldsn = sn;
/*           if (idbg.gt.3) print *,'oldsn=',oldsn */
/*           if (idbg.gt.2) call prse(ll,m,n,p,s,e) */
/* L1007: */
	    }
	    e[ll] = gg * sn;
/*         if (idbg.gt.3) print *,'e(ll)=',e(ll) */
	    s[ll] = gg * cs;
/*         if (idbg.gt.3) print *,'s(ll)=',s(ll) */

/*         test convergence */
	    if (*idbg > 0) {
	            printf("convergence decision for zero shift bottom to top\n");
		          printf("e(ll), threshold=%f,%f\n",e[ll],thresh);

		if ((d__1 = e[ll], fabs(d__1)) <= thresh) {
		        printf("***converged***\n");
		}
	    }
	    if ((d__1 = e[ll], fabs(d__1)) <= thresh) {
		e[ll] = 0.;
	    }
	}
    } else {
/*     (shift.ne.0, so do standard qr iteration) */
	*kount1 = *kount1 + m - ll;

/*       if idir=1, chase bulge from top to bottom */
	if (idir == 1) {
            doublereal c_b170 = 1.;
            
	    if (*idbg > 2) {
	            printf("qr with nonzero shift, top to bottom\n");
	    }
	    f = ((d__1 = s[ll], fabs(d__1)) - shift) * (d_sign(c_b170, s[ll])
		     + shift / s[ll]);
	    g = e[ll];
	    i__1 = m - 1;
	    for (k = ll; k <= i__1; ++k) {
/*           if (idbg.gt.2) print *,'qr inner loop, k=',k */
/*           if (idbg.gt.3) print *,'f,g=',f,g */
		ndrotg(&f, &g, &cs, &sn);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
		if (k != ll) {
		    e[k - 1] = f;
		}
/*           if (k.ne.ll .and. idbg.gt.3) print *,'e(k-1)=',e(
k-1) */
		f = cs * s[k] + sn * e[k];
/*           if (idbg.gt.3) print *,'f=',f */
		e[k] = cs * e[k] - sn * s[k];
/*           if (idbg.gt.3) print *,'e(k)=',e(k) */
		g = sn * s[k + 1];
/*           if (idbg.gt.3) print *,'g=',g */
		s[k + 1] = cs * s[k + 1];
/*           if (idbg.gt.3) print *,'s(k+1)=',s(k+1) */
		if (wantv) {
                    integer c__1 = 1;
		    drot(p, &v[k * v_dim1 + 1], &c__1, &v[(k + 1) * v_dim1 + 
			    1], &c__1, &cs, &sn);
		}
		ndrotg(&f, &g, &cs, &sn);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
		s[k] = f;
/*           if (idbg.gt.3) print *,'s(k)=',s(k) */
		f = cs * e[k] + sn * s[k + 1];
/*           if (idbg.gt.3) print *,'f=',f */
		s[k + 1] = -sn * e[k] + cs * s[k + 1];
/*           if (idbg.gt.3) print *,'s(k+1)=',s(k+1) */
		g = sn * e[k + 1];
/*           if (idbg.gt.3) print *,'g=',g */
		e[k + 1] = cs * e[k + 1];
/*           if (idbg.gt.3) print *,'e(k+1)=',e(k+1) */
/*           test for k.lt.n seems unnecessary since k=n cause
s zero */
/*           shift, so test removed from original code */
		if (wantu) {
                    integer c__1 = 1;
		    drot(n, &u[k * u_dim1 + 1], &c__1, &u[(k + 1) * u_dim1 + 
			    1], &c__1, &cs, &sn);
		}
/*           if (idbg.gt.2) call prse(ll,m,n,p,s,e) */
/* L1008: */
	    }
	    e[m - 1] = f;
/*         if (idbg.gt.3) print *,'e(m-1)=',e(m-1) */

/*         check convergence */
	    if (*idbg > 0) {
	            printf("convergence decision for shift top to bottom\n");
		          printf("e(m-1), threshold=%f,%f\n",e[m - 1],thresh);
		if ((d__1 = e[m - 1], fabs(d__1)) <= thresh) {
		  printf("***converged***\n");
		}
	    }
	    if ((d__1 = e[m - 1], fabs(d__1)) <= thresh) {
		e[m - 1] = 0.;
	    }
	} else {
/*       (idir=2, so chase bulge from bottom to top) */
            doublereal c_b170 = 1.;
	    if (*idbg > 2) {
	            printf("qr with nonzero shift, bottom to top\n");
	    }
	    f = ((d__1 = s[m], fabs(d__1)) - shift) * (d_sign(c_b170, s[m]) 
		    + shift / s[m]);
	    g = e[m - 1];
	    i__1 = ll + 1;
	    for (k = m; k >= i__1; --k) {
/*           if (idbg.gt.2) print *,'qr inner loop, k=',k */
/*           if (idbg.gt.3) print *,'f,g=',f,g */
		ndrotg(&f, &g, &cs, &sn);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
		if (k != m) {
		    e[k] = f;
		}
/*           if (k.ne.m .and. idbg.gt.3) print *,'e(k)=',e(k) 
*/
		f = cs * s[k] + sn * e[k - 1];
/*           if (idbg.gt.3) print *,'f=',f */
		e[k - 1] = -sn * s[k] + cs * e[k - 1];
/*           if (idbg.gt.3) print *,'e(k-1)=',e(k-1) */
		g = sn * s[k - 1];
/*           if (idbg.gt.3) print *,'g=',g */
		s[k - 1] = cs * s[k - 1];
/*           if (idbg.gt.3) print *,'s(k-1)=',s(k-1) */
		if (wantu && k <= *n) {
                    integer c__1 = 1;
		    d__1 = -sn;
		    drot(n, &u[(k - 1) * u_dim1 + 1], &c__1, &u[k * u_dim1 + 
			    1], &c__1, &cs, &d__1);
		}
		ndrotg(&f, &g, &cs, &sn);
/*           if (idbg.gt.3) print *,'f,cs,sn=',f,cs,sn */
		if (wantv) {
                    integer c__1 = 1;
		    d__1 = -sn;
		    drot(p, &v[(k - 1) * v_dim1 + 1], &c__1, &v[k * v_dim1 + 
			    1], &c__1, &cs, &d__1);
		}
		s[k] = f;
/*           if (idbg.gt.3) print *,'s(k)=',s(k) */
		f = sn * s[k - 1] + cs * e[k - 1];
/*           if (idbg.gt.3) print *,'f=',f */
		s[k - 1] = cs * s[k - 1] - sn * e[k - 1];
/*           if (idbg.gt.3) print *,'s(k-1)=',s(k-1) */
		if (k != ll + 1) {
		    g = sn * e[k - 2];
/*             if (idbg.gt.3) print *,'g=',g */
		    e[k - 2] = cs * e[k - 2];
/*             if (idbg.gt.3) print *,'e(k-2)=',e(k-2) */
		}
/*           if (idbg.gt.2) call prse(ll,m,n,p,s,e) */
/* L1009: */
	    }
	    e[ll] = f;
/*         if (idbg.gt.3) print *,'e(ll)=',e(ll) */

/*         test convergence */
	    if (*idbg > 0) {
	            printf("convergence decision for shift bottom to top\n");
		          printf("e(ll), threshold=%f,%f\n",e[ll],thresh);

		if ((d__1 = e[ll], fabs(d__1)) <= thresh) {
		        printf("***converged***\n");
		}
	    }
	    if ((d__1 = e[ll], fabs(d__1)) <= thresh) {
		e[ll] = 0.;
	    }
	}
    }

    if (*idbg > 1) {
            printf("s,e after qr\n");
	prse(&ll, &m, n, p, &s[1], &e[1]);
    }

/*     qr iteration finished, go back to check convergence */
    goto L999;

L998:

/*     make singular values positive */
    m = min(*n,*p);
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (s[i__] < 0.) {
	    s[i__] = -s[i__];
	    if (wantv) {
                integer c__1 = 1;
                doublereal c_b367 = -1.;
		dscal(p, &c_b367, &v[i__ * v_dim1 + 1], &c__1);
	    }
	}
/* L1010: */
    }

/*     sort singular values from largest at top to smallest */
/*     at bottom (use insertion sort) */
    i__1 = m - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*       scan for smallest s(j) */
	iisub = 1;
	smin = s[1];
	i__2 = m + 1 - i__;
	for (j = 2; j <= i__2; ++j) {
	    if (s[j] < smin) {
		iisub = j;
		smin = s[j];
	    }
/* L1012: */
	}
	if (iisub != m + 1 - i__) {
/*         swap singular values, vectors */
	    temp = s[m + 1 - i__];
	    s[m + 1 - i__] = s[iisub];
	    s[iisub] = temp;
	    if (wantv) {
                integer c__1 = 1;
		dswap(p, &v[(m + 1 - i__) * v_dim1 + 1], &c__1, &v[iisub * 
			v_dim1 + 1], &c__1);
	    }
	    if (wantu) {
                integer c__1 = 1;
                dswap(n, &u[(m + 1 - i__) * u_dim1 + 1], &c__1, &u[iisub * 
			u_dim1 + 1], &c__1);
	    }
	}
/* L1011: */
    }

/*     finished, return */
    return 0;

L997:
/*     maximum number of iterations exceeded */
    for (i__ = m - 1; i__ >= 1; --i__) {
	*info = i__ + 1;
	if (e[i__] != 0.) {
	    goto L996;
	}
/* L1013: */
    }
L996:
    return 0;
} /* ndsvd_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
prse(integer *ll, integer *m, integer *nrow, integer *ncol, doublereal *s, doublereal *e)
{
    /* Format strings */

    /* System generated locals */
    integer i__1;

    /* Builtin functions */

    /* Local variables */
    static integer i__;

    /* Fortran I/O blocks */


/*     debug routine to print s,e */
    /* Parameter adjustments */
    --e;
    --s;

    /* Function Body */
    printf("                      s(.)                       e(.) for ll,m=%ld,%ld\n",(*ll),(*m));
    i__1 = *m - 1;
    for (i__ = *ll; i__ <= i__1; ++i__) {
      printf("%26.17f %26.17f\n",s[i__],e[i__]);

/* L1: */
    }
    if (*m >= *ncol) {
      printf("%26.17f\n",s[*m]);
    }
    if (*m < *ncol) {
      printf("%26.17f %26.17f\n",s[*m],e[*m]);
    }
    return 0;
} /* prse_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
sig22(doublereal *a, doublereal *b, doublereal *c__, doublereal *sigmin, doublereal *sigmax, doublereal *snr, doublereal *csr, doublereal *snl, doublereal *csl)
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Local variables */
    static doublereal absa, absb, absc, acmn, acmx, sgna, sgnb, sgnc, cosl, 
	    sinl, cosr, temp, sinr, temp1, temp2, temp3, sgnmn, sgnmx, ac, ca;
    static integer ia;
    static doublereal absbac;
    static integer ib;
    static doublereal as, at, au;

    static doublereal bac;

/*     compute the singular value decomposition of the 2 by 2 */
/*     upper triangular matrix [[a,b];[0,c]] */
/*     inputs - */
/*       a,b,c - real*8 - matrix entries */
/*     outputs - */
/*       sigmin - real*8 - +-smaller singular value */
/*       sigmax - real*8 - +-larger singular value */
/*       snr, csr - real*8 - sin and cos of right rotation (see below) */
/*       snl, csl - real*8 - sin and cos of left rotation (see below) */

/*       [  csl  snl ]  * [ a b ] * [ csr  -snr ] = [ sigmax    0   ] */
/*       [ -snl  csl ]    [ 0 c ]   [ snr   csr ]   [    0   sigmin ] */

/*     barring over/underflow all output quantities are correct to */
/*     within a few units in their last places */

/*     let UF denote the underflow and OF the overflow threshold */
/*     let eps denote the machine precision */

/*     overflow is impossible unless the true value of sigmax exceeds */
/*     OF (or does so within a few units in its last place) */

/*     underflow cannot adversely effect sigmin,sigmax unless they are */
/*     less than UF/eps for conventional underflow */
/*     underflow cannot adversely effect sigmin,sigmax if underflow is */
/*     gradual and results normalized */

/*     overflow is impossible in computing sinr, cosr, sinl, cosl */
/*     underflow can adversely effect results only if sigmax<UF/eps */
/*     or true angle of rotation < UF/eps */

/*     note: if c=0, then csl=1. and snl=0. (needed in general svd) */


/*     local variables: */

    absa = fabs(*a);
    absb = fabs(*b);
    absc = fabs(*c__);
    sgna = d_sign(1.0, *a);
    sgnb = d_sign(1.0, *b);
    sgnc = d_sign(1.0, *c__);
    acmn = min(absa,absc);
    acmx = max(absa,absc);
/*     bad underflow possible if acmx<UF/eps and standard underflow */
/*     underflow impossible if underflow gradual */
/*     either at=0 or eps/2 <= at <= 1 */
/*     if no or gradual underflow, at nearly correctly rounded */
    at = acmx - acmn;
    if (at != (double)0.) {
	at /= acmx;
    }

/*     compute sigmin, sigmax */

    if (absb < acmx) {
/*         fabs(bac) <= 1, underflow possible */
	if (absa < absc) {
	    bac = *b / *c__;
	} else {
	    bac = *b / *a;
	}
/*         1 <= as <= 2, underflow and roundoff harmless */
	as = acmn / acmx + 1.;
/*         0 <= au <= 1, underflow possible */
	au = bac * bac;
/*         1 <= temp1 <= sqrt(5), underflow, roundoff harmless */
	temp1 = sqrt(as * as + au);
/*         0 <= temp2 <= 1, possible harmful underflow from at */
	temp2 = sqrt(at * at + au);
/*         1 <= temp <= sqrt(5) + sqrt(2) */
	temp = temp1 + temp2;
	*sigmin = acmn / temp;
	*sigmin += *sigmin;
	*sigmax = acmx * (temp / (double)2.);
    } else {
	if (absb == (double)0.) {
/*             matrix identically zero */
	    *sigmin = (double)0.;
	    *sigmax = (double)0.;
	} else {
/*             0 <= au <= 1, underflow possible */
	    au = acmx / absb;
	    if (au == 0.) {
/*                 either au=0 exactly or underflows */
/*                 sigmin only underflows if true value should
 */
/*                 overflow on product acmn*acmx impossible */
		*sigmin = acmx * acmn / absb;
		*sigmax = absb;
	    } else {
/*                 1 <= as <= 2, underflow and roundoff harmle
ss */
		as = acmn / acmx + 1.;
/*                 2 <= temp <= sqrt(5)+sqrt(2), possible harm
ful */
/*                 underflow from at */
/* Computing 2nd power */
		d__1 = as * au;
/* Computing 2nd power */
		d__2 = at * au;
		temp = sqrt(d__1 * d__1 + 1.) + sqrt(d__2 * d__2 + 1.);
/*                 0 < sigmin <= 2 */
		*sigmin = au + au;
/*                 bad underflow possible only if true sigmin 
near UF */
		*sigmin *= acmn / temp;
		*sigmax = absb * (temp / 2.);
	    }
	}
    }

/*     compute rotations */

    if (absb <= acmx) {
	if (at == 0.) {
/*             assume as = 2, since otherwise underflow will have 
*/
/*             contaminated at so much that we get a bad answer */
/*             anyway; this can only happen if sigmax < UF/eps */
/*             with conventional underflow; this cannot happen */
/*             with gradual underflow */
	    if (absb > 0.) {
/*                 0 <= absbac <= 1 */
		absbac = absb / acmx;
/*                 1 <= temp3 <= 1+sqrt(5), underflow harmless
 */
		temp3 = absbac + sqrt(au + 4.);
/*                 1/3 <= temp3 <= (1+sqrt(10))/2 */
		temp3 /= absbac * temp3 + 2.;
		sinr = d_sign(1.0, *b);
		cosr = d_sign(temp3, *a);
		sinl = d_sign(temp3, *c__);
		cosl = sinr;
		sgnmn = sgna * sgnb * sgnc;
		sgnmx = sgnb;
	    } else {
/*                 matrix diagonal */
		sinr = 0.;
		cosr = 1.;
		sinl = 0.;
		cosl = 1.;
		sgnmn = sgnc;
		sgnmx = sgna;
	    }
	} else {
/*             at .ne. 0, so eps/2 <= at <= 1 */
/*             eps/2 <= temp3 <= 1 + sqrt(10) */
	    temp3 = au + temp1 * temp2;
	    if (absa < absc) {
/*                 fabs(ac) <= 1 */
		ac = *a / *c__;
/*                 eps <= sinr <= sqrt(13)+3 */
/* Computing 2nd power */
		d__1 = as * at + au;
		sinr = sqrt(d__1 * d__1 + ac * 4. * ac * au) + as * at + au;
/*                 fabs(cosr) <= 2; if underflow, true cosr<UF/
eps */
		cosr = ac * bac;
		cosr += cosr;
/*                 eps/(3+sqrt(10)) <= sinl <= 1 */
		sinl = (as * at + temp3) / (ac * ac + 1. + temp3);
/*                 bad underflow possible only if sigmax < UF/
eps */
		sinl = *c__ * sinl;
		cosl = *b;
		sgnmn = sgna * sgnc;
		sgnmx = (double)1.;
	    } else {
/*                 fabs(ca) <= 1 */
		ca = *c__ / *a;
		sinr = *b;
		cosr = (as * at + temp3) / (ca * ca + 1. + temp3);
		cosr = *a * cosr;
/*                 fabs(sinl) <= 2; if underflow, true sinl<UF/
eps */
		sinl = ca * bac;
		sinl += sinl;
/*                 eps <= cosl <= sqrt(13)+3 */
/* Computing 2nd power */
		d__1 = as * at + au;
		cosl = sqrt(d__1 * d__1 + ca * 4. * ca * au) + as * at + au;
		sgnmn = sgna * sgnc;
		sgnmx = (double)1.;
	    }
	}
    } else {
	if (absa == 0.) {
	    cosr = 0.;
	    sinr = 1.;
	    ia = 0;
	} else {
	    sinr = *b;
/*             sigmin <= fabs(a)/sqrt(2), so no bad cancellation in
 */
/*             absa-sigmin; overflow extremely unlikely, and in an
y */
/*             event only if sigmax overflows as well */
	    cosr = (absa - *sigmin) * (d_sign(1.0, *a) + *sigmin / *a);
	    ia = 1;
	}
	if (absc == 0.) {
	    sinl = 0.;
	    cosl = 1.;
	    ib = 0;
	} else {
	    cosl = *b;
/*             sigmin <= fabs(c)/sqrt(2), so no bad cancellation in
 */
/*             absc-sigmin; overflow extremely unlikely, and in an
y */
/*             event only if sigmax overflows as well */
	    sinl = (absc - *sigmin) * (d_sign(1.0, *c__) + *sigmin / *c__);
	    ib = 1;
	}
	if (ia == 0 && ib == 0) {
	    sgnmn = (double)1.;
	    sgnmx = sgnb;
	} else if (ia == 0 && ib == 1) {
	    sgnmn = (double)1.;
	    sgnmx = (double)1.;
	} else if (ia == 1 && ib == 0) {
	    sgnmn = sgna * sgnc;
	    sgnmx = (double)1.;
	} else {
	    sgnmn = sgna * sgnb * sgnc;
	    sgnmx = sgnb;
	}
    }
    *sigmin = sgnmn * *sigmin;
    *sigmax = sgnmx * *sigmax;
    sndrtg(&cosr, &sinr, csr, snr);
    sndrtg(&cosl, &sinl, csl, snl);
    return 0;
} /* sig22_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
doublereal 
sigmin(doublereal *a, doublereal *b, doublereal *c__)
{
    /* System generated locals */
    doublereal ret_val, d__1, d__2;

    /* Local variables */
    static doublereal acmn, acmx, aa, ab, ac, as, at, au;

/*     compute smallest singular value of 2 by 2 matrix ((a,b);(0,c)) */
/*     answer is accurate to a few ulps if final answer */
/*     exceeds (underflow_threshold/macheps) */
/*     overflow is impossible */
    aa = fabs(*a);
    ab = fabs(*b);
    ac = fabs(*c__);
    acmn = min(aa,ac);
    if (acmn == 0.) {
	ret_val = 0.;
    } else {
	acmx = max(aa,ac);
	ab = fabs(*b);
	if (ab < acmx) {
	    as = acmn / acmx + 1.;
	    at = (acmx - acmn) / acmx;
/* Computing 2nd power */
	    d__1 = ab / acmx;
	    au = d__1 * d__1;
	    ret_val = acmn / (sqrt(as * as + au) + sqrt(at * at + au));
	    ret_val += ret_val;
	} else {
	    au = acmx / ab;
	    if (au == 0.) {
/*           possible harmful underflow */
/*           if exponent range asymmetric, true sigmin may not
 */
/*           underflow */
		ret_val = acmn * acmx / ab;
	    } else {
		as = acmn / acmx + 1.;
		at = (acmx - acmn) / acmx;
/* Computing 2nd power */
		d__1 = as * au;
/* Computing 2nd power */
		d__2 = at * au;
		ret_val = acmn / (sqrt(d__1 * d__1 + 1.) + sqrt(d__2 * d__2 + 
			1.));
		ret_val = au * ret_val;
		ret_val += ret_val;
	    }
	}
    }
    return ret_val;
} /* sigmin_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
sndrtg(doublereal *f, doublereal *g, doublereal *cs, doublereal *sn)
{
    /* System generated locals */
    doublereal d__1;

    /* Local variables */
    static doublereal t, tt;

/*     version of ndrotg, in which sign(f)=sign(cs),sign(g)=sign(sn) */
/*     cs, sn returned so that -sn*f+cs*g = 0 */
/*     and cs*f + sn*g = sqrt(f**2+g**2) */
    if (*f == 0. && *g == 0.) {
	*cs = 1.;
	*sn = 0.;
    } else {
	if (fabs(*f) > fabs(*g)) {
	    t = *g / *f;
	    tt = sqrt(t * t + 1.);
	    d__1 = 1. / tt;
	    *cs = d_sign(d__1, *f);
	    d__1 = t * *cs;
	    *sn = d_sign(d__1, *g);
	} else {
	    t = *f / *g;
	    tt = sqrt(t * t + 1.);
	    d__1 = 1. / tt;
	    *sn = d_sign(d__1, *g);
	    d__1 = t * *sn;
	    *cs = d_sign(d__1, *f);
	}
    }
    return 0;
} /* sndrtg_ */

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*              LINPACK and LAPACK routines */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */

/*  hqr3_loc.f orthes.f ortran.f   for computing triangular matrix */

/* Subroutine */ int 
hqr3lc(doublereal *a, doublereal *v, integer *n, integer *nlow, integer *nup, doublereal *eps, doublereal *er, doublereal *ei, integer *type__, integer *na, integer *nv, integer *imfd)
{
    /* System generated locals */
    integer a_dim1, a_offset, v_dim1, v_offset, i__1;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    static logical fail;
    static integer i__, l;
    static doublereal p, q, r__, s, t, w, x, y, z__, e1, e2;

    static integer nl, it, mu, nu;
/* HQR3 REDUCES THE UPPER HESSENBERG MATRIX A TO QUASI- */
/* TRIANGULAR FORM BY UNITARY SIMILARITY TRANSFORMATIONS. */
/* THE EIGENVALUES OF A, WHICH ARE CONTAINED IN THE 1X1 */
/* AND 2X2 DIAGONAL BLOCKS OF THE REDUCED MATRIX, ARE */
/* ORDERED IN assending or DESCENDING ORDER OF ALONG THE */
/* DIAGONAL.  THE TRANSFORMATIONS ARE ACCUMULATED IN THE */
/* ARRAY V.  HQR3 REQUIRES THE SUBROUTINES EXCHNG, */
/* SUBROUTINE) */
/*    *A       AN ARRAY THAT INITIALLY CONTAINS THE N X N */
/*             UPPER HESSENBERG MATRIX TO BE REDUCED.  ON */
/*             RETURN A CONTAINS THE REDUCED, QUASI- */
/*             TRIANGULAR MATRIX. */
/*    *V       AN ARRAY THAT CONTAINS A MATRIX INTO WHICH */
/*             THE REDUCING TRANSFORMATIONS ARE TO BE */
/*             MULTIPLIED. */
/*     N       THE ORDER OF THE MATRICES A AND V. */
/*     NLOW    A(NLOW,NLOW-1) AND A(NUP,+1,NUP) ARE */
/*     NUP     ASSUMED TO BE ZERO, AND ONLY ROWS NLOW */
/*             THROUGH NUP AND COLUMNS NLOW THROUGH */
/*             NUP ARE TRANSFORMED, RESULTING IN THE */
/*             CALCULATION OF EIGENVALUES NLOW */
/*             THROUGH NUP. */
/*     EPS     A CONVERGENCE CRITERION. */
/*    *ER      AN ARRAY THAT ON RETURN CONTAINS THE REAL */
/*             PARTS OF THE EIGENVALUES. */
/*    *EI      AN ARRAY THAT ON RETURN CONTAINS THE */
/*             IMAGINARY PARTS OF THE EIGENVALUES. */
/*    *TYPE    AN INTEGER ARRAY WHOSE I-TH ENTRY IS */
/*               0   IF THE I-TH EIGENVALUE IS REAL, */
/*               1   IF THE I-TH EIGENVALUE IS COMPLEX */
/*                   WITH POSITIVE IMAGINARY PART. */
/*               2   IF THE I-TH EIGENVALUE IS COMPLEX */
/*                   WITH NEGATIVE IMAGINARY PART, */
/*              -1   IF THE I-TH EIGENVALUE WAS NOT */
/*                   CALCULATED SUCCESSFULLY. */
/*     NA      THE FIRST DIMENSION OF THE ARRAY A. */
/*     NV      THE FIRST DIMENSION OF THE ARRAY V. */

/*     imfd    ascending or descending order of real part of eigenvalues 
*/
/*              -1  ascending (i.e. negative eigenvalues first) */
/*              +1  descending (positive eigenvalues) */


/* THE CONVERGENCE CRITERION EPS IS USED TO DETERMINE */
/* WHEN A SUBDIAGONAL ELEMENT OF A IS NEGLIGIBLE. */
/* SPECIFICALLY A(I+1,I) IS REGARDED AS NEGLIGIBLE */
/* IF */
/*        FABS(A(I+1),I)) .LE. EPS*(FABS(A(I,I))+FABS(A(I+1,I+1))). */
/* THIS MEANS THAT THE FINAL MATRIX RETURNED BY THE */
/* PROGRAM WILL BE EXACTLY SIMILAR TO A + E WHERE E IS */
/* OF ORDER EPS*NORM(A), FOR ANY REASONABLY BALANCED NORM */
/* SUCH AS THE ROW-SUM NORM. */
/* INTERNAL VARIABLES */
/* INITIALIZE. */
    /* Parameter adjustments */
    --type__;
    --ei;
    --er;
    a_dim1 = *na;
    a_offset = a_dim1 + 1;
    a -= a_offset;
    v_dim1 = *nv;
    v_offset = v_dim1 + 1;
    v -= v_offset;

    /* Function Body */
    i__1 = *nup;
    for (i__ = *nlow; i__ <= i__1; ++i__) {
	type__[i__] = -1;
/* L10: */
    }
    t = 0.;
/* MAIN LOOP. FIND AND ORDER EIGENVALUES. */
    nu = *nup;
L20:
    if (nu < *nlow) {
	goto L240;
    }
    it = 0;
/* QR LOOP.  FIND NEGLIGIBLE ELEMENTS AND PERFORM */
/* QR STEPS. */
L30:
/* SEARCH BACK FOR NEGLIGIBLE ELEMENTS. */
    l = nu;
L40:
    if (l == *nlow) {
	goto L50;
    }
    if ((d__1 = a[l + (l - 1) * a_dim1], fabs(d__1)) <= *eps * ((d__2 = a[l - 
	    1 + (l - 1) * a_dim1], fabs(d__2)) + (d__3 = a[l + l * a_dim1], 
	    fabs(d__3)))) {
	goto L50;
    }
    --l;
    goto L40;
L50:
/* TEST TO SEE IF AN EIGENVALUE OR A 2X2 BLOCK */
/* HAS BEEN FOUND. */
    x = a[nu + nu * a_dim1];
    if (l == nu) {
	goto L160;
    }
    y = a[nu - 1 + (nu - 1) * a_dim1];
    w = a[nu + (nu - 1) * a_dim1] * a[nu - 1 + nu * a_dim1];
    if (l == nu - 1) {
	goto L100;
    }
/* TEST ITERATION COUNT. IF IT IS 30 QUIT.  IF */
/* IT IS 10 OR 20 SET UP AN AD-HOC SHIFT. */
    if (it == 30) {
	goto L240;
    }
    if (it != 10 && it != 20) {
	goto L70;
    }
/* AD-HOC SHIFT. */
    t += x;
    i__1 = nu;
    for (i__ = *nlow; i__ <= i__1; ++i__) {
	a[i__ + i__ * a_dim1] -= x;
/* L60: */
    }
    s = (d__1 = a[nu + (nu - 1) * a_dim1], fabs(d__1)) + (d__2 = a[nu - 1 + (
	    nu - 2) * a_dim1], fabs(d__2));
    x = s * .75;
    y = x;
/* Computing 2nd power */
    d__1 = s;
    w = d__1 * d__1 * -.4375;
L70:
    ++it;
/* LOOK FOR TWO CONSECUTIVE SMALL SUB-DIAGONAL */
/* ELEMENTS. */
    nl = nu - 2;
L80:
    z__ = a[nl + nl * a_dim1];
    r__ = x - z__;
    s = y - z__;
    p = (r__ * s - w) / a[nl + 1 + nl * a_dim1] + a[nl + (nl + 1) * a_dim1];
    q = a[nl + 1 + (nl + 1) * a_dim1] - z__ - r__ - s;
    r__ = a[nl + 2 + (nl + 1) * a_dim1];
    s = fabs(p) + fabs(q) + fabs(r__);
    p /= s;
    q /= s;
    r__ /= s;
    if (nl == l) {
	goto L90;
    }
    if ((d__1 = a[nl + (nl - 1) * a_dim1], fabs(d__1)) * (fabs(q) + fabs(r__)) <=
	     *eps * fabs(p) * ((d__2 = a[nl - 1 + (nl - 1) * a_dim1], fabs(d__2)
	    ) + fabs(z__) + (d__3 = a[nl + 1 + (nl + 1) * a_dim1], fabs(d__3))))
	     {
	goto L90;
    }
    --nl;
    goto L80;
L90:
/* PERFORM A QR STEP BETWEEN NL AND NU. */
    qrstep(&a[a_offset], &v[v_offset], &p, &q, &r__, &nl, &nu, n, na, nv);
    goto L30;
/* 2X2 BLOCK FOUND. */
L100:
    if (nu != *nlow + 1) {
	a[nu - 1 + (nu - 2) * a_dim1] = 0.;
    }
    a[nu + nu * a_dim1] += t;
    a[nu - 1 + (nu - 1) * a_dim1] += t;
    type__[nu] = 0;
    type__[nu - 1] = 0;
    mu = nu;
/* LOOP TO POSITION  2X2 BLOCK. */
L110:
    nl = mu - 1;
/* ATTEMPT  TO SPLIT THE BLOCK INTO TWO REAL */
/* EIGENVALUES. */
    split(&a[a_offset], &v[v_offset], n, &nl, &e1, &e2, na, nv);
/* IF THE SPLIT WAS SUCCESSFUL, GO AND ORDER THE */
/* REAL EIGENVALUES. */
    if (a[mu + (mu - 1) * a_dim1] == 0.) {
	goto L170;
    }
/* TEST TO SEE IF THE BLOCK IS PROPERLY POSITIONED, */
/* AND IF NOT EXCHANGE IT */
    if (mu == *nup) {
	goto L230;
    }
    if (mu == *nup - 1) {
	goto L130;
    }
    if (a[mu + 2 + (mu + 1) * a_dim1] == 0.) {
	goto L130;
    }
/* THE NEXT BLOCK IS 2X2. */
/*     IF (A(MU-1,MU-1)*A(MU,MU)-A(MU-1,MU)*A(MU,MU-1).GE.A(MU+1, */
/*    * MU+1)*A(MU+2,MU+2)-A(MU+1,MU+2)*A(MU+2,MU+1)) GO TO 230 */

    if (*imfd == 1) {
	if (a[mu - 1 + (mu - 1) * a_dim1] + a[mu + mu * a_dim1] >= a[mu + 1 + 
		(mu + 1) * a_dim1] + a[mu + 2 + (mu + 2) * a_dim1]) {
	    goto L230;
	}
    } else {
	if (a[mu - 1 + (mu - 1) * a_dim1] + a[mu + mu * a_dim1] <= a[mu + 1 + 
		(mu + 1) * a_dim1] + a[mu + 2 + (mu + 2) * a_dim1]) {
	    goto L230;
	}
    }
    
    {
        integer c__2 = 2;
        exchng(&a[a_offset], &v[v_offset], n, &nl, &c__2, &c__2, eps, &fail, na, 
               nv);
    }
    if (! fail) {
	goto L120;
    }
    type__[nl] = -1;
    type__[nl + 1] = -1;
    type__[nl + 2] = -1;
    type__[nl + 3] = -1;
    goto L240;
L120:
    mu += 2;
    goto L150;
L130:
/* THE NEXT BLOCK IS 1X1. */
/*     IF (A(MU-1,MU-1)*A(MU,MU)-A(MU-1,MU)*A(MU,MU-1).GE.A(MU+1, */
/*    * MU+1)**2) GO TO 230 */

    if (*imfd == 1) {
	if (a[mu - 1 + (mu - 1) * a_dim1] + a[mu + mu * a_dim1] >= a[mu + 1 + 
		(mu + 1) * a_dim1] * 2.) {
	    goto L230;
	}
    } else {
	if (a[mu - 1 + (mu - 1) * a_dim1] + a[mu + mu * a_dim1] <= a[mu + 1 + 
		(mu + 1) * a_dim1] * 2.) {
	    goto L230;
	}
    }
    {
        integer c__1 = 1, c__2 = 2;
        exchng(&a[a_offset], &v[v_offset], n, &nl, &c__2, &c__1, eps, &fail, na, 
               nv);
    }
    if (! fail) {
	goto L140;
    }
    type__[nl] = -1;
    type__[nl + 1] = -1;
    type__[nl + 2] = -1;
    goto L240;
L140:
    ++mu;
L150:
    goto L110;
/* SINGLE EIGENVALUE FOUND. */
L160:
    nl = 0;
    a[nu + nu * a_dim1] += t;
    if (nu != *nlow) {
	a[nu + (nu - 1) * a_dim1] = 0.;
    }
    type__[nu] = 0;
    mu = nu;
/* LOOP TO POSITION ONE OR TWO REAL EIGENVALUES. */
L170:
/* POSITION THE EIGENVALUE LOCATED AT A(NL,NL). */
L180:
    if (mu == *nup) {
	goto L220;
    }
    if (mu == *nup - 1) {
	goto L200;
    }
    if (a[mu + 2 + (mu + 1) * a_dim1] == 0.) {
	goto L200;
    }
/* THE NEXT BLOCK IS 2X2. */
/*      IF (A(MU,MU)**2.GE.A(MU+1,MU+1)*A(MU+2,MU+2)-A(MU+1,MU+2)* */
/*    * A(MU+2,MU+1)) GO TO 220 */

    if (*imfd == 1) {
	if (a[mu + mu * a_dim1] * 2. >= a[mu + 1 + (mu + 1) * a_dim1] + a[mu 
		+ 2 + (mu + 2) * a_dim1]) {
	    goto L220;
	}
    } else {
	if (a[mu + mu * a_dim1] * 2. <= a[mu + 1 + (mu + 1) * a_dim1] + a[mu 
		+ 2 + (mu + 2) * a_dim1]) {
	    goto L220;
	}
    }

    {
        integer c__1 = 1, c__2 = 2;
        exchng(&a[a_offset], &v[v_offset], n, &mu, &c__1, &c__2, eps, &fail, na, 
               nv);
    }
    if (! fail) {
	goto L190;
    }
    type__[mu] = -1;
    type__[mu + 1] = -1;
    type__[mu + 2] = -1;
    goto L240;
L190:
    mu += 2;
    goto L210;
L200:
/* THE NEXT BLOCK IS 1X1. */
/*      IF (FABS(A(MU,MU)).GE.FABS(A(MU+1,MU+1))) GO TO 220 */

    if (*imfd == 1) {
	if (a[mu + mu * a_dim1] >= a[mu + 1 + (mu + 1) * a_dim1]) {
	    goto L220;
	}
    } else {
	if (a[mu + mu * a_dim1] <= a[mu + 1 + (mu + 1) * a_dim1]) {
	    goto L220;
	}
    }

    {
        integer c__1 = 1;
        exchng(&a[a_offset], &v[v_offset], n, &mu, &c__1, &c__1, eps, &fail, na, 
               nv);
    }
    ++mu;
L210:
    goto L180;
L220:
    mu = nl;
    nl = 0;
    if (mu != 0) {
	goto L170;
    }
/* GO BACK AND GET THE NEXT EIGENVALUE. */
L230:
    nu = l - 1;
    goto L20;
/* ALL THE EIGENVALUES HAVE BEEN FOUND AND ORDERED. */
/* COMPUTE THEIR VALUES AND TYPE. */
L240:
    if (nu < *nlow) {
	goto L260;
    }
    i__1 = nu;
    for (i__ = *nlow; i__ <= i__1; ++i__) {
	a[i__ + i__ * a_dim1] += t;
/* L250: */
    }
L260:
    nu = *nup;
L270:
    if (type__[nu] != -1) {
	goto L280;
    }
    --nu;
    goto L310;
L280:
    if (nu == *nlow) {
	goto L290;
    }
    if (a[nu + (nu - 1) * a_dim1] == 0.) {
	goto L290;
    }
/* 2X2 BLOCK. */
    i__1 = nu - 1;
    split(&a[a_offset], &v[v_offset], n, &i__1, &e1, &e2, na, nv);
    if (a[nu + (nu - 1) * a_dim1] == 0.) {
	goto L290;
    }
    er[nu] = e1;
    ei[nu - 1] = e2;
    er[nu - 1] = er[nu];
    ei[nu] = -ei[nu - 1];
    type__[nu - 1] = 1;
    type__[nu] = 2;
    nu += -2;
    goto L300;
L290:
/* SINGLE ROOT. */
    er[nu] = a[nu + nu * a_dim1];
    ei[nu] = 0.;
    --nu;
L300:
L310:
    if (nu >= *nlow) {
	goto L270;
    }
    return 0;
} /* hqr3lc_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
split(doublereal *a, doublereal *v, integer *n, integer *l, doublereal *e1, doublereal *e2, integer *na, integer *nv)
{
    /* System generated locals */
    integer a_dim1, a_offset, v_dim1, v_offset, i__1;
    doublereal d__1, d__2;

    /* Local variables */
    static integer i__, j;
    static doublereal p, q, r__, t, u, w, x, y, z__;
    static integer l1;

/* GIVEN THE UPPER HESSENBERG MATRIX A WITH A 2X2 BLOCK */
/* STARTING AT A(L,L), SPLIT DETERMINES IF THE */
/* CORRESPONDING EIGENVALUES ARE REAL OR COMPLEX. IF THEY */
/* ARE REAL, A ROTATION IS DETERMINED THAT REDUCES THE */
/* BLOCK TO UPPER TRIANGULAR FORM WITH THE EIGENVALUE */
/* OF LARGEST ABSOLUTE VALUE APPEARING FIRST.  THE */
/* ROTATION IS ACCUMULATED IN V.  THE EIGENVALUES (REAL */
/* ALTERED BY THE SUBROUTINE) */
/*    *A       THE UPPER HESSENVERG MATRIX WHOSE 2X2 */
/*             BLOCK IS TO BE SPLIT. */
/*    *V       THE ARRAY IN WHICH THE SPLITTING TRANS- */
/*             FORMATION IS TO BE ACCUMULATED. */
/*     N       THE ORDER OF THE MATRIX A. */
/*     L       THE POSITION OF THE 2X2 BLOCK. */
/*    *E1      ON RETURN IF THE EIGENVALUES ARE COMPLEX */
/*    *E2      E1 CONTAINS THEIR COMMON REAL PART AND */
/*             E2 CONTAINS THE POSITIVE IMAGINARY PART. */
/*             IF THE EIGENVALUES ARE REAL, E1 CONTAINS */
/*             THE ONE LARGEST IN ABSOLUTE VALUE AND E2 */
/*             CONTAINS THE OTHER ONE. */
/*     NA      THE FIRST DIMENSION OF THE ARRAY A. */
/*     NV      THE FIRST DIMENSION OF THE ARRAY V. */
/* INTERNAL VARIABLES */
    /* Parameter adjustments */
    a_dim1 = *na;
    a_offset = a_dim1 + 1;
    a -= a_offset;
    v_dim1 = *nv;
    v_offset = v_dim1 + 1;
    v -= v_offset;

    /* Function Body */
    x = a[*l + 1 + (*l + 1) * a_dim1];
    y = a[*l + *l * a_dim1];
    w = a[*l + (*l + 1) * a_dim1] * a[*l + 1 + *l * a_dim1];
    p = (y - x) / 2.;
/* Computing 2nd power */
    d__1 = p;
    q = d__1 * d__1 + w;
    if (q >= 0.) {
	goto L10;
    }
/* COMPLEX EIGENVALUE. */
    *e1 = p + x;
    *e2 = sqrt(-q);
    return 0;
L10:
/* TWO REAL EIGENVALUES.  SET UP TRANSFORMATION. */
    z__ = sqrt(q);
    if (p < 0.) {
	goto L20;
    }
    z__ = p + z__;
    goto L30;
L20:
    z__ = p - z__;
L30:
    if (z__ == 0.) {
	goto L40;
    }
    r__ = -w / z__;
    goto L50;
L40:
    r__ = 0.;
L50:
    if ((d__1 = x + z__, fabs(d__1)) >= (d__2 = x + r__, fabs(d__2))) {
	z__ = r__;
    }
    y = y - x - z__;
    x = -z__;
    t = a[*l + (*l + 1) * a_dim1];
    u = a[*l + 1 + *l * a_dim1];
    if (fabs(y) + fabs(u) <= fabs(t) + fabs(x)) {
	goto L60;
    }
    q = u;
    p = y;
    goto L70;
L60:
    q = x;
    p = t;
L70:
/* Computing 2nd power */
    d__1 = p;
/* Computing 2nd power */
    d__2 = q;
    r__ = sqrt(d__1 * d__1 + d__2 * d__2);
    if (r__ > 0.) {
	goto L80;
    }
    *e1 = a[*l + *l * a_dim1];
    *e2 = a[*l + 1 + (*l + 1) * a_dim1];
    a[*l + 1 + *l * a_dim1] = 0.;
    return 0;
L80:
    p /= r__;
    q /= r__;
/* PREMULTIPLY. */
    i__1 = *n;
    for (j = *l; j <= i__1; ++j) {
	z__ = a[*l + j * a_dim1];
	a[*l + j * a_dim1] = p * z__ + q * a[*l + 1 + j * a_dim1];
	a[*l + 1 + j * a_dim1] = p * a[*l + 1 + j * a_dim1] - q * z__;
/* L90: */
    }
/* POSTMULTIPLY. */
    l1 = *l + 1;
    i__1 = l1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ = a[i__ + *l * a_dim1];
	a[i__ + *l * a_dim1] = p * z__ + q * a[i__ + (*l + 1) * a_dim1];
	a[i__ + (*l + 1) * a_dim1] = p * a[i__ + (*l + 1) * a_dim1] - q * z__;
/* L100: */
    }
/* ACCUMULATE THE TRANSFORMATION IN V. */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ = v[i__ + *l * v_dim1];
	v[i__ + *l * v_dim1] = p * z__ + q * v[i__ + (*l + 1) * v_dim1];
	v[i__ + (*l + 1) * v_dim1] = p * v[i__ + (*l + 1) * v_dim1] - q * z__;
/* L110: */
    }
    a[*l + 1 + *l * a_dim1] = 0.;
    *e1 = a[*l + *l * a_dim1];
    *e2 = a[*l + 1 + (*l + 1) * a_dim1];
    return 0;
} /* split_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
exchng(doublereal *a, doublereal *v, integer *n, integer *l, integer *b1, integer *b2, doublereal *eps, logical *fail, integer *na, integer *nv)
{
    /* System generated locals */
    integer a_dim1, a_offset, v_dim1, v_offset, i__1;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    static integer i__, j, m;
    static doublereal p, q, r__, s, w, x, y, z__;
    static integer l1, it;


/* GIVEN THE UPPER HESSENBERG MATRIX A WITH CONSECUTIVE */
/* B1XB1 AND B2XB2 DIAGONAL BLOCKS (B1,B2 .LE. 2) */
/* STARTING AT A(L,L), EXCHNG PRODUCES A UNITARY */
/* SIMILARITY TRANSFORMATION THAT EXCHANGES THE BLOCKS */
/* ALONG WITH THEIR EIGENVALUES.  THE TRANSFORMATION */
/* IS ACCUMULATED IN V.  EXCHNG REQUIRES THE SUBROUTINE */
/*    *A       THE MATRIX WHOSE BLOCKS ARE TO BE */
/*             INTERCHANGED. */
/*    *V       THE ARRAY INTO WHICH THE TRANSFORMATIONS */
/*             ARE TO BE ACCUMULATED. */
/*     N       THE ORDER OF THE MATRIX A. */
/*     L       THE POSITION OF THE BLOCKS. */
/*     B1      AN INTEGER CONTAINING THE SIZE OF THE */
/*             FIRST BLOCK. */
/*     B2      AN INTEGER CONTAINING THE SIZE OF THE */
/*             SECOND BLOCK. */
/*     EPS     A CONVERGENCE CRITERION (CF. HQR3). */
/*    *FAIL    A LOGICAL VARIABLE WHICH IS FALSE ON A */
/*             NORMAL RETURN.  IF THIRTY ITERATIONS WERE */
/*             PERFORMED WITHOUT CONVERGENCE, FAIL IS SET */
/*             TO TRUE AND THE ELEMENT */
/*             A(L+B2,L+B2-1) CANNOT BE ASSUMED ZERO. */
/*     NA      THE FIRST DIMENSION OF THE ARRAY A. */
/*     NV      THE FIRST DIMENSION OF THE ARRAY V. */
/* INTERNAL VARIABLES. */
    /* Parameter adjustments */
    a_dim1 = *na;
    a_offset = a_dim1 + 1;
    a -= a_offset;
    v_dim1 = *nv;
    v_offset = v_dim1 + 1;
    v -= v_offset;

    /* Function Body */
    *fail = FALSE_;
    if (*b1 == 2) {
	goto L70;
    }
    if (*b2 == 2) {
	goto L40;
    }
/* INTERCHANGE 1X1 AND 1X1 BLOCKS. */
    l1 = *l + 1;
    q = a[*l + 1 + (*l + 1) * a_dim1] - a[*l + *l * a_dim1];
    p = a[*l + (*l + 1) * a_dim1];
/* Computing MAX */
    d__1 = fabs(p), d__2 = fabs(q);
    r__ = max(d__1,d__2);
    if (r__ == 0.) {
	return 0;
    }
    p /= r__;
    q /= r__;
/* Computing 2nd power */
    d__1 = p;
/* Computing 2nd power */
    d__2 = q;
    r__ = sqrt(d__1 * d__1 + d__2 * d__2);
    p /= r__;
    q /= r__;
    i__1 = *n;
    for (j = *l; j <= i__1; ++j) {
	s = p * a[*l + j * a_dim1] + q * a[*l + 1 + j * a_dim1];
	a[*l + 1 + j * a_dim1] = p * a[*l + 1 + j * a_dim1] - q * a[*l + j * 
		a_dim1];
	a[*l + j * a_dim1] = s;
/* L10: */
    }
    i__1 = l1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	s = p * a[i__ + *l * a_dim1] + q * a[i__ + (*l + 1) * a_dim1];
	a[i__ + (*l + 1) * a_dim1] = p * a[i__ + (*l + 1) * a_dim1] - q * a[
		i__ + *l * a_dim1];
	a[i__ + *l * a_dim1] = s;
/* L20: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	s = p * v[i__ + *l * v_dim1] + q * v[i__ + (*l + 1) * v_dim1];
	v[i__ + (*l + 1) * v_dim1] = p * v[i__ + (*l + 1) * v_dim1] - q * v[
		i__ + *l * v_dim1];
	v[i__ + *l * v_dim1] = s;
/* L30: */
    }
    a[*l + 1 + *l * a_dim1] = 0.;
    return 0;
L40:
/* INTERCHANGE 1X1 AND 2X2 BLOCKS. */
    x = a[*l + *l * a_dim1];
    p = 1.;
    q = 1.;
    r__ = 1.;
    i__1 = *l + 2;
    qrstep(&a[a_offset], &v[v_offset], &p, &q, &r__, l, &i__1, n, na, nv);
    it = 0;
L50:
    ++it;
    if (it <= 30) {
	goto L60;
    }
    *fail = TRUE_;
    return 0;
L60:
    p = a[*l + *l * a_dim1] - x;
    q = a[*l + 1 + *l * a_dim1];
    r__ = 0.;
    i__1 = *l + 2;
    qrstep(&a[a_offset], &v[v_offset], &p, &q, &r__, l, &i__1, n, na, nv);
    if ((d__1 = a[*l + 2 + (*l + 1) * a_dim1], fabs(d__1)) > *eps * ((d__2 = a[
	    *l + 1 + (*l + 1) * a_dim1], fabs(d__2)) + (d__3 = a[*l + 2 + (*l 
	    + 2) * a_dim1], fabs(d__3)))) {
	goto L50;
    }
    a[*l + 2 + (*l + 1) * a_dim1] = 0.;
    return 0;
L70:
/* INTERCHANGE 2X2 AND B2XB2 BLOCKS. */
    m = *l + 2;
    if (*b2 == 2) {
	++m;
    }
    x = a[*l + 1 + (*l + 1) * a_dim1];
    y = a[*l + *l * a_dim1];
    w = a[*l + 1 + *l * a_dim1] * a[*l + (*l + 1) * a_dim1];
    p = 1.;
    q = 1.;
    r__ = 1.;
    qrstep(&a[a_offset], &v[v_offset], &p, &q, &r__, l, &m, n, na, nv);
    it = 0;
L80:
    ++it;
    if (it <= 30) {
	goto L90;
    }
    *fail = TRUE_;
    return 0;
L90:
    z__ = a[*l + *l * a_dim1];
    r__ = x - z__;
    s = y - z__;
    p = (r__ * s - w) / a[*l + 1 + *l * a_dim1] + a[*l + (*l + 1) * a_dim1];
    q = a[*l + 1 + (*l + 1) * a_dim1] - z__ - r__ - s;
    r__ = a[*l + 2 + (*l + 1) * a_dim1];
    s = fabs(p) + fabs(q) + fabs(r__);
    p /= s;
    q /= s;
    r__ /= s;
    qrstep(&a[a_offset], &v[v_offset], &p, &q, &r__, l, &m, n, na, nv);
    if ((d__1 = a[m - 1 + (m - 2) * a_dim1], fabs(d__1)) > *eps * ((d__2 = a[m 
	    - 1 + (m - 1) * a_dim1], fabs(d__2)) + (d__3 = a[m - 2 + (m - 2) * 
	    a_dim1], fabs(d__3)))) {
	goto L80;
    }
    a[m - 1 + (m - 2) * a_dim1] = 0.;
    return 0;
} /* exchng_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
qrstep(doublereal *a, doublereal *v, doublereal *p, doublereal *q, doublereal *r__, integer *nl, integer *nu, integer *n, integer *na, integer *nv)
{
    /* System generated locals */
    integer a_dim1, a_offset, v_dim1, v_offset, i__1, i__2;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    static logical last;
    static integer i__, j, k;
    static doublereal s, x, y, z__;
    static integer nl2, nl3, num1;

/* QRSTEP PERFORMS ONE IMPLICIT QR STEP ON THE */
/* UPPER HESSENBERG MATRIX A.  THE SHIFT IS DETERMINED */
/* BY THE NUMBERS P,Q, AND R, AND THE STEP IS APPLIED TO */
/* ROWS AND COLUMNS NL THROUGH NU.  THE TRANSFORMATIONS */
/* SEQUENCE ARE (STARRED APRAMETERS ARE ALTERED BY THE */
/* SUBROUTINE) */
/*    *A       THE UPPER HESSENBERG MATRIX ON WHICH THE */
/*             QR STEP IS TO BE PERFORMED. */
/*    *V       THE ARRAY IN WHICH THE TRANSFORMATIONS */
/*             ARE TO BE ACCUMULATED */
/*    *Q */
/*    *R */
/*     NL      THE LOWER LIMIT OF THE STEP. */
/*     NU      THE UPPER LIMIT OF THE STEP. */
/*     N       THE ORDER OF THE MATRIX A. */
/*     NA      THE FIRST DIMENSION OF THE ARRAY A. */
/*     NV      THE FIRST DIMENSION OF THE ARRAY V. */
/* INTERNAL VARIABLES. */
    /* Parameter adjustments */
    a_dim1 = *na;
    a_offset = a_dim1 + 1;
    a -= a_offset;
    v_dim1 = *nv;
    v_offset = v_dim1 + 1;
    v -= v_offset;

    /* Function Body */
    nl2 = *nl + 2;
    i__1 = *nu;
    for (i__ = nl2; i__ <= i__1; ++i__) {
	a[i__ + (i__ - 2) * a_dim1] = 0.;
/* L10: */
    }
    if (nl2 == *nu) {
	goto L30;
    }
    nl3 = *nl + 3;
    i__1 = *nu;
    for (i__ = nl3; i__ <= i__1; ++i__) {
	a[i__ + (i__ - 3) * a_dim1] = 0.;
/* L20: */
    }
L30:
    num1 = *nu - 1;
    i__1 = num1;
    for (k = *nl; k <= i__1; ++k) {
/* DETERMINE THE TRANSFORMATION. */
	last = k == num1;
	if (k == *nl) {
	    goto L40;
	}
	*p = a[k + (k - 1) * a_dim1];
	*q = a[k + 1 + (k - 1) * a_dim1];
	*r__ = 0.;
	if (! last) {
	    *r__ = a[k + 2 + (k - 1) * a_dim1];
	}
	x = fabs(*p) + fabs(*q) + fabs(*r__);
	if (x == 0.) {
	    goto L130;
	}
	*p /= x;
	*q /= x;
	*r__ /= x;
L40:
/* Computing 2nd power */
	d__1 = *p;
/* Computing 2nd power */
	d__2 = *q;
/* Computing 2nd power */
	d__3 = *r__;
	s = sqrt(d__1 * d__1 + d__2 * d__2 + d__3 * d__3);
	if (*p < 0.) {
	    s = -s;
	}
	if (k == *nl) {
	    goto L50;
	}
	a[k + (k - 1) * a_dim1] = -s * x;
	goto L60;
L50:
	if (*nl != 1) {
	    a[k + (k - 1) * a_dim1] = -a[k + (k - 1) * a_dim1];
	}
L60:
	*p += s;
	x = *p / s;
	y = *q / s;
	z__ = *r__ / s;
	*q /= *p;
	*r__ /= *p;
/* PREMULTIPLY. */
	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
	    *p = a[k + j * a_dim1] + *q * a[k + 1 + j * a_dim1];
	    if (last) {
		goto L70;
	    }
	    *p += *r__ * a[k + 2 + j * a_dim1];
	    a[k + 2 + j * a_dim1] -= *p * z__;
L70:
	    a[k + 1 + j * a_dim1] -= *p * y;
	    a[k + j * a_dim1] -= *p * x;
/* L80: */
	}
/* POSTMULTIPLY. */
/* Computing MIN */
	i__2 = k + 3;
	j = min(i__2,*nu);
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    *p = x * a[i__ + k * a_dim1] + y * a[i__ + (k + 1) * a_dim1];
	    if (last) {
		goto L90;
	    }
	    *p += z__ * a[i__ + (k + 2) * a_dim1];
	    a[i__ + (k + 2) * a_dim1] -= *p * *r__;
L90:
	    a[i__ + (k + 1) * a_dim1] -= *p * *q;
	    a[i__ + k * a_dim1] -= *p;
/* L100: */
	}
/* ACCUMULATE THE TRANSFORMATION IN V. */
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    *p = x * v[i__ + k * v_dim1] + y * v[i__ + (k + 1) * v_dim1];
	    if (last) {
		goto L110;
	    }
	    *p += z__ * v[i__ + (k + 2) * v_dim1];
	    v[i__ + (k + 2) * v_dim1] -= *p * *r__;
L110:
	    v[i__ + (k + 1) * v_dim1] -= *p * *q;
	    v[i__ + k * v_dim1] -= *p;
/* L120: */
	}
L130:
	;
    }
    return 0;
} /* qrstep_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
orthes(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, doublereal *ort)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;

    /* Local variables */
    static doublereal f, g, h__;
    static integer i__, j, m;
    static doublereal scale;
    static integer la, ii, jj, mp, kp1;



/*     this subroutine is a translation of the algol procedure orthes, */
/*     num. math. 12, 349-368(1968) by martin and wilkinson. */
/*     handbook for auto. comp., vol.ii-linear algebra, 339-358(1971). */

/*     given a real general matrix, this subroutine */
/*     reduces a submatrix situated in rows and columns */
/*     low through igh to upper hessenberg form by */
/*     orthogonal similarity transformations. */

/*     on input */

/*        nm must be set to the row dimension of two-dimensional */
/*          array parameters as declared in the calling program */
/*          dimension statement. */

/*        n is the order of the matrix. */

/*        low and igh are integers determined by the balancing */
/*          subroutine  balanc.  if  balanc  has not been used, */
/*          set low=1, igh=n. */

/*        a contains the input matrix. */

/*     on output */

/*        a contains the hessenberg matrix.  information about */
/*          the orthogonal transformations used in the reduction */
/*          is stored in the remaining triangle under the */
/*          hessenberg matrix. */

/*        ort contains further information about the transformations. */
/*          only elements low through igh are used. */

/*     questions and comments should be directed to burton s. garbow, */
/*     mathematics and computer science div, argonne national laboratory 
*/

/*     this version dated august 1983. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    a_dim1 = *nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;
    --ort;

    /* Function Body */
    la = *igh - 1;
    kp1 = *low + 1;
    if (la < kp1) {
	goto L200;
    }

    i__1 = la;
    for (m = kp1; m <= i__1; ++m) {
	h__ = 0.;
	ort[m] = 0.;
	scale = 0.;
/*     .......... scale column (algol tol then not needed) .......... 
*/
	i__2 = *igh;
	for (i__ = m; i__ <= i__2; ++i__) {
/* L90: */
	    scale += (d__1 = a[i__ + (m - 1) * a_dim1], fabs(d__1));
	}

	if (scale == 0.) {
	    goto L180;
	}
	mp = m + *igh;
/*     .......... for i=igh step -1 until m do -- .......... */
	i__2 = *igh;
	for (ii = m; ii <= i__2; ++ii) {
	    i__ = mp - ii;
	    ort[i__] = a[i__ + (m - 1) * a_dim1] / scale;
	    h__ += ort[i__] * ort[i__];
/* L100: */
	}

	d__1 = sqrt(h__);
	g = -d_sign(d__1, ort[m]);
	h__ -= ort[m] * g;
	ort[m] -= g;
/*     .......... form (i-(u*ut)/h) * a .......... */
	i__2 = *n;
	for (j = m; j <= i__2; ++j) {
	    f = 0.;
/*     .......... for i=igh step -1 until m do -- .......... */
	    i__3 = *igh;
	    for (ii = m; ii <= i__3; ++ii) {
		i__ = mp - ii;
		f += ort[i__] * a[i__ + j * a_dim1];
/* L110: */
	    }

	    f /= h__;

	    i__3 = *igh;
	    for (i__ = m; i__ <= i__3; ++i__) {
/* L120: */
		a[i__ + j * a_dim1] -= f * ort[i__];
	    }

/* L130: */
	}
/*     .......... form (i-(u*ut)/h)*a*(i-(u*ut)/h) .......... */
	i__2 = *igh;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    f = 0.;
/*     .......... for j=igh step -1 until m do -- .......... */
	    i__3 = *igh;
	    for (jj = m; jj <= i__3; ++jj) {
		j = mp - jj;
		f += ort[j] * a[i__ + j * a_dim1];
/* L140: */
	    }

	    f /= h__;

	    i__3 = *igh;
	    for (j = m; j <= i__3; ++j) {
/* L150: */
		a[i__ + j * a_dim1] -= f * ort[j];
	    }

/* L160: */
	}

	ort[m] = scale * ort[m];
	a[m + (m - 1) * a_dim1] = scale * g;
L180:
	;
    }

L200:
    return 0;
} /* orthes_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* Subroutine */ int 
ortran(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, doublereal *ort, doublereal *z__)
{
    /* System generated locals */
    integer a_dim1, a_offset, z_dim1, z_offset, i__1, i__2, i__3;

    /* Local variables */
    static doublereal g;
    static integer i__, j, kl, mm, mp, mp1;



/*     this subroutine is a translation of the algol procedure ortrans, */
/*     num. math. 16, 181-204(1970) by peters and wilkinson. */
/*     handbook for auto. comp., vol.ii-linear algebra, 372-395(1971). */

/*     this subroutine accumulates the orthogonal similarity */
/*     transformations used in the reduction of a real general */
/*     matrix to upper hessenberg form by  orthes. */

/*     on input */

/*        nm must be set to the row dimension of two-dimensional */
/*          array parameters as declared in the calling program */
/*          dimension statement. */

/*        n is the order of the matrix. */

/*        low and igh are integers determined by the balancing */
/*          subroutine  balanc.  if  balanc  has not been used, */
/*          set low=1, igh=n. */

/*        a contains information about the orthogonal trans- */
/*          formations used in the reduction by  orthes */
/*          in its strict lower triangle. */

/*        ort contains further information about the trans- */
/*          formations used in the reduction by  orthes. */
/*          only elements low through igh are used. */

/*     on output */

/*        z contains the transformation matrix produced in the */
/*          reduction by  orthes. */

/*        ort has been altered. */

/*     questions and comments should be directed to burton s. garbow, */
/*     mathematics and computer science div, argonne national laboratory 
*/

/*     this version dated august 1983. */

/*     ------------------------------------------------------------------ 
*/

/*     .......... initialize z to identity matrix .......... */
    /* Parameter adjustments */
    z_dim1 = *nm;
    z_offset = z_dim1 + 1;
    z__ -= z_offset;
    --ort;
    a_dim1 = *nm;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    /* Function Body */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {

	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* L60: */
	    z__[i__ + j * z_dim1] = 0.;
	}

	z__[j + j * z_dim1] = 1.;
/* L80: */
    }

    kl = *igh - *low - 1;
    if (kl < 1) {
	goto L200;
    }
/*     .......... for mp=igh-1 step -1 until low+1 do -- .......... */
    i__1 = kl;
    for (mm = 1; mm <= i__1; ++mm) {
	mp = *igh - mm;
	if (a[mp + (mp - 1) * a_dim1] == 0.) {
	    goto L140;
	}
	mp1 = mp + 1;

	i__2 = *igh;
	for (i__ = mp1; i__ <= i__2; ++i__) {
/* L100: */
	    ort[i__] = a[i__ + (mp - 1) * a_dim1];
	}

	i__2 = *igh;
	for (j = mp; j <= i__2; ++j) {
	    g = 0.;

	    i__3 = *igh;
	    for (i__ = mp; i__ <= i__3; ++i__) {
/* L110: */
		g += ort[i__] * z__[i__ + j * z_dim1];
	    }
/*     .......... divisor below is negative of h formed in orthes.
 */
/*                double division avoids possible underflow ......
.... */
	    g = g / ort[mp] / a[mp + (mp - 1) * a_dim1];

	    i__3 = *igh;
	    for (i__ = mp; i__ <= i__3; ++i__) {
/* L120: */
		z__[i__ + j * z_dim1] += g * ort[i__];
	    }

/* L130: */
	}

L140:
	;
    }

L200:
    return 0;
} /* ortran_ */

