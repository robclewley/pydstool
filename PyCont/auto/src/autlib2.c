/* Autlib2.f -- translated by f2c (version 19970805).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "auto_f2c.h"
#include "auto_c.h"
//#include "malloc.h"
// DREW WUZ HERE
# include <time.h>

typedef struct {
  doublereal ***a;
  doublereal ***b;
  doublereal ***c;
  doublereal **d;
  doublereal ***a1;
  doublereal ***a2;
  doublereal ***s1;
  doublereal ***s2;
  doublereal ***bb;
  doublereal ***cc;
  doublereal **faa;
  doublereal ***ca1;

  integer *icf;
  integer *irf;
  integer *ipr;
  integer *icf11;
  integer *icf1;
  integer *icf2;
  integer *np;
} main_auto_storage_type;

void print_jacobian(iap_type iap,main_auto_storage_type data) {
  int i,j,k,l;
  int num_rows_A = iap.ndim * iap.ncol;
  int num_columns_A = iap.ndim * (iap.ncol + 1);
  int num_columns_B = iap.nfpr;
#ifndef MANIFOLD
  int num_rows_C = iap.nbc + iap.nint + 1;
#else
  int num_rows_C = iap.nbc + iap.nint + iap.nalc;
#endif
  int numblocks = iap.ntst;
  FILE *fp;
  static int num_calls=0;
  char filename[80];

  sprintf(filename,"jacobian%03d",num_calls);
  fp=fopen(filename,"w");
  num_calls++;

  for(i=0;i<numblocks;i++){
    for(j=0;j<num_rows_A;j++){
      /* Print zeros in front first */
      for(k=0;k<i*(num_columns_A-iap.ndim);k++)
	fprintf(fp,"%18.10e ",0.0);
      /* Now print line from block */
      for(k=0;k<num_columns_A;k++)
	fprintf(fp,"%18.10e ",data.a[i][j][k]);
      /* Now put zeros at end of line */
      for(k=i*(num_columns_A-iap.ndim)+num_columns_A;k<(num_columns_A-iap.ndim)*numblocks+iap.ndim;k++)
	fprintf(fp,"%18.10e ",0.0);
      /* Put in B */
      for(k=0;k<num_columns_B;k++)
	fprintf(fp,"%18.10e ",data.b[i][j][k]);
      fprintf(fp,"\n");
    }
  }

  /*For printing out C there needs to be a summation of the edge guys*/
  for(j=0;j<num_rows_C;j++) {
    /*The first num_rows_A columns are ok as the are*/
    for(k=0;k<(num_columns_A-iap.ndim);k++)
      fprintf(fp,"%18.10e ",data.c[0][j][k]);
    /* Now print out the rest of the blocks, doing a summation at the beginning of each */
    for(i=1;i<numblocks;i++) {
      for(k=0;k<iap.ndim;k++)
	fprintf(fp,"%18.10e ",data.c[i-1][j][k+ num_columns_A-iap.ndim] +
		data.c[i][k][j]);
      for(k=iap.ndim;k<num_columns_A-iap.ndim;k++)
	fprintf(fp,"%18.10e ",data.c[i][j][k]);
    }
    /*Now print out last column*/
    for(k=num_columns_A-iap.ndim;k<num_columns_A;k++)
      fprintf(fp,"%18.10e ",data.c[numblocks-1][j][k]);
    for(l=0;l<num_columns_B;l++)
      fprintf(fp,"%18.10e ",data.d[l][j]);
    fprintf(fp,"\n");
  }


  fclose(fp);

}

void print_ups_rlcur(iap_type iap,doublereal **ups,doublereal *rlcur) {
  FILE *fp;
  static int num_calls=0;
  char filename[80];
  int i,j;
  
  sprintf(filename,"ups_rlcur%03d",num_calls);
  fp=fopen(filename,"w");
  num_calls++;
  for(i=0;i<(iap.ndim)*(iap.ncol);i++)
    for(j=0;j<iap.ntst+1;j++)
      fprintf(fp,"%18.10e\n",ups[j][i]);
  for(i=0;i<iap.nfpr;i++)
    fprintf(fp,"%18.10e\n",rlcur[i]);

  fclose(fp);

}

void print_fa_fc(iap_type iap,doublereal **fa,doublereal *fc,char *filename) {
  FILE *fp;
  int i,j;
  int num_rows_A = iap.ndim * iap.ncol;
  int numblocks = iap.ntst;

  fp=fopen(filename,"w");

  for(i=0;i<numblocks;i++)
    for(j=0;j<num_rows_A;j++)
      fprintf(fp,"%18.10e\n",fa[j][i]);
  for(i=0;i<iap.nfpr+iap.ndim;i++)
    fprintf(fp,"%10.10e\n",fc[i]);

  fclose(fp);

}

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*           Setting up of the Jacobian and right hand side */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */

/*     ---------- ------ */
/* Subroutine */ int 
solvbv(integer *ifst, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), doublereal *rds, integer *nllv, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal *dtm, doublereal **fa, doublereal *fc, doublereal **p0, doublereal **p1, doublereal *thl, doublereal *thu)
{
  
  
  /* Local variables */
  
  integer ndim;
  logical ipar;
  integer ncol, nclm, nfpr, nint, nrow, ntst, ntst0;
#ifdef MANIFOLD
  integer nalc = iap->nalc;
#else
#define nalc 1
#endif
  
  doublereal **ff, **ft;
  
  integer nbc, iid, iam;
  doublereal det;
  integer ips, nrc;
  
  integer kwt;
  
  static main_auto_storage_type main_auto_storage={NULL,NULL,NULL,NULL,
						   NULL,NULL,NULL,NULL,
						   NULL,NULL,NULL,NULL,
						   NULL,NULL,NULL,NULL,
						   NULL,NULL,NULL};
  
  /*     N AX is the local N TSTX, which is smaller than the global N TSTX. */
  /*     NODES is the total number of nodes. */
  
  
  /* Sets up and solves the linear equations for one Newton/Chord iteration 
   */
  
  
  /* Most of the required memory is allocated below */
  /* This is an interesting section of code.  The main point
     is that setubv and conpar only get called when ifst
     is 1.  This is a optimization since you can solve
     the system using the previously factored jacobian.
     One thing to watch out for is that two seperate calls
     of solvbv_ talk to each other through these arrays,
     so it is only safe to get rid of them when ifst is
     1 (since their entries will then be recreated in conpar
     and setubv).
  */

  ff = DMATRIX(iap->ndim * iap->ncol, iap->ntst + 1);
  ft = DMATRIX(iap->ndim * iap->ncol, iap->ntst + 1);
    
  if (*ifst==1){
    /* The formulas used for the allocation are somewhat complex, but they
       are based on following macros (the space after the first letter is 
       for the scripts which detect these things automatically, the original
       name does not have the space:
       
       M 1AAR =  (((iap->ndim * iap->ncol ) + iap->ndim ) )      
       M 2AA  =	((iap->ndim * iap->ncol ) )                     
       N AX   =	(iap->ntst /NODES+1)                            
       M 1BB  =	(NPARX)                                         
       M 2BB  =	((iap->ndim * iap->ncol ) )                     
       M 1CC  =	((((iap->ndim * iap->ncol ) + iap->ndim ) ) )   
       M 2CC  =	(((iap->ndim +3) +NINTX+1) )                    
       M 1DD  =	(((iap->ndim +3) +NINTX+1) )                    
       M 2DD  =	(NPARX)                                         
       N RCX  =	((iap->ndim +3) +NINTX+1)                       
       N CLMX =	((iap->ndim * iap->ncol ) + iap->ndim )         
       N ROWX =	(iap->ndim * iap->ncol )                        
    */
    
    /* Free floating point arrays */
    FREE_DMATRIX_3D(main_auto_storage.a);
    FREE_DMATRIX_3D(main_auto_storage.b);
    FREE_DMATRIX_3D(main_auto_storage.c);
    FREE_DMATRIX(main_auto_storage.d);
    FREE_DMATRIX_3D(main_auto_storage.a1);
    FREE_DMATRIX_3D(main_auto_storage.a2);
    FREE_DMATRIX_3D(main_auto_storage.s1);
    FREE_DMATRIX_3D(main_auto_storage.s2);
    FREE_DMATRIX_3D(main_auto_storage.bb);
    FREE_DMATRIX_3D(main_auto_storage.cc);
    FREE_DMATRIX(main_auto_storage.faa);
    FREE_DMATRIX_3D(main_auto_storage.ca1);
    
    /* Free integer arrays */
    FREE(main_auto_storage.icf);
    FREE(main_auto_storage.irf);
    FREE(main_auto_storage.ipr);
    FREE(main_auto_storage.icf11);
    FREE(main_auto_storage.icf1);
    FREE(main_auto_storage.icf2);
    FREE(main_auto_storage.np);
    
    /*(M 1AAR*M 2AA*N AX) */
    main_auto_storage.a=DMATRIX_3D(iap->ntst + 1,
                                   iap->ncol * iap->ndim,
                                   (iap->ncol + 1) * iap->ndim); 
    /*(M 1BB*M 2BB*N AX)*/ 
    main_auto_storage.b=DMATRIX_3D(iap->ntst + 1, iap->ndim * iap->ncol, NPARX);
    /*(M 1CC*M 2CC*N AX)*/ 
    main_auto_storage.c=DMATRIX_3D(iap->ntst + 1,
                                   iap->nbc + iap->nint + nalc,
                                   (iap->ncol + 1) * iap->ndim);
    /*(M 1DD*M 2DD)*/ 
    main_auto_storage.d=DMATRIX(iap->nbc + iap->nint + nalc, NPARX);
    /*(iap->ndim * iap->ndim *N AX)*/ 
    main_auto_storage.a1=DMATRIX_3D(iap->ntst + 1, iap->ndim, iap->ndim);
    /*(iap->ndim * iap->ndim *N AX)*/ 
    main_auto_storage.a2=DMATRIX_3D(iap->ntst + 1, iap->ndim, iap->ndim);
    /*(iap->ndim * iap->ndim *N AX)*/ 
    main_auto_storage.s1=DMATRIX_3D(iap->ntst + 1, iap->ndim, iap->ndim);
    /*(iap->ndim * iap->ndim *N AX)*/ 
    main_auto_storage.s2=DMATRIX_3D(iap->ntst + 1, iap->ndim, iap->ndim);
    /*(iap->ndim *N PARX*N AX)*/ 
    main_auto_storage.bb=DMATRIX_3D(iap->ntst + 1, iap->ndim, NPARX);
    /*(N RCX* iap->ndim *N AX+1)*/ 
    main_auto_storage.cc=DMATRIX_3D(iap->ntst + 1, iap->nbc + iap->nint + nalc, iap->ndim);

    /*(iap->ndim *N AX)*/ 
    main_auto_storage.faa=DMATRIX(iap->ndim, iap->ntst +1);

    /*(iap->ndim * iap->ndim *K REDO)*/ 
    main_auto_storage.ca1=DMATRIX_3D(KREDO, iap->ndim, iap->ndim);
    
    /*(N CLMX*N AX)*/ 
    main_auto_storage.icf=(integer *)MALLOC(sizeof(integer)*(((iap->ndim * iap->ncol ) + iap->ndim ) * (iap->ntst +1) ) );
    /*(N ROWX*N AX)*/ 
    main_auto_storage.irf=(integer *)MALLOC(sizeof(integer)*((iap->ndim * iap->ncol ) * (iap->ntst +1) ) );
    /*(iap->ndim *N AX)*/ 
    main_auto_storage.ipr=(integer *)MALLOC(sizeof(integer)*(iap->ndim * (iap->ntst +1) ) );
    /*(iap->ndim *K REDO)*/ 
    main_auto_storage.icf11=(integer *)MALLOC(sizeof(integer)*(iap->ndim *KREDO) );
    /*(iap->ndim *N AX)*/ 
    main_auto_storage.icf1=(integer *)MALLOC(sizeof(integer)*(iap->ndim * (iap->ntst +1) ));
    /*(iap->ndim *N AX)*/ 
    main_auto_storage.icf2=(integer *)MALLOC(sizeof(integer)*(iap->ndim * (iap->ntst +1) )); 
    /*(2)*/ 
    main_auto_storage.np=(integer *)MALLOC(sizeof(integer)*(2) );
  }
  
  
  iam = iap->mynode;
  kwt = iap->numnodes;
  if (kwt > 1) {
    ipar = TRUE_;
  } else {
    ipar = FALSE_;
  }
  
  ndim = iap->ndim;
  ips = iap->ips;
  ntst = iap->ntst;
  ncol = iap->ncol;
  nbc = iap->nbc;
  nint = iap->nint;
  iid = iap->iid;
  nfpr = iap->nfpr;
  nrc = nbc + nint + 1;
  nrc = nbc + nint + nalc;
  nrow = ndim * ncol;
  nclm = nrow + ndim;
  
  if (kwt > ntst) {
    printf("NTST is less than the number of nodes\n");
    exit(0);
  } else {
    partition(&ntst, &kwt, main_auto_storage.np);
  }
  
  /*     NTST0 is the global one, NTST is the local one. */
  /*     The value of NTST may be different in different nodes. */
  ntst0 = ntst;
  ntst = main_auto_storage.np[iam];
  
  if (*ifst == 1) {
#ifdef USAGE
    struct rusage *setubv_usage;
    usage_start(&setubv_usage);
#endif
#ifndef MANIFOLD
    setubv(ndim, ips, ntst, ncol, nbc, nint, nfpr, nrc, nrow, nclm,
	   funi, bcni, icni, *ndxloc, iap, rap, par, icp, 
	   *rds, main_auto_storage.a, main_auto_storage.b, main_auto_storage.c, main_auto_storage.d, ft, fc, rlcur, 
	   rlold, rldot, ups, uoldps, udotps, upoldp, dups, 
	   dtm, thl, thu, p0, p1);
#else
    setubv(ndim, ips, ntst, ncol, nbc, nint, nalc, nfpr, nrc, nrow, nclm,
	   funi, bcni, icni, *ndxloc, iap, rap, par, icp, 
	   rds, main_auto_storage.a, main_auto_storage.b, main_auto_storage.c, main_auto_storage.d, ft, fc, rlcur, 
	   rlold, rldot, ups, uoldps, udotps, upoldp, dups, 
	   dtm, thl, thu, p0, p1);
#endif
#ifdef USAGE
    usage_end(setubv_usage,"all of setubv");
#endif
  } else {
#ifndef MANIFOLD
    setrhs(&ndim, &ips, &ntst, &ntst0, main_auto_storage.np, &ncol, &nbc, &nint, &
	   nfpr, &nrc, &nrow, &nclm, &iam, &kwt, &ipar, funi, bcni, icni,
	   ndxloc, iap, rap, par, icp, rds, ft, fc, rlcur, 
	   rlold, rldot, ups, uoldps, udotps, upoldp, dups, dtm, thl, 
	   thu, p0, p1);
#else
    setrhs(&ndim, &ips, &ntst, &ntst0, main_auto_storage.np, &ncol, &nbc, &nint, &nalc, &
	   nfpr, &nrc, &nrow, &nclm, &iam, &kwt, &ipar, funi, bcni, icni,
	   ndxloc, iap, rap, par, icp, rds, ft, fc, rlcur, 
	   rlold, rldot, ups, uoldps, udotps, upoldp, dups, dtm, thl, 
	   thu, p0, p1);
#endif
  }
  /*     The matrix D and FC are set to zero for all nodes except the first.
   */
  if (iam > 0) {
    setfcdd(ifst, main_auto_storage.d, fc, &nfpr, &nrc);
  }

#ifdef MATLAB_OUTPUT
  print_jacobian(*iap,main_auto_storage);
  {
    static num_calls = 0;
    char filename[80];
    sprintf(filename,"before%03d",num_calls);
    num_calls++;
    print_fa_fc(*iap,ft,fc,filename);
  }
#endif
  brbd(main_auto_storage.a, main_auto_storage.b, main_auto_storage.c, main_auto_storage.d, ft, fc, p0, p1, 
       ifst, &iid, nllv, &det, &ndim, &ntst, &nbc, &nrow, &nclm, &nfpr, &
       nrc, &iam, &kwt, &ipar, main_auto_storage.a1, main_auto_storage.a2, main_auto_storage.bb, 
       main_auto_storage.cc, main_auto_storage.faa, main_auto_storage.ca1, main_auto_storage.s1, main_auto_storage.s2, 
       main_auto_storage.icf11, main_auto_storage.ipr, main_auto_storage.icf1, main_auto_storage.icf2, 
       main_auto_storage.irf, main_auto_storage.icf);
  
  /*
    This is some stuff from the parallel version that isn't needed anymore 
    ----------------------------------------------------------------------
    lenft = ntst * nrow << 3;
    lenff = ntst0 * nrow << 3;
    jtmp1 = M 2AA;   I added spaces so these don't get flagged as header file macro dependancies
    jtmp2 = M 3AA;   I added spaces so these don't get flagged as header file macro dependancies
    lenff2 = jtmp1 * (jtmp2 + 1) << 3;
  */
  if (ipar) {
    /*        Global concatenation of the solution from each node. */
    integer tmp;
    gcol();
    tmp = iap->ntst + 1;
    faft(ff, fa, &ntst0, &nrow, ndxloc);
  } else {
    integer tmp;
    tmp = iap->ntst + 1;
    faft(ft, fa, &ntst0, &nrow, ndxloc);
  }
#ifdef MATLAB_OUTPUT
  {
    static num_calls = 0;
    char filename[80];
    sprintf(filename,"after%03d",num_calls);
    num_calls++;
    print_fa_fc(*iap,ft,fc,filename);
  }
#endif  

  rap->det = det;
  FREE_DMATRIX(ff);
  FREE_DMATRIX(ft);
  return 0;
} /* solvbv_ */


/*     ---------- ------- */
/* Subroutine */ int 
setfcdd(integer *ifst, doublereal **dd, doublereal *fc, integer *ncb, integer *nrc)
{
    /* Local variables */
  integer i, j;

  for (i = 0; i < *nrc; ++i) {
    if (*ifst == 1) {
      for (j = 0; j < *ncb; ++j) {
	dd[i][j] = 0.;
      }
    }
    fc[i] = 0.;
  }


  return 0;
} /* setfcdd_ */


/*     ---------- ---- */
/* Subroutine */ int 
faft(doublereal **ff, doublereal **fa, integer *ntst, integer *nrow, integer *ndxloc)
{
    /* Local variables */
  integer i, j;

  for (i = 0; i < *ntst; ++i) {
    for (j = 0; j < *nrow; ++j) {
      fa[i][j] = ff[j][i];
    }
  }

  return 0;
} /* faft_ */


/*     ---------- --------- */
/* Subroutine */ int 
partition(integer *n, integer *kwt, integer *m)
{
    /* Local variables */
  integer i, s, t;


  /*     Linear distribution of NTST over all nodes */

    /* Parameter adjustments */
    /*--m;*/

    
  t = *n / *kwt;
  s = *n % *kwt;

  for (i = 0; i < *kwt; ++i) {
    m[i] = t;
  }

  for (i = 0; i < s; ++i) {
    ++m[i];
  }

  return 0;
} /* partition_ */


/*     ------- -------- ------ */
integer 
mypart(integer *iam, integer *np)
{
  /* System generated locals */
  integer ret_val;

    /* Local variables */
  integer i, k;




  /*     Partition the mesh */


    /* Parameter adjustments */
    /*--np;*/

    
  k = 0;
  for (i = 0; i < *iam; ++i) {
    k += np[i];
  }
  ret_val = k;

  return ret_val;
} /* mypart_ */


/*     ---------- ------ */
#ifndef MANIFOLD
/* Subroutine */ int 
setrhs(integer *ndim, integer *ips, integer *na, integer *ntst, integer *np, integer *ncol, integer *nbc, integer *nint, integer *ncb, integer *nrc, integer *nra, integer *nca, integer *iam, integer *kwt, logical *ipar, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer *ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, doublereal **fa, doublereal *fc, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1)
#else
setrhs(integer *ndim, integer *ips, integer *na, integer *ntst, integer *np, integer *ncol, integer *nbc, integer *nint, integer *nalc, integer *ncb, integer *nrc, integer *nra, integer *nca, integer *iam, integer *kwt, logical *ipar, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer *ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, doublereal **fa, doublereal *fc, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1)
#endif
{
  integer i, j, k, l, m;
  integer mpart, i1, j1, k1, l1;

  doublereal rlsum;
  integer ib, ic, jj;
  integer ic1;

  integer jp1;
  integer ncp1;
  doublereal dt,ddt;

  doublereal *dicd = NULL, *ficd = NULL, *dfdp, *dfdu, *uold;
  doublereal *f;
  doublereal *u, **wploc;
  doublereal *wi, **wp, **wt;
  doublereal *dbc, *fbc, *uic, *uio, *prm, *uid, *uip, *ubc0, *ubc1;
#ifdef MANIFOLD
  integer udotps_off;
#endif

  if (iap->nint > 0)
  {
    dicd = (doublereal *)MALLOC(sizeof(doublereal)*(iap->nint)*(iap->ndim + NPARX));
    ficd = (doublereal *)MALLOC(sizeof(doublereal)*(iap->nint));
  }  
  dfdp = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim)*NPARX);
  dfdu = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim)*(iap->ndim));
  uold = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  f    = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  u    = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  wploc= DMATRIX(iap->ncol+1, iap->ncol);
  wi   = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ncol+1) );
  wp   = DMATRIX(iap->ncol+1, iap->ncol);
  wt   = DMATRIX(iap->ncol+1, iap->ncol);
  dbc  = (doublereal *)MALLOC(sizeof(doublereal)*(iap->nbc)*(2*iap->ndim + NPARX));
  fbc  = (doublereal *)MALLOC(sizeof(doublereal)*(iap->nbc));
  uic  = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  uio  = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  prm  = (doublereal *)MALLOC(sizeof(doublereal)*NPARX);
  uid  = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  uip  = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  ubc0 = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));
  ubc1 = (doublereal *)MALLOC(sizeof(doublereal)*(iap->ndim));


  /* Parameter adjustments */
  /*--np;*/
  /*--par;*/
  /*--icp;*/
  /*--fc;*/
  /*--rlcur;*/
  /*--rlold;*/
  /*--rldot;*/
  /*--dtm;*/
  /*--thl;*/
  /*--thu;*/
    
  *iam = iap->mynode;
  *kwt = iap->numnodes;
  if (*kwt > 1) {
    *ipar = TRUE_;
  } else {
    *ipar = FALSE_;
  }

  wint(*ncol + 1, wi);
  genwts(*ncol, iap->ncol + 1, wt, wp);
  /* Initialize to zero. */
  for (i = 0; i < *nrc; ++i) {
    fc[i] = 0.;
  }

  /* Set constants. */
  ncp1 = *ncol + 1;
  for (i = 0; i < *ncb; ++i) {
    par[icp[i]] = rlcur[i];
  }

  /* Generate FA : */

/*      Partition the mesh intervals. */
  mpart = mypart(iam, np);

  for (jj = 0; jj < *na; ++jj) {
    j = jj + mpart;
    jp1 = j + 1;
    dt = dtm[j];
    ddt = 1. / dt;
    for (ic = 0; ic < *ncol; ++ic) {
      for (ib = 0; ib < ncp1; ++ib) {
	wploc[ib][ic] = ddt * wp[ib][ic];
      }
    }
    for (ic = 0; ic < *ncol; ++ic) {
      for (k = 0; k < *ndim; ++k) {
	u[k] = wt[*ncol][ic] * ups[jp1][k];
	uold[k] = wt[*ncol][ic] * uoldps[jp1][k];
	for (l = 0; l < *ncol; ++l) {
	  l1 = l * *ndim + k;
	  u[k] += wt[l][ic] * ups[j][l1];
	  uold[k] += wt[l][ic] * uoldps[j][l1];
	}
      }
      /*     ** Time evolution computations (parabolic systems) */
      if (*ips == 14 || *ips == 16) {
	rap->tivp = rlold[0];
      }
      for (i = 0; i < NPARX; ++i) {
	prm[i] = par[i];
      }
      (*funi)(iap, rap, *ndim, u, uold, icp, prm, 2, f, 
	      dfdu, dfdp);
      ic1 = ic * *ndim;
      for (i = 0; i < *ndim; ++i) {
	fa[ic1 + i][jj] = f[i] - wploc[*ncol][ic] * ups[jp1][i];
	for (k = 0; k < *ncol; ++k) {
	  k1 = k * *ndim + i;
	  fa[ic1 + i][jj] -= wploc[k][ic] * ups[j][k1];
	}
      }
      /* L1: */
    }
    /* L2: */
  }

  /*     Generate FC : */

/*     Boundary conditions : */

  if (*nbc > 0) {
    for (i = 0; i < *ndim; ++i) {
      ubc0[i] = ups[0][i];
      ubc1[i] = ups[*ntst][i];
    }
    (*bcni)(iap, rap, *ndim, par, icp, *nbc, ubc0, ubc1, 
	    fbc, 2, dbc);
    for (i = 0; i < *nbc; ++i) {
      fc[i] = -fbc[i];
    }
    /*       Save difference : */
    for (j = 0; j < *ntst + 1; ++j) {
      for (i = 0; i < *nra; ++i) {
	dups[j][i] = ups[j][i] - uoldps[j][i];
      }
    }
  }

  /*     Integral constraints : */
  if (*nint > 0) {
    for (jj = 0; jj < *na; ++jj) {
      j = jj + mpart;
      jp1 = j + 1;
      for (k = 0; k < ncp1; ++k) {
	for (i = 0; i < *ndim; ++i) {
	  i1 = k * *ndim + i;
	  j1 = j;
	  if (k + 1 == ncp1) {
	    i1 = i;
	  }
	  if (k + 1 == ncp1) {
	    j1 = jp1;
	  }
	  uic[i] = ups[j1][i1];
	  uio[i] = uoldps[j1][i1];
	  uid[i] = udotps[j1][i1];
	  uip[i] = upoldp[j1][i1];
	}
	(*icni)(iap, rap, *ndim, par, icp, *nint, uic, 
		uio, uid, uip, ficd, 2, dicd);
	for (m = 0; m < *nint; ++m) {
	  fc[*nbc + m] -= dtm[j] * wi[k] * ficd[m];
	}
      }
    }
  }

  /*     Pseudo-arclength equation : */
#ifndef MANIFOLD
  rlsum = 0.;
  for (i = 0; i < *ncb; ++i) {
    rlsum += thl[icp[i]] * (rlcur[i] - rlold[i]) * rldot[i];
  }

  fc[-1 + *nrc] = *rds - rinpr(iap, ndim, ndxloc, udotps, 
			       dups, dtm, thu) - rlsum;
#else
  udotps_off=(iap->ntst + 1)*(iap->ndim * iap->ncol);
  for(m=0;m<*nalc;m++){
    rlsum = 0.;
    for (i = 0; i < *ncb; ++i) {
      rlsum += thl[icp[i]] * (rlcur[i] - rlold[i]) * rldot[i+m*NPARX];
    }

    fc[-1 + *nrc*m] = rds[m] - rinpr(iap, ndim, ndxloc, &(udotps[udotps_off*m]), 
			         dups, dtm, thu) - rlsum;
   }
#endif

  FREE(dicd );
  FREE(ficd );
  FREE(dfdp );
  FREE(dfdu );
  FREE(uold );
  FREE(f    );
  FREE(u    );
  FREE_DMATRIX(wploc);
  FREE(wi   );
  FREE_DMATRIX(wp);
  FREE_DMATRIX(wt);
  FREE(dbc  );
  FREE(fbc  );
  FREE(uic  );
  FREE(uio  );
  FREE(prm  );
  FREE(uid  );
  FREE(uip  );
  FREE(ubc0 );
  FREE(ubc1 );

  return 0;
} /* setrhs_ */


/*     ---------- ---- */
/* Subroutine */ int 
brbd(doublereal ***a, doublereal ***b, doublereal ***c, doublereal **d, doublereal **fa, doublereal *fc, doublereal **p0, doublereal **p1, integer *ifst, integer *idb, integer *nllv, doublereal *det, integer *nov, integer *na, integer *nbc, integer *nra, integer *nca, integer *ncb, integer *nrc, integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, doublereal **faa, doublereal ***ca1, doublereal ***s1, doublereal ***s2, integer *icf11, integer *ipr, integer *icf1, integer *icf2, integer *irf, integer *icf)
{
  doublereal **e;
  doublereal *fcc;
  doublereal *sol1,*sol2,*sol3;

  //DREW WUZ HERE: Condition number for jacobian
  integer i, j, k, ir, ic;
  
  integer cond = 0;
  
  doublereal *svde, *svds, svdu[1], *svdv;

  integer svdinf;
  integer zerorc;
  doublereal *svdwrk, *rwork;

  integer tmp = 1;
  doublereal tmp_tol = 1.0E-16;
  
  //doublereal timestart;
  //clock_t compcond;
  //DREW WUZ GONE

  e = DMATRIX(*nov + *nrc, *nov + *nrc);
  fcc = (doublereal *)MALLOC(sizeof(doublereal)*((*nov + *nrc) + (2*(*nov)*(*nov))+1));

  sol1 = (doublereal *)MALLOC(sizeof(doublereal)*(*nov)*(*na + 1));
  sol2 = (doublereal *)MALLOC(sizeof(doublereal)*(*nov)*(*na + 1));
  sol3 = (doublereal *)MALLOC(sizeof(doublereal)*(*nov)*(*na + 1));
  
  /* Local */

  /* Parameter adjustments */
  /*--icf;*/
  /*--irf;*/
  /*--icf2;*/
  /*--icf1;*/
  /*--ipr;*/
  /*--icf11;*/
  /*--s2;*/
  /*--s1;*/
  /*--ca1;*/
  /*--faa;*/
  /*--cc;*/
  /*--bb;*/
  /*--a2;*/
  /*--a1;*/
  /*--p1;*/
  /*--p0;*/
  /*--fc;*/
  /*--fa;*/
  /*--d;*/
  /*--c;*/
  /*--b;*/
  /*--a;*/

    
  if (*idb > 4 && *iam == 0) {
    print1(nov, na, nra, nca, ncb, nrc, a, b, c, d, &
    	   fa[0], fc);
  }
  if (*ifst == 1) {
#ifdef USAGE
    struct rusage *conpar_usage;
    usage_start(&conpar_usage);
#endif
      // DREW WUZ HERE: Condition number of submatrices in jacobian
      /* if (cond) {
          integer rdim = gData->iap.ncol*gData->iap.ndim;
          integer cdim = rdim + gData->iap.ndim;
          double sum = 0.0;
          
          rwork = (doublereal *)MALLOC(sizeof(doublereal)*rdim*cdim);
          svde = (doublereal *)MALLOC(sizeof(doublereal)*cdim);
          svds = (doublereal *)MALLOC(sizeof(doublereal)*(rdim+1));
          svdv = (doublereal *)MALLOC(sizeof(doublereal)*cdim*cdim);
          svdwrk = (doublereal *)MALLOC(sizeof(doublereal)*cdim);
          
          for (i=0; i<rdim; i++) {
              for (j=0; j<cdim; j++) {
                  rwork[i*cdim + j] = 0.0;
              }
          }
          
          for (k=0; k<gData->iap.ntst; k++) {
              for (i=0; i<rdim; i++) {
                  for (j=0; j<cdim; j++) {
                      rwork[i*cdim+j] = a[k][i][j];
                  }
              }              
              ezsvd(rwork, &rdim, &rdim, &cdim, svds, svde, svdu, &tmp, 
                svdv, &rdim, svdwrk, &tmp, &svdinf, &tmp_tol);
              sum += svds[0]/svds[rdim-1];
          }
          fprintf(stdout,"avg. cond = %lf\n", sum/k);
          fflush(stdout);
          
          FREE(rwork);
          FREE(svde);
          FREE(svds);
          FREE(svdv);
          FREE(svdwrk);
      } */
      // DREW WUZ HERE: Condition number for jacobian
      if (cond) {
          fprintf(stdout,"Creating jacobian...");
          //timestart = clock();
          //compcond = 0;
          integer mdim = gData->iap.ntst*gData->iap.ncol*gData->iap.ndim + gData->iap.nbc + gData->iap.nint + 1;
        
          rwork = (doublereal *)MALLOC(sizeof(doublereal)*mdim*mdim);
          svde = (doublereal *)MALLOC(sizeof(doublereal)*mdim);
          svds = (doublereal *)MALLOC(sizeof(doublereal)*(mdim+1));
          svdv = (doublereal *)MALLOC(sizeof(doublereal)*mdim*mdim);
          svdwrk = (doublereal *)MALLOC(sizeof(doublereal)*mdim);
    
          for (i=0; i<mdim; i++) {
              for (j=0; j<mdim; j++) {
                  rwork[i*mdim+j] = 0.0;
              }
          }
          for (k=0; k<gData->iap.ntst; k++) {
              ir = k*gData->iap.ncol*gData->iap.ndim;
              /* if ((clock()-timestart)/CLK_TCK >= 60/gData->iap.ntst) {
                  compcond = 0;
                  break;
              } */
              for (i=0; i<gData->iap.ncol*gData->iap.ndim; i++) {
                  ic = k*gData->iap.ncol*gData->iap.ndim;
                  for (j=0; j<(gData->iap.ncol+1)*gData->iap.ndim; j++) {
                      // Store a
                      rwork[(i+ir)*mdim + j+ic] = a[k][i][j];
                  }                  
                  // Store b
                  rwork[(i+ir)*mdim + mdim-2] = b[k][i][0];
                  rwork[(i+ir)*mdim + mdim-1] = b[k][i][1];                  
              }
              // Store c
              for (i=gData->iap.nbc + gData->iap.nint + 1; i>0; i--) {
                  ic = k*gData->iap.ncol*gData->iap.ndim;
                  for (j=0; j<(gData->iap.ncol+1)*gData->iap.ndim; j++) {
                      rwork[(mdim-i)*mdim + j+ic] += c[k][gData->iap.nbc + gData->iap.nint + 1 - i][j];
                  }
              }
          }
          // Store d
          for (i=gData->iap.nbc + gData->iap.nint + 1; i>0; i--) {
              rwork[(mdim-i)*mdim + mdim-2] = d[gData->iap.nbc + gData->iap.nint + 1 - i][0];
              rwork[(mdim-i)*mdim + mdim-1] = d[gData->iap.nbc + gData->iap.nint + 1 - i][1];
          }
          fprintf(stdout,"done!\n");
          
          // Check matrix for row/column of zeroes
          /* for (i=0; i<mdim; i++) {
              zerorc = 1;
              j = 0;
              while (zerorc && j<mdim) {
                  if (abs(rwork[i+mdim*j]) > 1e-10)
                      zerorc = 0;
                  j++;
              }
              if (zerorc) {
                  fprintf(stdout, "OOPS!  Zero row (i=%d)!\n", i);
              }
          } */
          
          if (1) {
              fprintf(stdout,"Checking condition number of jacobian...");
              fflush(stdout);
              ezsvd(rwork, &mdim, &mdim, &mdim, svds, svde, svdu, &tmp, 
                svdv, &mdim, svdwrk, &tmp, &svdinf, &tmp_tol);
              fprintf(stdout,"done!\n");
              fprintf(stdout,"  COND = %lf\n", svds[0]/svds[mdim-1]);
              fflush(stdout);
          } else {
              fprintf(stdout,"Passed time.\n");
              fflush(stdout);
          }
          FREE(rwork);
          FREE(svde);
          FREE(svds);
          FREE(svdv);
          FREE(svdwrk);
      }
      // DREW WUZ GONE
  
    conpar(nov, na, nra, nca, a, ncb, b, nbc, nrc, c, d, irf, icf);
#ifdef USAGE
    usage_end(conpar_usage,"all of conpar");
#endif
    copycp(*na, *nov, *nra, *nca, a, *ncb, b, *nrc, c, 
	   a1, a2, bb, cc, irf);
       
  }

  if (*nllv == 0) {
    conrhs(nov, na, nra, nca, a, nbc, nrc, c, fa, fc, 
	   irf, icf, iam);
    cpyrhs(*na, *nov, *nra, faa, fa, irf);
  } else {
#ifdef RANDY_FIX
    /* The faa array needs to be intialized as well, since it 
       it used in the dimrge_ rountine to print stuff out,
       and in the bcksub_ routine for actual computations! */
    {
      integer k;
      for(k=0;k<((*nov) * (*na + 1));k++)
	faa[k]=0.0;
    }
    setzero(fa, fc, na, nra, nrc);
#else
    setzero(fa, fc, na, nra, nrc);
    cpyrhs(*na, *nov, *nra, faa, fa, irf);
#endif
  }

  if (*ifst == 1) {
#ifdef USAGE
    struct rusage *reduce_usage;
    usage_start(&reduce_usage);
#endif
    reduce(iam, kwt, par, a1, a2, bb, cc, d, na, 
	   nov, ncb, nrc, s1, s2, ca1, icf1, icf2, 
	   icf11, ipr, nbc);
#ifdef USAGE
    usage_end(reduce_usage,"all of reduce");
#endif
  }

  if (*nllv == 0) {
    redrhs(iam, kwt, par, a1, a2, cc, faa, fc, na, 
	   nov, ncb, nrc, ca1, icf1, icf2, icf11, ipr,nbc);
  }

  dimrge(iam, kwt, par, e, cc, d, fc, ifst, na, 
     nrc, nov, ncb, idb, nllv, fcc, p0, p1, det, s1, a2,
	 faa, bb);

  bcksub(iam, kwt, par, s1, s2, a2, bb, faa, fc, 
	 fcc, sol1, sol2, sol3, na, nov, ncb, icf2);

  infpar(iam, par, a, b, fa, sol1, sol2, fc, na, nov, nra, 
	 nca, ncb, irf, icf);

  FREE_DMATRIX(e);
  FREE(fcc);
  FREE(sol1);
  FREE(sol2);
  FREE(sol3);
  return 0;
} /* brbd_ */


/*     ---------- ------- */
/* Subroutine */ int 
setzero(doublereal **fa, doublereal *fc, integer *na, integer *nra, integer *nrc)
{
    /* Local variables */
  integer i, j;

    /* Parameter adjustments */
    /*--fc;*/
    
  for (i = 0; i < *na; ++i) {
    for (j = 0; j < *nra; ++j) {
      fa[j][i] = 0.;
    }
  }

  for (i = 0; i < *nrc; ++i) {
    fc[i] = 0.;
  }

  return 0;
} /* setzero_ */


/*     ---------- ------ */
/* Subroutine */ int 
conrhs(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *nbc, integer *nrc, doublereal ***c, doublereal **fa, doublereal *fc, integer *irf, integer *icf, integer *iam)
{
  /* System generated locals */
  integer icf_dim1, irf_dim1;

    /* Local variables */
  integer nbcp1, i, icfic, irfir, m1, m2, ic, ir, irfirp, ir1, nex,
    irp;


    /* Parameter adjustments */
    /*--fc;*/
  irf_dim1 = *nra;
  icf_dim1 = *nca;
    
  nex = *nca - (*nov * 2);
  if (nex == 0) {
    return 0;
  }

  /* Condensation of right hand side. */

  nbcp1 = *nbc + 1;
  m1 = *nov + 1;
  m2 = *nov + nex;

  for (i = 0; i < *na; ++i) {
    for (ic = *nov; ic < m2; ++ic) {
      ir1 = ic - *nov + 1;
      irp = ir1 - 1;
      irfirp = ARRAY2D(irf, irp, i);
      icfic = ARRAY2D(icf, ic, i);
      for (ir = ir1; ir < *nra; ++ir) {
	irfir = ARRAY2D(irf, ir, i);
	if (a[i][irfir - 1][icfic - 1] != (double)0.) {
	  fa[irfir - 1][i] -= a[i][irfir - 1][icfic - 1] * fa[irfirp - 1][i];
	}
      }
      for (ir = *nbc; ir < *nrc; ++ir) {
	if (c[i][ir][icfic - 1] != (double)0.) {
	  fc[ir] -= c[i][ir][icfic - 1] * fa[irfirp - 1][i];
	}
      }
    }
  }

  return 0;
} /* conrhs_ */


/*     ---------- ------ */
/* Subroutine */ int 
copycp(integer na, integer nov, integer nra, integer nca, doublereal ***a, integer ncb, doublereal ***b, integer nrc, doublereal ***c, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, integer *irf)
{
  /* System generated locals */
  integer irf_dim1 = nra;  

  /* Local variables */
  integer i, j, k, irfir, ic, ir, ic1, nap1;
  
/* Local */

/* Copies the condensed sytem generated by CONPAR into workspace. */

  nap1 = na + 1;
  for (i = 0; i < na; ++i) {
    for (ir = 0; ir < nov; ++ir) {
      irfir = ARRAY2D(irf, nra - nov + ir, i);
      for (ic = 0; ic < nov; ++ic) {
	ic1 = nca - nov + ic;
	a1[i][ir][ic] = a[i][irfir - 1][ic];
	a2[i][ir][ic] = a[i][irfir - 1][ic1];
      }
      for (ic = 0; ic < ncb; ++ic) {
	bb[i][ir][ic] = b[i][irfir - 1][ic];
      }
    }
  }

  for (i = 0; i < nap1; ++i) {
    for (ir = 0; ir < nrc; ++ir) {
      for (ic = 0; ic < nov; ++ic) {
	if (i == 0) {
	  cc[i][ir][ic] = c[i][ir][ic];
	} else if (i + 1 == nap1) {
	  cc[i][ir][ic] = c[i - 1][ir][nra + ic];
	} else {
	  cc[i][ir][ic] = c[i][ir][ic] + c[i - 1][ir][nra + ic];
	}
      }
    }
  }
  
  // DREW WUZ HERE
  // Store jacobians along cycle (faster if used copy method)
  if (gData->sflow) {
      for (i=0; i<na; i++) {
          for (j=0; j<gData->iap.ndim; j++) {
              for (k=0; k<gData->iap.ndim; k++) {
                  gData->a1[i][j][k] = a1[i][j][k];
                  gData->a2[i][j][k] = a2[i][j][k];
              }
          }
      }
  }
  
  return 0;
} /* copycp_ */


/*     ---------- ------ */
/* Subroutine */ int 
cpyrhs(integer na, integer nov, integer nra, doublereal **faa, doublereal **fa, integer *irf)
{
  /* System generated locals */
  integer irf_dim1 = nra;

  /* Local variables */
  integer i, irfir, ir;

/*     **Copy the RHS */
    
  for (i = 0; i < na; ++i) {
    for (ir = 0; ir < nov; ++ir) {
      irfir = ARRAY2D(irf, nra - nov + ir, i);
      faa[ir][i] = fa[irfir - 1][i];
    }
  }

  return 0;
} /* cpyrhs_ */

/*     ---------- ------ */
/* Subroutine */ int 
redrhs(integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***cc, doublereal **faa, doublereal *fc, integer *na, integer *nov, integer *ncb, integer *nrc, doublereal ***ca1, integer *icf1, integer *icf2, integer *icf11, integer *ipr, integer *nbc)
{
  /* System generated locals */
  integer icf1_dim1, icf2_dim1, icf11_dim1, ipr_dim1;

  /* Local variables */
  integer niam, nlev;
  real xkwt;
  integer nbcp1, ipiv1, ipiv2, i;

  integer i1, i2, k1, l1, ic, ir;
  doublereal rm;
  logical master[KREDO];
  integer myleft[KREDO];
  logical worker[KREDO];
  doublereal buf[2];
  integer ism[KREDO], irm[KREDO];
  doublereal tmp;
  logical notsend;
  integer nap1, nam1, myright[KREDO], icp1;

  
    /* Parameter adjustments */
    /*--fc;*/
  ipr_dim1 = *nov;
  icf11_dim1 = *nov;
  icf2_dim1 = *nov;
  icf1_dim1 = *nov;

    
  nbcp1 = *nbc + 1;
  nap1 = *na + 1;
  nam1 = *na - 1;
  xkwt = (real) (*kwt);
  {
    real tmp = r_lg10(xkwt) / r_lg10(2.0);
    nlev = i_nint(&tmp);
  }
  notsend = TRUE_;

/* At each recursive level determine the master node (holding the pivot */
/* row after swapping), which will send the pivot row to the worker node 
*/
/* at distance 2**(K-1) from the master. Here K is the recursion level. */

  if (*par) {
    for (i = 0; i < nlev; ++i) {
      master[i] = FALSE_;
      worker[i] = FALSE_;
      k1 = pow_ii(2, i);
      niam = *iam / k1;
      if (notsend) {
	if (niam % 2 == 0) {
	  master[i] = TRUE_;
	  notsend = FALSE_;
	  ism[i] = (i + 1) + *iam + 10000;
	  irm[i] = ism[i] + k1;
	  myright[i] = *iam + k1;
	} else {
	  worker[i] = TRUE_;
	  ism[i] = (i + 1) + *iam + 10000;
	  irm[i] = ism[i] - k1;
	  myleft[i] = *iam - k1;
	}
      }
    }
  }

  /* Reduce concurrently in each node */
  for (i1 = 0; i1 < nam1; ++i1) {
    i2 = i1 + 1;
    for (ic = 0; ic < *nov; ++ic) {
      icp1 = ic + 1;
      ipiv1 = ARRAY2D(ipr, ic, i1);
      if (ipiv1 <= *nov) {
	tmp = faa[ic][i1];
	faa[ic][i1] = faa[ipiv1 - 1][i1];
	faa[ipiv1 - 1][i1] = tmp;
      } else {
	l1 = (ipiv1 - *nov) - 1;
	tmp = faa[ic][i1];
	faa[ic][i1] = faa[l1][i2];
	faa[l1][i2] = tmp;
      }
      for (ir = icp1; ir < *nov; ++ir) {
	l1 = ARRAY2D(icf2, ic, i1) - 1;
	rm = a2[i1][ir][l1];
	faa[ir][i1] -= rm * faa[ic][i1];
      }
      for (ir = 0; ir < *nov; ++ir) {
	l1 = ARRAY2D(icf1, ic, i2) - 1;
	rm = a1[i2][ir][l1];
	faa[ir][i2] -= rm * faa[ic][i1];
      }
      for (ir = nbcp1 - 1; ir < *nrc; ++ir) {
	l1 = ARRAY2D(icf2, ic, i1) - 1;
	rm = cc[i2][ir][l1];
	fc[ir] -= rm * faa[ic][i1];
      }
    }
  }

  /* Inter-node reduction needs communication between nodes */
  if (*par) {
    for (i = 0; i < nlev; ++i) {
      for (ic = 0; ic < *nov; ++ic) {
	icp1 = ic + 1;
	if (master[i]) {
	  ipiv1 = ARRAY2D(ipr, ic, (*na - 1));
	  if (ipiv1 <= *nov) {
	    buf[0] = faa[ipiv1 - 1][*na - 1];
	    faa[ipiv1 - 1][*na] = faa[ic][*na - 1];
	    faa[ic][*na - 1] = buf[0];
	    buf[1] = -1.;
	    csend();
	  } else {
	    buf[0] = faa[ic][*na - 1];
	    buf[1] = (doublereal) (ARRAY2D(ipr, ic, (*na - 1)) - *nov);
	    csend();
	    crecv();
	  }

	  for (ir = icp1; ir < *nov; ++ir) {
	    l1 = ARRAY2D(icf2, ic, (*na - 1)) - 1;
	    rm = a2[*na - 1][ir][l1];
	    faa[ir][*na - 1] -= rm * faa[ic][*na - 1];
	  }
	  for (ir = nbcp1 - 1; ir < *nrc; ++ir) {
	    l1 = ARRAY2D(icf2, ic, (*na - 1)) - 1;
	    rm = cc[nap1 - 1][ir][l1];
	    fc[ir] -= rm * faa[ic][*na - 1];
	  }
	}

	if (worker[i]) {
	  crecv();
	  ipiv2 = i_dnnt(&buf[1]);
	  if (ipiv2 < 0) {
	    tmp = buf[0];
	  } else {
	    tmp = faa[ipiv2 - 1][*na - 1];
	    faa[ipiv2 - 1][*na - 1] = buf[0];
	    csend();
	  }

	  for (ir = 0; ir < *nov; ++ir) {
	    l1 = ARRAY2D(icf11, ic, i) - 1;
	    rm = ca1[i][ir][l1];
	    faa[ir][*na - 1] -= rm * tmp;
	  }
	}
      }
      /*           **Synchronization at each recursion level among all n
		   odes */


    }

    l1 = *nrc - *nbc;
    gdsum();

  }

  return 0;
} /* redrhs_ */


/*     ---------- ------ */
/* Subroutine */ int 
dimrge(integer *iam, integer *kwt, logical *par, doublereal **e, doublereal ***cc, doublereal **d, doublereal *fc, integer *ifst, integer *na, integer *nrc, integer *nov, integer *ncb, integer *idb, integer *nllv, doublereal *fcc, doublereal **p0, doublereal **p1, doublereal *det, doublereal ***s, doublereal ***a2, doublereal **faa, doublereal ***bb)
{

  /* Local variables */

  integer i, j, k;

  integer novpi, novpj, k1, k2;

  integer novpj2, kc, kr, ncrloc, msglen1, msglen2, nap1;

  //DREW WUZ HERE: Condition number for jacobian
  integer ir, ic;
  
  integer cond = 0;
  
  doublereal *svde, *svds, svdu[1], *svdv;

  integer svdinf;
  integer zerorc;
  doublereal *svdwrk, *rwork;

  integer tmp = 1;
  doublereal tmp_tol = 1.0E-16;
  
  //doublereal timestart;
  //clock_t compcond;
  //DREW WUZ GONE

  double *xe;
  xe = (doublereal *)MALLOC(sizeof(doublereal)*(*nov + *nrc));


  /* Parameter adjustments */
  /*--fc;*/
  /*--xe;*/
  /*--fcc;*/
    
  nap1 = *na + 1;
  msglen1 = (*nrc * 8) * *nov;
  /* Computing 2nd power */
  msglen2 = (*nov + *nrc + ((*nov * *nov) * 2) + 1) * 8;
  ncrloc = *nrc + *nov;

  /* Send CC(1:NOV,1:NRC,1) in node 0 to node KWT-1 */

  if (*par) {
    if (*iam == 0) {
      csend();
    }
    if (*iam == *kwt - 1) {
      crecv();
    }
  }

  /* Copy */
  // DREW WUZ HERE:  NOTE: nov = ndim, na = ntst
  if (*iam == *kwt - 1) {
    for (i = 0; i < *nov; ++i) {
      for (j = 0; j < *nov; ++j) {
	novpj = *nov + j;
	e[i][j] = s[*na - 1][i][j];
	p0[j][i] = s[*na - 1][i][j];
	e[i][novpj] = a2[*na - 1][i][j];
	p1[j][i] = a2[*na - 1][i][j];
      }
      for (j = 0; j < *ncb; ++j) {
	novpj2 = (*nov * 2) + j;
	e[i][novpj2] = bb[*na - 1][i][j];
      }
    }

    for (i = 0; i < *nrc; ++i) {
      novpi = *nov + i;
      for (j = 0; j < *nov; ++j) {
	novpj = *nov + j;
	e[novpi][j] = cc[0][i][j];
	e[novpi][novpj] = cc[nap1 - 1][i][j];
      }
      for (j = 0; j < *ncb; ++j) {
	novpj2 = (*nov * 2) + j;
	e[novpi][novpj2] = d[i][j];
      }
    }

    for (i = 0; i < *nov; ++i) {
      xe[i] = faa[i][*na - 1];
    }

    for (i = 0; i < *nrc; ++i) {
      novpi = *nov + i;
      xe[novpi] = fc[i];
    }

    if (*idb >= 3) {
      fprintf(fp9," Residuals of reduced system:\n");	
	  
      fprintf(fp9," ");
      for (i = 0; i < ncrloc; ++i) {
	fprintf(fp9,"%11.3E",xe[i]);	
	if((i+ 1)%10==0)
	  fprintf(fp9,"\n ");
	    
      }
      fprintf(fp9,"\n");	
    }
    
    // DREW WUZ HERE
    if (cond) {
        rwork = (doublereal *)MALLOC(sizeof(doublereal)*ncrloc*ncrloc);
        svde = (doublereal *)MALLOC(sizeof(doublereal)*ncrloc);
        svds = (doublereal *)MALLOC(sizeof(doublereal)*(ncrloc+1));
        svdv = (doublereal *)MALLOC(sizeof(doublereal)*ncrloc*ncrloc);
        svdwrk = (doublereal *)MALLOC(sizeof(doublereal)*ncrloc);
        for (i=0; i<ncrloc; i++)
            for (j=0; j<ncrloc; j++)
                rwork[ncrloc*i+j]=e[i][j];
        ezsvd(rwork, &ncrloc, &ncrloc, &ncrloc, svds, svde, svdu, &tmp, 
          svdv, &ncrloc, svdwrk, &tmp, &svdinf, &tmp_tol);
        fprintf(stdout,"  COND = %lf\n", svds[0]/svds[ncrloc-1]);
        fflush(stdout);
        FREE(rwork);
        FREE(svde);
        FREE(svds);
        FREE(svdv);
        FREE(svdwrk);
    }
    // DREW WUZ GONE
    /* Reduced Jacobian consists of:
            [s1 a2 b]
            [c0 cn d]
    */
    
    if (*idb >= 4) {
    
      fprintf(fp9," Reduced Jacobian matrix:\n");	
	      
      for (i = 0; i < ncrloc; ++i) {
	int total_printed = 0;
	for (j = 0; j < ncrloc; ++j) {
	  if((total_printed != 0)&&(total_printed % 10 == 0))
	    fprintf(fp9,"\n");	
	  fprintf(fp9," %11.3E",e[i][j]);	
	  total_printed++;
	}
	fprintf(fp9,"\n");	
      }
    }

    /* Solve for FCC */
    if (*nllv == 0) {
      ge(ncrloc, ncrloc, *e, 1, 1, fcc, 1, xe, det);
    } else if (*nllv > 0) {
      nlvc(ncrloc, ncrloc, *nllv, e, fcc);
    } else {
      for (i = 0; i < ncrloc - 1; ++i) {
	xe[i] = 0.;
      }
      xe[-1 + ncrloc] = 1.;
      ge(ncrloc, ncrloc, *e, 1, 1, fcc, 1, xe, det);
    }
    if (*idb >= 4) {
      fprintf(fp9," Solution vector:\n");	
	  
      for (i = 0; i < ncrloc; ++i) {
	if((i!=0)&&(i%7==0))
	  fprintf(fp9,"\n");	
	fprintf(fp9," %11.3E",fcc[i]);	
      }
      fprintf(fp9,"\n");	
    }

    k1 = ncrloc;
    /* Computing 2nd power */
    k2 = k1 + (*nov) * (*nov);
    for (kr = 0; kr < *nov; ++kr) {
      for (kc = 0; kc < *nov; ++kc) {
	k = kr * *nov + kc;
	fcc[k1 + k] = p0[kc][kr];
	fcc[k2 + k] = p1[kc][kr];
      }
    }
    /* Computing 2nd power */
    fcc[ncrloc + ((*nov) * (*nov) * 2)] = *det;

  }

  /* Broadcast FCC from node KWT-1. The matrices P0 and P1 are */
  /* buffered in the tail of FCC so all nodes receive them. */
  if (*par) {
    if (*iam == *kwt - 1) {
      csend();
    } else {
      crecv();
    }
  }

  for (i = 0; i < *nrc; ++i) {
    fc[i] = fcc[*nov + i];
  }

  if (*iam < *kwt - 1) {
    k1 = ncrloc;
    /* Computing 2nd power */
    k2 = k1 + (*nov) * (*nov);
    for (kr = 1; kr <= *nov; ++kr) {
      for (kc = 1; kc <= *nov; ++kc) {
	k = kr * *nov + kc;
	p0[kc][kr] = fcc[k1 + k];
	p1[kc][kr] = fcc[k2 + k];
      }
    }
    /* Computing 2nd power */
    *det = fcc[ncrloc + ((*nov) * (*nov) * 2)];
  }
  /* free the memory*/
  /* Not the we have modified these parameter before, so
     we undo the modifications here and then free them. */
  /*xe \+= 1;*/
  FREE(xe);

  return 0;
} /* dimrge_ */


/*     ---------- ------ */
/* Subroutine */ int 
bcksub(integer *iam, integer *kwt, logical *par, doublereal ***s1, doublereal ***s2, doublereal ***a2, doublereal ***bb, doublereal **faa, doublereal *fc, doublereal *fcc, doublereal *sol1, doublereal *sol2, doublereal *sol3, integer *na, integer *nov, integer *ncb, integer *icf2)
{
  /* System generated locals */
  integer icf2_dim1, sol1_dim1, sol2_dim1, sol3_dim1;

    /* Local variables */
  integer niam, ibuf;
  logical even = FALSE_;
  integer nlev;
  logical hasright;
  doublereal xkwt;
  integer rmsgtype, smsgtype, i, k, l;

  integer nlist[2], itest, l1, l2;
  doublereal sm;
  integer msglen;

  logical master[KREDO];
  integer myleft, kp1;
  logical odd = FALSE_;
  integer ism, irm;
  logical hasleft, notsend;
  integer nam1, myright, nov2, nov3;
  double *buf=NULL;


    /* Parameter adjustments */
    /*--fc;*/
    /*--fcc;*/
  icf2_dim1 = *nov;
  sol3_dim1 = *nov;
  sol2_dim1 = *nov;
  sol1_dim1 = *nov;
    
  xkwt = (doublereal) (*kwt);
  {
    doublereal tmp = d_lg10(&xkwt) / r_lg10(2.0);
    nlev = i_dnnt(&tmp);
  }
  nov2 = *nov * 2;
  nov3 = *nov * 3;
  ibuf = (nov3 + 1) * 8;

  /* The backsubstitution in the reduction process is recursive. */
  notsend = TRUE_;

  /*At each recursion level determine the sender nodes (called MASTER here).
*/
  if (*par) {
    for (i = 0; i < nlev; ++i) {
      master[i] = FALSE_;
      niam = *iam / pow_ii(2, i);
      if (notsend) {
	if (niam % 2 == 0) {
	  master[i] = TRUE_;
	  notsend = FALSE_;
	}
      }
    }
  }

  if (*par) {

    /*Initialization for the master or sender node at the last recursion l
      evel.*/
    if (master[nlev - 1]) {
      for (l = 0; l < *nov; ++l) {
	ARRAY2D(sol1, l, (*na - 1)) = fcc[l];
	ARRAY2D(sol3, l, (*na - 1)) = fc[l];
      }
    }

    for (i = nlev - 1; i >= 0; --i) {
      if (master[i]) {
	ism = i + nlev + (*kwt * 4);
	irm = ism + 1;
	k = pow_ii(2, i - 1);
	/*              **Compute the ID of the receiving node */
	nlist[0] = *iam - k;
	nlist[1] = *iam + k;
	/*              **Receive solutions from previous level */
	if ((i + 1) < nlev) {
	  crecv();
	  niam = i_dnnt(&buf[nov3 + 1]);
	  if (*iam < niam) {
	    for (l = 0; l < *nov; ++l) {
	      ARRAY2D(sol1, l, (*na - 1)) = buf[l + 1];
	      ARRAY2D(sol3, l, (*na - 1)) = buf[*nov + l + 1];
	    }
	  } else {
	    for (l = 0; l < *nov; ++l) {
	      ARRAY2D(sol1, l, (*na - 1)) = buf[*nov + l + 1];
	      ARRAY2D(sol3, l, (*na - 1)) = buf[nov2 + l + 1];
	    }
	  }
	}
	/*              **Backsubstitute */
	for (k = *nov - 1; k >= 0; --k) {
	  kp1 = k + 1;
	  sm = 0.;
	  for (l = 0; l < *nov; ++l) {
	    sm += s1[*na - 1][k][l] * ARRAY2D(sol1, l, (*na - 1));
	    sm += s2[*na - 1][k][l] * ARRAY2D(sol3, l, (*na - 1));
	  }
	  for (l = 0; l < *ncb; ++l) {
	    sm += bb[*na - 1][k][l] * fc[*nov + l];
	  }
	  for (l = kp1; l < *nov; ++l) {
	    l1 = ARRAY2D(icf2, l, (*na - 1)) - 1;
	    sm += ARRAY2D(sol2, l1, (*na - 1)) * a2[*na - 1][k][l1];
	  }
	  l2 = ARRAY2D(icf2, k, (*na - 1)) - 1;
	  ARRAY2D(sol2, l2, (*na - 1)) = (faa[k][*na - 1] - sm) / a2[*na - 1][k][l2];
	}
	/*              **Send solutions to the next level */
	if (i + 1 > 1) {
	  for (l = 0; l < *nov; ++l) {
	    buf[l + 1] = ARRAY2D(sol1, l, (*na - 1));
	    buf[*nov + l + 1] = ARRAY2D(sol2, l, (*na - 1));
	    buf[nov2 + l + 1] = ARRAY2D(sol3, l, (*na - 1));
	  }
	  buf[nov3 + 1] = (doublereal) (*iam);
	  gsendx();
	}
      }
      /*           **Synchronization at each recursion level */


    }

    /* Define odd and even nodes */
    if (*iam % 2 == 0) {
      even = TRUE_;
    } else {
      odd = TRUE_;
    }

    /* Determine whether I have a right neighbor */
    if (*iam == *kwt - 1) {
      hasright = FALSE_;
    } else {
      hasright = TRUE_;
    }

    /* Determine whether I have a left neighbor */
    if (*iam == 0) {
      hasleft = FALSE_;
    } else {
      hasleft = TRUE_;
    }

    /* Define send message type */
    smsgtype = *iam + 1000;

    /* Define receive message type */
    rmsgtype = smsgtype - 1;

    /* Define my right neighbor */
    myleft = *iam - 1;
    myright = *iam + 1;
    msglen = *nov << 3;

    /* May only need odd sends to even */
    itest = 0;
    if (itest == 1) {
      if (odd && hasright) {
	csend();
      }
      if (even && hasleft) {
	crecv();
      }
    }

    /* Even nodes send and odd nodes receive */
    if (even && hasright) {
      csend();
    }
    if (odd && hasleft) {
      crecv();
    }

  } else {

    for (l = 0; l < *nov; ++l) {
      ARRAY2D(sol1, l, (*na - 1)) = fcc[l];
      ARRAY2D(sol2, l, (*na - 1)) = fc[l];
    }

  }

  if (*iam == *kwt - 1) {
    for (l = 0; l < *nov; ++l) {
      ARRAY2D(sol2, l, (*na - 1)) = fc[l];
    }
  }

  if (*na > 1) {
    for (l = 0; l < *nov; ++l) {
      ARRAY2D(sol1, l, (*na - 2)) = ARRAY2D(sol1, l, (*na - 1));
      ARRAY2D(sol3, l, (*na - 2)) = ARRAY2D(sol2, l, (*na - 1));
    }
  }

  /* Backsubstitution process; concurrently in each node. */
  nam1 = *na - 1;
  for (i = nam1 - 1; i >= 0; --i) {
    for (k = *nov - 1; k >= 0; --k) {
      sm = 0.;
      for (l = 0; l < *nov; ++l) {
	sm += ARRAY2D(sol1, l, i) * s1[i][k][l];
	sm += ARRAY2D(sol3, l, i) * s2[i][k][l];
      }
      for (l = 0; l < *ncb; ++l) {
	sm += fc[*nov + l] * bb[i][k][l];
      }
      for (l = k + 1; l < *nov; ++l) {
	l1 = ARRAY2D(icf2, l, i) - 1;
	sm += ARRAY2D(sol2, l1, i) * a2[i][k][l1];
      }
      l2 = ARRAY2D(icf2, k, i) - 1;
      ARRAY2D(sol2, l2, i) = (faa[k][i] - sm) / a2[i][k][l2];
    }
    for (l = 0; l < *nov; ++l) {
      ARRAY2D(sol1, l, (i + 1)) = ARRAY2D(sol2, l, i);
      if (i + 1 > 1) {
	ARRAY2D(sol3, l, (i - 1)) = ARRAY2D(sol2, l, i);
	ARRAY2D(sol1, l, (i - 1)) = ARRAY2D(sol1, l, i);
      }
    }
  }

  return 0;
} /* bcksub_ */


/*     ---------- ------ */
/* Subroutine */ int 
infpar(integer *iam, logical *par, doublereal ***a, doublereal ***b, doublereal **fa, doublereal *sol1, doublereal *sol2, doublereal *fc, integer *na, integer *nov, integer *nra, integer *nca, integer *ncb, integer *irf, integer *icf)
{
  /* System generated locals */
  integer irf_dim1, icf_dim1, sol1_dim1,
    sol2_dim1;

    /* Local variables */
  integer nram, icfj1, i, j;
  doublereal *x;
  integer nrapj, irfir, j1, novpj, icfnovpir, ir;
  doublereal sm;
  integer novpir, irp1;

  x = (doublereal *)MALLOC(sizeof(doublereal)*(*nra));


/* Determine the local varables by backsubstitition. */

    /* Parameter adjustments */
    /*--fc;*/
  sol2_dim1 = *nov;
  sol1_dim1 = *nov;
  irf_dim1 = *nra;
  icf_dim1 = *nca;
    
  nram = *nra - *nov;

/* Backsubstitution in the condensation of parameters; no communication. 
*/
  for (i = 0; i < *na; ++i) {
    for (ir = nram - 1; ir >= 0; --ir) {
      irp1 = ir + 1;
      sm = 0.;
      irfir = ARRAY2D(irf, ir, i) - 1;
      for (j = 0; j < *nov; ++j) {
	nrapj = *nra + j;
	sm += a[i][irfir][j] * ARRAY2D(sol1, j, i);
	sm += a[i][irfir][nrapj] * ARRAY2D(sol2, j, i);
      }
      for (j = 0; j < *ncb; ++j) {
	novpj = *nov + j;
	sm += b[i][irfir][j] * fc[novpj];
      }
      for (j = irp1; j < nram; ++j) {
	j1 = j + *nov;
	icfj1 = ARRAY2D(icf, j1, i) - 1;
	sm += a[i][irfir][icfj1] * x[icfj1];
      }
      novpir = *nov + ir;
      icfnovpir = ARRAY2D(icf, novpir, i) - 1;
      x[icfnovpir] = (fa[irfir][i] - sm) / a[i][irfir][icfnovpir];
    }
    /*        **Copy SOL1 and X into FA */
    for (j = 0; j < *nov; ++j) {
      fa[j][i] = ARRAY2D(sol1, j, i);
    }
    for (j = *nov; j < *nra; ++j) {
      fa[j][i] = x[j];
    }
  }
  FREE(x);

  return 0;
} /* infpar_ */


/*     ---------- --- */
/* Subroutine */ int 
rd0(integer *iam, integer *kwt, doublereal *d, integer *nrc)
{

  /* Local variables */
  integer niam;
  logical even[KREDO];
  doublereal xkwt;
  integer i, n;

  integer nredo, msglen, rmtype[KREDO], smtype[KREDO];
  logical odd[KREDO];

  doublereal *buf;

  logical notsend;
  integer myright[KREDO];

  buf = (doublereal *)MALLOC(sizeof(doublereal)*(*nrc));

/*     RECURSIVE DOUBLING PROCEDURE TO GET */
/*     THE GLOBAL SUM OF VECTORS FROM */
/*     EACH NODE. THE GLOBAL SUM IS ONLY AVAILABLE */
/*     IN THE LAST NODE */

/* Copying */
    /* Parameter adjustments */
    /*--d;*/

    
  xkwt = (doublereal) (*kwt);

  /* Determine the recursion level */
  {
    doublereal tmp = log(xkwt) / log((double)2.);
    nredo = i_dnnt(&tmp);
  }

/* At each recursion level determine the odd and even nodes */
  notsend = TRUE_;
  for (n = 0; n < nredo; ++n) {
    smtype[n] = n + 1000 + *iam + 1;
    rmtype[n] = smtype[n] - pow_ii(2, n);
    myright[n] = *iam + pow_ii(2, n);
    even[n] = FALSE_;
    odd[n] = FALSE_;
    niam = *iam / pow_ii(2, n);
    if (notsend) {
      if (niam % 2 == 0) {
	even[n] = TRUE_;
	notsend = FALSE_;
      } else {
	odd[n] = TRUE_;
      }
    }
  }

  niam = *nrc;
  msglen = niam * 8;
  for (n = 0; n < nredo; ++n) {
    /*        **Even nodes send and odd nodes receive from left to right 
     */
    if (even[n]) {
      csend();
    }
    if (odd[n]) {
      crecv();
      /*          ** Accumulate the partial sum in the current receiving
		  node */
      for (i = 0; i < niam; ++i) {
	d[i] += buf[i];
      }
    }
  }
  FREE(buf);
  return 0;
} /* rd0_ */

/*     ---------- ------ */
/* Subroutine */ int 
print1(integer *nov, integer *na, integer *nra, integer *nca, integer *ncb, integer *nrc, doublereal ***a, doublereal ***b, doublereal ***c, doublereal **d, doublereal **fa, doublereal *fc)
{
    

  /* Local variables */
  integer i, ic, ir;

  /* Parameter adjustments */
  /*--fc;*/
    
  fprintf(fp9,"AA , BB , FA (Full dimension) :\n");	
  /* should be 10.3f*/
  for (i = 0; i < *na; ++i) {
    fprintf(fp9,"I=%3ld\n",i + 1);
    for (ir = 0; ir < *nra; ++ir) {
      int total_written = 0;
      for (ic = 0; ic < *nca; ++ic) {
	if((total_written != 0) && (total_written%12 == 0))
	  fprintf(fp9,"\n");
	fprintf(fp9," %10.3E",a[i][ir][ic]);
	total_written++;
      }
      for (ic = 0; ic < *ncb; ++ic) {
	if((total_written != 0) && (total_written%12 == 0))
	  fprintf(fp9,"\n");
	fprintf(fp9," %10.3E",b[i][ir][ic]);
	total_written++;
      }
      if((total_written != 0) && (total_written%12 == 0))
	fprintf(fp9,"\n");
      fprintf(fp9," %10.3E",fa[ir][i]);	
      fprintf(fp9,"\n");	
    }
  }

  fprintf(fp9,"CC (Full dimension) :\n");	

  for (i = 0; i < *na; ++i) {
    fprintf(fp9,"I=%3ld\n",i + 1);	
    for (ir = 0; ir < *nrc; ++ir) {
      int total_written = 0;
      for (ic = 0; ic < *nca; ++ic) {
	if((total_written != 0) && (total_written%12 == 0))
	  fprintf(fp9,"\n");
	fprintf(fp9," %10.3E",c[i][ir][ic]);	
	total_written++;
      }
      fprintf(fp9,"\n");	
    }
  }

  fprintf(fp9,"DD , FC\n");	

  for (ir = 0; ir < *nrc; ++ir) {
    int total_written = 0;
    for (ic = 0; ic < *ncb; ++ic) {
      if((total_written != 0) && (total_written%12 == 0))
	fprintf(fp9,"\n");
      fprintf(fp9," %10.3E",d[ir][ic]);
      total_written++;
    }
    fprintf(fp9," %10.3E\n",fc[ir]);	
  }


  return 0;
} /* print1_ */


/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*         Dummy Routines for the Sequential Version */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
integer 
mynode(void)
{
  integer ret_val;
  ret_val = 0;
  return ret_val;
}

integer 
numnodes(void)
{
  integer ret_val;
  ret_val = 1;
  return ret_val;
}


/* Subroutine */ int 
gsync(void)
{
  return 0;
} /* gsync_ */

doublereal 
dclock(void)
{
  real ret_val;

  ret_val = (double)0.;
  return ret_val;
} 


/* Subroutine */ int 
csend(void)
{
  return 0;
} /* csend_ */


/* Subroutine */ int 
crecv(void)
{
  return 0;
} /* crecv_ */


/* Subroutine */ int 
gdsum(void)
{
  return 0;
} /* gdsum_ */


/* Subroutine */ int 
gsendx(void)
{
  return 0;
} /* gsendx_ */


/* Subroutine */ int 
gcol(void)
{
  return 0;
} /* gcol_ */


/* Subroutine */ int 
led(void)
{
  return 0;
} /* led_ */


/* Subroutine */ int 
setiomode(void)
{
  return 0;
} /* setiomode_ */






































