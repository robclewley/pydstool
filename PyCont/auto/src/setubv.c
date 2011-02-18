#include "auto_f2c.h"
#include "auto_c.h"
#include "auto_types.h"


#ifdef PTHREADS
/*The parallel version of make_fa is only tested on Pthreads
  This will probably work on the MPI version, but I want to keep
  it here until until I get a chance to test it.*/
#define PTHREADS_PARALLEL_FA

/*  There are not needed anymore, but I am going to keep
    them around for a bit until I am sure that everything
    works without the mutexs. 
    The homecont stuff doesn NOT currently worked multithreaded
    since it has several global variables that need to be
    gotten rid of.
*/
/*
  #define PTHREADS_USE_FUNI_MUTEX
  #define PTHREADS_USE_BCNI_MUTEX
  #define PTHREADS_USE_ICNI_MUTEX
*/
pthread_mutex_t mutex_for_funi = PTHREAD_MUTEX_INITIALIZER;
#endif

void *setubv_make_aa_bb_cc(void * arg)
{  
  /* System generated locals */
  integer dbc_dim1, dicd_dim1, dfdu_dim1, dfdp_dim1;
  
  /* Local variables */
  integer i, j, k, l, m;
  integer k1, l1;
  integer i1,j1;

  integer ib, ic, jj;
  doublereal dt;  
  integer ib1, ic1;
  integer jp1;
  doublereal ddt;
#ifdef MANIFOLD
  integer udotps_off;
#endif

  setubv_parallel_arglist *larg =  (setubv_parallel_arglist *)arg;

  doublereal *dicd, *ficd, *dfdp, *dfdu, *uold;
  doublereal *f;
  doublereal *u, **wploc;
  doublereal *dbc, *fbc, *uic, *uio, *prm, *uid, *uip, *ubc0, *ubc1;
  
  doublereal **ups = larg->ups;
  doublereal **upoldp = larg->upoldp;
  doublereal **udotps = larg->udotps;
  doublereal **uoldps = larg->uoldps;

  doublereal ***aa = larg->aa;
  doublereal ***bb = larg->bb;
  doublereal ***cc = larg->cc;

  doublereal **wp = larg->wp;
  doublereal **wt = larg->wt;
#ifdef USAGE
  struct rusage *setubv_make_aa_bb_cc_usage,*fa_usage;
  usage_start(&setubv_make_aa_bb_cc_usage);
#endif



  if (larg->nint > 0) {
      dicd = (doublereal *)MALLOC(sizeof(doublereal)*(larg->nint)*(larg->ndim + NPARX));
      ficd = (doublereal *)MALLOC(sizeof(doublereal)*(larg->nint));
  }
  else
      ficd = dicd = NULL;
  
  dfdp = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim)*NPARX);
  dfdu = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim)*(larg->ndim));
  uold = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  f    = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  u    = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  wploc= DMATRIX(larg->ncol+1, larg->ncol);
  dbc  = (doublereal *)MALLOC(sizeof(doublereal)*(larg->nbc)*(2*larg->ndim + NPARX));
  fbc  = (doublereal *)MALLOC(sizeof(doublereal)*(larg->nbc));
  uic  = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  uio  = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  prm  = (doublereal *)MALLOC(sizeof(doublereal)*NPARX);
  uid  = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  uip  = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  ubc0 = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));
  ubc1 = (doublereal *)MALLOC(sizeof(doublereal)*(larg->ndim));

  dicd_dim1 = larg->nint;
  dbc_dim1 = larg->nbc;
  dfdu_dim1 = larg->ndim;
  dfdp_dim1 = larg->ndim;

  /* Generate AA and BB: */
  
  /*      Partition the mesh intervals */
  /*jj will be replaced with loop_start and loop_end*/
  for (jj = larg->loop_start; jj < larg->loop_end; ++jj) {
    j = jj;
    jp1 = j + 1;
    dt = larg->dtm[j];
    ddt = 1. / dt;
    for (ic = 0; ic < larg->ncol; ++ic) {
      for (ib = 0; ib < larg->ncol + 1; ++ib) {
	wploc[ib][ic] = ddt * wp[ib][ic];
      }
    }
    /*this loop uses the loop_offset variable since up and uoldps
      and sent by the MPI version in their entirety, but
      loop_start and loop_end have been shifted.  The loop_offset
      variable contains the original value of loop_start and removes
      the shift*/
    for (ic = 0; ic < larg->ncol; ++ic) {
      for (k = 0; k < larg->ndim; ++k) {
	u[k] = wt[larg->ncol][ic] * ups[jp1 + larg->loop_offset][k];
	uold[k] = wt[larg->ncol][ic] * uoldps[jp1 + larg->loop_offset][k];
	for (l = 0; l < larg->ncol; ++l) {
	  l1 = l * larg->ndim + k;
	  u[k] += wt[l][ic] * ups[j + larg->loop_offset][l1];
	  uold[k] += wt[l][ic] * uoldps[j + larg->loop_offset][l1];
	}
      }

      for (i = 0; i < NPARX; ++i) {
	prm[i] = larg->par[i];
      }
      /*  
	  Ok this is a little wierd, so hold tight.  This function
	  is actually a pointer to a wrapper function, which eventually
	  calls the user defined func_.  Which wrapper is used
	  depends on what kind of problem it is.  The need for
	  the mutex is because some of these wrappers use a common
	  block for temporary storage 
	  NOTE!!!:  The icni and bcni wrappers do the same thing,
	  so if they ever get parallelized they need to be
	  checked as well.
      */
#ifdef PTHREADS_USE_FUNI_MUTEX      
#ifdef PTHREADS
      pthread_mutex_lock(&mutex_for_funi);
#endif
#endif
      (*(larg->funi))(larg->iap, larg->rap, larg->ndim, u, uold, larg->icp, prm, 2, f, dfdu, dfdp);
#ifdef PTHREADS_USE_FUNI_MUTEX      
#ifdef PTHREADS
      pthread_mutex_unlock(&mutex_for_funi);
#endif
#endif
      ic1 = ic * (larg->ndim);
      for (ib = 0; ib < larg->ncol + 1; ++ib) {
	double wt_tmp=wt[ib][ic];
	double wploc_tmp=wploc[ib][ic];
	ib1 = ib * larg->ndim;
	for (i = 0; i < larg->ndim; ++i) {
	  aa[jj][ic1 + i][ib1 + i] = wploc_tmp;
	  for (k = 0; k < larg->ndim; ++k) {
	    aa[jj][ic1 + i][ib1 + k] -= wt_tmp * ARRAY2D(dfdu, i, k);
	  }
	}
      }
      for (i = 0; i < larg->ndim; ++i) {
	for (k = 0; k < larg->ncb; ++k) {
	  bb[jj][ic1 + i][k] = -ARRAY2D(dfdp, i, larg->icp[k]);
	}
      }
    }
  
  }

  /*     Generate CC : */
  
  /*     Boundary conditions : */
  if (larg->nbc > 0) {
    for (i = 0; i < larg->ndim; ++i) {
      ubc0[i] = ups[0][i];
      ubc1[i] = ups[larg->na][i];
    }
    
#ifdef PTHREADS_USE_BCNI_MUTEX      
#ifdef PTHREADS
    pthread_mutex_lock(&mutex_for_funi);
#endif
#endif
    (*(larg->bcni))(larg->iap, larg->rap, larg->ndim, larg->par, 
	    larg->icp, larg->nbc, ubc0, ubc1, fbc, 2, dbc);
#ifdef PTHREADS_USE_BCNI_MUTEX      
#ifdef PTHREADS
    pthread_mutex_unlock(&mutex_for_funi);
#endif
#endif
    for (i = 0; i < larg->nbc; ++i) {
      for (k = 0; k < larg->ndim; ++k) {
	/*NOTE!!
	  This needs to split up.  Only the first processor does the first part
	  and only the last processors does the last part.*/
	if(larg->loop_offset + larg->loop_start == 0) {
	  cc[0][i][k] = ARRAY2D(dbc, i, k);
	}
	if(larg->loop_offset + larg->loop_end == larg->na) {
	  cc[larg->na-1 - larg->loop_offset][i][larg->nra + k] = 
	    ARRAY2D(dbc ,i , larg->ndim + k);
	}
      }
    }
  }
  
  /*     Integral constraints : */
  if (larg->nint > 0) {
    for (jj = larg->loop_start; jj < larg->loop_end; ++jj) {
      j = jj;
      jp1 = j + 1;
      for (k = 0; k < (larg->ncol + 1); ++k) {
	for (i = 0; i < larg->ndim; ++i) {
	  i1 = k * larg->ndim + i;
	  j1 = j;
	  if (k+1 == (larg->ncol + 1)) {
	    i1 = i;
	  }
	  if (k+1 == (larg->ncol + 1)) {
	    j1 = jp1;
	  }
	  uic[i] = ups[j1 + larg->loop_offset][i1];
	  uio[i] = uoldps[j1 + larg->loop_offset][i1];
	  uid[i] = udotps[j1 + larg->loop_offset][i1];
	  uip[i] = upoldp[j1 + larg->loop_offset][i1];
	}
	
#ifdef PTHREADS_USE_ICNI_MUTEX      
#ifdef PTHREADS
	pthread_mutex_lock(&mutex_for_funi);
#endif
#endif
	(*(larg->icni))(larg->iap, larg->rap, larg->ndim, larg->par, 
		larg->icp, larg->nint, 
		uic, uio, uid, uip, ficd, 2, dicd);
#ifdef PTHREADS_USE_ICNI_MUTEX      
#ifdef PTHREADS
	pthread_mutex_unlock(&mutex_for_funi);
#endif
#endif
	
	for (m = 0; m < larg->nint; ++m) {
	  for (i = 0; i < larg->ndim; ++i) {
	    k1 = k * larg->ndim + i;
	    cc[jj][larg->nbc + m][k1] = 
	      larg->dtm[j] * larg->wi[k ] * ARRAY2D(dicd, m, i);
	  }
	}
      }
    }
  }
  /*     Pseudo-arclength equation : */
#ifdef MANIFOLD
  udotps_off=larg->iap->ntst + 1;
#endif
  for (jj = larg->loop_start; jj < larg->loop_end; ++jj) {
#ifdef MANIFOLD
    for (m = 0; m < larg->nalc; ++m) {
#endif
    for (i = 0; i < larg->ndim; ++i) {
      for (k = 0; k < larg->ncol; ++k) {
	k1 = k * larg->ndim + i;
#ifndef MANIFOLD
	cc[jj][larg->nrc - 1][k1] = 
	  larg->dtm[jj] * larg->thu[i] * larg->wi[k] * 
	  udotps[jj + larg->loop_offset][k1];
#else
        cc[jj][larg->nrc - 1][k1] =
          larg->dtm[jj] * larg->thu[i] * larg->wi[k] *
          udotps[jj + larg->loop_offset + m * udotps_off][k1];
#endif
      }
#ifndef MANIFOLD
      cc[jj][larg->nrc -1][larg->nra + i] = 
	larg->dtm[jj] * larg->thu[i] * larg->wi[larg->ncol] * 
	udotps[jj + 1 + larg->loop_offset][i];
#else
      cc[jj][larg->nrc -1][larg->nra + i] =
        larg->dtm[jj] * larg->thu[i] * larg->wi[larg->ncol] *
        udotps[jj + 1 + larg->loop_offset + m*udotps_off][i];
      }
#endif
    }
  }


#ifdef PTHREADS_PARALLEL_FA
#ifdef USAGE
  usage_start(&fa_usage);
#endif

  setubv_make_fa(*larg);
  
#ifdef USAGE
  usage_end(fa_usage,"setubv make fa");
#endif
#endif
  FREE(dicd );
  FREE(ficd );
  FREE(dfdp );
  FREE(dfdu );
  FREE(uold );
  FREE(f    );
  FREE(u    );
  FREE_DMATRIX(wploc);
  FREE(dbc  );
  FREE(fbc  );
  FREE(uic  );
  FREE(uio  );
  FREE(prm  );
  FREE(uid  );
  FREE(uip  );
  FREE(ubc0 );
  FREE(ubc1 );

#ifdef USAGE
  usage_end(setubv_make_aa_bb_cc_usage,"in setubv worker");
#endif

  return NULL;

}

#ifdef PTHREADS
int 
setubv_threads_wrapper(setubv_parallel_arglist data)
{
  setubv_parallel_arglist *send_data;
  int i;
  pthread_t *th;
  void * retval;
  pthread_attr_t attr;
  int retcode;
#ifdef USAGE
  struct timeval *pthreads_create,*pthreads_join,*pthreads_all;
  time_start(&pthreads_create);
  time_start(&pthreads_all);
#endif
  th = (pthread_t *)MALLOC(sizeof(pthread_t)*global_num_procs);
  send_data = (setubv_parallel_arglist *)MALLOC(sizeof(setubv_parallel_arglist)*global_num_procs);
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr,PTHREAD_SCOPE_SYSTEM);

  for(i=0;i<global_num_procs;i++) {
    setubv_parallel_arglist_copy(&send_data[i],data);
    send_data[i].loop_start = (i*(data.na))/global_num_procs;
    send_data[i].loop_end = ((i+1)*(data.na))/global_num_procs;
    send_data[i].loop_offset = 0;
    retcode = pthread_create(&th[i], &attr, setubv_make_aa_bb_cc, (void *) &send_data[i]);
    if (retcode != 0) fprintf(stderr, "create %d failed %d\n", i, retcode);
  }
#ifdef USAGE
  time_end(pthreads_create,"setubv pthreads create",fp9);
  time_start(&pthreads_join);
#endif
  for(i=0;i<global_num_procs;i++) {
    retcode = pthread_join(th[i], &retval);
    if (retcode != 0) fprintf(stderr, "join %d failed %d\n", i, retcode);
  }  
  FREE(send_data);
  FREE(th);
#ifdef USAGE
  time_end(pthreads_join,"setubv pthreads join",fp9);
  time_end(pthreads_all,"setubv pthreads all",fp9);
#endif

  return 0;
}
#endif

#ifdef MPI
int 
setubv_mpi_wrapper(setubv_parallel_arglist data)
{
  integer loop_start,loop_end;
  integer loop_start_tmp,loop_end_tmp;
  integer loop_offset;
  int i,comm_size;
  int *aa_counts,*aa_displacements;
  int *bb_counts,*bb_displacements;
  int *cc_counts,*cc_displacements;
  int *dtm_counts,*dtm_displacements;

  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  aa_counts=(int *)MALLOC(sizeof(int)*comm_size);
  aa_displacements=(int *)MALLOC(sizeof(int)*comm_size);
  bb_counts=(int *)MALLOC(sizeof(int)*comm_size);
  bb_displacements=(int *)MALLOC(sizeof(int)*comm_size);
  cc_counts=(int *)MALLOC(sizeof(int)*comm_size);
  cc_displacements=(int *)MALLOC(sizeof(int)*comm_size);
  dtm_counts=(int *)MALLOC(sizeof(int)*comm_size);
  dtm_displacements=(int *)MALLOC(sizeof(int)*comm_size);
  aa_counts[0] = 0;
  aa_displacements[0] = 0;
  bb_counts[0] = 0;
  bb_displacements[0] = 0;
  cc_counts[0] = 0;
  cc_displacements[0] = 0;
  dtm_counts[0] = 0;
  dtm_displacements[0] = 0;

  
  for(i=1;i<comm_size;i++){
    
    /*Send message to get worker into setubv mode*/
    {
      int message=AUTO_MPI_SETUBV_MESSAGE;
      MPI_Send(&message,1,MPI_INT,i,0,MPI_COMM_WORLD);
    }
    loop_start = ((i-1)*(data.na))/(comm_size - 1);
    loop_end = ((i)*(data.na))/(comm_size - 1);
    aa_counts[i] = (data.nca)*(data.nra)*(loop_end-loop_start);
    aa_displacements[i] = (data.nca)*(data.nra)*loop_start;
    bb_counts[i] = (data.ncb)*(data.nra)*(loop_end-loop_start);
    bb_displacements[i] = (data.ncb)*(data.nra)*loop_start;
    cc_counts[i] = (data.nca)*(data.nrc)*(loop_end-loop_start);
    cc_displacements[i] = (data.nca)*(data.nrc)*loop_start;
    dtm_counts[i] = (loop_end-loop_start);
    dtm_displacements[i] = (loop_start);

    loop_start_tmp = 0;
    loop_end_tmp = loop_end-loop_start;
    MPI_Send(&loop_start_tmp ,1,MPI_LONG,i,0,MPI_COMM_WORLD);
    MPI_Send(&loop_end_tmp   ,1,MPI_LONG,i,0,MPI_COMM_WORLD);
    loop_offset = loop_start;
    MPI_Send(&loop_offset    ,1,MPI_LONG,i,0,MPI_COMM_WORLD);
  }

  {
    integer params[11];
    params[0]=data.na;
    params[1]=data.ndim;
    params[2]=data.ips;
    params[3]=data.ncol;
    params[4]=data.nbc;
    params[5]=data.nint;
    params[6]=data.ncb;
    params[7]=data.nrc;
    params[8]=data.nra;
    params[9]=data.nca;
    params[10]=data.ndxloc;
    MPI_Bcast(params     ,11,MPI_LONG,0,MPI_COMM_WORLD);
  }    

  {
    int position=0;
    void *buffer;
    int bufsize;
    int size_int,size_double;
    int niap,nrap;
    /* Here we compute the number of elements in the iap and rap structures.
       Since each of the structures is homogeneous we just divide the total
       size by the size of the individual elements.*/
    niap = sizeof(iap_type)/sizeof(integer);
    nrap = sizeof(rap_type)/sizeof(doublereal);
    MPI_Pack_size(niap+NPARX2,MPI_LONG,MPI_COMM_WORLD,&size_int);
    MPI_Pack_size(nrap+NPARX2+
		  (data.ndxloc)*(data.ndim)*(data.ncol)+
		  (data.ndxloc)*(data.ndim)*(data.ncol)+
		  (data.ncol + 1)*(data.ncol)+
		  (data.ncol + 1)*(data.ncol)+
		  (data.ncol + 1)+
		  (data.ndxloc)*(data.ndim)*(data.ncol)+
		  (data.ndxloc)*(data.ndim)*(data.ncol)+
		  (data.ndim)*8+
		  NPARX+
		  NPARX,
		  MPI_DOUBLE,MPI_COMM_WORLD,&size_double);
    bufsize = size_int + size_double;
    buffer=MALLOC((unsigned)bufsize);

    MPI_Pack(data.iap    ,niap,MPI_LONG,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.rap    ,nrap,MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    /**********************************************/
    MPI_Pack(data.par    ,NPARX2,MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.icp    ,NPARX2,MPI_LONG,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.ups    ,(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.uoldps ,(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.wp     ,(data.ncol + 1)*(data.ncol),MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.wt     ,(data.ncol + 1)*(data.ncol),MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.wi     ,(data.ncol + 1),MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.udotps ,(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.upoldp ,(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);

    MPI_Pack(data.thu    ,(data.ndim)*8,MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.thl    ,NPARX,MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    MPI_Pack(data.rldot  ,NPARX,MPI_DOUBLE,buffer,bufsize,&position,MPI_COMM_WORLD);
    
    MPI_Bcast(buffer     ,position,MPI_PACKED,0,MPI_COMM_WORLD);
  }

  MPI_Scatterv(data.dtm        ,dtm_counts,dtm_displacements,MPI_DOUBLE,
	       NULL,0,MPI_DOUBLE,
	       0,MPI_COMM_WORLD);

  /* Worker runs here */
  return 0;
}
#endif

int 
setubv_default_wrapper(setubv_parallel_arglist data)
{
  setubv_make_aa_bb_cc((void *)&data);
  return 0;
}

#ifndef MANIFOLD
int 
setubv(integer ndim, integer ips, integer na, integer ncol, integer nbc, integer nint, integer ncb, integer nrc, integer nra, integer nca, 
       FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, 
       doublereal rds, doublereal ***aa, doublereal ***bb, doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal *rlcur, 
       doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, 
       doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1)
#else
int 
setubv(integer ndim, integer ips, integer na, integer ncol, integer nbc, integer nint, integer nalc, integer ncb, integer nrc, integer nra, integer nca, 
       FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, 
       doublereal *rds, doublereal ***aa, doublereal ***bb, doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal *rlcur, 
       doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, 
       doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1)
#endif
{
  /* Local variables */
  integer i, j, k;

  doublereal *wi, **wp, **wt;
  
#ifdef USAGE
  struct rusage *initialization_usage,*fc_usage,*parallel_overhead_usage;
  usage_start(&initialization_usage);
#endif
  wi   = (doublereal *)MALLOC(sizeof(doublereal)*(ncol+1) );
  wp   = DMATRIX(ncol+1, ncol);
  wt   = DMATRIX(ncol+1, ncol);

  wint(ncol + 1, wi);
  genwts(ncol, ncol + 1, wt, wp);
  
  /* Initialize to zero. */
  for (i = 0; i < nrc; ++i) {
    fc[i] = 0.;
    for (k = 0; k < ncb; ++k) {
      dd[i][k] = 0.;
    }
  }

  /* Set constants. */
  for (i = 0; i < ncb; ++i) {
    par[icp[i]] = rlcur[i];
  }
  
  /*  NA is the local node's mesh interval number. */
  
  for (i = 0; i < na; ++i) {
    for (j = 0; j < nra; ++j) {
      for (k = 0; k < nca; ++k) {
	aa[i][j][k] = 0.;
      }
    }
    for (j = 0; j < nra; ++j) {
      for (k = 0; k < ncb; ++k) {
	bb[i][j][k] = 0.;
      }
    }
    for (j = 0; j < nrc; ++j) {
      for (k = 0; k < nca; ++k) {
	cc[i][j][k] = 0.;
      }
    }
  }

  /*     ** Time evolution computations (parabolic systems) */
  if (ips == 14 || ips == 16) {
    rap->tivp = rlold[0];
  } 
#ifdef USAGE
  usage_end(initialization_usage,"setubv initialization");
#endif

  {
    setubv_parallel_arglist arglist;
#ifndef MANIFOLD
    setubv_parallel_arglist_constructor(ndim, ips, na, ncol, nbc, nint, ncb, 
					nrc, nra, nca, funi, icni, ndxloc, iap, rap, 
					par, icp, aa, bb, cc, dd, fa, fc, ups, 
					uoldps, udotps, upoldp, dtm, wp, wt, wi, 
					thu, thl, rldot, bcni, &arglist);
#else
    setubv_parallel_arglist_constructor(ndim, ips, na, ncol, nbc, nint, nalc, ncb, 
					nrc, nra, nca, funi, icni, ndxloc, iap, rap, 
					par, icp, aa, bb, cc, dd, fa, fc, ups, 
					uoldps, udotps, upoldp, dtm, wp, wt, wi, 
					thu, thl, rldot, bcni, &arglist);
#endif
  
    switch(global_setubv_type) {

#ifdef PTHREADS
    case SETUBV_PTHREADS:
      setubv_threads_wrapper(arglist);
      break;
#endif

#ifdef MPI
    case SETUBV_MPI:
      if(global_verbose_flag)
	printf("Setubv MPI start\n");
      setubv_mpi_wrapper(arglist);
      if(global_verbose_flag)
	printf("Setubv MPI end\n");
      break;
#endif

    default:
      setubv_default_wrapper(arglist);
      break;
    }

#ifndef PTHREADS_PARALLEL_FA
#ifdef USAGE
  usage_start(&fa_usage);
#endif

  setubv_make_fa(arglist);
  
#ifdef USAGE
  usage_end(fa_usage,"setubv make fa");
#endif
#endif


#ifdef USAGE
    usage_start(&fc_usage);
#endif

    setubv_make_fc_dd(arglist,dups,rlcur,rlold,rds);

#ifdef USAGE
    usage_end(fc_usage,"setubv make fc");
#endif

  }

  FREE(wi   );
  FREE_DMATRIX(wp);
  FREE_DMATRIX(wt);
  return 0;
}

void setubv_make_fa(setubv_parallel_arglist larg) {
  integer i,j,k,l;
  integer ic,k1,ib;
  integer jj,jp1,l1,ic1;
  doublereal dt,ddt;

  doublereal **ups = larg.ups;

  doublereal **uoldps = larg.uoldps;

  doublereal **wp = larg.wp;

  doublereal **wt = larg.wt;
  
  doublereal **fa = larg.fa;
  
  doublereal **wploc= DMATRIX(larg.ncol+1, larg.ncol);
  
  doublereal *dfdp = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim)*NPARX);
  doublereal *dfdu = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim)*(larg.ndim));
  doublereal *u    = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  doublereal *uold = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  doublereal *f    = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  doublereal *prm  = (doublereal *)MALLOC(sizeof(doublereal)*NPARX);

  for (jj = larg.loop_start; jj < larg.loop_end; ++jj) {
    j = jj;
    jp1 = j + 1;
    dt = larg.dtm[j];
    ddt = 1. / dt;
    for (ic = 0; ic < larg.ncol; ++ic) {
      for (ib = 0; ib < larg.ncol + 1; ++ib) {
	wploc[ib][ic] = ddt * wp[ib][ic];
      }
    }
    for (ic = 0; ic < larg.ncol; ++ic) {
      for (k = 0; k < larg.ndim; ++k) {
	u[k] = wt[larg.ncol][ic] * ups[jp1][k];
	uold[k] = wt[larg.ncol][ic] * uoldps[jp1][k];
	for (l = 0; l < larg.ncol; ++l) {
	  l1 = l * larg.ndim + k;
	  u[k] += wt[l][ic] * ups[j + larg.loop_offset][l1];
	  uold[k] += wt[l][ic] * uoldps[j + larg.loop_offset][l1];
	}
      }

      for (i = 0; i < NPARX; ++i) {
	prm[i] = larg.par[i];
      }
#ifdef PTHREADS_USE_FUNI_MUTEX      
#ifdef PTHREADS
      pthread_mutex_lock(&mutex_for_funi);
#endif
#endif
      (*(larg.funi))(larg.iap, larg.rap, larg.ndim, u, uold, larg.icp, prm, 2, f, dfdu, dfdp);
#ifdef PTHREADS_USE_FUNI_MUTEX      
#ifdef PTHREADS
      pthread_mutex_unlock(&mutex_for_funi);
#endif
#endif

      ic1 = ic * (larg.ndim);
      for (i = 0; i < larg.ndim; ++i) {
	fa[ic1 + i][jj] = f[i] - wploc[larg.ncol][ic] * ups[jp1 + larg.loop_offset][i];
	for (k = 0; k < larg.ncol; ++k) {
	  k1 = k * larg.ndim + i;
	  fa[ic1 + i][jj] -= wploc[k][ic] * ups[j + larg.loop_offset][k1];
	}
      }
    }
  
  }
  FREE_DMATRIX(wploc);
  FREE(dfdp);
  FREE(dfdu);
  FREE(u);
  FREE(uold);
  FREE(f);
  FREE(prm);
  
}


#ifndef MANIFOLD
void setubv_make_fc_dd(setubv_parallel_arglist larg, doublereal **dups, doublereal *rlcur, 
	     doublereal *rlold, doublereal rds) {
#else
void setubv_make_fc_dd(setubv_parallel_arglist larg, doublereal **dups, doublereal *rlcur, 
	     doublereal *rlold, doublereal *rds) {
#endif
  integer i,j,jj,jp1,k,i1,m,j1;
  doublereal rlsum;

  doublereal **dd = larg.dd;

  doublereal **ups = larg.ups;

  doublereal **uoldps = larg.uoldps;
  
  doublereal **udotps = larg.udotps;
  
  doublereal **upoldp = larg.upoldp;
  
  integer dbc_dim1 = larg.nbc;
  doublereal *dbc  = (doublereal *)MALLOC(sizeof(doublereal)*(larg.nbc)*(2*larg.ndim + NPARX));
  doublereal *fbc  = (doublereal *)MALLOC(sizeof(doublereal)*(larg.nbc));
  doublereal *ubc0 = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  doublereal *ubc1 = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  integer dicd_dim1 = larg.nint;
  doublereal *dicd = NULL;
  doublereal *ficd = NULL;
  
  doublereal *uic  = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  doublereal *uio  = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  doublereal *uid  = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
  doublereal *uip  = (doublereal *)MALLOC(sizeof(doublereal)*(larg.ndim));
#ifdef MANIFOLD
  integer udotps_off;
#endif

  if (larg.nint > 0)
  {
       dicd = (doublereal *)MALLOC(sizeof(doublereal)*(larg.nint)*(larg.ndim + NPARX));
       ficd = (doublereal *)MALLOC(sizeof(doublereal)*(larg.nint));
  }
  
  /* Boundary condition part of FC */
  if (larg.nbc > 0) {
    for (i = 0; i < larg.ndim; ++i) {
      ubc0[i] = ups[0][i];
      ubc1[i] = ups[larg.na][i];
    }
    
    (*(larg.bcni))(larg.iap, larg.rap, larg.ndim, larg.par, 
		   larg.icp, larg.nbc, ubc0, ubc1, fbc, 2, dbc);
    for (i = 0; i < larg.nbc; ++i) {
      larg.fc[i] = -fbc[i];
      for (k = 0; k < larg.ncb; ++k) {
	dd[i][k] = 
	  ARRAY2D(dbc, i, (larg.ndim *2) + larg.icp[k]);
      }
    }
    /*       Save difference : */
    for (j = 0; j < larg.na + 1; ++j) {
      for (i = 0; i < larg.nra; ++i) {
	dups[j][i] = ups[j][i] - uoldps[j][i];
      }
    }
  }

  /* Integral constraint part of FC */
  if (larg.nint > 0) {
    for (jj = larg.loop_start; jj < larg.loop_end; ++jj) {
      j = jj;
      jp1 = j + 1;
      for (k = 0; k < (larg.ncol + 1); ++k) {
	for (i = 0; i < larg.ndim; ++i) {
	  i1 = k * larg.ndim + i;
	  j1 = j;
	  if (k+1 == (larg.ncol + 1)) {
	    i1 = i;
	  }
	  if (k+1 == (larg.ncol + 1)) {
	    j1 = jp1;
	  }
	  uic[i] = ups[j1][i1];
	  uio[i] = uoldps[j1][i1];
	  uid[i] = udotps[j1][i1];
	  uip[i] = upoldp[j1][i1];
	}
	
	(*(larg.icni))(larg.iap, larg.rap, larg.ndim, larg.par, 
		larg.icp, larg.nint, 
		uic, uio, uid, uip, ficd, 2, dicd);
	
	for (m = 0; m < larg.nint; ++m) {
	  larg.fc[larg.nbc + m] -= larg.dtm[j] * larg.wi[k] * ficd[m];
	  for (i = 0; i < larg.ncb; ++i) {
	    dd[larg.nbc + m][i] += 
	      larg.dtm[j] * larg.wi[k] * ARRAY2D(dicd, m, larg.ndim + larg.icp[i]);
	  }
	}
      }
    }
  }

#ifndef MANIFOLD
  for (i = 0; i < larg.ncb; ++i) {
    dd[larg.nrc-1][i] = larg.thl[larg.icp[i]] * larg.rldot[i];
  }

  rlsum = 0.;
  for (i = 0; i < larg.ncb; ++i) {
    rlsum += larg.thl[larg.icp[i]] * (rlcur[i] - rlold[i]) * larg.rldot[i];
  }

  larg.fc[larg.nrc-1] = rds - rinpr(larg.iap, &(larg.ndim), &(larg.ndxloc), larg.udotps, dups, larg.dtm, larg.thu) - rlsum;
#else
  udotps_off=(larg.iap->ntst + 1)*(larg.iap->ndim * larg.iap->ncol);
  for (m = 0; m < larg.nalc; ++m) {
      for (i = 0; i < larg.ncb; ++i) {
        dd[larg.nbc+larg.nint+m][i] = larg.thl[larg.icp[i]] * larg.rldot[i+m*NPARX];
      }

    rlsum = 0.;
    for (i = 0; i < larg.ncb; ++i) {
      rlsum += larg.thl[larg.icp[i]] * (rlcur[i] - rlold[i]) * larg.rldot[i+m*NPARX];
    }
    larg.fc[larg.nrc-1+m] = rds[m] - rinpr(larg.iap, &(larg.ndim), &(larg.ndxloc), larg.udotps, dups, larg.dtm, larg.thu) - rlsum;
   }
#endif

  FREE(dbc);
  FREE(fbc);
  FREE(ubc0);
  FREE(ubc1);
  FREE(dicd);
  FREE(ficd);
  FREE(uic);
  FREE(uio);
  FREE(uid);
  FREE(uip);

}

/* Copy a setubv_parallel_arglist */
void setubv_parallel_arglist_copy(setubv_parallel_arglist *output,
				  const setubv_parallel_arglist input) {
  memcpy(output,&input,sizeof(setubv_parallel_arglist));
}


/* Fill in a setubv_parallel_arglist for the individual variables */
#ifndef MANIFOLD
void setubv_parallel_arglist_constructor(integer ndim, integer ips, integer na, integer ncol, 
					 integer nbc, integer nint, integer ncb, integer nrc, integer nra, integer nca, 
					 FUNI_TYPE((*funi)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, 
					 integer *icp, doublereal ***aa, doublereal ***bb, 
					 doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal **ups, 
					 doublereal **uoldps, doublereal **udotps, 
					 doublereal **upoldp, doublereal *dtm, 
					 doublereal **wp, doublereal **wt, doublereal *wi,
					 doublereal *thu, doublereal *thl,
					 doublereal *rldot, BCNI_TYPE((*bcni)), setubv_parallel_arglist *data) {
#else
void setubv_parallel_arglist_constructor(integer ndim, integer ips, integer na, integer ncol, 
					 integer nbc, integer nint, integer nalc, integer ncb, integer nrc, integer nra, integer nca, 
					 FUNI_TYPE((*funi)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, 
					 integer *icp, doublereal ***aa, doublereal ***bb, 
					 doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal **ups, 
					 doublereal **uoldps, doublereal **udotps, 
					 doublereal **upoldp, doublereal *dtm, 
					 doublereal **wp, doublereal **wt, doublereal *wi,
					 doublereal *thu, doublereal *thl,
					 doublereal *rldot, BCNI_TYPE((*bcni)), setubv_parallel_arglist *data) {
#endif
  data->ndim   = ndim;
  data->ips    = ips;
  data->ncol   = ncol;
  data->nbc    = nbc;
  data->nint   = nint;
#ifdef MANIFOLD
  data->nalc   = nalc;
#endif
  data->ncb    = ncb;
  data->nrc    = nrc;
  data->nra    = nra;
  data->nca    = nca;
  data->na     = na;
  data->funi   = funi;
  data->icni   = icni;
  data->ndxloc = ndxloc;
  data->iap    = iap;
  data->rap    = rap;
  data->par    = par;
  data->icp    = icp;
  data->aa     = aa;
  data->bb     = bb;
  data->cc     = cc;
  data->dd     = dd;
  data->fa     = fa;
  data->fc     = fc;
  data->ups    = ups;
  data->uoldps = uoldps;
  data->udotps = udotps;
  data->upoldp = upoldp;
  data->dtm    = dtm;
  data->loop_start = 0;
  data->loop_end   = na;
  data->loop_offset = 0;
  data->wp     = wp;
  data->wt     = wt;
  data->wi     = wi;
  data->thu    = thu;
  data->thl    = thl;
  data->rldot  = rldot;
  data->bcni   = bcni;
}  









