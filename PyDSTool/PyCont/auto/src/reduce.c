#include "auto_f2c.h"
#include "auto_c.h"
#include "auto_types.h"

#ifdef PTHREADS
pthread_mutex_t reduce_mutex_for_d = PTHREAD_MUTEX_INITIALIZER;
#endif

/*This is the process function.  It is meant to be called either
  on a SMP using shared memory, or wrapped inside another
  routine for message passing*/

void *reduce_process(void * arg)
{
  integer icf_dim1, irf_dim1;
  
  /* Local variables */
  integer ipiv, jpiv, itmp;
  doublereal tpiv;
  integer i, j, l, k1, k2, m1, m2, ic, ir;
  doublereal rm;
  integer ir1, irp;
  doublereal piv;
  integer icp1;

  integer *nov, *nra, *nca;
  doublereal ***a;
  integer *ncb;
  doublereal ***b;
  integer *nbc, *nrc;
  doublereal ***c, **d;
  integer *irf, *icf;
  integer loop_start,loop_end;

#ifdef PTHREADS
  doublereal **dlocal;
#endif

#ifdef USAGE
  struct rusage *reduce_process_usage;
  usage_start(&reduce_process_usage);
#endif

  nov = ((reduce_parallel_arglist *)arg)->nov;
  nra = ((reduce_parallel_arglist *)arg)->nra;
  nca = ((reduce_parallel_arglist *)arg)->nca;
  a = ((reduce_parallel_arglist *)arg)->a;
  ncb = ((reduce_parallel_arglist *)arg)->ncb;
  b = ((reduce_parallel_arglist *)arg)->b;
  nbc = ((reduce_parallel_arglist *)arg)->nbc;
  nrc = ((reduce_parallel_arglist *)arg)->nrc;
  c = ((reduce_parallel_arglist *)arg)->c;
  d = ((reduce_parallel_arglist *)arg)->d;
  irf = ((reduce_parallel_arglist *)arg)->irf;
  icf = ((reduce_parallel_arglist *)arg)->icf;
  loop_start = ((reduce_parallel_arglist *)arg)->loop_start;
  loop_end = ((reduce_parallel_arglist *)arg)->loop_end;

#ifdef PTHREADS
  dlocal=DMATRIX(*nrc, *ncb);
#endif
  /* In the default case we don't need to do anything special */
  if(global_reduce_type == REDUCE_DEFAULT) {
    ;
  }
  /* In the message passing case we set d to be
     0.0, do a sum here, and then do the final
     sum (with the true copy of d) in the
     master */
  else if (global_reduce_type == REDUCE_MPI) {
    for(i=0;i<*nrc;i++)
      for (j=0; j<*ncb;j++)
        d[i][j]=0.0;
  }
  /* In the shared memory case we create a local
     variable for doing this threads part of the
     sum, then we do a final sum into shared memory
     at the end */
  else if (global_reduce_type == REDUCE_PTHREADS) {
#ifdef PTHREADS
    for(i=0;i<*nrc;i++)
      for (j=0; j<*ncb;j++)
          dlocal[i][j]=0.0;
#else
    ;
#endif
  }

  /* Note that the summation of the adjacent overlapped part of C */
  /* is delayed until REDUCE, in order to merge it with other communications.*/
  /* NA is the local NTST. */
  
  irf_dim1 = *nra;
  icf_dim1 = *nca;
  
  /* Condensation of parameters (Elimination of local variables). */
  m1 = *nov + 1;
  m2 = *nca - *nov;

  for (i = loop_start;i < loop_end; i++) {
    for (ic = m1; ic <= m2; ++ic) {
      ir1 = ic - *nov + 1;
      irp = ir1 - 1;
      icp1 = ic + 1;
      /*	     **Search for pivot (Complete pivoting) */
      piv = 0.0;
      ipiv = irp;
      jpiv = ic;
      for (k1 = irp; k1 <= *nra; ++k1) {
	int irf_k1_i = irf[-1 + k1 + i*irf_dim1];
	for (k2 = ic; k2 <= m2; ++k2) {
	  int icf_k2_i = icf[-1 + k2 + i*icf_dim1];
	  tpiv = a[i][-1 + irf_k1_i][-1 + icf_k2_i];
	  if (tpiv < 0.0) {
	    tpiv = -tpiv;
	  }
	  if (piv < tpiv) {
	    piv = tpiv;
	    ipiv = k1;
	    jpiv = k2;
	  }
	}
      }
      /*	     **Move indices */
      itmp = icf[-1 + ic + i*icf_dim1];
      icf[-1 + ic + i*icf_dim1] = icf[-1 + jpiv + i*icf_dim1];
      icf[-1 + jpiv + i*icf_dim1] = itmp;
      itmp = irf[-1 + irp + i*irf_dim1];
      irf[-1 + irp + i*irf_dim1] = irf[-1 + ipiv + i*irf_dim1];
      irf[-1 + ipiv + i*irf_dim1] = itmp;
      {
	int icf_ic_i = icf[-1 + ic + i*icf_dim1];
	int irf_irp_i = irf[-1 + irp + i*irf_dim1];
	doublereal *a_offset2 = a[i][-1 + irf_irp_i];
	doublereal *b_offset2 = b[i][-1 + irf_irp_i];
	/*	     **End of pivoting; elimination starts here */
	for (ir = ir1; ir <= *nra; ++ir) {
	  int irf_ir_i = irf[-1 + ir + i*irf_dim1];
	  doublereal *a_offset1 = a[i][-1 + irf_ir_i];
	  doublereal *b_offset1 = b[i][-1 + irf_ir_i];
	  rm = a_offset1[-1 + icf_ic_i]/a_offset2[-1 + icf_ic_i];
	  a[i][-1 + irf_ir_i][-1 + icf_ic_i] = rm;
	  if (rm != (double)0.) {
	    for (l = 0; l < *nov; ++l) {
                a_offset1[l] -= rm * a_offset2[l];
	    }
	    for (l = icp1 -1; l < *nca; ++l) {
	      int icf_l_i = icf[l + i*icf_dim1];
              a_offset1[-1 + icf_l_i] -= rm * a_offset2[-1 + icf_l_i];
	    }
	    for (l = 0; l < *ncb; ++l) {
	      b_offset1[l] -= rm * b_offset2[l];
	    }
	  }
	}
	for (ir = *nbc + 1; ir <= *nrc; ++ir) {
          doublereal *c_offset1 = c[i][-1 + ir];
          doublereal *d_offset1 = d[-1 + ir];
	  rm = c_offset1[-1 + icf_ic_i]/a_offset2[-1 + icf_ic_i];
	  c_offset1[-1 + icf_ic_i]=rm;
	  if (rm != (double)0.) {
	    for (l = 0; l < *nov; ++l) {
	      c_offset1[l] -= rm * a_offset2[l];
	    }
	    for (l = icp1 -1 ; l < *nca; ++l) {
	      int icf_l_i = icf[l + i*icf_dim1];
	      c_offset1[-1 + icf_l_i] -= rm * a_offset2[-1 + icf_l_i];
	    }
	    for (l = 0; l < *ncb; ++l) {
	      /* 
		 A little explanation of what is going on here
		 is in order I believe.  This array is
		 created by a summation across all workers,
		 hence it needs a mutex to avoid concurrent
		 writes (in the shared memory case) or a summation
		 in the master (in the message passing case).
		 Since mutex's can be somewhat slow, we will do the
		 summation into a local variable, and then do a
		 final summation back into global memory when the
		 main loop is done.
	      */
	      /* Nothing special for the default case */
	      if(global_reduce_type == REDUCE_DEFAULT) {
		d_offset1[l] -= rm * b_offset2[l];
	      }
	      /* In the message passing case we sum into d,
		 which is a local variable initialized to 0.0.
		 We then sum our part with the masters part
		 in the master. */
	      else if (global_reduce_type == REDUCE_MPI) {
		d_offset1[l] -= rm * b_offset2[l];
	      }
	      /* In the shared memory case we sum into a local
		 variable our contribution, and then sum
		 into shared memory at the end (inside a mutex */
	      else if (global_reduce_type == REDUCE_PTHREADS) {
#ifdef PTHREADS
		dlocal[l][-1 + ir] -= rm * b_offset2[l];

#else
		;
#endif
	      }
	    }
	  }
	}
      }
    }
  }
#ifdef PTHREADS
  /* This is were we sum into the global copy of the d
     array, in the shared memory case */
  if(global_reduce_type == REDUCE_PTHREADS) {
#ifdef PTHREADS
    pthread_mutex_lock(&reduce_mutex_for_d);
    for(i=0;i<*ncb;i++)
      for (j=0; j<*nrc;j++)
        d[i][j] += dlocal[i][j];
    pthread_mutex_unlock(&reduce_mutex_for_d);
    FREE_DMATRIX(dlocal);
#else
    ;
#endif
  }
#endif
#ifdef USAGE
  usage_end(reduce_process_usage,"in reduce worker");
#endif
  return NULL;
}

#ifdef PTHREADS
int reduce_threads_wrapper(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *ncb, 
			   doublereal ***b, integer *nbc, integer *nrc, doublereal ***c, doublereal **d, 
			   integer *irf, integer *icf)

{
  /* Aliases for the dimensions of the arrays */
  integer icf_dim1, irf_dim1;
  
  reduce_parallel_arglist *data;
  int i;
  pthread_t *th;
  void * retval;
  pthread_attr_t attr;
  int retcode;
#ifdef USAGE
  struct timeval *pthreads_wait;
  time_start(&pthreads_wait);
#endif
  
  data = (reduce_parallel_arglist *)MALLOC(sizeof(reduce_parallel_arglist)*global_num_procs);
  th = (pthread_t *)MALLOC(sizeof(pthread_t)*global_num_procs);

  irf_dim1 = *nra;
  icf_dim1 = *nca;

  for(i=0;i<global_num_procs;i++) {
    
    /*start and end of the computed loop*/
    data[i].loop_start = (i*(*na))/global_num_procs;
    data[i].loop_end = ((i+1)*(*na))/global_num_procs;
    
    /*3D Arrays*/ 
    data[i].a = a + data[i].loop_start;
    data[i].b = b + data[i].loop_start;
    data[i].c = c + data[i].loop_start;
    
    /*2D Arrays*/
    data[i].d = d;
    data[i].irf = irf + data[i].loop_start*irf_dim1;
    data[i].icf = icf + data[i].loop_start*icf_dim1;
    
    /*Scalars*/
    data[i].nbc = nbc;
    data[i].nrc = nrc;
    data[i].ncb = ncb;
    data[i].nov = nov;
    data[i].nra = nra;
    data[i].nca = nca;
    
    data[i].loop_end = data[i].loop_end - data[i].loop_start;
    data[i].loop_start = 0;
    
  }
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr,PTHREAD_SCOPE_SYSTEM);
  for(i=0;i<global_num_procs;i++) {
    retcode = pthread_create(&th[i], &attr, reduce_process, (void *) &data[i]);
    if (retcode != 0) fprintf(stderr, "create %d failed %d\n", i, retcode);
  }
  
  for(i=0;i<global_num_procs;i++) {
    retcode = pthread_join(th[i], &retval);
    if (retcode != 0) fprintf(stderr, "join %d failed %d\n", i, retcode);
  }  
  FREE(data);
  FREE(th);
#ifdef USAGE
  time_end(pthreads_wait,"reduce pthreads wrapper",fp9);
#endif
  return 0;
}
#endif

#ifdef MPI
int 
reduce_mpi_wrapper(integer *nov, integer *na, integer *nra, 
		   integer *nca, doublereal *a, integer *ncb, 
		   doublereal *b, integer *nbc, integer *nrc, 
		   doublereal *c, doublereal *d, integer *irf, integer *icf)

{
    integer loop_start,loop_end;
    integer loop_start_tmp,loop_end_tmp;
    int i,comm_size;
    int *a_counts,*a_displacements;
    int *b_counts,*b_displacements;
    int *c_counts,*c_displacements;
    int *irf_counts,*irf_displacements;
    int *icf_counts,*icf_displacements;


    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    a_counts=(int *)MALLOC(sizeof(int)*comm_size);
    a_displacements=(int *)MALLOC(sizeof(int)*comm_size);
    b_counts=(int *)MALLOC(sizeof(int)*comm_size);
    b_displacements=(int *)MALLOC(sizeof(int)*comm_size);
    c_counts=(int *)MALLOC(sizeof(int)*comm_size);
    c_displacements=(int *)MALLOC(sizeof(int)*comm_size);
    irf_counts=(int *)MALLOC(sizeof(int)*comm_size);
    irf_displacements=(int *)MALLOC(sizeof(int)*comm_size);
    icf_counts=(int *)MALLOC(sizeof(int)*comm_size);
    icf_displacements=(int *)MALLOC(sizeof(int)*comm_size);
    a_counts[0] = 0;
    a_displacements[0] = 0;
    b_counts[0] = 0;
    b_displacements[0] = 0;
    c_counts[0] = 0;
    c_displacements[0] = 0;
    irf_counts[0] = 0;
    irf_displacements[0] = 0;
    icf_counts[0] = 0;
    icf_displacements[0] = 0;

    for(i=1;i<comm_size;i++){
      
      /*Send message to get worker into reduce mode*/
      {
	int message=AUTO_MPI_REDUCE_MESSAGE;
	MPI_Send(&message,1,MPI_INT,i,0,MPI_COMM_WORLD);
      }
      loop_start = ((i-1)*(*na))/(comm_size - 1);
      loop_end = ((i)*(*na))/(comm_size - 1);
      a_counts[i] = (*nca)*(*nra)*(loop_end-loop_start);
      a_displacements[i] = (*nca)*(*nra)*loop_start;
      b_counts[i] = (*ncb)*(*nra)*(loop_end-loop_start);
      b_displacements[i] = (*ncb)*(*nra)*loop_start;
      c_counts[i] = (*nca)*(*nrc)*(loop_end-loop_start);
      c_displacements[i] = (*nca)*(*nrc)*loop_start;
      irf_counts[i] = (*nra)*(loop_end-loop_start);
      irf_displacements[i] = (*nra)*loop_start;
      icf_counts[i] = (*nca)*(loop_end-loop_start);
      icf_displacements[i] = (*nca)*loop_start;
      loop_start_tmp = 0;
      loop_end_tmp = loop_end-loop_start;
      MPI_Send(&loop_start_tmp ,1,MPI_LONG,i,0,MPI_COMM_WORLD);
      MPI_Send(&loop_end_tmp   ,1,MPI_LONG,i,0,MPI_COMM_WORLD);
    }
    {
      integer params[6];
      params[0]=*nov;
      params[1]=*nra;
      params[2]=*nca;
      params[3]=*ncb;
      params[4]=*nbc;
      params[5]=*nrc;

      
      MPI_Bcast(params        ,6,MPI_LONG,0,MPI_COMM_WORLD);
    }
    MPI_Scatterv(irf,irf_counts,irf_displacements,MPI_LONG,
		 NULL,0,MPI_LONG,
		 0,MPI_COMM_WORLD);
    MPI_Scatterv(icf,icf_counts,icf_displacements,MPI_LONG,
		 NULL,0,MPI_LONG,
		 0,MPI_COMM_WORLD);

    /* Worker is running now */

    {
      /*I create a temporary send buffer for the MPI_Reduce
	command.  This is because there isn't an
	asymmetric version (like MPI_Scatterv).*/
      double **dtemp;
      dtemp = DMATRIX(*nrc,*ncb);
      for(i=0;i<(*nrc);i++)
        for(j=0;i<(*ncb);i++)
          dtemp[i][j]=d[i][j];
      MPI_Reduce(dtemp[0],d[0],(*ncb)*(*nrc),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      FREE_DMATRIX(dtemp);
    }
    MPI_Gatherv(NULL,0,MPI_DOUBLE,
		a[0][0],a_counts,a_displacements,MPI_DOUBLE,
		0,MPI_COMM_WORLD);
    MPI_Gatherv(NULL,0,MPI_DOUBLE,
		b[0][0],b_counts,b_displacements,MPI_DOUBLE,
		0,MPI_COMM_WORLD);
    MPI_Gatherv(NULL,0,MPI_DOUBLE,
		c[0][0],c_counts,c_displacements,MPI_DOUBLE,
		0,MPI_COMM_WORLD);
    MPI_Gatherv(NULL,0,MPI_LONG,
		irf,irf_counts,irf_displacements,MPI_LONG,
		0,MPI_COMM_WORLD);
    MPI_Gatherv(NULL,0,MPI_LONG,
		icf,icf_counts,icf_displacements,MPI_LONG,
		0,MPI_COMM_WORLD);
    return 0;
}
#endif

int 
reduce_default_wrapper(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *ncb, doublereal ***b, integer *nbc, integer *nrc, doublereal ***c, doublereal **d, integer *irf, integer *icf)

{
    reduce_parallel_arglist data;
    data.nov = nov;
    data.nra = nra;
    data.nca = nca;
    data.a = a;
    data.ncb = ncb;
    data.b = b;
    data.nbc = nbc;
    data.nrc = nrc;
    data.c = c;
    data.d = d;
    data.irf = irf;
    data.icf = icf;
    data.loop_start = 0;
    data.loop_end = *na;
    reduce_process(&data);
    return 0;
}


/*
int 
reduce(integer *nov, integer *na, integer *nra, integer *nca, doublereal *a, integer *ncb, doublereal *b, integer *nbc, integer *nrc, doublereal *c, doublereal *d, integer *irf, integer *icf)
{
  integer icf_dim1, irf_dim1;
  
  integer i,j;
  integer nex;

  irf_dim1 = *nra;
  icf_dim1 = *nca;

  nex = *nca - (*nov << 1);
  if (nex == 0) {
    return 0;
  }
  
  for (i = 0; i <*na; ++i) {
    for (j = 0; j < *nra; ++j) {
      irf[j + i * irf_dim1] = j+1;
    }
    for (j = 0; j < *nca; ++j) {
      icf[j + i * icf_dim1] = j+1;
    }
  }

  switch(global_reduce_type) {
#ifdef PTHREADS
  case REDUCE_PTHREADS:
    reduce_threads_wrapper(nov, na, nra, nca, a, 
			    ncb, b, nbc, nrc, c, d,irf, icf);
    break;
#endif
#ifdef MPI
  case REDUCE_MPI:
    if(global_verbose_flag)
      printf("MPI reduce start\n");
    reduce_mpi_wrapper(nov, na, nra, nca, a, 
			ncb, b, nbc, nrc, c, d,irf, icf);
    if(global_verbose_flag)
      printf("MPI reduce end\n");
    break;
#endif
  default:
    reduce_default_wrapper(nov, na, nra, nca, a, 
			    ncb, b, nbc, nrc, c, d,irf, icf);
    break;
  }
  return 0;
} 
*/

int 
reduce(integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, doublereal **dd, integer *na, integer *nov, integer *ncb, integer *nrc, doublereal ***s1, doublereal ***s2, doublereal ***ca1, integer *icf1, integer *icf2, integer *icf11, integer *ipr, integer *nbc)
{
  /* System generated locals */
    integer icf1_dim1, icf2_dim1, icf11_dim1, ipr_dim1;

  doublereal zero, tpiv;
  real xkwt;
  integer nbcp1, ibuf1, ipiv1, jpiv1, ipiv2, jpiv2, i, k, l;

  integer i1, i2, k1, k2, i3, iprow, k3, l2, l3, ic, ir;
  doublereal rm;
  doublereal tmp;
  integer icp1;
  integer itmp;
  doublereal piv1, piv2;
  doublereal *buf=NULL;

#ifdef USAGE
  struct rusage *init, *mainloop,*pivoting,*elimination;
  usage_start(&init);
#endif

  /* Parameter adjustments */
  ipr_dim1 = *nov;
  icf11_dim1 = *nov;
  icf2_dim1 = *nov;
  icf1_dim1 = *nov;
    
  zero = 0.;
  nbcp1 = *nbc + 1;
  xkwt = (real) (*kwt);


  /* Initialization */

  for (i = 0; i < *na; ++i) {
    for (k1 = 0; k1 < *nov; ++k1) {
      ARRAY2D(icf1, k1, i) = k1 + 1;
      ARRAY2D(icf2, k1, i) = k1 + 1;
      ARRAY2D(ipr, k1, i) = k1 + 1;
      for (k2 = 0; k2 < *nov; ++k2) {
	s2[i][k1][k2] = 0.;
	s1[i][k1][k2] = 0.;
      }
    }
  }

  for (ir = 0; ir < *nov; ++ir) {
    for (ic = 0; ic < *nov; ++ic) {
      s1[0][ir][ic] = a1[0][ir][ic];
    }
  }
#ifdef USAGE
  usage_end(init,"reduce initialization");
  usage_start(&mainloop);
#endif

  /* The reduction process is done concurrently */
  for (i1 = 0; i1 < *na - 1; ++i1) {
    i2 = i1 + 1;
    i3 = i2 + 1;

    for (ic = 0; ic < *nov; ++ic) {
      icp1 = ic + 1;

      /* Complete pivoting; rows are swapped physically, columns swap in
	 dices */
      piv1 = zero;
      ipiv1 = ic + 1;
      jpiv1 = ic + 1;
      for (k1 = ic; k1 < *nov; ++k1) {
	for (k2 = ic; k2 < *nov; ++k2) {
	  tpiv = a2[i1][k1][ARRAY2D(icf2, k2, i1) - 1];
	  if (tpiv < zero) {
	    tpiv = -tpiv;
	  }
	  if (piv1 < tpiv) {
	    piv1 = tpiv;
	    ipiv1 = k1 + 1;
	    jpiv1 = k2 + 1;
	  }
	}
      }

      piv2 = zero;
      ipiv2 = 1;
      jpiv2 = ic + 1;
      for (k1 = 0; k1 < *nov; ++k1) {
	for (k2 = ic; k2 < *nov; ++k2) {
          tpiv = a1[i2][k1][ARRAY2D(icf1, k2, i2) - 1];
	  if (tpiv < zero) {
	    tpiv = -tpiv;
	  }
	  if (piv2 < tpiv) {
	    piv2 = tpiv;
	    ipiv2 = k1 + 1;
	    jpiv2 = k2 + 1;
	  }
	}
      }
      if (piv1 >= piv2) {
	ARRAY2D(ipr, ic, i1) = ipiv1;
	itmp = ARRAY2D(icf2, ic, i1);
	ARRAY2D(icf2, ic, i1) = ARRAY2D(icf2, (jpiv1 - 1), i1);
	ARRAY2D(icf2, (jpiv1 - 1), i1) = itmp;
	itmp = ARRAY2D(icf1, ic, i2);
	ARRAY2D(icf1, ic, i2) = ARRAY2D(icf1, (jpiv1 - 1), i2);
	ARRAY2D(icf1, (jpiv1 - 1), i2) = itmp;
	/* Swapping */
	for (l = 0; l < *nov; ++l) {
	  tmp = s1[i1][ic][l];
	  s1[i1][ic][l] = s1[i1][ipiv1 - 1][l];
	  s1[i1][ipiv1 - 1][l] = tmp;
	  if (l >= ic) {
	    tmp = a2[i1][ic][ARRAY2D(icf2, l, i1) - 1];
	    a2[i1][ic][ARRAY2D(icf2, l, i1) - 1] = 
	      a2[i1][ipiv1 - 1][ARRAY2D(icf2, l, i1) - 1];
	    a2[i1][ipiv1 - 1][ARRAY2D(icf2, l, i1) - 1] = tmp;
	  }
	  tmp = s2[i1][ic][l];
	  s2[i1][ic][l] = s2[i1][ipiv1 - 1][l];
	  s2[i1][ipiv1 - 1][l] = tmp;
	}

	for (l = 0; l < *ncb; ++l) {
	  tmp = bb[i1][ic][l];
	  bb[i1][ic][l] = bb[i1][ipiv1 - 1][l];
	  bb[i1][ipiv1 - 1][l] = tmp;
	}
      } else {
	ARRAY2D(ipr, ic, i1) = *nov + ipiv2;
	itmp = ARRAY2D(icf2, ic, i1);
	ARRAY2D(icf2, ic, i1) = ARRAY2D(icf2, (jpiv2 - 1), i1);
	ARRAY2D(icf2, (jpiv2 - 1), i1) = itmp;
	itmp = ARRAY2D(icf1, ic, i2);
	ARRAY2D(icf1, ic, i2) = ARRAY2D(icf1, (jpiv2 - 1), i2);
	ARRAY2D(icf1, (jpiv2 - 1), i2) = itmp;
	/* Swapping */
	for (l = 0; l < *nov; ++l) {
	  if (l >= ic) {
	    tmp = a2[i1][ic][ARRAY2D(icf2, l, i1) - 1];
	    a2[i1][ic][ARRAY2D(icf2, l, i1) - 1] = 
                a1[i2][ipiv2 - 1][ARRAY2D(icf2, l, i1) - 1];
	    a1[i2][ipiv2 - 1][ARRAY2D(icf2, l, i1) - 1] = tmp;
	  }
	  tmp = s2[i1][ic][l];
	  s2[i1][ic][l] = a2[i2][ipiv2 - 1][l];
	  a2[i2][ipiv2 - 1][l] = tmp;
	  tmp = s1[i1][ic][l];
	  s1[i1][ic][l] = s1[i2][ipiv2 - 1][l];
	  s1[i2][ipiv2 - 1][l] = tmp;
	}
	for (l = 0; l < *ncb; ++l) {
	  tmp = bb[i1][ic][l];
	  bb[i1][ic][l] = bb[i2][ipiv2 - 1][l];
	  bb[i2][ipiv2 - 1][l] = tmp;
	}
      }
      /* End of pivoting; Elimination starts here */

      for (ir = icp1; ir < *nov; ++ir) {
	/*for (ir = *nov - 1; ir >= icp1; ir--) {*/
	rm = a2[i1][ir][ARRAY2D(icf2, ic, i1) - 1] / 
	  a2[i1][ic][ARRAY2D(icf2, ic, i1) - 1];
	a2[i1][ir][ARRAY2D(icf2, ic, i1) - 1] = rm;

	if (rm != (double)0.) {
	  for (l = icp1; l < *nov; ++l) {
	    a2[i1][ir][ARRAY2D(icf2, l, i1) - 1] -= 
	      rm * a2[i1][ic][ARRAY2D(icf2, l, i1) - 1];
	  }

	  for (l = 0; l < *nov; ++l) {
	    s1[i1][ir][l] -= rm * s1[i1][ic][l];
	    s2[i1][ir][l] -= rm * s2[i1][ic][l];
	  }

	  for (l = 0; l < *ncb; ++l) {
	    bb[i1][ir][l] -= rm * bb[i1][ic][l];
	  }
	}
      }

      for (ir = 0; ir < *nov; ++ir) {
	/*for (ir = *nov - 1; ir >= 0; ir--) {*/
	rm = a1[i2][ir][ARRAY2D(icf1, ic, i2) - 1] / 
	  a2[i1][ic][ARRAY2D(icf2, ic, i1) - 1];
	a1[i2][ir][ARRAY2D(icf1, ic, i2) - 1] = rm;

	if (rm != (double)0.) {
	  for (l = icp1; l < *nov; ++l) {
	    a1[i2][ir][ARRAY2D(icf1, l, i2) - 1] -= 
	      rm * a2[i1][ic][ARRAY2D(icf2, l, i1) - 1];
	  }
	  for (l = 0; l < *nov; ++l) {
	    s1[i2][ir][l] -= rm * s1[i1][ic][l];
	    a2[i2][ir][l] -= rm * s2[i1][ic][l];
	  }
	  for (l = 0; l < *ncb; ++l) {
            bb[i2][ir][l] -= rm * bb[i1][ic][l];
	  }
	}
      }

      for (ir = nbcp1 - 1; ir < *nrc; ++ir) {
	/*for (ir = *nrc - 1; ir >= nbcp1 - 1; ir--) {*/
	rm = cc[i2][ir][ARRAY2D(icf2, ic, i1) - 1] / 
	  a2[i1][ic][ARRAY2D(icf2, ic, i1) - 1];
	cc[i2][ir][ARRAY2D(icf2, ic, i1) - 1] = rm;

	if (rm != (double)0.) {
	  for (l = icp1; l < *nov; ++l) {
	    cc[i2][ir][ARRAY2D(icf2, l, i1) - 1] -= 
	      rm * a2[i1][ic][ARRAY2D(icf2, l, i1) - 1];
	  }
	  for (l = 0; l < *nov; ++l) {
	    cc[0][ir][l] -= rm * s1[i1][ic][l];
	    cc[i3][ir][l] -= rm * s2[i1][ic][l];
	  }
	  for (l = 0; l < *ncb; ++l) {
	    dd[ir][l] -= rm * bb[i1][ic][l];
	  }
	}
      }

      /* L2: */
    }
    /* L3: */
  }

  /* Initialization */
  for (i = 0; i < *nov; ++i) {
    ARRAY2D(icf2, i, (*na - 1)) = i + 1;
  }
#ifdef USAGE
  usage_end(mainloop,"reduce mainloop");
#endif    

#ifdef DEBUG
  {
    FILE *icf1_fp,*icf2_fp,*ipr_fp,*s1_fp,*s2_fp;
    FILE *a1_fp,*a2_fp,*bb_fp,*cc_fp,*dd_fp;
    int i,j,k;
    char *prefix="test";
    char filename[80];

    strcpy(filename,prefix);
    strcat(filename,".icf1");
    icf1_fp = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".icf2");
    icf2_fp = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".ipr");
    ipr_fp  = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".s1");
    s1_fp   = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".s2");
    s2_fp   = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".a1");
    a1_fp   = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".a2");
    a2_fp   = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".bb");
    bb_fp   = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".cc");
    cc_fp   = fopen(filename,"w");

    strcpy(filename,prefix);
    strcat(filename,".dd");
    dd_fp   = fopen(filename,"w");

    for (i = 0; i < *na; ++i) {
      for (j = 0; j < *nov; ++j) {
	fprintf(icf1_fp,"%d \n",ARRAY2D(icf1, j, i));
	fprintf(icf2_fp,"%d \n",ARRAY2D(icf2, j, i));
	fprintf(ipr_fp, "%d \n",ARRAY2D(ipr, j, i));
	for (k = 0; k < *nov; ++k) {
	  fprintf(s1_fp,"%d \n",s1[i][j][k]);
	  fprintf(s2_fp,"%d \n",s2[i][j][k]);
	  fprintf(a1_fp,"%d \n",a1[i][j][k]);
	  fprintf(a2_fp,"%d \n",a2[i][j][k]);
	}
	for (k = 0; k < *ncb;k++) {
	  fprintf(bb_fp,"%d \n",bb[i][j][k]);
	}
	for (k = 0; k < *nrc;k++) {
	  fprintf(cc_fp,"%d \n",cc[i][k][j]);
	}
      }
    }
    for(i=0;i < *nrc;i++) {
      for(j=0;j < *ncb;j++) {
	  fprintf(dd_fp,"%d \n",dd[i][j]);
      }
    }
  }
  exit(0);
#endif
  return 0;
} 








