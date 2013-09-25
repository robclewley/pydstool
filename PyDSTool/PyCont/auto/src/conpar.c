#include "auto_f2c.h"
#include "auto_c.h"
#include "auto_types.h"

#ifdef PTHREADS
pthread_mutex_t mutex_for_d = PTHREAD_MUTEX_INITIALIZER;
#endif

/*This is the process function.  It is meant to be called either
  on a SMP using shared memory, or wrapped inside another
  routine for message passing*/

void *conpar_process(void * arg)
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
  struct rusage *conpar_process_usage;
  usage_start(&conpar_process_usage);
#endif

  nov = ((conpar_parallel_arglist *)arg)->nov;
  nra = ((conpar_parallel_arglist *)arg)->nra;
  nca = ((conpar_parallel_arglist *)arg)->nca;
  a = ((conpar_parallel_arglist *)arg)->a;
  ncb = ((conpar_parallel_arglist *)arg)->ncb;
  b = ((conpar_parallel_arglist *)arg)->b;
  nbc = ((conpar_parallel_arglist *)arg)->nbc;
  nrc = ((conpar_parallel_arglist *)arg)->nrc;
  c = ((conpar_parallel_arglist *)arg)->c;
  d = ((conpar_parallel_arglist *)arg)->d;
  irf = ((conpar_parallel_arglist *)arg)->irf;
  icf = ((conpar_parallel_arglist *)arg)->icf;
  loop_start = ((conpar_parallel_arglist *)arg)->loop_start;
  loop_end = ((conpar_parallel_arglist *)arg)->loop_end;

#ifdef PTHREADS
  dlocal=DMATRIX(*nrc, *ncb);
#endif
  /* In the default case we don't need to do anything special */
  if(global_conpar_type == CONPAR_DEFAULT) {
    ;
  }
  /* In the message passing case we set d to be
     0.0, do a sum here, and then do the final
     sum (with the true copy of d) in the
     master */
  else if (global_conpar_type == CONPAR_MPI) {
    for(i=0;i<*nrc;i++)
      for (j=0; j<*ncb;j++)
          d[i][j]=0.0;
  }
  /* In the shared memory case we create a local
     variable for doing this threads part of the
     sum, then we do a final sum into shared memory
     at the end */
  else if (global_conpar_type == CONPAR_PTHREADS) {
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
	  a_offset1[-1 + icf_ic_i] = rm;
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
	      if(global_conpar_type == CONPAR_DEFAULT) {
		d_offset1[l] -= rm * b_offset2[l];
	      }
	      /* In the message passing case we sum into d,
		 which is a local variable initialized to 0.0.
		 We then sum our part with the masters part
		 in the master. */
	      else if (global_conpar_type == CONPAR_MPI) {
		d_offset1[l] -= rm * b_offset2[l];
	      }
	      /* In the shared memory case we sum into a local
		 variable our contribution, and then sum
		 into shared memory at the end (inside a mutex */
	      else if (global_conpar_type == CONPAR_PTHREADS) {
#ifdef PTHREADS
		dlocal[-1 + ir][l] -= rm * b_offset2[l];

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
  if(global_conpar_type == CONPAR_PTHREADS) {
#ifdef PTHREADS
    pthread_mutex_lock(&mutex_for_d);
    for(i=0;i<*nrc;i++)
      for (j=0; j<*ncb;j++)
        d[i][j] += dlocal[i][j];
    pthread_mutex_unlock(&mutex_for_d);
    FREE_DMATRIX(dlocal);
#else
    ;
#endif
  }
#endif
#ifdef USAGE
  usage_end(conpar_process_usage,"in conpar worker");
#endif
  return NULL;
}

#ifdef PTHREADS
int conpar_threads_wrapper(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *ncb, 
			   doublereal ***b, integer *nbc, integer *nrc, doublereal ***c, doublereal **d, 
			   integer *irf, integer *icf)

{
  /* Aliases for the dimensions of the arrays */
  integer icf_dim1, irf_dim1;
  
  conpar_parallel_arglist *data;
  int i;
  pthread_t *th;
  void * retval;
  pthread_attr_t attr;
  int retcode;
#ifdef USAGE
  struct timeval *pthreads_wait;
  time_start(&pthreads_wait);
#endif
  
  data = (conpar_parallel_arglist *)MALLOC(sizeof(conpar_parallel_arglist)*global_num_procs);
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
    retcode = pthread_create(&th[i], &attr, conpar_process, (void *) &data[i]);
    if (retcode != 0) fprintf(stderr, "create %d failed %d\n", i, retcode);
  }
  
  for(i=0;i<global_num_procs;i++) {
    retcode = pthread_join(th[i], &retval);
    if (retcode != 0) fprintf(stderr, "join %d failed %d\n", i, retcode);
  }  
  FREE(data);
  FREE(th);
#ifdef USAGE
  time_end(pthreads_wait,"conpar pthreads wrapper",fp9);
#endif
  return 0;
}
#endif

#ifdef MPI
int 
conpar_mpi_wrapper(integer *nov, integer *na, integer *nra, 
		   integer *nca, doublereal ***a, integer *ncb, 
		   doublereal ***b, integer *nbc, integer *nrc, 
		   doublereal ***c, doublereal *d, integer *irf, integer *icf)

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
      
      /*Send message to get worker into conpar mode*/
      {
	int message=AUTO_MPI_CONPAR_MESSAGE;
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
conpar_default_wrapper(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *ncb, doublereal ***b, integer *nbc, integer *nrc, doublereal ***c, doublereal **d, integer *irf, integer *icf)

{
    conpar_parallel_arglist data;
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
    conpar_process(&data);
    return 0;
}


int 
conpar(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *ncb, doublereal ***b, integer *nbc, integer *nrc, doublereal ***c, doublereal **d, integer *irf, integer *icf)
{
  /* Aliases for the dimensions of the arrays */
  integer icf_dim1, irf_dim1;
  
  /* Local variables */
  integer i,j;
  integer nex;

  /*<	      NEX=NCA-2*NOV >*/
  irf_dim1 = *nra;
  icf_dim1 = *nca;

  /* Function Body */
  nex = *nca - (*nov << 1);
  if (nex == 0) {
    return 0;
  }
  
  /*     Initialization */
  for (i = 0; i <*na; ++i) {
    for (j = 0; j < *nra; ++j) {
      irf[j + i * irf_dim1] = j+1;
    }
    for (j = 0; j < *nca; ++j) {
      icf[j + i * icf_dim1] = j+1;
    }
  }

  switch(global_conpar_type) {
#ifdef PTHREADS
  case CONPAR_PTHREADS:
    conpar_threads_wrapper(nov, na, nra, nca, a, 
			    ncb, b, nbc, nrc, c, d,irf, icf);
    break;
#endif
#ifdef MPI
  case CONPAR_MPI:
    if(global_verbose_flag)
      printf("MPI conpar start\n");
    conpar_mpi_wrapper(nov, na, nra, nca, a, 
			ncb, b, nbc, nrc, c, d,irf, icf);
    if(global_verbose_flag)
      printf("MPI conpar end\n");
    break;
#endif
  default:
    conpar_default_wrapper(nov, na, nra, nca, a, 
			    ncb, b, nbc, nrc, c, d,irf, icf);
    break;
  }
  return 0;
} 








