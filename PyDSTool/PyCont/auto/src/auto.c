#include "auto_f2c.h"
#include "auto_c.h"
#include "assert.h"

FILE *fp9;
FILE *fp12;
int global_conpar_type=CONPAR_DEFAULT;
int global_setubv_type=SETUBV_DEFAULT;
int global_reduce_type=REDUCE_DEFAULT;
int global_num_procs=1;
int global_verbose_flag=0;

AutoData *gData = NULL;

#ifdef FLOATING_POINT_TRAP
#include <fpu_control.h>
/* This is a x86 specific function only used for debugging.
   It turns on various floating point traps.  */
static int trapfpe()
{
  fpu_control_t traps;
  traps = _FPU_DEFAULT & (~(_FPU_MASK_IM | _FPU_MASK_ZM | _FPU_MASK_OM));
  _FPU_SETCW(traps);
}
#endif


/*************************
*** MAIN AUTO ENTRANCE ***
**************************/

int AUTO(AutoData *Data)
{
  struct timeval  *time0,*time1;
  integer icp[NPARX2];
  doublereal par[NPARX2], thl[NPARX];
  iap_type *iap;
  rap_type *rap;
  doublereal *thu;
  integer *iuz;
  doublereal *vuz;
  function_list list;

  integer i, j, k;

  // Initialize structures and constants
  gData = Data;

  iap = &(Data->iap);
  rap = &(Data->rap);

  Data->sp_len = Data->num_sp + (1 + floor(iap->nmx/iap->npr));
  Data->sp_inc = 5;

#ifdef USAGE
  struct rusage *init_usage,*total_usage;
  usage_start(&init_usage);
  usage_start(&total_usage);
#endif

#ifdef FLOATING_POINT_TRAP
  trapfpe();
#endif

#ifdef PTHREADS
  global_conpar_type = CONPAR_PTHREADS;
  global_setubv_type = SETUBV_PTHREADS;
  global_reduce_type = REDUCE_PTHREADS;
#endif

  fp9 = fopen("fort.9","w");
  if(fp9 == NULL) {
    fprintf(stderr,"Error:  Could not open fort.9\n");
    exit(1);
  }


  /* Initialization : */

  iap->mynode = mynode();
  iap->numnodes = numnodes();
  if (iap->numnodes > 1) {
    iap->parallel_flag = 1;
  } else {
    iap->parallel_flag = 0;
  }


    /* NOTE:  thu is allocated inside this function, and the
       pointer is passed back.  I know this is ugly, but
       this function does a bit of work to get thu setup correctly,
       as well as figuring out the size the array should be.
       What really should happen is to have one function which
       reads fort.2 and another fuction which initializes the array.
       That way the allocation could happen between the two calls.
    */
    init0(iap, rap, par, icp, thl, &thu, &iuz, &vuz);

    /* Find restart label and determine type of restart point. */
    if (iap->irs > 0) {
      logical found = FALSE_;

      findlb(iap, rap, iap->irs, &(iap->nfpr), &found);
      if (! found) {
	if (iap->mynode == 0) {
	  fprintf(stderr,"\nRestart label %4ld not found\n",iap->irs);
	}
	exit(0);
      }
    }
    set_function_pointers(*iap,&list);
    init1(iap, rap, icp, par);
    chdim(iap);

    /* Create the allocations for the global structures used in
       autlib3.c and autlib5.c.  These are purely an efficiency thing.
       The allocation and deallocation of these scratch areas takes
       up a nontrivial amount of time if done directly in the
       wrapper functions in autlib3.c*/
    allocate_global_memory(*iap);

    /* ---------------------------------------------------------- */
    /* ---------------------------------------------------------- */
    /*  One-parameter continuations */
    /* ---------------------------------------------------------- */
    /* ---------------------------------------------------------- */

#ifdef USAGE
    usage_end(init_usage,"main initialization");
#endif

    if (Data->print_input)
        PrintInput(Data, par, icp);

    // Initialize output variables
    if(list.type==AUTOAE)
        Data->u = DMATRIX(iap->nmx, iap->ndim);
    else {
        // Solution measures
        Data->usm = (doublereal ***)MALLOC((2+(int)(log2(Data->nsm)))*sizeof(doublereal **));
        Data->usm[0] = DMATRIX(iap->nmx, iap->ndim);    // MAX
        Data->usm[1] = DMATRIX(iap->nmx, iap->ndim);    // MIN
        for (i=0; i<(int)(log2(Data->nsm)); i++)
            Data->usm[2+i] = DMATRIX(iap->nmx, iap->ndim);

        // Jacobian of flow
        if (Data->sjac) {
            Data->c0 = DMATRIX_3D(iap->nmx, iap->ndim, iap->ndim);
            Data->c1 = DMATRIX_3D(iap->nmx, iap->ndim, iap->ndim);
        }

        // Jacobian of flow along cycles (temporary storage)
        if (Data->sflow) {
            Data->a1 = DMATRIX_3D(iap->ntst, iap->ndim, iap->ndim);
            Data->a2 = DMATRIX_3D(iap->ntst, iap->ndim, iap->ndim);
        }

        // Number of iterations
        if (Data->snit) {
            Data->nit = (integer *)MALLOC(iap->nmx*sizeof(integer));
        }
    }
    Data->par = DMATRIX(iap->nmx, iap->nicp);
    if (iap->isp >= 1) {
        Data->ev = DCMATRIX(iap->nmx, iap->ndim);
        for (i=0; i<iap->nmx; i++) {
            for (j=0; j<iap->ndim; j++) {
                Data->ev[i][j].r = NAN; // This is a flag for bad floquet multipliers
                Data->ev[i][j].i = NAN;
            }
        }
    }
    Data->num_u = 0;

    if (Data->sp == NULL)
        Data->num_sp = 0;

    Data->sp = (AutoSPData *)REALLOC(Data->sp, (Data->sp_len)*sizeof(AutoSPData));

    for (i=Data->num_sp; i<Data->sp_len; i++) {
        Data->sp[i].u = NULL;
        Data->sp[i].icp = NULL;
        Data->sp[i].ups = NULL;
        Data->sp[i].udotps = NULL;
        Data->sp[i].rldot = NULL;
        Data->sp[i].a1 = NULL;
        Data->sp[i].a2 = NULL;
    }

    if(list.type==AUTOAE)
      autoae(iap, rap, par, icp, list.aelist.funi, list.aelist.stpnt, list.aelist.pvli, thl, thu, iuz, vuz);
    if(list.type==AUTOBV)
      autobv(iap, rap, par, icp, list.bvlist.funi, list.bvlist.bcni,
	     list.bvlist.icni, list.bvlist.stpnt, list.bvlist.pvli, thl, thu, iuz, vuz);

    // Testing output
    if (Data->print_output)
        PrintOutput(Data);

#ifdef USAGE
    usage_end(total_usage,"total");

#endif
    //time_end(time0,"Total Time ",fp9);
    fprintf(fp9,"----------------------------------------------");
    fprintf(fp9,"----------------------------------------------\n");
    //time_end(time1,"",stdout);


  //}
  FREE(thu);
  FREE(iuz);
  FREE(vuz);
  fclose(fp9);

  // Clean up special solution points that were allocated and not used
  Data->sp = (AutoSPData *)REALLOC(Data->sp, (Data->num_sp)*sizeof(AutoSPData));
  assert(Data->sp);
  Data->sp_len = Data->num_sp;

  return 1;
}


/************
*** Print ***
*************/

int PrintInput(AutoData *Data, doublereal *par, integer *icp) {
    integer i;

    fprintf(stdout,"*******************************\n");
    fprintf(stdout,"********* INIT (IAP) **********\n");
    fprintf(stdout,"*******************************\n\n");

    fprintf(stdout,"Data->iap.ndim = %d\n", Data->iap.ndim);
    fprintf(stdout,"Data->iap.ips = %d\n", Data->iap.ips);
    fprintf(stdout,"Data->iap.irs = %d\n", Data->iap.irs);
    fprintf(stdout,"Data->iap.ilp = %d\n", Data->iap.ilp);
    fprintf(stdout,"Data->iap.ntst = %d\n", Data->iap.ntst);
    fprintf(stdout,"Data->iap.ncol = %d\n", Data->iap.ncol);
    fprintf(stdout,"Data->iap.iad = %d\n", Data->iap.iad);
    fprintf(stdout,"Data->iap.iads = %d\n", Data->iap.iads);
    fprintf(stdout,"Data->iap.isp = %d\n", Data->iap.isp);
    fprintf(stdout,"Data->iap.isw = %d\n", Data->iap.isw);
    fprintf(stdout,"Data->iap.iplt = %d\n", Data->iap.iplt);
    fprintf(stdout,"Data->iap.nbc = %d\n", Data->iap.nbc);
    fprintf(stdout,"Data->iap.nint = %d\n", Data->iap.nint);
    fprintf(stdout,"Data->iap.nmx = %d\n", Data->iap.nmx);
    fprintf(stdout,"Data->iap.nuzr = %d\n", Data->iap.nuzr);
    fprintf(stdout,"Data->iap.npr = %d\n", Data->iap.npr);
    fprintf(stdout,"Data->iap.mxbf = %d\n", Data->iap.mxbf);
    fprintf(stdout,"Data->iap.iid = %d\n", Data->iap.iid);
    fprintf(stdout,"Data->iap.itmx = %d\n", Data->iap.itmx);
    fprintf(stdout,"Data->iap.itnw = %d\n", Data->iap.itnw);
    fprintf(stdout,"Data->iap.nwtn = %d\n", Data->iap.nwtn);
    fprintf(stdout,"Data->iap.jac = %d\n", Data->iap.jac);
    fprintf(stdout,"Data->iap.ndm = %d\n", Data->iap.ndm);
    fprintf(stdout,"Data->iap.nbc0 = %d\n", Data->iap.nbc0);
    fprintf(stdout,"Data->iap.nnt0 = %d\n", Data->iap.nnt0);
    fprintf(stdout,"Data->iap.iuzr = %d\n", Data->iap.iuzr);
    fprintf(stdout,"Data->iap.itp = %d\n", Data->iap.itp);
    fprintf(stdout,"Data->iap.itpst = %d\n", Data->iap.itpst);
    fprintf(stdout,"Data->iap.nfpr = %d\n", Data->iap.nfpr);
    fprintf(stdout,"Data->iap.ibr = %d\n", Data->iap.ibr);
    fprintf(stdout,"Data->iap.nit = %d\n", Data->iap.nit);
    fprintf(stdout,"Data->iap.ntot = %d\n", Data->iap.ntot);
    fprintf(stdout,"Data->iap.nins = %d\n", Data->iap.nins);
    fprintf(stdout,"Data->iap.istop = %d\n", Data->iap.istop);
    fprintf(stdout,"Data->iap.nbif = %d\n", Data->iap.nbif);
    fprintf(stdout,"Data->iap.ipos = %d\n", Data->iap.ipos);
    fprintf(stdout,"Data->iap.lab = %d\n", Data->iap.lab);
    fprintf(stdout,"Data->iap.nicp = %d\n", Data->iap.nicp);

    fprintf(stdout,"********** END (IAP) ***********\n\n");

    fprintf(stdout,"*******************************\n");
    fprintf(stdout,"********* INIT (RAP) **********\n");
    fprintf(stdout,"*******************************\n\n");

    fprintf(stdout,"Data->rap.ds = %lf\n", Data->rap.ds);
    fprintf(stdout,"Data->rap.dsmin = %lf\n", Data->rap.dsmin);
    fprintf(stdout,"Data->rap.dsmax = %lf\n", Data->rap.dsmax);
    fprintf(stdout,"Data->rap.dsold = %lf\n", Data->rap.dsold);
    fprintf(stdout,"Data->rap.rl0 = %lf\n", Data->rap.rl0);
    fprintf(stdout,"Data->rap.rl1 = %lf\n", Data->rap.rl1);
    fprintf(stdout,"Data->rap.a0 = %lf\n", Data->rap.a0);
    fprintf(stdout,"Data->rap.a1 = %lf\n", Data->rap.a1);
    fprintf(stdout,"Data->rap.amp = %lf\n", Data->rap.amp);
    fprintf(stdout,"Data->rap.epsl = %lf\n", Data->rap.epsl);
    fprintf(stdout,"Data->rap.epsu = %lf\n", Data->rap.epsu);
    fprintf(stdout,"Data->rap.epss = %lf\n", Data->rap.epss);
    fprintf(stdout,"Data->rap.det = %lf\n", Data->rap.det);
    fprintf(stdout,"Data->rap.tivp = %lf\n", Data->rap.tivp);
    fprintf(stdout,"Data->rap.fldf = %lf\n", Data->rap.fldf);
    fprintf(stdout,"Data->rap.hbff = %lf\n", Data->rap.hbff);
    fprintf(stdout,"Data->rap.biff = %lf\n", Data->rap.biff);
    fprintf(stdout,"Data->rap.spbf = %lf\n", Data->rap.spbf);

    fprintf(stdout,"********** END (RAP) ***********\n\n");

    fprintf(stdout,"*******************************\n");
    fprintf(stdout,"********* INIT (PAR) **********\n");
    fprintf(stdout,"*******************************\n\n");

    if (par != NULL)
        for (i=0; i<2*NPARX; i++)
          fprintf(stdout,"%d ", par[i]);
    fprintf(stdout,"\n");

    fprintf(stdout,"********** END (PAR) ***********\n\n");


    fprintf(stdout,"*******************************\n");
    fprintf(stdout,"********* INIT (ICP) **********\n");
    fprintf(stdout,"*******************************\n\n");

    if (icp != NULL)
        for (i=0; i<Data->iap.nicp; i++)
          fprintf(stdout,"%d ", icp[i]);
    fprintf(stdout,"\n");

    fprintf(stdout,"********** END (ICP) ***********\n\n");

    return 1;
}

int PrintOutput(AutoData *Data) {
    iap_type *iap;
    rap_type *rap;
    integer i, j, k;

    iap = &(Data->iap);
    rap = &(Data->rap);

    printf("NUM POINTS = %d\n", Data->num_u);
    if (Data->u != NULL) {
        for (i=0; i<Data->num_u; i++) {
         for (j=0; j<iap->ndim; j++)
             printf("%14.6E ",Data->u[i][j]);
         for (j=0; j<iap->nicp; j++)
             printf("%14.6E ",Data->par[i][j]);
         printf("\n");
        }
    }

    if (Data->usm != NULL) {
        if (Data->usm[0] != NULL) {
            printf("MAX:\n");
            for (i=0; i<Data->num_u; i++) {
             for (j=0; j<iap->ndim; j++)
                 printf("%14.6E ",Data->usm[0][i][j]);
            }
        }
        if (Data->usm[1] != NULL) {
            printf("MIN:\n");
            for (i=0; i<Data->num_u; i++) {
             for (j=0; j<iap->ndim; j++)
                 printf("%14.6E ",Data->usm[1][i][j]);
            }
        }
        printf("PAR:\n");
        for (i=0; i<Data->num_u; i++) {
            for (j=0; j<iap->nicp; j++)
                printf("%14.6E ",Data->par[i][j]);
            printf("\n");
        }
    }

    printf("SPECIAL POINTS = %d\n", Data->num_sp);
    for (i=0; i<Data->num_sp; i++) {
        printf("%5ld",Data->sp[i].ibr);
        printf("%5ld",Data->sp[i].mtot);
        printf("%5ld",Data->sp[i].itp);
        printf("%5ld",Data->sp[i].lab);
        printf("%5ld",Data->sp[i].nfpr);
        printf("%5ld",Data->sp[i].isw);
        printf("%5ld",Data->sp[i].ntpl);
        printf("%5ld",Data->sp[i].nar);
        printf("%5ld",Data->sp[i].nrowpr);
        printf("%5ld",Data->sp[i].ntst);
        printf("%5ld",Data->sp[i].ncol);
        printf("%5ld",Data->sp[i].nparx);
        printf("\n");

        for (j=0; j<Data->sp[i].nar; j++) {
         printf("%14.6E ",Data->sp[i].u[j]);
        }
        printf("\n");
        if (Data->sp[i].icp != NULL) {
         for (j=0; j<Data->sp[i].nfpr; j++) {
             printf("%14.6E ",Data->sp[i].par[Data->sp[i].icp[j]]);
         }
        } else {
         for (j=0; j<Data->sp[i].nparx; j++) {
             printf("%14.6E ",Data->sp[i].par[j]);
         }
        }
        printf("\n");

        if (Data->sp[i].ups != NULL) {
            printf("%d %d %d %d \n",i,Data->sp[i].ncol, Data->sp[i].ntst, Data->sp[i].nar);
            for (j=0; j<Data->sp[i].ncol*Data->sp[i].ntst+1; j++) {
                for (k=0; k<Data->sp[i].nar; k++)
                    printf("%19.10E ",Data->sp[i].ups[j][k]);
                printf("\n");
            }
        }
        printf("\n");
        if (Data->sp[i].rldot != NULL) {
         for (j=0; j<Data->sp[i].nfpr; j++)
             printf("%19.10E ",Data->sp[i].rldot[j]);
        }
        printf("\n\n");
        if (Data->sp[i].udotps != NULL) {
         for (j=0; j<Data->sp[i].ncol*Data->sp[i].ntst+1; j++) {
             for (k=0; k<Data->sp[i].nar-1; k++) {
                 printf("%19.10E ",Data->sp[i].udotps[j][k]);
             }
             printf("\n");
         }
        }
        printf("\n");
    }

    return 1;
}

/************
*** Setup ***
*************/


int BlankData(AutoData *Data) {
    Data->iap.ndim = 0;
    Data->iap.ips = 0;
    Data->iap.irs = 0;
    Data->iap.ilp = 0;
    Data->iap.ntst = 0;
    Data->iap.ncol = 0;
    Data->iap.iad = 0;
    Data->iap.iads = 0;
    Data->iap.isp = 0;
    Data->iap.isw = 0;
    Data->iap.iplt = 0;
    Data->iap.nbc = 0;
    Data->iap.nint = 0;
#ifdef MANIFOLD
    Data->nalc = 0;
#endif
    Data->iap.nmx = 0;
    Data->iap.nuzr = 0;
    Data->iap.npr = 0;
    Data->iap.mxbf = 0;
    Data->iap.iid = 0;
    Data->iap.itmx = 0;
    Data->iap.itnw = 0;
    Data->iap.nwtn = 0;
    Data->iap.jac = 0;
    Data->iap.ndm = 0;
    Data->iap.nbc0 = 0;
    Data->iap.nnt0 = 0;
    Data->iap.iuzr = 0;
    Data->iap.itp = 0;
    Data->iap.itpst = 0;
    Data->iap.nfpr = 0;
    Data->iap.ibr = 0;
    Data->iap.nit = 0;
    Data->iap.ntot = 0;
    Data->iap.nins = 0;
    Data->iap.istop = 0;
    Data->iap.nbif = 0;
    Data->iap.ipos = 0;
    Data->iap.lab = 0;
    Data->iap.nicp = 0;
    Data->iap.mynode = 0;
    Data->iap.numnodes = 0;
    Data->iap.parallel_flag = 0;

    Data->rap.ds = 0.0;
    Data->rap.dsmin = 0.0;
    Data->rap.dsmax = 0.0;
    Data->rap.dsold = 0.0;
    Data->rap.rl0 = 0.0;
    Data->rap.rl1 = 0.0;
    Data->rap.a0 = 0.0;
    Data->rap.a1 = 0.0;
    Data->rap.amp = 0.0;
    Data->rap.epsl = 0.0;
    Data->rap.epsu = 0.0;
    Data->rap.epss = 0.0;
    Data->rap.det = 0.0;
    Data->rap.tivp = 0.0;
    Data->rap.fldf = 0.0;
    Data->rap.hbff = 0.0;
    Data->rap.biff = 0.0;
    Data->rap.spbf = 0.0;

    Data->icp = NULL;

    Data->nthl = 0;
    Data->ithl = NULL;
    Data->thl = NULL;

    Data->nthu = 0;
    Data->ithu = NULL;
    Data->thu = NULL;

    Data->iuz = NULL;
    Data->vuz = NULL;

    Data->num_sp = 0;
    Data->sp = NULL;

    Data->num_u = 0;
    Data->nsm = 0;
    Data->sjac = 0;
    Data->sflow = 0;
    Data->snit = 0;
    Data->u = NULL;
    Data->usm = NULL;
    Data->par = NULL;
    Data->ev = NULL;
    Data->c0 = NULL;
    Data->c1 = NULL;
    Data->a1 = NULL;
    Data->a2 = NULL;
    Data->nit = NULL;

    Data->npar = 0;
    Data->sp_len = 0;
    Data->sp_inc = 0;
    Data->sp_ind = 0;

    Data->print_input = 0;
    Data->print_output = 0;
    Data->verbosity = 0;
}

int DefaultData(AutoData *Data)
{
    integer i;

    // Setup parameters
    Data->iap.ndim = 2;
    Data->iap.ips = 1;
    Data->iap.irs = 1;
    Data->iap.ilp = 1;
    Data->iap.ntst = 50;
    Data->iap.ncol = 4;
    Data->iap.iad = 3;
    Data->iap.iads = 1;
    Data->iap.isp = 1;
    Data->iap.isw = 1;
    Data->iap.iplt = 0;
    Data->iap.nbc = 0;
    Data->iap.nint = 0;
    Data->iap.nmx = 100;
    Data->iap.nuzr = 0;
    Data->iap.npr = 100;
    Data->iap.mxbf = 0;
    Data->iap.iid = 2;
    Data->iap.itmx = 8;
    Data->iap.itnw = 5;

    Data->iap.nwtn = 3;

    Data->iap.jac = 0;

    Data->rap.ds = 0.010000;
    Data->rap.dsmin = 0.005000;
    Data->rap.dsmax = 0.050000;
    Data->rap.rl0 = -Inf;
    Data->rap.rl1 = Inf;
    Data->rap.a0 = -Inf;
    Data->rap.a1 = Inf;
    Data->rap.epsl = 0.000001;
    Data->rap.epsu = 0.000001;
    Data->rap.epss = 0.000100;

    Data->verbosity = 1;
    Data->nsm = 1;

    if ((Data->icp == NULL) && (Data->iap.nicp == 0)) {
        Data->iap.nicp = 1;
        Data->icp = (integer *)MALLOC(Data->iap.nicp*sizeof(integer));
        Data->icp[0] = 0;
    }

    if ((Data->ithl == NULL) && (Data->thl == NULL) && (Data->nthl == 0)) {
        Data->nthl = 1;
        Data->ithl = (integer *)MALLOC(Data->nthl*sizeof(integer));
        Data->ithl[0] = 10;
        Data->thl = (doublereal *)MALLOC(Data->nthl*sizeof(doublereal));
        Data->thl[0] = 0.;
    }

    // TEMPORARY (use for vanderPol4)
    /* Data->iap.nuzr = 4;
    Data->iuz = (integer *)MALLOC(Data->iap.nuzr*sizeof(integer));
    Data->vuz = (doublereal *)MALLOC(Data->iap.nuzr*sizeof(doublereal));
    for (i=0; i<Data->iap.nuzr; i++) {
        Data->iuz[i] = 1;
        Data->vuz[i] = pow(10,-1*(i+1));
    } */
}

/* Uses Data info to initialize certain aspects of special point, so make sure Data is initialized and
populated with proper parameter values before calling this function. */
int CreateSpecialPoint(AutoData *Data, integer itp, integer lab, doublereal *u,
                       integer npar, integer *ipar, doublereal *par, integer *icp,
                       doublereal *ups, doublereal *udotps, doublereal *rldot) {
    integer i, j, k, nsp;

    nsp = Data->num_sp;
    if (nsp == Data->sp_len) {
        Data->sp_len += 1;
        Data->sp = (AutoSPData *)REALLOC(Data->sp, Data->sp_len*sizeof(AutoSPData));
        Data->sp[nsp].u = (doublereal *)MALLOC((Data->iap.ndim+1)*sizeof(doublereal));
        Data->sp[nsp].ups = NULL;
        Data->sp[nsp].udotps = NULL;
        Data->sp[nsp].rldot = NULL;
        Data->sp[nsp].icp = NULL;
        Data->sp[nsp].a1 = NULL;
        Data->sp[nsp].a2 = NULL;
    }

    Data->sp[nsp].ibr = 1;
    Data->sp[nsp].mtot = 1;
    Data->sp[nsp].itp = itp;
    Data->sp[nsp].lab = lab;
    Data->sp[nsp].nfpr = Data->iap.nicp;
    Data->sp[nsp].isw = Data->iap.isw;
    Data->sp[nsp].ntpl = Data->iap.ncol*Data->iap.ntst+1;
    Data->sp[nsp].nar = Data->iap.ndim+1;
    Data->sp[nsp].nrowpr = 0;
    Data->sp[nsp].ntst = Data->iap.ntst;
    Data->sp[nsp].ncol = Data->iap.ncol;
    Data->sp[nsp].nparx = NPARX;

    Data->npar = npar-1;      // Number of parameters in model (subtract period T)

    // u
    for (i=0; i<Data->iap.ndim+1; i++)
        Data->sp[nsp].u[i] = u[i];

    // par
    for (i=0; i<NPARX; i++)
        Data->sp[nsp].par[i] = 0.0;
    for (i=0; i<npar; i++)
        Data->sp[nsp].par[ipar[i]] = par[i];

    // icp
    if (icp != NULL) {
        Data->sp[nsp].icp = (integer *)MALLOC(Data->iap.nicp*sizeof(integer));
        for (i=0; i<Data->iap.nicp; i++)
            Data->sp[nsp].icp[i] = icp[i];
    }

    // ups
    if (ups != NULL) {
        Data->sp[nsp].ups = DMATRIX(Data->iap.ncol*Data->iap.ntst+1, Data->iap.ndim+1);
        for (i=0; i<Data->iap.ncol*Data->iap.ntst+1; i++)
            for (j=0; j<Data->iap.ndim+1; j++)
                Data->sp[nsp].ups[i][j] = ups[i*(Data->iap.ndim+1)+j];
    }

    // udotps
    if (udotps != NULL) {
        Data->sp[nsp].udotps = DMATRIX(Data->iap.ncol*Data->iap.ntst+1, Data->iap.ndim);
        for (i=0; i<Data->iap.ncol*Data->iap.ntst+1; i++)
            for (j=0; j<Data->iap.ndim; j++)
                Data->sp[nsp].udotps[i][j] = udotps[i*(Data->iap.ndim)+j];
    }

    // rldot
    if (rldot != NULL) {
        Data->sp[nsp].rldot = (doublereal *)MALLOC((Data->iap.nicp)*sizeof(doublereal));
        for (i=0; i<Data->iap.nicp; i++)
            Data->sp[nsp].rldot[i] = rldot[i];
    }

    // a1 and a2
    if (Data->sflow) {
        Data->sp[nsp].a1 = DMATRIX_3D(Data->iap.ntst, Data->iap.ndim, Data->iap.ndim);
        Data->sp[nsp].a2 = DMATRIX_3D(Data->iap.ntst, Data->iap.ndim, Data->iap.ndim);
    }

    Data->num_sp += 1;
	return 1;
}


/**************
*** CLEANUP ***
***************/


int CleanupParams(AutoData *Data) {
    int result = 1;

    if (Data->icp != NULL) {
        FREE(Data->icp);
        Data->icp = NULL;
        Data->iap.nicp = 0;
    }

    if (Data->ithl != NULL) {
        FREE(Data->ithl);
        Data->ithl = NULL;
        Data->nthl = 0;
    }

    if (Data->thl != NULL) {
        FREE(Data->thl);
        Data->thl = NULL;
        Data->nthl = 0;
    }

    if (Data->ithu != NULL) {
        FREE(Data->ithu);
        Data->ithu = NULL;
        Data->nthu = 0;
    }

    if (Data->thu != NULL) {
        FREE(Data->thu);
        Data->thu = NULL;
        Data->nthu = 0;
    }

    if (Data->iuz != NULL) {
        FREE(Data->iuz);
        Data->iuz = NULL;
    }

    if (Data->vuz != NULL) {
        FREE(Data->vuz);
        Data->vuz = NULL;
    }

    return result;
}

int CleanupSolution(AutoData *Data) {
    int i;
    int result = 1;

    if (Data->u != NULL) {
        FREE_DMATRIX(Data->u);
        Data->u = NULL;
    }

    if (Data->usm != NULL) {
        if (Data->usm[0] != NULL) {
            FREE_DMATRIX(Data->usm[0]);
            Data->usm[0] = NULL;
        }
        if (Data->usm[1] != NULL) {
            FREE_DMATRIX(Data->usm[1]);
            Data->usm[1] = NULL;
        }
        for (i=0; i<(int)(log2(Data->nsm)); i++) {
            FREE_DMATRIX(Data->usm[2+i]);
            Data->usm[2+i] = NULL;
        }
        FREE(Data->usm);
        Data->usm = NULL;
    }

    if (Data->par != NULL) {
        FREE_DMATRIX(Data->par);
        Data->par = NULL;
    }

    if (Data->ev != NULL) {
        FREE_DCMATRIX(Data->ev);
        Data->ev = NULL;
    }

    if (Data->c0 != NULL) {
        FREE_DMATRIX_3D(Data->c0);
        Data->c0 = NULL;
    }

    if (Data->c1 != NULL) {
        FREE_DMATRIX_3D(Data->c1);
        Data->c1 = NULL;
    }

    if (Data->a1 != NULL) {
        FREE_DMATRIX_3D(Data->a1);
        Data->a1 = NULL;
    }

    if (Data->a2 != NULL) {
        FREE_DMATRIX_3D(Data->a2);
        Data->a2 = NULL;
    }

    if (Data->nit != NULL) {
        FREE(Data->nit);
        Data->nit = NULL;
    }

    return result;
}

int CleanupSpecialPoints(AutoData *Data) {
    AutoSPData *SPData = Data->sp;
    int result = 1;
    int i;

    if (SPData != NULL) {
        for (i = 0; i < Data->sp_len; i++) {
            if (SPData[i].icp != NULL) {
                FREE(SPData[i].icp);
                SPData[i].icp = NULL;
            }

            if (SPData[i].u != NULL) {
                FREE(SPData[i].u);
                SPData[i].u = NULL;
            }

            if (SPData[i].rldot != NULL) {
                FREE(SPData[i].rldot);
                SPData[i].rldot = NULL;
            }

            if (SPData[i].ups != NULL) {
                FREE_DMATRIX(SPData[i].ups);
                SPData[i].ups = NULL;
            }

            if (SPData[i].udotps != NULL) {
                FREE_DMATRIX(SPData[i].udotps);
                SPData[i].udotps = NULL;
            }

            if (SPData[i].a1 != NULL) {
                FREE_DMATRIX_3D(SPData[i].a1);
                SPData[i].a1 = NULL;
            }

            if (SPData[i].a2 != NULL) {
                FREE_DMATRIX_3D(SPData[i].a2);
                SPData[i].a2 = NULL;
            }
        }

        FREE(Data->sp);
        Data->sp = NULL;
    }

    return result;
}

int CleanupAll(AutoData *Data) {
    int result = 1;

    if (Data != NULL) {
        result = result && (CleanupParams(Data));
        result = result && (CleanupSolution(Data));
        result = result && (CleanupSpecialPoints(Data));

        FREE(Data);
        Data = NULL;
    }

    return result;
}
