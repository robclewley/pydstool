#include <stdio.h>
#include <unistd.h>
#ifdef PTHREADS
#include <pthread.h>
#endif
#ifdef MPI
#include <signal.h>
#include <mpi.h>
#include "auto_mpi.h"
#endif
#include <string.h>
#include <float.h>
#include <math.h>

#ifndef __AUTO_C_H__
#define __AUTO_C_H__

#define Inf     DBL_MAX

#define NPARX (150) /* allows for 110 parameters */
#define NBIFX (20)
#define KREDO (1)  /*get rid of*/
#define NPARX2 (NPARX*2)  /*get rid of*/
#define HMACH (1.0e-7)
#define RSMALL (1.0e-30)
#define RLARGE (1.0e+30)
#define HMACH1 (HMACH+1.0e0)
#define M1SB (NBIFX) 
#define LEFT (1)
#define RIGHT (2)
#define QZMATZ (.FALSE.)
#define QZEPS1 (0.0E0)
#define HMACHHO (1.0e-13)

extern FILE *fp2;
extern FILE *fp9;
extern FILE *fp10;
extern FILE *fp12;

#define CONPAR_DEFAULT  0
#define CONPAR_PTHREADS 1
#define CONPAR_MPI      2
#define SETUBV_DEFAULT  0
#define SETUBV_PTHREADS 1
#define SETUBV_MPI      2
#define REDUCE_DEFAULT  0
#define REDUCE_PTHREADS 1
#define REDUCE_MPI      2

extern int global_conpar_type;
extern int global_setubv_type;
extern int global_reduce_type;
extern int global_num_procs;
extern int global_verbose_flag;

typedef struct {
  /* 1 */ integer ndim;
  /* 2 */ integer ips;
  /* 3 */ integer irs;
  /* 4 */ integer ilp;
  /* 5 */ integer ntst;
  /* 6 */ integer ncol;
  /* 7 */ integer iad;
  /* 8 */ integer iads;
  /* 9 */ integer isp;
  /* 10 */ integer isw;
  /* 11 */ integer iplt;
  /* 12 */ integer nbc;
  /* 13 */ integer nint;
#ifdef MANIFOLD
  /* 13a*/ integer nalc;    /* The number of arclength constraints (k) */
#endif
  /* 14 */ integer nmx;
  /* 15 */ integer nuzr;
  /* 16 */ integer npr;
  /* 17 */ integer mxbf;
  /* 18 */ integer iid;
  /* 19 */ integer itmx;
  /* 20 */ integer itnw;
  /* 21 */ integer nwtn;
  /* 22 */ integer jac;
  /* 23 */ integer ndm;
  /* 24 */ integer nbc0;
  /* 25 */ integer nnt0;
  /* 26 */ integer iuzr;
  /* 27 */ integer itp;
  /* 28 */ integer itpst;
  /* 29 */ integer nfpr;
  /* 30 */ integer ibr;
  /* 31 */ integer nit;
  /* 32 */ integer ntot;
  /* 33 */ integer nins;
  /* 34 */ integer istop;
  /* 35 */ integer nbif;
  /* 36 */ integer ipos;
  /* 37 */ integer lab;
  /* 41 */ integer nicp;
  /* The following are not set in init_.  
     They have to do with the old parallel version. */
  /* 38 */ integer mynode;
  /* 39 */ integer numnodes;
  /* 40 */ integer parallel_flag;
} iap_type;

typedef struct {
  /* 1 */ doublereal ds;
  /* 2 */ doublereal dsmin;
  /* 3 */ doublereal dsmax;
  /* There is no 4 */
  /* 5 */ doublereal dsold;
  /* 6 */ doublereal rl0;
  /* 7 */ doublereal rl1;
  /* 8 */ doublereal a0;
  /* 9 */ doublereal a1;
  /* 10 */ doublereal amp;
  /* 11 */ doublereal epsl;
  /* 12 */ doublereal epsu;
  /* 13 */ doublereal epss;
  /* 14 */ doublereal det;
  /* 15 */ doublereal tivp;
  /* 16 */ doublereal fldf;
  /* 17 */ doublereal hbff;
  /* 18 */ doublereal biff;
  /* 19 */ doublereal spbf;
} rap_type;

typedef struct {
    integer ibr;    // The index of the branch.
    integer mtot;   // The index of the point (modulo 10000).
    integer itp;    // The type of point.
    integer lab;    // The label of the point.
    integer nfpr;   // The number of free parameters used in the computation.
    integer isw;    // The value of ISW used in the computation.
    integer ntpl;   // The number of points in the time interval [0,1] for which solution values are written.
    integer nar;    // The number of values written per point. (NAR=NDIM+1, since T and U(i), i=1,..,NDIM are written).
    integer nrowpr; // AUTO specific output formatting parameter (not really needed).  Included for completion.
    integer ntst;   // The number of time intervals used in the discretization.
    integer ncol;   // The number of collocation points used.
    integer nparx;  // The dimension of the par array (and the number of values in the parameter block).
    integer *icp;   // Index of free parameters.
    
    doublereal *u;          // State variables (including t in first position, dimension = nar)
    doublereal par[NPARX];  // Parameters.
    // NOTE: I store ups differently than auto.  In auto, all collocation
    // points are stored in one row.  I put them in separate rows (analogous to
    // when auto prints to file).  This explains the wacko indices in code.  May want
    // to change this if code gets out of hand.
    doublereal **ups;       // Cycle.
    doublereal **udotps;    // Cycle branch direction. (dimension ntpl)
    doublereal *rldot;      // ??? (dimension nfpr)
    
    doublereal ***a1,
               ***a2;       // Decomposition of flow maps:  J[i] = a2[i]^(-1)a1[i]
} AutoSPData;

typedef struct {
    /********
    * INPUT *
    *********/
    iap_type iap;       // Auto parameters (type integer)
    rap_type rap;       // Auto parameters (type doublereal)
    integer *icp;       // Free parameters
    integer nthl;       // Number of parameters with pseudo-arclength weight not equal to 1
    integer *ithl;      // Indices of parameters with pseudo-arclength weight not equal to 1
    doublereal *thl;    // Weights of parameters with pseudo-arclength weight not equal to 1
    integer nthu;       // Number of parameters with stepsize weight not equal to 1
    integer *ithu;      // Indices of parameters with stepsize weight not equal to 1
    doublereal *thu;    // Weights of parameters with stepsize weight not equal to 1
    integer *iuz;       // Indices of parameter bounds (used for output, termination, ...)
    doublereal *vuz;    // Bounds on parameters (used for output, termination, ...)

    /*********
    * OUTPUT *
    **********/
    int num_u;              // Number of points along solution branch
    int num_sp;             // Number of special points.
    int nsm;                // Solution measures (nsm = 1 + avg*2 + nm2*4)
    doublereal **u;         // AE: Solution branch (state variables)
    doublereal ***usm;      // BVP: Solution measures (0: max, 1: min, avg, nm2)
    doublereal **par;       // Solution branch (parameters)
    doublecomplex **ev;     // Eigenvalues/Floquet multipliers
    int sjac;               // Flag for saving of jacobian
    doublereal ***c0,       // Decomposition of jacobian:  J = c1^(-1)c0
               ***c1;
    int sflow;              // Flag for saving of flow maps along cycles
    doublereal ***a1,
               ***a2;       // Temporary storage of flow maps
    int snit;               // Flag for saving number of iterations
    int *nit;               // Number of iterations
    AutoSPData *sp;         // Special points

    /************
    * CONSTANTS *
    *************/
    int npar;           // For some strange reason, this isn't stored in AUTO.  What is it, you ask?
                        //      THE NUMBER OF PARAMETERS IN THE MODEL.  :)  (Of course, I could just
                        //      have no idea where it is stored.  Very possible...)
    int sp_len;         // Minimum number of special points (used for labeling of points and plotting
                        //      of cycles.  Value = 2 + floor(nmx/npr), 2 for start and end of curve.
    int sp_inc;         // Increment by which special points are reallocated.
    int sp_ind;         // Index of special point with correct label (set in findlb and used in
                        //      readlb, rsptbv, stpnbv).    
    /********
    * DEBUG *
    *********/
    int print_input;   // If 1, will print all input to stdout
    int print_output;  // If 1, will print all output to stdout
    
    int verbosity;     // If 0, no output to screen.
} AutoData;

extern AutoData *gData;

#ifdef __PYTHON__
#   include <Python.h>
#   define MALLOC(size) (PyMem_Malloc(size))
#   define REALLOC(ptr, size) (PyMem_Realloc(ptr, size))
#   define FREE(ptr) PyMem_Free(ptr)
#else
#   define MALLOC(size) (malloc(size))
#   define REALLOC(ptr, size) (realloc(ptr, size))
#   define FREE(ptr) (free(ptr))
#endif

#ifdef USAGE
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

/*This is the type for all functions which can be used as "funi" the function
  which evaluates the right hand side of the equations and generates
  the Jacobian*/
#define FUNI_TYPE(X) int X(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp)

/*This is the type for all functions which can be used as "bcni" the function
  which evaluates the boundary conditions */
#define BCNI_TYPE(X) int X(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nbc, const doublereal *u0, const doublereal *u1, doublereal *f, integer ijac, doublereal *dbc)

/*This is the type for all functions which can be used as "icni" the function
  which evaluates kernel of the integral constraints */
#define ICNI_TYPE(X) int X(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *f, integer ijac, doublereal *dint)

/*This is the type for all functions which can be used as additional
  output functions for algebraic problems */
#define PVLI_TYPE_AE(X) int X(iap_type *iap, rap_type *rap, doublereal *u, doublereal *par)

/*This is the type for all functions which can be used as additional
  output functions for BVPs */
#define PVLI_TYPE_BVP(X) int X(iap_type *iap, rap_type *rap, integer *icp, doublereal *dtm, integer *ndxloc, doublereal **ups, integer *ndim, doublereal **p0, doublereal **p1, doublereal *par)

/* This is the type for all functions that can be used at starting points
   for algebraic problems */
#define STPNT_TYPE_AE(X) int X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *u)

/* This is the type for all functions that can be used at starting points
   for BVPs */
#define STPNT_TYPE_BVP(X) int X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsrs, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu)

/*This is the type for all functions which can be used to detect
  special points for algebraic problems */
#define FNCS_TYPE_AE(X) doublereal X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, logical *chng, FUNI_TYPE((*funi)), integer *m1aaloc, doublereal **aa, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *uold, doublereal *udot, doublereal *rhs, doublereal *dfdu, doublereal *dfdp, integer *iuz, doublereal *vuz)

/*This is the type for all functions which can be used to detect
  special points for BVPS */
#define FNCS_TYPE_BVP(X) doublereal X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, logical *chng, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), doublereal **p0, doublereal **p1, doublecomplex *ev, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal **dups, doublereal *tm, doublereal *dtm, doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz)

#define AUTOAE 0
#define AUTOBV 1

typedef struct {
  FUNI_TYPE((*funi));
  BCNI_TYPE((*bcni));
  ICNI_TYPE((*icni));
  STPNT_TYPE_BVP((*stpnt));
  PVLI_TYPE_BVP((*pvli));
} autobv_function_list;

typedef struct {
  FUNI_TYPE((*funi));
  STPNT_TYPE_AE((*stpnt));
  PVLI_TYPE_AE((*pvli));
} autoae_function_list;

typedef struct {
  int type;
  autobv_function_list bvlist;
  autoae_function_list aelist;
} function_list;

/* auto.c */
int AUTO(AutoData *Data);
int PrintInput(AutoData *Data, doublereal *par, integer *icp);
int PrintOutput(AutoData *Data);
int BlankData(AutoData *Data);
int DefaultData(AutoData *Data);
int CreateSpecialPoint(AutoData *Data, integer itp, integer lab, doublereal *u, 
                       integer npar, integer *ipar, doublereal *par, integer *icp,
                       doublereal *ups, doublereal *udotps, doublereal *rldot);
int CleanupParams(AutoData *Data);
int CleanupSolution(AutoData *Data);
int CleanupSpecialPoints(AutoData *Data);
int CleanupAll(AutoData *Data);

/* autlib1.c */
void time_start(struct timeval **);
void time_end(struct timeval *, char *, FILE *fp);
#ifdef USAGE
void usage_start(struct rusage **);
void usage_end(struct rusage *, char *);
#endif
void allocate_global_memory(const iap_type);
int init0(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *thl, doublereal **thu_pointer, integer **iuz_pointer, doublereal **vuz_pointer);
int chdim(iap_type *iap);
int autoae(iap_type *iap, rap_type *rap, doublereal *par, 
integer *icp, 
FUNI_TYPE((*funi)), 
STPNT_TYPE_AE((*stpnt)), 
PVLI_TYPE_AE((*pvli)), 
doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);

int autobv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), STPNT_TYPE_BVP((*stpnt)), PVLI_TYPE_BVP((*pvli)), doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
int init1(iap_type *iap, rap_type *rap, integer *icp, doublereal *par);
int cnrlae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), STPNT_TYPE_AE((*stpnt)), PVLI_TYPE_AE((*pvli)), doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
STPNT_TYPE_AE(stpnus);
STPNT_TYPE_AE(stpnae);
int stprae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rds, integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *thl, doublereal *thu);
int contae(iap_type *iap, rap_type *rap, doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *uold, doublereal *udot);
int solvae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rds, integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *thl, doublereal *thu);
int lcspae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FNCS_TYPE_AE((*fncs)), FUNI_TYPE((*funi)), integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *q, doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
int mueller(doublereal *q0, doublereal *q1, doublereal *q, doublereal *s0, doublereal *s1, doublereal *s, doublereal *rds);
FNCS_TYPE_AE(fnbpae);
FNCS_TYPE_AE(fnlpae);
FNCS_TYPE_AE(fnhbae);
FNCS_TYPE_AE(fnuzae);
int stbif(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *m1aaloc, doublereal **aa, integer m1sbloc, doublereal **stud, doublereal **stu, doublereal *stla, doublereal *stld, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *udot, doublereal *dfdu, doublereal *dfdp, doublereal *thl, doublereal *thu);
int swpnt(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, integer m1sbloc, doublereal **stud, doublereal **stu, doublereal *stla, doublereal *stld, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *udot);
int swprc(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *rds, doublereal *thl, doublereal *thu);
int sthd(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *thl, doublereal *thu);
int headng(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer iunit, integer *n1, integer *n2);
int stplae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rlcur, doublereal *u);
int wrline(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *icu, integer *ibr, integer *ntot, integer *lab, doublereal *vaxis, doublereal *u);
int wrtsp8(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *lab, doublereal *rlcur, doublereal *u);
int wrjac(iap_type *iap, integer *n, integer *m1aaloc, doublereal **aa, doublereal *rhs);
int msh(const iap_type *iap, const rap_type *rap, doublereal *tm);
int genwts(const integer ncol, const integer n1, doublereal **wt, doublereal **wp);
int cpnts(const integer ncol, doublereal *zm);
int cntdif(integer *n, doublereal *d);
int wint(const integer n, doublereal *wi);
int adptds(iap_type *iap, rap_type *rap, doublereal *rds);
int adapt(iap_type *iap, rap_type *rap, integer *nold, integer *ncold, integer *nnew, integer *ncnew, doublereal *tm, doublereal *dtm, integer *ndxloc, doublereal **ups, doublereal **vps);
int interp(iap_type *iap, rap_type *rap, integer *ndim, integer *n, integer *nc, doublereal *tm, integer *ndxloc, doublereal **ups, integer *n1, integer *nc1, doublereal *tm1, doublereal **ups1, doublereal *tm2, integer *itm1);
int newmsh(iap_type *iap, rap_type *rap, integer *ndxloc, doublereal **ups, integer *nold, integer *ncold, doublereal *tmold, doublereal *dtmold, integer *nnew, doublereal *tmnew, integer *iper);
int ordr(iap_type *iap, rap_type *rap, integer *n, doublereal *tm, integer *n1, doublereal *tm1, integer *itm1);
int intwts(iap_type *iap, rap_type *rap, integer *n, doublereal *z__, doublereal *x, doublereal *wts);
int eqdf(iap_type *iap, rap_type *rap, integer *ntst, integer *ndim, integer *ncol, doublereal *dtm, integer *ndxloc, doublereal **ups, doublereal *eqf, integer *iper);
int eig(iap_type *iap, integer *ndim, integer *m1a, doublereal *a, doublecomplex *ev, integer *ier);
int nlvc(integer n, integer m, integer k, doublereal **a, doublereal *u);
int nrmlz(integer *ndim, doublereal *v);
doublereal api(doublereal r__);
int ge(integer n, integer m1a, doublereal *a, integer nrhs, integer ndxloc, doublereal *u, integer m1f, doublereal *f, doublereal *det);
int newlab(iap_type *iap, rap_type *rap);
int findlb(iap_type *iap, const rap_type *rap, integer irs, integer *nfpr, logical *found);
int readlb(const iap_type *iap, const rap_type *rap, doublereal *u, doublereal *par);
doublereal rinpr(iap_type *iap, integer *ndim1, integer *ndxloc, doublereal **ups, doublereal **vps, doublereal *dtm, doublereal *thu);
doublereal rnrmsq(iap_type *iap, integer *ndim1, integer *ndxloc, doublereal **ups, doublereal *dtm, doublereal *thu);
doublereal rintg(iap_type *iap, integer *ndxloc, integer ic, doublereal **ups, doublereal *dtm);
doublereal rnrm2(iap_type *iap, integer *ndxloc, integer ic, doublereal **ups, doublereal *dtm);
doublereal rmxups(iap_type *iap, integer *ndxloc, integer i, doublereal **ups);
doublereal rmnups(iap_type *iap, integer *ndxloc, integer i, doublereal **ups);
int scaleb(iap_type *iap, integer *icp, integer *ndxloc, doublereal **dvps, doublereal *rld, doublereal *dtm, doublereal *thl, doublereal *thu);
int cnrlbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), STPNT_TYPE_BVP((*stpnt)), PVLI_TYPE_BVP((*pvli)), doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
int contbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal *dtm, doublereal *thl, doublereal *thu);
int extrbv(iap_type *iap, rap_type *rap, FUNI_TYPE((*funi)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps);
int stupbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **upoldp);
int stepbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), PVLI_TYPE_BVP((*pvli)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal *tm, doublereal *dtm, doublereal **p0, doublereal **p1, doublereal *thl, doublereal *thu);
int rsptbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), STPNT_TYPE_BVP((*stpnt)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *tm, doublereal *dtm, doublecomplex *ev, integer *nodir, doublereal *thl, doublereal *thu);
STPNT_TYPE_BVP(stpnbv);
STPNT_TYPE_BVP(stpnub);
int setrtn(iap_type *iap, integer *ntst, integer *ndxloc, doublereal **ups, doublereal *par);
int stdrbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal *dtm, integer iperp, doublereal **p0, doublereal **p1, doublereal *thl, doublereal *thu);
int lcspbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FNCS_TYPE_BVP((*fncs)), FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), PVLI_TYPE_BVP((*pvli)), doublereal *q, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal *tm, doublereal *dtm, doublereal **p0, doublereal **p1, doublecomplex *ev, doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
FNCS_TYPE_BVP(fnlpbv);
FNCS_TYPE_BVP(fnbpbv);
FNCS_TYPE_BVP(fnspbv);
FNCS_TYPE_BVP(fnuzbv);
int tpspbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublecomplex *ev);
int stplbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **c0, doublereal **c1);
int wrtbv8(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *dtm);
int wrtbv9(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rlcur, integer *ndxloc, doublereal **ups, doublereal *tm, doublereal *dtm, doublereal *thl, doublereal *thu);
PVLI_TYPE_AE(pvlsae);
PVLI_TYPE_BVP(pvlsbv);
int setpae(iap_type *iap, rap_type *rap);
int setpbv(iap_type *iap, rap_type *rap, doublereal *dtm);
int autim0(doublereal *t);
int autim1(doublereal *t);
/* sometimes here ups is just one vector .... *double, and sometimes a **double */
doublereal getp(char *code, integer ic, void *u_or_ups);
int set_function_pointers(const iap_type,function_list *);
/* autlib2.c */
int solvbv(integer *ifst, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), doublereal *rds, integer *nllv, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal *dtm, doublereal **fa, doublereal *fc, doublereal **p0, doublereal **p1, doublereal *thl, doublereal *thu);
int setfcdd(integer *ifst, doublereal **dd, doublereal *fc, integer *ncb, integer *nrc);
int faft(doublereal **ff, doublereal **fa, integer *ntst, integer *nrow, integer *ndxloc);
int partition(integer *n, integer *kwt, integer *m);
integer mypart(integer *iam, integer *np);
#ifndef MANIFOLD
int setrhs(integer *ndim, integer *ips, integer *na, integer *ntst, integer *np, integer *ncol, integer *nbc, integer *nint, integer *ncb, integer *nrc, integer *nra, integer *nca, integer *iam, integer *kwt, logical *ipar, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer *ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, doublereal **fa, doublereal *fc, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#else
int setrhs(integer *ndim, integer *ips, integer *na, integer *ntst, integer *np, integer *ncol, integer *nbc, integer *nint, integer *nalc, integer *ncb, integer *nrc, integer *nra, integer *nca, integer *iam, integer *kwt, logical *ipar, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer *ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, doublereal **fa, doublereal *fc, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#endif
int brbd(doublereal ***a, doublereal ***b, doublereal ***c, doublereal **d, doublereal **fa, doublereal *fc, doublereal **p0, doublereal **p1, integer *ifst, integer *idb, integer *nllv, doublereal *det, integer *nov, integer *na, integer *nbc, integer *nra, integer *nca, integer *ncb, integer *nrc, integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, doublereal **faa, doublereal ***ca1, doublereal ***s1, doublereal ***s2, integer *icf11, integer *ipr, integer *icf1, integer *icf2, integer *irf, integer *icf);
int setzero(doublereal **fa, doublereal *fc, integer *na, integer *nra, integer *nrc);
int conrhs(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *nbc, integer *nrc, doublereal ***c, doublereal **fa, doublereal *fc, integer *irf, integer *icf, integer *iam);
int copycp(integer na, integer nov, integer nra, integer nca, doublereal ***a, integer ncb, doublereal ***b, integer nrc, doublereal ***c, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, integer *irf);
int cpyrhs(integer na, integer nov, integer nra, doublereal **faa, doublereal **fa, integer *irf);
int redrhs(integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***cc, doublereal **faa, doublereal *fc, integer *na, integer *nov, integer *ncb, integer *nrc, doublereal ***ca1, integer *icf1, integer *icf2, integer *icf11, integer *ipr, integer *nbc);
int dimrge(integer *iam, integer *kwt, logical *par, doublereal **e, doublereal ***cc, doublereal **d, doublereal *fc, integer *ifst, integer *na, integer *nrc, integer *nov, integer *ncb, integer *idb, integer *nllv, doublereal *fcc, doublereal **p0, doublereal **p1, doublereal *det, doublereal ***s, doublereal ***a2, doublereal **faa, doublereal ***bb);
int bcksub(integer *iam, integer *kwt, logical *par, doublereal ***s1, doublereal ***s2, doublereal ***a2, doublereal ***bb, doublereal **faa, doublereal *fc, doublereal *fcc, doublereal *sol1, doublereal *sol2, doublereal *sol3, integer *na, integer *nov, integer *ncb, integer *icf2);
int infpar(integer *iam, logical *par, doublereal ***a, doublereal ***b, doublereal **fa, doublereal *sol1, doublereal *sol2, doublereal *fc, integer *na, integer *nov, integer *nra, integer *nca, integer *ncb, integer *irf, integer *icf);
int rd0(integer *iam, integer *kwt, doublereal *d, integer *nrc);
int print1(integer *nov, integer *na, integer *nra, integer *nca, integer *ncb, integer *nrc, doublereal ***a, doublereal ***b, doublereal ***c, doublereal **d, doublereal **fa, doublereal *fc);
integer mynode(void);
integer numnodes(void);
int gsync(void);
doublereal dclock(void);
int csend(void);
int crecv(void);
int gdsum(void);
int gsendx(void);
int gcol(void);
int led(void);
int setiomode(void);
/* autlib3.c */
FUNI_TYPE(fnlp);
int fflp(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnlp);
FUNI_TYPE(fnc1);
STPNT_TYPE_AE(stpnc1);
FUNI_TYPE(fnc2);
int ffc2(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnc2);
FUNI_TYPE(fnds);
FUNI_TYPE(fnti);
FUNI_TYPE(fnhd);
int ffhd(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnhd);
FUNI_TYPE(fnhb);
int ffhb(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnhb);
FUNI_TYPE(fnhw);
int ffhw(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnhw);
FUNI_TYPE(fnps);
BCNI_TYPE(bcps);
ICNI_TYPE(icps);
int pdble(const iap_type *iap, const rap_type *rap, integer *ndim, integer *ntst, integer *ncol, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *par);
STPNT_TYPE_BVP(stpnps);
FUNI_TYPE(fnws);
int ffws(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp, integer ndm, doublereal *dfu, doublereal *dfp);
FUNI_TYPE(fnwp);
int stpnwp(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fnsp);
int ffsp(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp, integer ndm, doublereal *dfu, doublereal *dfp);
FUNI_TYPE(fnpe);
int ffpe(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp, integer ndm, doublereal *dfu, doublereal *dfp);
ICNI_TYPE(icpe);
FUNI_TYPE(fnpl);
int ffpl(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcpl);
ICNI_TYPE(icpl);
int stpnpl(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fnpd);
int ffpd(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcpd);
ICNI_TYPE(icpd);
int stpnpd(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fntr);
int fftr(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
int bctr(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nbc, const doublereal *u0,const  doublereal *u1, doublereal *f, integer ijac, doublereal *dbc);
ICNI_TYPE(ictr);
int stpntr(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fnpo);
int ffpo(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const doublereal *upold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcpo);
ICNI_TYPE(icpo);
int fipo(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, integer nnt0, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *fi, doublereal *dint, integer ndmt, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_BVP(stpnpo);
FUNI_TYPE(fnbl);
int ffbl(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcbl);
int fbbl(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nbc, integer nbc0, const doublereal *u0, const doublereal *u1, doublereal *f, doublereal *dbc);
ICNI_TYPE(icbl);
int fibl(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, integer nnt0, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *f, doublereal *dint);
STPNT_TYPE_BVP(stpnbl);
FUNI_TYPE(funi);
BCNI_TYPE(bcni);
ICNI_TYPE(icni);
int fopi(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp);
/* autlib4.c */
int flowkm(integer ndim, doublereal **c0, doublereal **c1, integer iid, doublecomplex *ev);
int dhhpr(integer k, integer j, integer n, doublereal *x, integer incx, doublereal *beta, doublereal *v);
int dhhap(integer k, integer j, integer n, integer q, doublereal *beta, doublereal *v, integer job, doublereal **a, integer lda);
/* autlib5.c */
FUNI_TYPE(fnho);
int ffho(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu);
BCNI_TYPE(bcho);
int fbho(const iap_type *iap, integer ndim, doublereal *par, const integer *icp, integer nbc, integer nbc0, const doublereal *u0, const doublereal *u1, doublereal *fb);
ICNI_TYPE(icho);
int fiho(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, integer nnt0, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *fi);
int inho(iap_type *iap, integer *icp, doublereal *par);
int preho(iap_type *iap, rap_type *rap, doublereal *par, const integer *icp, integer ndx, integer *ntsr, integer *nar, integer ncolrs, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *dtm);
int stpnho(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
int stpho(iap_type *iap, integer *icp, doublereal *u, doublereal *par, doublereal t);
PVLI_TYPE_BVP(pvlsho);
doublereal psiho(const iap_type *iap, integer is, doublereal **rr, doublereal **ri, doublereal ***v, doublereal ***vt, const integer *icp, doublereal *par, const doublereal *pu0, const doublereal *pu1);
int eigho(integer isign, integer itrans, doublereal *rr, doublereal *ri, doublereal **vret, const doublereal *xequib, const integer *icp, doublereal *par, integer ndm);
int prjctn(doublereal **bound, const doublereal *xequib, const integer *icp, doublereal *par, integer imfd, integer is, integer itrans, integer ndm);
/* eispack.c */
int rg(integer nm, integer n, doublereal *a, doublereal *wr, doublereal *wi, integer matz, doublereal *z__, integer *iv1, doublereal *fv1, integer *ierr);
int hqr(integer *nm, integer *n, integer *low, integer *igh, doublereal *h__, doublereal *wr, doublereal *wi, integer *ierr);
int hqr2(integer *nm, integer *n, integer *low, integer *igh, doublereal *h__, doublereal *wr, doublereal *wi, doublereal *z__, integer *ierr);
int cdiv(doublereal *ar, doublereal *ai, doublereal *br, doublereal *bi, doublereal *cr, doublereal *ci);
int balanc(integer *nm, integer *n, doublereal *a, integer *low, integer *igh, doublereal *scale);
int balbak(integer *nm, integer *n, integer *low, integer *igh, doublereal *scale, integer *m, doublereal *z__);
int elmhes(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, integer *int__);
int eltran(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, integer *int__, doublereal *z__);
int qzhes(integer nm, integer n, doublereal *a, doublereal *b, logical matz, doublereal *z__);
int qzit(integer nm, integer n, doublereal *a, doublereal *b, doublereal eps1, logical matz, doublereal *z__, integer *ierr);
int qzval(integer nm, integer n, doublereal *a, doublereal *b, doublereal *alfr, doublereal *alfi, doublereal *beta, logical matz, doublereal *z__);
doublereal epslon(doublereal x);
doublereal dnrm2(integer *n, doublereal *dx, integer *incx);
doublereal ddot(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy);
int dscal(integer *n, doublereal *da, doublereal *dx, integer *incx);
integer idamax(integer *n, doublereal *dx, integer *incx);
int daxpy(integer *n, doublereal *da, doublereal *dx, integer *incx, doublereal *dy, integer *incy);
int drot(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy, doublereal *c, doublereal *s);
int dswap(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy);
int dgemc(integer *m, integer *n, doublereal *a, integer *lda, doublereal *b, integer *ldb, logical *trans);
int xerbla(char *srname, integer *info, integer srname_len);
logical lsame(char *ca, char *cb, integer ca_len, integer cb_len);
int dgemm(char *transa, char *transb, integer *m, integer *n, integer *k, doublereal *alpha, doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *beta, doublereal *c, integer *ldc, integer transa_len, integer transb_len);
int ezsvd(doublereal *x, integer *ldx, integer *n, integer *p, doublereal *s, doublereal *e, doublereal *u, integer *ldu, doublereal *v, integer *ldv, doublereal *work, integer *job, integer *info, doublereal *tol);
int ndrotg(doublereal *f, doublereal *g, doublereal *cs, doublereal *sn);
int ndsvd(doublereal *x, integer *ldx, integer *n, integer *p, doublereal *s, doublereal *e, doublereal *u, integer *ldu, doublereal *v, integer *ldv, doublereal *work, integer *job, integer *info, integer *maxitr, doublereal *tol, integer *idbg, integer *ifull, integer *kount, integer *kount1, integer *kount2, integer *skip, integer *limshf, doublereal *maxsin, integer *iidir);
int prse(integer *ll, integer *m, integer *nrow, integer *ncol, doublereal *s, doublereal *e);
int sig22(doublereal *a, doublereal *b, doublereal *c, doublereal *sigmin, doublereal *sigmax, doublereal *snr, doublereal *csr, doublereal *snl, doublereal *csl);
doublereal sigmin(doublereal *a, doublereal *b, doublereal *c);
int sndrtg(doublereal *f, doublereal *g, doublereal *cs, doublereal *sn);
int hqr3lc(doublereal *a, doublereal *v, integer *n, integer *nlow, integer *nup, doublereal *eps, doublereal *er, doublereal *ei, integer *type__, integer *na, integer *nv, integer *imfd);
int split(doublereal *a, doublereal *v, integer *n, integer *l, doublereal *e1, doublereal *e2, integer *na, integer *nv);
int exchng(doublereal *a, doublereal *v, integer *n, integer *l, integer *b1, integer *b2, doublereal *eps, logical *fail, integer *na, integer *nv);
int qrstep(doublereal *a, doublereal *v, doublereal *p, doublereal *q, doublereal *r__, integer *nl, integer *nu, integer *n, integer *na, integer *nv);
int orthes(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, doublereal *ort);
int ortran(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, doublereal *ort, doublereal *z__);


/* problem defined functions*/
extern int func(integer ndim, const doublereal *u, const integer *icp, 
	 const doublereal *par, integer ijac, 
	 doublereal *f, doublereal *dfdu, doublereal *dfdp);
extern int stpnt(integer ndim, doublereal t, 
	  doublereal *u, doublereal *par);
extern int bcnd(integer ndim, const doublereal *par, const integer *icp, integer nbc, 
	 const doublereal *u0, const doublereal *u1, integer ijac,
	 doublereal *f, doublereal *dbc);
extern int icnd(integer ndim, const doublereal *par, const integer *icp, integer nint, 
	 const doublereal *u, const doublereal *uold, const doublereal *udot, 
	 const doublereal *upold, integer ijac,
	 doublereal *fi, doublereal *dint);
extern int fopt(integer ndim, const doublereal *u, const integer *icp, 
	 const doublereal *par, integer ijac, 
	 doublereal *fs, doublereal *dfdu, doublereal *dfdp);
// This is a dirty trick with mismatching prototypes -
// sometimes u has to be **double and sometimes *double
extern int pvls(integer ndim, const void *u, doublereal *par);
/* conpar.c */
void *conpar_process(void *);
int conpar(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *ncb, doublereal ***b, integer *nbc, integer *nrc, doublereal ***c, doublereal **d, integer *irf, integer *icf);


/* reduce.c */
int reduce(integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, doublereal **dd, integer *na, integer *nov, integer *ncb, integer *nrc, doublereal ***s1, doublereal ***s2, doublereal ***ca1, integer *icf1, integer *icf2, integer *icf11, integer *ipr, integer *nbc);

/*setubv.c */
#include "auto_types.h"
void *setubv_make_aa_bb_cc(void *);
#ifndef MANIFOLD
int setubv(integer ndim, integer ips, integer na, integer ncol, integer nbc, integer nint, integer ncb, integer nrc, integer nra, integer nca, 
	   FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, 
	   doublereal rds, doublereal ***aa, doublereal ***bb, doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal *rlcur, 
	   doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, 
	   doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#else
int setubv(integer ndim, integer ips, integer na, integer ncol, integer nbc, integer nint, integer nalc, integer ncb, integer nrc, integer nra, integer nca, 
	   FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, 
	   doublereal *rds, doublereal ***aa, doublereal ***bb, doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal *rlcur, 
	   doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, 
	   doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#endif
void setubv_parallel_arglist_copy(setubv_parallel_arglist *output, const setubv_parallel_arglist input);
#ifndef MANIFOLD
void setubv_parallel_arglist_constructor(integer ndim, integer ips, integer na, integer ncol, 
					 integer nbc, integer nint, integer ncb, integer nrc, integer nra, integer nca, 
					 FUNI_TYPE((*funi)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, 
					 integer *icp, doublereal ***aa, doublereal ***bb, 
					 doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal **ups, 
					 doublereal **uoldps, doublereal **udotps, 
					 doublereal **upoldp, doublereal *dtm, 
					 doublereal **wp, doublereal **wt, doublereal *wi,
					 doublereal *thu, doublereal *thl, doublereal *rldot, BCNI_TYPE((*bcni)),
					 setubv_parallel_arglist *data);
#else
void setubv_parallel_arglist_constructor(integer ndim, integer ips, integer na, integer ncol, 
					 integer nbc, integer nint, integer nalc, integer ncb, integer nrc, integer nra, integer nca, 
					 FUNI_TYPE((*funi)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, 
					 integer *icp, doublereal ***aa, doublereal ***bb, 
					 doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal **ups, 
					 doublereal **uoldps, doublereal **udotps, 
					 doublereal **upoldp, doublereal *dtm, 
					 doublereal **wp, doublereal **wt, doublereal *wi,
					 doublereal *thu, doublereal *thl, doublereal *rldot, BCNI_TYPE((*bcni)),
					 setubv_parallel_arglist *data);
#endif
void setubv_make_fa(setubv_parallel_arglist larg);
#ifndef MANIFOLD
void setubv_make_fc_dd(setubv_parallel_arglist larg,doublereal **dups, doublereal *rlcur, 
		       doublereal *rlold, doublereal rds);
#else
void setubv_make_fc_dd(setubv_parallel_arglist larg,doublereal **dups, doublereal *rlcur, 
		       doublereal *rlold, doublereal *rds);
#endif


/*worker.c*/
int mpi_worker();
int mpi_setubv_worker();
int mpi_conpar_worker();
#include "auto_types.h"
int set_funi_and_icni(iap_type *,setubv_parallel_arglist *);

#ifdef AUTO_CONSTRUCT_DESCTRUCT
int user_construct(int argc, char **argv);
int user_destruct();
#endif

/*dmatrix.c*/
doublereal **dmatrix(integer n_rows, integer n_cols);
doublereal ***dmatrix_3d(integer n_levels, integer n_rows, integer n_cols);
void free_dmatrix(doublereal **m);
void free_dmatrix_3d(doublereal ***m);

doublereal **DMATRIX(integer n_rows, integer n_cols);
doublecomplex **DCMATRIX(integer n_rows, integer n_cols);
doublereal ***DMATRIX_3D(integer n_levels, integer n_rows, integer n_cols);
void FREE_DMATRIX(doublereal **m);
void FREE_DCMATRIX(doublecomplex **m);
void FREE_DMATRIX_3D(doublereal ***m);

/* slower allocation routines to facilitate debugging - don't work with MPI! */
/* mess up eispack interfacing as well - can mess up Floquet multipliers!    */

doublereal **dmatrix_debug(integer n_rows, integer n_cols);
doublereal ***dmatrix_3d_debug(integer n_levels, integer n_rows, integer n_cols);
void free_dmatrix_debug(doublereal **m);
void free_dmatrix_3d_debug(doublereal ***m);

/*fcon.c*/
int prepare_cycle(AutoData *Data, double *cycle, int len, double *ups_out, double *udotps_out, double *rldot_out);
void fcon_cntdif(integer n, double *d);
int fcon_eqdf(integer ntst, integer ndim, integer ncol, double *dtm, double *ups, double *eqf, integer iper);
int fcon_init(integer *iap);
int fcon_adapt(integer *iap, integer nold, integer ncold, integer nnew, integer ncnew, double *tm, double *dtm, double *ups, double *vps);
int fcon_interp(integer ndim, integer n, integer nc, double *tm, double *ups, integer n1, integer nc1, double *tm1, double *ups1, double *tm2, integer *itm1);
int fcon_newmsh(integer ndim, double *ups, integer nold, integer ncold, double *tmold, double *dtmold, integer nnew, double *tmnew, integer iper);
int fcon_wrtbv8(integer *iap, double *par, integer icp, double rldot, double *ups, double *udotps, double *tm, double *dtm,
                double *ups_out, double *udotps_out, double *rldot_out);
int fcon_intwts(integer n, double z, double *x, double *wts);
int fcon_ordr(integer n, double * tm, integer n1, double *tm1, integer *itm1);

#ifndef DMATRIX_C
#ifdef MALLOC_DEBUG
#define dmatrix dmatrix_debug
#define dmatrix_3d dmatrix_3d_debug
#define free_dmatrix free_dmatrix_debug
#define free_dmatrix_3d free_dmatrix_3d_debug
#endif
#endif

#endif





