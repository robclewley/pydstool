#include "radau5mod.h"

/* To be set at compilation time */

extern int N_AUXVARS;
extern int N_EVENTS;

double *gICs = NULL;
double globalt0 = 0;
double **gBds = NULL; /* Holds upper and lower bounds on phase space variables and
			 parameters. gBds[] has length phaseDim + paramDim; each of
			 gBds[i] is a length 2 double array. gBds[i][0] is the upper bound,
			 gBds[i][1] is the lower bound. */

IData *gIData = NULL;

ContSolFunType gContSolFun = &contsolfun;

/**************************************
 **************************************
 *            MAIN ROUTINES
 **************************************
 **************************************/

PyObject* Integrate(double *ic, double t, double hinit, double hmax,
		    double safety,
		    double jacRecompute, double newtonStop,
		    double stepChangeLB, double stepChangeUB,
		    double stepSizeLB, double stepSizeUB,
		    int hessenberg,  int maxNewton,
		    int newtonStart, int index1dim, int index2dim,
		    int index3dim, int stepSizeStrategy,
		    int DAEstructureM1, int DAEstructureM2,
		    int useJac, int useMass, int verbose,
		    int calcAux, int calcSpecTimes) {
  int i, j;
  double stats[7];
  double hlast = -1;
  int idid = 0;                   /* Return code from radau5. Codes are:
				     1 - Successful
				     2 - Successful, interrupted by solout
				     -1 - Input not consistent
				     -2 - Larger max steps is needed
				     -3 - Step size becomes too small
				     -4 - Matrix is repeatedly singular
				  */

  int iout = DENSE_OUTPUT_CALL; /* Call solout at each step */
  int itol = VECTOR_ERR_TOL;      /* rtol, atol are vectors; must pass point to int */
  int *ipar = NULL; /* No integer parameter array */

  int phaseDim = 0;
  double tinit = 0;
  double tend = 0;

  int ijac = 0; /* if we setup Jac to return correctly oriented jacobian */
  int imas = 0; /* supposed to be identity, so not DAE */
  int mljac = 0;
  int mujac = 0; /* Bandedness not accounted for yet. */
  int mlmas = 0;
  int mumas = 0;

  int lwork = 0;
  int liwork = 0;

  FILE *ErrOut = NULL;

  assert( gIData );
  assert( gICs );
  assert( ic );

  if( gIData->isInitBasic != 1 || gIData->isInitIntegData != 1 ) {
    return PackOut(gIData, gICs, FAILURE, stats, hlast, idid);
  }

  /* Set whether to calculate output at specific times on this run */
  gIData->calcSpecTimes = calcSpecTimes;
  gIData->calcAux = calcAux;

  for( i = 0; i < gIData->phaseDim; i++ ) {
    gIData->gIC[i] = ic[i];
  }

  /* Call RADAU5 */

  if( verbose == 1 )
    ErrOut = stderr;

  import_array();


  phaseDim = gIData->phaseDim;
  tinit = t;
  tend = gIData->tEnd;
  ijac = (useJac && gIData->hasJac) ? 1 : 0;
  imas = (useMass && gIData->hasMass) ? 1 : 0;

  /* Set the direction of integration */
  gIData->direction = (t < gIData->tEnd ) ? 1 : -1;

  /* Call RADAU5 */
  if( InitializeRadauOptions( gIData, UROUND, safety,
			      jacRecompute, newtonStop,
			      stepChangeLB, stepChangeUB,
			      hmax, stepSizeLB, stepSizeUB,
			      hessenberg, gIData->maxPts, maxNewton,
			      newtonStart, index1dim, index2dim,
			      index3dim, stepSizeStrategy,
			      DAEstructureM1, DAEstructureM2 )
      != SUCCESS ) {
        return PackOut(gIData, gICs, FAILURE, stats, hlast, idid);
  }

  mljac = gIData->jacLowerBandwidth;
  mujac = gIData->jacUpperBandwidth;

  mlmas = gIData->masLowerBandwidth;
  mumas = gIData->masUpperBandwidth;

  lwork = gIData->workArrayLen;
  liwork = gIData->intWorkArrayLen;

  radau5_(&phaseDim, vfield, &tinit, gIData->gIC, &tend, &hinit, gIData->gRTol,
	  gIData->gATol, &itol, vfieldjac, &ijac, &mljac, &mujac,
	  vfieldmas, &imas, &mlmas, &mumas, radau_solout, &iout,
	  gIData->gWorkArray, &lwork, gIData->gIntWorkArray, &liwork, gIData->gParams,
	  ipar, &idid, radau_adjust_h);

  gIData->hasRun = 1;

  /* Stats are:
     0 - Num function evals
     1 - Num Jacobian evals
     2 - Num computed steps
     3 - Num accepted steps
     4 - Num rejected steps
     5 - Num LU decompositions
     6 - Num Forward-Backward substitutions
  */
  for( i = 0; i < 7; i++ ) {
    stats[i] = gIData->gIntWorkArray[i + 13];
  }

  /* FORTRAN Radau5 routine puts last H step in hinit */
  hlast = hinit;

  if( N_AUXVARS > 0  && calcAux != 0 ) {
    AuxVarCalc( gIData );
  }

  return PackOut(gIData, gICs, SUCCESS, stats, hlast, idid);
}



/**************************************
 **************************************
 *       INTEGRATION HELPERS
 **************************************
 **************************************/


/* Just a way to dereference the first argument */
void radau_adjust_h(double *t, double *h) {
	adjust_h(gIData, *t, h);
}

void vfield(int *n, double *t, double *x, double *f,
	    double *rpar, int *ipar) {

  FillCurrentExtInputValues(gIData, *t);

  vfieldfunc(*n, (unsigned) gIData->paramDim, *t, x,
	     rpar, f, (unsigned) gIData->extraSpaceSize, gIData->gExtraSpace,
	     (unsigned) gIData->nExtInputs, gIData->gCurrentExtInputVals);
}

void vfieldjac(int *n, double *t, double *x, double *df, int *ldf,
	       double *rpar, int *ipar) {
  double **f = NULL;

  setJacPtrs(gIData, df);
  f = gIData->gJacPtrs;

  FillCurrentExtInputValues(gIData, *t);

  jacobian(*n, (unsigned) gIData->paramDim, *t, x, rpar, f,
	   (unsigned) gIData->extraSpaceSize, gIData->gExtraSpace,
	   (unsigned) gIData->nExtInputs, gIData->gCurrentExtInputVals);
}

/* Phase space dim n, pointer to mass array write location am,
   int mass matrix lower bandwidth lmas,
   double real-valued parameters rpar
   int int-valued parameters ipar */
void vfieldmas(int *n, double *am, int *lmas, double *rpar, int *ipar, double *t, double *x) {
  double **f = NULL;

  setMassPtrs(gIData, am);
  f = gIData->gMassPtrs;

  FillCurrentExtInputValues(gIData, *t);

  massMatrix(*n, (unsigned) gIData->paramDim, *t, x, rpar, f,
	     (unsigned) gIData->extraSpaceSize, gIData->gExtraSpace,
	     (unsigned) gIData->nExtInputs, gIData->gCurrentExtInputVals);
}

/* Continous solution function for interpolation
   Takes phase space index and time */
double contsolfun(unsigned k, double x) {
  int i = (int) k + 1;
  double t = x;

  /* contdata and contdim are set by solout */
  return contr5_(&i, &t, gIData->contdata, gIData->contdim);
}


void refine(int n, double xold, double x) {
  int i, j, r = gIData->refine;
  double dx = (x - xold) / (r+1);
  double t = xold;


  gIData->refineBufIdx = 0;

  for (j = 0; j < r; j++) {
    t = t+dx;
    for (i = 0; i < n; i++) {
      gIData->gYout[i] = gIData->cContSolFun(i, t);
    }
    BufferRefinePoint(gIData, t, gIData->gYout, gIData->refineBufIdx++);
  }
}


void radau_solout(int *nr, double *xold, double *x, double *y, double *cont,
		  int *lrc, int *n, double *rpar, int *ipar, int *irtrn) {
  int TotEvts = 0;
  int termFound = 0;
  *irtrn = 0;  // stop-flag unset

  /* Reset event flags */
  gIData->eventFound = 0;

  gIData->lastTime = *x;
  gIData->lastPoint = y;

  /* Update globals with cont'n data for this point */
  gIData->contdata = cont;
  gIData->contdim = lrc;


  /* Just output the initial point */
  if (gIData->timeIdx == 0) {
    OutputPoint(gIData, gIData->lastTime, gIData->lastPoint);
    return;
  }

  /* Check for events */
  if( gIData->haveActive > 0 ) {
    TotEvts = DetectEvents(gIData, *xold, gIData->lastTime, gIData->lastPoint, irtrn, &termFound);
    /* If negative TotEvts, exit because we've exceeded maxevtpts */
    if( TotEvts < 0 ) {
      *irtrn = -8;
      return;
    }
  }

  /* Do refinement */
  if ( gIData->refine > 0)
    refine(*n, *xold, gIData->lastTime);

  SavePoints(gIData, TotEvts, termFound);

  /* Check limit on number of points */
  if (gIData->timeIdx >= gIData->maxPts)
    *irtrn = -6;             /* Max points return code */
}

int CompareEvents(const void *element1, const void *element2) {
  int *v1 = (int *) element1;
  int *v2 = (int *) element2;

  return (gIData->gNTEvtFoundTimes[*v1] < gIData->gNTEvtFoundTimes[*v2] ) ? -1 :
    (gIData->gNTEvtFoundTimes[*v1] > gIData->gNTEvtFoundTimes[*v2]) ? 1 : 0;
}

