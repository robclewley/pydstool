#include "dop853mod.h"

/* To be set at compilation time */

extern int N_AUXVARS;
extern int N_EVENTS;
extern int N_EXTINPUTS;

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

PyObject* Integrate(double *ic, double t, double hinit,
		    double hmax, double safety,
		    double fac1, double fac2, double beta,
		    int verbose, int calcAux, int calcSpecTimes,
		    int checkBounds, int boundCheckMaxSteps, double *magBound) {
  int i;
  double stats[4];
  double hlast = -1;
  int idid = 0;                   /* return code from dop853 */
  unsigned *dense_i = 0;          /* no need for indices of dense components */
  FILE *ErrOut = NULL;


  assert( gIData );
  assert( gICs );
  assert( gBds );
  assert(ic);

  if( gIData->isInitBasic != 1 || gIData->isInitIntegData != 1 ) {
    return PackOut(gIData, gICs, FAILURE, stats, hlast, idid);
  }

  /* Set whether to calculate output at specific times on this run */
  gIData->calcSpecTimes = calcSpecTimes;
  gIData->calcAux = calcAux;

  for( i = 0; i < gIData->phaseDim; i++ ) {
    gIData->gIC[i] = ic[i];
  }

  _init_numpy();


  if( verbose == 1 )
    ErrOut = stderr;

  /* Set the direction of integration */
  gIData->direction = (t < gIData->tEnd ) ? 1 : -1;

  /* Call DOP853 */

  if( InitializeDopriOptions( gIData, checkBounds, boundCheckMaxSteps, magBound )
      != SUCCESS ) {
    return PackOut(gIData, gICs, FAILURE, stats, hlast, idid);
  }

  idid = dop853(gIData->phaseDim, &vfield, t, gIData->gIC, gIData->gParams,
		gIData->tEnd, gIData->gRTol, gIData->gATol, VECTOR_ERR_TOL,
		&dopri_solout, DENSE_OUTPUT_CALL, ErrOut,
		UROUND, safety, fac1, fac2, beta,
		hmax, hinit, gIData->maxPts,
		1, TEST_STIFF, gIData->phaseDim, dense_i, gIData->phaseDim,
		gIData->checkBounds, gIData->boundsCheckMaxSteps, gIData->gMagBound,
		&dopri_adjust_h);

  gIData->hasRun = 1;

  stats[0] = (double) nfcnRead();
  stats[1] = (double) nstepRead();
  stats[2] = (double) naccptRead();
  stats[3] = (double) nrejctRead();
  hlast = hRead();

  if( N_AUXVARS > 0 && calcAux != 0 ) {
    AuxVarCalc( gIData );
  }

  return PackOut(gIData, gICs, SUCCESS, stats, hlast, idid);
}




/**************************************
 **************************************
 *       INTEGRATION HELPERS
 **************************************
 **************************************/

void dopri_adjust_h( double t, double *h ) {
	adjust_h(gIData, t, h);
}

void vfield(unsigned n, double x, double *y, double *p, double *f) {

  FillCurrentExtInputValues(gIData, x);

  vfieldfunc(n, (unsigned) gIData->paramDim, x, y,
	     p, f, (unsigned) gIData->extraSpaceSize, gIData->gExtraSpace,
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
   return contd8(k, x);
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


void dopri_solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn) {
  int TotEvts = 0;
  int termFound = 0;
  *irtrn = 0;  // stop-flag unset

  gIData->eventFound = 0;

  gIData->lastTime = x;
  gIData->lastPoint = y;

  /* Just output the initial point */
  if (gIData->timeIdx == 0) {
    OutputPoint(gIData, gIData->lastTime, gIData->lastPoint);
    return;
  }

  /* Check for events */
  if( gIData->haveActive > 0 ) {
    TotEvts = DetectEvents(gIData, xold, gIData->lastTime, gIData->lastPoint, irtrn, &termFound);
    /* If negative TotEvts, exit because we've exceeded the maxevtpts */
    if( TotEvts < 0 ) {
      *irtrn = -8;
      return;
    }
  }

  /* Do refinement */
  if ( gIData->refine > 0)
    refine(n, xold, gIData->lastTime);

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
