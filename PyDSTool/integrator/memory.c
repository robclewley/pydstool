#include "memory.h"

/* Basic initialization that must be done before any methods of the
   integrator can be used. Every parameter set here must be left unchanged
   for the life of the integrator, except Pars.

   This function should be called ONLY onece: when the integrator is created.

   It assumes a blank (all 0's and NULLs) IData struct. (BlankIData)

*/

int InitializeBasic( IData *GS, int PhaseDim, int ParamDim, int nAux, int nEvents,
		     int nExtInputs, int HasJac, int HasJacP, int HasMass, int extraSize) {
  assert( GS );
  /* Check to see if we're already basically initialized; if so, check that we
     haven't changed dimensions (we assume we're PyMem_Malloc'd here) */
  if( PhaseDim == 0 || ParamDim < 0 ) {
      return FAILURE;
  }

  /* We should not be initialized yet, but we will check that critical values
     aren't being changed anyway. */
  if( GS->isInitBasic ) {
    if( PhaseDim != GS->phaseDim || ParamDim != GS->paramDim ) {
      return FAILURE;
    }
  }

  GS->phaseDim = PhaseDim;
  GS->paramDim = ParamDim;

  GS->gIC = NULL;
  GS->gIC = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
  assert(GS->gIC);

  /* We malloc space for the parameter array, but
     we don't do so for the global ICs. That is handled in a differnt
     function ( ) */
  GS->gParams = NULL;
  GS->gParams = (double *)PyMem_Malloc(GS->paramDim*sizeof(double));
  assert(GS->gParams);

  /* Set total number of aux vars */
  GS->nAuxVars = nAux;
  GS->gAuxPoints = NULL;

  /* These may not change, but use of jacobians, etc. in integration
     can be set by passing arguments to the appropriate integrate function.
  */
  GS->hasJac = (( HasJac > 0 ) ? 1 : 0);
  GS->hasJacP = (( HasJacP > 0 ) ? 1 : 0);
  GS->hasMass = (( HasMass > 0 ) ? 1 : 0);

  /* We save jacobians, etc. in column-major format. These
     pointers into the 1-D arrays are provided for convenience.
  */

  GS->gJacPtrs = NULL;
  if( GS->hasJac == 1 ) {
    GS->gJacPtrs = (double **)PyMem_Malloc(GS->phaseDim*sizeof(double *));
    assert(GS->gJacPtrs);
  }

  GS->gMassPtrs = NULL;
  if( GS->hasMass == 1 ) {
    GS->gMassPtrs = (double **)PyMem_Malloc(GS->phaseDim*sizeof(double *));
    assert(GS->gMassPtrs);
  }

  /* Set total number of events, assign event function ptrs
   to the array. These cannot change, but may be made active/inactive. */
  assert(nEvents >= 0);
  GS->nEvents = nEvents;
  GS->gEventFcnArray = NULL;
  if( GS->nEvents > 0 ) {
    GS->gEventFcnArray = (EvFunType *)PyMem_Malloc(GS->nEvents * sizeof(EvFunType));
    assert(GS->gEventFcnArray);
    assignEvents(GS->gEventFcnArray);
  }

  /* Set the total number of expected external inputs */
  assert(nExtInputs >= 0);
  GS->nExtInputs = nExtInputs;

  /* Any extra working space needed by the vfield, aux funcs, etc.
     must be assigned only once. */
  assert(extraSize >= 0);
  GS->extraSpaceSize = extraSize;
  GS->gExtraSpace = NULL;
  if( GS->extraSpaceSize > 0 ) {
    GS->gExtraSpace = PyMem_Malloc(GS->extraSpaceSize*sizeof(double));
    assert(GS->gExtraSpace);
  }

  GS->isInitBasic = 1;
  return SUCCESS;
}

/* Here we free all of the memory that was allocated during
   the InitializeBasic call and set initBasic to 0.
   After this function is called, the integrator is no longer usable
   and should be deleted. (But the IData struct, etc. have not
   been freed.)
*/

void CleanupBasic( IData *GS ) {

  if( GS != NULL ) {

    if( GS->gExtraSpace != NULL ) {
      PyMem_Free(GS->gExtraSpace);
      GS->gExtraSpace = NULL;
    }

    if( GS->gEventFcnArray != NULL ) {
      PyMem_Free(GS->gEventFcnArray);
      GS->gEventFcnArray = NULL;
    }

    if(GS->gMassPtrs != NULL) {
      PyMem_Free(GS->gMassPtrs);
      GS->gMassPtrs = NULL;
    }

    if(GS->gJacPtrs != NULL) {
      PyMem_Free(GS->gJacPtrs);
      GS->gJacPtrs = NULL;
    }

    if( GS->gParams != NULL ) {
      PyMem_Free(GS->gParams);
      GS->gParams = NULL;
    }

    if( GS->gIC != NULL ) {
      PyMem_Free(GS->gIC);
      GS->gIC = NULL;
    }


    GS->isInitBasic = 0;
  }
}

int SetRunParams( IData *GS, double *Pars, double *ICs, double **Bds, double *y0, double gTime,
		  double *GlobalTime0, double tstart, double tend,
		  int refine, int nSpecTimes, double *specTimes, double *upperBounds, double *lowerBounds) {

  int i;

  assert( GS );

  /* We only check initBasic because we don't rely on
     any other mallocs having been performed.  */
  assert( GS->isInitBasic == 1 );

  /* Copy in parameters, ICs, etc. for this run */
  /* Parameters were malloc'd in InitBasic */
  if( GS->paramDim > 0 ) {
    assert(GS->gParams);
    for( i = 0; i < GS->paramDim; i++ ) {
      GS->gParams[i] = Pars[i];
    }
  }

  /* We expect IC to be global ICs, initialized ahead of time (initialize basic) */
  /* Save initial conditions */
  for( i = 0; i < GS->phaseDim; i++ ) {
    ICs[i] = y0[i];
  }

  /* We expect Bds to be global Bounds, initialized ahead of time (initialize basic) */
  /* Save the current upper and lower bounds on parameters. */
  for( i = 0; i < GS->phaseDim + GS->paramDim; i++ ) {
    Bds[0][i] = lowerBounds[i];
    Bds[1][i] = upperBounds[i];
  }

  /* Set the global time (also a global variable) */
  *GlobalTime0 = gTime;

  /* Make sure that all of our global variables are set to initial values at beginning of
     call. NB: This was the source of a bug that popped up when the program was run
     twice in succession. */
  GS->tStart = tstart;
  GS->tEnd = tend;

  GS->lastTime = tstart;

  /* Set the direction of integration */
  GS->direction = (GS->tStart < GS->tEnd ) ? 1 : -1;

  /* Calling SetRunParams forces a new run -- this means we must reset the indices */
  GS->hasRun = 0;
  GS->timeIdx = 0; GS->pointsIdx = 0; GS->auxIdx = 0;
  GS->eventFound = 0;

  nSpecTimes = ( nSpecTimes < 0 ) ? 0 : nSpecTimes;
  if( nSpecTimes > 0 ) {
    if( GS->specTimesLen > 0 ) {
      double *temp = NULL;
      assert(GS->gSpecTimes);
      temp = (double *)PyMem_Realloc(GS->gSpecTimes, nSpecTimes*sizeof(double));
      assert(temp);
    }
    else {
      if( GS->gSpecTimes != NULL ) {
	PyMem_Free( GS->gSpecTimes );
	GS->gSpecTimes = NULL;
      }
      GS->gSpecTimes = (double *)PyMem_Malloc(nSpecTimes*sizeof(double));
      assert(GS->gSpecTimes);
    }
    GS->specTimesLen = nSpecTimes;
    for( i = 0; i < GS->specTimesLen; i++ ) {
      GS->gSpecTimes[i] = specTimes[i];
    }
  }
  else if ( GS->specTimesLen > 0 ) {
    if( GS->gSpecTimes != NULL ) {
      PyMem_Free(GS->gSpecTimes);
      GS->gSpecTimes = NULL;
    }
    GS->specTimesLen = nSpecTimes;
  }


  /* If refinement is on, allocate space for buffering refinement
     points and times */
  refine = ( refine < 0 ) ? 0 : refine;
  GS->refineBufIdx = 0;
  if( refine > 0 ) {
    if( refine > GS->refine ) {
      if( GS->gRefineTimes != NULL ) {
	for( i = 0; i < GS->refine; i++ ) {
	  if( GS->gRefinePoints[i] != NULL ) {
	    PyMem_Free( GS->gRefinePoints[i] );
	    GS->gRefinePoints[i] = NULL;
	  }
	}
	PyMem_Free( GS->gRefinePoints );
	GS->gRefinePoints = NULL;
      }

      GS->refine = refine;

      if( GS->gRefineTimes == NULL ) {
	GS->gRefinePoints = (double **)PyMem_Malloc(GS->refine*sizeof(double *));
	assert(GS->gRefinePoints);
	for( i = 0; i < GS->refine; i++ ) {
	  GS->gRefinePoints[i] = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
	  assert(GS->gRefinePoints[i]);
	}
	GS->gRefineTimes = (double *)PyMem_Malloc(GS->refine*sizeof(double));
	assert(GS->gRefineTimes);
      }
    }
    /* Note in this case we are not reallocing the refine times.
       Since this is a 1-D double array, free will take care of it easily. */
    else if (refine < GS->refine) {
      for( i = refine; i < GS->refine; i++ ) {
	if( GS->gRefinePoints[i] != NULL ) {
	  PyMem_Free(GS->gRefinePoints[i]);
	  GS->gRefinePoints[i] = NULL;
	}
      }
      GS->refine = refine;
    }
  }
  else {
    if ( refine < GS->refine ) {
      for( i = 0; i < GS->refine; i++ ) {
        if( GS->gRefinePoints[i] != NULL ) {
	  PyMem_Free(GS->gRefinePoints[i]);
          GS->gRefinePoints[i] = NULL;
        }
      }
      PyMem_Free(GS->gRefinePoints);
      GS->gRefinePoints = NULL;
    }
  }

  GS->isInitRunParams = 1;

  return SUCCESS;
}

void CleanupRunParams( IData *GS ) {

  int i;

  assert(GS);

  if(GS->gSpecTimes != NULL) {
    PyMem_Free(GS->gSpecTimes);
    GS->gSpecTimes = NULL;
  }

  if(GS->gRefinePoints != NULL) {
    for(i = GS->refine-1; i >= 0; i--) {
      if(GS->gRefinePoints[i] != NULL) {
	PyMem_Free(GS->gRefinePoints[i]);
      }
    }
    PyMem_Free(GS->gRefinePoints);
    GS->gRefinePoints = NULL;
  }

  if(GS->gRefineTimes != NULL) {
    PyMem_Free(GS->gRefineTimes);
    GS->gRefineTimes = NULL;
  }

  GS->specTimesLen = 0;
  GS->refine = 0;

  GS->isInitRunParams = 0;
}

int InitIntegData( IData *GS,  int Maxpts, double *atol, double *rtol,
		   ContSolFunType ContSolFun) {

  int i;

  /* Check that InitIntegData is being called in the right order? */
  assert(GS->isInitBasic == 1);

  assert(GS->isInitIntegData == 0);

  GS->gPoints = NULL; GS->gTimeV = NULL; GS->gYout = NULL;

  GS->lastPoint = NULL;

  /* Assign global variables */
  GS->maxPts = Maxpts;

  /* Allocate space for maximum number of points to be calculated during integration
     NB: Change this to throw Python exception on failure? */
  GS->gPoints = (double **)PyMem_Malloc(GS->maxPts*sizeof(double *));
  assert(GS->gPoints);
  for( i = 0; i < GS->maxPts; i++ ) {
    GS->gPoints[i] = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
    assert(GS->gPoints[i]);
  }

  GS->gTimeV = (double *)PyMem_Malloc(GS->maxPts*sizeof(double));
  assert(GS->gTimeV);

  GS->gYout = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
  assert(GS->gYout);

  /* Radau specific initializations */
#ifdef __RADAU__
  if( GS->hasJac || GS->hasMass ) {
    GS->workArrayLen = (GS->phaseDim)*(5*(GS->phaseDim) + 12) + 20;
    GS->intWorkArrayLen = 3*(GS->phaseDim) + 20;
  }
  else {
    GS->workArrayLen = (GS->phaseDim)*(4*(GS->phaseDim) + 12) + 20;
    GS->intWorkArrayLen = 3*(GS->phaseDim) + 20;
  }

  GS->gWorkArray = (double *)PyMem_Malloc(GS->workArrayLen*sizeof(double));
  assert(GS->gWorkArray);

  GS->gIntWorkArray = (int *)PyMem_Malloc(GS->intWorkArrayLen*sizeof(int));
  assert(GS->gIntWorkArray);

  GS->ijac = 0; /* Currently ignored */
  /* Currently, only full matrices are supported */
  if( GS->hasJac ) {
    GS->jacLowerBandwidth = GS->phaseDim;
  }
  else {
    GS->jacLowerBandwidth = GS->phaseDim;
  }
  GS->jacUpperBandwidth = 0;

  GS->imas = 0; /* Currently ignored */
  /* Currently, only full matrices are supported */
  if( GS->hasMass ) {
    GS->masLowerBandwidth = GS->phaseDim;
  }
  else {
    GS->masLowerBandwidth = 0;
  }
  GS->masUpperBandwidth = 0;
#endif

  /* Dopri specific initializations */
#ifdef __DOPRI__
  GS->gMagBound = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
  assert(GS->gMagBound);
  GS->checkBounds = 0;
  GS->boundsCheckMaxSteps = 0;
#endif

  GS->gATol = NULL;  GS->gRTol = NULL;

  GS->gATol = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
  assert(GS->gATol);
  GS->gRTol = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
  assert(GS->gRTol);
  for( i = 0; i < GS->phaseDim; i++ ) {
    GS->gATol[i] = atol[i];
    GS->gRTol[i] = rtol[i];
  }

  GS->cContSolFun = NULL;
  GS->cContSolFun = ContSolFun; assert(GS->cContSolFun);

  GS->isInitIntegData = 1;

  return SUCCESS;
}

void CleanupIData( IData *GS ) {
  int i;

  if( GS != NULL ) {

    /* PyMem_Free C memory used for integration, event detection. Try to
       do this in reverse order of allocation for possible memory efficiencies. */

    /* PyMem_Free auxilliary points */
    if(GS->gAuxPoints != NULL) {
      for( i = GS->pointsIdx-1; i >= 0; i-- ) {
	if( GS->gAuxPoints[i] != NULL ) {
	  PyMem_Free(GS->gAuxPoints[i]);
	}
      }
      PyMem_Free(GS->gAuxPoints);
      GS->gAuxPoints = NULL;
    }

    if(GS->gRTol != NULL) {
      PyMem_Free(GS->gRTol);
      GS->gRTol = NULL;
    }

    if(GS->gATol != NULL) {
      PyMem_Free(GS->gATol);
      GS->gATol = NULL;
    }

    /* Radau specific PyMem_Frees */
    #ifdef __RADAU__

    if(GS->gIntWorkArray != NULL) {
      PyMem_Free(GS->gIntWorkArray);
      GS->gIntWorkArray = NULL;
    }

    if(GS->gWorkArray != NULL) {
      PyMem_Free(GS->gWorkArray);
      GS->gWorkArray = NULL;
    }
    #endif

    /* Dopri specific PyMem_Frees */
    #ifdef __DOPRI__

    if(GS->gMagBound != NULL) {
      PyMem_Free(GS->gMagBound);
      GS->gMagBound = NULL;
    }
    #endif

    if(GS->gYout != NULL) {
      PyMem_Free(GS->gYout);
      GS->gYout = NULL;
    }

    if(GS->gTimeV != NULL) {
      PyMem_Free(GS->gTimeV);
      GS->gTimeV = NULL;
    }

    if(GS->gPoints != NULL){
      for( i = GS->maxPts-1; i >= 0; i-- ) {
	if(GS->gPoints[i] != NULL) {
	  PyMem_Free(GS->gPoints[i]);
	}
      }
      PyMem_Free(GS->gPoints);
      GS->gPoints = NULL;
    }

    GS->auxIdx = 0;
    GS->maxPts = 0;

    GS->isInitIntegData = 0;
  }
}

int InitializeExtInputs( IData *GS, int nExtInputs, int *extInputLens, double *extInputVals,
			 double *extInputTimes ) {
  int i, j, curridx;

  assert(GS);
  assert( nExtInputs == GS->nExtInputs );

   /* Only change external inputs if there were external inputs originally;
     cannot add/subtract external inputs after initial run. We assume that if the inputs are
     being changed, that the input array has the correct number of components. */
  if( GS->nExtInputs > 0 ) {

      assert(extInputLens);
      assert(extInputVals);
      assert(extInputTimes);

      /* Free the old memory. We leave the outermost pointers,
	 since we assume that the number of external inputs is the same as before. */
      if( GS->gExtInputTimes != NULL ) {
	for( i = 0; i < GS->nExtInputs; i++ ) {
	  if( GS->gExtInputTimes[i] != NULL ) {
	    PyMem_Free(GS->gExtInputTimes[i]);
	  }
	}
      }

      if( GS->gExtInputVals != NULL ) {
	for( i = 0; i < GS->nExtInputs; i++ ) {
	  if( GS->gExtInputVals[i] != NULL ) {
	    PyMem_Free(GS->gExtInputVals[i]);
	  }
	}
      }

      /* If we have external inputs, allocate space for them and copy
	 the inputs into internal memory space */
      if( GS->gExtInputLens == NULL ) {
	GS->gExtInputLens = (int *)PyMem_Malloc(GS->nExtInputs*sizeof(int));
	assert(GS->gExtInputLens);
      }

      for( i = 0; i < GS->nExtInputs; i++ ) {
	/* If extInputLen[i] invalid, ignore it */
	if( extInputLens[i] > 0 ) {
	  GS->gExtInputLens[i] = extInputLens[i];
	}
	else {
	  GS->gExtInputLens[i] = 0;
	}
      }

      /* Malloc space for the values and times; we assume that the inputs have
	 already been checked for appropriate time orientation */
      if( GS->gExtInputVals == NULL ) {
	GS->gExtInputVals = (double **)PyMem_Malloc(GS->nExtInputs*sizeof(double *));
	assert(GS->gExtInputVals);
      }
      if( GS->gExtInputTimes == NULL ) {
	GS->gExtInputTimes = (double **)PyMem_Malloc(GS->nExtInputs*sizeof(double *));
	assert(GS->gExtInputTimes);
      }

      curridx = 0;
      for( i = 0; i < GS->nExtInputs; i++ ) {
	if( GS->gExtInputLens[i] > 0 ) {
	  GS->gExtInputVals[i] = (double *)PyMem_Malloc(GS->gExtInputLens[i]*sizeof(double));
	  assert(GS->gExtInputVals[i]);
	  GS->gExtInputTimes[i] = (double *)PyMem_Malloc(GS->gExtInputLens[i]*sizeof(double));
	  assert(GS->gExtInputTimes[i]);
	  for( j = 0; j < GS->gExtInputLens[i]; j++ ) {
	    GS->gExtInputVals[i][j] = extInputVals[curridx + j];
	    GS->gExtInputTimes[i][j] = extInputTimes[curridx + j];
	  }
	  curridx += GS->gExtInputLens[i];
	}
	else {
	  GS->gExtInputVals[i] = NULL;
	  GS->gExtInputTimes[i] = NULL;
	}
      }

      /* Malloc space as necessary for the values to be passed to vfieldfunc, etc.
	 and the current index */
      if( GS->gCurrentExtInputIndex == NULL ) {
	GS->gCurrentExtInputIndex = (int *)PyMem_Malloc(GS->nExtInputs*sizeof(int));
	assert(GS->gCurrentExtInputIndex);
	for( i = 0; i < GS->nExtInputs; i++ ) {
	  GS->gCurrentExtInputIndex[i] = 0;
	}
      }

      if( GS->gCurrentExtInputVals == NULL ) {
	GS->gCurrentExtInputVals = (double *)PyMem_Malloc(GS->nExtInputs*sizeof(double));
	assert(GS->gCurrentExtInputVals);
      }

  }

  //  for( i = 0; i < GS->nExtInputs; i++ ) {
  //  for(j = 0; j < GS->gExtInputLens[i]; j++ ) {


  GS->isInitExtInputs = 1;

  return SUCCESS;
}

void CleanupExtInputs( IData *GS ) {
  int i;

  if( GS != NULL ) {
    if( GS->nExtInputs > 0 ) {

      if( GS->gCurrentExtInputVals != NULL ) {
	PyMem_Free(GS->gCurrentExtInputVals);
	GS->gCurrentExtInputVals = NULL;
      }

      if( GS->gCurrentExtInputIndex != NULL ) {
	PyMem_Free(GS->gCurrentExtInputIndex);
	GS->gCurrentExtInputIndex = NULL;
      }

      if( GS->gExtInputTimes != NULL ) {
	for( i = 0; i < GS->nExtInputs; i++ ) {
	  if( GS->gExtInputTimes[i] != NULL ) {
	    PyMem_Free(GS->gExtInputTimes[i]);
	  }
	}
	PyMem_Free(GS->gExtInputTimes);
	GS->gExtInputTimes = NULL;
      }

      if( GS->gExtInputVals != NULL ) {
	for( i = 0; i < GS->nExtInputs; i++ ) {
	  if( GS->gExtInputVals[i] != NULL ) {
	    PyMem_Free(GS->gExtInputVals[i]);
	  }
	}
	PyMem_Free(GS->gExtInputVals);
	GS->gExtInputVals = NULL;
      }

      if( GS->gExtInputLens != NULL ) {
	PyMem_Free(GS->gExtInputLens);
	GS->gExtInputLens = NULL;
      }

    }

    GS->isInitExtInputs = 0;
  }
}

int InitializeEvents( IData *GS, int Maxevtpts, int *EventActive, int *EventDir, int *EventTerm,
		      double *EventInterval, double *EventDelay, double *EventTol,
		      int *Maxbisect, double EventNearCoef ) {

  int i, j, k;

  assert(GS);

  GS->activeTerm = 0; GS->activeNonTerm = 0;

  GS->gTermIndices = NULL; GS->gNonTermIndices = NULL;

  GS->gTempPoint = NULL; GS->gNTEvtFound = NULL;
  GS->gNTEvtFoundOrder = NULL;
  GS->gNTEvtFoundTimes = NULL;

  GS->gCheckableEvents = NULL; GS->gCheckableEventCounts = NULL;

  GS->gEventY = NULL;
  GS->gEventPoints = NULL; GS->gEventTimes = NULL;

  GS->gEventPointBuf = NULL; GS->gEventTimeBuf = NULL;
  GS->eventBufIdx = 0;


  if( EventNearCoef > 0 )
    GS->eventNearCoef = EventNearCoef;
  else
    GS->eventNearCoef = EVENT_NEARNESS_FACTOR;


  GS->gEventY = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
  assert(GS->gEventY);

  /* Count the number of active events and the number of active
     terminal events. */
  GS->haveActive = 0;
  GS->activeTerm = 0;

  GS->maxEvtPts = Maxevtpts;
  assert(GS->maxEvtPts >= 0 );


  /* From SetEventPtrs */
  GS->gEventActive = NULL;
  GS->gEventDir = NULL;
  GS->gEventTerm = NULL;
  GS->gEventDelay = NULL;
  GS->gEventTol = NULL;
  GS->gMaxBisect = NULL;

  if( GS->nEvents > 0 ) {

    GS->gEventActive = (int *)PyMem_Malloc(GS->nEvents*sizeof(int));
    assert(GS->gEventActive);
    for( i = 0; i < GS->nEvents; i++ ) {
      GS->gEventActive[i] = EventActive[i];
    }

    GS->gEventDir = (int *)PyMem_Malloc(GS->nEvents*sizeof(int));
    assert(GS->gEventDir);
    for( i = 0; i < GS->nEvents; i++ ) {
      GS->gEventDir[i] = EventDir[i];
    }

    GS->gEventTerm = (int *)PyMem_Malloc(GS->nEvents*sizeof(int));
    assert(GS->gEventTerm);
    for( i = 0; i < GS->nEvents; i++ ) {
      GS->gEventTerm[i] = EventTerm[i];
    }

    GS->gEventInterval = (double *)PyMem_Malloc(GS->nEvents*sizeof(double));
    assert(GS->gEventInterval);
    for( i = 0; i < GS->nEvents; i++ ) {
      GS->gEventInterval[i] = EventInterval[i];
    }

    GS->gEventDelay = (double *)PyMem_Malloc(GS->nEvents*sizeof(double));
    assert(GS->gEventDelay);
    for( i = 0; i < GS->nEvents; i++ ) {
      GS->gEventDelay[i] = EventDelay[i];
    }

    GS->gEventTol = (double *)PyMem_Malloc(GS->nEvents*sizeof(double));
    assert(GS->gEventTol);
    for( i = 0; i < GS->nEvents; i++ ) {
      GS->gEventTol[i] = EventTol[i];
    }

    GS->gMaxBisect = (int *)PyMem_Malloc(GS->nEvents*sizeof(int));
    assert(GS->gMaxBisect);
    for( i = 0; i < GS->nEvents; i++ ) {
      GS->gMaxBisect[i] = Maxbisect[i];
    }
  }

  /* End from set event pointers */


  if( GS->nEvents > 0 ) {
    for( i = 0; i < GS->nEvents; i++ ) {
      if( GS->gEventActive[i] != 0 ) {
	GS->haveActive++;
	if( GS->gEventTerm[i] == 1 ) {
	  GS->activeTerm++;
	}
      }
    }
  }

  /*  If there are some active events, we will PyMem_Malloc space
      for them and record where they are */
  if( GS->haveActive > 0 ) {
    int j;

    /* PyMem_Malloc space for flag that we found this event on the last
       call to detect events */

    /* PyMem_Malloc space for buffering found events and times
       before merging them into trajectory. */
    GS->gEventPointBuf = (double **)PyMem_Malloc(GS->haveActive*sizeof(double *));
    assert(GS->gEventPointBuf);
    for( i = 0; i < GS->haveActive; i++ ) {
      GS->gEventPointBuf[i] = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
      assert(GS->gEventPointBuf[i]);
    }

    GS->gEventTimeBuf = (double *)PyMem_Malloc(GS->haveActive*sizeof(double));
    assert(GS->gEventTimeBuf);

    /* PyMem_Malloc space for recording the indices of active events and
       how many instances of each have been recorded. */
    GS->gCheckableEvents = (int *)PyMem_Malloc(GS->haveActive*sizeof(int));
    assert(GS->gCheckableEvents);
    GS->gCheckableEventCounts = (int *)PyMem_Malloc(GS->haveActive*sizeof(int));
    assert(GS->gCheckableEventCounts);
    for( i = 0; i < GS->haveActive; i++ ) {
      GS->gCheckableEvents[i] = 0;
      GS->gCheckableEventCounts[i] = 0;
    }

    /* PyMem_Malloc space for recording indices of active terminal events */
    if( GS->activeTerm > 0 ) {
      GS->gTermIndices = (int *)PyMem_Malloc(GS->activeTerm*sizeof(int));
      assert(GS->gTermIndices);
      for( i = 0; i < GS->activeTerm; i++ ) {
	GS->gTermIndices[i] = 0;
      }
    }
    /* Record the number of active, nonterminal events for convenience later */
    GS->activeNonTerm = GS->haveActive - GS->activeTerm;

    /* PyMem_Malloc space for use in sorting events in case multiple events
       occur between integration steps */
    if( GS->activeNonTerm > 0 ) {
      GS->gNonTermIndices = (int *)PyMem_Malloc(GS->activeNonTerm*sizeof(int));
      assert(GS->gNonTermIndices);

      GS->gNTEvtFound = (int *)PyMem_Malloc(GS->activeNonTerm*sizeof(int));
      assert(GS->gNTEvtFound);

      GS->gNTEvtFoundOrder = (int *)PyMem_Malloc(GS->activeNonTerm*sizeof(int));
      assert(GS->gNTEvtFoundOrder);

      GS->gNTEvtFoundTimes = (double *)PyMem_Malloc(GS->activeNonTerm*sizeof(double));
      assert(GS->gNTEvtFoundTimes);

      for( i = 0; i < GS->activeNonTerm; i++ ) {
	GS->gNonTermIndices[i] = 0;
	GS->gNTEvtFound[i] = 0;
	GS->gNTEvtFoundOrder[i] = 0;
	GS->gNTEvtFoundTimes[i] = 0;
      }
    }

    GS->gTempPoint = (double *)PyMem_Malloc(GS->phaseDim*sizeof(double));
    assert(GS->gTempPoint);

    /* Record of how many of each active event we've caught */
    for( i = 0; i < GS->haveActive; i++ ) {
      GS->gCheckableEventCounts[i] = 0;
    }

    /* Allocate space for anticipated events.
       Only allocate space for active events.*/
    GS->gEventPoints = (double ***)PyMem_Malloc(GS->haveActive*sizeof(double **));
    assert(GS->gEventPoints);
    GS->gEventTimes = (double **)PyMem_Malloc(GS->haveActive*sizeof(double *));
    assert(GS->gEventTimes);

    for( i = 0; i < GS->haveActive; i++ ) {
      GS->gEventPoints[i] = (double **)PyMem_Malloc(GS->phaseDim*sizeof(double *));
      assert(GS->gEventPoints[i]);
      if( GS->maxEvtPts > 0 ) {
	GS->gEventTimes[i] = (double *)PyMem_Malloc(GS->maxEvtPts*sizeof(double));
	assert(GS->gEventTimes[i]);
	for(j = 0; j < GS->phaseDim; j++) {
	  GS->gEventPoints[i][j] = (double *)PyMem_Malloc(GS->maxEvtPts*sizeof(double));
	  assert(GS->gEventPoints[i][j]);
	}
      }
    }
  }

  /* Record the indices of which events are active in a smaller array.
     These are 'checkable' events. Record also which ones are terminal,
     nonterminal in smaller arrays. */
  for( i = 0, j = 0, k = 0; i < GS->nEvents; i++ ) {
    if( GS->gEventActive[i] != 0 ) {
      GS->gCheckableEvents[j] = i;
      if( GS->gEventTerm[i] > 0 ) {
	GS->gTermIndices[k] = j;
	k++;
      }
      else {
	GS->gNonTermIndices[j-k] = j;
      }
      j++;
    }
  }

  GS->isInitEvents = 1;

  return SUCCESS;
}

void CleanupEvents( IData *GS ) {
  int i, j;

  if( GS != NULL ) {

    if(GS->gEventPoints != NULL) {
      for( i = GS->haveActive-1; i >= 0; i-- ) {
	if( GS->gEventPoints[i] != NULL ) {
	  for( j = GS->phaseDim-1; j >= 0; j-- ) {
	    if( GS->gEventPoints[i][j] != NULL ) {
	      PyMem_Free(GS->gEventPoints[i][j]);
	    }
	  }
	  PyMem_Free(GS->gEventPoints[i]);
	}
      }
      PyMem_Free(GS->gEventPoints);
      GS->gEventPoints = NULL;
    }

    if(GS->gEventTimes != NULL) {
      for(i = GS->haveActive-1; i >= 0; i--) {
	if(GS->gEventTimes[i] != NULL) {
	  PyMem_Free(GS->gEventTimes[i]);
	}
      }
      PyMem_Free(GS->gEventTimes);
      GS->gEventTimes = NULL;
    }

    if(GS->gTempPoint != NULL) {
      PyMem_Free(GS->gTempPoint);
      GS->gTempPoint = NULL;
    }

    if(GS->gNTEvtFoundTimes != NULL) {
      PyMem_Free(GS->gNTEvtFoundTimes);
      GS->gNTEvtFoundTimes = NULL;
    }

    if(GS->gNTEvtFoundOrder != NULL) {
      PyMem_Free(GS->gNTEvtFoundOrder);
      GS->gNTEvtFoundOrder = NULL;
    }

    if(GS->gNTEvtFound != NULL) {
      PyMem_Free(GS->gNTEvtFound);
      GS->gNTEvtFound = NULL;
    }

    if(GS->gNonTermIndices != NULL) {
      PyMem_Free(GS->gNonTermIndices);
      GS->gNonTermIndices = NULL;
    }

    if(GS->gTermIndices != NULL) {
      PyMem_Free(GS->gTermIndices);
      GS->gTermIndices = NULL;
    }

    if(GS->gCheckableEventCounts != NULL) {
      PyMem_Free(GS->gCheckableEventCounts);
      GS->gCheckableEventCounts = NULL;
    }

    if(GS->gCheckableEvents != NULL) {
      PyMem_Free(GS->gCheckableEvents);
      GS->gCheckableEvents = NULL;
    }

    if(GS->gEventTimeBuf != NULL) {
      PyMem_Free(GS->gEventTimeBuf);
      GS->gEventTimeBuf = NULL;
    }

    if(GS->gEventPointBuf != NULL) {
      for(i = GS->haveActive-1; i >= 0; i--) {
	if(GS->gEventPointBuf[i] != NULL) {
	  PyMem_Free(GS->gEventPointBuf[i]);
	}
      }
      PyMem_Free(GS->gEventPointBuf);
      GS->gEventPointBuf = NULL;
    }

    if(GS->gMaxBisect != NULL) {
      PyMem_Free(GS->gMaxBisect);
      GS->gMaxBisect = NULL;
    }

    if(GS->gEventTol != NULL) {
      PyMem_Free(GS->gEventTol);
      GS->gEventTol = NULL;
    }

    if(GS->gEventDelay != NULL){
      PyMem_Free(GS->gEventDelay);
      GS->gEventDelay = NULL;
    }
    if(GS->gEventInterval != NULL) {
      PyMem_Free(GS->gEventInterval);
      GS->gEventInterval = NULL;
    }

    if(GS->gEventTerm != NULL) {
      PyMem_Free(GS->gEventTerm);
      GS->gEventTerm = NULL;
    }

    if(GS->gEventDir != NULL) {
      PyMem_Free(GS->gEventDir);
      GS->gEventDir = NULL;
    }

    if(GS->gEventActive != NULL) {
      PyMem_Free(GS->gEventActive);
      GS->gEventActive = NULL;
    }

    if(GS->gEventY != NULL) {
      PyMem_Free(GS->gEventY);
      GS->gEventY = NULL;
    }

    GS->maxEvtPts = 0;
    GS->eventBufIdx = 0;
    GS->haveActive = 0;
    GS->activeTerm = 0;
    GS->activeNonTerm = 0;

    GS->isInitEvents = 0;
  }
}

void CleanupAll( IData *GS, double *ICs, double **Bds ) {
  int i = 0;

  if( Bds != NULL ) {
    for( i = 0; i < 2; i++ ) {
      if( Bds[i] != NULL ) {
	PyMem_Free( Bds[i] );
	Bds[i] = NULL;
      }
    }
    PyMem_Free( Bds );
    Bds = NULL;
  }

  if( ICs != NULL ) {
    PyMem_Free( ICs );
    ICs = NULL;
  }

  if( GS != NULL ) {
    CleanupRunParams( GS );
    CleanupExtInputs( GS );
    CleanupEvents( GS );
    CleanupIData( GS );
    CleanupBasic( GS );

    PyMem_Free(GS);
    GS = NULL;
 }
}

int SetContParams( IData *GS, double tend, double *pars, double **Bds, double *upperBounds, double *lowerBounds ) {

  int i;

  assert(GS);
  assert(pars);
  assert(GS->gParams);
  assert(GS->hasRun != 0);

  GS->tStart = GS->tEnd;
  GS->tEnd = tend;

  for( i = 0; i < GS->paramDim; i++ ) {
    GS->gParams[i] = pars[i];
  }

  for( i = 0; i < GS->phaseDim + GS->paramDim; i++ ) {
    Bds[0][i] = lowerBounds[i];
    Bds[1][i] = upperBounds[i];
  }

  return SUCCESS;
}

int ResetIndices( IData *GS ) {
  int i;

  assert( GS );

  /* From internal state */
  GS->hasRun = 0;

  /* From Integ */
  GS->timeIdx = 0;
  GS->pointsIdx = 0;


  /* From Events */
  GS->eventBufIdx = 0;
  GS->eventFound = 0;

  if( GS->haveActive > 0 ) {
    assert(GS->gCheckableEventCounts);
    for( i = 0; i < GS->haveActive; i++ ) {
      GS->gCheckableEventCounts[i] = 0;
    }
  }

  /* From Run Params */
  GS->refineBufIdx = 0;
  GS->specTimesIdx = 0;

  /* From Aux */
  GS->auxIdx = 0;

  /* From Ext Inputs */
  if( GS->nExtInputs > 0 ) {
    if( GS->gCurrentExtInputIndex != NULL ) {
      for( i = 0; i < GS->nExtInputs; i++ ) {
	GS->gCurrentExtInputIndex[i] = 0;
      }
    }
  }

  return SUCCESS;
}

void BlankIData( IData *GS ) {

  assert(GS);

  /* Internal state flags */
  GS->isInitBasic = 0;
  GS->isInitIntegData = 0;
  GS->isInitEvents = 0;
  GS->isInitExtInputs = 0;
  GS->isInitRunParams = 0;
  GS->hasRun = 0;

  /* Basic parameters */
  GS->phaseDim = 0;
  GS->paramDim = 0;
  GS->hasJac = 0;
  GS->hasJacP = 0;
  GS->hasMass = 0;
  GS->gIC = NULL;
  GS->nEvents = 0;
  GS->gEventFcnArray = NULL;
  GS->nAuxVars = 0;
  GS->extraSpaceSize = 0;
  GS->gExtraSpace = NULL;
  GS->gJacPtrs = NULL;
  GS->gMassPtrs = NULL;

  /* Point saving (initInteg) */
  GS->maxPts = 0;
  GS->timeIdx = 0;
  GS->pointsIdx = 0;
  GS->lastTime = 0;
  GS->lastPoint = NULL;
  GS->gATol = NULL;
  GS->gRTol = NULL;
  GS->gPoints = NULL;
  GS->gTimeV = NULL;
  GS->gYout = NULL;
  GS->cContSolFun = NULL;

  /* Run parameters */
  GS->tStart = 0;
  GS->tEnd = 0;
  GS->refine = 0;
  GS->direction = 0;
  GS->gParams = NULL;
  GS->gRefinePoints = NULL;
  GS->gRefineTimes = NULL;
  GS->refineBufIdx = 0;
  GS->gSpecTimes = NULL;
  GS->specTimesLen = 0;
  GS->specTimesIdx = 0;
  GS->calcSpecTimes = 0;
  GS->calcAux = 0;

  /* Events */
  GS->maxEvtPts = 0;
  GS->haveActive = 0;
  GS->activeTerm = 0;
  GS->activeNonTerm = 0;
  GS->gTermIndices = NULL;
  GS->gNonTermIndices = NULL;
  GS->gEventActive = NULL;
  GS->gEventDir = NULL;
  GS->gEventTerm = NULL;
  GS->gMaxBisect = NULL;
  GS->gEventTol = NULL;
  GS->gEventDelay = NULL;
  GS->gEventInterval = NULL;
  GS->gTempPoint = NULL;
  GS->gNTEvtFound = NULL;
  GS->gNTEvtFoundOrder = NULL;
  GS->gNTEvtFoundTimes = NULL;
  GS->gCheckableEvents = NULL;
  GS->gCheckableEventCounts = NULL;
  GS->eventFound = 0;
  GS->eventT = 0;
  GS->gEventY = NULL;
  GS->gEventPoints = NULL;
  GS->gEventTimes = NULL;
  GS->gEventPointBuf = NULL;
  GS->gEventTimeBuf = NULL;
  GS->eventBufIdx = 0;
  GS->eventNearCoef = 0;

  /* Aux variables */
  GS->gAuxPoints = NULL;
  GS->auxIdx = 0;

  /* External inputs */
  GS->nExtInputs = 0;
  GS->gExtInputLens = NULL;
  GS->gExtInputVals = NULL;
  GS->gExtInputTimes = NULL;
  GS->gCurrentExtInputVals = NULL;
  GS->gCurrentExtInputIndex = NULL;

  /* Radau specific */
#ifdef __RADAU__
  GS->workArrayLen = 0;
  GS->intWorkArrayLen = 0;
  GS->gWorkArray = NULL;
  GS->gIntWorkArray = NULL;

  GS->ijac = 0;
  GS->jacLowerBandwidth = 0;
  GS->jacUpperBandwidth = 0;

  GS->imas = 0;
  GS->masLowerBandwidth = 0;
  GS->masUpperBandwidth = 0;

  GS->contdim = NULL;
  GS->contdata = NULL;
#endif

  /* Dopri specific */
#ifdef __DOPRI__
  GS->checkBounds = 0;
  GS->boundsCheckMaxSteps = 0;
  GS->gMagBound = NULL;
#endif

}


void setJacPtrs( IData *GS, double *jacSpace ) {
  int i;
  assert( GS );

  if( GS->hasJac == 1 && GS->phaseDim > 0 ) {
    assert(GS->gJacPtrs);
    assert(jacSpace);
    for( i = 0; i < GS->phaseDim; i++ ) {
      GS->gJacPtrs[i] = &jacSpace[i*GS->phaseDim];
    }
  }
}

/* Currently assumes full mass matrix */
void setMassPtrs( IData *GS, double *massSpace ) {
  int i;
  assert( GS );

  if( GS->hasMass == 1 && GS->phaseDim > 0 ) {
    assert(GS->gMassPtrs);
    assert(massSpace);
    for( i = 0; i < GS->phaseDim; i++ ) {
      GS->gMassPtrs[i] = &massSpace[i*GS->phaseDim];
    }
  }
}



#ifdef __RADAU__
int InitializeRadauOptions( IData *GS, double uround, double safety,
			    double jacRecompute, double newtonStop,
			    double stepChangeLB, double stepChangeUB,
			    double hmax, double stepSizeLB, double stepSizeUB,
			    int hessenberg, int maxSteps, int maxNewton,
			    int newtonStart, int index1dim, int index2dim,
			    int index3dim, int stepSizeStrategy,
			    int DAEstructureM1, int DAEstructureM2 ) {

  int i;

  assert( GS );
  if( GS->isInitIntegData != 1 )
    return FAILURE;

  for( i = 0; i < 20; i++ ) {
    GS->gWorkArray[i] = 0;
    GS->gIntWorkArray[i] = 0;
  }

  GS->gWorkArray[0] = uround; /* Rounding unit, default 1e-16 */
  GS->gWorkArray[1] = safety; /* Stepsize prediction safety factor, default 0.9 */
  GS->gWorkArray[2] = jacRecompute; /* Parameter to determine when jacobian should
				      be recomputed. Set ~= 0.1 when jacobian evals
				      are costly. Set <= 0.001 for small systems.
				      Set < 0 to force recomputation at every step.
				      Default is 0.001 */
  GS->gWorkArray[3] = newtonStop; /* Stopping criterion for Newton's method, should be
				    < 1. Smaller values mean slower computation, but
				    safer. Default is min(0.03, sqrt(rtol[0])) */
  GS->gWorkArray[4] = stepChangeLB;
  GS->gWorkArray[5] = stepChangeUB; /* If stepChangeLB < hnew/hold < stepChangeUB,
				      then stepsize is not changed. Together with
				      large jacRecompute, this saves LU decomps
				      and compute times for large systems. For small
				      systems, stepChangeLB = 1, stepChangeUB = 1.2
				      might be good; for large full systems,
				      stepChangeLB = 0.99, stepChangeUB = 2.0
				      might be good. Default is stepChangeLB = 1,
				      stepChangeUB = 1.2 */

  GS->gWorkArray[6] = hmax; /* Max stepsize, default = tend - t0 */
  GS->gWorkArray[7] = stepSizeLB;
  GS->gWorkArray[8] = stepSizeUB; /* New stepsize is chosen subject to
				    stepSizeLB < hnew/hold < stepSizeUB,
				    Default is stepSizeLB = 0.2,
				    stepSizeUB = 8.0 */

  GS->gIntWorkArray[0] = hessenberg; /* If != 0, transforms jacobian to Hessenberg
				     form. Good for large systems with full
				     jacobian. Does not work for banded jacobian
				     (mljac < phaseDim) or implicit (imas = 1)
				     systems. */
  GS->gIntWorkArray[1] = maxSteps; /* Max number of steps. If 0, default is
				   100000 */
  GS->gIntWorkArray[2] = maxNewton; /* Max number of newton iterations per step
				    to solve implicit systems. Default is 7 */
  GS->gIntWorkArray[3] = newtonStart; /* If == 0, starting value for newton's method
				      is extrapolated collocation solution.
				      If != 0, zero starting values used. This is
				      recommended if newton has difficulty converging.
				      (e.g. when Nstep > Naccpt + Nreject in output
				      stats) */
  GS->gIntWorkArray[4] = index1dim; /* Dimension of index 1 variables. Must be > 0.
				    For ODEs, this is phaseDim. Default is phaseDim.
				 */
  GS->gIntWorkArray[5] = index2dim; /* Dimension of index 2 variables. Default 0 */
  GS->gIntWorkArray[6] = index3dim; /* Dimension of index 3 variables. Default 0 */

  GS->gIntWorkArray[7] = stepSizeStrategy; /* Step size selection strategy.
					   If == 1, Mod. Predictive controller
					   (Gustafsson)
					   If == 2, Classical step size control
					   1 seems safer; 2 produces faster runs
					   on simple problems. */
  GS->gIntWorkArray[8] = DAEstructureM1; /* DAE Structure M1. See radau5.f header.
					 Default 0. */
  GS->gIntWorkArray[8] = DAEstructureM2; /* DAE Structure M2. See radau5.f header.
					  Default = M1. */


  return SUCCESS;

}
#endif

#ifdef __DOPRI__
int InitializeDopriOptions( IData *GS, int checkBounds, int boundsCheckMaxSteps,
			    double *magBound) {

  int i;

  assert( GS );
  if( GS->isInitIntegData != 1 )
    return FAILURE;

  assert(magBound);

  GS->checkBounds = checkBounds;
  GS->boundsCheckMaxSteps = boundsCheckMaxSteps;

  for( i = 0; i < GS->phaseDim; i++ ) {
    GS->gMagBound[i] = magBound[i];
  }

  return SUCCESS;
}

#endif
