#include "interface.h"
extern int N_AUXVARS;
extern int N_EVENTS; 
extern int N_EXTINPUTS;

extern EvFunType *gEventFunction;
extern ContSolFunType gContSolFun;

extern IData *gIData;
extern double *gICs;
extern double globalt0;
extern double **gBds;

PyObject* Vfield(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyArrayObject *PointsOut = NULL;
  
  double *ftemp = NULL;
  int i; 

  /* for NumArray compatibility */
  import_libnumarray();

  if( (gIData == NULL) || (gIData->isInitBasic == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else if( (gIData->nExtInputs > 0) && (gIData->isInitExtInputs == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else {
    OutObj = PyTuple_New(1);
    assert(OutObj);

    ftemp = (double *)PyMem_Malloc((gIData->phaseDim)*sizeof(double));
    assert(ftemp);

    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    vfieldfunc(gIData->phaseDim, gIData->paramDim, t, x, p, ftemp, 
	       gIData->extraSpaceSize, gIData->gExtraSpace, 
	       gIData->nExtInputs, gIData->gCurrentExtInputVals);
    
    PointsOut = NA_NewArray(NULL, tFloat64, 1, gIData->phaseDim);
    assert(PointsOut);
    for( i = 0; i < gIData->phaseDim; i++ ) {
      NA_set1_Float64(PointsOut, i, ftemp[i]);
    }

    PyTuple_SetItem(OutObj, 0, (PyObject *)PointsOut);

    PyMem_Free(ftemp);
    return OutObj;
  }
}

PyObject* Jacobian(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyArrayObject *JacOut = NULL;

  double *jactual = NULL, **jtemp = NULL;
  int i, j, n;

  /* for NumArray compatibility */
  import_libnumarray();

  if( (gIData == NULL) || (gIData->isInitBasic == 0) || (gIData->hasJac == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else if( (gIData->nExtInputs > 0) && (gIData->isInitExtInputs == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else {
    OutObj = PyTuple_New(1);
    assert(OutObj);

    n = gIData->phaseDim;
    jactual = (double *)PyMem_Malloc(n*n*sizeof(double));
    assert(jactual);
    jtemp = (double **)PyMem_Malloc(n*sizeof(double *));
    assert(jtemp);
    for( i = 0; i < n; i++ ) {
      jtemp[i] = jactual;
      jactual += n;
    }
    

    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    /* Assume jacobian is returned in column-major format */
    jacobian(gIData->phaseDim, gIData->paramDim, t, x, p, jtemp, 
	     gIData->extraSpaceSize, gIData->gExtraSpace, 
	     gIData->nExtInputs, gIData->gCurrentExtInputVals);

    JacOut = NA_NewArray(NULL, tFloat64, 2, gIData->phaseDim, gIData->phaseDim);
    assert(JacOut);
    /* Return jacobian in row-major format */
    for( i = 0; i < gIData->phaseDim; i++ ) {
      for( j = 0; j < gIData->phaseDim; j++ ) {
	NA_set2_Float64(JacOut, i, j, jtemp[j][i]);
      }
    } 

    PyTuple_SetItem(OutObj, 0, (PyObject *)JacOut);

    PyMem_Free(jtemp[0]);
    PyMem_Free(jtemp);

    return OutObj;
  }
}

PyObject* JacobianP(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyArrayObject *JacPOut = NULL;

  double *jactual = NULL, **jtemp = NULL;
  int i, j, n, m;

  /* for NumArray compatibility */
  import_libnumarray();

  if( (gIData == NULL) || (gIData->isInitBasic == 0) 
      || (gIData->hasJacP == 0) || (gIData->paramDim == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else if( (gIData->nExtInputs > 0) && (gIData->isInitExtInputs == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else {
    OutObj = PyTuple_New(1);
    assert(OutObj);

    n = gIData->phaseDim;
    m = gIData->paramDim;
    jactual = (double *)PyMem_Malloc(n*m*sizeof(double));
    assert(jactual);
    jtemp = (double **)PyMem_Malloc(n*sizeof(double *));
    assert(jtemp);
    for( i = 0; i < n; i++ ) {
      jtemp[i] = jactual;
      jactual += m;
    }
    
    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    jacobianParam(gIData->phaseDim, gIData->paramDim, t, x, p, jtemp, 
		  gIData->extraSpaceSize, gIData->gExtraSpace, 
		  gIData->nExtInputs, gIData->gCurrentExtInputVals);
    
    JacPOut = NA_NewArray(NULL, tFloat64, 2, gIData->paramDim, gIData->phaseDim);
    assert(JacPOut);
    for( i = 0; i < gIData->phaseDim; i++ ) {
      for( j = 0; j < gIData->paramDim; j++ ) {
	NA_set2_Float64(JacPOut, j, i, jtemp[i][j]);
      }
    } 

    PyTuple_SetItem(OutObj, 0, (PyObject *)JacPOut);
    
    PyMem_Free(jtemp[0]);
    PyMem_Free(jtemp);
    
    return OutObj;
  }
}

PyObject* AuxFunc(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyArrayObject *AuxOut = NULL;
  
  double *ftemp = NULL;
  int i; 

  /* for NumArray compatibility */
  import_libnumarray();

  if( (gIData == NULL) || (gIData->isInitBasic == 0) || (gIData->nAuxVars == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else if( (gIData->nExtInputs > 0) && (gIData->isInitExtInputs == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else {
    OutObj = PyTuple_New(1);
    assert(OutObj);

    ftemp = (double *)PyMem_Malloc((gIData->nAuxVars)*sizeof(double));
    assert(ftemp);

    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    auxvars(gIData->phaseDim, gIData->phaseDim, t, x, p, ftemp, 
	    gIData->extraSpaceSize, gIData->gExtraSpace, 
	    gIData->nExtInputs, gIData->gCurrentExtInputVals);
    
    AuxOut = NA_NewArray(NULL, tFloat64, 1, gIData->nAuxVars);
    assert(AuxOut);
    for( i = 0; i < gIData->nAuxVars; i++ ) {
      NA_set1_Float64(AuxOut, i, ftemp[i]);
    }

    PyTuple_SetItem(OutObj, 0, (PyObject *)AuxOut);

    PyMem_Free(ftemp);
    return OutObj;
  }
}

PyObject* MassMatrix(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyArrayObject *MassOut = NULL;

  double *jactual = NULL, **jtemp = NULL;
  int i, j, n;

  /* for NumArray compatibility */
  import_libnumarray();
  
  if( (gIData == NULL) || (gIData->isInitBasic == 0) || (gIData->hasMass == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else if( (gIData->nExtInputs > 0) && (gIData->isInitExtInputs == 0) ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else {
    OutObj = PyTuple_New(1);
    assert(OutObj);
    n = gIData->phaseDim;
    jactual = (double *)PyMem_Malloc(n*n*sizeof(double));
    assert(jactual);
    jtemp = (double **)PyMem_Malloc(n*sizeof(double *));
    assert(jtemp);
    for( i = 0; i < n; i++ ) {
      jtemp[i] = jactual;
      jactual += n;
    }
    
    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    /* Assume jacobian is returned in column-major format */
    massMatrix(gIData->phaseDim, gIData->paramDim, t, x, p, jtemp, 
	       gIData->extraSpaceSize, gIData->gExtraSpace, 
	       gIData->nExtInputs, gIData->gCurrentExtInputVals);

    MassOut = NA_NewArray(NULL, tFloat64, 2, gIData->phaseDim, gIData->phaseDim);
    assert(MassOut);
    /* Return mass matrix in row-major format */
    for( i = 0; i < gIData->phaseDim; i++ ) {
      for( j = 0; j < gIData->phaseDim; j++ ) {
	NA_set2_Float64(MassOut, i, j, jtemp[j][i]);
      }
    } 

    PyTuple_SetItem(OutObj, 0, (PyObject *)MassOut);

    PyMem_Free(jtemp[0]);
    PyMem_Free(jtemp);

    return OutObj;
  }
}

/**************************************
 **************************************
 *       INITIALIZATION ROUTINES
 **************************************
 **************************************/

PyObject* InitBasic(int PhaseDim, int ParamDim, int nAux, int nEvents, int nExtInputs, 	     
		    int HasJac, int HasJacP, int HasMass, int extraSize) {

  PyObject *OutObj = NULL;
  int i = 0;

  if( gIData == NULL) {
    gIData = (IData *)PyMem_Malloc(sizeof(IData));
    assert(gIData);
    BlankIData(gIData);  
  }

  assert(nAux == N_AUXVARS);
  assert(nEvents == N_EVENTS);
  assert(nExtInputs == N_EXTINPUTS);

  if( !InitializeBasic( gIData, PhaseDim, ParamDim, nAux, nEvents, nExtInputs,
		       HasJac, HasJacP, HasMass, extraSize) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }

  if( gICs == NULL ) {
    gICs = (double *)PyMem_Malloc(gIData->phaseDim*sizeof(double));
    assert(gICs); 
  }

  if( gBds == NULL ) {
    gBds = (double **)PyMem_Malloc(2*sizeof(double *));
    assert(gBds);
    for( i = 0; i < 2; i ++ ) {
      gBds[i] = (double *)PyMem_Malloc((gIData->phaseDim + gIData->paramDim)*sizeof(double));
      assert(gBds[i]);
    }
  }

  if( !ResetIndices( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }  
  else {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  return OutObj;
}

PyObject* InitEvents( int Maxevtpts, int *EventActive, int *EventDir, int *EventTerm,
		      double *EventInterval, double *EventDelay, double *EventTol,
		      int *Maxbisect, double EventNearCoef) {
  PyObject *OutObj = NULL;

  if( (gIData == NULL) ||  gIData->isInitBasic != 1 ){
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }

  if( !InitializeEvents( gIData, Maxevtpts, EventActive, EventDir, EventTerm,
			EventInterval, EventDelay, EventTol, Maxbisect, EventNearCoef ) ){
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }
  if( !ResetIndices( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }
  else {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  return OutObj;

}

PyObject* InitInteg(int Maxpts, double *atol, double *rtol ) {
  PyObject *OutObj = NULL;

  int i = 0;

  if( (gIData == NULL) ||  gIData->isInitBasic != 1 ){
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }
  if( gICs == NULL ) {
    gICs = (double *)PyMem_Malloc(gIData->phaseDim*sizeof(double));
    assert(gICs); 
  }
  if( gBds == NULL ) {
    gBds = (double **)PyMem_Malloc(2*sizeof(double *));
    assert(gBds);
    for( i = 0; i < 2; i ++ ) {
      gBds[i] = (double *)PyMem_Malloc((gIData->phaseDim + gIData->paramDim)*sizeof(double));
      assert(gBds[i]);
    }
  }
  if( !InitIntegData( gIData, Maxpts, atol, rtol, gContSolFun ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }
  if( !ResetIndices( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }   
  else {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
    return OutObj;
  }
}

PyObject* InitExtInputs(int nExtInputs, int *extInputLens, double *extInputVals, 
			double *extInputTimes ) {
  PyObject *OutObj = NULL;


  if( (gIData == NULL) ||  gIData->isInitBasic != 1 ){
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }  
  if( !InitializeExtInputs( gIData, nExtInputs, extInputLens, extInputVals, extInputTimes ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }
  else {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
    return OutObj;
  }
}

PyObject* SetRunParameters(double *ic, double *pars, double gt0, double t0, 
			   double tend, int refine, int specTimeLen, double *specTimes, 
			   double *upperBounds, double *lowerBounds) {
  PyObject *OutObj = NULL;

  int i = 0;

  if( (gIData == NULL) ||  gIData->isInitBasic != 1 ){
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }
  if( gICs == NULL ) {
    gICs = (double *)PyMem_Malloc(gIData->phaseDim*sizeof(double));
    assert(gICs); 
  }
  if( gBds == NULL ) {
    gBds = (double **)PyMem_Malloc(2*sizeof(double *));
    assert(gBds);
    for( i = 0; i < 2; i ++ ) {
      gBds[i] = (double *)PyMem_Malloc((gIData->phaseDim + gIData->paramDim)*sizeof(double));
      assert(gBds[i]);
    }
  }
  if( !SetRunParams( gIData, pars, gICs, gBds, ic, gt0, &globalt0, 
		     t0, tend, refine, specTimeLen, specTimes, upperBounds, lowerBounds ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }
  if( !ResetIndices( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }  
  else {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
    return OutObj;
  }
}

PyObject* SetContParameters(double tend, double *pars, double *upperBounds, double *lowerBounds) {
  PyObject *OutObj = NULL;

  if( (gIData == NULL) ||  gIData->isInitBasic != 1 || gICs == NULL || gBds == NULL){
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }

  if( !SetContParams( gIData, tend, pars, gBds, upperBounds, lowerBounds ) ) {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
    return OutObj;
  }

  else {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
    return OutObj;
  }
}


/**************************************
 **************************************
 *       CLEANUP ROUTINES
 **************************************
 **************************************/


PyObject* CleanUp( void ) {
  PyObject *OutObj = NULL;
  
  if( CleanupAll( gIData, gICs, gBds ) ) {
    gIData = NULL; gICs = NULL; gBds = NULL;
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  else {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }
  return OutObj;

}

PyObject* ClearExtInputs( void ) {
  PyObject *OutObj = NULL;
  
  if( CleanupExtInputs( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  else {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }
  return OutObj;

}

PyObject* ClearEvents( void ) {
  PyObject *OutObj = NULL;
  
  if( CleanupEvents( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  else {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }
  return OutObj;

}

PyObject* ClearInteg( void ) {
  PyObject *OutObj = NULL;
  
  if( CleanupIData( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  else {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }
  return OutObj;

}

PyObject* ClearParams( void ) {
  PyObject *OutObj = NULL;

  if( CleanupRunParams( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  else {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }
  return OutObj;

}

PyObject* Reset( void ) {
  PyObject *OutObj = NULL;

  if( ResetIndices( gIData ) ) {
    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);
  }
  else {
    OutObj = Py_BuildValue("(i)", 0);
    assert(OutObj);
  }
  return OutObj;

}

/******************************
 *
 * Packout routine returns the
 * points computed by 
 * the integrator.
 *
 ******************************/

PyObject* PackOut( IData *GS, double *ICs,
		    int state, double *stats, double hlast, int idid ) {
  int i, j;
  int numEvents = 0;

  /* Python objects to be returned at end of integration */
  PyObject *OutObj = NULL; /* Overall PyTuple output object */ 
  PyArrayObject *TimeOut = NULL; /* Trajectory times */
  PyArrayObject *PointsOut = NULL; /* Trajectory points */
  PyArrayObject *StatsOut = NULL; /* */
  PyObject *hout = NULL;
  PyObject *idout = NULL;

  PyObject *EventPointsOutTuple = NULL;
  PyObject *EventTimesOutTuple = NULL;

  PyArrayObject **EventPointsOutArray = NULL;
  PyArrayObject **EventTimesOutArray = NULL;
    
  PyArrayObject *AuxOut = NULL;

  assert(GS);

  if( state == FAILURE ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  
 /* for NumArray compatibility */
  import_libnumarray();

  EventPointsOutTuple = PyTuple_New(GS->nEvents);
  assert(EventPointsOutTuple);
  EventTimesOutTuple = PyTuple_New(GS->nEvents); 
  assert(EventTimesOutTuple);

  
  /* Copy out points and then PyMem_Free them */
  PointsOut = NA_NewArray(NULL, tFloat64, 2, GS->phaseDim, GS->pointsIdx); 
  assert(PointsOut);
  /* WARNING -- modified the order of the dimensions here! */
  for(i = 0; i < GS->pointsIdx; i++) {
    for(j = 0; j < GS->phaseDim; j++) {
      NA_set2_Float64(PointsOut, j, i, GS->gPoints[i][j]);
    }
  }
  
  /* Copy out times and the PyMem_Free them */
  TimeOut = NA_NewArray(NULL, tFloat64, 1, GS->timeIdx);
  assert(TimeOut);
  for( i = 0; i < GS->timeIdx; i++ ) {
    NA_set1_Float64(TimeOut, i, GS->gTimeV[i]);
  }

  /* Copy out auxilliary points */
  if( GS->nAuxVars > 0 && GS->calcAux == 1 ) {
    /* WARNING: The order of the dimensions 
       is switched here! */
    AuxOut = NA_NewArray(NULL, tFloat64, 2, GS->nAuxVars, GS->pointsIdx);
    assert(AuxOut);
    for( i = 0; i < GS->pointsIdx; i++ ) {
      for( j = 0; j < GS->nAuxVars; j++ ) {
	NA_set2_Float64(AuxOut, j, i, GS->gAuxPoints[i][j]);
      }
    }
  }
  
  /* Allocate and copy out integration stats */
  /* Number of stats different between Radau, Dopri */
  StatsOut = NA_NewArray(stats, tFloat64, 1, 7);
  assert(StatsOut);

  /* Setup the tuple for the event points and times (separate from trajectory).
     Must remember to keep the reference count for Py_None up to date*/
  for( i = 0; i < GS->nEvents; i++ ) {
    PyTuple_SetItem(EventPointsOutTuple, i, Py_None);
    Py_INCREF(Py_None);
    PyTuple_SetItem(EventTimesOutTuple, i, Py_None);
    Py_INCREF(Py_None);
  }
  
  /* Count the number of events for which something was caught */
  numEvents = 0;
  for( i = 0; i < GS->haveActive; i++ ) {
    if( GS->gCheckableEventCounts[i] > 0 ) {
      numEvents++;
    }
  }
  /* Only allocate memory for events if something was caught */
  if( numEvents > 0 ) {
    /* Allocate separate arrays for each event */
    EventPointsOutArray = PyMem_Malloc(sizeof(PyArrayObject *) * numEvents);
    assert(EventPointsOutArray);
    EventTimesOutArray = PyMem_Malloc(sizeof(PyArrayObject *) * numEvents);
    assert(EventTimesOutArray);
    
    /* Lower reference count for Py_None from INCREFs in initialization of
       OutTuples above. Decrease by 2*numevents) */
    for( i = 0; i < numEvents; i++ ) {
      Py_DECREF(Py_None);
      Py_DECREF(Py_None);
    }

    for( i = 0, j = 0; i < GS->haveActive; i++ ) {
      /* i tracks which of the active events we are (possibly) saving,
	 j tracks which of the python arrays we are (possibly) storing the ith
	 event's points in. */
      if( GS->gCheckableEventCounts[i] > 0 ) {
	int k, l;
	/* Get the number of points caught for this event, and which one (in list
	   of all active and nonactive events) it is */
	int EvtCt = GS->gCheckableEventCounts[i], EvtIdx = GS->gCheckableEvents[i];
	
	/* Copy the event times, points into a python array */
	EventTimesOutArray[j] = NA_NewArray(NULL, tFloat64, 1, EvtCt);
	assert(EventTimesOutArray[j]);
	/* WARNING -- modified the order of the dimensions here! */
	EventPointsOutArray[j] = NA_NewArray(NULL, tFloat64, 2, GS->phaseDim, EvtCt);
	assert(EventPointsOutArray[j]);
	for( k = 0; k < EvtCt; k++ ) {
	  NA_set1_Float64(EventTimesOutArray[j], k, GS->gEventTimes[i][k]);
	  for( l = 0; l < GS->phaseDim; l++ ) {
	    NA_set2_Float64(EventPointsOutArray[j], l, k, GS->gEventPoints[i][l][k]);
	  }
	}
	/* The returned python tuple has slots for all events, not just active
	   ones, which is why we insert into the tuple at position EvtIdx */
	PyTuple_SetItem(EventPointsOutTuple, EvtIdx, 
			(PyObject *)(EventPointsOutArray[j]));
	PyTuple_SetItem(EventTimesOutTuple, EvtIdx, 
			(PyObject *)(EventTimesOutArray[j]));
	j++; /* Only go to the next python array if we saved something for the ith event.
		Otherwise, for loop will take us to next event to see if empty or not. */
      }
    }
  }

  /* Pack points, times, stats, events, etc. into a 7 tuple */
  OutObj = PyTuple_New(8);
  assert(OutObj);

  PyTuple_SetItem(OutObj, 0, (PyObject *)TimeOut);

  PyTuple_SetItem(OutObj, 1, (PyObject *)PointsOut);

  if( GS->nAuxVars > 0 && GS->calcAux == 1) {
    PyTuple_SetItem(OutObj, 2, (PyObject *)AuxOut);
  }
  else {
    PyTuple_SetItem(OutObj, 2, Py_None);
    Py_INCREF(Py_None);
  }

  PyTuple_SetItem(OutObj, 3, (PyObject *)StatsOut);

  PyTuple_SetItem(OutObj, 4, PyFloat_FromDouble(hlast));

  PyTuple_SetItem(OutObj, 5, PyFloat_FromDouble((double)idid));

  PyTuple_SetItem(OutObj, 6, EventTimesOutTuple);

  PyTuple_SetItem(OutObj, 7, EventPointsOutTuple);


  return OutObj;
}


