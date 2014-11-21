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
  PyObject *PointsOut = NULL;
  
  double *ftemp = NULL;
  int i; 

  import_array();

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
    
    npy_intp dims[1] = {gIData->phaseDim};
    PointsOut = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, ftemp);
    if(!PointsOut) {
        PyMem_Free(ftemp);
        Py_INCREF(Py_None);
        return Py_None;
    }
    PyArray_UpdateFlags((PyArrayObject *)PointsOut, NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA);
    PyTuple_SetItem(OutObj, 0, PointsOut);
    return OutObj;
  }
}

PyObject* Jacobian(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyObject *JacOut = NULL;

  double *jactual = NULL, **jtemp = NULL;
  int i, j, n;

  import_array();

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
      jtemp[i] = jactual + i * n;
    }
    

    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    /* Assume jacobian is returned in column-major format */
    jacobian(gIData->phaseDim, gIData->paramDim, t, x, p, jtemp, 
	     gIData->extraSpaceSize, gIData->gExtraSpace, 
	     gIData->nExtInputs, gIData->gCurrentExtInputVals);

    PyMem_Free(jtemp);
    npy_intp dims[2] = {gIData->phaseDim, gIData->phaseDim};
    JacOut = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, jactual);
    if(JacOut) {
        PyArray_UpdateFlags((PyArrayObject *)JacOut, NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA);
        PyTuple_SetItem(OutObj, 0, PyArray_Transpose((PyArrayObject *)JacOut, NULL));
        return OutObj;
    }
    else {
        PyMem_Free(jactual);
        Py_INCREF(Py_None);
        return Py_None;
    }
  }
}


PyObject* JacobianP(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyObject *JacPOut = NULL;

  double *jactual = NULL, **jtemp = NULL;
  int i, j, n, m;

  import_array();

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
      jtemp[i] = jactual + m * i;
    }
    
    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    /* Assume jacobianParam is returned in column-major format */
    jacobianParam(gIData->phaseDim, gIData->paramDim, t, x, p, jtemp, 
		  gIData->extraSpaceSize, gIData->gExtraSpace, 
		  gIData->nExtInputs, gIData->gCurrentExtInputVals);

    PyMem_Free(jtemp);

    npy_intp dims[2] = {gIData->paramDim, gIData->phaseDim};
    JacPOut = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, jactual);
    if(JacPOut) {
        PyArray_UpdateFlags((PyArrayObject *)JacPOut, NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA);
        PyTuple_SetItem(OutObj, 0, PyArray_Transpose((PyArrayObject *)JacPOut, NULL));
        return OutObj;
    } else {
        PyMem_Free(jactual);
        Py_INCREF(Py_None);
        return Py_None;
    }
  }
}

PyObject* AuxFunc(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyObject *AuxOut = NULL;
  
  double *ftemp = NULL;
  int i; 

  import_array();

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

    npy_intp dims[1] = {gIData->nAuxVars};
    AuxOut = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, ftemp);
    if(AuxOut) {
        PyArray_UpdateFlags((PyArrayObject *)AuxOut, NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA);
        PyTuple_SetItem(OutObj, 0, AuxOut);
        return OutObj;
    }
    else {
        PyMem_Free(ftemp);
        Py_INCREF(Py_None);
        return Py_None;
    }
    return OutObj;
  }
}

PyObject* MassMatrix(double t, double *x, double *p) {
  PyObject *OutObj = NULL;
  PyObject *MassOut = NULL;

  double *mmactual = NULL, **mmtemp = NULL;
  int i, j, n;

  import_array();
  
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
    mmactual = (double *)PyMem_Malloc(n*n*sizeof(double));
    assert(mmactual);
    mmtemp = (double **)PyMem_Malloc(n*sizeof(double *));
    assert(mmtemp);
    for( i = 0; i < n; i++ ) {
      mmtemp[i] = mmactual + n * i;
    }
    
    if( gIData->nExtInputs > 0 ) {
      FillCurrentExtInputValues( gIData, t );
    }

    /* Assume massMatrix is returned in column-major format */
    massMatrix(gIData->phaseDim, gIData->paramDim, t, x, p, mmtemp, 
	       gIData->extraSpaceSize, gIData->gExtraSpace, 
	       gIData->nExtInputs, gIData->gCurrentExtInputVals);

    PyMem_Free(mmtemp);

    npy_intp dims[2] = {gIData->phaseDim, gIData->phaseDim};
    MassOut = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, mmactual);
    if(MassOut) {
        PyArray_UpdateFlags((PyArrayObject *)MassOut, NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA);
        PyTuple_SetItem(OutObj, 0, PyArray_Transpose((PyArrayObject *)MassOut, NULL));
        return OutObj;
    } else {
        PyMem_Free(mmactual);
        Py_INCREF(Py_None);
        return Py_None;
    }
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
  PyObject *TimeOut = NULL; /* Trajectory times */
  PyObject *PointsOut = NULL; /* Trajectory points */
  PyObject *StatsOut = NULL; /* */
  PyObject *hout = NULL;
  PyObject *idout = NULL;

  PyObject *EventPointsOutTuple = NULL;
  PyObject *EventTimesOutTuple = NULL;

  PyObject *AuxOut = NULL;

  assert(GS);

  if( state == FAILURE ) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  import_array();

  EventPointsOutTuple = PyTuple_New(GS->nEvents);
  assert(EventPointsOutTuple);
  EventTimesOutTuple = PyTuple_New(GS->nEvents); 
  assert(EventTimesOutTuple);

  
  /* Copy out points */
  npy_intp p_dims[2] = {GS->phaseDim, GS->pointsIdx};
  PointsOut = PyArray_SimpleNew(2, p_dims, NPY_DOUBLE);
  assert(PointsOut);
  /* WARNING -- modified the order of the dimensions here! */
  for(i = 0; i < GS->pointsIdx; i++) {
    for(j = 0; j < GS->phaseDim; j++) {
      *((double *) PyArray_GETPTR2((PyArrayObject *)PointsOut, j, i)) = GS->gPoints[i][j];
    }
  }
  
  /* Copy out times */
  npy_intp t_dims[1] = {GS->timeIdx};
  TimeOut = PyArray_SimpleNew(1, t_dims, NPY_DOUBLE);
  assert(TimeOut);
  for( i = 0; i < GS->timeIdx; i++ ) {
      *((double *) PyArray_GETPTR1((PyArrayObject *)TimeOut, i)) = GS->gTimeV[i];
  }

  /* Copy out auxilliary points */
  npy_intp au_dims[2] = {GS->nAuxVars, GS->pointsIdx};
  if( GS->nAuxVars > 0 && GS->calcAux == 1 ) {
    /* WARNING: The order of the dimensions 
       is switched here! */
    AuxOut = PyArray_SimpleNew(2, au_dims, NPY_DOUBLE);
    assert(AuxOut);
    for( i = 0; i < GS->pointsIdx; i++ ) {
      for( j = 0; j < GS->nAuxVars; j++ ) {
	*((double *) PyArray_GETPTR2((PyArrayObject *)AuxOut, j, i)) = GS->gAuxPoints[i][j];
      }
    }
  }
  
  /* Allocate and copy out integration stats */
  /* Number of stats different between Radau, Dopri */
  npy_intp s_dims[1] = {7};
  StatsOut = PyArray_SimpleNewFromData(1, s_dims, NPY_DOUBLE, stats);
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
    /* Lower reference count for Py_None from INCREFs in initialization of
       OutTuples above. Decrease by 2*numevents) */
    for( i = 0; i < numEvents; i++ ) {
      Py_DECREF(Py_None);
      Py_DECREF(Py_None);
    }

    for( i = 0; i < GS->haveActive; i++ ) {
      if( GS->gCheckableEventCounts[i] > 0 ) {
	int k, l;
	/* Get the number of points caught for this event, and which one (in list
	   of all active and nonactive events) it is */
	int EvtCt = GS->gCheckableEventCounts[i], EvtIdx = GS->gCheckableEvents[i];
    npy_intp et_dims[1] = {EvtCt};
    npy_intp ep_dims[2] = {GS->phaseDim, EvtCt};
	
	/* Copy the event times, points into a python array */
	PyObject *e_times = PyArray_SimpleNewFromData(1, et_dims, NPY_DOUBLE, GS->gEventTimes[i]);
	assert(e_times);
	PyObject *e_points = PyArray_SimpleNew(2, ep_dims, NPY_DOUBLE);
	assert(e_points);
	for( k = 0; k < EvtCt; k++ ) {
	  for( l = 0; l < GS->phaseDim; l++ ) {
          *((double *) PyArray_GETPTR2((PyArrayObject *)e_points, l, k)) = GS->gEventPoints[i][l][k];
	  }
	}
	/* The returned python tuple has slots for all events, not just active
	   ones, which is why we insert into the tuple at position EvtIdx */
	PyTuple_SetItem(EventPointsOutTuple, EvtIdx, e_points);
	PyTuple_SetItem(EventTimesOutTuple, EvtIdx, e_times);
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


