/**************************************
 **************************************
 *            EVENT FINDING
 **************************************
 **************************************/
#include "eventFinding.h"

double dummyevent(unsigned n, double x, double *y, double *p) {
    return 0.0;
}

/* Takes IData struct (global), previous integrated time point, current integrated time
   point, current integrated trajectory point, pointer to error return code.
   Returns -(Total Number of events found) if a terminal event occurred,
   +(Total Number of events found) otherwise.
   Main Event detection loop: Tries each active terminal event, then each active
   nonterminal event, saving events that are caught into the appropriate
   event arrays as well as the event point buffer for inclusion in the trajectory
   later. */
int DetectEvents(IData *GS, double xold, double x, double *y, int *irtrn, int *termFound) {
  int EvtIdx, EvtCt, CheckIdx;
  double dir = x - xold; /* Determine if integrating forwards or backwards in time */
  int FirstTerm = -1; /* Index of first (temporally) terminal event */
  int i, j, sortedIdx;
  int NTEvtFoundCt = 0; /* Number of non-terminal events caught */
  double *point;
  double time;

  point = GS->gTempPoint;

  /* Reset eventBufIdx to ensure we record to correct position. */
  GS->eventBufIdx = 0;

  /* GS->lastTime should be set to the last valid time already */
  /* Check terminal events first */
  if( GS->activeTerm > 0 ) {
    for( i = 0; i < GS->activeTerm; i++ ) {
      /* Shorthand for the index into Events of the terminal event we're checking */
      EvtIdx = GS->gCheckableEvents[GS->gTermIndices[i]];

      /* Shorthand for the count of the terminal event we're checking;
	 should always be 0 */
      EvtCt = GS->gCheckableEventCounts[GS->gTermIndices[i]];

      /* Index into CheckableEvents of the terminal event we're checking */
      CheckIdx = GS->gTermIndices[i];

      /* At each check, write to eventT,Y. */
      if( CheckEventFctn(GS, xold, EvtIdx, EvtCt, CheckIdx ) ) {

	/* See if this point is sooner than the previous one */
	if( ( dir > 0 ) ? ( GS->eventT < GS->lastTime ) :
	    ( GS->eventT > GS->lastTime ) ) {

	  /* If this point is sooner than the previous, save it as the
	     current last point */
	  GS->lastTime = GS->eventT;

	  for( j = 0; j < GS->phaseDim; j++ ) {
	    GS->lastPoint[j] = GS->gEventY[j];
	  }
	  /* Record the index of this event as the first terminal event */
	  FirstTerm = i;
	}
      }
    }
  }

  /* Save the soonest terminal event in the event points array at the
     event's index */
  if( FirstTerm >= 0 ) {
    *irtrn = -5; /* Set Terminal event code */

    /* Count of any terminal event should always be 0, but check anyway */
    EvtCt = GS->gCheckableEventCounts[GS->gTermIndices[FirstTerm]];

    /* Don't record more than the max num of points */
    if( EvtCt >= GS->maxEvtPts ) {
      *irtrn = -7; /* Too many event points return code */
      *termFound = 0;
      return -1;
    }
    else {
      GS->gEventTimes[GS->gTermIndices[FirstTerm]][EvtCt] = GS->lastTime;
      for( i = 0; i < GS->phaseDim; i++ ) {
	GS->gEventPoints[GS->gTermIndices[FirstTerm]][i][EvtCt] = GS->lastPoint[i];
      }
      GS->gCheckableEventCounts[GS->gTermIndices[FirstTerm]]++;
    }
  }


  /* Check nonterminal events between xold and latest point. Note that
     if we've found a terminal event above, the latest point is that terminal
     event. */
  for( i = 0; i < GS->activeNonTerm; i++ ) {
    /* Index into Events of nonterminal event we're checking */
    EvtIdx = GS->gCheckableEvents[GS->gNonTermIndices[i]];

    /* Current count of the nonterminal event we're checking */
    EvtCt = GS->gCheckableEventCounts[GS->gNonTermIndices[i]];
    /* Index into CheckableEvents of the nonterminal event we're checking */
    CheckIdx = GS->gNonTermIndices[i];

    if( CheckEventFctn(GS, xold, EvtIdx, EvtCt, CheckIdx) ) {
      if( EvtCt >= GS->maxEvtPts ) {
	*irtrn = -7; /* Too many event points return code */
	return -1;
      }
      else {
	/* Save the index gNonTermIndices[i] of the found event; this
	 array will be sorted temporally later */
	GS->gNTEvtFound[NTEvtFoundCt] = GS->gNonTermIndices[i];

	/* Just add the current tally of found nonterminal events at the
	   next position in the Order array. This array will be used to
	   sort events temporally later -- as indices into NTEvtFound */
	GS->gNTEvtFoundOrder[NTEvtFoundCt] = NTEvtFoundCt;

	/* Save the actual time of the current NT event found */
	GS->gNTEvtFoundTimes[NTEvtFoundCt] = GS->eventT;

	/* Track the number of NT events we've found. */
	NTEvtFoundCt++;

	/* Save the NT event's time and point in the Event arrays; increment its count */
	GS->gEventTimes[GS->gNonTermIndices[i]][EvtCt] = GS->eventT;
	for( j = 0; j < GS->phaseDim; j++ ) {
	  GS->gEventPoints[GS->gNonTermIndices[i]][j][EvtCt] = GS->gEventY[j];
	}
	GS->gCheckableEventCounts[GS->gNonTermIndices[i]]++;
      }
    }
  }

  /* If nonterminal events were found, insert them into the trajectory points
     array in the correct order */
  if( NTEvtFoundCt > 0 ) {
    /* If just one non-terminal event, no need to sort */
     if ( NTEvtFoundCt == 1 ) {
       time = GS->gEventTimes[GS->gNTEvtFound[0]]
	 [GS->gCheckableEventCounts[GS->gNTEvtFound[0]]-1];
      /* Note we have to do the extra copy because of where the j is */
      for( j = 0; j < GS->phaseDim; j++ ) {
	point[j] = GS->gEventPoints[GS->gNTEvtFound[0]][j]
	  [GS->gCheckableEventCounts[GS->gNTEvtFound[0]]-1];
      }

      BufferEventPoint(GS, time, point, GS->eventBufIdx++);

    }
    /* Otherwise, we need to sort the NT points temporally before inserting them
       into the trajectory. */
    else {
      /* sort the points according to time */
      qsort((void *) GS->gNTEvtFoundOrder, (size_t) NTEvtFoundCt, sizeof(int), CompareEvents);

      /* if we are integrating forward, insert in ascending order */
      if( (x-xold) > 0 ) {
	for( i = 0; i < NTEvtFoundCt; i++ ) {
	  /* Get the index into events of the next (temporally) found NT event */
	  sortedIdx = GS->gNTEvtFound[GS->gNTEvtFoundOrder[i]];

	  /* Save this NT point into the trajectory */
	  time = GS->gEventTimes[sortedIdx][GS->gCheckableEventCounts[sortedIdx]-1];
	  for( j = 0; j < GS->phaseDim; j++ ) {
	    point[j] = GS->gEventPoints[sortedIdx][j]
	      [GS->gCheckableEventCounts[sortedIdx]-1];
	  }
	  /* Write out the point */
	  BufferEventPoint(GS, time, point, GS->eventBufIdx++);
	}
      }
      /* otherwise, in descending order */
      else {
      	for( i = NTEvtFoundCt-1; i >= 0; i-- ) {
	  /* Get the index into events of the next (temporally) found NT event */
	  sortedIdx = GS->gNTEvtFound[GS->gNTEvtFoundOrder[i]];

	  /* Save this NT point into the trajectory */
	  time = GS->gEventTimes[sortedIdx][GS->gCheckableEventCounts[sortedIdx]-1];
	  for( j = 0; j < GS->phaseDim; j++ ) {
	    point[j] = GS->gEventPoints[sortedIdx]
	      [j][GS->gCheckableEventCounts[sortedIdx]-1];
	  }
	  /* Write out the point */
	  BufferEventPoint(GS, time, point, GS->eventBufIdx++);
	}
      }
    }
  }

  /* Add in the terminal event, at the end of the trajectory points array, if necessary */
  if( FirstTerm >= 0 ) {
    /* Count of any terminal event should always be 0, but check anyway */
    EvtCt = GS->gCheckableEventCounts[GS->gTermIndices[FirstTerm]];

    time = GS->gEventTimes[GS->gTermIndices[FirstTerm]][EvtCt-1];
    for( j = 0; j < GS->phaseDim; j++ ) {
      point[j] = GS->gEventPoints[GS->gTermIndices[FirstTerm]][j][EvtCt-1];
    }
    BufferEventPoint(GS, time, point, GS->eventBufIdx++);
  }


  /* If terminal events were found, return Number of term + Non-term events found */
  if ( FirstTerm >= 0 ) {
    *termFound = 1;
    return NTEvtFoundCt + 1;
  }
  /* If no term events found, return +(Non-term events found)
     Will be 0 if no events were found. */
  else {
    *termFound = 0;
    return NTEvtFoundCt;
  }
}


/* Use this to check that (1) a given time is not too close to the
   last found event; (2) if that's ok, then whether a given event
   occurred */
int CheckEventFctn(IData *GS, double xold, int idx, int count, int CheckIdx) {
  int found = 0; /* Default that there was no event found */
  double PrevTime = 0.0;
  double x = 0.0;

  x = GS->lastTime;

  /* First check that there has been sufficient delay from t0 before
     trying to detect the event */

  if ( ( x > xold ) ? ((x - GS->tStart) > GS->gEventDelay[idx]) :
       ((GS->tStart - x) > GS->gEventDelay[idx]) ) {


    /* If this event has not been detected before, try to find it */
    if( count == 0 ) {
      if ( FindEvent(GS, idx, xold) != 0 ) {
	if( ( x > xold ) ? ((GS->eventT - GS->tStart) > GS->gEventDelay[idx]) :
	    ((GS->tStart - GS->eventT) > GS->gEventDelay[idx]) )
	  found = 1;
	else
	  found = 0;
      }
    }

    /* If this event has been detected before, check that there is
       sufficient delay before trying to find it again. */
    else {
      PrevTime = GS->gEventTimes[CheckIdx][count-1];
      /* Check event interval has elapsed (in proper direction - backwards or forwards time) */
      if ( (x > xold) ?  ((x-PrevTime) > GS->gEventInterval[idx])
	   : ((PrevTime-x) > GS->gEventInterval[idx]) ) {
	if ( FindEvent(GS, idx, xold) != 0 ) {
	  found = 1;
	}
      }
    }
  }

  if (found != 0)
    GS->eventFound = 1;

  return found;
}



//////////////NOTE TO SELF -- THIS ONE NEEDS LOTS OF WORK!!!!

int FindEvent(IData *GS, int evtIdx, double xold) {
  /*
  //    k : index of event function
  //    n : space dimension
  // xold : last time value (before this step)
  //    x : xold + h
  //    y : f(x)
  //
  // Returns : 0 if event not found.
  //         :-3 if coordinate event found
  // 	:-4 if event function found
  //            (the neg. signs signals DOPRI to stop)
  //
  // Sets:  xfound, yfound to the x,y values where event was found
  */

  double dx, jdx, targetval, oldval, newval;
  unsigned i,j;
  int eventsign;

  /* Don't check if first point */
  if (GS->timeIdx < 1)
    return 0;

  /* targetval is left over from when coordinate events were distinct from
     function events. We will leave it in case separating them is desired later.
     The target value for any event function is 0.0, marking the event crossing. */

  targetval = 0.0;
  eventsign = GS->gEventDir[evtIdx];

  /* Value of the event function at the current integrated point */
  FillCurrentExtInputValues(GS, GS->lastTime);
  newval = (GS->gEventFcnArray[evtIdx])(GS->phaseDim, GS->lastTime, GS->lastPoint,
					GS->gParams, GS->extraSpaceSize, GS->gExtraSpace,
					GS->nExtInputs, GS->gCurrentExtInputVals);
  /* Value of the event function at the previous integrated point (or previous event point,
     depending */
  FillCurrentExtInputValues(GS, GS->gTimeV[GS->timeIdx-1]);
  oldval = (GS->gEventFcnArray[evtIdx])(GS->phaseDim, GS->gTimeV[GS->timeIdx-1],
					GS->gPoints[GS->timeIdx-1], GS->gParams,
					GS->extraSpaceSize, GS->gExtraSpace,
					GS->nExtInputs, GS->gCurrentExtInputVals);


  /*  Check for event right at LastTime  */
  /* There is potential double-saving problem here, but it should occur
     so infrequently that we don't have to deal with it now. 9/20/05 */
  if ( (newval == targetval) &&
       ( (eventsign >= 0 && newval > oldval) ||
	 (eventsign <= 0 && newval < oldval) ))    {

    /* If there was an event, save its time and value */
    GS->eventT = GS->lastTime;
    for( i = 0; i < GS->phaseDim; i++ ) {
      GS->gEventY[i] = GS->lastPoint[i];
    }
    return 1;
  }

  /* Check for event on interval (xold, LastTime) */
  if ( ((newval-targetval)*(oldval-targetval) < 0) &&
       ((eventsign == 0) ||
	(eventsign > 0 && newval > targetval ) ||
	(eventsign < 0 && newval < targetval ) ) ){

    BinSearch(GS, evtIdx, xold, GS->lastTime, oldval, newval, targetval);

    /* Record the point at the event time found by BinSearch using
       contsol interpolation provided by integrator. */
    for( i = 0; i < GS->phaseDim; i++ ) {
	GS->gEventY[i] = GS->cContSolFun(i, GS->eventT);
    }
    return 2;
  }

  /* near an event do a double check:              */
  /* check for event in sub-intervals of (xold, x)  */
  if (fabs(newval - targetval) < GS->eventNearCoef * GS->gATol[evtIdx] ) {
      dx = (GS->lastTime - xold)/SUBINTERVALS;
      jdx = xold + dx;
      for (j = 1; j < SUBINTERVALS; j++, jdx += dx, oldval = newval)
	{
	  /* compute newval at jdx = xold + j*dx */
	  for (i = 0; i < GS->phaseDim; i++) {
	    GS->gEventY[i] = GS->cContSolFun(i, jdx);
	  }

	  /* Calculate new event function value at this subinterval time pt */
	  FillCurrentExtInputValues(GS, jdx);
	  newval = (GS->gEventFcnArray[evtIdx])(GS->phaseDim, jdx,
						GS->gEventY, GS->gParams,
						GS->extraSpaceSize, GS->gExtraSpace,
						GS->nExtInputs, GS->gCurrentExtInputVals);

	  /* check for event right on jdx */
	  if ( (newval == targetval ) &&
	       ((eventsign >= 0 && newval > oldval) ||
		(eventsign <= 0 && newval < oldval) )) {
	    /* if the event occured at time jdx, we have already saved the event point
	       value, just save the time */
	    GS->eventT = jdx;
	    return 3;
	  }

	  /* Check for event on (xold+(j-1)*dx, xold+j*dx) */
	  if ( ((newval-targetval)*(oldval-targetval) < 0) &&
	       ((eventsign == 0) ||
		(eventsign > 0 && newval > targetval ) ||
		(eventsign < 0 && newval < targetval ) )) {

	    BinSearch(GS, evtIdx, jdx-dx, jdx, oldval, newval, targetval);
	    /* Save the point at the time set by Binsearch */
	    for( i = 0; i < GS->phaseDim; i++ ) {
	      GS->gEventY[i] = GS->cContSolFun(i,GS->eventT);
	    }
	    return 4;
	  }
	}  /* for subintervals */
  }  /* if near event */

  return 0; /* ==> no events found */
}


/* Flag tells whether a coordinate event or not -- now obsolete */
/* Question? What to do with xfound -- was a 'global' object variable */

void BinSearch(IData *GS, int evtIdx, double x1, double x2, double v1, double v2,
		 double vo) {
  unsigned safecnt=0, i;
  double x3, v3;

  while ( safecnt++ < GS->gMaxBisect[evtIdx]) {
     /* NB: 1e-9 may be arbitary limit to be set elsewhere */
     if (fabs(v2-v1) < BINSEARCH_LIMIT) {
       x3 = (x1+x2)/2;
     }
     else {
       x3 = x1 + (vo - v1) * (x2 - x1) / (v2-v1);
     }
     /* We only have function events */

     for (i=0; i < GS->phaseDim; i++) {
       GS->gEventY[i] = GS->cContSolFun(i, x3);
     }

     FillCurrentExtInputValues(GS, x3);
     v3 = (GS->gEventFcnArray[evtIdx])(GS->phaseDim, x3, GS->gEventY, GS->gParams,
				       GS->extraSpaceSize, GS->gExtraSpace,
				       GS->nExtInputs, GS->gCurrentExtInputVals);

     /* Stop the binary search if we are smaller than the event tolerance */
     if ( fabs(v3-vo) < GS->gEventTol[evtIdx] )
       break;

     if ( ( v1 > vo && vo > v3 ) || ( v1 < vo && vo < v3 ) )  {
       x2 = x3;
       v2 = v3;
     }
     else {
       x1 = x3;
       v1 = v3;
     }
   }

   /* NB: Need to adjust this to give python exception
      in case this problem below arises */

   /*   if (safecnt>= gMaxBisect[evtIdx])
	mexWarnMsgTxt("Event search limit was reached before tolerance was met.");
   */

   GS->eventT = x3;
}
