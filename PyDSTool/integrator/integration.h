#ifndef __INTEGRATION__
#define __INTEGRATION__

#include "vfield.h"
#include "events.h"
#include <stdio.h>
#include "Python.h"
#include <math.h>
#include <numpy/arrayobject.h>

/* Somewhat arbitrary round-off tolerance */
#define ROUND_TOL 2.3e-13

#define FAILURE 0
#define SUCCESS 1
#define EVENT_NEARNESS_FACTOR 1000.0

/* Structure to hold trajectory data calculated by the 
   integration routine, including event points, refinement points
   and auxilliary function calculations. Expected to be used as 
   a global object. 
   
   Naming convention: Pointers named "gXXXX" are to be malloc'd in 
   an initialize function and free'd later.
   Pointers named "cXXXX" hold addresses of items allocated elsewhere
*/


typedef struct IntegratorData {

  /****************************/
  /* Internal state flags     */
  /****************************/
  int isInitBasic;
  int isInitIntegData;
  int isInitEvents;
  int isInitExtInputs;
  int isInitRunParams;
  int hasRun;

  /****************************/
  /* Basic parameters         */
  /****************************/
  int phaseDim; /* Phase space dimension of the vector field being integrated. */
  int paramDim; /* Dimension of parameter space of vector field being integrated. */

  int hasJac; /* Does the vectorfield definition include a jacobian function? 
		 1 = Yes, 0 = No */
  int hasJacP; /* Does the vectorfield definition include a jacobian wrt parameters function? 
		 1 = Yes, 0 = No */
  int hasMass; /* Does the vectorfield definition include a mass matrix function? 
		 1 = Yes, 0 = No */

  double *gIC; /* Will hold the initial point; at the of the integration this will hold the 
		  last integrated point */

  int nEvents;  /* Total number of event functions defined in vfield.c file */
  EvFunType *gEventFcnArray; /* Array of ptrs to event functions */

  int nAuxVars; /* Number of auxilliary variables defined in vfield.c file; length
		   of array returned by Aux func. */

  int extraSpaceSize;  /* Length of extra space array */
  double *gExtraSpace; /* 'work' array available to all functions in vfield.c file */ 

  double **gJacPtrs;   /* Pointers into 1-D array of jacobian values, for convenience */
  double **gMassPtrs;  /* Pointers into 1-D array of mass matrix values, for convenience */

  /****************************/
  /* Point saving (initInteg) */
  /****************************/
  int maxPts; /* The maximum number of points of the trajectory to calculate, including
		 event and refinement points. */

  int timeIdx; /* Current index into array of time points in the trajectory. Should
		  match PointsIdx. */

  int pointsIdx; /* Current index into array of phase space points in the trajectory. 
		    Should match TimeIdx. */

  double lastTime;   /* Last (prev) time recorded by integrator or event finding */ 
  double *lastPoint; /* Last (prev) point recorded by integrator or event finding */

  double *gATol; /* Array of absolute error tolerances; size PhaseDim */
  double *gRTol; /* Array of relative error tolerances; size PhaseDim */

  double **gPoints; /* Array where integrated points, event points, refine points, etc. saved */
  double *gTimeV;   /* Array where integrated times, event times, refine times, etc. saved */
  double *gYout;    /* Temp storage area for outputing points to gPoints array when saving */

  ContSolFunType cContSolFun; /* Pointer to continuous solution function (used in solout, events) */

  /****************************/
  /* Run parameters           */
  /****************************/
  double tStart; /* The starting time for integration. */
  double tEnd;   /* End time for integration */
  int refine; /* Number of refinement points to interpolate between integration points. */
  
  int direction; /* Forwards = 1 or backwards = -1 in time; 0 means unset */

  double *gParams; /* Array of parameters for the vector field being 
		      integrated. Used in calls to event functions. Copied in from python. */

  double **gRefinePoints; /* Buffer of points refining integration; size is refine*PhaseDim */
  double *gRefineTimes;   /* Buffer of times refining integration; size is refine */
  int refineBufIdx;       /* Current index into refine buffers */

  double *gSpecTimes; /* Specific times at which to evaluate the trajectory */
  int specTimesLen; /* Length of gSpecTimes */
  int specTimesIdx; /* Current index into array of specific times */

  int calcSpecTimes; /* Flag for whether specific times should be calculated */
  int calcAux;       /* Flag for whether aux variables should be calculated on this run */

  /****************************/
  /* Event saving             */
  /****************************/
 
  int maxEvtPts;      /* Max number of event points to save */
  int haveActive;     /* Number of events active */
  int activeTerm;     /* Number of active terminal events */
  int activeNonTerm;  /* Number of active non-terminal events */

  int *gCheckableEvents;     /* Indices (into gEventFcnArray) of active events, size nActive */
  int *gCheckableEventCounts;/* Number of times each active event has been found */
  int *gTermIndices;    /* Indices (into gCheckableEvents) of terminal events */
  int *gNonTermIndices; /* Indices (into gCheckableEvents) of terminal events */

  int *gEventActive; /* 0 inactive, 1 active, size nEvents; copied from python */
  int *gEventDir;    /* -1 decreasing, 1 increasing, 0 either, size nEvents; copied from python */
  int *gEventTerm;   /* 0 nonterminal, 1 terminal, size nEvents; copied from python */   

  int *gMaxBisect;        /* Max number of bisection steps, size nEvents; copied from python */
  double *gEventTol;      /* Event detection tolerance, size nEvents; copied from python */
  double *gEventDelay;    /* Delay from start of integration before looking for event, size nEvents
			     copied from python */
  double *gEventInterval; /* Delay after finding an event before looking again, size nEvents;
			     copied from python */

  double *gTempPoint;
  int *gNTEvtFound;
  int *gNTEvtFoundOrder;
  double *gNTEvtFoundTimes;

  int eventFound;
  double eventT; /* Time of the found event; formerly xfound */
  double *gEventY;/* Value of the found event; formerly yfound */
  double ***gEventPoints;
  double **gEventTimes;
  double **gEventPointBuf;
  double *gEventTimeBuf;
  int eventBufIdx;

  double eventNearCoef;

  /****************************/
  /* Aux variables            */
  /****************************/
  double **gAuxPoints;
  int auxIdx;

  /****************************/
  /* External inputs          */
  /****************************/
  int nExtInputs; /* Number of external inputs */
  int *gExtInputLens; /* Length of the external inputs */

  double **gExtInputVals; /* Arrays holding the external input values */
  double **gExtInputTimes; /* Arrays holding the times associated with external input values */

  double *gCurrentExtInputVals; /* Array that holds the current external input value at time t,
				   calculated from input values and times */
  
  int *gCurrentExtInputIndex; /* Indices into the ExtInputTimes/Vals arrays for 
				 faster calculations. */

  /****************************/
  /* Radau specific           */
  /****************************/
#ifdef __RADAU__
  int workArrayLen; /* size should be 4*n*n + 12*n + 20, n = phaseDim */
  int intWorkArrayLen;/* size should be 3n + 20 */
  double *gWorkArray; /* work array */
  int *gIntWorkArray; /* integer work array */

  int ijac; /* Analytic jacobian flag -- same as hasJac */
  int jacLowerBandwidth; /* Jacobian lower bandwidth; corresponds to mljac */
  int jacUpperBandwidth; /* corresponds to mujac */
  
  int imas; /* Identity mass matrix flag */
  int masLowerBandwidth; /* corresponds to mlmas */
  int masUpperBandwidth; /* corresponds to mumas */

  int *contdim; /* = 4*n */
  double *contdata; /* continous solution data */
#endif

  /****************************/
  /* Dopri specific           */
  /****************************/
#ifdef __DOPRI__
  int checkBounds;
  int boundsCheckMaxSteps;
  double *gMagBound;
#endif

} IData;

void AuxVarCalc( IData *GS );

void OutputPoint( IData *GS, double t, double *y );

void BufferRefinePoint( IData *GS, double t, double *y, int idx );

void BufferEventPoint( IData *GS, double t, double *y, int idx );

void SavePoints( IData *GS, int TotEvents, int foundTerm );

void FillCurrentExtInputValues( IData *GS, double t );

double GetCurrentExtInputValue( IData *GS, double t, int idx );

int FindCurrentExtInputIndex( IData *GS, double t, int idx );

void adjust_h( IData *GS, double t, double *h );

PyObject* PackOut( IData *GS, double *ICs,
		   int state, double *stats, double hlast, int idid );

#endif

