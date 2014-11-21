#ifndef __MEMORY__
#define __MEMORY__

#include "integration.h"
#include "vfield.h"
#include "events.h"
#include <stdio.h>
#include "Python.h"
#include <numpy/arrayobject.h>

#define FAILURE 0
#define SUCCESS 1
#define EVENT_NEARNESS_FACTOR 1000.0

int InitializeBasic( IData *GS, int PhaseDim, int ParamDim, int nAux, int nEvents, int nExtInputs, 	     
		     int HasJac, int HasJacP, int HasMass, int extraSize);

int CleanupBasic( IData *GS );

int SetRunParams( IData *GS, double *Pars, double *ICs, double **Bds, double *y0, double gTime, 
		  double *GlobalTime0, double tstart, double tend,
		  int refine, int nSpecTimes, double *specTimes, 
		  double *upperBounds, double *lowerBounds );

int CleanupRunParams( IData *GS );

int InitIntegData( IData *GS,  int Maxpts, double *atol, double *rtol, 
		   ContSolFunType ContSolFun);

int CleanupIData( IData *GS );

int InitializeEvents( IData *GS, int Maxevtpts, int *EventActive, int *EventDir, int *EventTerm,
		      double *EventInterval, double *EventDelay, double *EventTol,
		      int *Maxbisect, double EventNearCoef );

int CleanupEvents( IData *GS );

int InitializeExtInputs( IData *GS, int nExtInputs, int *extInputLens, double *extInputVals,
			 double *extInputTimes );

int CleanupExtInputs( IData *GS );

int CleanupAll( IData *GS, double *ICs, double **Bds);

int SetContParams( IData *GS, double tend, double *pars, 
		   double **Bds, double *upperBounds, double *lowerBounds);


int ResetIndices( IData *GS );

void BlankIData( IData *GS );

void setJacPtrs( IData *GS, double *jacSpace );

void setMassPtrs( IData *GS, double *massSpace );


#ifdef __RADAU__
int InitializeRadauOptions( IData *GS, double uround, double safety, 
			    double jacRecompute, double newtonStop, 
			    double stepChangeLB, double stepChangeUB,
			    double hmax, double stepSizeLB, double stepSizeUB,
			    int hessenberg, int maxSteps, int maxNewton,
			    int newtonStart, int index1dim, int index2dim,
			    int index3dim, int stepSizeStrategy, 
			    int DAEstructureM1, int DAEstructureM2 );
#endif

#ifdef __DOPRI__
int InitializeDopriOptions( IData *GS, int checkBounds, int boundsCheckMaxSteps, 
			    double *magBound);

#endif

#endif
