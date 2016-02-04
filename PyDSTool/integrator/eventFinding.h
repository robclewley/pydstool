#ifndef __EVENTFINDING__
#define __EVENTFINDING__

#include "vfield.h"
#include "events.h"
#include "integration.h"

#define BINSEARCH_LIMIT 1e-9
#define SUBINTERVALS 10

/* Prototype for event functions */

int DetectEvents(IData *GS, double xold, double x, double *y, int *irtrn, int *termFound);

int CheckEventFctn(IData *GS, double xold, int idx, int count, int CheckIdx);

int FindEvent(IData *GS, int evtIdx, double xold);

void BinSearch(IData *GS, int evtIdx, double x1, double x2, double v1, double v2, 
	       double vo);
#endif
