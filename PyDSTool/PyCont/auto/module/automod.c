#include "automod.h"

extern AutoData *gIData;

/**************************************
 **************************************
 *       COMPUTE
 **************************************
 **************************************/

int Compute( void ) {

    if (!AUTO(gIData))
        return 0;
    else {
        return 1;
    }
}

/**************************************
 **************************************
 *       Python -> C
 **************************************
 **************************************/

int Initialize( void ) {
    if (gIData == NULL) {
      gIData = (AutoData *)PyMem_Malloc(sizeof(AutoData));
      assert(gIData);
    }

    BlankData(gIData);
    DefaultData(gIData);
    
    return 1;
}

/* Needs to set main data for auto computation and before call to SetInitPoint */
int SetData(int ips, int ilp, int isw, int isp, int sjac, int sflow, int nsm, int nmx, int ndim, 
                  int ntst, int ncol, int iad, double epsl, double epsu, double epss, int itmx,
                  int itnw, double ds, double dsmin, double dsmax, int npr, int iid,
                  int nicp, int *icp, int nuzr, int *iuz, double *vuz) {
    int i;

    gIData->iap.ips = ips;
    gIData->iap.ilp = ilp;
    gIData->iap.isw = isw;
    gIData->iap.isp = isp;
    gIData->iap.nmx = nmx;
    gIData->iap.ndim = ndim;
    gIData->iap.ntst = ntst;
    gIData->iap.ncol = ncol;
    gIData->iap.iad = iad;
    gIData->iap.itmx = itmx;
    gIData->iap.itnw = itnw;
    gIData->iap.npr = npr;
    gIData->iap.iid = iid;
    gIData->iap.nicp = nicp;
        
    gIData->rap.epsl = epsl;
    gIData->rap.epsu = epsu;
    gIData->rap.epss = epss;
    gIData->rap.ds = ds;
    gIData->rap.dsmin = dsmin;
    gIData->rap.dsmax = dsmax;
    
    gIData->sjac = sjac;
    gIData->sflow = sflow;
    gIData->snit = 1;
    gIData->nsm = nsm;
    
    gIData->icp = (integer *)REALLOC(gIData->icp,gIData->iap.nicp*sizeof(integer));
    for (i=0; i<gIData->iap.nicp; i++)
        gIData->icp[i] = icp[i];
    
    gIData->iap.nuzr = nuzr;
    if (gIData->iap.nuzr > 0) {
        gIData->iuz = (integer *)REALLOC(gIData->iuz,gIData->iap.nuzr*sizeof(integer));
        gIData->vuz = (doublereal *)REALLOC(gIData->vuz,gIData->iap.nuzr*sizeof(doublereal));
    }
    for (i=0; i<gIData->iap.nuzr; i++) {
        gIData->iuz[i] = iuz[i];
        gIData->vuz[i] = vuz[i];
    }

    return 1;
}

// Assumes there has been a call to SetData before this.
int SetInitPoint(double *u, int npar, int *ipar, double *par, int *icp, int nups,
                       double *ups, double *udotps, double *rldot, int adaptcycle) {
    /* Still think I should get rid of typemap(freearg) and send address to pointer in
    prepare_cycle so I can realloc if necessary and not have to use this my_... crap */
    integer itp;
    integer i, j;
    integer SUCCESS = 1;
    double *my_ups = ups;
    double *my_udotps = udotps;
    double *my_rldot = rldot;

    if (ups == NULL)
        itp = 3;
    else {
        itp = 9;
        
        if (adaptcycle || udotps == NULL || rldot == NULL) {
            // NOTE: AUTO allocates (ntst+1)*ncol points, but only displays ntpl=ntst*ncol+1
            my_ups = (double *)MALLOC(gIData->iap.ncol*(gIData->iap.ntst+1)*(gIData->iap.ndim+1)*sizeof(double));
            my_udotps = (double *)MALLOC(gIData->iap.ncol*(gIData->iap.ntst+1)*gIData->iap.ndim*sizeof(double));
            my_rldot = (double *)MALLOC(gIData->iap.nicp*sizeof(double));
            prepare_cycle(gIData, ups, nups, my_ups, my_udotps, my_rldot);
        }
    }
    
    if (!CreateSpecialPoint(gIData, itp, 1, u, npar, ipar, par, icp, my_ups, my_udotps, my_rldot)) {
        fprintf(stderr,"*** Warning [interface.c]: Problem in CreateSpecialPoint.\n");
        fflush(stderr);
        SUCCESS = 0;
    }
    
    if ((SUCCESS) && (itp == 9) && (adaptcycle || udotps == NULL || rldot == NULL)) {
        integer nmx, npr, verbosity;
        double ds, dsmin, dsmax;
        
        // Adjust rldot and sp.nfpr = 1 (AUTO detects this to get udotps and rldot for
        //  starting point)
        gIData->sp[0].nfpr = 1;

        // Remove from here till } once findPeriodicOrbit is created
        FREE(my_ups);
        FREE(my_udotps);
        FREE(my_rldot);
        
        // Make sure initial point is on curve...
        nmx = gIData->iap.nmx;
        npr = gIData->iap.npr;
        ds = gIData->rap.ds;
        dsmin = gIData->rap.dsmin;
        dsmax = gIData->rap.dsmax;
        verbosity = gIData->verbosity;
        
        gIData->iap.nmx = 3;
        gIData->iap.npr = 3;
        gIData->rap.ds = min(1e-4, gIData->rap.ds);
        gIData->rap.dsmin = min(1e-4, gIData->rap.dsmin);
        gIData->rap.dsmax = min(1e-4, gIData->rap.dsmax);
        gIData->verbosity = 0;
        
        AUTO(gIData);
        CleanupSolution(gIData);
        
        gIData->iap.nmx = nmx;
        gIData->iap.npr = npr;
        gIData->rap.ds = ds;
        gIData->rap.dsmin = dsmin;
        gIData->rap.dsmax = dsmax;
        gIData->verbosity = verbosity;
        
        // Check for NaNs
        for (i=0; i<gIData->sp[0].nar; i++) {
            if (isnan(gIData->sp[1].u[i])) {
                fprintf(stderr,"*** Warning [interface.c]: NaNs in auto solution.\n");
                fflush(stderr);
                SUCCESS = 0;
                break;
            }
        }
        
        if (SUCCESS) {
            for (i=0; i<gIData->sp[0].nar; i++)
                gIData->sp[0].u[i] = gIData->sp[1].u[i];
            
            for (i=0; i<gIData->iap.ntst*gIData->iap.ncol+1; i++)
                for (j=0; j<gIData->iap.ndim+1; j++)
                    gIData->sp[0].ups[i][j] = gIData->sp[1].ups[i][j];
            
            for (i=0; i<gIData->iap.ntst*gIData->iap.ncol+1; i++)
                for (j=0; j<gIData->iap.ndim; j++)
                    gIData->sp[0].udotps[i][j] = gIData->sp[1].udotps[i][j];
            
            for (i=0; i<gIData->iap.nicp; i++)
                gIData->sp[0].rldot[i] = gIData->sp[1].rldot[i];
        }
        
        gIData->sp[0].nfpr = gIData->iap.nicp;
        
        FREE(gIData->sp[1].icp);
        FREE(gIData->sp[1].u);
        FREE(gIData->sp[1].rldot);
        FREE_DMATRIX(gIData->sp[1].ups);
        FREE_DMATRIX(gIData->sp[1].udotps);
        if (gIData->sflow) {
            FREE_DMATRIX_3D(gIData->sp[1].a1);
            FREE_DMATRIX_3D(gIData->sp[1].a2);
        }
        gIData->sp[1].icp = NULL;
        gIData->sp[1].u = NULL;
        gIData->sp[1].rldot = NULL;
        gIData->sp[1].ups = NULL;
        gIData->sp[1].udotps = NULL;
        gIData->sp[1].a1 = NULL;
        gIData->sp[1].a2 = NULL;
        
        gIData->num_sp = 1;
        gIData->sp_len = 1;
        
        gIData->sp = (AutoSPData *)REALLOC(gIData->sp, (gIData->num_sp)*sizeof(AutoSPData));
    }
    
    return SUCCESS;
}

/* int Reset( void ) {
    int result = 1;

    result = result && (CleanupSolution(gIData));
    result = result && (CleanupSpecialPoints(gIData));
    
    DefaultData(gIData);
    
    return result;
} */

/******************************************
 ******************************************
 *       C -> Python
 *
 * Most functions use NumPy INPLACE_ARRAY 
 * routines (see %apply commands in 
 * automod.i). Ultimately, gIData should be 
 * implemented as a class structure, with
 * these functions as methods.  Maybe
 * cython would be perfect for this.
 * 
 ******************************************
 ******************************************/

int getSolutionNum(void)
{
	return gIData->num_u;
}

void getSolutionVar(double *A, int nd1, int nd2, int nd3)
{
	int i, j, k;
	
	for (i=0; i<nd1; i++)
		for (j=0; j<nd2; j++)
			for (k=0; k<nd3; k++)
				if (gIData->u != NULL)
					A[i*nd2*nd3+j*nd3+k] = gIData->u[i][j];
				else
					A[i*nd2*nd3+j*nd3+k] = gIData->usm[k][i][j];
}

void getSolutionPar(double *A, int nd1, int nd2)
{
	int i, j;
	
	for (i=0; i<nd1; i++)
		for (j=0; j<nd2; j++)
			A[i*nd2+j] = gIData->par[i][j];
}

void getFloquetMultipliers(double *A, int nd1, int nd2, int nd3)
{
	int i, j, k;
	
	for (i=0; i<nd1; i++)
		for (j=0; j<nd2; j++)
			for (k=0; k<nd3; k++)
				if (k == 0)
					A[i*nd2*nd3+j*nd3+k] = gIData->ev[i][j].r;
				else
					A[i*nd2*nd3+j*nd3+k] = gIData->ev[i][j].i;	
}

void getJacobians(double *A, int nd1, int nd2, int nd3)
{
	int i, j, k, l;
	int ndim = gIData->iap.ndim;
	
	for (i=0; i<nd1; i++)
		for (j=0; j<nd2; j++)
			for (k=0; k<ndim; k++)
				for (l=0; l<ndim; l++)
					if (i == 0)
						A[i*nd2*nd3+j*nd3+k*ndim+l] = gIData->c0[j][k][l];
					else
						A[i*nd2*nd3+j*nd3+k*ndim+l] = gIData->c1[j][k][l];
}

void getNumIters(int *A, int nd1, int nd2)
{
/* Needs to be 2D on Python side.  Should change in future. */
	int i;
	
	for (i=0; i<nd1; i++)
		A[i] = gIData->nit[i];
}

int getSpecPtNum(void)
{
	return gIData->num_sp;
}

void getSpecPtDims(int i, int *A, int nd1)
{
	A[0] = gIData->sp[i].mtot-1;
	A[1] = gIData->sp[i].itp;
	A[2] = gIData->sp[i].ntpl;
	A[3] = gIData->sp[i].nar;
	A[4] = gIData->sp[i].nfpr;
}

void getSpecPtFlags(int i, int *A, int nd1)
{
	A[0] = (gIData->sp[i].ups != NULL);
	A[1] = (gIData->sp[i].udotps != NULL);
	A[2] = (gIData->sp[i].rldot != NULL);
	A[3] = (gIData->sflow);
}

void getSpecPt_ups(int i, double *A, int nd1, int nd2)
{
	int j, k;
	
	for (j=0; j<nd1; j++)
		for (k=0; k<nd2; k++)
			A[j*nd2+k] = gIData->sp[i].ups[j][k];
}

void getSpecPt_udotps(int i, double *A, int nd1, int nd2)
{
	int j, k;
	
	for (j=0; j<nd1; j++)
		for (k=0; k<nd2; k++)
			A[j*nd2+k] = gIData->sp[i].udotps[j][k];
}

void getSpecPt_rldot(int i, double *A, int nd1)
{
	int j;
	
	for (j=0; j<nd1; j++)
		A[j] = gIData->sp[i].rldot[j];
}

void getSpecPt_flow1(int i, double *A, int nd1, int nd2, int nd3)
{
	int j, k, l;
	
	for (j=0; j<nd1; j++)
		for (k=0; k<nd2; k++)
			for (l=0; l<nd3; l++)
				A[j*nd2*nd3+k*nd3+l] = gIData->sp[i].a1[j][k][l];	
}

void getSpecPt_flow2(int i, double *A, int nd1, int nd2, int nd3)
{
	int j, k, l;
	
	for (j=0; j<nd1; j++)
		for (k=0; k<nd2; k++)
			for (l=0; l<nd3; l++)
				A[j*nd2*nd3+k*nd3+l] = gIData->sp[i].a2[j][k][l];	
}

/**************************************
 **************************************
 *       CLEANUP
 **************************************
 **************************************/

/* int ClearParams( void ) {
	int result;
    
    result = CleanupParams(gIData);

    return result;
}

int ClearSolution( void ) {
    int result;
    
    result = CleanupSolution(gIData);
    
    return result;
}

int ClearSpecialPoints( void ) {
    int result;

    result = CleanupSpecialPoints(gIData);
    
    return result;
} */

int ClearAll( void ) {
    int result;
    
    result = CleanupAll(gIData);
	if (result)
        gIData = NULL;
    
    return result;
}
