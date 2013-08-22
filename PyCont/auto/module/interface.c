#include "interface.h"

extern AutoData *gIData;

/**************************************
 **************************************
 *       INITIALIZATION
 **************************************
 **************************************/

PyObject* Initialize( void ) {

    PyObject *OutObj = NULL;

    if (gIData == NULL) {
      gIData = (AutoData *)PyMem_Malloc(sizeof(AutoData));
      assert(gIData);
    }

    BlankData(gIData);
    DefaultData(gIData);

    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);

    return OutObj;
}

/* Needs to set main data for auto computation and before call to SetInitPoint */
PyObject* SetData(int ips, int ilp, int isw, int isp, int sjac, int sflow, int nsm, int nmx, int ndim,
                  int ntst, int ncol, int iad, double epsl, double epsu, double epss, int itmx,
                  int itnw, double ds, double dsmin, double dsmax, int npr, int iid,
                  int nicp, int *icp, int nuzr, int *iuz, double *vuz) {
    int i;
    PyObject *OutObj = NULL;

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

    OutObj = Py_BuildValue("(i)", 1);
    assert(OutObj);

    return OutObj;
}

// Assumes there has been a call to SetData before this.
PyObject* SetInitPoint(double *u, int npar, int *ipar, double *par, int *icp, int nups,
                       double *ups, double *udotps, double *rldot, int adaptcycle) {
    /* Still think I should get rid of typemap(freearg) and send address to pointer in
    prepare_cycle so I can realloc if necessary and not have to use this my_... crap */
    PyObject *OutObj = NULL;
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

    OutObj = Py_BuildValue("i", SUCCESS);
    assert(OutObj);

    return OutObj;
}

PyObject* Reset( void ) {
    int result = 1;
    PyObject *OutObj = NULL;

    result = result && (CleanupSolution(gIData));
    result = result && (CleanupSpecialPoints(gIData));

    DefaultData(gIData);

    OutObj = Py_BuildValue("(i)", result);
    assert(OutObj);

    return OutObj;
}

/**************************************
 **************************************
 *       CLEANUP
 **************************************
 **************************************/

PyObject* ClearParams( void ) {
    PyObject *OutObj = NULL;

    if (!CleanupParams(gIData)) {
        OutObj = Py_BuildValue("(i)", 0);
        assert(OutObj);
    } else {
        OutObj = Py_BuildValue("(i)", 1);
        assert(OutObj);
    }

    return OutObj;
}

PyObject* ClearSolution( void ) {
    PyObject *OutObj = NULL;

    if (!CleanupSolution(gIData)) {
        OutObj = Py_BuildValue("(i)", 0);
        assert(OutObj);
    } else {
        OutObj = Py_BuildValue("(i)", 1);
        assert(OutObj);
    }

    return OutObj;
}

PyObject* ClearSpecialPoints( void ) {
    PyObject *OutObj = NULL;

    if (!CleanupSpecialPoints(gIData)) {
        OutObj = Py_BuildValue("(i)", 0);
        assert(OutObj);
    } else {
        OutObj = Py_BuildValue("(i)", 1);
        assert(OutObj);
    }

    return OutObj;
}

PyObject* ClearAll( void ) {
    PyObject *OutObj = NULL;

    if (!CleanupAll(gIData)) {
        OutObj = Py_BuildValue("(i)", 0);
        assert(OutObj);
    } else {
        OutObj = Py_BuildValue("(i)", 1);
        assert(OutObj);
        gIData = NULL;
    }

    return OutObj;
}
