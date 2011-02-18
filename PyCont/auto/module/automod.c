#include "automod.h"

AutoData *gIData = NULL;

/**************************************
 **************************************
 *       COMPUTE
 **************************************
 **************************************/

PyObject* Compute( void ) {
    
    if (!AUTO(gIData))
        return PackOut(0);
    else {
        return PackOut(1);
    }
}

/**************************************
 **************************************
 *       PACKOUT
 **************************************
 **************************************/

PyObject* PackOut(int state) {
    int i, j, k, l;
    int ind;
    int vardim;
    int extdim = 0;
    
    Complex64 evtemp;
    
    PyObject *OutObj = NULL; /* Overall PyTuple output object */
    PyArrayObject **VarOut = NULL;
    PyArrayObject *ParOut = NULL;
    PyArrayObject *EvOut = NULL;
    PyArrayObject *JacOut0 = NULL;
    PyArrayObject *JacOut1 = NULL;
    PyArrayObject *NitOut = NULL;
    
    PyObject **SPOutTuple = NULL;
    PyObject **SPOutFlowTuple = NULL;
    
    PyArrayObject **SPOut_ups = NULL;
    PyArrayObject **SPOut_udotps = NULL;
    PyArrayObject **SPOut_rldot = NULL;
    PyArrayObject ***SPOut_Flow1 = NULL;
    PyArrayObject ***SPOut_Flow2 = NULL;
    
    assert(gIData);
    
    if (state == FAILURE) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    /* for NumArray compatibility */
    import_libnumarray();
    
    // Initialize and store VarOut and ParOut
    if (gIData->u != NULL)
        vardim = 1; // AE
    else
        vardim = 2+(int)(log2(gIData->nsm)); // BVP (will increase later -- user controlled, e.g. average, L2, ...)
    VarOut = (PyArrayObject **)PyMem_Malloc(vardim*sizeof(PyArrayObject *));
    for (i=0; i<vardim; i++) {
        VarOut[i] = NA_NewArray(NULL, tFloat64, 2, gIData->num_u, gIData->iap.ndim);
        assert(VarOut[i]);
    }
    ParOut = NA_NewArray(NULL, tFloat64, 2, gIData->num_u, gIData->iap.nicp);
    assert(ParOut);
    
    for(i = 0; i < gIData->num_u; i++) {
        for(j = 0; j < gIData->iap.ndim; j++) {
            if (gIData->u != NULL)
                NA_set2_Float64(VarOut[0], i, j, gIData->u[i][j]);
            else
                for(k = 0; k < vardim; k++)
                    NA_set2_Float64(VarOut[k], i, j, gIData->usm[k][i][j]);
        }
        
        for(j=0; j < gIData->iap.nicp; j++)
            NA_set2_Float64(ParOut, i, j, gIData->par[i][j]);
    }
    
    // Initialize and store eigenvalues, if computed
    if (gIData->ev != NULL) {
        EvOut = NA_NewArray(NULL, tComplex64, 2, gIData->num_u, gIData->iap.ndim);
        for (i = 0; i < gIData->num_u; i++) {
            for (j = 0; j < gIData->iap.ndim; j++) {
                evtemp.r = gIData->ev[i][j].r;
                evtemp.i = gIData->ev[i][j].i;
                NA_set2_Complex64(EvOut, i, j, evtemp);
            }
        }
        extdim++;
    }
    
    // Initialize and store jacobian
    if (gIData->sjac) {
        JacOut0 = NA_NewArray(NULL, tFloat64, 2, gIData->num_u, gIData->iap.ndim*gIData->iap.ndim);
        JacOut1 = NA_NewArray(NULL, tFloat64, 2, gIData->num_u, gIData->iap.ndim*gIData->iap.ndim);
        for (i = 0; i < gIData->num_u; i++) {
            for (j = 0; j < gIData->iap.ndim; j++) {
                for (k = 0; k < gIData->iap.ndim; k++) {
                    NA_set2_Float64(JacOut0, i, k+j*gIData->iap.ndim, gIData->c0[i][j][k]);
                    NA_set2_Float64(JacOut1, i, k+j*gIData->iap.ndim, gIData->c1[i][j][k]);
                }
            }
        }
        extdim += 2;
    }
    
    // Initialize and store number of iterations
    if (gIData->snit) {
        NitOut = NA_NewArray(NULL, tInt64, 2, gIData->num_u, 1);
        for (i = 0; i < gIData->num_u; i++) {
            NA_set2_Int64(NitOut, i, 0, gIData->nit[i]);
        }
        extdim += 1;
    }
    
    // Initialize and store SPOut
    SPOutTuple = PyMem_Malloc(gIData->num_sp*sizeof(PyObject *));
    assert(SPOutTuple);
    for (i=0; i<gIData->num_sp; i++) {
        SPOutTuple[i] = PyTuple_New(6);
        assert(SPOutTuple);
    }
    
    SPOut_ups = PyMem_Malloc(gIData->num_sp*sizeof(PyArrayObject *));
    assert(SPOut_ups);
    
    SPOut_udotps = PyMem_Malloc(gIData->num_sp*sizeof(PyArrayObject *));
    assert(SPOut_udotps);
    
    SPOut_rldot = PyMem_Malloc(gIData->num_sp*sizeof(PyArrayObject *));
    assert(SPOut_rldot);
    
    if (gIData->sflow) {
        SPOutFlowTuple = PyMem_Malloc(gIData->num_sp*sizeof(PyObject *));
        assert(SPOutFlowTuple);
        for (i=0; i<gIData->num_sp; i++) {
            SPOutFlowTuple[i] = PyTuple_New(2*gIData->iap.ntst);
            assert(SPOutFlowTuple[i]);
        }
        
        SPOut_Flow1 = PyMem_Malloc(gIData->num_sp*sizeof(PyArrayObject **));
        SPOut_Flow2 = PyMem_Malloc(gIData->num_sp*sizeof(PyArrayObject **));
        assert(SPOut_Flow1);
        assert(SPOut_Flow2);
        
        for (i=0; i<gIData->num_sp; i++) {
            SPOut_Flow1[i] = PyMem_Malloc(gIData->iap.ntst*sizeof(PyArrayObject *));
            SPOut_Flow2[i] = PyMem_Malloc(gIData->iap.ntst*sizeof(PyArrayObject *));
            assert(SPOut_Flow1[i]);
            assert(SPOut_Flow2[i]);
        }
    }
    
    for(i = 0; i < gIData->num_sp; i++) {
        ind = 0;    // Used so I can insert more info w/o worrying about indices, just ordering
        // Index
        PyTuple_SetItem(SPOutTuple[i], ind++, PyInt_FromLong(gIData->sp[i].mtot-1));
        
        // Type of point
        PyTuple_SetItem(SPOutTuple[i], ind++, PyInt_FromLong(gIData->sp[i].itp));
        
        // ups
        if (gIData->sp[i].ups != NULL) {
            SPOut_ups[i] = NA_NewArray(NULL, tFloat64, 2, gIData->sp[i].ntpl, gIData->sp[i].nar);
            assert(SPOut_ups[i]);
            for (k = 0; k < gIData->sp[i].ntpl; k++)
                for (l = 0; l < gIData->sp[i].nar; l++)
                    NA_set2_Float64(SPOut_ups[i], k, l, gIData->sp[i].ups[k][l]);
            PyTuple_SetItem(SPOutTuple[i], ind++, SPOut_ups[i]);
        } else {
            PyTuple_SetItem(SPOutTuple[i], ind++, Py_None);
            Py_INCREF(Py_None);
        }
        
        // udotps
        if (gIData->sp[i].udotps != NULL) {
            SPOut_udotps[i] = NA_NewArray(NULL, tFloat64, 2, gIData->sp[i].ntpl, gIData->sp[i].nar-1);
            assert(SPOut_udotps[i]);
            for (k = 0; k < gIData->sp[i].ntpl; k++)
                for (l = 0; l < gIData->sp[i].nar-1; l++)
                    NA_set2_Float64(SPOut_udotps[i], k, l, gIData->sp[i].udotps[k][l]);
            PyTuple_SetItem(SPOutTuple[i], ind++, SPOut_udotps[i]);
        } else {
            PyTuple_SetItem(SPOutTuple[i], ind++, Py_None);
            Py_INCREF(Py_None);
        }
        
        // rldot
        if (gIData->sp[i].rldot != NULL) {
            SPOut_rldot[i] = NA_NewArray(NULL, tFloat64, 1, gIData->sp[i].nfpr);
            assert(SPOut_rldot[i]);
            for (j = 0; j < gIData->sp[i].nfpr; j++)
                NA_set1_Float64(SPOut_rldot[i], j, gIData->sp[i].rldot[j]);
            PyTuple_SetItem(SPOutTuple[i], ind++, SPOut_rldot[i]);
        } else {
            PyTuple_SetItem(SPOutTuple[i], ind++, Py_None);
            Py_INCREF(Py_None);
        }
        
        // sflow (a1 and a2)
        if (gIData->sflow) {
            for (j = 0; j < gIData->iap.ntst; j++) {
                SPOut_Flow1[i][j] = NA_NewArray(NULL, tFloat64, 2, gIData->sp[i].nar-1, gIData->sp[i].nar-1);
                SPOut_Flow2[i][j] = NA_NewArray(NULL, tFloat64, 2, gIData->sp[i].nar-1, gIData->sp[i].nar-1);
                assert(SPOut_Flow1[i][j]);
                assert(SPOut_Flow2[i][j]);
                for (k = 0; k < gIData->sp[i].nar-1; k++) {
                    for (l = 0; l < gIData->sp[i].nar-1; l++) {
                        NA_set2_Float64(SPOut_Flow1[i][j], k, l, gIData->sp[i].a1[j][k][l]);
                        NA_set2_Float64(SPOut_Flow2[i][j], k, l, gIData->sp[i].a2[j][k][l]);
                    }
                }
                PyTuple_SetItem(SPOutFlowTuple[i], 2*j, SPOut_Flow1[i][j]);
                PyTuple_SetItem(SPOutFlowTuple[i], 2*j+1, SPOut_Flow2[i][j]);
            }
            PyTuple_SetItem(SPOutTuple[i], ind++, SPOutFlowTuple[i]);
        } else {
            PyTuple_SetItem(SPOutTuple[i], ind++, Py_None);
            Py_INCREF(Py_None);
        }
    }
    
    // Wrap it up
    OutObj = PyTuple_New(1+vardim+extdim+gIData->num_sp);
    assert(OutObj);
    
    for (i=0; i<vardim; i++)
        PyTuple_SetItem(OutObj, i, (PyObject *)VarOut[i]);
    PyTuple_SetItem(OutObj, vardim, (PyObject *)ParOut);
    
    if (EvOut != NULL)
        PyTuple_SetItem(OutObj, 1+vardim, (PyObject *)EvOut);
    
    if (gIData->sjac) {
        //PyTuple_SetItem(OutObj, 1+vardim+extdim-2, (PyObject *)JacOut0);
        //PyTuple_SetItem(OutObj, 1+vardim+extdim-1, (PyObject *)JacOut1);
        PyTuple_SetItem(OutObj, 1+vardim+extdim-3, (PyObject *)JacOut0);
        PyTuple_SetItem(OutObj, 1+vardim+extdim-2, (PyObject *)JacOut1);
    }
    
    if (gIData->snit) {
        PyTuple_SetItem(OutObj, 1+vardim+extdim-1, (PyObject *)NitOut);
    }
    
    for (i = 0; i < gIData->num_sp; i++)
        PyTuple_SetItem(OutObj, 1+vardim+extdim+i, SPOutTuple[i]);
    
    // Free
    PyMem_Free(VarOut);
    PyMem_Free(SPOut_ups);
    PyMem_Free(SPOut_udotps);
    PyMem_Free(SPOut_rldot);
    if (gIData->sflow) {
        PyMem_Free(SPOutFlowTuple);
        for (i=0; i<gIData->num_sp; i++) {
            PyMem_Free(SPOut_Flow1[i]);
            PyMem_Free(SPOut_Flow2[i]);
        }
        PyMem_Free(SPOut_Flow1);
        PyMem_Free(SPOut_Flow2);
    }
    PyMem_Free(SPOutTuple);
    
    // Ship it out
    return OutObj;
}

/**************************************
 **************************************
 *    REQUIRED AUTO FUNCTIONS
 **************************************
 **************************************/

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int func(integer ndim, const doublereal *u, const integer *icp,
         const doublereal *par, integer ijac,
         doublereal *f, doublereal *dfdu, doublereal *dfdp) {

  /* Jacobian and Parameter Jacobian loops clearly slow things down here. */
  
  integer i;
  doublereal **jac;
  doublereal **jacp;
  
  vfieldfunc(ndim, 0, 0, u, par, f, 0, NULL, 0, NULL);
  
  // Jacobian
  jac = (doublereal **)MALLOC(ndim*sizeof(doublereal *));
  jac[0] = dfdu;
  for (i=1; i<ndim; i++) jac[i] = jac[i-1] + ndim;
  jacobian(ndim, 0, 0, u, par, jac, 0, NULL, 0, NULL);
  FREE(jac);
  
  // Parameter Jacobian
  jacp = (doublereal **)MALLOC(gIData->npar*sizeof(doublereal *));
  jacp[0] = dfdp;
  for (i=1; i<gIData->npar; i++) {
      if (i != 10)
          jacp[i] = jacp[i-1] + ndim;
      else
          jacp[i] = jacp[i-1] + 41*ndim;    // Index = 50
  }
  jacobianParam(ndim, 0, 0, u, par, jacp, 0, NULL, 0, NULL);
  FREE(jacp);
  
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int stpnt(integer ndim, doublereal t, doublereal *u, doublereal *par) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int bcnd(integer ndim, const doublereal *par, const integer *icp,
         integer nbc, const doublereal *u0, const doublereal *u1, integer ijac,
         doublereal *fb, doublereal *dbc) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int icnd(integer ndim, const doublereal *par, const integer *icp,
         integer nint, const doublereal *u, const doublereal *uold,
         const doublereal *udot, const doublereal *upold, integer ijac,
         doublereal *fi, doublereal *dint) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int fopt(integer ndim, const doublereal *u, const integer *icp,
         const doublereal *par, integer ijac,
         doublereal *fs, doublereal *dfdu, doublereal *dfdp) {
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int pvls(integer ndim, const void *u, doublereal *par) {
    u = (doublereal *)u;
    return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
