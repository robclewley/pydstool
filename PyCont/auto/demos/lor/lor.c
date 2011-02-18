/*
 * Author:   Drew LaMar
 * Date:     3 June 2006
 * Revision: 0.3.0
*/

#include "auto_f2c.h"
#include "auto_c.h"

int main(int argc, char *argv[])
{
    AutoData *Data;
    doublereal *cycle;
    doublereal *ups, *udotps, *rldot;
    doublereal period;
    integer i, j;
    doublereal u[3] = {0., 0., 0.};
    integer ipar[4] = {0, 1, 2, 10};
    doublereal par[4] = {280., 2.6666666666666665, 10., 0.4332};
    
    Data = (AutoData *)MALLOC(sizeof(AutoData));
    
    BlankData(Data);
    DefaultData(Data);
    Data->iap.irs = 1;
    Data->iap.ips = 2;
    Data->iap.ilp = 0;
    Data->iap.ndim = 3;
    Data->iap.nicp = 2;
    Data->iap.ntst = 20;
    Data->iap.ncol = 4;
    Data->iap.isp = 2;
    Data->rap.ds = -0.5;
    Data->rap.dsmin = 0.01;
    Data->rap.dsmax = 25.0;
    Data->rap.epsl = 1e-7;
    Data->rap.epsu = 1e-7;
    Data->rap.epss = 0.0001;
    Data->iap.nmx = 50;
    Data->rap.rl0 = 200.;
    Data->rap.rl1 = 400.;
    
    // Load data from lor.dat
    FILE *fp = fopen("lor.dat","r");
    if (fp == NULL) {
        fprintf(stdout,"Error:  Could not open lor.dat\n");
        exit(1);
    }
    cycle = (doublereal *)MALLOC(4*117*sizeof(doublereal));
    for (j=0; j<117; j++) {
        fscanf(fp,"%lf",&cycle[4*j]);
        for (i=0; i<3; i++) {
            fscanf(fp,"%lf",&cycle[1+i+j*4]);
        }
    }
    fclose(fp);
    
    // Tweak times
    period = cycle[4*116] - cycle[0];
    for (i=116; i>=0; i--)
        cycle[4*i] = (cycle[4*i] - cycle[0])/period;
    
    ups = (doublereal *)MALLOC((Data->iap.ncol*Data->iap.ntst+1)*(Data->iap.ndim+1)*sizeof(doublereal));
    udotps = (doublereal *)MALLOC((Data->iap.ncol*Data->iap.ntst+1)*Data->iap.ndim*sizeof(doublereal));
    rldot = (doublereal *)MALLOC(Data->iap.nicp*sizeof(doublereal));
    prepare_cycle(Data,cycle, 117, ups, udotps, rldot);
        
    Data->icp = (integer *)REALLOC(Data->icp,Data->iap.nicp*sizeof(integer));
    Data->icp[1] = 10;
    
    // Create special point
    CreateSpecialPoint(Data,9,1,u,4,ipar,par,Data->icp,ups,udotps,rldot);
    
    AUTO(Data);
    
    CleanupAll(Data);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/*   lor :     The Lorenz Equations */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int func (integer ndim, const doublereal *u, const integer *icp,
          const doublereal *par, integer ijac,
          doublereal *f, doublereal *dfdu, doublereal *dfdp)
{
  f[0] = par[2] * (u[1] - u[0]);
  f[1] = par[0] * u[0] - u[1] - u[0] * u[2];
  f[2] = u[0] * u[1] - par[1] * u[2];

  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int stpnt (integer ndim, doublereal t,
           doublereal *u, doublereal *par)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int bcnd (integer ndim, const doublereal *par, const integer *icp,
          integer nbc, const doublereal *u0, const doublereal *u1, integer ijac,
          doublereal *fb, doublereal *dbc)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int icnd (integer ndim, const doublereal *par, const integer *icp,
          integer nint, const doublereal *u, const doublereal *uold,
          const doublereal *udot, const doublereal *upold, integer ijac,
          doublereal *fi, doublereal *dint)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int fopt (integer ndim, const doublereal *u, const integer *icp,
          const doublereal *par, integer ijac,
          doublereal *fs, doublereal *dfdu, doublereal *dfdp)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int pvls (integer ndim, const void *u,
          doublereal *par)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

