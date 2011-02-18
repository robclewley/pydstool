/*
The DsTool program is the property of:
 
                             Cornell University 
                        Center of Applied Mathematics 
                              Ithaca, NY 14853
                      dstool_bugs@macomb.tn.cornell.edu
 
and may be used, modified and distributed freely, subject to the following
restrictions:
 
       Any product which incorporates source code from the DsTool
       program or utilities, in whole or in part, is distributed
       with a copy of that source code, including this notice. You
       must give the recipients all the rights that you have with
       respect to the use of this software. Modifications of the
       software must carry prominent notices stating who changed
       the files and the date of any change.
 
DsTool is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of FITNESS FOR A PARTICULAR PURPOSE.
The software is provided as is without any obligation on the part of Cornell 
faculty, staff or students to assist in its use, correction, modification or
enhancement.

2001/Nov/12, Bart Oldeman, these matrix allocation routines adapted and
used in AUTO2000.

There are _debug versions and normal versions here. Use the debug
versions explicitly or by a #define in auto_c.h. The advantage
of the normal versions is that they allocate big blocks so they
are faster. The debug versions come in handy with bounds checkers
like "electric fence" (or often, even without them) so AUTO can bomb
out with a segmentation fault at the offending place and gdb is happy
to point out where that was.

NOTE: the _debug versions don't work with MPI.

ALSO they don't work with some of the eispack routines, and mess up
Floquet multipliers!
*/

/*
 * dmatrix.c 
 */

#define DMATRIX_C

#include <stdlib.h>
#include <stdio.h>
#include <auto_f2c.h>
#include <auto_c.h>

/*
 * dmatrix()
 *
 * memory allocation for a matrix of doubles
 * returns m so that m[0][0],...,m[n_rows-1][n_cols-1] are the valid locations
 *
 * last modified:  8/21/91  paw
 */

doublereal **
dmatrix(integer n_rows, integer n_cols)
{
  integer i, total_pts;
  doublereal **m;

  if (n_rows<=0 || n_cols<=0) return(NULL);
  total_pts = n_rows*n_cols;
  
  if ( (m = (doublereal **) malloc( (unsigned) (n_rows * sizeof(doublereal *)))) == NULL)
    {
      printf("dmatrix: memory allocation failure!\n");
    }
  else
    {
      if ( (m[0] = (doublereal *) malloc( (unsigned) (total_pts * sizeof(doublereal)))) == NULL)
	{
	  free(m);
	  m = NULL;
	  printf("dmatrix: memory allocation failure!\n");
	}
      else
	  for (i=1; i<n_rows; i++) m[i] = m[i-1] + n_cols;
    }
  return(m);
}

doublereal **
dmatrix_debug(integer n_rows, integer n_cols)
{
  integer i;
  doublereal **m;

  if (n_rows<=0 || n_cols<=0) return(NULL);
  
  if ( (m = malloc( (unsigned) ((n_rows+1) * sizeof(doublereal *)))) == NULL)
    {
      printf("dmatrix: memory allocation failure!\n");
    }
  else
    {
      for (i=0; i<n_rows; i++)
        m[i] = malloc(n_cols * sizeof(doublereal));
      m[n_rows] = NULL;
    }
  return(m);
}

doublereal ***
dmatrix_3d(integer n_levels, integer n_rows, integer n_cols)
{
  integer i, total_ptrs;
  doublereal ***m;

  if (n_levels<=0 || n_rows<=0 || n_cols<=0) return(NULL);
  total_ptrs = n_levels*n_rows;
  
  if ( (m = (doublereal ***) malloc( (unsigned) (n_levels * sizeof(doublereal **)))) == NULL)
    {
      printf("dmatrix_3d: memory allocation failure!\n");
    }
  else
    {
      if ( (m[0] = dmatrix( (unsigned) total_ptrs, (unsigned) n_cols)) == NULL)
	{
	  free(m);
	  m = NULL;
	  printf("dmatrix_3d: memory allocation failure!\n");
	}
      else
	  for (i=1; i<n_levels; i++) m[i] = m[i-1] + n_rows;
    }
  return(m);
}

doublereal ***
dmatrix_3d_debug(integer n_levels, integer n_rows, integer n_cols)
{
  integer i;
  doublereal ***m;

  if (n_levels<=0 || n_rows<=0 || n_cols<=0) return(NULL);
  
  if ( (m = malloc( (unsigned) ((n_levels+1) * sizeof(doublereal **)))) == NULL)
    {
      printf("dmatrix_3d: memory allocation failure!\n");
    }
  else
    {
      for (i=0; i<n_levels; i++)
        m[i] = dmatrix( (unsigned) n_rows, (unsigned) n_cols);
      m[n_levels] = NULL;
    }
  return(m);
}

/*
 * free_dmatrix()
 *
 * frees memory allocated by dmatrix()
 *
 * last modified: 8/21/91  paw
 */
void
free_dmatrix(doublereal **m)
{
  if (m==NULL) return;
  free( (char *) (m[0]) );
  free( (char *) (m) );
}

void
free_dmatrix_debug(doublereal **m)
{
  integer i;
    
  if (m==NULL) return;

  for (i = 0; m[i] != NULL; i++)
      free(m[i]);
  free( (char *) (m) );
}

void
free_dmatrix_3d(doublereal ***m)
{
  if (m==NULL) return;
  free_dmatrix(m[0]);
  free( (char *) (m) );
}

void
free_dmatrix_3d_debug(doublereal ***m)
{
  integer i;
    
  if (m==NULL) return;

  for (i = 0; m[i] != NULL; i++)
      free_dmatrix(m[i]);
  free( (char *) (m) );
}

/* Modified dmatrix, free_dmatrix: 4/20/06 */
doublereal **DMATRIX(integer n_rows, integer n_cols)
{
  integer i, total_pts;
  doublereal **m;

  if (n_rows<=0 || n_cols<=0) return(NULL);
  total_pts = n_rows*n_cols;
  
  if ( (m = (doublereal **) MALLOC( (unsigned) (n_rows * sizeof(doublereal *)))) == NULL)
    {
      printf("DMATRIX: memory allocation failure!\n");
    }
  else
    {
      if ( (m[0] = (doublereal *) MALLOC( (unsigned) (total_pts * sizeof(doublereal)))) == NULL)
	{
	  FREE(m);
	  m = NULL;
	  printf("DMATRIX: memory allocation failure!\n");
	}
      else
	  for (i=1; i<n_rows; i++) m[i] = m[i-1] + n_cols;
    }
  return(m);
}

doublecomplex **DCMATRIX(integer n_rows, integer n_cols)
{
  integer i, total_pts;
  doublecomplex **m;

  if (n_rows<=0 || n_cols<=0) return(NULL);
  total_pts = n_rows*n_cols;
  
  if ( (m = (doublecomplex **) MALLOC( (unsigned) (n_rows * sizeof(doublecomplex *)))) == NULL)
    {
      printf("DCMATRIX: memory allocation failure!\n");
    }
  else
    {
      if ( (m[0] = (doublecomplex *) MALLOC( (unsigned) (total_pts * sizeof(doublecomplex)))) == NULL)
	{
	  FREE(m);
	  m = NULL;
	  printf("DCMATRIX: memory allocation failure!\n");
	}
      else
	  for (i=1; i<n_rows; i++) m[i] = m[i-1] + n_cols;
    }
  return(m);
}

doublereal ***
DMATRIX_3D(integer n_levels, integer n_rows, integer n_cols)
{
  integer i, total_ptrs;
  doublereal ***m;

  if (n_levels<=0 || n_rows<=0 || n_cols<=0) return(NULL);
  total_ptrs = n_levels*n_rows;
  
  if ( (m = (doublereal ***) MALLOC( (unsigned) (n_levels * sizeof(doublereal **)))) == NULL)
    {
      printf("dmatrix_3d: memory allocation failure!\n");
    }
  else
    {
      if ( (m[0] = DMATRIX( (unsigned) total_ptrs, (unsigned) n_cols)) == NULL)
	{
	  FREE(m);
	  m = NULL;
	  printf("dmatrix_3d: memory allocation failure!\n");
	}
      else
	  for (i=1; i<n_levels; i++) m[i] = m[i-1] + n_rows;
    }
  return(m);
}

/*
 * FREE_DMATRIX()
 *
 * FREEs memory allocated by DMATRIX()
 *
 * last modified: 4/15/06  mdl
 */
void FREE_DMATRIX(doublereal **m)
{
  if (m==NULL) return;
  FREE( (char *) (m[0]) );
  FREE( (char *) (m) );
}

void FREE_DCMATRIX(doublecomplex **m)
{
  if (m==NULL) return;
  FREE( (char *) (m[0]) );
  FREE( (char *) (m) );
}

void
FREE_DMATRIX_3D(doublereal ***m)
{
  if (m==NULL) return;
  FREE_DMATRIX(m[0]);
  FREE( (char *) (m) );
}

