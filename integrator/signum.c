#include <signum.h>

/* signum: returns the sign of x; returns 0 if x==0. */
double signum(double x)
{
  if (x<0) {
    return -1;
  }
  else if (x==0) {
    return 0;
  }
  else if (x>0) {
    return 1;
  }
  else {
    /* must be that x is Not-a-Number */
    return x;
  }
}
