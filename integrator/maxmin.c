#include <maxmin.h>
#include <stdlib.h>
#include <stdarg.h>

double maxof(int n_args, ...){
        register int i;
        double max, a;
        va_list ap;

        va_start(ap, n_args);
        max = va_arg(ap, double);
        for(i = 2; i <= n_args; i++) {
                if((a = va_arg(ap, double)) > max)
                        max = a;
        }

        va_end(ap);
        return max;
}

double minof(int n_args, ...){
        register int i;
        double min, a;
        va_list ap;

        va_start(ap, n_args);
        min = va_arg(ap, double);
        for(i = 2; i <= n_args; i++) {
                if((a = va_arg(ap, double)) < min)
                        min = a;
        }

        va_end(ap);
        return min;
}