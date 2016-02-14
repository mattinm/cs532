#ifndef KMEANS_H
#define KMEANS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
# define M_PI 3.14159265358979323846264338327
#endif

#ifndef COMP_PRECISION
# define COMP_PRECISION 0.0001
#endif

#define RANDOM_DOUBLE(min, max) ((double)(min) + (double)rand() / RAND_MAX * ((double)(max) - (double)(min)))

#ifndef NOPRINT 
# define DEBUG_PRINTF(x) printf x
#else
# define DEBUG_PRINTF(x) do {} while (0)
#endif /* NOPRINTF */

#endif /* KMEANS_H */
