#ifndef FITNESS_DISTANCES_H_INCLUDED
#define FITNESS_DISTANCES_H_INCLUDED

#include "interface.h"

typedef double *(*distance_table_fn)(const struct LGPInput *const in);

// Distance functions
double *euclidean_distances(const struct LGPInput *const in);
double *manhattan_distances(const struct LGPInput *const in);
double *chebyshev_distances(const struct LGPInput *const in);
double *cosine_distances(const struct LGPInput *const in);

#endif