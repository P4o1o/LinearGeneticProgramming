#ifndef PROB_H_INCLUDED
#define PROB_H_INCLUDED
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <float.h>

void random_seed_init(void);

// PROBABILITY
typedef uint64_t prob;
#define MAX_PROB (((prob) RAND_MAX) + 1) // this is 1.0 probability
#define MIN_PROB 0 // this is 0.0 probability
#define PROB_PRECISION 1.0 / MAX_PROB
#define PROBABILITY(val) ((prob)(((double) MAX_PROB) * (val))) // give the probability in prob (uint64_t) from the double from 0.0 to 1.0 rappresenting it
#define INVERSE_PROB(p) (MAX_PROB - (p)) // takes a probability expressed in prob (uint64_t)
#define WILL_HAPPEN(p) ((p) > (prob) rand()) // takes a probability expressed in prob (uint64_t), returns 1 if the event happen else 0

// RANDOM INTEGERS
#define RAND_BOUNDS(min, max) ((min) + (rand() % ((max) - (min) + 1)))
#define RAND_UPTO(max) (rand() % ((max) + 1))

// RANDOM DOUBLE
#define RAND_DOUBLE() (DBL_MIN + ((double)rand() / RAND_MAX) * (DBL_MAX - DBL_MIN))
#define RAND_DBL_BOUNDS(min, max) (min + ((double)rand() / RAND_MAX) * (max - min))

#endif