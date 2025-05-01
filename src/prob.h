#ifndef PROB_H_INCLUDED
#define PROB_H_INCLUDED
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <float.h>
#include <omp.h>
#include "macros.h"

uint32_t random(void);

void random_init(const uint32_t seed, uint64_t thread_num);

#define RANDOM_MAX 0xFFFFFFFF

// PROBABILITY
typedef uint64_t prob;
#define MAX_PROB (((prob) RANDOM_MAX) + 1) // this is 1.0 probability
#define MIN_PROB 0 // this is 0.0 probability
#define PROB_PRECISION 1.0 / MAX_PROB
#define PROBABILITY(val) ((prob)(((double) MAX_PROB) * (val))) // give the probability in prob (uint64_t) from the double from 0.0 to 1.0 rappresenting it
#define INVERSE_PROB(p) (MAX_PROB - (p)) // takes a probability expressed in prob (uint64_t)
#define WILL_HAPPEN(p) ((p) > (prob) random()) // takes a probability expressed in prob (uint64_t), returns 1 if the event happen else 0

// RANDOM INTEGERS
#define RAND_BOUNDS(min, max) ((min) + ((uint64_t) random() % ((max) - (min) + (uint64_t) 1)))
#define RAND_UPTO(max) ((uint64_t) random() % ((max) + (uint64_t) 1))

// RANDOM DOUBLE
#define RAND_DOUBLE() (DBL_MIN + ((double)random() / RANDOM_MAX) * (DBL_MAX - DBL_MIN))
#define RAND_DBL_BOUNDS(min, max) (min + ((double)random() / RANDOM_MAX) * (max - min))

#define STATE_SIZE 624 

struct RandEngine{
    #if defined(INCLUDE_AVX512F)
        alignas(64) uint32_t state[STATE_SIZE];
    #elif defined(INCLUDE_AVX2)
        alignas(32) uint32_t state[STATE_SIZE];
    #elif defined(INCLUDE_SSE2)
        alignas(16) uint32_t state[STATE_SIZE];
    #else
        uint32_t state[STATE_SIZE];
    #endif
    uint64_t index;
};

extern struct RandEngine rand_engine[MAX_OMP_THREAD];

#endif