#ifndef PROB_H_INCLUDED
#define PROB_H_INCLUDED
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <float.h>

uint64_t random(void);

void random_init(const uint32_t seed);

// PROBABILITY
typedef uint64_t prob;
#define MAX_PROB (((prob) RAND_MAX) + 1) // this is 1.0 probability
#define MIN_PROB 0 // this is 0.0 probability
#define PROB_PRECISION 1.0 / MAX_PROB
#define PROBABILITY(val) ((prob)(((double) MAX_PROB) * (val))) // give the probability in prob (uint64_t) from the double from 0.0 to 1.0 rappresenting it
#define INVERSE_PROB(p) (MAX_PROB - (p)) // takes a probability expressed in prob (uint64_t)
#define WILL_HAPPEN(p) ((p) > (prob) rand()) // takes a probability expressed in prob (uint64_t), returns 1 if the event happen else 0

// RANDOM INTEGERS
#define RAND_BOUNDS(min, max) ((min) + ((uint64_t) rand() % ((max) - (min) + (uint64_t) 1)))
#define RAND_UPTO(max) ((uint64_t) rand() % ((max) + (uint64_t) 1))

// RANDOM DOUBLE
#define RAND_DOUBLE() (DBL_MIN + ((double)rand() / RAND_MAX) * (DBL_MAX - DBL_MIN))
#define RAND_DBL_BOUNDS(min, max) (min + ((double)rand() / RAND_MAX) * (max - min))


#define N 624

#if defined(__AVX512F__) || defined(__AVX512DQ__) || defined(__AVX2__)
	#include <immintrin.h>
#else
    #if defined(__SSE2__) /* GNU/Clang */ \
        || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)  /* MSVC */ 
            #include <xmmintrin.h>
            #include <emmintrin>
    #endif
#endif

struct RandEngine{
    alignas(32) uint32_t state[N];
    uint64_t index;
    #if defined(__AVX512F__)
        __m512i state256[N/8];
    #else
        #if defined(__AVX2__)
            __m256i state256[N/4];
        #else
            #if defined(__SSE2__)
                __m128i state128[N/2];
            #endif
    #endif
};

extern struct RandEngine rand_engine;

#endif