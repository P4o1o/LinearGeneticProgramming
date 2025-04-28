#ifndef PROB_H_INCLUDED
#define PROB_H_INCLUDED
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <float.h>

uint32_t random(void);

void random_init(const uint32_t seed);

#define RANDOM_MAX 0xFFFFFFFF

// PROBABILITY
typedef uint64_t prob;
#define MAX_PROB (((prob) RANDOM_MAX) + 1) // this is 1.0 probability
#define MIN_PROB 0 // this is 0.0 probability
#define PROB_PRECISION 1.0 / MAX_PROB
#define PROBABILITY(val) ((prob)(((double) MAX_PROB) * (val))) // give the probability in prob (uint64_t) from the double from 0.0 to 1.0 rappresenting it
#define INVERSE_PROB(p) (MAX_PROB - (p)) // takes a probability expressed in prob (uint64_t)
#define WILL_HAPPEN(p) ((p) > (prob) rand()) // takes a probability expressed in prob (uint64_t), returns 1 if the event happen else 0

// RANDOM INTEGERS
#define RAND_BOUNDS(min, max) ((min) + ((uint64_t) rand() % ((max) - (min) + (uint64_t) 1)))
#define RAND_UPTO(max) ((uint64_t) rand() % ((max) + (uint64_t) 1))

// RANDOM DOUBLE
#define RAND_DOUBLE() (DBL_MIN + ((double)rand() / RANDOM_MAX) * (DBL_MAX - DBL_MIN))
#define RAND_DBL_BOUNDS(min, max) (min + ((double)rand() / RANDOM_MAX) * (max - min))


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
    union RandState{
    #if defined(__AVX512F__)
        alignas(64) uint32_t i32[N];
        alignas(64) __m512i avx512[N/16];
    #else
        #if defined(__AVX2__)
            alignas(32) __m256i avx256[N/8];
            alignas(32) uint32_t i32[N];
        #else
            #if defined(__SSE2__)
                alignas(18) __m128i sse128[N/4];
                alignas(18) uint32_t i32[N];
            #else
                uint32_t i32[N];
            #endif
        #endif
    #endif
    }state;
    uint64_t index;
};

extern struct RandEngine rand_engine;

#endif