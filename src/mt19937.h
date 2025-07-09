#ifndef MT19937_H_INCLUDED
#define MT19937_H_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "macros.h"

#define MT19937_STATE_SIZE ((size_t) 624) 

struct MT19937{
    #if defined(INCLUDE_AVX512F)
        alignas(64) uint32_t state[MT19937_STATE_SIZE + 13];
    #elif defined(INCLUDE_AVX2)
        alignas(32) uint32_t state[MT19937_STATE_SIZE + 5];
    #elif defined(INCLUDE_SSE2)
        alignas(16) uint32_t state[MT19937_STATE_SIZE + 1];
    #else
        uint32_t state[MT19937_STATE_SIZE];
    #endif
    size_t next;
};

void init_MT19937(const uint32_t seed, struct MT19937 * rand_engine);
uint32_t get_MT19937(struct MT19937 * rand_engine);
#if defined(INCLUDE_AVX512F)
    __m512i get_16_MT19937(struct MT19937 * rand_engine);
#endif
#if defined(INCLUDE_AVX2) | defined(INCLUDE_AVX512F)
    __m256i get_8_MT19937(struct MT19937 * rand_engine);
#endif
#if defined(INCLUDE_SSE2) | defined(INCLUDE_AVX2) | defined(INCLUDE_AVX512F)
    __m128i get_4_MT19937(struct MT19937 * rand_engine);
#endif

#define MT19937_64_STATE_SIZE ((size_t) 312) 

struct MT19937_64{
    #if defined(INCLUDE_AVX512F)
        alignas(64) uint64_t state[MT19937_64_STATE_SIZE + 4];
    #elif defined(INCLUDE_AVX2)
        alignas(32) uint64_t state[MT19937_64_STATE_SIZE + 1];
    #elif defined(INCLUDE_SSE2)
        alignas(16) uint64_t state[MT19937_64_STATE_SIZE + 1];
    #else
        uint64_t state[MT19937_64_STATE_SIZE];
    #endif
    size_t next;
};

void init_MT19937_64(const uint64_t seed, struct MT19937_64 * rand_engine);
uint64_t get_MT19937_64(struct MT19937_64 * rand_engine);
#if defined(INCLUDE_AVX512F)
    __m512i get_8_MT19937_64(struct MT19937_64 * rand_engine);
#endif
#if defined(INCLUDE_AVX2) | defined(INCLUDE_AVX512F)
    __m256i get_4_MT19937_64(struct MT19937_64 * rand_engine);
#endif
#if defined(INCLUDE_SSE2) | defined(INCLUDE_AVX2) | defined(INCLUDE_AVX512F)
    __m128i get_2_MT19937_64(struct MT19937_64 * rand_engine);
#endif

#endif
