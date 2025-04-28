#include "prob.h"

#define M 122
#define W 32
#define R 31
#define A 0x9908B0DFU
#define U 11
#define S 7
#define T 15
#define L 18
#define B 0x9D2C5680U
#define C 0xEFC60000U
#define F 1812433253U

#define UMASK (0xFFFFFFFFU << R)
#define LMASK (0xFFFFFFFFU >> (W - R))

struct RandEngine rand_engine;

uint32_t random(void){
    uint32_t res;
#pragma omp critical
    {
        if(rand_engine.index >= N){
            #if defined(__AVX512F__)
                for(uint64_t i = 1; i < N/16; i++){
                    __m512i x = rand_engine.state.avx512[i];
                    __m512i x_next = rand_engine.state.avx512[(i + 1) % (N/16)];
                    __m512i mag = _mm512_and_si512(x, _mm512_set1_epi32(UMASK));
                    mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi32(LMASK)));
                    __m512i xA = _mm512_srli_epi32(mag, 1);
                    __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi32(1));
                    xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi32(A)));
                    rand_engine.state.avx512[i] = _mm512_xor_si512(rand_engine.state.avx512[(i + M/16) % (N/16)], xA);
                }
            #elif defined(__AVX2__)
                for (uint64_t i = 0; i < N/8; i++) {
                    __m256i x = rand_engine.state.state256[i];
                    __m256i x_next = rand_engine.state.state256[(i + 1) % (N/8)];
                    __m256i mag = _mm256_and_si256(x, _mm256_set1_epi32(UMASK));
                    mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi32(LMASK)));
                    __m256i xA = _mm256_srli_epi32(mag, 1);
                    __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi32(1));
                    xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi32(A)));
                    rand_engine.state.state256[i] = _mm256_xor_si256(rand_engine.state.state256[(i + M/8) % (N/8)], xA);
                }
            #elif defined(INCLUDE_SSE2)
                for (uint64_t i = 0; i < N/4; i++) {
                    __m128i x = rand_engine.state.state128[i];
                    __m128i x_next = rand_engine.state.state128[(i + 1) % (N/4)];
                    __m128i mag = _mm_and_si128(x, _mm_set1_epi32(UMASK));
                    mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi32(LMASK)));
                    __m128i xA = _mm_srli_epi32(mag, 1);
                    __m128i mask = _mm_and_si128(mag, _mm_set1_epi32(1));
                    xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi32(A)));
                    rand_engine.state.state128[i] = _mm_xor_si128(rand_engine.state.state128[(i + M/4) % (N/4)], xA);
                }
            #else
                for(uint64_t i = 1; i < N; i++){
                    uint32_t x = (rand_engine.state.i32[i] & UMASK) | (rand_engine.state.i32[(i + 1) % N] & LMASK);
                    uint32_t xA = x >> 1;
                    xA ^= (x & 1) * A;
                    rand_engine.state.i32[i] = rand_engine.state.i32[(i + M) % N] ^ xA;
                    #if defined(__GNUC__) || defined(__clang__)
                        __builtin_prefetch((const void*)&(rand_engine.state[(i + 8) % N]), 0, 3);
                    #endif
                }
            #endif
            rand_engine.index = 0;
        }
        res = rand_engine.state.i32[rand_engine.index];
        rand_engine.index += 1;
    }
    res ^= (res >> U);
    res ^= (res << S) & B;
    res ^= (res << T) & C;
    res ^= (res >> L);
    return res;
}

void random_init(const uint32_t seed) {
    rand_engine.state.i32[0] = seed;
    for(uint64_t i = 1; i < N; i++){
        rand_engine.state.i32[i] = F * (rand_engine.state.i32[i - 1] ^ (rand_engine.state.i32[i - 1] >> (W - 2))) + i;
    }
    rand_engine.index = N;
}