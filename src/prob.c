#include "prob.h"

#define M 397
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

struct RandEngine rand_engine[NUMBER_OF_OMP_THREADS];

uint32_t random(void){
    uint64_t thread_num = omp_get_thread_num();
    uint32_t res;
    if(rand_engine[thread_num].index >= STATE_SIZE){
        #if defined(INCLUDE_AVX512F)
            for(uint64_t i = 0; i < STATE_SIZE/16; i++){
                uint64_t idx = i * 16;
                __m512i x = _mm512_load_si512(rand_engine[thread_num].state + idx);
                __m512i x_next = _mm512_loadu_si512(rand_engine[thread_num].state + ((idx + 1) % (STATE_SIZE)));
                __m512i mag = _mm512_and_si512(x, _mm512_set1_epi32(UMASK));
                mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi32(LMASK)));
                __m512i xA = _mm512_srli_epi32(mag, 1);
                __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi32(1));
                xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi32(A)));
                __m512i lastpiece =  _mm512_loadu_si512(rand_engine[thread_num].state + ((idx + M) % (STATE_SIZE)));
                __m512i result = _mm512_xor_si512(lastpiece, xA);
                _mm512_store_si512(rand_engine[thread_num].state + idx, result);
            }
        #elif defined(INCLUDE_AVX2)
            for (uint64_t i = 0; i < STATE_SIZE/8; i++) {
                uint64_t idx = i * 8;
                __m256i x = _mm256_load_si256((const __m256i*)(rand_engine[thread_num].state + idx));
                __m256i x_next = _mm256_loadu_si256((const __m256i*)(rand_engine[thread_num].state + ((idx + 1) % (STATE_SIZE))));
                __m256i mag = _mm256_and_si256(x, _mm256_set1_epi32(UMASK));
                mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi32(LMASK)));
                __m256i xA = _mm256_srli_epi32(mag, 1);
                __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi32(1));
                xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi32(A)));
                __m256i lastpiece = _mm256_loadu_si256((const __m256i*)(rand_engine[thread_num].state + ((idx + M) % (STATE_SIZE))));
                __m256i result = _mm256_xor_si256(lastpiece, xA);
                _mm256_store_si256((__m256i*)(rand_engine[thread_num].state + idx), result);
            }
        #elif defined(INCLUDE_SSE2)
            for (uint64_t i = 0; i < STATE_SIZE/4; i++) {
                uint64_t idx = i * 4;
                __m128i x = _mm_load_si128((__m128i*)(rand_engine[thread_num].state + idx));
                __m128i x_next = _mm_loadu_si128((__m128i*)(rand_engine[thread_num].state + ((idx + 1) % (STATE_SIZE))));
                __m128i mag = _mm_and_si128(x, _mm_set1_epi32(UMASK));
                mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi32(LMASK)));
                __m128i xA = _mm_srli_epi32(mag, 1);
                __m128i mask = _mm_and_si128(mag, _mm_set1_epi32(1));
                xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi32(A)));
                __m128i lastpiece = _mm_loadu_si128((__m128i*)(rand_engine[thread_num].state + ((idx + M) % (STATE_SIZE))));
                __m128i result = _mm_xor_si128(lastpiece, xA);
                _mm_store_si128((__m128i*)(rand_engine[thread_num].state + idx), result);
            }
        #else
            for(uint64_t i = 0; i < STATE_SIZE; i++){
                uint32_t x = (rand_engine[thread_num].state[i] & UMASK) | (rand_engine[thread_num].state[(i + 1) % STATE_SIZE] & LMASK);
                uint32_t xA = x >> 1;
                xA ^= (x & 1) * A;
                rand_engine[thread_num].state[i] = rand_engine[thread_num].state[(i + M) % STATE_SIZE] ^ xA;
                #if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch((const void*)&(rand_engine[thread_num].state[(i + 8) % STATE_SIZE]), 0, 3);
                #endif
            }
        #endif
        rand_engine[thread_num].index = 0;
    }
    res = rand_engine[thread_num].state[rand_engine[thread_num].index];
    rand_engine[thread_num].index += 1;
    res ^= (res >> U);
    res ^= (res << S) & B;
    res ^= (res << T) & C;
    res ^= (res >> L);
    return res;
}

void random_init(const uint32_t seed, uint64_t thread_num) {
    rand_engine[thread_num].state[0] = seed;
    for(uint64_t i = 1; i < STATE_SIZE; i++){
        rand_engine[thread_num].state[i] = F * (rand_engine[thread_num].state[i - 1] ^ (rand_engine[thread_num].state[i - 1] >> (W - 2))) + i;
    }
    rand_engine[thread_num].index = STATE_SIZE;
}
