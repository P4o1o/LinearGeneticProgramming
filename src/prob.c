#include "prob.h"

#define N 624
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

struct RandEngine rand_engine;

uint64_t random(void){
    uint32_t res;
#pragma omp critical
    {
        if(rand_engine.index >= N){
            for(uint64_t i = 1; i < N; i++){
                uint32_t x = (rand_engine.state[i] & UMASK) | (rand_engine.state[(i + 1) % N] & LMASK);
                uint32_t xA = x >> 1;
                xA ^= (x & 1) * A;
                rand_engine.state[i] = rand_engine.state[(i + M) % N] ^ xA;
                #if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
                    _mm_prefetch((const char*)&(rand_engine.state[(i + 8) % N]), _MM_HINT_T0);
                #else
                    #if defined(__GNUC__) || defined(__clang__)
                        __builtin_prefetch((const void*)&(rand_engine.state[(i + 8) % N]), 0, 3);
                    #endif
                #endif
            }
            rand_engine.index = 0;
        }
        res = rand_engine.state[rand_engine.index];
        rand_engine.index += 1;
    }
    res ^= (res >> U);
    res ^= (res << S) & B;
    res ^= (res << T) & C;
    res ^= (res >> L);
    return res;
}

void random_init(const uint32_t seed) {
    rand_engine.state[0] = seed;
    for(uint64_t i = 1; i < N; i++){
        rand_engine.state[i] = F * (rand_engine.state[i - 1] ^ (rand_engine.state[i - 1] >> (W - 2))) + i;
    }
    rand_engine.index = N;
}