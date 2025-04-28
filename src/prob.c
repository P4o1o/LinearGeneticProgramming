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
    uint64_t i = rand_engine.index;
    if(i >= N){
        for(uint64_t j = 1; j < N; j++){
            uint32_t x = (rand_engine.state[j] & UMASK) | (rand_engine.state[(j + 1) % N] & LMASK);
            uint32_t xA = x >> 1;
            xA ^= (x & 1) * A;
            rand_engine.state[j] = rand_engine.state[(j + M) % N] ^ xA;
            #if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
                _mm_prefetch((const char*)&(rand_engine.state[(j + 8) % N]), _MM_HINT_T0);
            #else
                #if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch((const void*)&(rand_engine.state[(j + 8) % N]), 0, 3);
                #endif
            #endif
        }
        rand_engine.index = 0;
        i = 0;
    }
    rand_engine.index = i + 1;
    uint32_t y = rand_engine.state[i];
    y ^= (y >> U);
    y ^= (y << S) & B;
    y ^= (y << T) & C;
    y ^= (y >> L);
    return y;
}

void random_init(const uint32_t seed) {
    rand_engine.state[0] = seed;
    for(uint64_t i = 1; i < N; i++){
        rand_engine.state[i] = F * (rand_engine.state[i - 1] ^ (rand_engine.state[i - 1] >> (W - 2))) + i;
    }
    rand_engine.index = N;
}