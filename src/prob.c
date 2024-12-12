#include "prob.h"

#define SIPROUND \
        v0 += v1; \
        v1 = (v1 << 13) | (v1 >> 51); \
        v1 ^= v0; \
        v0 = (v0 << 32) | (v0 >> 32); \
        v2 += v3; \
        v3 = (v3 << 16) | (v3 >> 48);     \
        v3 ^= v2; \
        v2 += v1; \
        v1 = (v1 << 17) | (v0 >> 47); \
        v1 ^= v2; \
        v2 = (v2 << 32) | (v2 >> 32); \
        v0 += v3; \
        v3 = (v3 << 21) | (v0 >> 43); \
        v3 ^= v0;

#define HASHKEY0_RAND 0x3df52ab9c5671a23ULL
#define HASHKEY1_RAND 0x7a8321f0bc9a8533ULL

void random_seed_init(void) {
    uint64_t v0 = HASHKEY0_RAND ^ 0x736f6d6570736575ULL;
    uint64_t v1 = HASHKEY1_RAND ^ 0x646f72616e646f6dULL;
    uint64_t v2 = HASHKEY0_RAND ^ 0x6c7967656e657261ULL;
    uint64_t v3 = HASHKEY1_RAND ^ 0x7465646279746573ULL;
    uint64_t epoch = time(NULL);
    v3 ^= epoch;
    SIPROUND;
    SIPROUND;
    v0 ^= epoch;
    uint64_t cpuclock = clock();
    v3 ^= cpuclock;
    SIPROUND;
    SIPROUND;
    v0 ^= cpuclock;
    v2 ^= 0xff;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    SIPROUND;
	srand(v0 ^ v1 ^ v2 ^ v3);
}