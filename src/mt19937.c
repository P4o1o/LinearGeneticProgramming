#include "mt19937.h"

#define MIDDLE_WORD_R32 ((size_t) 397)
#define WORD_SIZE_R32 32
#define SEPARATOR_POINT_R32 31
#define TWIST_COEFF_R32 0x9908B0DFU
#define U_TEMPERING_R32 11
#define S_TEMPERING_R32 7
#define T_TEMPERING_R32 15
#define L_TEMPERING_R32 18
#define B_MASK_R32 0x9D2C5680U
#define C_MASK_R32 0xEFC60000U
#define INITIAL_MULTIPLIER_R32 1812433253U

#define U_MASK_R32 (0xFFFFFFFFU << SEPARATOR_POINT_R32)
#define L_MASK_R32 (0xFFFFFFFFU >> (WORD_SIZE_R32 - SEPARATOR_POINT_R32))

void init_MT19937(const uint32_t seed, struct MT19937 * rand_engine) {
    rand_engine->state[0] = seed;
    for(size_t i = 1; i < MT19937_STATE_SIZE; i++){
        rand_engine->state[i] = INITIAL_MULTIPLIER_R32 * (rand_engine->state[i - 1] ^ (rand_engine->state[i - 1] >> (WORD_SIZE_R32 - 2))) + i;
    }
    rand_engine->next = MT19937_STATE_SIZE;
}

static inline void twist_MT19937(struct MT19937 * rand_engine){
    #if defined(INCLUDE_AVX512F)
        __m512i x = _mm512_load_si512(rand_engine->state);
        __m512i x_next = _mm512_loadu_si512(rand_engine->state + 1);
        __m512i mag = _mm512_and_si512(x, _mm512_set1_epi32(U_MASK_R32));
        mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi32(L_MASK_R32)));
        __m512i xA = _mm512_srli_epi32(mag, 1);
        __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi32(1));
        xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi32(TWIST_COEFF_R32)));
        __m512i lastpiece =  _mm512_loadu_si512(rand_engine->state + MIDDLE_WORD_R32);
        __m512i result = _mm512_xor_si512(lastpiece, xA);
        _mm512_store_si512(rand_engine->state, result);
        memcpy(rand_engine->state + MT19937_64_STATE_SIZE, rand_engine->state, 13 * sizeof(uint32_t));
        for(size_t idx = 16; idx < MT19937_STATE_SIZE - MIDDLE_WORD_R32; idx += 16){
            uint32_t *now = rand_engine->state + idx
            __m512i x = _mm512_load_si512(now);
            __m512i x_next = _mm512_loadu_si512(now + 1);
            __m512i mag = _mm512_and_si512(x, _mm512_set1_epi32(U_MASK_R32));
            mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi32(L_MASK_R32)));
            __m512i xA = _mm512_srli_epi32(mag, 1);
            __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi32(1));
            xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi32(TWIST_COEFF_R32)));
            __m512i lastpiece =  _mm512_loadu_si512(now + MIDDLE_WORD_R32);
            __m512i result = _mm512_xor_si512(lastpiece, xA);
            _mm512_store_si512(now, result);
        }
        for(size_t idx = MT19937_STATE_SIZE - MIDDLE_WORD_R32 + 13; idx < MT19937_STATE_SIZE; idx += 16){
            uint32_t *now = rand_engine->state + idx
            __m512i x = _mm512_load_si512(now);
            __m512i x_next = _mm512_loadu_si512(now + 1);
            __m512i mag = _mm512_and_si512(x, _mm512_set1_epi32(U_MASK_R32));
            mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi32(L_MASK_R32)));
            __m512i xA = _mm512_srli_epi32(mag, 1);
            __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi32(1));
            xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi32(TWIST_COEFF_R32)));
            __m512i lastpiece =  _mm512_loadu_si512(now + MIDDLE_WORD_R32 - MT19937_STATE_SIZE);
            __m512i result = _mm512_xor_si512(lastpiece, xA);
            _mm512_store_si512(now, result);
        }
    #elif defined(INCLUDE_AVX2)
        __m256i x = _mm256_load_si256((const __m256i*)rand_engine->state);
        __m256i x_next = _mm256_loadu_si256((const __m256i*)(rand_engine->state + 1));
        __m256i mag = _mm256_and_si256(x, _mm256_set1_epi32(U_MASK_R32));
        mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi32(L_MASK_R32)));
        __m256i xA = _mm256_srli_epi32(mag, 1);
        __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi32(1));
        xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi32(TWIST_COEFF_R32)));
        __m256i lastpiece = _mm256_loadu_si256((const __m256i*)(rand_engine->state + MIDDLE_WORD_R32));
        __m256i result = _mm256_xor_si256(lastpiece, xA);
        _mm256_store_si256((__m256i*)rand_engine->state, result);
        memcpy(rand_engine->state + MT19937_64_STATE_SIZE, rand_engine->state, 5 * sizeof(uint32_t));
        for (size_t idx = 8; idx < MT19937_STATE_SIZE - MIDDLE_WORD_R32; idx += 8) {
            uint32_t *now = rand_engine->state + idx
            __m256i x = _mm256_load_si256((const __m256i*) now);
            __m256i x_next = _mm256_loadu_si256((const __m256i*)(now + 1));
            __m256i mag = _mm256_and_si256(x, _mm256_set1_epi32(U_MASK_R32));
            mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi32(L_MASK_R32)));
            __m256i xA = _mm256_srli_epi32(mag, 1);
            __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi32(1));
            xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi32(TWIST_COEFF_R32)));
            __m256i lastpiece = _mm256_loadu_si256((const __m256i*)(now + MIDDLE_WORD_R32));
            __m256i result = _mm256_xor_si256(lastpiece, xA);
            _mm256_store_si256((__m256i*) now, result);
        }
        for(size_t idx = MT19937_STATE_SIZE - MIDDLE_WORD_R32 + 5; idx < MT19937_STATE_SIZE; idx += 8){
            uint32_t *now = rand_engine->state + idx
            __m256i x = _mm256_load_si256((const __m256i*) now);
            __m256i x_next = _mm256_loadu_si256((const __m256i*)(now + 1));
            __m256i mag = _mm256_and_si256(x, _mm256_set1_epi32(U_MASK_R32));
            mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi32(L_MASK_R32)));
            __m256i xA = _mm256_srli_epi32(mag, 1);
            __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi32(1));
            xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi32(TWIST_COEFF_R32)));
            __m256i lastpiece = _mm256_loadu_si256((const __m256i*)(now + MIDDLE_WORD_R32 - MT19937_STATE_SIZE));
            __m256i result = _mm256_xor_si256(lastpiece, xA);
            _mm256_store_si256((__m256i*) now, result);
        }
    #elif defined(INCLUDE_SSE2)
        __m128i x = _mm_load_si128((__m128i*)(rand_engine->state));
        __m128i x_next = _mm_loadu_si128((__m128i*)(rand_engine->state + 1));
        __m128i mag = _mm_and_si128(x, _mm_set1_epi32(U_MASK_R32));
        mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi32(L_MASK_R32)));
        __m128i xA = _mm_srli_epi32(mag, 1);
        __m128i mask = _mm_and_si128(mag, _mm_set1_epi32(1));
        xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi32(TWIST_COEFF_R32)));
        __m128i lastpiece = _mm_loadu_si128((__m128i*)(rand_engine->state + MIDDLE_WORD_R32));
        __m128i result = _mm_xor_si128(lastpiece, xA);
        _mm_store_si128((__m128i*)rand_engine->state, result);
        rand_engine->state[MT19937_STATE_SIZE] = rand_engine->state[0];
        for (size_t idx = 4; idx < MT19937_STATE_SIZE - MIDDLE_WORD_R32; idx += 4) {
            uint32_t *now = rand_engine->state + idx
            __m128i x = _mm_load_si128((__m128i*) now);
            __m128i x_next = _mm_loadu_si128((__m128i*)(now + 1));
            __m128i mag = _mm_and_si128(x, _mm_set1_epi32(U_MASK_R32));
            mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi32(L_MASK_R32)));
            __m128i xA = _mm_srli_epi32(mag, 1);
            __m128i mask = _mm_and_si128(mag, _mm_set1_epi32(1));
            xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi32(TWIST_COEFF_R32)));
            __m128i lastpiece = _mm_loadu_si128((__m128i*)(now + MIDDLE_WORD_R32));
            __m128i result = _mm_xor_si128(lastpiece, xA);
            _mm_store_si128((__m128i*) now, result);
        }
        for(size_t idx = MT19937_STATE_SIZE - MIDDLE_WORD_R32 + 1; idx < MT19937_STATE_SIZE; idx += 4){
            uint32_t *now = rand_engine->state + idx
            __m128i x = _mm_load_si128((__m128i*)(now));
            __m128i x_next = _mm_loadu_si128((__m128i*)(now + 1));
            __m128i mag = _mm_and_si128(x, _mm_set1_epi32(U_MASK_R32));
            mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi32(L_MASK_R32)));
            __m128i xA = _mm_srli_epi32(mag, 1);
            __m128i mask = _mm_and_si128(mag, _mm_set1_epi32(1));
            xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi32(TWIST_COEFF_R32)));
            __m128i lastpiece = _mm_loadu_si128((__m128i*)(now + MIDDLE_WORD_R32 - MT19937_STATE_SIZE));
            __m128i result = _mm_xor_si128(lastpiece, xA);
            _mm_store_si128((__m128i*)(now), result);
        }
    #elif defined(INCLUDE_NEON)
        uint32x4_t x = vld1q_u32(rand_engine->state);
        uint32x4_t x_next = vld1q_u32(rand_engine->state + 1);
        uint32x4_t mag = vandq_u32(x, vdupq_n_u32(U_MASK_R32));
        mag = vorrq_u32(mag, vandq_u32(x_next, vdupq_n_u32(L_MASK_R32)));
        uint32x4_t xA = vshrq_n_u32(mag, 1);
        uint32x4_t mask = vandq_u32(mag, vdupq_n_u32(1));
        xA = veorq_u32(xA, vandq_u32(mask, vdupq_n_u32(TWIST_COEFF_R32)));
        uint32x4_t lastpiece = vld1q_u32(rand_engine->state + MIDDLE_WORD_R32);
        uint32x4_t result = veorq_u32(lastpiece, xA);
        vst1q_u32(rand_engine->state, result);
        rand_engine->state[MT19937_STATE_SIZE] = rand_engine->state[0];
        for (size_t idx = 4; idx < MT19937_STATE_SIZE - MIDDLE_WORD_R32; idx += 4) {
            uint32_t *now = rand_engine->state + idx
            uint32x4_t x = vld1q_u32(now);
            uint32x4_t x_next = vld1q_u32(now + 1);
            uint32x4_t mag = vandq_u32(x, vdupq_n_u32(U_MASK_R32));
            mag = vorrq_u32(mag, vandq_u32(x_next, vdupq_n_u32(L_MASK_R32)));
            uint32x4_t xA = vshrq_n_u32(mag, 1);
            uint32x4_t mask = vandq_u32(mag, vdupq_n_u32(1));
            xA = veorq_u32(xA, vandq_u32(mask, vdupq_n_u32(TWIST_COEFF_R32)));
            uint32x4_t lastpiece = vld1q_u32(now + MIDDLE_WORD_R32);
            uint32x4_t result = veorq_u32(lastpiece, xA);
            vst1q_u32(now, result);
        }
        for (size_t idx = MT19937_STATE_SIZE - MIDDLE_WORD_R32 + 1; idx < MT19937_STATE_SIZE; idx += 4) {
            uint32_t *now = rand_engine->state + idx
            uint32x4_t x = vld1q_u32(now);
            uint32x4_t x_next = vld1q_u32(now + 1);
            uint32x4_t mag = vandq_u32(x, vdupq_n_u32(U_MASK_R32));
            mag = vorrq_u32(mag, vandq_u32(x_next, vdupq_n_u32(L_MASK_R32)));
            uint32x4_t xA = vshrq_n_u32(mag, 1);
            uint32x4_t mask = vandq_u32(mag, vdupq_n_u32(1));
            xA = veorq_u32(xA, vandq_u32(mask, vdupq_n_u32(TWIST_COEFF_R32)));
            uint32x4_t lastpiece = vld1q_u32(now + MIDDLE_WORD_R32 - MT19937_STATE_SIZE);
            uint32x4_t result = veorq_u32(lastpiece, xA);
            vst1q_u32(now, result);
        }
    #else
        size_t i;
        for(i = 0; i < MT19937_STATE_SIZE - MIDDLE_WORD_R32; i++){
            uint32_t x = (rand_engine->state[i] & U_MASK_R32) | (rand_engine->state[i + 1] & L_MASK_R32);
            uint32_t xA = x >> 1;
            xA ^= (x & 1) * TWIST_COEFF_R32;
            rand_engine->state[i] = rand_engine->state[(i + MIDDLE_WORD_R32)] ^ xA;
        }
        for(; i < MT19937_STATE_SIZE - 1; i++){
            uint32_t x = (rand_engine->state[i] & U_MASK_R32) | (rand_engine->state[i + 1] & L_MASK_R32);
            uint32_t xA = x >> 1;
            xA ^= (x & 1) * TWIST_COEFF_R32;
            rand_engine->state[i] = rand_engine->state[(i + MIDDLE_WORD_R32) - MT19937_STATE_SIZE] ^ xA;
        }
        uint32_t x = (rand_engine->state[MT19937_STATE_SIZE - 1] & U_MASK_R32) | (rand_engine->state[0] & L_MASK_R32);
        uint32_t xA = x >> 1;
        xA ^= (x & 1) * TWIST_COEFF_R32;
        rand_engine->state[MT19937_STATE_SIZE - 1] = rand_engine->state[MIDDLE_WORD_R32 - 1] ^ xA;

    #endif
    rand_engine->next = 0;
}

uint32_t get_MT19937(struct MT19937 * rand_engine){
    uint32_t res;
    if(rand_engine->next >= MT19937_STATE_SIZE){
        twist_MT19937(rand_engine);
    }
    res = rand_engine->state[rand_engine->next];
    rand_engine->next++;
    res ^= (res >> U_TEMPERING_R32);
    res ^= (res << S_TEMPERING_R32) & B_MASK_R32;
    res ^= (res << T_TEMPERING_R32) & C_MASK_R32;
    res ^= (res >> L_TEMPERING_R32);
    return res;
}

#if defined(INCLUDE_AVX512F)
__m512i get_16_MT19937(struct MT19937 * rand_engine){
    __m512i res;
    if(rand_engine->next >= MT19937_STATE_SIZE + 16){
        twist_MT19937(rand_engine);
        res = _mm512_load_si512(rand_engine->state + rand_engine->next);
    }else if(rand_engine->next % 16 == 0){
        res = _mm512_load_si512(rand_engine->state + rand_engine->next);
    }else{
        res = _mm512_loadu_si512(rand_engine->state + rand_engine->next);
    }
    rand_engine->next += 16;
    res = _mm512_xor_si512(res, _mm512_srli_epi32(res, U_TEMPERING_R32));
    res = _mm512_xor_si512(res, _mm512_and_si512(_mm512_slli_epi32(res, S_TEMPERING_R32), _mm512_set1_epi32(B_MASK_R32)));
    res = _mm512_xor_si512(res, _mm512_and_si512(_mm512_slli_epi32(res, T_TEMPERING_R32), _mm512_set1_epi32(C_MASK_R32)));
    res = _mm512_xor_si512(res, _mm512_srli_epi32(res, L_TEMPERING_R32));
    return res;
}
#endif

#if defined(INCLUDE_AVX2) || defined(INCLUDE_AVX512F)
__m256i get_8_MT19937(struct MT19937 * rand_engine){
    __m256i res;
    if(rand_engine->next >= MT19937_STATE_SIZE + 8){
        twist_MT19937(rand_engine);
        res = _mm256_load_si256((const __m256i*)(rand_engine->state + rand_engine->next));
    }else if(rand_engine->next % 8 == 0){
        res = _mm256_load_si256((const __m256i*)(rand_engine->state + rand_engine->next));
    }else{
        res = _mm256_loadu_si256((const __m256i*)(rand_engine->state + rand_engine->next));
    }
    rand_engine->next += 8;
    res = _mm256_xor_si256(res, _mm256_srli_epi32(res, U_TEMPERING_R32));
    res = _mm256_xor_si256(res, _mm256_and_si256(_mm256_slli_epi32(res, S_TEMPERING_R32), _mm256_set1_epi32(B_MASK_R32)));
    res = _mm256_xor_si256(res, _mm256_and_si256(_mm256_slli_epi32(res, T_TEMPERING_R32), _mm256_set1_epi32(C_MASK_R32)));
    res = _mm256_xor_si256(res, _mm256_srli_epi32(res, L_TEMPERING_R32));
    return res;
}
#endif

#if defined(INCLUDE_SSE2) || defined(INCLUDE_AVX2) || defined(INCLUDE_AVX512F)
__m128i get_4_MT19937(struct MT19937 * rand_engine){
    __m128i res;
    if(rand_engine->next >= MT19937_STATE_SIZE + 4){
        twist_MT19937(rand_engine);
        res = _mm_load_si128((__m128i*) (rand_engine->state + rand_engine->next));
    }else if(rand_engine->next % 4 == 0){
        res = _mm_load_si128((__m128i*) (rand_engine->state + rand_engine->next));
    }else{
        res = _mm_loadu_si128((__m128i*) (rand_engine->state + rand_engine->next));
    }
    rand_engine->next += 4;
    res = _mm_xor_si128(res, _mm_srli_epi32(res, U_TEMPERING_R32));
    res = _mm_xor_si128(res, _mm_and_si128(_mm_slli_epi32(res, S_TEMPERING_R32), _mm_set1_epi32(B_MASK_R32)));
    res = _mm_xor_si128(res, _mm_and_si128(_mm_slli_epi32(res, T_TEMPERING_R32), _mm_set1_epi32(C_MASK_R32)));
    res = _mm_xor_si128(res, _mm_srli_epi32(res, L_TEMPERING_R32));
    return res;
}
#elif defined(INCLUDE_NEON)
uint32x4_t get_4_MT19937(struct MT19937 * rand_engine){
    if(rand_engine->next >= MT19937_STATE_SIZE + 4){
        twist_MT19937(rand_engine);
    }
    uint32x4_t res = vld1q_u32(rand_engine->state + rand_engine->next);
    rand_engine->next += 4;
    res = veorq_u32(res, vshrq_n_u32(res, U_TEMPERING_R32));
    res = veorq_u32(res, vandq_u32(vshlq_n_u32(res, S_TEMPERING_R32), vdupq_n_u32(B_MASK_R32)));
    res = veorq_u32(res, vandq_u32(vshlq_n_u32(res, T_TEMPERING_R32), vdupq_n_u32(C_MASK_R32)));
    res = veorq_u32(res, vshrq_n_u32(res, L_TEMPERING_R32));
    return res;
}
#endif

#define MIDDLE_WORD_R64 ((size_t) 156)
#define WORD_SIZE_R64 64
#define SEPARATOR_POINT_R64 31
#define TWIST_COEFF_R64 0xB5026F5AA96619E9UL
#define U_TEMPERING_R64 29
#define S_TEMPERING_R64 17
#define T_TEMPERING_R64 37
#define L_TEMPERING_R64 43
#define B_MASK_R64 0x71D67FFFEDA60000UL
#define C_MASK_R64 0xFFF7EEE000000000UL
#define INITIAL_MULTIPLIER_R64 6364136223846793005UL

#define U_MASK_R64 (0xFFFFFFFFFFFFFFFFUL << SEPARATOR_POINT_R64)
#define L_MASK_R64 (0xFFFFFFFFFFFFFFFFUL >> (WORD_SIZE_R64 - SEPARATOR_POINT_R64))

void init_MT19937_64(const uint64_t seed, struct MT19937_64 * rand_engine){
    rand_engine->state[0] = seed;
    for(size_t i = 1; i < MT19937_64_STATE_SIZE; i++){
        rand_engine->state[i] = INITIAL_MULTIPLIER_R64 * (rand_engine->state[i - 1] ^ (rand_engine->state[i - 1] >> (WORD_SIZE_R64 - 2))) + i;
    }
    rand_engine->next = MT19937_64_STATE_SIZE;
}

static inline void twist_MT19937_64(struct MT19937_64 * rand_engine){
    #if defined(INCLUDE_AVX512F)
        __m512i x = _mm512_load_si512(rand_engine->state);
        __m512i x_next = _mm512_loadu_si512(rand_engine->state + 1);
        __m512i mag = _mm512_and_si512(x, _mm512_set1_epi64(U_MASK_R64));
        mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi64(L_MASK_R64)));
        __m512i xA = _mm512_srli_epi64(mag, 1);
        __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi64(1UL));
        xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi64(TWIST_COEFF_R64)));
        __m512i lastpiece =  _mm512_loadu_si512(rand_engine->state + MIDDLE_WORD_R64);
        __m512i result = _mm512_xor_si512(lastpiece, xA);
        _mm512_store_si512(rand_engine->state, result);
        memcpy(rand_engine->state + MT19937_64_STATE_SIZE, rand_engine->state, 4 * sizeof(uint64_t));
        for (size_t idx = 8; idx < MT19937_64_STATE_SIZE - MIDDLE_WORD_R64; idx += 8){
            uint64_t *now = rand_engine->state + idx;
            __m512i x = _mm512_load_si512(now);
            __m512i x_next = _mm512_loadu_si512(now + 1);
            __m512i mag = _mm512_and_si512(x, _mm512_set1_epi64(U_MASK_R64));
            mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi64(L_MASK_R64)));
            __m512i xA = _mm512_srli_epi64(mag, 1);
            __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi64(1UL));
            xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi64(TWIST_COEFF_R64)));
            __m512i lastpiece =  _mm512_loadu_si512(now + MIDDLE_WORD_R64);
            __m512i result = _mm512_xor_si512(lastpiece, xA);
            _mm512_store_si512(now, result);
        }
        for(size_t idx = MT19937_64_STATE_SIZE - MIDDLE_WORD_R64 + 4; idx < MT19937_64_STATE_SIZE; idx += 8){
            uint64_t *now = rand_engine->state + idx;
            __m512i x = _mm512_load_si512(now);
            __m512i x_next = _mm512_loadu_si512(now + 1);
            __m512i mag = _mm512_and_si512(x, _mm512_set1_epi64(U_MASK_R64));
            mag = _mm512_or_si512(mag, _mm512_and_si512(x_next, _mm512_set1_epi64(L_MASK_R64)));
            __m512i xA = _mm512_srli_epi64(mag, 1);
            __m512i mask = _mm512_and_si512(mag, _mm512_set1_epi64(1UL));
            xA = _mm512_xor_si512(xA, _mm512_and_si512(mask, _mm512_set1_epi64(TWIST_COEFF_R64)));
            __m512i lastpiece =  _mm512_loadu_si512(now + MIDDLE_WORD_R64 - MT19937_64_STATE_SIZE);
            __m512i result = _mm512_xor_si512(lastpiece, xA);
            _mm512_store_si512(now, result);
        }
    #elif defined(INCLUDE_AVX2)
        __m256i x = _mm256_load_si256((const __m256i*)rand_engine->state);
        __m256i x_next = _mm256_loadu_si256((const __m256i*)(rand_engine->state + 1));
        __m256i mag = _mm256_and_si256(x, _mm256_set1_epi64x(U_MASK_R64));
        mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi64x(L_MASK_R64)));
        __m256i xA = _mm256_srli_epi64(mag, 1);
        __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi64x(1UL));
        xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi64x(TWIST_COEFF_R64)));
        __m256i lastpiece = _mm256_load_si256((const __m256i*)(rand_engine->state + MIDDLE_WORD_R64));
        __m256i result = _mm256_xor_si256(lastpiece, xA);
        _mm256_store_si256((__m256i*) rand_engine->state, result);
        rand_engine->state[MT19937_64_STATE_SIZE] = rand_engine->state[0];
        for (size_t idx = 4; idx < MT19937_64_STATE_SIZE - MIDDLE_WORD_R64; idx += 4) {
            uint64_t *now = rand_engine->state + idx;
            __m256i x = _mm256_load_si256((const __m256i*) now);
            __m256i x_next = _mm256_loadu_si256((const __m256i*)(now + 1));
            __m256i mag = _mm256_and_si256(x, _mm256_set1_epi64x( U_MASK_R64));
            mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi64x(L_MASK_R64)));
            __m256i xA = _mm256_srli_epi64(mag, 1);
            __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi64x(1UL));
            xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi64x(TWIST_COEFF_R64)));
            __m256i lastpiece = _mm256_load_si256((const __m256i*)(now + MIDDLE_WORD_R64));
            __m256i result = _mm256_xor_si256(lastpiece, xA);
            _mm256_store_si256((__m256i*) now, result);
        }
        for(size_t idx = MT19937_64_STATE_SIZE - MIDDLE_WORD_R64; idx < MT19937_64_STATE_SIZE; idx += 4){
            uint64_t *now = rand_engine->state + idx;
            __m256i x = _mm256_load_si256((const __m256i*) now);
            __m256i x_next = _mm256_loadu_si256((const __m256i*)(now + 1));
            __m256i mag = _mm256_and_si256(x, _mm256_set1_epi64x(U_MASK_R64));
            mag = _mm256_or_si256(mag, _mm256_and_si256(x_next, _mm256_set1_epi64x(L_MASK_R64)));
            __m256i xA = _mm256_srli_epi64(mag, 1);
            __m256i mask = _mm256_and_si256(mag, _mm256_set1_epi64x(1UL));
            xA = _mm256_xor_si256(xA, _mm256_and_si256(mask, _mm256_set1_epi64x(TWIST_COEFF_R64)));
            __m256i lastpiece = _mm256_load_si256((const __m256i*)(now + MIDDLE_WORD_R64 - MT19937_64_STATE_SIZE));
            __m256i result = _mm256_xor_si256(lastpiece, xA);
            _mm256_store_si256((__m256i*) now, result);
        }
    #elif defined(INCLUDE_SSE2)
        __m128i x = _mm_load_si128((__m128i*)(rand_engine->state));
        __m128i x_next = _mm_loadu_si128((__m128i*)(rand_engine->state + 1));
        __m128i mag = _mm_and_si128(x, _mm_set1_epi64x(U_MASK_R64));
        mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi64x(L_MASK_R64)));
        __m128i xA = _mm_srli_epi64(mag, 1);
        __m128i mask = _mm_and_si128(mag, _mm_set1_epi64x(1UL));
        xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi64x(TWIST_COEFF_R64)));
        __m128i lastpiece = _mm_load_si128((__m128i*)(rand_engine->state + MIDDLE_WORD_R64));
        __m128i result = _mm_xor_si128(lastpiece, xA);
        _mm_store_si128((__m128i*)rand_engine->state, result);
        rand_engine->state[MT19937_64_STATE_SIZE] = rand_engine->state[0];
        for (size_t idx = 2; idx < MT19937_64_STATE_SIZE - MIDDLE_WORD_R64; idx += 2) {
            uint64_t *now = rand_engine->state + idx;
            __m128i x = _mm_load_si128((__m128i*) now);
            __m128i x_next = _mm_loadu_si128((__m128i*)(now + 1));
            __m128i mag = _mm_and_si128(x, _mm_set1_epi64x(U_MASK_R64));
            mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi64x(L_MASK_R64)));
            __m128i xA = _mm_srli_epi64(mag, 1);
            __m128i mask = _mm_and_si128(mag, _mm_set1_epi64x(1UL));
            xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi64x(TWIST_COEFF_R64)));
            __m128i lastpiece = _mm_load_si128((__m128i*)(now + MIDDLE_WORD_R64));
            __m128i result = _mm_xor_si128(lastpiece, xA);
            _mm_store_si128((__m128i*) now, result);
        }
        for(size_t idx = MT19937_64_STATE_SIZE - MIDDLE_WORD_R64; idx < MT19937_64_STATE_SIZE; idx += 2){
            uint64_t *now = rand_engine->state + idx;
            __m128i x = _mm_load_si128((__m128i*)(now));
            __m128i x_next = _mm_loadu_si128((__m128i*)(now + 1));
            __m128i mag = _mm_and_si128(x, _mm_set1_epi64x(U_MASK_R64));
            mag = _mm_or_si128(mag, _mm_and_si128(x_next, _mm_set1_epi64x(L_MASK_R64)));
            __m128i xA = _mm_srli_epi64(mag, 1);
            __m128i mask = _mm_and_si128(mag, _mm_set1_epi64x(1UL));
            xA = _mm_xor_si128(xA, _mm_and_si128(mask, _mm_set1_epi64x(TWIST_COEFF_R64)));
            __m128i lastpiece = _mm_load_si128((__m128i*)(now + MIDDLE_WORD_R64 - MT19937_64_STATE_SIZE));
            __m128i result = _mm_xor_si128(lastpiece, xA);
            _mm_store_si128((__m128i*) now, result);
        }
    #elif defined(INCLUDE_NEON)
        uint64x2_t x = vld1q_u64(rand_engine->state);
        uint64x2_t x_next = vld1q_u64(rand_engine->state + 1);
        uint64x2_t mag = vandq_u64(x, vdupq_n_u64(U_MASK_R64));
        mag = vorrq_u64(mag, vandq_u64(x_next, vdupq_n_u64(L_MASK_R64)));
        uint64x2_t xA = vshrq_n_u64(mag, 1);
        uint64x2_t mask = vandq_u64(mag, vdupq_n_u64(1ULL));
        xA = veorq_u64(xA, vandq_u64(mask, vdupq_n_u64(TWIST_COEFF_R64)));
        uint64x2_t lastpiece = vld1q_u64(rand_engine->state + MIDDLE_WORD_R64);
        uint64x2_t result = veorq_u64(lastpiece, xA);
        vst1q_u64(rand_engine->state, result);
        rand_engine->state[MT19937_64_STATE_SIZE] = rand_engine->state[0];
        for (size_t idx = 2; idx < MT19937_64_STATE_SIZE - MIDDLE_WORD_R64; idx += 2) {
            uint64_t *now = rand_engine->state + idx;
            uint64x2_t x = vld1q_u64(now);
            uint64x2_t x_next = vld1q_u64(now + 1);
            uint64x2_t mag = vandq_u64(x, vdupq_n_u64(U_MASK_R64));
            mag = vorrq_u64(mag, vandq_u64(x_next, vdupq_n_u64(L_MASK_R64)));
            uint64x2_t xA = vshrq_n_u64(mag, 1);
            uint64x2_t mask = vandq_u64(mag, vdupq_n_u64(1ULL));
            xA = veorq_u64(xA, vandq_u64(mask, vdupq_n_u64(TWIST_COEFF_R64)));
            uint64x2_t lastpiece = vld1q_u64(now + MIDDLE_WORD_R64);
            uint64x2_t result = veorq_u64(lastpiece, xA);
            vst1q_u64(now, result);
        }
        for (size_t idx = MT19937_64_STATE_SIZE - MIDDLE_WORD_R64 + 1; idx < MT19937_64_STATE_SIZE; idx += 2) {
            uint64_t *now = rand_engine->state + idx;
            uint64x2_t x = vld1q_u64(now);
            uint64x2_t x_next = vld1q_u64(now + 1);
            uint64x2_t mag = vandq_u64(x, vdupq_n_u64(U_MASK_R64));
            mag = vorrq_u64(mag, vandq_u64(x_next, vdupq_n_u64(L_MASK_R64)));
            uint64x2_t xA = vshrq_n_u64(mag, 1);
            uint64x2_t mask = vandq_u64(mag, vdupq_n_u64(1ULL));
            xA = veorq_u64(xA, vandq_u64(mask, vdupq_n_u64(TWIST_COEFF_R64)));
            uint64x2_t lastpiece = vld1q_u64(now + MIDDLE_WORD_R64 - MT19937_64_STATE_SIZE);
            uint64x2_t result = veorq_u64(lastpiece, xA);
            vst1q_u64(now, result);
        }
    #else
        size_t i;
        for(i = 0; i < MT19937_64_STATE_SIZE - MIDDLE_WORD_R64; i++){
            uint64_t x = (rand_engine->state[i] & U_MASK_R64) | (rand_engine->state[i + 1] & L_MASK_R64);
            uint64_t xA = x >> 1;
            xA ^= (x & 1) * TWIST_COEFF_R64;
            rand_engine->state[i] = rand_engine->state[(i + MIDDLE_WORD_R64)] ^ xA;
        }
        for(; i < MT19937_64_STATE_SIZE - 1; i++){
            uint64_t x = (rand_engine->state[i] & U_MASK_R64) | (rand_engine->state[i + 1] & L_MASK_R64);
            uint64_t xA = x >> 1;
            xA ^= (x & 1) * TWIST_COEFF_R64;
            rand_engine->state[i] = rand_engine->state[(i + MIDDLE_WORD_R64) - MT19937_64_STATE_SIZE] ^ xA;
        }
        uint64_t x = (rand_engine->state[MT19937_64_STATE_SIZE - 1] & U_MASK_R64) | (rand_engine->state[0] & L_MASK_R64);
        uint64_t xA = x >> 1;
        xA ^= (x & 1) * TWIST_COEFF_R64;
        rand_engine->state[MT19937_64_STATE_SIZE - 1] = rand_engine->state[MIDDLE_WORD_R64 - 1] ^ xA;

    #endif
    rand_engine->next = 0;
}

uint64_t get_MT19937_64(struct MT19937_64 * rand_engine){
    uint64_t res;
    if(rand_engine->next >= MT19937_64_STATE_SIZE){
        twist_MT19937_64(rand_engine);
    }
    res = rand_engine->state[rand_engine->next];
    rand_engine->next++;
    res ^= (res >> U_TEMPERING_R64);
    res ^= (res << S_TEMPERING_R64) & B_MASK_R64;
    res ^= (res << T_TEMPERING_R64) & C_MASK_R64;
    res ^= (res >> L_TEMPERING_R64);
    return res;
}


#if defined(INCLUDE_AVX512F)
__m512i get_8_MT19937_64(struct MT19937_64 * rand_engine){
    __m512i res;
    if(rand_engine->next >= MT19937_64_STATE_SIZE + 8){
        twist_MT19937_64(rand_engine);
        res = _mm512_load_si512(rand_engine->state + rand_engine->next);
    }else if(rand_engine->next % 8 == 0){
        res = _mm512_load_si512(rand_engine->state + rand_engine->next);
    }else{
        res = _mm512_loadu_si512(rand_engine->state + rand_engine->next);
    }
    rand_engine->next += 8;
    res = _mm512_xor_si512(res, _mm512_srli_epi64(res, U_TEMPERING_R64));
    res = _mm512_xor_si512(res, _mm512_and_si512(_mm512_slli_epi64(res, S_TEMPERING_R64), _mm512_set1_epi64(B_MASK_R64)));
    res = _mm512_xor_si512(res, _mm512_and_si512(_mm512_slli_epi64(res, T_TEMPERING_R64), _mm512_set1_epi64(C_MASK_R64)));
    res = _mm512_xor_si512(res, _mm512_srli_epi64(res, L_TEMPERING_R64));
    return res;
}
#endif

#if defined(INCLUDE_AVX2) || defined(INCLUDE_AVX512F)
__m256i get_4_MT19937_64(struct MT19937_64 * rand_engine){
    __m256i res;
    if(rand_engine->next >= MT19937_64_STATE_SIZE + 4){
        twist_MT19937_64(rand_engine);
        res = _mm256_load_si256((const __m256i*)(rand_engine->state + rand_engine->next));
    }else if(rand_engine->next % 4 == 0){
        res = _mm256_load_si256((const __m256i*)(rand_engine->state + rand_engine->next));
    }else{
        res = _mm256_loadu_si256((const __m256i*)(rand_engine->state + rand_engine->next));
    }
    rand_engine->next += 4;
    res = _mm256_xor_si256(res, _mm256_srli_epi64(res, U_TEMPERING_R64));
    res = _mm256_xor_si256(res, _mm256_and_si256(_mm256_slli_epi64(res, S_TEMPERING_R64), _mm256_set1_epi64x(B_MASK_R64)));
    res = _mm256_xor_si256(res, _mm256_and_si256(_mm256_slli_epi64(res, T_TEMPERING_R64), _mm256_set1_epi64x(C_MASK_R64)));
    res = _mm256_xor_si256(res, _mm256_srli_epi64(res, L_TEMPERING_R64));
    return res;
}
#endif

#if defined(INCLUDE_SSE2) || defined(INCLUDE_AVX2) || defined(INCLUDE_AVX512F)
__m128i get_2_MT19937_64(struct MT19937_64 * rand_engine){
    __m128i res;
    if(rand_engine->next >= MT19937_64_STATE_SIZE + 2){
        twist_MT19937_64(rand_engine);
        res = _mm_load_si128((__m128i*) (rand_engine->state + rand_engine->next));
    }else if(rand_engine->next % 2 == 0){
        res = _mm_load_si128((__m128i*) (rand_engine->state + rand_engine->next));
    }else{
        res = _mm_loadu_si128((__m128i*) (rand_engine->state + rand_engine->next));
    }
    rand_engine->next += 2;
    res = _mm_xor_si128(res, _mm_srli_epi64(res, U_TEMPERING_R64));
    res = _mm_xor_si128(res, _mm_and_si128(_mm_slli_epi64(res, S_TEMPERING_R64), _mm_set1_epi64x(B_MASK_R64)));
    res = _mm_xor_si128(res, _mm_and_si128(_mm_slli_epi64(res, T_TEMPERING_R64), _mm_set1_epi64x(C_MASK_R64)));
    res = _mm_xor_si128(res, _mm_srli_epi64(res, L_TEMPERING_R64));
    return res;
}
#elif defined(INCLUDE_NEON)
uint64x2_t get_2_MT19937_64(struct MT19937_64 * rand_engine){
    if(rand_engine->next >= MT19937_64_STATE_SIZE + 2){
        twist_MT19937_64(rand_engine);
    }
    uint64x2_t res = vld1q_u64(rand_engine->state + rand_engine->next);
    rand_engine->next += 2;
    res = veorq_u64(res, vshrq_n_u64(res, U_TEMPERING_R64));
    res = veorq_u64(res, vandq_u64(vshlq_n_u64(res, S_TEMPERING_R64), vdupq_n_u64(B_MASK_R64)));
    res = veorq_u64(res, vandq_u64(vshlq_n_u64(res, T_TEMPERING_R64), vdupq_n_u64(C_MASK_R64)));
    res = veorq_u64(res, vshrq_n_u64(res, L_TEMPERING_R64));
    return res;
}
#endif

