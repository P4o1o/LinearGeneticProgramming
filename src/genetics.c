#include "genetics.h"

const uint64_t VECT_ALIGNMENT_WRAPPER = VECT_ALIGNMENT;

struct Instruction rand_instruction(const struct LGPInput *const in, const uint64_t prog_size){
    ASSERT(prog_size > 0);
    ASSERT(in->rom_size > 0);
    const struct Operation op = in->instr_set.op[RAND_UPTO(in->instr_set.size - 1)];
    const uint8_t opcode = op.code;
    union Memblock tmp;
    uint32_t addr;
    switch(op.addr){
        case 1:
            addr = random();
        break;
        case 2:
            addr = RAND_UPTO(in->ram_size - 1);
        break;
        case 3:
            addr = RAND_UPTO(prog_size);
        break;
        case 4:
            addr = RAND_UPTO(in->rom_size - 1);
        break;
        case 5:
            tmp.f64 = RAND_DOUBLE();
            addr = tmp.i64;
        break;
        case 0:
            addr = 0;
        break;
        default:
            ASSERT(0);
        break;
    }
    uint8_t regs[3] = {0, 0, 0};
    for (uint64_t j = 0; j < op.regs; j++){
        regs[j] = RAND_UPTO(REG_NUM - 1);
    }
    const struct Instruction res = { .op = opcode, .reg = {regs[0], regs[1], regs[2]}, .addr = addr};
    return res;
}

unsigned int equal_program(const struct Program *const prog1, const struct Program *const prog2){
	if(prog1->size != prog2->size){
		return 0;
	}
    uint64_t* program1 = (uint64_t*) prog1->content;
    uint64_t* program2 = (uint64_t*) prog2->content;
	for(uint64_t i = 0; i < prog1->size; i++){
		if(program1[i] != program2[i]){
            return 0;
        }
	}
	return 1;
}

#define PRIME1 0x9E3779B185EBCA87ULL
#define PRIME2 0xC2B2AE3D27D4EB4FULL

static inline uint64_t roll_left(const uint64_t num, const uint64_t step){
    return ((num << step) | (num >> (64 - step)));
}

static inline uint64_t xxh_roll(const uint64_t previous, const uint64_t input){
    return roll_left(previous + input * PRIME2, 31) * PRIME1;
}

#if defined(INCLUDE_AVX512F)

    UNUSED_ATTRIBUTE static inline __m512i avx512_mul_epi64(const __m512i a, const __m512i b){
        const __m512i lo = _mm512_mul_epu32(a, b);
        const __m512i a_shuf = _mm512_shuffle_epi32(a, _MM_SHUFFLE(3,1,2,0));
        const __m512i b_shuf = _mm512_shuffle_epi32(b, _MM_SHUFFLE(3,1,2,0));
        const __m512i hi = _mm512_mul_epu32(a_shuf, b_shuf);
        const __m512i hi_shifted = _mm512_slli_epi64(hi, 32);
        return _mm512_mask_blend_epi32(0xAAAA, lo, hi_shifted);
    }

    static inline __m512i avx512_xxh_roll(const __m512i previous, const __m512i input){
        const __m512i prime2 = _mm512_set1_epi64(PRIME2);
        const __m512i prime1 = _mm512_set1_epi64(PRIME1);
        #if defined(INCLUDE_AVX512DQ)
            const __m512i multiplied = _mm512_mullo_epi64(input, prime2);
        #else
            const __m512i multiplied = avx512_mul_epi64(input, prime2);
        #endif
        const __m512i added = _mm512_add_epi64(previous, multiplied);
        const __m512i rolled = _mm512_rol_epi64(added, 31);
        #if defined(INCLUDE_AVX512DQ)
            return _mm512_mullo_epi64(rolled, prime1);
        #else
            return avx512_mul_epi64(rolled, prime1);
        #endif
        
    }
#endif

#if defined(INCLUDE_AVX2) || defined(INCLUDE_AVX512F)

    UNUSED_ATTRIBUTE static inline __m256i avx256_mul_epi64(const __m256i a, const __m256i b){
        const __m256i lo = _mm256_mul_epu32(a, b);
        const __m256i a_shuf = _mm256_shuffle_epi32(a, _MM_SHUFFLE(3,1,2,0));
        const __m256i b_shuf = _mm256_shuffle_epi32(b, _MM_SHUFFLE(3,1,2,0));
        const __m256i hi = _mm256_mul_epu32(a_shuf, b_shuf);
        const __m256i hi_shifted = _mm256_slli_epi64(hi, 32);
        return _mm256_blend_epi32(lo, hi_shifted, 0xAA);
    }

    static inline __m256i avx256_xxh_roll(const __m256i previous, const __m256i input){
        const __m256i prime2 = _mm256_set1_epi64x(PRIME2);
        const __m256i prime1 = _mm256_set1_epi64x(PRIME1);
        #if defined(INCLUDE_AVX512DQ) && defined(INCLUDE_AVX512VL)
            const __m256i multiplied = _mm256_mullo_epi64(input, prime2);
        #else
            const __m256i multiplied = avx256_mul_epi64(input, prime2);
        #endif
        const __m256i added = _mm256_add_epi64(previous, multiplied);
        const __m256i rolled = _mm256_or_si256(_mm256_slli_epi64(added, 31), _mm256_slli_epi64(added, 33));
        #if defined(INCLUDE_AVX512DQ) && defined(INCLUDE_AVX512VL)
            return _mm256_mullo_epi64(rolled, prime1);
        #else
            return avx256_mul_epi64(rolled, prime1);
        #endif
    }
#elif defined(INCLUDE_SSE2)

    static inline __m128i sse2_mul_epi64(const __m128i a, const __m128i b){
        const __m128i lo = _mm_mul_epu32(a, b);
        const __m128i a_shuf = _mm_shuffle_epi32(a, _MM_SHUFFLE(3,1,2,0));
        const __m128i b_shuf = _mm_shuffle_epi32(b, _MM_SHUFFLE(3,1,2,0));
        const  __m128i hi = _mm_mul_epu32(a_shuf, b_shuf);
        const __m128i hi_shifted = _mm_slli_epi64(hi, 32);
        #if defined(INCLUDE_SSE4_1)
            return _mm_blend_epi16(lo, hi_shifted, 0xA);
        #else
            return _mm_add_epi64(lo, hi_shifted);
        #endif
    }

    static inline __m128i sse2_xxh_roll(const __m128i previous, const __m128i input){
        const __m128i prime2 = _mm_set1_epi64x(PRIME2);
        const __m128i prime1 = _mm_set1_epi64x(PRIME1);
        const __m128i multiplied = sse2_mul_epi64(input, prime2);
        const __m128i added = _mm_add_epi64(previous, multiplied);
        const __m128i rolled = _mm_or_si128(_mm_slli_epi64(added, 31), _mm_slli_epi64(added, 33));
        return sse2_mul_epi64(rolled, prime1);
    }
#endif

uint64_t xxhash_program(const struct Program *const prog){
    const uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    const uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    const uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;
    uint64_t * input = (uint64_t*) prog->content;
    const uint64_t *const end = input + prog->size;
    uint64_t hash;
    if (prog->size >= 4) {
        #if defined(INCLUDE_AVX512F)
            __m512i acc = _mm512_set_epi64(
                HASH_SEED + PRIME1 + PRIME2, HASH_SEED + PRIME2, HASH_SEED, HASH_SEED - PRIME1,
                HASH_SEED + PRIME1 + PRIME2, HASH_SEED + PRIME2, HASH_SEED, HASH_SEED - PRIME1
            );
            while(input + 8 <= end){
                __m512i data = _mm512_load_si512(input);
                acc = avx512_xxh_roll(acc, data);
                input += 8;
            }
            alignas(64) uint64_t counter[8];
            _mm512_storeu_si512(counter, acc);
            if (input + 4 <= end){
                __m256i data = _mm256_load_si256((const __m256i *)input);
                __m256i small_acc = _mm512_castsi512_si256(acc);
                small_acc = avx256_xxh_roll(small_acc, data);
                _mm256_store_si256((__m256i *) counter, small_acc);
                input += 4;
            }
        #elif defined(INCLUDE_AVX2)
            __m256i acc = _mm256_set_epi64x(HASH_SEED + PRIME1 + PRIME2, HASH_SEED + PRIME2, HASH_SEED, HASH_SEED - PRIME1);
            while(input + 4 <= end){
                __m256i data = _mm256_load_si256((const __m256i *)input);
                acc = avx256_xxh_roll(acc, data);
                input += 4;
            }
            alignas(32) uint64_t counter[4];
            _mm256_store_si256((__m256i *)counter, acc);
        #elif defined(INCLUDE_SSE2)
            __m128i small_acc1 = _mm_set_epi64x(HASH_SEED + PRIME1 + PRIME2, HASH_SEED + PRIME2);
            __m128i small_acc2 = _mm_set_epi64x(HASH_SEED, HASH_SEED - PRIME1);
            while(input + 4 <= end){
                __m128i data = _mm_load_si128((__m128i*)(input));
                small_acc1 = sse2_xxh_roll(small_acc1, data);
                input += 2;
                small_acc2 = sse2_xxh_roll(small_acc2, data);
                input += 2;
            }
            alignas(16) uint64_t counter[4];
            _mm_store_si128((__m128i *)(counter), small_acc1);
            _mm_store_si128((__m128i *)(counter + 2), small_acc2);
        #else
            uint64_t counter[4];
            counter[0] = HASH_SEED + PRIME1 + PRIME2;
            counter[1] = HASH_SEED + PRIME2;
            counter[2] = HASH_SEED;
            counter[3] = HASH_SEED - PRIME1;
            while(input + 4 <= end){
                counter[0] = xxh_roll(counter[0], *input);
                input += 1;
                counter[1] = xxh_roll(counter[1], *input);
                input += 1;
                counter[2] = xxh_roll(counter[2], *input);
                input += 1;
                counter[3] = xxh_roll(counter[3], *input);
                input += 1;
            }
        #endif
        hash = roll_left(counter[0], 1) + roll_left(counter[1], 7) + roll_left(counter[2], 12) + roll_left(counter[3], 18);
    }else{
        hash = PRIME5 + HASH_SEED;
    }
    hash += (prog->size * 8);

    for(; input < end; input += 1){
        hash = roll_left(hash ^ xxh_roll(0, *input), 27) * PRIME1 + PRIME4;
    }

    hash ^= hash >> 33;
    hash *= PRIME2;
    hash ^= hash >> 29;
    hash *= PRIME3;
    hash ^= hash >> 32;
    return hash;
}

inline void free_individual(struct Individual *ind){
    aligned_free(ind->prog.content);
}
void free_population(struct Population *pop){
    for (uint64_t i = 0; i < pop->size; i++){
        free_individual(&(pop->individual[i]));
    }
    aligned_free(pop->individual);
}
void free_lgp_input(struct LGPInput *in){
    free(in->memory);
}
