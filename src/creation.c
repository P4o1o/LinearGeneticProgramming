#include "creation.h"

static inline uint64_t instr_to_u64(const struct Instruction inst){
    const union InstrToU64 res = { .instr = inst};
    return res.u64;
}

static inline unsigned int equal_program(const struct Program *const prog1, const struct Program *const prog2){
	if(prog1->size != prog2->size){
		return 0;
	}
	for(uint64_t i = 0; i < prog1->size; i++){
		if(instr_to_u64(prog1->content[i]) != instr_to_u64(prog2->content[i])){
            return 0;
        }
	}
	return 1;
}

#define HASH_SEED 0ULL

#define PRIME1 0x9E3779B185EBCA87ULL
#define PRIME2 0xC2B2AE3D27D4EB4FULL

static inline uint64_t roll_left(const uint64_t num, const uint64_t counter){
    return ((num << counter) | (num >> (64 - counter)));
}

static inline uint64_t xxh_roll(const uint64_t previous, const uint64_t input){
    return roll_left(previous + input * PRIME2, 31) * PRIME1;
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    static inline __m512i avx512_xxh_roll(const __m512i previous, const __m512i input){
        const __m512i prime2 = _mm512_set1_epi64(PRIME2);
        const __m512i prime1 = _mm512_set1_epi64(PRIME1);
        const __m512i multiplied = _mm512_mullo_epi64(input, prime2);
        const __m512i added = _mm512_add_epi64(previous, multiplied);
        const __m512i rolled = _mm512_rol_epi64(added, 31);
        return _mm512_mullo_epi64(rolled, prime1);
    }
#endif

static inline uint64_t xxhash_program(const struct Program *const prog){
    const uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    const uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    const uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;
    uint64_t * input = (uint64_t*) prog->content;
    const uint64_t *const end = input + prog->size;
    uint64_t hash;
    if (prog->size >= 4) {
        uint64_t v1 = HASH_SEED + PRIME1 + PRIME2;
        uint64_t v2 = HASH_SEED + PRIME2;
        uint64_t v3 = HASH_SEED;
        uint64_t v4 = HASH_SEED - PRIME1;
        #if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512i acc = _mm512_set_epi64(v4, v3, v2, v1, v4, v3, v2, v1);
            while(input + 8 <= end){
                __m512i data = _mm512_load_si512(input);
                acc = avx512_xxh_roll(acc, data);
                input += 8;
            }
            uint64_t tmp[8];
            _mm512_storeu_si512(tmp, acc);
            v1 = tmp[0];
            v2 = tmp[1];
            v3 = tmp[2];
            v4 = tmp[3];
            if (input + 4 <= end){
                v1 = xxh_roll(v1, *input);
                input += 1;
                v2 = xxh_roll(v2, *input);
                input += 1;
                v3 = xxh_roll(v3, *input);
                input += 1;
                v4 = xxh_roll(v4, *input);
                input += 1;
            }
        #else
            while(input + 4 <= end){
                v1 = xxh_roll(v1, *input);
                input += 1;
                v2 = xxh_roll(v2, *input);
                input += 1;
                v3 = xxh_roll(v3, *input);
                input += 1;
                v4 = xxh_roll(v4, *input);
                input += 1;
            }
        #endif
        hash = roll_left(v1, 1) + roll_left(v2, 7) + roll_left(v3, 12) + roll_left(v4, 18);
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


static inline struct Program rand_program(const struct LGPInput *const in, const uint64_t minsize, const uint64_t maxsize) {
    ASSERT(minsize > 0);
    ASSERT(minsize <= maxsize);
    ASSERT(maxsize <= MAX_PROGRAM_SIZE);
    ASSERT(in->rom_size > 0);
	struct Program res = { .size = RAND_BOUNDS(minsize, maxsize) };
    ASSERT(minsize <= res.size);
    ASSERT(res.size <= maxsize);
	for (uint64_t i = 0; i < res.size; i++) {
        res.content[i] = rand_instruction(in, res.size);
	}
    res.content[res.size + 1] = (struct Instruction) {.op = I_EXIT, .reg = {0, 0, 0}, .addr = 0};
	return res;
}

struct LGPResult rand_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct FitnessAssesment *const fitness, const uint64_t max_clock) {
    ASSERT(params->pop_size > 0);
    ASSERT(0 < params->minsize);
    ASSERT(params->minsize <= params->maxsize);
    ASSERT(params->maxsize <= MAX_PROGRAM_SIZE);
	struct Population pop = {.size = params->pop_size};
	pop.individual = (struct Individual *) malloc(sizeof(struct Individual) * pop.size);
	if (pop.individual == NULL) {
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic,1)
	for (uint64_t i = 0; i < pop.size; i++) {
        struct Program prog = rand_program(in, params->minsize, params->maxsize);
        ASSERT(params->minsize <= prog.size);
        ASSERT(prog.size <= params->maxsize);
		pop.individual[i] = (struct Individual){ .prog = prog, .fitness = fitness->fn(in, &prog, max_clock)};
	}
    struct LGPResult res = {.generations = 0, .pop = pop, .evaluations = pop.size};
	return res;
}

static inline uint64_t next_power_of_two(uint64_t x) {
    ASSERT(x > 0);
    for (uint64_t shift = 1; shift < sizeof(uint64_t)*8; shift <<= 1)
        x &&= (x >> shift);
    return x << 1;
}

struct LGPResult unique_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct FitnessAssesment *const fitness, const uint64_t max_clock){
    ASSERT(params->pop_size > 0);
    ASSERT(0 < params->minsize);
    ASSERT(params->minsize <= params->maxsize);
    ASSERT(params->maxsize <= MAX_PROGRAM_SIZE);
	struct Population pop = {.size = params->pop_size};
    pop.individual = (struct Individual *) malloc(sizeof(struct Individual) * pop.size);
	if (pop.individual == NULL) {
		MALLOC_FAIL;
	}
    struct ProgramSet progmap = {.capacity = next_power_of_two(params->pop_size), .size = 0};
    uint64_t mask = progmap.capacity - 1;
    progmap.table = (struct ProgramSetNode *) malloc(sizeof(struct ProgramSetNode) * progmap.capacity);
    if(progmap.table == NULL){
        MALLOC_FAIL;
    }
    memset(progmap.table, 0, sizeof(struct ProgramSetNode) * progmap.capacity);
#pragma omp parallel for schedule(dynamic,1)
    for (uint64_t i = 0; i < pop.size; i++) {
        struct Program prog;
        uint64_t found;
        do{
            prog = rand_program(in, params->minsize, params->maxsize);
            ASSERT(params->minsize <= prog.size);
            ASSERT(prog.size <= params->maxsize);
            uint64_t hash = xxhash_program(&prog);
            uint64_t index = hash & mask;
            uint64_t added;
            for(;;){
                added = 0;
#pragma omp critical
                {
                    if(progmap.table[index].prog.size == 0){
                        progmap.table[index].prog = prog;
                        progmap.table[index].hash = hash;
                        added = 1;
                    }
                }
                if(added){
                    found = 0;
                    break;
                }else if(progmap.table[index].hash == hash && equal_program(&(progmap.table[index].prog), &prog)){
                    found = 1;
                    break;
                }
                index = (index + 1) & mask;
            }
        }while(found);
        pop.individual[i] = (struct Individual){ .prog = prog, .fitness = fitness->fn(in, &prog, max_clock)};
    }
    free(progmap.table);
    struct LGPResult res = {.generations = 0, .pop = pop, .evaluations = pop.size};
    return res;
}