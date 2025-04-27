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


static inline uint64_t hash_program(const struct Program *const prog){
    uint64_t hash = prog->size;
    for(uint64_t i = 0; i < prog->size; i++){
		hash += instr_to_u64(prog->content[i]);
	}
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
