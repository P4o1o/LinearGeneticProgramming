#include "creation.h"

static inline struct Program rand_program(const struct LGPInput *const in, const uint64_t minsize, const uint64_t maxsize) {
    ASSERT(minsize > 0);
    ASSERT(minsize <= maxsize);
    ASSERT(in->rom_size > 0);
    uint64_t size = RAND_BOUNDS(minsize, maxsize);
	struct Program res = { .size = size};
    size++;
    #if VECT_ALIGNMENT != 0
        uint64_t align = VECT_ALIGNMENT / 8;
        size = (size + align - 1) & ~(align - 1);
        ASSERT(size % align == 0);
    #endif
    ASSERT(size > res.size);
    res.content = aligned_alloc(VECT_ALIGNMENT, sizeof(struct Instruction) * size);
    if (res.content == NULL) {
        MALLOC_FAIL;
    }
    ASSERT(minsize <= res.size);
    ASSERT(res.size <= maxsize);
	for (uint64_t i = 0; i < res.size; i++) {
        res.content[i] = rand_instruction(in, res.size);
	}
    for(uint64_t i = res.size; i < size; i++){
        res.content[i] = (struct Instruction) {.op = I_EXIT, .reg = {0, 0, 0}, .addr = 0};
    }
	return res;
}

struct LGPResult rand_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct Fitness *const fitness, const uint64_t max_clock, const struct FitnessParams *const fitness_param) {
    ASSERT(params->pop_size > 0);
    ASSERT(0 < params->minsize);
    ASSERT(params->minsize <= params->maxsize);
	struct Population pop = {.size = params->pop_size};
	pop.individual = (struct Individual *) malloc(sizeof(struct Individual) * pop.size);
	if (pop.individual == NULL) {
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_OMP_THREADS)
	for (uint64_t i = 0; i < pop.size; i++) {
        struct Program prog = rand_program(in, params->minsize, params->maxsize);
        ASSERT(params->minsize <= prog.size);
        ASSERT(prog.size <= params->maxsize);
		pop.individual[i] = (struct Individual){ .prog = prog, .fitness = fitness->fn(in, &prog, max_clock, fitness_param)};
	}
    struct LGPResult res = {.generations = 0, .pop = pop, .evaluations = pop.size};
	return res;
}

static inline uint64_t next_power_of_two(uint64_t x) {
    ASSERT(x > 0);
    for (uint64_t shift = 1; shift < sizeof(uint64_t)*8; shift <<= 1)
        x |= (x >> shift);
    return x + 1;
}

struct LGPResult unique_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct Fitness *const fitness, const uint64_t max_clock, const struct FitnessParams *const fitness_param){
    ASSERT(params->pop_size > 0);
    ASSERT(0 < params->minsize);
    ASSERT(params->minsize <= params->maxsize);
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
#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_OMP_THREADS)
    for (uint64_t i = 0; i < pop.size; i++) {
        struct Program prog;
        uint64_t not_found;
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
                    not_found = 0;
                    break;
                }else if(progmap.table[index].hash == hash && equal_program(&(progmap.table[index].prog), &prog)){
                    not_found = 1;
                    free(prog.content);
                    break;
                }
                index = (index + 1) & mask;
            }
        }while(not_found);
        pop.individual[i] = (struct Individual){ .prog = prog, .fitness = fitness->fn(in, &prog, max_clock, fitness_param)};
    }
    free(progmap.table);
    struct LGPResult res = {.generations = 0, .pop = pop, .evaluations = pop.size};
    return res;
}

struct LGPMultiResult rand_multipopulation(const struct LGPInput *const in, const struct InitializationParams *const params, const struct MultiFitness *const multifitness, const uint64_t max_clock) {
    ASSERT(params->pop_size > 0);
    ASSERT(0 < params->minsize);
    ASSERT(params->minsize <= params->maxsize);
	struct MultiPopulation pop = {.size = params->pop_size};
	pop.individual = (struct MultiIndividual *) malloc(sizeof(struct MultiIndividual) * pop.size);
	if (pop.individual == NULL) {
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_OMP_THREADS)
	for (uint64_t i = 0; i < pop.size; i++) {
        struct Program prog = rand_program(in, params->minsize, params->maxsize);
        ASSERT(params->minsize <= prog.size);
        ASSERT(prog.size <= params->maxsize);
		pop.individual[i] = (struct MultiIndividual){ .prog = prog, .fitness = eval_multifitness(in, &pop.individual[i].prog, max_clock, multifitness)};
	}
    struct LGPMultiResult res = {.generations = 0, .pop = pop, .evaluations = pop.size};
	return res;
}
