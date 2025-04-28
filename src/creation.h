#ifndef CREATION_H_INCLUDED
#define CREATION_H_INCLUDED

#include "genetics.h"
#include "fitness.h"

struct InitializationParams{
	const uint64_t pop_size; // size of the initial Population
	const uint64_t minsize; // minimum size of a program generated in the initialization_func
	const uint64_t maxsize; // maximum size of a program generated in the initialization_func
};

typedef struct LGPResult (*initialization_fn)(const struct LGPInput *const, const struct InitializationParams *const, const struct FitnessAssesment *const, const uint64_t);

struct LGPResult unique_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct FitnessAssesment *const fitness, const uint64_t max_clock);
struct LGPResult rand_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct FitnessAssesment *const fitness, const uint64_t max_clock);

// used in unique_population

#if defined(__AVX512F__) | defined(__AVX512DQ__)
	#include <immintrin.h>
#endif

union InstrToU64{
	const struct Instruction instr;
	const uint64_t u64;
};

struct ProgramSetNode{
	struct Program prog;
	uint64_t hash;
};

struct ProgramSet{
	struct ProgramSetNode *table;
	uint64_t size;
	const uint64_t capacity;
};

#endif