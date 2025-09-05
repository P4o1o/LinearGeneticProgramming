#ifndef CREATION_H_INCLUDED
#define CREATION_H_INCLUDED

#include "genetics.h"
#include "fitness.h"

struct InitializationParams{
	const uint64_t pop_size; // size of the initial Population
	const uint64_t minsize; // minimum size of a program generated in the initialization_func
	const uint64_t maxsize; // maximum size of a program generated in the initialization_func
};

typedef struct LGPResult (*initialization_fn)(const struct LGPInput *const, const struct InitializationParams *const, const struct Fitness *const, const uint64_t, const struct FitnessParams *const);

struct LGPResult unique_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct Fitness *const fitness, const uint64_t max_clock, const struct FitnessParams *const fitness_param);
struct LGPResult rand_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct Fitness *const fitness, const uint64_t max_clock, const struct FitnessParams *const fitness_param);


typedef struct LGPMultiResult (*multi_initialization_fn)(const struct LGPInput *const, const struct InitializationParams *const, const struct MultiFitness *const, const uint64_t);

struct LGPMultiResult rand_multipopulation(const struct LGPInput *const in, const struct InitializationParams *const params, const struct MultiFitness *const multifitness, const uint64_t max_clock);


// used in unique_population

struct ProgramSetNode{
	struct Program prog;
	uint64_t hash;
};

struct ProgramSet{
	struct ProgramSetNode *restrict table;
	uint64_t size;
	const uint64_t capacity;
};

#endif
