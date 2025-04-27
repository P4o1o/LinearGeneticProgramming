#ifndef FITNESS_H_INCLUDED
#define FITNESS_H_INCLUDED

#include "genetics.h"

typedef double (*fitness_fn)(const struct LGPInput *const, const struct Program *const, const uint64_t);

double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock);

enum FitnessType{
	MINIMIZE = 0,
	MAXIMIZE = 1,


	FITNESS_TYPE_NUM
};

struct FitnessAssesment{
	const fitness_fn fn;
	const enum FitnessType type;
};

extern const struct FitnessAssesment MSE;

#endif