#ifndef FITNESS_H_INCLUDED
#define FITNESS_H_INCLUDED

#include "genetics.h"

union FitnessParams{
	const double threshold; // used in threshold_accuracy
};

typedef double (*fitness_fn)(const struct LGPInput *const, const struct Program *const, const uint64_t, const union FitnessParams *const params);

double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);

enum FitnessType{
	MINIMIZE = 0,
	MAXIMIZE = 1,


	FITNESS_TYPE_NUM
};

struct FitnessAssesment{
	const fitness_fn fn;
	const enum FitnessType type;
	const char *name; // name of the fitness function, used for printing
};

extern const struct FitnessAssesment MSE;
extern const struct FitnessAssesment RMSE;
extern const struct FitnessAssesment MAE;
extern const struct FitnessAssesment RSQUARED;
extern const struct FitnessAssesment ACCURACY;
extern const struct FitnessAssesment THRESHOLD_ACCURACY;

#endif
