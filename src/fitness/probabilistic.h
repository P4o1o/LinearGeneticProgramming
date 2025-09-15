#ifndef FITNESS_PROBABILISTIC_H_INCLUDED
#define FITNESS_PROBABILISTIC_H_INCLUDED

#include "interface.h"

// PROBABILISTIC FITNESS FUNCTION PROTOTYPES
double binary_cross_entropy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double gaussian_log_likelihood(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double brier_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double hinge_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);

// PROBABILISTIC STEP FUNCTION PROTOTYPES
union FitnessStepResult cross_entropy_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult gaussian_likelihood_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult brier_score_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult hinge_loss_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);

// PROBABILISTIC INIT_ACC FUNCTION PROTOTYPES (from regression)
union FitnessStepResult init_acc_f64(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);

// PROBABILISTIC COMBINE FUNCTION PROTOTYPES (from regression)
// PROBABILISTIC COMBINE FUNCTION PROTOTYPES (from regression)
int sum_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const uint64_t clocks, const uint64_t input_num, const struct FitnessParams *const params);

// PROBABILISTIC FINALIZE FUNCTION PROTOTYPES
double negative_mean_input_and_dim(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double mean_input_and_dim(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);

// PROBABILISTIC FITNESS STRUCT EXPORTS
extern const struct Fitness BINARY_CROSS_ENTROPY;
extern const struct Fitness GAUSSIAN_LOG_LIKELIHOOD;
extern const struct Fitness BRIER_SCORE;
extern const struct Fitness HINGE_LOSS;

#endif
