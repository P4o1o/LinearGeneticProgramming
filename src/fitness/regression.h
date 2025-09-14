#ifndef FITNESS_REGRESSION_H_INCLUDED
#define FITNESS_REGRESSION_H_INCLUDED

#include "interface.h"

// REGRESSION FITNESS FUNCTION PROTOTYPES
double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double length_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double clock_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double symmetric_mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double logcosh(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double worst_case_error(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double huber_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double pinball_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double pearson_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);

// REGRESSION STEP FUNCTION PROTOTYPES
union FitnessStepResult quadratic_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult absolute_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult absolute_percent_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult symmetric_absolute_percent_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult logcosh_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult huber_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult pinball_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);

// REGRESSION INIT_ACC FUNCTION PROTOTYPES
union FitnessStepResult init_acc_f64(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult init_acc_r_2(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult pearson_init_acc(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);

// REGRESSION COMBINE FUNCTION PROTOTYPES
int sum_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int sum_float_clock_pen(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int max_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int r_squared_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int pearson_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);

// REGRESSION FINALIZE FUNCTION PROTOTYPES
double mean_input_and_dim(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double sqrt_mean_input_and_dim(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double percent_mean_input_and_dim(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double mean_input_and_dim_length_pen(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double max_over_ressize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double r_squared_finalize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double pearson_finalize(const union FitnessStepResult * const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);

// UTILITY FUNCTION PROTOTYPES
union FitnessStepResult return_info(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
int compare_doubles(const void *a, const void *b);

// REGRESSION FITNESS STRUCT EXPORTS
extern const struct Fitness MSE;
extern const struct Fitness RMSE;
extern const struct Fitness LENGTH_PENALIZED_MSE;
extern const struct Fitness CLOCK_PENALIZED_MSE;
extern const struct Fitness MAE;
extern const struct Fitness MAPE;
extern const struct Fitness SYMMETRIC_MAPE;
extern const struct Fitness LOGCOSH;
extern const struct Fitness WORST_CASE_ERROR;
extern const struct Fitness HUBER_LOSS;
extern const struct Fitness R_SQUARED;
extern const struct Fitness PINBALL_LOSS;
extern const struct Fitness PEARSON_CORRELATION;

#endif
