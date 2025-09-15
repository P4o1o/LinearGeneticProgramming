#ifndef FITNESS_CLASSIFICATION_H_INCLUDED
#define FITNESS_CLASSIFICATION_H_INCLUDED

#include "interface.h"

// CLASSIFICATION FITNESS FUNCTION PROTOTYPES
double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double strict_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double strict_binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double strict_threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double balanced_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double g_mean(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double f1_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double f_beta_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double matthews_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double cohens_kappa(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);

// CLASSIFICATION STEP FUNCTION PROTOTYPES
union FitnessStepResult exact_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult binary_sign_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult threshold_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult binary_classification_confusion(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);

// CLASSIFICATION INIT_ACC FUNCTION PROTOTYPES
union FitnessStepResult init_acc_i64(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult init_acc_confusion(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);

// CLASSIFICATION COMBINE FUNCTION PROTOTYPES
int sum_uint64(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const uint64_t clocks, const uint64_t input_num, const struct FitnessParams *const params);
int sum_confusion(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const uint64_t clocks, const uint64_t input_num, const struct FitnessParams *const params);
int strict_sample_match(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const uint64_t clocks, const uint64_t input_num, const struct FitnessParams *const params);

// CLASSIFICATION FINALIZE FUNCTION PROTOTYPES
double rate_per_input(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double rate_per_sample(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double confusion_accuracy(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double confusion_f1_score(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double confusion_f_beta_score(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double confusion_balanced_accuracy(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double confusion_g_mean(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double confusion_matthews_correlation(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);
double confusion_cohens_kappa(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);

// CLASSIFICATION FITNESS STRUCT EXPORTS
extern const struct Fitness ACCURACY;
extern const struct Fitness STRICT_ACCURACY;
extern const struct Fitness BINARY_ACCURACY;
extern const struct Fitness STRICT_BINARY_ACCURACY;
extern const struct Fitness THRESHOLD_ACCURACY;
extern const struct Fitness STRICT_THRESHOLD_ACCURACY;
extern const struct Fitness BALANCED_ACCURACY;
extern const struct Fitness G_MEAN;
extern const struct Fitness F1_SCORE;
extern const struct Fitness F_BETA_SCORE;
extern const struct Fitness MATTHEWS_CORRELATION;
extern const struct Fitness COHENS_KAPPA;

#endif
