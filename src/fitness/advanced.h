#ifndef FITNESS_ADVANCED_H_INCLUDED
#define FITNESS_ADVANCED_H_INCLUDED

#include "interface.h"

// ADVANCED FITNESS FUNCTION PROTOTYPES
double adversarial_perturbation_sensitivity(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double conditional_value_at_risk(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);

// ADVANCED INIT_ACC FUNCTION PROTOTYPES
union FitnessStepResult vect_f64_init_acc(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);

// ADVANCED FINALIZE FUNCTION PROTOTYPES
double value_at_risk_finalize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);

// ADVANCED FITNESS STRUCT EXPORTS
extern const struct Fitness ADVERSARIAL_PERTURBATION_SENSITIVITY;
extern const struct Fitness CONDITIONAL_VALUE_AT_RISK;

#endif
