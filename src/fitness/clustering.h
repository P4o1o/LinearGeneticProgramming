#ifndef FITNESS_CLUSTERING_H_INCLUDED
#define FITNESS_CLUSTERING_H_INCLUDED

#include "interface.h"

// CLUSTERING PIPELINE FUNCTIONS
union FitnessStepResult clustering_init_acc(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult clustering_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
int clustering_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const uint64_t clocks, const uint64_t input_num, const struct FitnessParams *const params);
int k_clustering_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const uint64_t clocks, const uint64_t input_num, const struct FitnessParams *const params);
double clustering_finalize(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);

// CLUSTERING FINALIZE FUNCTION PROTOTYPES
double silhouette_finalize(const union FitnessStepResult *const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);

double silhouette_score(const struct LGPInput *const input, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);

// EXTERN FITNESS STRUCT DECLARATIONS FOR CLUSTERING
extern const struct Fitness SILHOUETTE_SCORE;

#endif // FITNESS_CLUSTERING_H_INCLUDED
