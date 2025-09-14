#ifndef FITNESS_CLUSTERING_H_INCLUDED
#define FITNESS_CLUSTERING_H_INCLUDED

#include "interface.h"

// CLUSTERING PIPELINE FUNCTIONS
union FitnessStepResult clustering_init_acc(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult clustering_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
int clustering_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t input_idx);
double clustering_finalize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);

// Standard clustering fitness function
double simple_clustering_fitness(const struct LGPInput *const input, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);

// EXTERN FITNESS STRUCT DECLARATIONS FOR CLUSTERING
extern const struct Fitness SILHOUETTE_SCORE;
extern const struct Fitness CALINSKI_HARABASZ_INDEX;
extern const struct Fitness DAVIES_BOULDIN_INDEX;
extern const struct Fitness DUNN_INDEX;
extern const struct Fitness INERTIA;
extern const struct Fitness ADJUSTED_RAND_INDEX;
extern const struct Fitness FUZZY_PARTITION_COEFFICIENT;
extern const struct Fitness FUZZY_PARTITION_ENTROPY;

#endif // FITNESS_CLUSTERING_H_INCLUDED
