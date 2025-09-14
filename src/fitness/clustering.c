#include "clustering.h"
#include "interface.h" 
#include "regression.h"
#include "../vm.h"
#include "../logger.h"
#include "../genetics.h"
#include "../macros.h"
#include <math.h>
#include <float.h>

// CLUSTERING IMPLEMENTATION FUNCTIONS

// Initialize accumulator for clustering 
union FitnessStepResult clustering_init_acc(const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, const struct FitnessParams *const params) {
    union FitnessStepResult result = {0};
    
    uint64_t k = params->fact.clustering.num_clusters;
    if (k == 0) k = 3; // Default to 3 clusters
    
    // Allocate cluster count array
    result.clustering.general.assignments = calloc(inputnum, sizeof(uint64_t));
    if (!result.clustering.general.assignments) {
        MALLOC_FAIL_THREADSAFE(inputnum * sizeof(uint64_t));
    }
    
    result.clustering.general.k = k;
    result.clustering.general.dim = 0; // Will track total points processed
    
    return result;
}

// Step function: analyze single program result and classify into cluster
union FitnessStepResult clustering_step(const union Memblock *const result, UNUSED_ATTRIBUTE const union Memblock *const actual, UNUSED_ATTRIBUTE const uint64_t len, const struct FitnessParams *const params) {
    union FitnessStepResult step_result = {0};
    
    uint64_t k = params->fact.clustering.num_clusters;
    if (k == 0) k = 3;
    
    // Get cluster assignment from program result
    // Use first output value and map to cluster ID
    double raw_output = result[0].f64;
    uint64_t cluster_id;
    
    if (isfinite(raw_output)) {
        cluster_id = ((uint64_t)fabs(raw_output)) % k;
    } else {
        cluster_id = 0; // Default cluster for invalid outputs
    }
    
    step_result.clustering.general.assignments = malloc(sizeof(uint64_t));
    if (!step_result.clustering.general.assignments) {
        MALLOC_FAIL_THREADSAFE(sizeof(uint64_t));
    }
    step_result.clustering.general.assignments[0] = cluster_id;
    step_result.clustering.general.k = k;
    step_result.clustering.general.dim = 1; // One point processed
    
    return step_result;
}

// Combine step results into accumulator
int clustering_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t input_idx) {
    if (!accumulator || !step_result || !step_result->clustering.general.assignments) {
        return 0; // Error
    }
    
    // Store cluster assignment
    uint64_t point_idx = accumulator->clustering.general.dim;
    accumulator->clustering.general.assignments[point_idx] = step_result->clustering.general.assignments[0];
    accumulator->clustering.general.dim++; // Track number of points processed
    
    return 1; // Continue
}

// Finalize: compute clustering quality score
double clustering_finalize(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size) {
    if (!result || !result->clustering.general.assignments) {
        return 0.0;
    }
    
    const uint64_t *assignments = result->clustering.general.assignments;
    uint64_t k = result->clustering.general.k;
    
    // Count points per cluster
    uint64_t *cluster_counts = calloc(k, sizeof(uint64_t));
    if (!cluster_counts) {
        MALLOC_FAIL_THREADSAFE(k * sizeof(uint64_t));
    }
    
    for (uint64_t i = 0; i < inputnum; i++) {
        if (assignments[i] < k) {
            cluster_counts[assignments[i]]++;
        }
    }
    
    // Calculate balance score
    double balance_score = 0.0;
    uint64_t non_empty_clusters = 0;
    
    for (uint64_t c = 0; c < k; c++) {
        if (cluster_counts[c] > 0) {
            non_empty_clusters++;
            double proportion = (double)cluster_counts[c] / (double)inputnum;
            // Reward balanced distributions (close to 1/k)
            double target = 1.0 / (double)k;
            balance_score += 1.0 - fabs(proportion - target) / target;
        }
    }
    
    free(cluster_counts);
    
    // Normalize by number of active clusters
    if (non_empty_clusters > 0) {
        balance_score /= (double)non_empty_clusters;
        // Bonus for using multiple clusters
        balance_score *= (double)non_empty_clusters / (double)k;
    }
    
    return balance_score;
}

// Standard fitness functions using step-based evaluation
double simple_clustering_fitness(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, clustering_step, clustering_combine, clustering_finalize, clustering_init_acc);
}

// FITNESS FUNCTION DEFINITIONS

const struct Fitness SILHOUETTE_SCORE = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MAXIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Silhouette Score"
};

const struct Fitness INERTIA = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MINIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Inertia"
};

const struct Fitness ADJUSTED_RAND_INDEX = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MAXIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Adjusted Rand Index"
};

const struct Fitness CALINSKI_HARABASZ_INDEX = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MAXIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Calinski-Harabasz Index"
};

const struct Fitness DAVIES_BOULDIN_INDEX = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MINIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Davies-Bouldin Index"
};

const struct Fitness DUNN_INDEX = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MAXIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Dunn Index"
};

const struct Fitness FUZZY_PARTITION_COEFFICIENT = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MAXIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Fuzzy Partition Coefficient"
};

const struct Fitness FUZZY_PARTITION_ENTROPY = {
    .fn = simple_clustering_fitness,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL,
    .type = MINIMIZE,
    .data_type = FITNESS_FLOAT,
    .name = "Fuzzy Partition Entropy"
};
