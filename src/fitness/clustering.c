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
union FitnessStepResult clustering_init_acc(const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params) {
    union FitnessStepResult result;
    result.clustering.assignments = calloc(inputnum, sizeof(uint64_t));
    if (!result.clustering.assignments) {
        MALLOC_FAIL_THREADSAFE(inputnum * sizeof(uint64_t));
    }
    return result;
}

// Step function: analyze single program result and classify into cluster
union FitnessStepResult clustering_step(const union Memblock *const result, UNUSED_ATTRIBUTE const union Memblock *const actual, UNUSED_ATTRIBUTE const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params) {
    union FitnessStepResult step_result;
    step_result.clustering.single_assignment = result[0].i64;
    return step_result;
}

// Combine step results into accumulator
int clustering_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const uint64_t clocks, const uint64_t input_idx, UNUSED_ATTRIBUTE const struct FitnessParams *const params) {
    accumulator->clustering.assignments[input_idx] = step_result->clustering.single_assignment;
    return 1; // Continue
}

int k_clustering_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const uint64_t clocks, const uint64_t input_idx, const struct FitnessParams *const params) {
    ASSERT(params->fact.clustering.num_clusters > 1);
    if(step_result->clustering.single_assignment >= params->fact.clustering.num_clusters) {
        free(accumulator->clustering.assignments);
        accumulator->clustering.assignments = NULL;
        return 0; // Stop
    }
    accumulator->clustering.assignments[input_idx] = step_result->clustering.single_assignment;
    return 1; // Continue
}

// Silhouette Score finalize - calcola qualità clustering via silhouette analysis
double silhouette_finalize(const union FitnessStepResult *const result, const struct LGPInput *const in, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size, UNUSED_ATTRIBUTE const uint64_t input_num, UNUSED_ATTRIBUTE const struct FitnessParams *const params) {
    if(result->clustering.assignments == NULL) {
        return -1.0;
    }
    const uint64_t *assignments = result->clustering.assignments;
    uint64_t inputnum = in->input_num;
    uint64_t max_cluster_id = 0;
    for (uint64_t i = 0; i < inputnum; i++) {
        if (assignments[i] > max_cluster_id) max_cluster_id = assignments[i];
    }
    uint64_t num_clusters = max_cluster_id + 1;
    
    if (num_clusters < 2) {
        free(result->clustering.assignments);
        return -1.0; // Serve almeno 2 cluster per silhouette
    }
    
    uint64_t *cluster_sizes = calloc(num_clusters, sizeof(uint64_t));
    if (!cluster_sizes)
        MALLOC_FAIL_THREADSAFE(num_clusters * sizeof(uint64_t));
    
    for (uint64_t i = 0; i < inputnum; i++) {
        cluster_sizes[assignments[i]]++;
    }
    
    uint64_t non_empty_clusters = 0;
    for (uint64_t c = 0; c < num_clusters; c++) {
        if (cluster_sizes[c] > 0) non_empty_clusters++;
    }
    
    if (non_empty_clusters < 2) {
        free(cluster_sizes);
        free(result->clustering.assignments);
        return -1.0;
    }
    
    double total_silhouette = 0.0;
    uint64_t valid_points = 0;

    double *distances = params->fact.clustering.distance_table;
    
    for (uint64_t i = 0; i < inputnum; i++) {
        uint64_t cluster_i = assignments[i];        
        
        // Se il cluster ha solo 1 punto, silhouette = 0
        if (cluster_sizes[cluster_i] <= 1) {
            valid_points++;
            continue;
        }
        
        double intra_cluster_dist = 0.0;
        uint64_t intra_count = 0;
        
        // Calcola distanza media intra-cluster (a_i) usando pre-computed distances
        for (uint64_t j = 0; j < inputnum; j++) {
            if (i != j && assignments[j] == cluster_i) {
                intra_cluster_dist += distances[i * inputnum + j];
                intra_count++;
            }
        }
        
        if (intra_count > 0) {
            intra_cluster_dist /= intra_count;
        }
        
        // Calcola distanza media al cluster più vicino (b_i)
        double min_inter_cluster_dist = DBL_MAX;
        
        for (uint64_t c = 0; c < num_clusters; c++) {
            if (c != cluster_i && cluster_sizes[c] > 0) {
                double inter_cluster_dist = 0.0;
                uint64_t inter_count = 0;
                
                for (uint64_t j = 0; j < inputnum; j++) {
                    if (assignments[j] == c) {
                        inter_cluster_dist += distances[i * inputnum + j];
                        inter_count++;
                    }
                }
                
                if (inter_count > 0) {
                    inter_cluster_dist /= inter_count;
                    if (inter_cluster_dist < min_inter_cluster_dist) {
                        min_inter_cluster_dist = inter_cluster_dist;
                    }
                }
            }
        }
        
        // Calcola silhouette per questo punto: s_i = (b_i - a_i) / max(a_i, b_i)
        if (min_inter_cluster_dist < DBL_MAX) {
            double max_dist = fmax(intra_cluster_dist, min_inter_cluster_dist);
            if (max_dist > 0.0) {
                double silhouette_i = (min_inter_cluster_dist - intra_cluster_dist) / max_dist;
                total_silhouette += silhouette_i;
            }
        }
        valid_points++;
    }
    
    free(cluster_sizes);
    free(result->clustering.assignments);

    // Ritorna silhouette medio
    if (valid_points > 0) {
        return total_silhouette / valid_points;
    } else {
        return -1.0;
    }
}

double silhouette_score(const struct LGPInput *const input, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(input, prog, max_clock, params, clustering_step, k_clustering_combine, silhouette_finalize, clustering_init_acc);
}

const struct Fitness SILHOUETTE_SCORE = {
    .name = "silhouette_score",
    .init_acc = clustering_init_acc,
    .step = clustering_step,
    .combine = k_clustering_combine,
    .finalize = silhouette_finalize,
    .type = MAXIMIZE,
    .data_type = FITNESS_INT,
    .fn = silhouette_score
};