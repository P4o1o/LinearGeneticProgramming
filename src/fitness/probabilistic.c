#include "probabilistic.h"
#include "regression.h"  // For shared functions
#include "../macros.h"

// PROBABILISTIC STEP FUNCTION IMPLEMENTATIONS

inline union FitnessStepResult cross_entropy_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params){
    double cross_entropy = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double y_pred = result[i].f64;
        double y_true = actual[i].f64;
        
        // Clamp predictions to avoid log(0)
        y_pred = fmax(params->fact.tolerance, fmin(1.0 - params->fact.tolerance, y_pred));
        
        cross_entropy += -(y_true * log(y_pred) + (1.0 - y_true) * log(1.0 - y_pred));
    }
    return (union FitnessStepResult){.total_f64 = cross_entropy};
}

inline union FitnessStepResult gaussian_likelihood_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params){
    double log_likelihood = 0.0;
    double sigma = params->fact.sigma;
    double sigma_sq = sigma * sigma;
    
    for(uint64_t i = 0; i < len; i++){
        double diff = result[i].f64 - actual[i].f64;
        log_likelihood += -0.5 * (diff * diff) / sigma_sq - 0.5 * log(2.0 * M_PI * sigma_sq);
    }
    return (union FitnessStepResult){.total_f64 = log_likelihood};
}

inline union FitnessStepResult brier_score_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    double brier = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double y_pred = result[i].f64;
        double y_true = actual[i].f64;
        double diff = y_pred - y_true;
        brier += diff * diff;
    }
    return (union FitnessStepResult){.total_f64 = brier};
}

inline union FitnessStepResult hinge_loss_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    double hinge = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double y_pred = result[i].f64;
        double y_true = actual[i].f64; // Should be -1 or +1
        double margin = y_true * y_pred;
        hinge += fmax(0.0, 1.0 - margin);
    }
    return (union FitnessStepResult){.total_f64 = hinge};
}

// PROBABILISTIC FINALIZE FUNCTION IMPLEMENTATIONS

inline double negative_mean_input_and_dim(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return isfinite(result->total_f64) ? -(result->total_f64 / (double)(inputnum * ressize)) : -DBL_MAX;
}

// PROBABILISTIC FITNESS FUNCTION IMPLEMENTATIONS

double binary_cross_entropy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, cross_entropy_step, sum_float, mean_input_and_dim, init_acc_f64);
}

double gaussian_log_likelihood(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, gaussian_likelihood_step, sum_float, negative_mean_input_and_dim, init_acc_f64);
}

double brier_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, brier_score_step, sum_float, mean_input_and_dim, init_acc_f64);
}

double hinge_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, hinge_loss_step, sum_float, mean_input_and_dim, init_acc_f64);
}

// PROBABILISTIC FITNESS STRUCT DEFINITIONS

const struct Fitness BINARY_CROSS_ENTROPY = {
    .fn = binary_cross_entropy,
    .type = MINIMIZE,
    .name = "Binary Cross Entropy",
    .data_type = FITNESS_PROB,
    .step = cross_entropy_step,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};

const struct Fitness GAUSSIAN_LOG_LIKELIHOOD = {
    .fn = gaussian_log_likelihood,
    .type = MAXIMIZE,
    .name = "Gaussian Log Likelihood",
    .data_type = FITNESS_FLOAT,
    .step = gaussian_likelihood_step,
    .combine = sum_float,
    .finalize = negative_mean_input_and_dim,
    .init_acc = init_acc_f64
};

const struct Fitness BRIER_SCORE = {
    .fn = brier_score,
    .type = MINIMIZE,
    .name = "Brier Score",
    .data_type = FITNESS_PROB,
    .step = brier_score_step,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};

const struct Fitness HINGE_LOSS = {
    .fn = hinge_loss,
    .type = MINIMIZE,
    .name = "Hinge Loss",
    .data_type = FITNESS_FLOAT,
    .step = hinge_loss_step,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};
