#include "classification.h"
#include "../macros.h"

// CLASSIFICATION INIT_ACC IMPLEMENTATIONS

union FitnessStepResult init_acc_i64(UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    return (union FitnessStepResult){.total_u64 = 0};
}

union FitnessStepResult init_acc_confusion(UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    return (union FitnessStepResult){.confusion = {.true_pos = 0, .false_pos = 0, .false_neg = 0, .true_neg = 0}};
}

// CLASSIFICATION STEP FUNCTION IMPLEMENTATIONS

inline union FitnessStepResult exact_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    uint64_t correct = 0;
    for(uint64_t i = 0; i < len; i++){
        if(result[i].i64 == actual[i].i64)
            correct++;
    }
    return (union FitnessStepResult){.total_u64 = correct};
}

inline union FitnessStepResult binary_sign_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    uint64_t correct = 0;
    for(uint64_t i = 0; i < len; i++){
        uint64_t res_sign = result[i].i64 & (((uint64_t)1) << 63);
        uint64_t act_sign = actual[i].i64 & (((uint64_t)1) << 63);
        if(res_sign == act_sign)
            correct++;
    }
    return (union FitnessStepResult){.total_u64 = correct};
}

inline union FitnessStepResult threshold_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params){
    uint64_t correct = 0;
    for(uint64_t i = 0; i < len; i++){
        double diff = fabs(result[i].f64 - actual[i].f64);
        if(diff <= params->fact.threshold)
            correct++;
    }
    return (union FitnessStepResult){.total_u64 = correct};
}

inline union FitnessStepResult binary_classification_confusion(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    uint64_t tp = 0, fp = 0, fn = 0, tn = 0;
    for(uint64_t i = 0; i < len; i++){
        uint64_t res_positive = (result[i].i64 & (((uint64_t)1) << 63)) == 0; // 0 = positive, 1 = negative
        uint64_t act_positive = (actual[i].i64 & (((uint64_t)1) << 63)) == 0;
        
        if(act_positive && res_positive) tp++;
        else if(act_positive && !res_positive) fn++;
        else if(!act_positive && res_positive) fp++;
        else tn++;
    }
    return (union FitnessStepResult){.confusion = {.true_pos = tp, .false_pos = fp, .false_neg = fn, .true_neg = tn}};
}

// CLASSIFICATION COMBINE FUNCTION IMPLEMENTATIONS

inline int sum_uint64(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    accumulator->total_u64 += step_result->total_u64;
    return 1;
}

inline int sum_confusion(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    accumulator->confusion.true_pos += step_result->confusion.true_pos;
    accumulator->confusion.false_pos += step_result->confusion.false_pos;
    accumulator->confusion.false_neg += step_result->confusion.false_neg;
    accumulator->confusion.true_neg += step_result->confusion.true_neg;
    return 1;
}

inline int strict_sample_match(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    // For strict accuracy - count if ALL labels in this sample are correct
    uint64_t expected_matches = params->end - params->start;
    if(step_result->total_u64 == expected_matches)
        accumulator->total_u64++;
    return 1;
}

// CLASSIFICATION FINALIZE FUNCTION IMPLEMENTATIONS

inline double rate_per_input(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return (double)result->total_u64 / (double)(inputnum * ressize);
}

inline double rate_per_sample(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return (double)result->total_u64 / (double)inputnum;
}

inline double confusion_accuracy(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t total = result->confusion.true_pos + result->confusion.false_pos + result->confusion.false_neg + result->confusion.true_neg;
    if(total == 0) return 0.0;
    return (double)(result->confusion.true_pos + result->confusion.true_neg) / (double)total;
}

inline double confusion_f1_score(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t tp = result->confusion.true_pos;
    uint64_t fp = result->confusion.false_pos;
    uint64_t fn = result->confusion.false_neg;

    if(tp == 0) return 0.0;
    
    double precision = (double)tp / (double)(tp + fp);
    double recall = (double)tp / (double)(tp + fn);
    
    if(precision + recall == 0.0) return 0.0;
    return 2.0 * precision * recall / (precision + recall);
}

inline double confusion_f_beta_score(const union FitnessStepResult *const result, const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t tp = result->confusion.true_pos;
    uint64_t fp = result->confusion.false_pos;
    uint64_t fn = result->confusion.false_neg;

    if(tp == 0) return 0.0;
    
    double precision = (double)tp / (double)(tp + fp);
    double recall = (double)tp / (double)(tp + fn);
    double beta = params->fact.beta;
    double beta_sq = beta * beta;
    
    if(precision + recall == 0.0) return 0.0;
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall);
}

inline double confusion_balanced_accuracy(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t tp = result->confusion.true_pos;
    uint64_t fp = result->confusion.false_pos;
    uint64_t fn = result->confusion.false_neg;
    uint64_t tn = result->confusion.true_neg;

    double sensitivity = (tp + fn > 0) ? (double)tp / (double)(tp + fn) : 0.0;
    double specificity = (tn + fp > 0) ? (double)tn / (double)(tn + fp) : 0.0;
    
    return (sensitivity + specificity) / 2.0;
}

inline double confusion_g_mean(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t tp = result->confusion.true_pos;
    uint64_t fp = result->confusion.false_pos;
    uint64_t fn = result->confusion.false_neg;
    uint64_t tn = result->confusion.true_neg;

    double sensitivity = (tp + fn > 0) ? (double)tp / (double)(tp + fn) : 0.0;
    double specificity = (tn + fp > 0) ? (double)tn / (double)(tn + fp) : 0.0;
    
    return sqrt(sensitivity * specificity);
}

inline double confusion_matthews_correlation(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t tp = result->confusion.true_pos;
    uint64_t fp = result->confusion.false_pos;
    uint64_t fn = result->confusion.false_neg;
    uint64_t tn = result->confusion.true_neg;

    double numerator = (double)(tp * tn) - (double)(fp * fn);
    double denominator = sqrt((double)(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    
    if(denominator == 0.0) return 0.0;
    return numerator / denominator;
}

inline double confusion_cohens_kappa(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t tp = result->confusion.true_pos;
    uint64_t fp = result->confusion.false_pos;
    uint64_t fn = result->confusion.false_neg;
    uint64_t tn = result->confusion.true_neg;
    uint64_t total = tp + fp + fn + tn;
    
    if(total == 0) return 0.0;
    
    double po = (double)(tp + tn) / (double)total;
    double pe = ((double)(tp + fp) * (tp + fn) + (double)(tn + fp) * (tn + fn)) / ((double)total * total);
    
    if(pe == 1.0) return 0.0;
    return (po - pe) / (1.0 - pe);
}

// CLASSIFICATION FITNESS FUNCTION IMPLEMENTATIONS

double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, exact_match, sum_uint64, rate_per_input, init_acc_i64);
}

double strict_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, exact_match, strict_sample_match, rate_per_sample, init_acc_i64);
}

double binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_sign_match, sum_uint64, rate_per_input, init_acc_i64);
}

double strict_binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_sign_match, strict_sample_match, rate_per_sample, init_acc_i64);
}

double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, threshold_match, sum_uint64, rate_per_input, init_acc_i64);
}

double strict_threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, threshold_match, strict_sample_match, rate_per_sample, init_acc_i64);
}

double f1_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_f1_score, init_acc_confusion);
}

double f_beta_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_f_beta_score, init_acc_confusion);
}

double matthews_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_matthews_correlation, init_acc_confusion);
}

double balanced_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_balanced_accuracy, init_acc_confusion);
}

double g_mean(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_g_mean, init_acc_confusion);
}

double cohens_kappa(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_cohens_kappa, init_acc_confusion);
}

// CLASSIFICATION FITNESS STRUCT DEFINITIONS

const struct Fitness ACCURACY = {
    .fn = accuracy,
    .type = MAXIMIZE,
    .name = "Accuracy",
    .data_type = FITNESS_INT,
    .step = exact_match,
    .combine = sum_uint64,
    .finalize = rate_per_input,
    .init_acc = init_acc_i64
};

const struct Fitness STRICT_ACCURACY = {
    .fn = strict_accuracy,
    .type = MAXIMIZE,
    .name = "Strict Accuracy",
    .data_type = FITNESS_INT,
    .step = exact_match,
    .combine = strict_sample_match,
    .finalize = rate_per_sample,
    .init_acc = init_acc_i64
};

const struct Fitness BINARY_ACCURACY = {
    .fn = binary_accuracy,
    .type = MAXIMIZE,
    .name = "Binary Accuracy",
    .data_type = FITNESS_INT,
    .step = binary_sign_match,
    .combine = sum_uint64,
    .finalize = rate_per_input,
    .init_acc = init_acc_i64
};

const struct Fitness STRICT_BINARY_ACCURACY = {
    .fn = strict_binary_accuracy,
    .type = MAXIMIZE,
    .name = "Strict Binary Accuracy",
    .data_type = FITNESS_INT,
    .step = binary_sign_match,
    .combine = strict_sample_match,
    .finalize = rate_per_sample,
    .init_acc = init_acc_i64
};

const struct Fitness THRESHOLD_ACCURACY = {
    .fn = threshold_accuracy,
    .type = MAXIMIZE,
    .name = "Threshold Accuracy",
    .data_type = FITNESS_FLOAT,
    .step = threshold_match,
    .combine = sum_uint64,
    .finalize = rate_per_input,
    .init_acc = init_acc_i64
};

const struct Fitness STRICT_THRESHOLD_ACCURACY = {
    .fn = strict_threshold_accuracy,
    .type = MAXIMIZE,
    .name = "Strict Threshold Accuracy",
    .data_type = FITNESS_FLOAT,
    .step = threshold_match,
    .combine = strict_sample_match,
    .finalize = rate_per_sample,
    .init_acc = init_acc_i64
};

const struct Fitness F1_SCORE = {
    .fn = f1_score,
    .type = MAXIMIZE,
    .name = "F1 Score",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_f1_score,
    .init_acc = init_acc_confusion
};

const struct Fitness F_BETA_SCORE = {
    .fn = f_beta_score,
    .type = MAXIMIZE,
    .name = "F-Beta Score",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_f_beta_score,
    .init_acc = init_acc_confusion
};

const struct Fitness MATTHEWS_CORRELATION = {
    .fn = matthews_correlation,
    .type = MAXIMIZE,
    .name = "Matthews Correlation",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_matthews_correlation,
    .init_acc = init_acc_confusion
};

const struct Fitness BALANCED_ACCURACY = {
    .fn = balanced_accuracy,
    .type = MAXIMIZE,
    .name = "Balanced Accuracy",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_balanced_accuracy,
    .init_acc = init_acc_confusion
};

const struct Fitness G_MEAN = {
    .fn = g_mean,
    .type = MAXIMIZE,
    .name = "G-mean",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_g_mean,
    .init_acc = init_acc_confusion
};

const struct Fitness COHENS_KAPPA = {
    .fn = cohens_kappa,
    .type = MAXIMIZE,
    .name = "Cohen's Kappa",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_cohens_kappa,
    .init_acc = init_acc_confusion
};
