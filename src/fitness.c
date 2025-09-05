#include "fitness.h"
#include "macros.h"

static inline double eval_fitness(
    const struct LGPInput *const in,
    const struct Program *const prog,
    const uint64_t max_clock,
    const struct FitnessParams * const params,
    const fitness_step step,
    const fitness_combine combine,
    const fitness_finalize finalize,
    const fitness_init_acc init_acc
){
    ASSERT(prog->size > 0);
    ASSERT(in->ram_size > 0);
    ASSERT(in->input_num > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    uint64_t result_size = params->end - params->start;
    ASSERT(result_size <= in->ram_size);
    union FitnessStepResult accumulator = init_acc(in->input_num, result_size, params);
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        uint64_t clocks = run_vm(&vm, max_clock);
        union Memblock *result = &vm.ram[params->start];
        union Memblock *actual = &in->memory[(in->rom_size + in->res_size)* i + in->rom_size + params->start];
        union FitnessStepResult step_res = step(result, actual, result_size, params);
        if(! combine(&accumulator, &step_res, params, clocks)){
            break;
        }
    }
    free(vm.ram);
    return finalize(&accumulator, params, result_size, in->input_num, prog->size);
}

double *eval_multifitness(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct MultiFitness * const fitness){
    ASSERT(prog->size > 0);
    ASSERT(in->ram_size > 0);
    ASSERT(in->input_num > 0);
    ASSERT(fitness->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    uint64_t result_size = fitness->params->end - fitness->params->start;
    ASSERT(result_size <= in->ram_size);
    union FitnessStepResult *accumulator = malloc(sizeof(union FitnessStepResult) * fitness->size);
    if (accumulator == NULL) {
        free(vm.ram);
        MALLOC_FAIL_THREADSAFE;
    }
    for (uint64_t i = 0; i < fitness->size; i++) {
        accumulator[i] = fitness->functions[i].init_acc(in->input_num, result_size, &fitness->params[i]);
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        uint64_t clocks = run_vm(&vm, max_clock);
        for (uint64_t j = 0; j < fitness->size; j++) {
            union FitnessStepResult step_res = fitness->functions[j].step(
                &vm.ram[fitness->params[j].start],
                &in->memory[(in->rom_size + in->res_size)* i + in->rom_size + fitness->params[j].start],
                result_size,
                &fitness->params[j]
            );
            if(! fitness->functions[j].combine(&accumulator[j], &step_res, &fitness->params[j], clocks)){
                break;
            }
        }
    }
    free(vm.ram);
    double *results = malloc(sizeof(double) * fitness->size);
    if (results == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for (uint64_t j = 0; j < fitness->size; j++) {
        results[j] = fitness->functions[j].finalize(&accumulator[j], &fitness->params[j], result_size, in->input_num, prog->size);
    }
    free(accumulator);
    return results;
}


// ACCUMULATOR INITIALIZATION

union FitnessStepResult init_acc_i64(UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    return (union FitnessStepResult){.total_u64 = 0};
}

union FitnessStepResult init_acc_f64(UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    return (union FitnessStepResult){.total_f64 = 0};
}


union FitnessStepResult init_acc_confusion(UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    return (union FitnessStepResult){.confusion = {.true_pos = 0, .false_pos = 0, .false_neg = 0, .true_neg = 0}};
}

union FitnessStepResult init_acc_r_2(const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
	union FitnessStepResult res = {
		.r_2 = {
			.means = (double *)malloc(ressize * sizeof(double)),
            .real_vals = (double *)malloc(ressize * inputnum * sizeof(double)),
			.ss_res = 0.0
		}
	};
	if(res.r_2.means == NULL || res.r_2.real_vals == NULL){
		MALLOC_FAIL_THREADSAFE;
	}
	for(uint64_t i = 0; i < ressize; i++){
		res.r_2.means[i] = 0.0;
	}
	return res;
}

inline union FitnessStepResult pearson_init_acc(UNUSED_ATTRIBUTE const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    union FitnessStepResult result = {
        .pearson = {
            .sum_x = malloc(ressize * sizeof(double)),
            .sum_y = malloc(ressize * sizeof(double)),
            .sum_xy = malloc(ressize * sizeof(double)),
            .sum_x2 = malloc(ressize * sizeof(double)),
            .sum_y2 = malloc(ressize * sizeof(double)),
        }
    };
    if (result.pearson.sum_x == NULL || result.pearson.sum_y == NULL || result.pearson.sum_xy == NULL || result.pearson.sum_x2 == NULL || result.pearson.sum_y2 == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    memset(result.pearson.sum_x, 0, ressize * sizeof(double));
    memset(result.pearson.sum_y, 0, ressize * sizeof(double));
    memset(result.pearson.sum_xy, 0, ressize * sizeof(double));
    memset(result.pearson.sum_x2, 0, ressize * sizeof(double));
    return result;
}

inline union FitnessStepResult vect_f64_init_acc(UNUSED_ATTRIBUTE const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params) {
    union FitnessStepResult result =  {
        .vect_f64 = malloc(ressize * sizeof(double))
    };
    if (result.vect_f64 == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    memset(result.vect_f64, 0, ressize * sizeof(double));
    return result;
}

// STEP FUNCTIONS

inline union FitnessStepResult quadratic_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    double mse = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double diff = actual[i].f64 - result[i].f64;
        mse += (diff * diff);
    }
    return (union FitnessStepResult){.total_f64 = mse};
}

inline union FitnessStepResult absolute_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    double mae = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double diff = actual[i].f64 - result[i].f64;
        mae += fabs(diff);
    }
    return (union FitnessStepResult){.total_f64 = mae};
}

inline union FitnessStepResult absolute_percent_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    double mae = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double diff = actual[i].f64 - result[i].f64;
        if(actual[i].f64 != 0.0) {
            mae += fabs(diff / actual[i].f64);
        }else{
            mae += fabs(diff);
        }
    }
    return (union FitnessStepResult){.total_f64 = mae};
}

inline union FitnessStepResult return_info(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    return (union FitnessStepResult){
        .info = {
            .result = ((union Memblock *)result), // Cast to suppress the warning
            .actual = ((union Memblock *)actual), // Cast to suppress the warning
            .len = len
        }
    };
}

inline union FitnessStepResult symmetric_absolute_percent_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    double smape = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double res = result[i].f64;
        double act = actual[i].f64;
        double denominator = (fabs(act) + fabs(res)) / 2.0;
        if(denominator != 0.0) {
            smape += fabs((act - res) / denominator);
        }
    }
    return (union FitnessStepResult){.total_f64 = smape};
}

inline union FitnessStepResult logcosh_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    double logcosh = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double diff = result[i].f64 - actual[i].f64;
        logcosh += diff + log((1.0 + exp(-2.0 * diff)) / 2.0); // more stable than log(cosh(diff))
    }
    return (union FitnessStepResult){.total_f64 = logcosh};
}

inline union FitnessStepResult huber_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params){
    double huber = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double diff = fabs(result[i].f64 - actual[i].f64);
        if (diff <= params->fact.delta)
            huber += 0.5 * diff * diff;
        else
            huber += params->fact.delta * (diff - 0.5 * params->fact.delta);
    }
    return (union FitnessStepResult){.total_f64 = huber};
}

inline union FitnessStepResult pinball_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params){
    double pinball = 0.0;
    for(uint64_t i = 0; i < len; i++){
        double diff = result[i].f64 - actual[i].f64;
        if(diff >= 0.0)
            pinball += params->fact.quantile * diff;
        else
            pinball += (params->fact.quantile - 1.0) * diff;
    }
    return (union FitnessStepResult){.total_f64 = pinball};
}

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

// COMBINE FUNCTIONS

inline int sum_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    if(!isfinite(step_result->total_f64)) {
        accumulator->total_f64 = INFINITY;
        return 0;
    }
    accumulator->total_f64 += step_result->total_f64;
    return 1;
}


inline int sum_float_clock_pen(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks){
    if(!isfinite(step_result->total_f64)) {
        accumulator->total_f64 = INFINITY;
        return 0;
    }
    accumulator->total_f64 += step_result->total_f64 + params->fact.alpha * (double)clocks;
    return 1;
}

// Additional combine functions

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

inline int max_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    if(!isfinite(step_result->total_f64)) {
        accumulator->total_f64 = INFINITY;
        return 0;
    }
    if(step_result->total_f64 > accumulator->total_f64)
        accumulator->total_f64 = step_result->total_f64;
    return 1;
}

inline int strict_sample_match(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    // For strict accuracy - count if ALL labels in this sample are correct
    uint64_t expected_matches = params->end - params->start;
    if(step_result->total_u64 == expected_matches)
        accumulator->total_u64++;
    return 1;
}


int r_squared_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    for(uint64_t i = 0; i < step_result->info.len; i++){
        if(!isfinite(step_result->info.result[i].f64)) {
            free(accumulator->r_2.means);
            accumulator->r_2.ss_res = DBL_MAX;
            return 0;
        }
        accumulator->r_2.means[i] += step_result->info.actual[i].f64;
        accumulator->r_2.ss_res += (step_result->info.actual[i].f64 - step_result->info.result[i].f64) * (step_result->info.actual[i].f64 - step_result->info.result[i].f64);
        accumulator->r_2.real_vals[accumulator->r_2.count] = step_result->info.actual[i].f64;
        accumulator->r_2.count++;
    }
    return 1;
}

inline int pearson_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    for(uint64_t j = 0; j < step_result->info.len; j++){
        double result = step_result->info.result[j].f64;
        if (!(isfinite(result))){
            free(accumulator->pearson.sum_x);
            free(accumulator->pearson.sum_y);
            free(accumulator->pearson.sum_xy);
            free(accumulator->pearson.sum_x2);
            free(accumulator->pearson.sum_y2);
            return 0;
        }
        double actual_value = step_result->info.actual[j].f64;
        accumulator->pearson.sum_x[j] += result;
        accumulator->pearson.sum_y[j] += actual_value;
        accumulator->pearson.sum_xy[j] += result * actual_value;
        accumulator->pearson.sum_x2[j] += result * result;
        accumulator->pearson.sum_y2[j] += actual_value * actual_value;
    }
    return 1;
}

// FINALIZE FUNCTIONS

inline double mean_input_and_dim(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return isfinite(result->total_f64) ? result->total_f64 / (double)(inputnum * ressize) : DBL_MAX;
}

inline double sqrt_mean_input_and_dim(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return isfinite(result->total_f64) ? sqrt(result->total_f64 / (double)(inputnum * ressize)) : DBL_MAX;
}

inline double percent_mean_input_and_dim(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return isfinite(result->total_f64) ? result->total_f64 * 100.0 / (double)(inputnum * ressize) : DBL_MAX;
}

inline double mean_input_and_dim_length_pen(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size){
    return isfinite(result->total_f64) ? (result->total_f64 / (double)(inputnum * ressize)) + params->fact.alpha * (double)prog_size : DBL_MAX;
}

// Additional finalize functions

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

inline double negative_mean_input_and_dim(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return isfinite(result->total_f64) ? -(result->total_f64 / (double)(inputnum * ressize)) : -DBL_MAX;
}

inline double max_over_ressize(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    return isfinite(result->total_f64) ? result->total_f64 / (double)ressize : DBL_MAX;
}

double r_squared_finalize(const union FitnessStepResult *const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    if(! isfinite(result->r_2.ss_res)){
        free(result->r_2.means);
        return -DBL_MAX;
    }
    double ss_tot = 0.0;
    for(uint64_t i = 0; i < inputnum; i++){
        for(uint64_t j = 0; j < ressize; j++){
            double actual_value = result->r_2.real_vals[i * ressize + j];
            double mean =  result->r_2.means[j] / inputnum;
            ss_tot += (actual_value - mean) * (actual_value - mean);
        }
    }
    free(result->r_2.means);
    if(! isfinite(ss_tot) || ss_tot <= 0.0)
        return -DBL_MAX;
    else 
        return 1.0 - (result->r_2.ss_res / ss_tot);
}

inline double pearson_finalize(const union FitnessStepResult * const result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size){
    uint64_t valid = 0;
    double total = 0.0;
    for(uint64_t j = 0; j < ressize; j++){
        double numerator = (inputnum * result->pearson.sum_xy[j]) - (result->pearson.sum_x[j] * result->pearson.sum_y[j]);
        double denominator = sqrt((inputnum * result->pearson.sum_x2[j] - result->pearson.sum_x[j] * result->pearson.sum_x[j]) * (inputnum * result->pearson.sum_y2[j] - result->pearson.sum_y[j] * result->pearson.sum_y[j]));
        if(denominator != 0.0){
            valid++;
            total += numerator / denominator;
        }
    }
    free(result->pearson.sum_x);
    free(result->pearson.sum_y);
    free(result->pearson.sum_xy);
    free(result->pearson.sum_x2);
    free(result->pearson.sum_y2);
    if(valid > 0)
        return total / (double)valid;
    else
        return DBL_MAX;
}

inline int compare_doubles(const void *a, const void *b) {
    double diff = (*(double *)a - *(double *)b);
    return (diff > 0) - (diff < 0);
}

inline double value_at_risk_finalize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const uint64_t prog_size) {
    qsort(result->vect_f64, inputnum, sizeof(double), compare_doubles);
    double error = 0.0;
    uint64_t count = (uint64_t) ceil(params->fact.alpha * inputnum);
    for (uint64_t i = inputnum - count; i < inputnum; i++) {
        error += result->vect_f64[i] /((double)ressize);
    }
    free(result->vect_f64);
    if (!(isfinite(error)))
        return DBL_MAX;
    return error / count;
}


// FITNESS DECLARATIONS

inline double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, mean_input_and_dim, init_acc_f64);
}

const struct Fitness MSE = {
    .fn = mse,
    .type = MINIMIZE,
    .name = "MSE",
    .data_type = FITNESS_FLOAT,
    .step = quadratic_error,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};

double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, sqrt_mean_input_and_dim, init_acc_f64);
}
const struct Fitness RMSE = {
    .fn = rmse,
    .type = MINIMIZE,
    .name = "RMSE",
    .data_type = FITNESS_FLOAT,
    .step = quadratic_error,
    .combine = sum_float,
    .finalize = sqrt_mean_input_and_dim,
    .init_acc = init_acc_f64
};

double length_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, mean_input_and_dim_length_pen, init_acc_f64);
}
const struct Fitness LENGTH_PENALIZED_MSE = {
    .fn = length_penalized_mse,
    .type = MINIMIZE,
    .name = "Length Penalized MSE",
    .data_type = FITNESS_FLOAT,
    .step = quadratic_error,
    .combine = sum_float,
    .finalize = mean_input_and_dim_length_pen,
    .init_acc = init_acc_f64
};

double clock_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float_clock_pen, mean_input_and_dim, init_acc_f64);
}
const struct Fitness CLOCK_PENALIZED_MSE = {
    .fn = clock_penalized_mse,
    .type = MINIMIZE,
    .name = "Clock Penalized MSE",
    .data_type = FITNESS_FLOAT,
    .step = quadratic_error,
    .combine = sum_float_clock_pen,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};

double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, absolute_error, sum_float, mean_input_and_dim, init_acc_f64);
}

const struct Fitness MAE = {
    .fn = mae,
    .type = MINIMIZE, 
    .name = "MAE",
    .data_type = FITNESS_FLOAT,
    .step = absolute_error,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};

double mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, absolute_percent_error, sum_float, percent_mean_input_and_dim, init_acc_f64);
}

const struct Fitness MAPE = {
    .fn = mape,
    .type = MINIMIZE,
    .name = "MAPE",
    .data_type = FITNESS_FLOAT,
    .step = absolute_percent_error,
    .combine = sum_float,
    .finalize = percent_mean_input_and_dim,
    .init_acc = init_acc_f64
};


double symmetric_mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, symmetric_absolute_percent_error, sum_float, percent_mean_input_and_dim, init_acc_f64);
}

const struct Fitness SYMMETRIC_MAPE = {
    .fn = symmetric_mape,
    .type = MINIMIZE,
    .name = "Symmetric MAPE",
    .data_type = FITNESS_FLOAT,
    .step = symmetric_absolute_percent_error,
    .combine = sum_float,
    .finalize = percent_mean_input_and_dim,
    .init_acc = init_acc_f64
};

double logcosh(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, logcosh_error, sum_float, mean_input_and_dim, init_acc_f64);
}

const struct Fitness LOGCOSH = {
    .fn = logcosh,
    .type = MINIMIZE,
    .name = "LogCosh",
    .data_type = FITNESS_FLOAT,
    .step = logcosh_error,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};

double huber_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, huber_error, sum_float, mean_input_and_dim, init_acc_f64);
}

const struct Fitness HUBER_LOSS = {
    .fn = huber_loss,
    .type = MINIMIZE,
    .name = "Huber Loss",
    .data_type = FITNESS_FLOAT,
    .step = huber_error,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};  

double worst_case_error(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, quadratic_error, max_float, max_over_ressize, init_acc_f64);
}

const struct Fitness WORST_CASE_ERROR = {
    .fn = worst_case_error,
    .type = MINIMIZE,
    .name = "Worst Case Error",
    .data_type = FITNESS_FLOAT,
    .step = quadratic_error,
    .combine = max_float,
    .finalize = max_over_ressize,
    .init_acc = init_acc_f64
};

double pinball_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, pinball_error, sum_float, mean_input_and_dim, init_acc_f64);
}

const struct Fitness PINBALL_LOSS = {
    .fn = pinball_loss,
    .type = MINIMIZE,
    .name = "Pinball Loss",
    .data_type = FITNESS_FLOAT,
    .step = pinball_error,
    .combine = sum_float,
    .finalize = mean_input_and_dim,
    .init_acc = init_acc_f64
};

double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, exact_match, sum_uint64, rate_per_input, init_acc_i64);
}

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


double strict_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, exact_match, strict_sample_match, rate_per_sample, init_acc_i64);
}

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

double binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_sign_match, sum_uint64, rate_per_input, init_acc_i64);
}

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


double strict_binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_sign_match, strict_sample_match, rate_per_sample, init_acc_i64);
}

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


double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, threshold_match, sum_uint64, rate_per_input, init_acc_i64);
}

const struct Fitness THRESHOLD_ACCURACY = {
    .fn = threshold_accuracy,
    .type = MAXIMIZE,
    .name = "Threshold Accuracy",
    .data_type = FITNESS_FLOAT,
    .step = threshold_match,
    .combine = sum_uint64,
    .finalize = rate_per_input,
    .init_acc = init_acc_f64
};


double strict_threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, threshold_match, strict_sample_match, rate_per_sample, init_acc_i64);
}

const struct Fitness STRICT_THRESHOLD_ACCURACY = {
    .fn = strict_threshold_accuracy,
    .type = MAXIMIZE,
    .name = "Strict Threshold Accuracy",
    .data_type = FITNESS_FLOAT,
    .step = threshold_match,
    .combine = strict_sample_match,
    .finalize = rate_per_sample,
    .init_acc = init_acc_f64
};


double f1_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_f1_score, init_acc_confusion);
}

const struct Fitness F1_SCORE = {
    .fn = f1_score,
    .type = MAXIMIZE,
    .name = "F1 Score",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_f1_score,
    .init_acc = init_acc_i64
};

double f_beta_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_f_beta_score, init_acc_confusion);
}

const struct Fitness F_BETA_SCORE = {
    .fn = f_beta_score,
    .type = MAXIMIZE,
    .name = "F-Beta Score",
    .data_type = FITNESS_SIGN_BIT,
    .step = binary_classification_confusion,
    .combine = sum_confusion,
    .finalize = confusion_f_beta_score,
    .init_acc = init_acc_i64
};

double binary_cross_entropy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, cross_entropy_step, sum_float, mean_input_and_dim, init_acc_f64);
}

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

double gaussian_log_likelihood(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, gaussian_likelihood_step, sum_float, negative_mean_input_and_dim, init_acc_f64);
}

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

double brier_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, brier_score_step, sum_float, mean_input_and_dim, init_acc_f64);
}

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


double hinge_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, hinge_loss_step, sum_float, mean_input_and_dim, init_acc_f64);
}

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

double matthews_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_matthews_correlation, init_acc_confusion);
}

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

double balanced_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_balanced_accuracy, init_acc_confusion);
}

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


double g_mean(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_g_mean, init_acc_confusion);
}

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

double cohens_kappa(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    return eval_fitness(in, prog, max_clock, params, binary_classification_confusion, sum_confusion, confusion_cohens_kappa, init_acc_confusion);
}

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

double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with uniform average
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double ss_res = 0.0;
    double ss_tot = 0.0;
    uint64_t delta = params->end - params->start;
    vm.ram = malloc(sizeof(union Memblock) * delta);
    double *means = malloc(sizeof(double) * (delta));
    if (vm.ram == NULL || means == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    memset(means, 0, sizeof(double) * (delta));
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(means);
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            means[j - params->start] += (actual_value / (double)in->input_num);
            ss_res += (actual_value - result) * (actual_value - result);
        }
    }
    free(vm.ram);
    for(uint64_t i = 0; i < in->input_num; i++){
        for(uint64_t j = params->start; j < params->end; j++){
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            ss_tot += (actual_value - means[j - params->start]) * (actual_value - means[j - params->start]);
        }
    }
    free(means);
    if(! isfinite(ss_res) || ! isfinite(ss_tot) || ss_tot <= 0.0)
        return DBL_MAX;

    return 1.0 - (ss_res / ss_tot);
}

const struct Fitness R_SQUARED = {
    .fn = r_squared,
    .type = MAXIMIZE,
    .name = "R^2",
    .data_type = FITNESS_FLOAT,
    .step = return_info,
    .combine = r_squared_combine,
    .finalize = r_squared_finalize,
    .init_acc = init_acc_r_2
};

double pearson_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with uniform average
    return eval_fitness(in, prog, max_clock, params, return_info, pearson_combine, pearson_finalize, pearson_init_acc);
}

const struct Fitness PEARSON_CORRELATION = {
    .fn = pearson_correlation,
    .type = MAXIMIZE,
    .name = "Pearson Correlation",
    .data_type = FITNESS_FLOAT,
    .step = return_info,
    .combine = pearson_combine,
    .finalize = pearson_finalize,
    .init_acc = pearson_init_acc
};

double conditional_value_at_risk(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with vectorial distance of the output as error
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, value_at_risk_finalize, vect_f64_init_acc);
}

const struct Fitness CONDITIONAL_VALUE_AT_RISK = {
    .fn = conditional_value_at_risk,
    .type = MINIMIZE,
    .name = "Conditional Value at Risk",
    .data_type = FITNESS_FLOAT,
    .step = quadratic_error,
    .combine = sum_float,
    .finalize = value_at_risk_finalize,
    .init_acc = vect_f64_init_acc
};

double adversarial_perturbation_sensibility(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with vectorial distance of the output as error
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double sum_error = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    union Memblock *altered_rom = malloc(sizeof(union Memblock) * in->rom_size);
    if (vm.ram == NULL || altered_rom == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        double error = 0.0;
        run_vm(&vm, max_clock);
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(altered_rom);
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = result - actual_value;
            error += diff * diff;
        }
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        for(uint64_t j = 0; j < in->rom_size; j++){
            altered_rom[j].f64 = in->memory[(in->rom_size + in->res_size)* i + j].f64 + params->fact.perturbation_vector[j];
        }
        vm.rom = altered_rom;
        run_vm(&vm, max_clock);
        double error_pos = 0.0;
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(altered_rom);
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = result - actual_value;
            error_pos += diff * diff;
        }
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        for(uint64_t j = 0; j < in->rom_size; j++){
            altered_rom[j].f64 = in->memory[(in->rom_size + in->res_size)* i + j].f64 - params->fact.perturbation_vector[j];
        }
        run_vm(&vm, max_clock);
        double error_neg = 0.0;
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(altered_rom);
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = result - actual_value;
            error_neg += diff * diff;
        }
        double max_error = fmax(error_pos, error_neg);
        sum_error += fabs(max_error - error);
    }
    free(vm.ram);
    free(altered_rom);
    if(isfinite(sum_error))
        return sum_error / (double)(in->input_num * (params->end - params->start));
    else
        return DBL_MAX;
}

const struct Fitness ADVERSARIAL_PERTURBATION_SENSITIVITY = {
    .fn = adversarial_perturbation_sensibility,
    .type = MINIMIZE,
    .name = "Adversarial Perturbation Sensibility",
    .data_type = FITNESS_FLOAT,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL
};