#include "regression.h"
#include "../macros.h"

// REGRESSION INIT_ACC IMPLEMENTATIONS

union FitnessStepResult init_acc_f64(UNUSED_ATTRIBUTE const uint64_t inputnum, UNUSED_ATTRIBUTE const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params){
    return (union FitnessStepResult){.total_f64 = 0};
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
		MALLOC_FAIL_THREADSAFE(sizeof(double) * (ressize + ressize * inputnum));
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
        MALLOC_FAIL_THREADSAFE(sizeof(double) * 5 * ressize);
    }
    memset(result.pearson.sum_x, 0, ressize * sizeof(double));
    memset(result.pearson.sum_y, 0, ressize * sizeof(double));
    memset(result.pearson.sum_xy, 0, ressize * sizeof(double));
    memset(result.pearson.sum_x2, 0, ressize * sizeof(double));
    return result;
}

// REGRESSION STEP FUNCTION IMPLEMENTATIONS

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

// REGRESSION COMBINE FUNCTION IMPLEMENTATIONS

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

inline int max_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, UNUSED_ATTRIBUTE const struct FitnessParams *const params, UNUSED_ATTRIBUTE const uint64_t clocks){
    if(!isfinite(step_result->total_f64)) {
        accumulator->total_f64 = INFINITY;
        return 0;
    }
    if(step_result->total_f64 > accumulator->total_f64)
        accumulator->total_f64 = step_result->total_f64;
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

// REGRESSION FINALIZE FUNCTION IMPLEMENTATIONS

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

// UTILITY FUNCTION IMPLEMENTATIONS

inline int compare_doubles(const void *a, const void *b) {
    double diff = (*(double *)a - *(double *)b);
    return (diff > 0) - (diff < 0);
}

// REGRESSION FITNESS FUNCTION IMPLEMENTATIONS

inline double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, mean_input_and_dim, init_acc_f64);
}

double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, sqrt_mean_input_and_dim, init_acc_f64);
}

double length_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, mean_input_and_dim_length_pen, init_acc_f64);
}

double clock_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float_clock_pen, mean_input_and_dim, init_acc_f64);
}

double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, absolute_error, sum_float, mean_input_and_dim, init_acc_f64);
}

double mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    return eval_fitness(in, prog, max_clock, params, absolute_percent_error, sum_float, percent_mean_input_and_dim, init_acc_f64);
}

double symmetric_mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, symmetric_absolute_percent_error, sum_float, percent_mean_input_and_dim, init_acc_f64);
}

double logcosh(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, logcosh_error, sum_float, mean_input_and_dim, init_acc_f64);
}

double huber_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, huber_error, sum_float, mean_input_and_dim, init_acc_f64);
}

double worst_case_error(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, quadratic_error, max_float, max_over_ressize, init_acc_f64);
}

double pinball_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    return eval_fitness(in, prog, max_clock, params, pinball_error, sum_float, mean_input_and_dim, init_acc_f64);
}

double pearson_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with uniform average
    return eval_fitness(in, prog, max_clock, params, return_info, pearson_combine, pearson_finalize, pearson_init_acc);
}

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
        MALLOC_FAIL_THREADSAFE(sizeof(union Memblock) * delta + sizeof(double) * (delta));
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

// REGRESSION FITNESS STRUCT DEFINITIONS

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
