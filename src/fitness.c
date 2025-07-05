#include "fitness.h"

inline double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mse = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_mse = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
        mse += (actual_mse * actual_mse);
    }
    free(vm.ram);
    if(isfinite(mse))
        return mse / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment MSE = {.fn = mse, .type = MINIMIZE, .name = "MSE"};

double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params) {
    double mse_val = mse(in, prog, max_clock, params);
    return isfinite(mse_val) ? sqrt(mse_val) : DBL_MAX;
}
const struct FitnessAssesment RMSE = {.fn = rmse, .type = MINIMIZE, .name = "RMSE"};

double lenght_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params) {
    double base = mse(in, prog, max_clock, params);
    if (!isfinite(base)) return DBL_MAX;
    return base + params->alpha * (double)prog->size;
}
const struct FitnessAssesment LENGHT_PENALIZED_MSE = {.fn   = lenght_penalized_mse, .type = MINIMIZE, .name = "Lenght Penalized MSE"};

double clock_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params) {
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mse = 0.0;
    uint64_t clock = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        uint64_t actual_clock = run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_mse = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
        mse += (actual_mse * actual_mse);
        clock += actual_clock;
    }
    free(vm.ram);
    if(isfinite(mse))
        return (mse + params->alpha * clock) / (double)in->input_num;
    else
        return DBL_MAX;
}
const struct FitnessAssesment CLOCK_PENALIZED_MSE = {.fn   = clock_penalized_mse, .type = MINIMIZE, .name = "Clock Penalized MSE"};

double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mae = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_mae = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
        mae += fabs(actual_mae);
    }
    free(vm.ram);
    if(isfinite(mae))
        return mae / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment MAE = {.fn = mae, .type = MINIMIZE, .name = "MAE"};

double mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mape = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        if(actual)
            mape += fabs((actual - result) / actual);
        else
            mape += fabs((actual - result));
    }
    free(vm.ram);
    if(isfinite(mape))
        return mape * 100 / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment MAPE = {.fn = mape, .type = MINIMIZE, .name = "MAPE"};


double symmetric_mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mape = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        double denominator = (fabs(actual) + fabs(result)) / 2;
        if(denominator)
            mape += fabs((actual - result) / denominator);
    }
    free(vm.ram);
    if(isfinite(mape))
        return mape * 100 / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment SYMMETRIC_MAPE = {.fn = symmetric_mape, .type = MINIMIZE, .name = "MAPE"};

double logcosh(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double logcosh = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        logcosh += log(cosh(actual_value - result));
    }
    free(vm.ram);
    if(isfinite(logcosh))
        return logcosh / (double)in->input_num;
    else
        return DBL_MAX;
}
const struct FitnessAssesment LOGCOSH = {.fn = logcosh, .type = MINIMIZE, .name = "LogCosh"};

double huber_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double huber_loss = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        double diff = fabs(actual_value - result);
        if (diff <= params->delta)
            huber_loss += 0.5 * diff * diff;
        else
            huber_loss += params->delta * (diff - 0.5 * params->delta);
    }
    free(vm.ram);
    if(isfinite(huber_loss))
        return huber_loss / (double)in->input_num;
    else
        return DBL_MAX;
}
const struct FitnessAssesment HUBER_LOSS = {.fn = huber_loss, .type = MINIMIZE, .name = "Huber Loss"};

double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double ss_res = 0.0;
    double ss_tot = 0.0;
    double mean = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        mean += actual_value;
        ss_res += (actual_value - result) * (actual_value - result);
    }
    free(vm.ram);
    mean /= (double)in->input_num;
    for(uint64_t i = 0; i < in->input_num; i++){
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        ss_tot += (actual_value - mean) * (actual_value - mean);
    }
    if(ss_tot <= 0.0 || ! isfinite(ss_tot))
        return DBL_MAX;

    return 1.0 - (ss_res / ss_tot);
}

const struct FitnessAssesment RSQUARED = {.fn = r_squared, .type = MAXIMIZE, .name = "R^2"};

double worst_case_error(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double worst_case_error = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double error = fabs(in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result);
        if(error > worst_case_error)
            worst_case_error = error;
    }
    free(vm.ram);
    if(isfinite(worst_case_error))
        return worst_case_error;
    else
        return DBL_MAX;
}

const struct FitnessAssesment WORST_CASE_ERROR = {.fn = worst_case_error, .type = MINIMIZE, .name = "Worst Case Error"};

double pinball_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double pinball_loss = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        double diff = actual_value - result;
        if (diff >= 0)
            pinball_loss += params->quantile * diff;
        else
            pinball_loss += (params->quantile - 1.0) * (diff);
    }
    free(vm.ram);
    if(isfinite(pinball_loss))
        return pinball_loss / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment PINBALL_LOSS = {.fn = pinball_loss, .type = MINIMIZE, .name = "Pinball Loss"};

double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t correct = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64;
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64;
        if (actual_value == result)
            correct++;
    }
    free(vm.ram);
    return (double)correct / (double)in->input_num;
}

double pearson_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        sum_x += result;
        sum_y += actual_value;
        sum_xy += result * actual_value;
        sum_x2 += result * result;
        sum_y2 += actual_value * actual_value;
    }
    free(vm.ram);
    double numerator = (in->input_num * sum_xy) - (sum_x * sum_y);
    double denominator = sqrt((in->input_num * sum_x2 - sum_x * sum_x) * (in->input_num * sum_y2 - sum_y * sum_y));
    if(denominator == 0.0)
        return DBL_MAX;

    return numerator / denominator;
}

const struct FitnessAssesment PEARSON_CORRELATION = {.fn = pearson_correlation, .type = MAXIMIZE, .name = "Pearson Correlation"};

const struct FitnessAssesment ACCURACY = {.fn = accuracy, .type = MAXIMIZE, .name = "Accuracy"};

double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t correct = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        if (fabs(actual_value - result) <= params->threshold)
            correct++;
    }
    free(vm.ram);
    return (double)correct / (double)in->input_num;
}

const struct FitnessAssesment THRESHOLD_ACCURACY = {.fn = threshold_accuracy, .type = MAXIMIZE, .name = "Threshold Accuracy"};

double f1_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        if (actual_value && result)
            true_positive++;
        else if (actual_value && (! result))
            false_negative++;
        else if ((! actual_value) && result)
            false_positive++;
    }
    free(vm.ram);
    uint64_t pred_positive = true_positive + false_positive;
    uint64_t positive = true_positive + false_negative;
    if( pred_positive == 0 || positive == 0)
        return 0.0;
    double precision = (double)true_positive / (double)(pred_positive);
    double recall = (double)true_positive / (double)(positive);
    if(precision + recall == 0.0)
        return 0.0;

    return 2.0 * precision * recall / (precision + recall);
}

const struct FitnessAssesment F1_SCORE = {.fn = f1_score, .type = MAXIMIZE, .name = "F1 Score"};

double f_beta_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        if (actual_value && result)
            true_positive++;
        else if (actual_value && (! result))
            false_negative++;
        else if ((! actual_value) && result)
            false_positive++;
    }
    free(vm.ram);
    uint64_t pred_positive = true_positive + false_positive;
    uint64_t positive = true_positive + false_negative;
    if( pred_positive == 0 || positive == 0)
        return 0.0;
    double precision = (double)true_positive / (double)(pred_positive);
    double recall = (double)true_positive / (double)(positive);
    if(precision + recall == 0.0)
        return 0.0;
    double beta2 = params->beta * params->beta;
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall);
}

const struct FitnessAssesment F_BETA_SCORE = {.fn = f_beta_score, .type = MAXIMIZE, .name = "F-Beta Score"};

double binary_cross_entropy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double bce = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if ((! (isfinite(result))) || result < 0 || result > 1){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        double p = result < params->tollerance ? params->tollerance : (result > 1.0 - params->tollerance ? 1.0 - params->tollerance : result);
        bce += -actual_value * log(p) - (1.0 - actual_value) * log(1.0 - p);
    }
    free(vm.ram);
    if(isfinite(bce))
        return bce / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment BINARY_CROSS_ENTROPY = {.fn = binary_cross_entropy, .type = MINIMIZE, .name = "Binary Cross Entropy"};

double gaussian_log_likelihood(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params) {
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double sigma_squared = params->sigma * params->sigma;
    double C = -0.5 * log(2 * M_PI * sigma_squared);
    double sum = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        double err = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
        sum += C - 0.5 * (err * err)/(sigma_squared);
    }
    free(vm.ram);
    if(isfinite(sum))
        return sum;
    else
        return 0;
}

const struct FitnessAssesment GAUSSIAN_LOG_LIKELIHOOD = {.fn = gaussian_log_likelihood, .type = MAXIMIZE, .name = "Gaussian Log Likelihood"};

double brier_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double sum = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if ((! (isfinite(result))) || result < 0 || result > 1){
            free(vm.ram);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        double error = result - actual_value;
        sum += error * error;
    }
    free(vm.ram);
    if(isfinite(sum))
        return sum / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment BRIER_SCORE = {.fn = brier_score, .type = MINIMIZE, .name = "Brier Score"};


double hinge_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double sum = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64 & (1UL << 63UL);
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64;
        sum += fmax(0.0, 1.0 - actual_value * result);
    }
    free(vm.ram);
    if(isfinite(sum))
        return sum / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment HINGE_LOSS = {.fn = hinge_loss, .type = MINIMIZE, .name = "Hinge Loss"};

double matthews_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params) {
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        if (actual_value && result)
            true_positive++;
        else if (actual_value && (! result))
            false_negative++;
        else if ((! actual_value) && result)
            false_positive++;
        else
            true_negative++;
    }
    free(vm.ram);
    double numerator = (double) true_positive * (double) true_negative - (double) false_positive * (double) false_negative;
    double denominator = sqrt(
        (double) (true_positive + false_positive) *
        (double) (true_positive + false_negative) *
        (double) (true_negative + false_positive) *
        (double) (true_negative + false_negative)
    );
    if(denominator == 0.0)
        return DBL_MAX;

    return numerator / denominator;
}

const struct FitnessAssesment MATTHEWS_CORRELATION = {.fn   = matthews_correlation, .type = MAXIMIZE, .name = "Matthews Correation"};

double balanced_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params) {
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        if (actual_value && result)
            true_positive++;
        else if (actual_value && (! result))
            false_negative++;
        else if ((! actual_value) && result)
            false_positive++;
        else
            true_negative++;
    }
    free(vm.ram);
    double sensitivity = (true_positive + false_negative) ? (double) true_positive / (double) (true_positive + false_negative) : 0.0;
    double specificity = (true_negative + false_positive) ? (double) true_negative / (double) (true_negative + false_positive) : 0.0;
    return 0.5 * (sensitivity + specificity);
}

const struct FitnessAssesment BALANCED_ACCURACY = {.fn = balanced_accuracy, .type = MAXIMIZE, .name = "Balanced Accuracy"};


double g_mean(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params) {
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        if (actual_value && result)
            true_positive++;
        else if (actual_value && (! result))
            false_negative++;
        else if ((! actual_value) && result)
            false_positive++;
        else
            true_negative++;
    }
    free(vm.ram);
    double sensitivity = (true_positive + false_negative) ? (double) true_positive / (double) (true_positive + false_negative) : 0.0;
    double specificity = (true_negative + false_positive) ? (double) true_negative / (double) (true_negative + false_positive) : 0.0;
    return sqrt(sensitivity * specificity);
}

const struct FitnessAssesment G_MEAN = {.fn = g_mean, .type = MAXIMIZE, .name = "G-mean"};

double cohens_kappa(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params) {
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64 & (((uint64_t) 1) << ((uint64_t) 63));
        if (actual_value && result)
            true_positive++;
        else if (actual_value && (! result))
            false_negative++;
        else if ((! actual_value) && result)
            false_positive++;
        else
            true_negative++;
    }
    free(vm.ram);
    double observed_agreement = (double)(true_positive + true_negative) / (double)in->input_num;
    double expected_agreement = ((double)(true_positive + false_positive) / (double)in->input_num) * ((double)(true_positive + false_negative) / (double)in->input_num)
                                + ((double)(false_positive + true_negative) / (double)in->input_num) * ((double)(false_negative + true_negative) / (double)in->input_num);
    if(expected_agreement == 1.0)
        return DBL_MAX;

    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement);
}

const struct FitnessAssesment COHENS_KAPPA = {.fn = cohens_kappa, .type = MAXIMIZE, .name = "Cohen's Kappa"};

double adversarial_perturbation_sensibility(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params) {
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
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            return DBL_MAX;
        }
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        for(uint64_t j = 0; j < in->rom_size; j++){
            altered_rom[j].f64 = in->memory[(in->rom_size + in->res_size)* i + j].f64 + params->perturbation_vector[j];
        }
        vm.rom = altered_rom;
        run_vm(&vm, max_clock);
        double resultpos = vm.ram[0].f64;
        if (!(isfinite(resultpos))){
            free(vm.ram);
            free(altered_rom);
            return DBL_MAX;
        }
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        for(uint64_t j = 0; j < in->rom_size; j++){
            altered_rom[j].f64 = in->memory[(in->rom_size + in->res_size)* i + j].f64 - params->perturbation_vector[j];
        }
        run_vm(&vm, max_clock);
        double resultneg = vm.ram[0].f64;
        if (!(isfinite(resultneg))){
            free(vm.ram);
            free(altered_rom);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        double error_pos = fabs(resultpos - actual_value);
        double error_neg = fabs(resultneg - actual_value);
        double error = fabs(result - actual_value);
        double max_error = fmax(error_pos, error_neg);
        sum_error += fabs(max_error - error);
    }
    free(vm.ram);
    free(altered_rom);
    if(isfinite(sum_error))
        return sum_error / in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment ADVERSARIAL_PERTURBATION_SENSIBILITY = {.fn = adversarial_perturbation_sensibility, .type = MINIMIZE, .name = "Adversarial Perturbation Sensibility"};

static inline int compare_doubles(const void *a, const void *b) {
    double diff = (*(double *)a - *(double *)b);
    return (diff > 0) - (diff < 0);
}


double conditional_value_at_risk(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params) {
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t count = (uint64_t) ceil(params->alpha * in->input_num);
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    double *results = malloc(sizeof(double) * in->input_num);
    if (vm.ram == NULL || results == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
            free(vm.ram);
            free(results);
            return DBL_MAX;
        }
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        results[i] = fabs(actual_value - result);
    }
    free(vm.ram);
    if(count == 0){
        free(results);
        return DBL_MAX;
    }
    qsort(results, in->input_num, sizeof(double), compare_doubles);
    double error = 0.0;
    for (uint64_t i = in->input_num - count; i < in->input_num; i++) {
        error += results[i];
    }
    free(results);
    if (!(isfinite(error)))
        return DBL_MAX;
    return error / count;
}

const struct FitnessAssesment CONDITIONAL_VALUE_AT_RISK = {.fn = conditional_value_at_risk, .type = MINIMIZE, .name = "Conditional Value at Risk"};