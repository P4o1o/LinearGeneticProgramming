#include "fitness.h"

inline double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_mse = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64 - result;
            mse += (actual_mse * actual_mse);
        }
    }
    free(vm.ram);
    if(isfinite(mse))
        return mse / ((double)in->input_num * (double)(params->end - params->start));
    else
        return DBL_MAX;
}

const struct Fitness MSE = {.fn = mse, .type = MINIMIZE, .name = "MSE"};

double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    double mse_val = mse(in, prog, max_clock, params);
    return isfinite(mse_val) ? sqrt(mse_val) : DBL_MAX;
}
const struct Fitness RMSE = {.fn = rmse, .type = MINIMIZE, .name = "RMSE"};

double length_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    double base = mse(in, prog, max_clock, params);
    if (!isfinite(base)) return DBL_MAX;
    return base + params->fact.alpha * (double)prog->size;
}
const struct Fitness LENGTH_PENALIZED_MSE = {.fn   = length_penalized_mse, .type = MINIMIZE, .name = "Length Penalized MSE"};

double clock_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
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
        clock += actual_clock;
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_mse = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64 - result;
            mse += (actual_mse * actual_mse);
        }
    }
    free(vm.ram);
    if(isfinite(mse))
        return (mse + params->fact.alpha * clock) / (((double)in->input_num * (double)(params->end - params->start)));
    else
        return DBL_MAX;
}
const struct Fitness CLOCK_PENALIZED_MSE = {.fn   = clock_penalized_mse, .type = MINIMIZE, .name = "Clock Penalized MSE"};

double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_mae = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64 - result;
            mae += fabs(actual_mae);
        }
    }
    free(vm.ram);
    if(isfinite(mae))
        return mae / (((double)in->input_num * (double)(params->end - params->start)));
    else
        return DBL_MAX;
}

const struct Fitness MAE = {.fn = mae, .type = MINIMIZE, .name = "MAE"};

double mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            if(actual)
                mape += fabs((actual - result) / actual);
            else
                mape += fabs((actual - result));
        }
    }
    free(vm.ram);
    if(isfinite(mape))
        return mape * 100 / (((double)in->input_num * (double)(params->end - params->start)));
    else
        return DBL_MAX;
}

const struct Fitness MAPE = {.fn = mape, .type = MINIMIZE, .name = "MAPE"};


double symmetric_mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double denominator = (fabs(actual) + fabs(result)) / 2;
            if(denominator)
                mape += fabs((actual - result) / denominator);
        }
    }
    free(vm.ram);
    if(isfinite(mape))
        return mape * 100 / (((double)in->input_num * (double)(params->end - params->start)));
    else
        return DBL_MAX;
}

const struct Fitness SYMMETRIC_MAPE = {.fn = symmetric_mape, .type = MINIMIZE, .name = "MAPE"};

double logcosh(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = result - actual_value;
            logcosh += diff + log((1 + exp(-2 * diff)) / 2); // more stable than log(cosh(diff))
        }
    }
    free(vm.ram);
    if(isfinite(logcosh))
        return logcosh / (((double)in->input_num * (double)(params->end - params->start)));
    else
        return DBL_MAX;
}
const struct Fitness LOGCOSH = {.fn = logcosh, .type = MINIMIZE, .name = "LogCosh"};

double huber_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = fabs(result - actual_value);
            if (diff <= params->fact.delta)
                huber_loss += 0.5 * diff * diff;
            else
                huber_loss += params->fact.delta * (diff - 0.5 * params->fact.delta);
        }
    }
    free(vm.ram);
    if(isfinite(huber_loss))
        return huber_loss / (((double)in->input_num * (double)(params->end - params->start)));
    else
        return DBL_MAX;
}
const struct Fitness HUBER_LOSS = {.fn = huber_loss, .type = MINIMIZE, .name = "Huber Loss"};  

double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with uniform average
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double ss_res = 0.0;
    double ss_tot = 0.0;
    uint64_t delta = params->end - params->start;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
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


const struct Fitness R_SQUARED = {.fn = r_squared, .type = MAXIMIZE, .name = "R^2"};

double worst_case_error(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
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
        double error = 0.0;
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = fabs(result - actual_value);
            error += diff;
        }
        if(error > worst_case_error)
            worst_case_error = error;
    }
    free(vm.ram);
    if(isfinite(worst_case_error))
        return worst_case_error / (double)(params->end - params->start);
    else
        return DBL_MAX;
}

const struct Fitness WORST_CASE_ERROR = {.fn = worst_case_error, .type = MINIMIZE, .name = "Worst Case Error"};

double pinball_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with uniform average
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = fabs(result - actual_value);
            if(diff >= 0)
                pinball_loss += params->fact.quantile * diff;
            else
                pinball_loss += (params->fact.quantile - 1.0) * diff;
        }
    }
    free(vm.ram);
    if(isfinite(pinball_loss))
        return pinball_loss / ((double)in->input_num * (double)(params->end - params->start));
    else
        return DBL_MAX;
}

const struct Fitness PINBALL_LOSS = {.fn = pinball_loss, .type = MINIMIZE, .name = "Pinball Loss"};

double pearson_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with uniform average
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t delta = params->end - params->start;
    double *sum_x = malloc(delta * sizeof(double));
    double *sum_y = malloc(delta * sizeof(double));
    double *sum_xy = malloc(delta * sizeof(double));
    double *sum_x2 = malloc(delta * sizeof(double));
    double *sum_y2 = malloc(delta * sizeof(double));
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL || sum_x == NULL || sum_y == NULL || sum_xy == NULL || sum_x2 == NULL || sum_y2 == NULL) {
        MALLOC_FAIL_THREADSAFE;
    }
    memset(sum_x, 0, delta * sizeof(double));
    memset(sum_y, 0, delta * sizeof(double));
    memset(sum_xy, 0, delta * sizeof(double));
    memset(sum_x2, 0, delta * sizeof(double));
    memset(sum_y2, 0, delta * sizeof(double));
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        for(uint64_t j = 0; j < delta; j++){
            double result = vm.ram[params->start + j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                free(sum_x);
                free(sum_y);
                free(sum_xy);
                free(sum_x2);
                free(sum_y2);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + params->start + j].f64;
            sum_x[j] += result;
            sum_y[j] += actual_value;
            sum_xy[j] += result * actual_value;
            sum_x2[j] += result * result;
            sum_y2[j] += actual_value * actual_value;
        }
    }
    free(vm.ram);
    uint64_t valid = 0;
    double total = 0.0;
    for(uint64_t j = 0; j < delta; j++){
        double numerator = (in->input_num * sum_xy[j]) - (sum_x[j] * sum_y[j]);
        double denominator = sqrt((in->input_num * sum_x2[j] - sum_x[j] * sum_x[j]) * (in->input_num * sum_y2[j] - sum_y[j] * sum_y[j]));
        if(denominator != 0.0){
            valid++;
            total += numerator / denominator;
        }
    }
    free(sum_x);
    free(sum_y);
    free(sum_xy);
    free(sum_x2);
    free(sum_y2);
    if(valid > 0)
        return total / (double)valid;
    else
        return DBL_MAX;
}

const struct Fitness PEARSON_CORRELATION = {.fn = pearson_correlation, .type = MAXIMIZE, .name = "Pearson Correlation"};

double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a integer value representing a class label
    // It counts how many times the program output matches the actual value.
    // Vectorial version with per-label accuracy
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64;
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64;
            if (actual_value == result)
                correct++;
        }
    }
    free(vm.ram);
    return (double)correct / ((double)in->input_num * (double)(params->end - params->start));
}

const struct Fitness ACCURACY = {.fn = accuracy, .type = MAXIMIZE, .name = "Accuracy"};


double strict_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a integer value representing a class label
    // It counts how many times the program output matches the actual value.
    // Vectorial version with strict-per-sample accuracy
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
        uint64_t match = 1;
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64;
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64;
            if (actual_value != result){
                match = 0;
                break;
            }
        }
        correct += match;
    }
    free(vm.ram);
    return (double)correct / (double)in->input_num;
}

const struct Fitness STRICT_ACCURACY = {.fn = strict_accuracy, .type = MAXIMIZE, .name = "Strict Accuracy"};

double binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // It counts how many times the program output matches the actual value.
    // Vectorial version with per-label accuracy
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64  & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value == result)
                correct++;
        }
    }
    free(vm.ram);
    return (double)correct / ((double)in->input_num * (double)(params->end - params->start));
}

const struct Fitness BINARY_ACCURACY = {.fn = binary_accuracy, .type = MAXIMIZE, .name = "Binary Accuracy"};


double strict_binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // It counts how many times the program output matches the actual value.
    // Vectorial version with strict-per-sample accuracy
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
        uint64_t match = 1;
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value != result){
                match = 0;
                break;
            }
        }
        correct += match;
    }
    free(vm.ram);
    return (double)correct / (double)in->input_num;
}

const struct Fitness STRICT_BINARY_ACCURACY = {.fn = strict_binary_accuracy, .type = MAXIMIZE, .name = "Strict Binary Accuracy"};


double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a floating point value.
    // It counts how many times the program output is within a certain threshold of the actual value.
    // The threshold is defined in the params.
    // Vectorial version with per-label accuracy
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            if (fabs(actual_value - result) <= params->fact.threshold)
                correct++;
        }
    }
    free(vm.ram);
    return (double)correct / ((double)in->input_num * (double)(params->end - params->start));
}

const struct Fitness THRESHOLD_ACCURACY = {.fn = threshold_accuracy, .type = MAXIMIZE, .name = "Threshold Accuracy"};


double strict_threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a floating point value.
    // It counts how many times the program output is within a certain threshold of the actual value.
    // The threshold is defined in the params.
    // Vectorial version with strict-per-sample accuracy
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
        uint64_t match = 1;
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            if (fabs(actual_value - result) > params->fact.threshold){
                match = 0;
                break;
            }
        }
        correct += match;
    }
    free(vm.ram);
    return (double)correct / (double)in->input_num;
}

const struct Fitness STRICT_THRESHOLD_ACCURACY = {.fn = strict_threshold_accuracy, .type = MAXIMIZE, .name = "Strict Threshold Accuracy"};


double f1_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // It counts true positives, false positives and false negatives.
    // The result is the F1 score, which is the harmonic mean of precision and recall.
    // Vectorial version with per-label counting (multilabel classification)
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value && result)
                true_positive++;
            else if (actual_value && (! result))
                false_negative++;
            else if ((! actual_value) && result)
                false_positive++;
        }
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

const struct Fitness F1_SCORE = {.fn = f1_score, .type = MAXIMIZE, .name = "F1 Score"};

double f_beta_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // It counts true positives, false positives and false negatives.
    // The result is the F beta score, which is the weighted harmonic mean of precision and recall.
    // Vectorial version with per-label counting (multilabel classification)
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value && result)
                true_positive++;
            else if (actual_value && (! result))
                false_negative++;
            else if ((! actual_value) && result)
                false_positive++;
        }
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
    double beta2 = params->fact.beta * params->fact.beta;
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall);
}

const struct Fitness F_BETA_SCORE = {.fn = f_beta_score, .type = MAXIMIZE, .name = "F-Beta Score"};

double binary_cross_entropy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a floating point value in [0, 1] representing a probability.
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result)) || result < 0 || result > 1){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double p = result < params->fact.tolerance ? params->fact.tolerance : (result > 1.0 - params->fact.tolerance ? 1.0 - params->fact.tolerance : result);
            bce += -actual_value * log(p) - (1.0 - actual_value) * log(1.0 - p);
        }
    }
    free(vm.ram);
    if(isfinite(bce))
        return bce / (double)(in->input_num * (params->end - params->start));
    else
        return DBL_MAX;
}

const struct Fitness BINARY_CROSS_ENTROPY = {.fn = binary_cross_entropy, .type = MINIMIZE, .name = "Binary Cross Entropy"};

double gaussian_log_likelihood(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value.
    // Vectorial version with uniform average (fields of the output vector are considered independent).
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double sigma_squared = params->fact.sigma * params->fact.sigma;
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
        double error = 0.0;
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double err = result - actual_value;
            error += err * err;
        }
        sum += C - 0.5 * error / sigma_squared;
    }
    free(vm.ram);
    if(isfinite(sum))
        return sum;
    else
        return 0;
}

const struct Fitness GAUSSIAN_LOG_LIKELIHOOD = {.fn = gaussian_log_likelihood, .type = MAXIMIZE, .name = "Gaussian Log Likelihood"};

double brier_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a floating point value in [0, 1] representing a probability.
    // while the real output is a binary value.
    // Vectorial version with multilabel classification
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
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result)) || result < 0 || result > 1){
                free(vm.ram);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double err = result - actual_value;
            sum += err * err;
        }
    }
    free(vm.ram);
    if(isfinite(sum))
        return sum / (double)(in->input_num * (params->end - params->start));
    else
        return DBL_MAX;
}

const struct Fitness BRIER_SCORE = {.fn = brier_score, .type = MINIMIZE, .name = "Brier Score"};


double hinge_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params){
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // Vectorial version with multilabel classification
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            sum += fmax(0.0, 1.0 - actual_value * result);
        }
    }
    free(vm.ram);
    if(isfinite(sum))
        return sum / (double)(in->input_num * (params->end - params->start));
    else
        return DBL_MAX;
}

const struct Fitness HINGE_LOSS = {.fn = hinge_loss, .type = MINIMIZE, .name = "Hinge Loss"};

double matthews_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // Vectorial version with per-label counting (multilabel classification)
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value && result)
                true_positive++;
            else if (actual_value && (! result))
                false_negative++;
            else if ((! actual_value) && result)
                false_positive++;
            else
                true_negative++;
        }
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

const struct Fitness MATTHEWS_CORRELATION = {.fn   = matthews_correlation, .type = MAXIMIZE, .name = "Matthews Correlation"};

double balanced_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // Vectorial version with per-label counting (multilabel classification)
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value && result)
                true_positive++;
            else if (actual_value && (! result))
                false_negative++;
            else if ((! actual_value) && result)
                false_positive++;
            else
                true_negative++;
        }
    }
    free(vm.ram);
    double sensitivity = (true_positive + false_negative) ? (double) true_positive / (double) (true_positive + false_negative) : 0.0;
    double specificity = (true_negative + false_positive) ? (double) true_negative / (double) (true_negative + false_positive) : 0.0;
    return 0.5 * (sensitivity + specificity);
}

const struct Fitness BALANCED_ACCURACY = {.fn = balanced_accuracy, .type = MAXIMIZE, .name = "Balanced Accuracy"};


double g_mean(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // Vectorial version with per-label counting (multilabel classification)
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value && result)
                true_positive++;
            else if (actual_value && (! result))
                false_negative++;
            else if ((! actual_value) && result)
                false_positive++;
            else
                true_negative++;
        }
    }
    free(vm.ram);
    double sensitivity = (true_positive + false_negative) ? (double) true_positive / (double) (true_positive + false_negative) : 0.0;
    double specificity = (true_negative + false_positive) ? (double) true_negative / (double) (true_negative + false_positive) : 0.0;
    return sqrt(sensitivity * specificity);
}

const struct Fitness G_MEAN = {.fn = g_mean, .type = MAXIMIZE, .name = "G-mean"};

double cohens_kappa(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a zero or positive integer for true and a negative integer for false.
    // Vectorial version with per-label counting (multilabel classification)
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
        for(uint64_t j = params->start; j < params->end; j++){
            uint64_t result = vm.ram[j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].i64 & (((uint64_t) 1) << ((uint64_t) 63));
            if (actual_value && result)
                true_positive++;
            else if (actual_value && (! result))
                false_negative++;
            else if ((! actual_value) && result)
                false_positive++;
            else
                true_negative++;
        }
    }
    free(vm.ram);
    double observed_agreement = (double)(true_positive + true_negative) / (double)in->input_num;
    double expected_agreement = ((double)(true_positive + false_positive) / (double)in->input_num) * ((double)(true_positive + false_negative) / (double)in->input_num)
                                + ((double)(false_positive + true_negative) / (double)in->input_num) * ((double)(false_negative + true_negative) / (double)in->input_num);
    if(expected_agreement == 1.0)
        return DBL_MAX;

    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement);
}

const struct Fitness COHENS_KAPPA = {.fn = cohens_kappa, .type = MAXIMIZE, .name = "Cohen's Kappa"};

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

const struct Fitness ADVERSARIAL_PERTURBATION_SENSIBILITY = {.fn = adversarial_perturbation_sensibility, .type = MINIMIZE, .name = "Adversarial Perturbation Sensibility"};

static inline int compare_doubles(const void *a, const void *b) {
    double diff = (*(double *)a - *(double *)b);
    return (diff > 0) - (diff < 0);
}


double conditional_value_at_risk(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with vectorial distance of the output as error
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t count = (uint64_t) ceil(params->fact.alpha * in->input_num);
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
        results[i] = 0.0;
        for(uint64_t j = params->start; j < params->end; j++){
            double result = vm.ram[j].f64;
            if (!(isfinite(result))){
                free(vm.ram);
                free(results);
                return DBL_MAX;
            }
            double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size + j].f64;
            double diff = result - actual_value;
            results[i] += diff * diff;
        }
    }
    free(vm.ram);
    qsort(results, in->input_num, sizeof(double), compare_doubles);
    double error = 0.0;
    for (uint64_t i = in->input_num - count; i < in->input_num; i++) {
        error += (results[i] / ((double)params->end - (double)params->start));
    }
    free(results);
    if (!(isfinite(error)))
        return DBL_MAX;
    return error / count;
}

const struct Fitness CONDITIONAL_VALUE_AT_RISK = {.fn = conditional_value_at_risk, .type = MINIMIZE, .name = "Conditional Value at Risk"};