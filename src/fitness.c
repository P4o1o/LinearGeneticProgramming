#include "fitness.h"

inline double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mse = 0.0;
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * RAM_SIZE);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result)))
            return DBL_MAX;
        double actual_mse = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
        mse += (actual_mse * actual_mse);
    }
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

double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mae = 0.0;
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * RAM_SIZE);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result)))
            return DBL_MAX;
        double actual_mae = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
        mae += fabs(actual_mae);
    }
    if(isfinite(mae))
        return mae / (double)in->input_num;
    else
        return DBL_MAX;
}

const struct FitnessAssesment MAE = {.fn = mae, .type = MINIMIZE, .name = "MAE"};

double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double ss_res = 0.0;
    double ss_tot = 0.0;
    double mean = 0.0;

    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * RAM_SIZE);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result)))
            return DBL_MAX;
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        mean += actual_value;
        ss_res += (actual_value - result) * (actual_value - result);
    }
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

double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t correct = 0;
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * RAM_SIZE);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        uint64_t result = vm.ram[0].i64;
        uint64_t actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].i64;
        if (actual_value == result)
            correct++;
    }
    return (double)correct / (double)in->input_num;
}

const struct FitnessAssesment ACCURACY = {.fn = accuracy, .type = MAXIMIZE, .name = "Accuracy"};

double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params){
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    uint64_t correct = 0;
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * RAM_SIZE);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        double actual_value = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64;
        if (fabs(actual_value - result) <= params->threshold)
            correct++;
    }
    return (double)correct / (double)in->input_num;
}

const struct FitnessAssesment THRESHOLD_ACCURACY = {.fn = threshold_accuracy, .type = MAXIMIZE, .name = "Threshold Accuracy"};