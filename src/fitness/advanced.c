#include "advanced.h"
#include "regression.h"  // For shared functions
#include "../macros.h"

// ADVANCED INIT_ACC IMPLEMENTATIONS

inline union FitnessStepResult vect_f64_init_acc(UNUSED_ATTRIBUTE const uint64_t inputnum, const uint64_t ressize, UNUSED_ATTRIBUTE const struct FitnessParams *const params) {
    union FitnessStepResult result =  {
        .vect_f64 = malloc(ressize * sizeof(double))
    };
    if (result.vect_f64 == NULL) {
        MALLOC_FAIL_THREADSAFE(sizeof(double) * ressize);
    }
    memset(result.vect_f64, 0, ressize * sizeof(double));
    return result;
}

// ADVANCED FINALIZE FUNCTION IMPLEMENTATIONS

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

// ADVANCED FITNESS FUNCTION IMPLEMENTATIONS

double conditional_value_at_risk(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with vectorial distance of the output as error
    return eval_fitness(in, prog, max_clock, params, quadratic_error, sum_float, value_at_risk_finalize, vect_f64_init_acc);
}

double adversarial_perturbation_sensitivity(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params) {
    // This fitness function assumes that the program outputs a double value
    // Vectorial version with vectorial distance of the output as error
    ASSERT(prog->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    double sum_error = 0.0;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    union Memblock *altered_rom = malloc(sizeof(union Memblock) * in->rom_size);
    if (vm.ram == NULL || altered_rom == NULL) {
        MALLOC_FAIL_THREADSAFE(sizeof(union Memblock) * in->ram_size + sizeof(union Memblock) * in->rom_size);
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

// ADVANCED FITNESS STRUCT DEFINITIONS

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

const struct Fitness ADVERSARIAL_PERTURBATION_SENSITIVITY = {
    .fn = adversarial_perturbation_sensitivity,
    .type = MINIMIZE,
    .name = "Adversarial Perturbation Sensitivity",
    .data_type = FITNESS_FLOAT,
    .step = NULL,
    .combine = NULL,
    .finalize = NULL,
    .init_acc = NULL
};
