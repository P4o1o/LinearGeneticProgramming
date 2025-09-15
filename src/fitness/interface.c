#include "interface.h"
#include "../macros.h"

double *eval_multifitness(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct MultiFitness * const fitness){
    ASSERT(prog->size > 0);
    ASSERT(in->ram_size > 0);
    ASSERT(in->input_num > 0);
    ASSERT(fitness->size > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE(sizeof(union Memblock) * in->ram_size);
    }
    uint64_t result_size = fitness->params->end - fitness->params->start;
    ASSERT(result_size <= in->ram_size);
    union FitnessStepResult *accumulator = malloc(sizeof(union FitnessStepResult) * fitness->size);
    if (accumulator == NULL) {
        free(vm.ram);
        MALLOC_FAIL_THREADSAFE(sizeof(union FitnessStepResult) * fitness->size);
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
            if(! fitness->functions[j].combine(&accumulator[j], &step_res, clocks, i, &fitness->params[j])){
                break;
            }
        }
    }
    free(vm.ram);
    double *results = malloc(sizeof(double) * fitness->size);
    if (results == NULL) {
        MALLOC_FAIL_THREADSAFE(sizeof(double) * fitness->size);
    }
    for (uint64_t j = 0; j < fitness->size; j++) {
        results[j] = fitness->functions[j].finalize(&accumulator[j], in, result_size, prog->size, in->input_num, &fitness->params[j]);
    }
    free(accumulator);
    return results;
}

void free_distance_table(struct FitnessParams *const params) {
    free(params->fact.clustering.distance_table);
}
