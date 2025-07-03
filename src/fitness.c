#include "fitness.h"

double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock){
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

const struct FitnessAssesment MSE = {.fn = mse, .type = MINIMIZE};
