#include "genetics.h"


void print_program(const struct Program* prog){
	printf("\n");
	for (uint64_t i = 0; i < prog->size; i++) {
		printf("%s ", INSTRSET[prog->content.op].name);
        for(uint64_t i = 0; i < INSTRSET[prog->content.op].regs; i++){
            printf("REG-%d ", prog->content.reg[i]);
        }
        if(INSTRSET[prog->content.op].addr){
            printf("ADDR-%d ", prog->content.reg[i]);
        }
		printf("\n");
	}
}

double mse(const struct LGPInput in, const struct Program* prog, uint64_t max_clock){
    if (individ->dna_len == 0)
		return DBL_MAX;
    struct VirtualMachine vm;
    vm.program = prog;
    vm.vmem = malloc(sizeof(union memblock) * in.mem_size);
    double mse = 0;
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memcpy(vm.vmem, in->memories[(in->mem_size + in->res_size)* i], in.mem_size* sizeof(union memblock));
        run_vm(&vm, max_clock);
        double result = vm.vmem[in.result_addr];
        if (!(isfinite(result))){
            free_env(actual_env);
            free(vm.vmem);
			return DBL_MAX;
        }
		double actual_mse = in->memories[(in->mem_size + in->res_size)* i + in->mem_size].f64 - result;
		mse += (actual_mse * actual_mse);
    }
    free(vm.vmem);
    if(isfinite(mse))
		return mse / (double)in->input_size;
	else
		return DBL_MAX;
}

inline void population_fitness(const struct LGPInput* in, const struct Population *pop, const length_t already_calc, fitness_fn fitness_func){
    #pragma omp parallel for schedule(dynamic,1)
        for (length_t i = already_calc; i < pop->size; i++) {
            pop->individ[i].fitness = fitness_func(in, &pop->individuals[i]);
        }
    }