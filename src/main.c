#include <stdio.h>
#include "evolution.h"
#include "psb2.h"
 

static inline double get_time_sec() {
	#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L // >= C11
		struct timespec ts;
		if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
			return ts.tv_sec + ts.tv_nsec * 1e-9;
		}
	#endif
	return (double)clock() / (double)CLOCKS_PER_SEC;    
}

int main(int argc, char *argv[]){
	random_init(7, 0);
	for(uint64_t i = 0; i < MAX_OMP_THREAD; i++){
		uint32_t seed = random();
		printf("seed %ld: %0x\n", i, seed);
		random_init(seed, i);
	}
	const struct LGPOptions par = {
		.fitness = MSE,
		.selection = elitism,
		.select_param = (union SelectionParams) {.size = 5000},
		.initialization_func = unique_population,
		.init_params = (struct InitializationParams) {
			.pop_size = 10000,
			.minsize = 5,
			.maxsize = 20
		},
		.target = 1e-27,
		.mutation_prob = 1.0,
		.max_mutation_len = 10,
		.crossover_prob = 1.0,
		.max_clock = 2200,
		.max_individ_len = MAX_PROGRAM_SIZE,
		.generations = 300,
		.verbose = 1
	};
	struct Operation opset[9] = {OP_ADD_F, OP_SUB_F, OP_MUL_F, OP_DIV_F, OP_SQRT, OP_LOAD_RAM_F, OP_LOAD_ROM_F, OP_STORE_RAM_F, OP_MOV_F};
	struct InstructionSet instr_set = (struct InstructionSet) {
		.size = 9, .op = opset,
	};
	struct LGPInput in = vector_distance(&instr_set, 2, 100);
	double start = get_time_sec();
	const struct LGPResult res = evolve(&in, &par);
	double end = get_time_sec();
	free(in.memory);
	printf("Solution:\n");
	print_program(&(res.pop.individual[res.best_individ].prog));
	printf("Time: %lf, evaluations: %lu, eval/sec: %lf\n", end - start, res.evaluations, ((double) res.evaluations) / (end - start));
	free(res.pop.individual);
	return 0;
}
