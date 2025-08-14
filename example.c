#include <stdio.h>
#include "src/evolution.h"
#include "src/psb2.h"
#include "src/prob.h"
 

static inline double get_time_sec(void) {
	#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L // >= C11
		struct timespec ts;
		if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
			return ts.tv_sec + ts.tv_nsec * 1e-9;
		}
	#endif
	return (double)clock() / (double)CLOCKS_PER_SEC;    
}

int main(UNUSED_ATTRIBUTE int argc, UNUSED_ATTRIBUTE char *argv[]){
	random_init(0x47afeb91, 0);
	for(uint64_t i = 0; i < NUMBER_OF_OMP_THREADS; i++){
		uint32_t seed = random();
		printf("seed %ld: %0x\n", i, seed);
		random_init(seed, i);
	}
	const struct LGPOptions par = {
		.fitness = MSE,
		.fitness_param = (struct FitnessParams) {
			.start = 0,
			.end = 1,
		},
		.selection = tournament,
		.select_param = (union SelectionParams) {.size = 3},
		.initialization_func = unique_population,
		.init_params = (struct InitializationParams) {
			.pop_size = 1000,
			.minsize = 2,
			.maxsize = 5
		},
		.target = 1e-27,
		.mutation_prob = 0.76,
		.max_mutation_len = 5,
		.crossover_prob = 0.95,
		.max_clock = 5000,
		.max_individ_len = 50,
		.generations = 10,
		.verbose = 1
	};
	struct Operation opset[18] = {
		OP_ADD_F, OP_SUB_F, OP_MUL_F, OP_DIV_F, OP_POW, OP_LOAD_ROM_F, OP_STORE_RAM_F, OP_MOV_F,
		OP_JMP, OP_JMP_Z, OP_JMP_NZ, OP_JMP_G, OP_JMP_L, OP_CMP, OP_TEST, OP_CLC,
		OP_JMP_NEXIST, OP_JMP_EXIST
	};
	struct InstructionSet instr_set = (struct InstructionSet) {
		.size = 18, .op = opset,
	};
	struct LGPInput in = vector_distance(&instr_set, 2, 100);
	double start = get_time_sec();
	const struct LGPResult res = evolve(&in, &par);
	double end = get_time_sec();
	free(in.memory);
	printf("Solution:\n");
	print_program(&(res.pop.individual[res.best_individ].prog));
	printf("Time: %lf, evaluations: %lu, eval/sec: %lf\n", end - start, res.evaluations, ((double) res.evaluations) / (end - start));
	aligned_free(res.pop.individual);
	return 0;
}
