#include <stdio.h>
#include "genetics.h"
#include "psb2.h"

int main(int argc, char *argv[]){

	const struct LGPOptions par = {
		.fitness = MSE,
		.selection_func = tournament,
		.select_param = (union SelectionParams) {.size = 3},
		.initialization_func = rand_population,
		.init_params = (struct InitializationParams) {
			.pop_size = 400,
			.minsize = 5,
			.maxsize = 20
		},
		.target = 1e-27,
		.mutation_prob = 0.33,
		.max_mutation_len = 10,
		.crossover_prob = 0.69,
		.max_clock = 2200,
		.max_individ_len = MAX_PROGRAM_SIZE,
		.generations = 300,
		.verbose = 1
	};
	struct InstructionSet instr_set = (struct InstructionSet) {.size = INSTR_NUM, .op = INSTRSET,};
	struct LGPInput in = vector_distance(&instr_set, 2, 100);
	const struct LGPResult res = evolve(&in, &par);
	free(in.memory);
	free(res.pop.individual);
	return 0;
}

