#include "psb2.h"


struct LGPInput vector_distance(const struct InstructionSet *const instr_set, const uint64_t vector_len, const uint64_t instances){
	uint64_t input_num = instances;
	uint64_t rom_size = vector_len * 2;
    uint64_t block_size = rom_size + 1;
	union Memblock *memory = malloc(input_num * block_size * sizeof(union Memblock));
#pragma omp parallel for schedule(static,1) num_threads(MAX_OMP_THREAD)
	for (uint64_t i = 0; i < input_num; i++) {
        memory[i * block_size + rom_size].f64 = 0.0;
		for (uint64_t j = 0; j < vector_len; j++) {
			memory[i * block_size + j].f64 = RAND_DBL_BOUNDS(-100.0, 100.0);
			memory[i * block_size + vector_len + j].f64 = RAND_DBL_BOUNDS(-100.0, 100.0);
			double diff = memory[i * block_size + j].f64 - memory[i * block_size + vector_len + j].f64;
			memory[i * block_size + rom_size].f64 += diff * diff;
		}
		memory[i * block_size + rom_size].f64 = sqrt(memory[i * block_size + rom_size].f64);
	}
    struct LGPInput res ={.memory = memory, .rom_size = rom_size, .input_num = input_num, .res_size = 1, .instr_set = *instr_set};
	return res;
}
