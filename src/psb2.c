#include "psb2.h"


struct LGPInput vector_distance(const struct InstructionSet *const instr_set, const uint64_t vector_len, const uint64_t instances){
	uint64_t input_num = instances;
	uint64_t rom_size = vector_len * 2;
    uint64_t block_size = rom_size + 1;
	union Memblock *memory = malloc(input_num * block_size * sizeof(union Memblock));
#pragma omp parallel for schedule(static,1) num_threads(NUMBER_OF_OMP_THREADS)
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
    struct LGPInput res ={.memory = memory, .rom_size = rom_size, .input_num = input_num, .res_size = 1, .instr_set = *instr_set, .ram_size = 1};
	return res;
}


struct LGPInput bouncing_balls(const struct InstructionSet *const instr_set, const uint64_t instances){
	uint64_t input_num = instances;
	uint64_t rom_size = 3;
	uint64_t block_size = rom_size + 1;
	union Memblock *memory = malloc(input_num * block_size * sizeof(union Memblock));
#pragma omp parallel for schedule(static,1) num_threads(NUMBER_OF_OMP_THREADS)
	for (uint64_t i = 0; i < input_num; i++) {
		memory[i * block_size].f64 = RAND_DBL_BOUNDS(1.01, 100.0); // first bounce
		memory[i * block_size + 1].f64 = RAND_DBL_BOUNDS(1.00, 100.0); // second bounce
		memory[i * block_size + 2].i64 = ((uint64_t)RAND_BOUNDS(1, 20)); // number of bounces;
		memory[i * block_size + 3].f64 = memory[i * block_size].f64; // result
		double r = memory[i * block_size + 1].f64 / memory[i * block_size].f64;
		for (uint64_t j = 1; j < memory[i * block_size + 2].i64; j++) {
			memory[i * block_size + 3].f64 += 2.0 * pow(r, j) * memory[i * block_size].f64;
		}
	}
	struct LGPInput res = {.memory = memory, .rom_size = rom_size, .input_num = input_num, .res_size = 1, .instr_set = *instr_set, .ram_size = 1};
	return res;
}

struct LGPInput dice_game(const struct InstructionSet *const instr_set, const uint64_t instances){
	// https://www.karlin.mff.cuni.cz/~nagy/NMSA202/dice1.pdf pg.70
	uint64_t input_num = instances;
	uint64_t rom_size = 2;
	uint64_t block_size = rom_size + 1;
	union Memblock *memory = malloc(input_num * block_size * sizeof(union Memblock));
#pragma omp parallel for schedule(static,1) num_threads(NUMBER_OF_OMP_THREADS)
	for (uint64_t i = 0; i < input_num; i++) {
		memory[i * block_size].i64 = RAND_BOUNDS(1, 1000); // first bounce
		memory[i * block_size + 1].i64 = RAND_BOUNDS(1, 1000); // second bounce
		memory[i * block_size + 2].i64 = (memory[i * block_size].i64 > memory[i * block_size + 1].i64) ? 1.0 - ((memory[i * block_size + 1].i64 + 1.0) / (2.0 * memory[i * block_size].i64)) : (memory[i * block_size].i64 - 1.0) / (2.0 * memory[i * block_size + 1].i64); // result
	}
	struct LGPInput res = {.memory = memory, .rom_size = rom_size, .input_num = input_num, .res_size = 1, .instr_set = *instr_set, .ram_size = 1};
	return res;
}

struct LGPInput shopping_list(const struct InstructionSet *const instr_set, const uint64_t num_of_items, const uint64_t instances){
	uint64_t input_num = instances;
	uint64_t rom_size = 2 * num_of_items;
	uint64_t block_size = rom_size + 1;
	union Memblock *memory = malloc(input_num * block_size * sizeof(union Memblock));
#pragma omp parallel for schedule(static,1) num_threads(NUMBER_OF_OMP_THREADS)
	for (uint64_t i = 0; i < input_num; i++) {
		memory[i * block_size + rom_size].f64 = 0.0; // result
		for (uint64_t j = 0; j < rom_size; j += 2) {
			memory[i * block_size + j].f64 = RAND_DBL_BOUNDS(1.0, 50.0); // price of item
			memory[i * block_size + j + 1].f64 = RAND_DBL_BOUNDS(1.0, 100.0); // scount of item
			memory[i * block_size + rom_size].f64 += memory[i * block_size + j].f64 * memory[i * block_size + j + 1].f64 / 100.0; // total price
		}
		
	}
	struct LGPInput res = {.memory = memory, .rom_size = rom_size, .input_num = input_num, .res_size = 1, .instr_set = *instr_set, .ram_size = 1};
	return res;
}

struct LGPInput snow_day(const struct InstructionSet *const instr_set, const uint64_t instances){
	uint64_t input_num = instances;
	uint64_t rom_size = 4;
	uint64_t block_size = rom_size + 1;
	union Memblock *memory = malloc(input_num * block_size * sizeof(union Memblock));
#pragma omp parallel for schedule(static,1) num_threads(NUMBER_OF_OMP_THREADS)
	for (uint64_t i = 0; i < input_num; i++) {
		memory[i * block_size].i64 = RAND_BOUNDS(0, 20); // hours of snow
		memory[i * block_size + 1].f64 = RAND_DBL_BOUNDS(0.0, 20.0); // snow in the ground
		memory[i * block_size + 2].f64 = RAND_DBL_BOUNDS(0.0, 10.0); // snow fall rate
		memory[i * block_size + 3].f64 = RAND_DBL_BOUNDS(0.0, 1.0); // snow melting rate
		memory[i * block_size + 4].f64 = memory[i * block_size + 1].f64; // result
		for (uint64_t j = 1; j < memory[i * block_size].i64; j++) {
			memory[i * block_size + 4].f64 += memory[i * block_size + 2].f64;
			memory[i * block_size + 4].f64 *= (1.0 - memory[i * block_size + 3].f64);
		}
	}
	struct LGPInput res = {.memory = memory, .rom_size = rom_size, .input_num = input_num, .res_size = 1, .instr_set = *instr_set, .ram_size = 1};
	return res;
}
