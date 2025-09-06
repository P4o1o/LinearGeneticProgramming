#ifndef GENETICS_H_INCLUDED
#define GENETICS_H_INCLUDED

#include "prob.h"
#include "vm.h"
#include "logger.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern const uint64_t VECT_ALIGNMENT_WRAPPER;

struct Program{
	struct Instruction *content;
    uint64_t size;
};

struct Individual{
	struct Program prog;
    double fitness;
};

struct Population{
    struct Individual *individual;
    uint64_t size;
};

struct MultiIndividual{
	struct Program prog;
    double *fitness;
};

struct MultiPopulation{
	struct MultiIndividual *individual;
    uint64_t size;
	uint64_t fitness_size;
};

struct InstructionSet{
	const uint64_t size;
	const struct Operation *op;
};

struct LGPInput{
	const uint64_t input_num;
	const uint64_t rom_size; // size of the problem data in each input
	const uint64_t res_size; // size of the solution in each input
	const uint64_t ram_size; // size of the RAM, must be >= res_size
	const struct InstructionSet instr_set;
	union Memblock *restrict memory; //problem1, solution1, problem2, solution2, problem3, ...
};

struct LGPResult{
	const struct Population pop; // resulting pupulation
	const uint64_t evaluations; // number of the total evaluations done in this evolution
	const uint64_t generations; // number of generations the evolution has done
	const uint64_t best_individ; // index of the best individual in the Population
};

struct LGPMultiResult{
	const struct MultiPopulation pop; // resulting pupulation
	const uint64_t evaluations; // number of the total evaluations done in this evolution
	const uint64_t generations; // number of generations the evolution has done
	const uint64_t best_individ; // index of the best individual in the Population
};

struct Instruction rand_instruction(const struct LGPInput *const in, const uint64_t prog_size);

#define HASH_SEED 0x5ab26229f0294a21ULL

unsigned int equal_program(const struct Program *const prog1, const struct Program *const prog2);
uint64_t xxhash_program(const struct Program *const prog);
void free_individual(struct Individual *ind);
void free_population(struct Population *pop);
void free_lgp_input(struct LGPInput *in);

#endif
