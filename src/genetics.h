#ifndef GENETICS_H_INCLUDED
#define GENETICS_H_INCLUDED

#include "prob.h"
#include "vm.h"
#include "logger.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#define MAX_PROGRAM_SIZE 254

union InstrToU64{
	const struct Instruction instr;
	const uint64_t u64;
};

struct Program{
    struct Instruction content[MAX_PROGRAM_SIZE + 1];
    uint64_t size;
};

struct ProgramCouple{
	struct Program prog[2];
};

struct Individual{
	struct Program prog;
    double fitness;
};

void print_program(const struct Program* prog);

struct Population{
    struct Individual *individual;
    uint64_t size;
};

struct LGPInput{
	const uint64_t input_num;
	const uint64_t rom_size;
	const uint64_t res_size;
	const uint64_t op_size;
	const struct Operation *op;
	const union ConstMemblock *memory;
};

typedef double (*fitness_fn)(const struct LGPInput *const, const struct Program *const, const uint64_t);

double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock);

enum FitnessType{
	MINIMIZE,
	MAXIMIZE
};

struct FitnessAssesment{
	fitness_fn fn;
	enum FitnessType type;
};

extern const struct FitnessAssesment MSE;

struct LGPResult{
	const struct Population pop; // resulting pupulation
	const uint64_t evaluations; // number of the total evaluations done in this evolution
	const uint64_t generations; // number of generations the evolution has done
	const uint64_t best_individ; // index of the best individual in the Population
};

struct InitializationParams{
	const uint64_t pop_size; // size of the initial Population
	const uint64_t minsize; // minimum size of a program generated in the initialization_func
	const uint64_t maxsize; // maximum size of a program generated in the initialization_func
};

typedef struct LGPResult (*initialization_fn)(const struct LGPInput *const, const struct InitializationParams *const, const struct FitnessAssesment *const, const uint64_t);

// used in unique_population

struct PrgTableNode{
	const struct Program prog;
	const uint64_t hash;
};

struct ProgramTable{
	struct PrgTableNode *table;
	const uint64_t size;
};
struct LGPResult unique_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct FitnessAssesment *const fitness, const uint64_t max_clock);
struct LGPResult rand_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct FitnessAssesment *const fitness, const uint64_t max_clock);

struct FitnessSharingParams{ // parameters for selections based on fitness sharing
    const double alpha;
    const double beta;
    const double sigma;
    const uint64_t size;
};

union SelectionParams{
    const uint64_t size; // size of the tournament, size of the elite for elitism, size of sampling in roulette_selection
    const double val; // percentual_elitism
    const struct FitnessSharingParams fs_params; // fitness_sharing_tournament, fitness_sharing_roulette
};

typedef void (*selection_fn)(struct Population*, const union SelectionParams*, const enum FitnessType);

void tournament(struct Population* initial, const union SelectionParams* tourn_size, const enum FitnessType ftype);
void elitism(struct Population* initial, const union SelectionParams* new_size, const enum FitnessType ftype);
void percentual_elitism(struct Population* initial, const union SelectionParams *elite_size, const enum FitnessType ftype);
void roulette_selection(struct Population* initial, const union SelectionParams* elite_size, const enum FitnessType ftype);
void fitness_sharing_tournament(struct Population* initial, const union SelectionParams *params, const enum FitnessType ftype);
void fitness_sharing_roulette(struct Population* initial, const union SelectionParams *params, const enum FitnessType ftype);
void fitness_sharing_elitism(struct Population* initial, const union SelectionParams *params, const enum FitnessType ftype);


struct LGPOptions {
	const struct FitnessAssesment fitness;	// fitness function for program evaluation
	const selection_fn selection_func; // selection function to be used
	const initialization_fn initialization_func; // function for create an initial Population
	const union SelectionParams select_param; // parameters for the selection function
	const struct InitializationParams init_params; // parameters for the initialization function
	const struct Population initial_pop; // if initialization_func == NULL then start with initial_pop
	const double tollerance; // the evolution stops if tollerance > mse of the best individual
	const double mutation_prob; // mutation propability
	const double crossover_prob; // crossover propability
    const uint64_t max_clock;
    const uint64_t max_individ_len;
	const uint64_t max_mutation_len; // maximum lenght of the new cromosomes added by the mutation
	const uint64_t generations; // maximum generation of execution
	const unsigned verbose; // if 0 doesn't print anything else for every generations print "number of generation, best individual's mse, Population size and number of evaluations"
};

struct LGPResult evolve(const struct LGPInput *const in, const struct LGPOptions *const args);

#endif