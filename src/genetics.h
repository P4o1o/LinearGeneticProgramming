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

#define MAX_PROGRAM_SIZE 255

struct Program{
    struct Instruction *content[MAX_PROGRAM_SIZE];
    uint64_t size;
    double fitness;
};

void print_program(const struct Program* prog);

struct Population{
    struct Program individ;
    uint64_t size;
};

struct LGPInput{
	uint64_t input_num;
	uint64_t mem_size;
	uint64_t res_size;
	union memblock *memories;
};

typedef double (*fitness_fn)(const struct LGPInput in, const struct Program* prog);

struct LGPResult{
	struct Population pop; // resulting pupulation
	uint64_t evaluations; // number of the total evaluations done in this evolution
	uint64_t generations; // number of generations the evolution has done
	uint64_t best_individ; // index of the best individual in the population
};

typedef struct genetic_result (*initialization)(const struct LGPInput* in, const length_t pop_size, const length_t dna_minsize, const length_t dna_maxsize);

struct FitnessSharingParams{ // parameters for selections based on fitness sharing
    double alpha;
    double beta;
    double sigma;
    length_t size;
};

union SelectionParams{
    length_t size; // size of the tournament, size of the elite for elitism, size of sampling in roulette_selection
    double val; // percentual_elitism
    struct FitnessSharingParams fs_params; // fitness_sharing_tournament, fitness_sharing_roulette
};

typedef struct population(*selection)(struct Population*, const union SelectionParams*);

struct LGPOptions {
	fitness_fn fitness_func	// fitness function for program evaluation
	selection select_type; // selection function to be used
	initialization init_type; // function for create an initial population
	double tollerance; // the evolution stops if tollerance > mse of the best individual
	double mutation_prob; // mutation propability
	double crossover_prob; // crossover propability
	union SelectionParams select_param; // parameters for the selection function
    uint64_t max_clock;
	uint64_t max_mutation_len; // maximum lenght of the new cromosomes added by the mutation
	uint64_t initial_pop_size; // size of the initial population
	uint64_t min_init_size; // minimum size of an individual generated in the initial population
	uint64_t max_init_size; // maximum size of an individual generated in the initial population
	uint64_t generations; // maximum generation of execution
	unsigned verbose; // if 0 doesn't print anything else for every generations print "number of generation, best individual's mse, population size and number of evaluations"
};

struct LGPResult evolve(const struct LGPInput* in, const struct LGPOptions* args);

#endif