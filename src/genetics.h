#ifndef GENETICS_H_INCLUDED
#define GENETICS_H_INCLUDED

#include "prob.h"
#include "operations.h"
#include "logger.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

typedef uint32_t length_t;

typedef uint16_t op_type;

struct cromosome {
	union argtype args; // arguments, see operations.h
	env_index res; // index on witch the result will be stored
	op_type op; // index of the operation in the genetic_env 
};

struct individual {
	struct cromosome* dna; // list of instructions
	length_t dna_len; // list length
};

struct genetic_env{
	struct operation *ops; // list of operations
    op_type ops_size; // size of the list of operations
	env_index env_size; // size of the virtual enviroment in witch the results will be stored
};

struct genetic_env simple_genv(const env_index size); // with +, -, *, safe_div

void print_individual(const struct genetic_env* genv, const struct individual* individ);

struct individual remove_trash(const struct genetic_env *genv, const struct individual* ind); // Function that eliminate instructions that uneffects the output

double predict(const struct genetic_env *genv, const struct individual *individ, const double *X, const env_index x_len);

struct population{
	struct individual *individuals;
	length_t size;
};

void free_population(struct population* pop);

struct single_input {
	double* x; // input vector for an individual evaluation
	double y; // correct output that should be generated
};

struct genetic_input {
	struct genetic_env genv;
	struct single_input *data;
	length_t input_size; // how many single_input to use
	env_index x_len; // lenght of every x in single_input
};

void free_genetic_input(struct genetic_input* in);

double get_mse(const struct genetic_input* in, const struct individual *individ);

void mse_population(const struct genetic_input* in, const struct population *pop, double **mse, const length_t already_calc);

struct individual extract_best(const struct genetic_input* in, const struct population* pop);


struct genetic_result{
	struct population pop; // resulting pupulation
	double *mse; // list of mse, pop[i] has mse[i] mse
	uint64_t evaluations; // number of the total evaluations done in this evolution
	length_t generations; // number of generations the evolution has done
	length_t best_individ; // index of the best individual in the population
};

void free_genetic_result(struct genetic_result *res);

#endif