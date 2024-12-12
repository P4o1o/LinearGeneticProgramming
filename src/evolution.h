#ifndef EVOLUTION_H_INCLUDED
#define EVOLUTION_H_INCLUDED

#include "genetics.h"
#include "selections.h"
#include "creations.h"

struct genetic_options {
	selection select_type; // selection function to be used
	initialization init_type; // function for create an initial population
	double tollerance; // the evolution stops if tollerance > mse of the best individual
	double mutation_prob; // mutation propability
	double crossover_prob; // crossover propability
	union selection_params select_param; //parameters for the selection function
	length_t evolution_cycles; // how many times in a single generation mutation and crossover operation shold be tried
	length_t mut_max_len; // maximum lenght of the new cromosomes added by the mutation
	length_t initial_pop_size; // size of the initial population
	length_t dna_minsize; // minimum size of an individual generated in the initial population
	length_t dna_maxsize; // maximum size of an individual generated in the initial population
	length_t generations; // maximum generation of execution
	unsigned verbose; // if 0 doesn't print anything else for evry generations print "number of generation, best individual's mse, population size and number of evaluations"
};

struct genetic_result evolve(const struct genetic_input* in, const struct genetic_options* args);

#endif