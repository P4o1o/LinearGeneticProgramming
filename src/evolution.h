#ifndef EVOLUTION_H_INCLUDED
#define EVOLUTION_H_INCLUDED

#include "genetics.h"
#include "selection.h"
#include "creation.h"

struct LGPOptions {
	const struct FitnessAssesment fitness;	// fitness function for program evaluation
	const struct Selection selection; // selection function to be used
	const initialization_fn initialization_func; // function for create an initial Population
	const union SelectionParams select_param; // parameters for the selection function
	const struct InitializationParams init_params; // parameters for the initialization function
	const struct Population initial_pop; // if initialization_func == NULL then start with initial_pop
	const double target; // the evolution stops if tollerance > mse of the best individual
	const double mutation_prob; // mutation propability
	const double crossover_prob; // crossover propability
    const uint64_t max_clock;
    const uint64_t max_individ_len;
	const uint64_t max_mutation_len; // maximum lenght of the new cromosomes added by the mutation
	const uint64_t generations; // maximum generation of execution
	const unsigned verbose; // if 0 doesn't print anything else for every generations print "number of generation, best individual's mse, Population size and number of evaluations"
};

struct LGPResult evolve(const struct LGPInput *const in, const struct LGPOptions *const args);

void print_program(const struct Program *const prog);

// used in crossover
struct ProgramCouple{
	struct Program prog[2];
};

#endif
