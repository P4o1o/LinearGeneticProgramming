#ifndef CREATIONS_H_INCLUDED
#define CREATIONS_H_INCLUDED

#include "genetics.h"

typedef struct genetic_result (*initialization)(const struct genetic_input* in, const length_t pop_size, const length_t dna_minsize, const length_t dna_maxsize);

struct genetic_result unique_population(const struct genetic_input* in, const length_t pop_size, const length_t dna_minsize, const length_t dna_maxsize);
struct genetic_result rand_population(const struct genetic_input* in, const length_t pop_size, const length_t dna_minsize, const length_t dna_maxsize);


// used in unique_population

struct evaluated_set_node{
	struct individual ind;
	uint64_t hash;
	double mse;
	struct evaluated_set_node *next;
};

struct evaluated_set{
	struct evaluated_set_node **list;
	length_t capacity;
	length_t size;
};

// for unreachable code (used only in hash_individ)
#ifdef __GNUC__
	#define UNREACHABLE __builtin_unreachable()
#else
#ifdef _MSC_VER
	#define UNREACHABLE __assume(0);
#else
	[[noreturn]] inline void unreachable(){}
	#define UNREACHABLE unreachable()
#endif
#endif

union doubletouint{
    double d;
    uint64_t u;
};

#endif