#ifndef SELECTIONS_H_INCLUDED
#define SELECTIONS_H_INCLUDED
#include "genetics.h"
#include "prob.h"

struct fitness_sharing_params{ // parameters for selections based on fitness sharing
    double alpha;
    double beta;
    double sigma;
    length_t size;
};

union selection_params{
    length_t size; // size of the tournament, size of the elite for elitism, size of sampling in roulette_selection
    double val;// percentual_elitism
    struct fitness_sharing_params fs_params; // fitness_sharing_tournament, fitness_sharing_roulette
};


typedef struct population(*selection)(const struct genetic_env* genv, struct population*, double**, const union selection_params*);

struct population tournament(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params* tourn_size);
struct population elitism(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params* new_size);
struct population percentual_elitism(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *elite_size);
struct population roulette_selection(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params* elite_size);
struct population fitness_sharing_tournament(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *params);
struct population fitness_sharing_roulette(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *params);

#endif