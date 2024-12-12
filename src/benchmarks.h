#ifndef BENCHMARKS_H_INCLUDED
#define BENCHMARKS_H_INCLUDED

#include "evolution.h"
#include "float_psb2.h"

void print_evolution(struct genetic_input *in, struct genetic_options *params);

void fprint_head_evolution(FILE *file);

void fprint_evolution(FILE *file, struct genetic_input *in, struct genetic_options *params);

void test_selections(FILE *file, struct genetic_input *in, struct genetic_options *params, length_t selectsize, double selectval, length_t test4selections);

void dice_game_with_minmax();

void dice_game_with_selection(selection func, union selection_params *params);

void shopping_with_percent(env_index bag_size);

void bounching_balls_with_power();

void snow_day_simple_genv();

void solve_vector_distance(length_t dimensions);

void test_selections_on_dice_game(FILE *file, length_t test4selections);

#endif