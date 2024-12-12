#include "evolution.h"
#include "benchmarks.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
	//dice_game_with_minmax();
	//shopping_with_percent(5);
	//solve_vector_distance(4);
	//snow_day_simple_genv();
	//bounching_balls_with_power();

/*
	// Shopping List (5 items) save stat on file
	struct genetic_env shop_genv;
	shop_genv.env_size = 10;
	shop_genv.ops_size = 5;
	shop_genv.ops = malloc(sizeof(struct operation) * shop_genv.ops_size);
	shop_genv.ops[0] = Addition;
	shop_genv.ops[1] = Subtraction;
	shop_genv.ops[2] = Multiplication;
	shop_genv.ops[3] = SafeDivision;
	shop_genv.ops[4] = Percentage;

	struct genetic_input shop_in = shopping_list(&shop_genv, 5, 1000);
	struct genetic_options shop_params;
	shop_params.tollerance = 0.0000000001;
	shop_params.generations = 100;
	shop_params.initial_pop_size = 200000;
	shop_params.init_type = unique_population;
	shop_params.dna_minsize = 3;
	shop_params.dna_maxsize = 5;
	shop_params.evolution_cycles = 3;
	shop_params.crossover_prob = 0.4;
	shop_params.mutation_prob = 0.7;
	shop_params.mut_max_len = 5;
	shop_params.verbose = 1;

	FILE * file = fopen("selections_shopping.csv", "a");
	//fprint_head_evolution(file);
	printf("Start\n");
	test_selections(file, &shop_in, &shop_params, 10000, 0.20, 5);
	fclose(file);
	free_genetic_input(&shop_in);	
*/

	// Shopping List (2 items) show evolution
	env_index items = 2;
	struct genetic_env shop_genv;
	shop_genv.env_size = items * 2;
	shop_genv.ops_size = 5;
	shop_genv.ops = malloc(sizeof(struct operation) * shop_genv.ops_size);
	shop_genv.ops[0] = Addition;
	shop_genv.ops[1] = Subtraction;
	shop_genv.ops[2] = Multiplication;
	shop_genv.ops[3] = SafeDivision;
	shop_genv.ops[4] = Percentage;

	struct genetic_input shop_in = shopping_list(&shop_genv, items, 1000);
	struct genetic_options shop_params;
	shop_params.tollerance = 0.0000000001;
	shop_params.generations = 200;
	shop_params.initial_pop_size = 3000;
	shop_params.init_type = unique_population;
	shop_params.dna_minsize = 3;
	shop_params.dna_maxsize = 5;
	shop_params.evolution_cycles = 3;
	shop_params.crossover_prob = 0.4;
	shop_params.mutation_prob = 0.6;
	shop_params.mut_max_len = 5;
	shop_params.verbose = 1;
	shop_params.select_type = tournament;
	shop_params.select_param.size = 5;
	print_evolution(&shop_in, &shop_params);
	
	free_genetic_input(&shop_in);

	return 0;
}