#include "evolution.h"
#include "benchmarks.h"
#include <stdio.h>

#ifdef _WIN32
    #include <windows.h>
    #define SLEEP(seconds) Sleep((seconds) * 1000)
#else
    #include <unistd.h>
    #define SLEEP(seconds) sleep(seconds)
#endif

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
	
	shop_params.select_type = fitness_sharing_tournament;
	shop_params.select_param.fs_params.size = 5;
	shop_params.select_param.fs_params.alpha = 0.5;
	shop_params.select_param.fs_params.beta = 5.0;
	shop_params.select_param.fs_params.sigma = 3.0;

	shop_params.select_type = tournament;
	shop_params.select_param.size = 5;

	print_evolution(&shop_in, &shop_params);
	
	free_genetic_input(&shop_in);
	*/
	
	struct genetic_input input;
	input.input_size = 100;
	input.x_len = 3;

	size_t test_num = 40;

	struct genetic_env genv;
	genv.env_size = 5;
	genv.ops_size = 5;
	genv.ops = malloc(sizeof(struct operation) * genv.ops_size);
	genv.ops[0] = Addition;
	genv.ops[1] = Subtraction;
	genv.ops[2] = Multiplication;
	genv.ops[3] = SafeDivision;
	genv.ops[4] = Power;
	
	input.genv = genv;
	
	save_problem_data("balls-3x100.bin", bouncing_balls(&genv, input.input_size));

	input.data = load_data("balls-3x100.bin", input.x_len, input.input_size);

	struct genetic_options params;
	params.tollerance = 1e-12;
	params.generations = 2300;
	params.initial_pop_size = 4000;
	params.init_type = unique_population;
	params.dna_minsize = 2;
	params.dna_maxsize = 5;
	params.evolution_cycles = 1;
	params.crossover_prob = 0.77; // 10000 new childrend
	params.mutation_prob = 0.96;
	params.mut_max_len = 5;
	params.verbose = 0;
	/*
	params.select_type = fitness_sharing_tournament;
	params.select_param.fs_params.size = 5;
	params.select_param.fs_params.alpha = 0.5;
	params.select_param.fs_params.beta = 5.0;
	params.select_param.fs_params.sigma = 3.0;
	*/
	params.select_type = elitism;
	params.select_param.size = 10000;

	//FILE * file = fopen("balls-3x100.csv", "a");
	
	//fprint_head_evolution(file);

	printf("Start\n");
/*
	for(size_t i = 1; i <= test_num; i++){
		fprint_evolution(file, &input, &params);
		printf("Done %lu of %lu\n", i, test_num);
		fflush(file);
		SLEEP(5);
	}
	fclose(file);
*/
	print_evolution(&input, &params);
	free_genetic_input(&input);

	return 0;
}