#include "benchmarks.h"
#include "time.h"

void print_evolution(struct genetic_input *in, struct genetic_options *params){
	clock_t start, end;
	printf("\nStarting Evolution\n");
	start = clock();
	struct genetic_result res = evolve(in, params);
	end = clock();
	double exectime = ((double) (end - start)) / ((double) CLOCKS_PER_SEC);
	print_individual(&in->genv, &res.pop.individuals[res.best_individ]);
	printf("mse is %e, total evaluations %ld, time of execution (in seconds) %lf\n", res.mse[res.best_individ], res.evaluations, exectime);
	free_genetic_result(&res);

}

void fprint_evolution(FILE *file, struct genetic_input *in, struct genetic_options *params){
	clock_t start, end;
	start = clock();
	struct genetic_result res = evolve(in, params);
	end = clock();
	char *select;
	char select_pars[10];
	if(params->select_type == elitism){
		select = "elitism";
		snprintf(select_pars, 10, "%d", params->select_param.size);
	}else if(params->select_type == tournament){
		select = "tournament";
		snprintf(select_pars, 10, "%d", params->select_param.size);
	}else if(params->select_type == percentual_elitism){
		select = "percentual_elitism";
		snprintf(select_pars, 10, "%lf", params->select_param.val);
	}else if(params->select_type == roulette_selection){
		select = "roulette";
		snprintf(select_pars, 10, "%d", params->select_param.size);
	}else if(params->select_type == fitness_sharing_tournament){
		select = "fs_roulette";
		snprintf(select_pars, 10, "%d", params->select_param.fs_params.size);
	}else if(params->select_type == fitness_sharing_roulette){
		select = "fs_tournament";
		snprintf(select_pars, 10, "%d", params->select_param.fs_params.size);
	}else UNREACHABLE;
	double exectime = ((double) (end - start)) / ((double) CLOCKS_PER_SEC);
	fprintf(file,
		"%s, %s, %d, %d, %d, %d, %d, %lf, %lf, %d, %d, %e, %ld, %lf, %d\n",
		select, select_pars, (params->tollerance > res.mse[res.best_individ]), params->initial_pop_size,
		params->dna_minsize, params->dna_maxsize, params->evolution_cycles, params->crossover_prob,
		params->mutation_prob, params->mut_max_len, in->input_size, res.mse[res.best_individ],
		res.evaluations, exectime, res.generations);
	free_genetic_result(&res);

}

void fprint_head_evolution(FILE *file){
	fprintf(file, "select_type,select_args,found,initial_pop,in_min_len,in_max_len,ev_cycles,cross_prob,mut_prob,max_mut_len,tests,mse,evaluations,exec_time,gen\n");
}

void test_selections(FILE *file, struct genetic_input *in, struct genetic_options *params, length_t selectsize, double selectval, size_t test4selections){
	length_t tournsize = (length_t) (1.0 / selectval);
	for(size_t i = 1; i <= test4selections; i++){
		params->select_param.size = selectsize;
		params->select_type = elitism;
		fprint_evolution(file, in, params);
		params->select_param.size = selectsize;
		params->select_type = roulette_selection;
		//print_evolution(file, in, params);
		params->select_param.size = tournsize;
		params->select_type = tournament;
		fprint_evolution(file, in, params);
		params->select_param.val = selectval;
		params->select_type = percentual_elitism;
		fprint_evolution(file, in, params);
		/*
		params->select_param.fs_params.size = selectsize;
		params->select_param.fs_params.alpha = 1.0;
		params->select_param.fs_params.beta = 2.0;
		params->select_param.fs_params.sigma = 2.5;
		params->select_type = fitness_sharing_roulette;
		fprint_evolution(file, in, params);
		params->select_param.fs_params.size = tournsize;
		params->select_param.fs_params.alpha = 1.0;
		params->select_param.fs_params.beta = 2.0;
		params->select_param.fs_params.sigma = 2.5;
		params->select_type = fitness_sharing_tournament;
		fprint_evolution(file, in, params);
		*/
		printf("\ndone %ld of %ld", i, test4selections);
	}
}

void dice_game_with_minmax(){
    // DICE GAME
	struct genetic_env dice_genv;
	dice_genv.env_size = 4;
	dice_genv.ops_size = 6;
	dice_genv.ops = malloc(sizeof(struct operation) * dice_genv.ops_size);
	dice_genv.ops[0] = Addition;
	dice_genv.ops[1] = Subtraction;
	dice_genv.ops[2] = Multiplication;
	dice_genv.ops[3] = SafeDivision;
	dice_genv.ops[4] = Minimum;
	dice_genv.ops[5] = Maximum;

	struct genetic_input dice_game_in = dice_game(&dice_genv, 1000);

	struct genetic_options dice_params;
	dice_params.tollerance = 0.00000000000001;
	dice_params.generations = 500;
	dice_params.initial_pop_size = 5000;
	dice_params.init_type = unique_population;
	dice_params.select_type = elitism;
	dice_params.select_param.size = 1000;
	dice_params.dna_minsize = 2;
	dice_params.dna_maxsize = 3;
	dice_params.evolution_cycles = 2;
	dice_params.crossover_prob = 0.3;
	dice_params.mutation_prob = 0.9;
	dice_params.mut_max_len = 3;
	dice_params.verbose = 1;

	print_evolution(&dice_game_in, &dice_params);
	
	free_genetic_input(&dice_game_in);
}

void dice_game_with_selection(selection func, union selection_params *params){
    // DICE GAME
	struct genetic_env dice_genv = simple_genv(2);

	struct genetic_input dice_game_in = dice_game(&dice_genv, 500);

	struct genetic_options dice_params;
	dice_params.tollerance = 0.00000000000001;
	dice_params.generations = 100;
	dice_params.initial_pop_size = 3000;
	dice_params.init_type = unique_population;
	dice_params.select_type = func;
	dice_params.select_param = *params;
	dice_params.dna_minsize = 2;
	dice_params.dna_maxsize = 3;
	dice_params.evolution_cycles = 2;
	dice_params.crossover_prob = 0.4;
	dice_params.mutation_prob = 0.9;
	dice_params.mut_max_len = 3;
	dice_params.verbose = 1;

	print_evolution(&dice_game_in, &dice_params);
	free_genetic_input(&dice_game_in);
}

void shopping_with_percent(env_index bag_size){
    // SHOPPING LIST
	printf("Creating Environment\n");
	struct genetic_env shop_genv;
	shop_genv.env_size = bag_size * 2;
	shop_genv.ops_size = 5;
	shop_genv.ops = malloc(sizeof(struct operation) * shop_genv.ops_size);
	shop_genv.ops[0] = Addition;
	shop_genv.ops[1] = Subtraction;
	shop_genv.ops[2] = Multiplication;
	shop_genv.ops[3] = SafeDivision;
	shop_genv.ops[4] = Percentage;

	struct genetic_input shop_in = shopping_list(&shop_genv, bag_size, 1000);
	struct genetic_options shop_params;
	shop_params.tollerance = 0.0000000001;
	shop_params.generations = 400;
	shop_params.initial_pop_size = 300000;
	shop_params.init_type = unique_population;
	shop_params.select_type = tournament;
	shop_params.select_param.size = 5;
	shop_params.dna_minsize = 1;
	shop_params.dna_maxsize = (shop_genv.env_size > 4) ? (length_t) shop_genv.env_size : 5;
	shop_params.evolution_cycles = 3;
	shop_params.crossover_prob = 0.4;
	shop_params.mutation_prob = 0.7;
	shop_params.mut_max_len = 5;
	shop_params.verbose = 1;

	print_evolution(&shop_in, &shop_params);

	free_genetic_input(&shop_in);

}

void bounching_balls_with_power(){
	// BOUNCING BALLS
	struct genetic_env balls_genv;
	balls_genv.env_size = 12;
	balls_genv.ops_size = 5;
	balls_genv.ops = malloc(sizeof(struct operation) * balls_genv.ops_size);
	balls_genv.ops[0] = Addition;
	balls_genv.ops[1] = Subtraction;
	balls_genv.ops[2] = Multiplication;
	balls_genv.ops[3] = SafeDivision;
	balls_genv.ops[4] = Power;

	struct genetic_input balls_in = bouncing_balls(&balls_genv, 100);
	struct genetic_options balls_params;
	balls_params.tollerance = 0.0000000001;
	balls_params.generations = 400;
	balls_params.initial_pop_size = 20000;
	balls_params.init_type = unique_population;
	balls_params.select_type = tournament;
	balls_params.select_param.size = 5;
	balls_params.dna_minsize = 3;
	balls_params.dna_maxsize = 5;
	balls_params.evolution_cycles = 2;
	balls_params.crossover_prob = 0.565;
	balls_params.mutation_prob = 0.865;
	balls_params.mut_max_len = 5;
	balls_params.verbose = 1;

	print_evolution(&balls_in, &balls_params);
	
	free_genetic_input(&balls_in);
}

void snow_day_simple_genv(){
	// SNOW DAY
	struct genetic_env snow_genv = simple_genv(5);

	struct genetic_input snow_day_in = snow_day(&snow_genv, 100);

	struct genetic_options snow_params;
	snow_params.tollerance = 0.0000000001;
	snow_params.generations = 300;
	snow_params.initial_pop_size = 20000;
	snow_params.init_type = unique_population;
	snow_params.select_type = tournament;
	snow_params.select_param.size = 5;
	snow_params.dna_minsize = 1;
	snow_params.dna_maxsize = 5;
	snow_params.evolution_cycles = 3;
	snow_params.crossover_prob = 0.4;
	snow_params.mutation_prob = 0.6;
	snow_params.mut_max_len = 5;
	snow_params.verbose = 1;

	print_evolution(&snow_day_in, &snow_params);
	
	free_genetic_input(&snow_day_in);
}

void solve_vector_distance(length_t dimensions){
	struct genetic_env vgenv;
	vgenv.env_size = dimensions * 2;
	vgenv.ops_size = 5;
	vgenv.ops = malloc(sizeof(struct operation) * vgenv.ops_size);
	vgenv.ops[0] = Addition;
	vgenv.ops[1] = Subtraction;
	vgenv.ops[2] = Multiplication;
	vgenv.ops[3] = SafeDivision;
	vgenv.ops[4] = SquareRoot;
	struct genetic_input vin = vector_distance(&vgenv, dimensions, 500);

	struct genetic_options vec_params;
	vec_params.tollerance = 1.0e-20;
	vec_params.generations = 200;
	vec_params.initial_pop_size = 100000;
	vec_params.init_type = unique_population;
	vec_params.select_type = tournament;
	vec_params.select_param.size = 5;
	vec_params.dna_minsize = 2;
	vec_params.dna_maxsize = 2 * dimensions;
	vec_params.evolution_cycles = 2;
	vec_params.crossover_prob = 0.6;
	vec_params.mutation_prob = 1.0;
	vec_params.mut_max_len = 5;
	vec_params.verbose = 1;

	print_evolution(&vin, &vec_params);
	
	free_genetic_input(&vin);
}