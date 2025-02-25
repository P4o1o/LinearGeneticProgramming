#include "evolution.h"

inline struct individual mutation(const struct genetic_env *genv, struct individual* individ, length_t mut_max_len) {
	length_t mutation_len = RAND_BOUNDS(0, mut_max_len);
	length_t start = RAND_UPTO(individ->dna_len);
	length_t last_piece = individ->dna_len - start;
	length_t substitution = RAND_BOUNDS(0, last_piece);
	struct individual mutated;
	mutated.dna_len = individ->dna_len + mutation_len - substitution;
	if(mutated.dna_len == 0){
		mutated.dna_len = 1;
		mutation_len = 1;
	}
	mutated.dna = (struct cromosome*)malloc(sizeof(struct cromosome) * mutated.dna_len);
	if (mutated.dna == NULL) {
		MALLOC_FAIL_THREADSAFE;
	}
	memset(mutated.dna, 0, sizeof(struct cromosome) * mutated.dna_len);
	if (start)
		memcpy(mutated.dna, individ->dna, sizeof(struct cromosome) * start);
	length_t end_mutation = start + mutation_len;
	for (length_t j = start; j < end_mutation; j++) {
		mutated.dna[j].res = RAND_UPTO(genv->env_size - 1);
		mutated.dna[j].op = RAND_UPTO(genv->ops_size - 1);
        if(genv->ops[mutated.dna[j].op].arity == -1)
            mutated.dna[j].args.imm = RAND_DOUBLE;
		for (int32_t k = 0; k < genv->ops[mutated.dna[j].op].arity; k++)
			mutated.dna[j].args.reg[k] = RAND_UPTO(genv->env_size - 1);
	}
	length_t restart = start + substitution;
	last_piece -= substitution;
	if (last_piece)
		memcpy(mutated.dna + end_mutation, individ->dna + restart, sizeof(struct cromosome) * last_piece);
	return mutated;
}

inline void crossover(struct individual* individ0, struct individual *individ1, struct individual *new0, struct individual *new1) {
	length_t start_i = RAND_UPTO(individ0->dna_len - 1);
	length_t end_i = RAND_BOUNDS(start_i + 1, individ0->dna_len);
	length_t start_j = RAND_UPTO(individ1->dna_len - 1);
	length_t end_j = RAND_BOUNDS(start_j + 1, individ1->dna_len);
	// FIRST CROSSOVER
	length_t slice_j_size = end_j - start_j;
	length_t first_i_slice_j = start_i + slice_j_size;
	length_t last_of_i = individ0->dna_len - end_i;
	new0->dna_len = first_i_slice_j + last_of_i;
	new0->dna = (struct cromosome*)malloc(sizeof(struct cromosome) * new0->dna_len);
	// SECOND CROSSOVER
	length_t slice_i_size = end_i - start_i;
	length_t first_j_slice_i = start_j + slice_i_size;
	length_t last_of_j = individ1->dna_len - end_j;
	new1->dna_len = first_j_slice_i + last_of_j;
	new1->dna = (struct cromosome*)malloc(sizeof(struct cromosome) * new1->dna_len);
	if (new0->dna == NULL || new1->dna == NULL) {
        MALLOC_FAIL_THREADSAFE;
	}
	// FIRST CROSSOVER
	if (start_i)
		memcpy(new0->dna, individ0->dna, sizeof(struct cromosome) * start_i);
	memcpy(new0->dna + start_i, individ1->dna + start_j, sizeof(struct cromosome) * slice_j_size);
	if (last_of_i)
		memcpy(new0->dna + first_i_slice_j, individ0->dna + end_i, sizeof(struct cromosome) * last_of_i);
	// SECOND CROSSOVER
	if (start_j)
		memcpy(new1->dna, individ1->dna, sizeof(struct cromosome) * start_j);
	memcpy(new1->dna + start_j, individ0->dna + start_i, sizeof(struct cromosome) * slice_i_size);
	if (last_of_j)
		memcpy(new1->dna + first_j_slice_i, individ1->dna + end_j, sizeof(struct cromosome) * last_of_j);

}



struct genetic_result evolve(const struct genetic_input* in, const struct genetic_options* args) {
	prob mut_prob = PROBABILITY(args->mutation_prob);
	prob cross_prob = PROBABILITY(args->crossover_prob);
	random_seed_init();
	// POPULATION INITIALIZATION
	struct genetic_result res = args->init_type(in, args->initial_pop_size, args->dna_minsize, args->dna_maxsize);
	length_t mse_calculated = res.pop.size;
	double winner = DBL_MAX;
	res.best_individ = 0;
	for (length_t i = 0; i < res.pop.size; i++) {
		if (res.mse[i] < winner){
			winner = res.mse[i];
			res.best_individ = i;
		}
	}
	if(args->verbose)
		printf("\nGeneration 0, best_mse %lf, population_size %d", winner, res.pop.size);
	if (winner <= args->tollerance){
		return res;
	}
    // GENERATIONS LOOP
    for (res.generations = 1; res.generations <= args->generations; res.generations++) {
        // EVOLUTION
        res.pop.individuals = (struct individual*) realloc(res.pop.individuals, sizeof(struct individual) * res.pop.size * 4 *  args->evolution_cycles);
        if (res.pop.individuals == NULL){
			MALLOC_FAIL;
		}
		const length_t oldsize = res.pop.size;
#pragma omp parallel for schedule(dynamic,1) collapse(2)
		for (length_t i = 0; i < oldsize; i += 1) {
			for(length_t cycle = 0; cycle < args->evolution_cycles; cycle++){
				// MUTATION
				if (WILL_HAPPEN(mut_prob)) {
					struct individual mutated = mutation(&(in->genv), &(res.pop.individuals[i]), args->mut_max_len);
					struct individual clean = remove_trash(&(in->genv), &mutated); // if clean.size == 0 => empty individual
					free(mutated.dna);
					if(clean.dna_len){
#pragma omp critical
						{
								res.pop.individuals[res.pop.size] = clean;
								res.pop.size += 1;
						}
					}else{
						free(clean.dna);
					}
				}
				// CROSS OVER
				if (WILL_HAPPEN(cross_prob)) {
					struct individual new0;
					struct individual new1;
					length_t j = RAND_UPTO(oldsize - 1);
					crossover(&(res.pop.individuals[i]), &(res.pop.individuals[j]), &new0, &new1);
					struct individual clean0 = remove_trash(&(in->genv), &new0);
					free(new0.dna);
					if(clean0.dna_len != 0){
#pragma omp critical
						{
							res.pop.individuals[res.pop.size] = clean0;
							res.pop.size += 1;
						}
					}else{
						free(clean0.dna);
					}
					struct individual clean1 = remove_trash(&(in->genv), &new1);
					free(new1.dna);
					if(clean1.dna_len != 0){
#pragma omp critical
						{
							res.pop.individuals[res.pop.size] = clean1;
							res.pop.size += 1;
						}
					}else{
							free(clean1.dna);
						}
				}
			}
		}
        res.pop.individuals = (struct individual*) realloc(res.pop.individuals, sizeof(struct individual) * res.pop.size);
		res.mse = (double*) realloc(res.mse, sizeof(double) * res.pop.size);
        if (res.pop.individuals == NULL || res.mse == NULL){
			MALLOC_FAIL;
	    }
		// MSE EVALUATION
		mse_population(in, &res.pop, &res.mse, mse_calculated);
		res.evaluations += (uint64_t)(res.pop.size - mse_calculated);
		double best_mse = DBL_MAX;
		for (length_t i = 0; i < res.pop.size; i++) {
			if (res.mse[i] < best_mse){
				res.best_individ = i;
				best_mse = res.mse[i];
			}
		}
		if(args->verbose)
			printf("\nGeneration %d, best_mse %lf, population_size %d, evaluations %ld", res.generations, best_mse, res.pop.size, res.evaluations);
		if (best_mse <= args->tollerance)
			break;
		// SELECTION
        res.pop = args->select_type(&in->genv, &res.pop, &res.mse, &args->select_param);
		mse_calculated = res.pop.size; // mse already calculated
    }
	double best_mse = DBL_MAX;
	for (length_t i = 0; i < res.pop.size; i++) {
		if (res.mse[i] < best_mse){
			res.best_individ = i;
			best_mse = res.mse[i];
		}
	}
    return res;
}
