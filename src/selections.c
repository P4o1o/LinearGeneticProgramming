#include "selections.h"

inline length_t invert_mse(double* mse, length_t size, double * inv_mse){
	double worst = 0.0;
	double best = DBL_MAX;
	length_t winner = 0;
	for(length_t j = 0; j < size; j++){
		if(mse[j] < DBL_MAX && mse[j] > worst)
			worst = mse[j];
		if(mse[j] < best){
			winner = j;
			best = mse[j];
		}
	}
#pragma omp parallel for schedule(static,1)
	for(length_t j = 0; j < size; j++){
		if(mse[j] < DBL_MAX){
			inv_mse[j] = worst - mse[j];
		}else{
			inv_mse[j] = 0.0;
		}
	}
	return winner;
}

inline double edit_distance(const struct genetic_env* genv, const struct individual* ind1, const struct individual* ind2){
	length_t row_len = ind2->dna_len + 1;
	double *table = malloc((ind1->dna_len + 1) * row_len * sizeof(double));
	if(table == NULL)
		MALLOC_FAIL_THREADSAFE;
	for(length_t j = 0; j <= ind2->dna_len; j++){
		table[j] = (double) j;
	}
	for(length_t i = 1; i <= ind1->dna_len; i++){
		table[i * row_len] = (double) i;
	}
	for(length_t i = 1; i <= ind1->dna_len; i++){
		for(length_t j = 1; j <= ind2->dna_len; j++){
			double cost = 0.0;
			if(ind1->dna[i - 1].op == ind2->dna[j - 1].op){
				for(int32_t k = 0; k < genv->ops[ind1->dna[i - 1].op].arity; k++){
					if(ind1->dna[i - 1].args.reg[k] != ind2->dna[j - 1].args.reg[k]){
						cost += 0.75 / (double) k;
					}
				}
				if(genv->ops[ind1->dna[i - 1].op].arity == -1){
					if(ind1->dna[i - 1].args.imm != ind2->dna->args.imm)
						cost = 0.75;
				}
				if(ind1->dna[i - 1].res != ind2->dna[j - 1].res){
					cost += 0.25;
				}
			}else{
				cost = 1.0;				
			}
			double subst = table[(i - 1) * row_len + j - 1] + cost;
			double delete = table[(i - 1) * row_len + j] + 1.0;
			double insert = table[i * row_len + j - 1] + 1.0;
			table[i * row_len + j] = (subst < insert && subst < delete) ? subst : (insert  < delete) ? insert : delete;
		}
	}
	double result = table[ind1->dna_len * row_len + ind2->dna_len];
	free(table);
	return result;
}

inline double *distances_table(const struct genetic_env* genv, const struct population *pop){
	double *distance_tab = malloc(sizeof(double) * ((size_t) pop->size) * ((size_t) pop->size));
	if(distance_tab == NULL)
		MALLOC_FAIL;
#pragma omp parallel for collapse(2)
	for(length_t i = 0; i < pop->size; i++){
		for(length_t j = i; j < pop->size; j++){
			if(i == j){
				distance_tab[i * pop->size + j] = 0.0;
			}else{
				distance_tab[i * pop->size + j] = edit_distance(genv, &pop->individuals[i], &pop->individuals[j]);
				distance_tab[j * pop->size + i] = distance_tab[i * pop->size + j];
			}
		}
	}
	return distance_tab;
}

inline void shuffle_population(struct population* pop, double* mse) {
	for (length_t i = pop->size - 1; i > 0; i--) {
		length_t j = RAND_UPTO(i - 1);
		struct individual tmp = pop->individuals[i];
		double tmp_mse = mse[i];
		pop->individuals[i] = pop->individuals[j];
		mse[i] = mse[j];
		pop->individuals[j] = tmp;
		mse[j] = tmp_mse;
	}
}

inline void parallel_merge_sort(struct population *pop, double* mse) {
	length_t width;
    for (width = 1; width < pop->size; width *= 2) {
#pragma omp parallel
		{
			struct individual *temp_pop = (struct individual *) malloc( 2 * width * sizeof(struct individual));
			double *temp_mse = (double *) malloc( 2 * width * sizeof(double));
			if (temp_pop == NULL || temp_mse == NULL) {
#pragma omp critical
				{
					MALLOC_FAIL;
				}
			}
#pragma omp for
			for (length_t i = 0; i < pop->size; i += 2 * width) {
				length_t left = i;
				length_t mid = i + width - 1;
				length_t right = (i + 2 * width - 1 < pop->size) ? i + 2 * width - 1 : pop->size - 1;
				if (mid < right) {
					length_t i1 = left, i2 = mid + 1, k = 0;
					while (i1 <= mid && i2 <= right) {
						if (mse[i1] <= mse[i2]) {
							temp_mse[k] = mse[i1];
							temp_pop[k] = pop->individuals[i1];
							i1 += 1;
						} else {
							temp_mse[k] = mse[i2];
							temp_pop[k] = pop->individuals[i2];
							i2 += 1;
						}
						k +=1;
					}
					while (i1 <= mid) {
						temp_mse[k] = mse[i1];
						temp_pop[k] = pop->individuals[i1];
						i1 += 1;
						k += 1;
					}
					while (i2 <= right) {
						temp_mse[k] = mse[i2];
						temp_pop[k] = pop->individuals[i2];
						i2 += 1;
						k += 1;
					}
					for (length_t j = left; j <= right; j++) {
						mse[j] = temp_mse[j - left];
						pop->individuals[j] = temp_pop[j - left];
					}
				}
			}
			free(temp_mse);
			free(temp_pop);
		}
    }
}

struct population tournament(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params* tourn_size) {
	shuffle_population(initial, *mse);
	length_t tournaments = initial->size / tourn_size->size;
	length_t small_tourn = initial->size % tourn_size->size;
	struct population res;
	res.size = tournaments + (small_tourn != 0);
	res.individuals = (struct individual*)malloc(sizeof(struct individual) * res.size);
	double *saved_mse = (double*) malloc(sizeof(double) * res.size);
	if (res.individuals == NULL || saved_mse == NULL){
		MALLOC_FAIL;
    }
#pragma omp parallel for schedule(static,1)
	for (length_t i = 0; i < tournaments; i++) {
		length_t winner = 0;
		for (length_t j = 1; j < tourn_size->size; j++) {
			if ((*mse)[i * tourn_size->size + winner] > (*mse)[i * tourn_size->size + j]) {
				free(initial->individuals[i * tourn_size->size + winner].dna);
				winner = j;
			}
			else
				free(initial->individuals[i * tourn_size->size + j].dna);
		}
		res.individuals[i] = initial->individuals[i * tourn_size->size + winner];
		saved_mse[i] = (*mse)[i * tourn_size->size + winner];
	}
	if (small_tourn) {
		length_t winner = 1; // initial->size - 1
		for (length_t i = 2; i <= small_tourn; i++) {
			if ((*mse)[initial->size - winner] > (*mse)[initial->size - i]) {
				free(initial->individuals[initial->size - winner].dna);
				winner = i;
			}
			else
				free(initial->individuals[initial->size - i].dna);
		}
		res.individuals[res.size - 1] = initial->individuals[initial->size - winner];
		saved_mse[res.size - 1] = (*mse)[initial->size - winner];
	}
	free(*mse);
	*mse = saved_mse;
	free(initial->individuals);
	return res;
}

struct population elitism(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *elite_size) {
	if(initial->size <= elite_size->size)
		return *initial;
	parallel_merge_sort(initial, *mse);
	struct population res;
	res.size = elite_size->size;
	for(length_t i = res.size; i < initial->size; i++){
		free(initial->individuals[i].dna);
	}
	res.individuals = realloc(initial->individuals, sizeof(struct individual) * res.size);
	*mse = realloc(*mse, sizeof(double) * res.size);
	if(res.individuals == NULL || *mse == NULL)
		MALLOC_FAIL;
	initial->individuals = NULL;
	return res;
}

struct population percentual_elitism(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *elite_size) {
	struct population res;
	res.size = (length_t)(elite_size->val * ((double) initial->size));
	if(res.size < 2)
		return *initial;
	parallel_merge_sort(initial, *mse);
	for(length_t i = res.size; i < initial->size; i++){
		free(initial->individuals[i].dna);
	}
	res.individuals = realloc(initial->individuals, sizeof(struct individual) * res.size);
	*mse = realloc(*mse, sizeof(double) * res.size);
	if(res.individuals == NULL || *mse == NULL)
		MALLOC_FAIL;
	initial->individuals = NULL;
	return res;
}

struct population roulette_selection(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *new_size) {
	struct population res;
	res.size = new_size->size;
	res.individuals = (struct individual*) malloc(new_size->size * sizeof(struct individual));
	double *mse_survived = (double*) malloc(sizeof(double) * new_size->size);
	double *inv_mse = (double*) malloc(sizeof(double) * initial->size);
	if (res.individuals == NULL || mse_survived == NULL || inv_mse == NULL){
		MALLOC_FAIL;
	}
	shuffle_population(initial, *mse);
	double worst = 0.0;
	double best = DBL_MAX;
	length_t winner = 0;
	for(length_t j = 0; j < initial->size; j++){
		if((*mse)[j] < DBL_MAX && (*mse)[j] > worst)
			worst = (*mse)[j];
		if((*mse)[j] < best){
			winner = j;
			best = (*mse)[j];
		}
	}
	inv_mse[0] = ((*mse)[0] < DBL_MAX) ? worst - ((*mse)[0]) : 0.0;
	for(length_t j = 1; j < initial->size; j++){
		if((*mse)[j] < DBL_MAX){
			inv_mse[j] = worst - ((*mse)[j]);
		}else{
			inv_mse[j] = 0.0;
		}
		inv_mse[j] += inv_mse[j - 1];
	}
	if(! (inv_mse[initial->size - 1] > 0.0)){
#pragma omp parallel for schedule(static,1)
		for(length_t j = 0; j < initial->size; j++){
			inv_mse[j] = (double)(j + 1);
		}
	}
	res.individuals[0].dna_len = initial->individuals[winner].dna_len;
	res.individuals[0].dna = (struct cromosome *) malloc(sizeof(struct cromosome) * res.individuals[0].dna_len);
	if(res.individuals[0].dna == NULL)
		MALLOC_FAIL;
	memcpy(res.individuals[0].dna, initial->individuals[winner].dna, sizeof(struct cromosome) * res.individuals[0].dna_len);
	mse_survived[0] = (*mse)[winner];
#pragma omp parallel for schedule(static,1)
	for(length_t i = 1; i < new_size->size; i++){
		double selected = RAND_DBL_BOUNDS(0.0, inv_mse[initial->size - 1]);
		for (length_t j = 0; j < initial->size; j++) {
			if (selected <= inv_mse[j]) {
				res.individuals[i].dna_len = initial->individuals[j].dna_len;
				res.individuals[i].dna = (struct cromosome *) malloc(sizeof(struct cromosome) * res.individuals[i].dna_len);
				if(res.individuals[i].dna == NULL){
#pragma omp critical
					{
						MALLOC_FAIL;
					}
				}
				memcpy(res.individuals[i].dna, initial->individuals[j].dna, sizeof(struct cromosome) * res.individuals[i].dna_len);
				mse_survived[i] = (*mse)[j];
				break;
			}
		}
	}
	free(inv_mse);
	free(*mse);
#pragma omp parallel for schedule(static,1) 
	for (length_t j = 0; j < initial->size; j++) {
		free(initial->individuals[j].dna);
	}
	free(initial->individuals);
	*mse = mse_survived;
	return res;
}

inline double *fitness_sharing(const struct genetic_env* genv, struct population* initial, double* fitness, const union selection_params *params){
	double * dtab = distances_table(genv, initial);
	double *res = malloc(sizeof(double) * initial->size);
	if(res == NULL)
		MALLOC_FAIL;
#pragma omp parallel for schedule(static, 1)
	for(length_t i = 0; i < initial->size; i++){
		double sharing = 0.0;
		for(length_t j = 0; j < initial->size; j++){
			if(dtab[i * initial->size + j] == 0){
				sharing += 1;
			}else if(dtab[i * initial->size + j] < params->fs_params.sigma){
				sharing += 1.0 - pow((dtab[i * initial->size + j] / params->fs_params.sigma), params->fs_params.alpha);
			}
		}
		if(sharing == 0.0)
			sharing = 1.0;
		res[i] = pow(fitness[i], params->fs_params.beta) * sharing;
	}
	free(dtab);
	return res;
}


struct population fitness_sharing_tournament(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *params){
	shuffle_population(initial, *mse);
	length_t tournaments = initial->size / params->fs_params.size;
	length_t small_tourn = initial->size % params->fs_params.size;
	struct population res;
	res.size = tournaments + (small_tourn != 0);
	res.individuals = (struct individual*)malloc(sizeof(struct individual) * res.size);
	double *saved_mse = (double*) malloc(sizeof(double) * res.size);
	if (res.individuals == NULL || saved_mse == NULL){
		MALLOC_FAIL;
    }
	double *fs = fitness_sharing(genv, initial, *mse, params);
#pragma omp parallel for schedule(static,1)
	for (length_t i = 0; i < tournaments; i++) {
		length_t winner = 0;
		for (length_t j = 1; j < params->fs_params.size; j++) {
			if (fs[i * params->fs_params.size + winner] > fs[i * params->fs_params.size + j]) {
				free(initial->individuals[i * params->fs_params.size + winner].dna);
				winner = j;
			}
			else
				free(initial->individuals[i * params->fs_params.size + j].dna);
		}
		res.individuals[i] = initial->individuals[i * params->fs_params.size + winner];
		saved_mse[i] = (*mse)[i * params->fs_params.size + winner];
	}
	if (small_tourn) {
		length_t winner = 1; // initial->size - 1
		for (length_t i = 2; i <= small_tourn; i++) {
			if (fs[initial->size - winner] > fs[initial->size - i]) {
				free(initial->individuals[initial->size - winner].dna);
				winner = i;
			}
			else
				free(initial->individuals[initial->size - i].dna);
		}
		res.individuals[res.size - 1] = initial->individuals[initial->size - winner];
		saved_mse[res.size - 1] = (*mse)[initial->size - winner];
	}
	free(fs);
	free(*mse);
	*mse = saved_mse;
	free(initial->individuals);
	return res;
}

struct population fitness_sharing_roulette(const struct genetic_env* genv, struct population* initial, double** mse, const union selection_params *params) {
	struct population res;
	res.size = params->fs_params.size;
	res.individuals = (struct individual*) malloc(params->fs_params.size * sizeof(struct individual));
	double *mse_survived = (double*) malloc(sizeof(double) * params->fs_params.size);
	double *inv_fs = (double*) malloc(sizeof(double) * initial->size);
	if (res.individuals == NULL || mse_survived == NULL || inv_fs == NULL){
		MALLOC_FAIL;
	}
	double *fs = fitness_sharing(genv, initial, *mse, params);
	length_t winner = invert_mse(fs, initial->size, inv_fs);
	free(fs);
	for(length_t j = 1; j < initial->size; j++){
		inv_fs[j] += inv_fs[j - 1];
	}
	if(! (inv_fs[initial->size - 1] > 0.0)){
#pragma omp parallel for schedule(static,1)
		for(length_t j = 0; j < initial->size; j++){
			inv_fs[j] = (double)(j + 1);
		}
	}
	res.individuals[0].dna_len = initial->individuals[winner].dna_len;
	res.individuals[0].dna = (struct cromosome *) malloc(sizeof(struct cromosome) * res.individuals[0].dna_len);
	if(res.individuals[0].dna == NULL)
		MALLOC_FAIL;
	memcpy(res.individuals[0].dna, initial->individuals[winner].dna, sizeof(struct cromosome) * res.individuals[0].dna_len);
	mse_survived[0] = (*mse)[winner];
#pragma omp parallel for schedule(static,1)
	for(length_t i = 1; i < params->fs_params.size; i++){
		double winner = RAND_DBL_BOUNDS(0.0, inv_fs[initial->size - 1]);
		for (length_t j = 0; j < initial->size; j++) {
			if (winner <= inv_fs[j]) {
				res.individuals[i].dna_len = initial->individuals[j].dna_len;
				res.individuals[i].dna = (struct cromosome *) malloc(sizeof(struct cromosome) * res.individuals[i].dna_len);
				if(res.individuals[i].dna == NULL){
#pragma omp critical
					{
						MALLOC_FAIL;
					}
				}
				memcpy(res.individuals[i].dna, initial->individuals[j].dna, sizeof(struct cromosome) * res.individuals[i].dna_len);
				mse_survived[i] = (*mse)[j];
				break;
			}
		}
	}
	free(inv_fs);
	free(*mse);
#pragma omp parallel for schedule(static,1) 
	for (length_t j = 0; j < initial->size; j++) {
		free(initial->individuals[j].dna);
	}
	free(initial->individuals);
	*mse = mse_survived;
	return res;
}
