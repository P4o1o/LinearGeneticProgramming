#include "genetics.h"

struct genetic_env simple_genv(const env_index size){
	struct genetic_env genv;
	genv.float_reg = size;
	genv.ops_size = 4;
	genv.ops = malloc(sizeof(struct operation) * genv.ops_size);
	if(genv.ops == NULL)
		MALLOC_FAIL;
	genv.ops[0] = Addition;
	genv.ops[1] = Subtraction;
	genv.ops[2] = Multiplication;
	genv.ops[3] = SafeDivision;
	return genv;
}

void print_individual(const struct genetic_env* genv, const struct individual* individ) {
	printf("\n");
	for (length_t i = 0; i < individ->dna_len; i++) {
		printf("reg %d = %s ", individ->dna[i].res, genv->ops[individ->dna[i].op].name);
		if (genv->ops[individ->dna[i].op].arity == -1)
			printf("%lf", individ->dna[i].args.imm);
		for (int32_t j = 0; j < genv->ops[individ->dna[i].op].arity; j++)
			printf("reg %d ", individ->dna[i].args.reg[j]);
		printf("\n");
	}
}

struct individual remove_trash(const struct genetic_env *genv, const struct individual* ind){
	env_index *used = malloc(sizeof(env_index) * genv->float_reg);
	if(used == NULL)
		MALLOC_FAIL_THREADSAFE;
	memset(used, 0, sizeof(env_index) * genv->float_reg);
	used[0] = 1;
	struct individual res;
	res.dna = malloc(sizeof(struct cromosome) * ind->dna_len);
	if(res.dna == NULL)
		MALLOC_FAIL_THREADSAFE;
	res.dna_len = 0;
	for(length_t i = ind->dna_len; i > 0; i--){
		if(genv->ops[ind->dna[i - 1].op].state_changer || used[ind->dna[i - 1].res]){
			res.dna[res.dna_len] = ind->dna[i - 1];
			res.dna_len += 1;
			used[ind->dna[i - 1].res] = 0;
			for(int32_t j = 0; j < genv->ops[ind->dna[i - 1].op].arity; j++){
				used[ind->dna[i - 1].args.reg[j]] = 1;
			}
		}
	}
	free(used);
	if(res.dna_len == 0){
		return res;
	}
	if(ind->dna_len > res.dna_len){
		res.dna = realloc(res.dna, sizeof(struct cromosome) * res.dna_len);
		if(res.dna == NULL)
			MALLOC_FAIL_THREADSAFE;
	}
	for(length_t i = 0; i < res.dna_len/2; i++){
		struct cromosome tmp = res.dna[i];
		res.dna[i] = res.dna[res.dna_len - 1 - i];
		res.dna[res.dna_len - 1 - i] = tmp;
	}
	return res;
}

inline struct virtual_env create_env(const struct genetic_env* genv){
	struct virtual_env env;
	env.freg = (double*) malloc(((size_t)genv->float_reg) * sizeof(double));
	env.ireg = (int64_t*) malloc(((size_t)genv->int_reg) * sizeof(int64_t));
    if(env.freg == NULL || env.ireg == NULL){
		MALLOC_FAIL_THREADSAFE;
    }
	return env;
}

inline void setup_env(const struct genetic_input* in, struct virtual_env env, length_t index){
	memset(env.freg, 0, in->genv.float_reg * sizeof(double));
	memcpy(env.freg, in->data[index].x, in->x_len * sizeof(double));
	memset(env.ireg, 0, in->genv.int_reg * sizeof(int64_t));
	env.flag = 0;
}

inline void free_env(struct virtual_env env){
	free(env.freg);
	free(env.ireg);
}

double predict(const struct genetic_env *genv, const struct individual *individ, const double *X, const env_index x_len){
    struct virtual_env actual_env = create_env(genv);
	memset(actual_env.freg, 0, genv->float_reg * sizeof(double));
	memcpy(actual_env.freg, X, x_len * sizeof(double));
	memset(actual_env.ireg, 0, genv->int_reg * sizeof(int64_t));
	actual_env.flag = 0;
    for (length_t j = 0; j < individ->dna_len; j++) {
        genv->ops[individ->dna[j].op].function(&actual_env, individ->dna[j].res, &individ->dna[j].args);
    }
	double result = actual_env.freg[0];
	free_env(actual_env);
    return result;
}

void free_population(struct population* pop) {
	for (length_t i = 0; i < pop->size; i++)
		free(pop->individuals[i].dna);
	free(pop->individuals);
	pop->size = 0;
}

void free_genetic_input(struct genetic_input* in){
	for (length_t i = 0; i < in->input_size; i++) {
		free(in->data[i].x);
	}
	free(in->data);
	free(in->genv.ops);
}

inline double get_mse(const struct genetic_input* in, const struct individual *individ) {
	if (individ->dna_len == 0)
		return DBL_MAX;
	double mse = 0;
	struct virtual_env actual_env = create_env(&(in->genv));
	for (length_t k = 0; k < in->input_size; k++) {
		setup_env(in, actual_env, k);
		for (length_t j = 0; j < individ->dna_len; j++) {
			in->genv.ops[individ->dna[j].op].function(&actual_env, individ->dna[j].res, &individ->dna[j].args);
		}
		if (!(isfinite(actual_env.freg[0]))){
            free_env(actual_env);
			return DBL_MAX;
        }
		double actual_mse = in->data[k].y - actual_env.freg[0];
		mse += actual_mse * actual_mse;
	}
    free_env(actual_env);
	if(isfinite(mse))
		return mse / (double)in->input_size;
	else
		return DBL_MAX;
}

void mse_population(const struct genetic_input* in, const struct population *pop, double **mse, const length_t already_calc){
#pragma omp parallel for schedule(dynamic,1)
	for (length_t i = already_calc; i < pop->size; i++) {
		(*mse)[i] = get_mse(in, &pop->individuals[i]);
	}
}

struct individual extract_best(const struct genetic_input* in, const struct population* pop) {
	length_t winner = 0;
	double* mse = (double*)malloc(sizeof(double) * pop->size);
	if (mse == NULL) {
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic)
	for (length_t i = 0; i < pop->size; i++) {
		mse[i] = get_mse(in, &pop->individuals[i]);
#pragma omp critical
        {
            if(mse[winner] > mse[i]) {
                winner = i;
            }
        }
	}
	free(mse);
	return pop->individuals[winner];
}


void free_genetic_result(struct genetic_result *res){
	free_population(&res->pop);
	free(res->mse);
}