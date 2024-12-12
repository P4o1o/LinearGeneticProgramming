#include "creations.h"

inline unsigned int equal_individ(const struct genetic_env* genv, const struct individual* ind1, const struct individual* ind2){
	if(ind1->dna_len != ind2->dna_len){
		return 0;
	}
	for(length_t i = 0; i < ind1->dna_len; i++){
		if(ind1->dna[i].op != ind2->dna[i].op || ind1->dna[i].res != ind2->dna[i].res){
			return 0;
		}
		for(int32_t j = 0; j < genv->ops[ind1->dna[i].op].arity; j++){
				if(ind1->dna[i].args.reg[j] != ind2->dna[i].args.reg[j]){
					return 0;
				}
			}
		if(genv->ops[ind1->dna[i].op].arity == -1){
			if(ind1->dna[i].args.imm != ind2->dna[i].args.imm){
					return 0;
			}
		}
	}
	return 1;
}

#define SIPROUND \
        v0 += v1; \
        v1 = (v1 << 13) | (v1 >> 51); \
        v1 ^= v0; \
        v0 = (v0 << 32) | (v0 >> 32); \
        v2 += v3; \
        v3 = (v3 << 16) | (v3 >> 48);     \
        v3 ^= v2; \
        v2 += v1; \
        v1 = (v1 << 17) | (v0 >> 47); \
        v1 ^= v2; \
        v2 = (v2 << 32) | (v2 >> 32); \
        v0 += v3; \
        v3 = (v3 << 21) | (v0 >> 43); \
        v3 ^= v0;

inline uint64_t hash_individ(const struct genetic_env* genv, const struct individual* individ){
	uint64_t key0 = 0x3df52ab9c5671a23ULL;
	uint64_t key1 = 0x7a8321f0bc9a8533ULL;

	uint64_t v0 = key0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = key1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = key0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = key1 ^ 0x7465646279746573ULL;

	for(length_t i = 0; i < individ->dna_len; i++){
		uint64_t instruction = (((uint64_t) individ->dna[i].op) << 48) | (((uint64_t) individ->dna[i].res) << 32);
        union doubletouint val;
		switch(genv->ops[individ->dna[i].op].arity){
			case 2:
				instruction |= (uint64_t) individ->dna[i].args.reg[1];
			// FALL THROUGHT
			case 1:
				instruction |= (((uint64_t) individ->dna[i].args.reg[0]) << 16);
			break;

			case 0:
			break;

			case -1:
                val.d = individ->dna[i].args.imm;
				instruction |= (val.u >> 32);
			break;

			default:
			UNREACHABLE;
		}
		v3 ^= instruction;
		SIPROUND;
		SIPROUND;
		v0 ^= instruction;
	}
	v3 ^= individ->dna_len;
	SIPROUND;
	SIPROUND;
	v0 ^= individ->dna_len;
	v2 ^= 0xff;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    SIPROUND;
	return v0 ^ v1 ^ v2 ^ v3;
}

inline struct individual rand_individual(const struct genetic_env* genv, const length_t minsize, const length_t maxsize) {
	struct individual res;
	res.dna_len = RAND_BOUNDS(minsize, maxsize);
	res.dna = (struct cromosome*) malloc(sizeof(struct cromosome) * res.dna_len);
	if (res.dna == NULL) {
		MALLOC_FAIL_THREADSAFE;
	}
	for (length_t i = 0; i < res.dna_len; i++) {
		res.dna[i].res = RAND_UPTO(genv->env_size - 1);
		res.dna[i].op = RAND_UPTO(genv->ops_size - 1);
        if(genv->ops[res.dna[i].op].arity == -1)
            res.dna[i].args.imm = RAND_DOUBLE;
		for (int32_t k = 0; k < genv->ops[res.dna[i].op].arity; k++)
			res.dna[i].args.reg[k] = RAND_UPTO(genv->env_size - 1);
	}
	return res;
}

struct genetic_result rand_population(const struct genetic_input* in, const length_t pop_size, const length_t dna_minsize, const length_t dna_maxsize) {
	struct genetic_result res;
	res.generations = 0;
	res.pop.size = pop_size;
	res.pop.individuals = (struct individual*)malloc(sizeof(struct individual) * res.pop.size);
	res.mse = (double*) malloc(sizeof(double) * res.pop.size);
	if (res.pop.individuals == NULL || res.mse == NULL) {
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic,1)
	for (length_t i = 0; i < res.pop.size; i++) {
		do {
			struct individual tmp = rand_individual(&(in->genv), dna_minsize, dna_maxsize);
			res.pop.individuals[i] = remove_trash(&in->genv, &tmp);
			free(tmp.dna);
			if ((res.mse[i] = get_mse(in, &res.pop.individuals[i])) != DBL_MAX) {
				break;
			}
            free(res.pop.individuals[i].dna);
		} while (1);
	}
	res.evaluations = (uint64_t) res.pop.size;
	return res;
}

struct genetic_result unique_population(const struct genetic_input* in, const length_t pop_size, const length_t dna_minsize, const length_t dna_maxsize){
	struct evaluated_set map;
	map.capacity = pop_size * 2;
	map.list = (struct evaluated_set_node **) malloc(sizeof(struct evaluated_set_node *) * map.capacity);
	if(map.list == NULL){
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(static,1)
	for(length_t i = 0; i < map.capacity; i++){
		map.list[i] = NULL;
	}
	map.size = 0;
	uint64_t evals = 0;
#pragma omp parallel for schedule(dynamic,1)
	for (length_t i = 0; i < pop_size; i++) {
		do {
			struct individual tmp = rand_individual(&(in->genv), dna_minsize, dna_maxsize);
			struct individual ind = remove_trash(&in->genv, &tmp);
			free(tmp.dna);
			if(ind.dna_len == 0){
				free(ind.dna);
				continue;
			}
			uint64_t hash = hash_individ(&in->genv, &ind);
			length_t i = (length_t)(hash % map.capacity);
			double act_mse = get_mse(in, &ind);
			unsigned int found = 0;
			if(act_mse != DBL_MAX){
#pragma omp critical
				{
					evals += 1;
					struct evaluated_set_node *actual_node = map.list[i];
					while(actual_node != NULL){
						if(actual_node->hash == hash)
							if(equal_individ(&in->genv, &ind, & actual_node->ind)){
								found = 1;
								break;
							}
						actual_node = actual_node->next;
					}
					if(! found){
						actual_node = malloc(sizeof(struct evaluated_set_node));
						if(actual_node == NULL){
							MALLOC_FAIL;
						}
						actual_node->hash = hash;
						actual_node->ind = ind;
						actual_node->next = map.list[i];
						actual_node->mse = act_mse;
						map.list[i] = actual_node;
						map.size += 1;
					}			
				}
				if(! found) // OpenMP non mi permette di scrivere il break all'interno della sezione critica
					break;
			}
			free(ind.dna);
		} while (1);
	}
	struct genetic_result res;
	res.generations = 0;
	res.pop.size = 0;
	res.pop.individuals = (struct individual*) malloc(sizeof(struct individual) * map.size);
	res.mse = (double*) malloc(sizeof(double) * map.size);
	if(res.pop.individuals == NULL || res.mse == NULL){
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic,1)
	for(length_t i = 0; i < map.capacity; i++){
		struct evaluated_set_node *actual_node = map.list[i];
		while(actual_node != NULL){
#pragma omp critical
			{
				res.pop.individuals[res.pop.size] = actual_node->ind;
				res.mse[res.pop.size] = actual_node->mse;
				res.pop.size += 1;
			}
			struct evaluated_set_node * tmp = actual_node;
			actual_node = actual_node->next;
			free(tmp);
		}
	}
	free(map.list);
	res.evaluations = evals;
	return res;
}
