#include "selection.h"

static inline double edit_distance(const struct Program *const prog1, const struct Program *const prog2){
	uint64_t row_len = prog2->size + 1;
	double *table = malloc((prog1->size + 1) * row_len * sizeof(double));
	if(table == NULL)
		MALLOC_FAIL_THREADSAFE;
	for(uint64_t j = 0; j <= prog2->size; j++){
		table[j] = (double) j;
	}
	for(uint64_t i = 1; i <= prog1->size; i++){
		table[i * row_len] = (double) i;
	}
	for(uint64_t i = 1; i <= prog1->size; i++){
		for(uint64_t j = 1; j <= prog2->size; j++){
			double cost = 0.0;
			if(prog1->content[i - 1].op == prog2->content[j - 1].op){
                if(prog1->content[i - 1].addr != prog2->content[i - 1].addr)
                    cost = 0.5;
				if(prog1->content[i - 1].reg[0] != prog2->content[i - 1].reg[0]){
					cost += 0.2;
				}
                if(prog1->content[i - 1].reg[1] != prog2->content[i - 1].reg[1]){
					cost += 0.15;
				}
                if(prog1->content[i - 1].reg[2] != prog2->content[i - 1].reg[2]){
					cost += 0.15;
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
	double result = table[prog1->size * row_len + prog2->size];
	free(table);
	return result;
}

static inline double *distances_table(const struct Population *const pop){
	double *distance_tab = malloc(sizeof(double) * ((size_t) pop->size) * ((size_t) pop->size));
	if(distance_tab == NULL)
		MALLOC_FAIL;
#pragma omp parallel for collapse(2) num_threads(MAX_OMP_THREAD)
	for(uint64_t i = 0; i < pop->size; i++){
		for(uint64_t j = i; j < pop->size; j++){
			if(i == j){
				distance_tab[i * pop->size + j] = 0.0;
			}else{
				distance_tab[i * pop->size + j] = edit_distance(&(pop->individual[i].prog), &(pop->individual[j].prog));
				distance_tab[j * pop->size + i] = distance_tab[i * pop->size + j];
			}
		}
	}
	return distance_tab;
}

#define DECLARE_FITNESS_SHARING(NAME, OPER) \
static inline double *fitness_sharing_##NAME(struct Population *const initial, const union SelectionParams *const params){ \
	double * dtab = distances_table(initial); \
	double *res = malloc(sizeof(double) * initial->size); \
	if(res == NULL) \
		MALLOC_FAIL; \
    _Pragma("omp parallel for schedule(static, 1) num_threads(MAX_OMP_THREAD)") \
	for(uint64_t i = 0; i < initial->size; i++){ \
		double sharing = 0.0; \
		for(uint64_t j = 0; j < initial->size; j++){ \
			if(dtab[i * initial->size + j] == 0){ \
				sharing += 1; \
			}else if(dtab[i * initial->size + j] < params->fs_params.sigma){ \
				sharing += 1.0 - pow((dtab[i * initial->size + j] / params->fs_params.sigma), params->fs_params.alpha); \
			} \
		} \
		if(sharing == 0.0) \
			sharing = 1.0; \
		res[i] = pow(initial->individual[i].fitness, params->fs_params.beta) OPER sharing; \
	} \
	free(dtab); \
	return res; \
} /* END MACRO */

DECLARE_FITNESS_SHARING(MAXIMIZE, /)
DECLARE_FITNESS_SHARING(MINIMIZE, *)


static inline void shuffle_population(struct Population* pop) {
    ASSERT(pop->size > 0);
	for (uint64_t i = pop->size - 1; i > 0; i--) {
		uint64_t j = RAND_UPTO(i - 1);
		struct Individual tmp = pop->individual[i];
		pop->individual[i] = pop->individual[j];
		pop->individual[j] = tmp;
	}
}

#define DECLARE_MERGE_SORT(NAME, CMP) \
static inline void merge_sort_##NAME(struct Population *pop) { \
	uint64_t width; \
    for (width = 1; width < pop->size; width *= 2) { \
        _Pragma("omp parallel num_threads(MAX_OMP_THREAD)") \
		{ \
			struct Individual *temp_pop = (struct Individual *) malloc( 2 * width * sizeof(struct Individual)); \
			if (temp_pop == NULL) { \
                MALLOC_FAIL_THREADSAFE; \
			} \
            _Pragma("omp for") \
			for (uint64_t i = 0; i < pop->size; i += 2 * width) { \
				uint64_t left = i; \
				uint64_t mid = i + width - 1; \
				uint64_t right = (i + 2 * width - 1 < pop->size) ? i + 2 * width - 1 : pop->size - 1; \
				if (mid < right) { \
					uint64_t i1 = left, i2 = mid + 1, k = 0; \
					while (i1 <= mid && i2 <= right) { \
						if (pop->individual[i1].fitness CMP pop->individual[i2].fitness) { \
							temp_pop[k] = pop->individual[i1]; \
							i1 += 1; \
						} else { \
							temp_pop[k] = pop->individual[i2]; \
							i2 += 1; \
						} \
						k +=1; \
					} \
					while (i1 <= mid) { \
						temp_pop[k] = pop->individual[i1]; \
						i1 += 1; \
						k += 1; \
					} \
					while (i2 <= right) { \
						temp_pop[k] = pop->individual[i2]; \
						i2 += 1; \
						k += 1; \
					} \
					for (uint64_t j = left; j <= right; j++) { \
						pop->individual[j] = temp_pop[j - left]; \
					} \
				} \
			} \
			free(temp_pop); \
		} \
    } \
} /* END MACRO */

DECLARE_MERGE_SORT(MINIMIZE, <=)

DECLARE_MERGE_SORT(MAXIMIZE, >=)

#define DECLARE_FITNESS_SHARING_MERGE_SORT(NAME, CMP) \
static inline void fitness_sharing_merge_sort_##NAME(struct Population *pop, const double *const fitness_sharing) { \
	uint64_t width; \
    for (width = 1; width < pop->size; width *= 2) { \
        _Pragma("omp parallel num_threads(MAX_OMP_THREAD)") \
		{ \
			struct Individual *temp_pop = (struct Individual *) malloc( 2 * width * sizeof(struct Individual)); \
			if (temp_pop == NULL) { \
                MALLOC_FAIL_THREADSAFE; \
			} \
            _Pragma("omp for") \
			for (uint64_t i = 0; i < pop->size; i += 2 * width) { \
				uint64_t left = i; \
				uint64_t mid = i + width - 1; \
				uint64_t right = (i + 2 * width - 1 < pop->size) ? i + 2 * width - 1 : pop->size - 1; \
				if (mid < right) { \
					uint64_t i1 = left, i2 = mid + 1, k = 0; \
					while (i1 <= mid && i2 <= right) { \
						if (fitness_sharing[i1] CMP fitness_sharing[i2]) { \
							temp_pop[k] = pop->individual[i1]; \
							i1 += 1; \
						} else { \
							temp_pop[k] = pop->individual[i2]; \
							i2 += 1; \
						} \
						k +=1; \
					} \
					while (i1 <= mid) { \
						temp_pop[k] = pop->individual[i1]; \
						i1 += 1; \
						k += 1; \
					} \
					while (i2 <= right) { \
						temp_pop[k] = pop->individual[i2]; \
						i2 += 1; \
						k += 1; \
					} \
					for (uint64_t j = left; j <= right; j++) { \
						pop->individual[j] = temp_pop[j - left]; \
					} \
				} \
			} \
			free(temp_pop); \
		} \
    } \
} /* END MACRO */

DECLARE_FITNESS_SHARING_MERGE_SORT(MINIMIZE, <=)

DECLARE_FITNESS_SHARING_MERGE_SORT(MAXIMIZE, >=)

#define DECLARE_elitism(TYPE) \
void elitism_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    if(initial->size <= params->size) \
		return; \
    merge_sort_##TYPE(initial); \
    initial->size = params->size; \
} /* END MACRO */

#define DECLARE_percentual_elitism(TYPE) \
void percentual_elitism_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    uint64_t final_size = (uint64_t)(params->val * ((double) initial->size)); \
    if(final_size == 0) \
		return; \
    merge_sort_##TYPE(initial); \
    initial->size = final_size; \
} /* END MACRO */

#define DECLARE_fitness_sharing_elitism(TYPE) \
void fitness_sharing_elitism_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    if(initial->size <= params->size) \
		return; \
    double *fitness_sharing  = fitness_sharing_##TYPE(initial, params); \
    fitness_sharing_merge_sort_##TYPE(initial, fitness_sharing); \
    free(fitness_sharing); \
    initial->size = params->size; \
} /* END MACRO */

#define DECLARE_fitness_sharing_percentual_elitism(TYPE) \
void fitness_sharing_percentual_elitism_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    uint64_t final_size = (uint64_t)(params->val * ((double) initial->size)); \
    if(final_size == 0) \
		return; \
    double *fitness_sharing  = fitness_sharing_##TYPE(initial, params); \
    fitness_sharing_merge_sort_##TYPE(initial, fitness_sharing); \
    free(fitness_sharing); \
    initial->size = final_size; \
} /* END MACRO */

static inline uint64_t cmp_tournament_MINIMIZE(const double one, const double two){
    return one > two;
}

static inline uint64_t cmp_tournament_MAXIMIZE(const double one, const double two){
    return one < two;
}

#define DECLARE_tournament(TYPE) \
void tournament_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    ASSERT(initial->size > 0); \
    shuffle_population(initial); \
    uint64_t tournaments = initial->size / params->size; \
	uint64_t small_tourn = initial->size % params->size; \
	struct Population res = { .size = tournaments + (small_tourn != 0) }; \
    ASSERT(res.size > 0); \
	res.individual = (struct Individual*) malloc(sizeof(struct Individual) * res.size); \
	if (res.individual == NULL){ \
		MALLOC_FAIL; \
    } \
    _Pragma("omp parallel for schedule(static,1) num_threads(MAX_OMP_THREAD)") \
    for (uint64_t i = 0; i < tournaments; i++) { \
        uint64_t winner = 0; \
        for (uint64_t j = 1; j < params->size; j++) { \
            if (cmp_tournament_##TYPE(initial->individual[i * params->size + winner].fitness, initial->individual[i * params->size + j].fitness)) { \
                winner = j; \
            } \
        } \
        res.individual[i] = initial->individual[i * params->size + winner]; \
    } \
    if (small_tourn) { \
        uint64_t winner = 1; /* initial->size - 1 */ \
        for (uint64_t i = 2; i <= small_tourn; i++) { \
            if (cmp_tournament_##TYPE(initial->individual[initial->size - winner].fitness, initial->individual[initial->size - i].fitness)) { \
                winner = i; \
            } \
        } \
        res.individual[res.size - 1] = initial->individual[initial->size - winner]; \
    } \
    memcpy(initial->individual, res.individual, res.size * sizeof(struct Individual)); \
    initial->size = res.size; \
    free(res.individual); \
} /* END MACRO */

#define DECLARE_fitness_sharing_tournament(TYPE) \
void fitness_sharing_tournament_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    ASSERT(initial->size > 0); \
    shuffle_population(initial); \
    uint64_t tournaments = initial->size / params->size; \
	uint64_t small_tourn = initial->size % params->size; \
	struct Population res = { .size = tournaments + (small_tourn != 0) }; \
    ASSERT(res.size > 0); \
	res.individual = (struct Individual*) malloc(sizeof(struct Individual) * res.size); \
	if (res.individual == NULL){ \
		MALLOC_FAIL; \
    } \
    double *fs = fitness_sharing_##TYPE(initial, params); \
    _Pragma("omp parallel for schedule(static,1) num_threads(MAX_OMP_THREAD)") \
    for (uint64_t i = 0; i < tournaments; i++) { \
        uint64_t winner = 0; \
        for (uint64_t j = 1; j < params->size; j++) { \
            if (cmp_tournament_##TYPE(fs[i * params->size + winner], fs[i * params->size + j])) { \
                winner = j; \
            } \
        } \
        res.individual[i] = initial->individual[i * params->size + winner]; \
    } \
    if (small_tourn) { \
        uint64_t winner = 1; /* initial->size - 1 */ \
        for (uint64_t i = 2; i <= small_tourn; i++) { \
            if (cmp_tournament_##TYPE(fs[initial->size - winner], fs[initial->size - i])) { \
                winner = i; \
            } \
        } \
        res.individual[res.size - 1] = initial->individual[initial->size - winner]; \
    } \
    free(fs); \
    memcpy(initial->individual, res.individual, res.size * sizeof(struct Individual)); \
    initial->size = res.size; \
    free(res.individual); \
} /* END MACRO */

static inline double probability_roulette_MAXIMIZE(const double fitness, const struct DoubleCouple maxmin){
    if(fitness != -DBL_MAX)
        return fitness / maxmin.val[0];
    else
        return 0;
}

static inline double probability_roulette_MINIMIZE(const double fitness, const struct DoubleCouple maxmin){
    if(fitness != DBL_MAX)
        return (maxmin.val[0] - fitness) / (maxmin.val[0] - maxmin.val[1]);
    else
        return 0;
}

static inline struct DoubleCouple get_info_roulette_MAXIMIZE(struct Population * initial){
    double best = -DBL_MAX;
    for(uint64_t j = 0; j < initial->size; j++){
        if(initial->individual[j].fitness < best){
			best = initial->individual[j].fitness;
		}
    }
    return (struct DoubleCouple){.val = {best, -DBL_MAX}};
}

static inline struct DoubleCouple get_info_roulette_MINIMIZE(struct Population * initial){
    double best = DBL_MAX;
    double worst = -DBL_MAX;
    for(uint64_t j = 0; j < initial->size; j++){
        if(initial->individual[j].fitness != DBL_MAX && initial->individual[j].fitness > worst){
            worst = initial->individual[j].fitness;
        }
        if(initial->individual[j].fitness < best){
			best = initial->individual[j].fitness;
		}
    }
    return (struct DoubleCouple){.val = {best, worst}};
}

#define DECLARE_roulette(TYPE) \
void roulette_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    struct Population res; \
	res.individual = (struct Individual*) malloc(initial->size * sizeof(struct Individual)); \
    if(res.individual == NULL){ \
        MALLOC_FAIL; \
    } \
	const struct DoubleCouple maxmin = get_info_roulette_##TYPE(initial); \
    if(maxmin.val[0] == maxmin.val[1]){ \
        res.individual[0] = initial->individual[0]; \
        uint64_t last_elem = 0; \
        _Pragma("omp parallel for schedule(dynamic,1) num_threads(MAX_OMP_THREAD)") \
        for(uint64_t i = 1; i < initial->size; i++){ \
            prob probability = PROBABILITY(0.5); \
            if(WILL_HAPPEN(probability)){ \
                _Pragma("omp atomic") \
                    ++last_elem; \
                res.individual[last_elem] = initial->individual[i]; \
            } \
        } \
        res.size = last_elem + 1; \
    }else{ \
        uint64_t last_elem = -1; \
        _Pragma("omp parallel for schedule(dynamic,1) num_threads(MAX_OMP_THREAD)") \
        for(uint64_t i = 0; i < initial->size; i++){ \
            double doubleprob = probability_roulette_##TYPE(initial->individual[i].fitness, maxmin); \
            prob probability = PROBABILITY(doubleprob); \
            if(WILL_HAPPEN(probability)){ \
                _Pragma("omp atomic") \
                    ++last_elem; \
                res.individual[last_elem] = initial->individual[i]; \
            } \
        } \
        res.size = last_elem + 1; \
    } \
    memcpy(initial->individual, res.individual, res.size * sizeof(struct Individual)); \
    initial->size = res.size; \
    free(res.individual); \
} /* END MACRO */


static inline struct DoubleCouple get_info_fitness_sharing_roulette_MAXIMIZE(double * fs, uint64_t size){
    double best = -DBL_MAX;
    for(uint64_t j = 0; j < size; j++){
        if(fs[j] < best){
			best = fs[j];
		}
    }
    return (struct DoubleCouple){.val = {best, -DBL_MAX}};
}

static inline struct DoubleCouple get_info_fitness_sharing_roulette_MINIMIZE(double * fs, uint64_t size){
    double best = DBL_MAX;
    double worst = -DBL_MAX;
    for(uint64_t j = 0; j < size; j++){
        if(fs[j] != DBL_MAX && fs[j] > worst){
            worst = fs[j];
        }
        if(fs[j] < best){
			best = fs[j];
		}
    }
    return (struct DoubleCouple){.val = {best, worst}};
}

#define DECLARE_fitness_sharing_roulette(TYPE) \
void fitness_sharing_roulette_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    struct Population res; \
	res.individual = (struct Individual*) malloc(initial->size * sizeof(struct Individual)); \
    if(res.individual == NULL){ \
        MALLOC_FAIL; \
    } \
    double *fs = fitness_sharing_##TYPE(initial, params); \
	const struct DoubleCouple maxmin = get_info_fitness_sharing_roulette_##TYPE(fs, initial->size); \
    if(maxmin.val[0] == maxmin.val[1]){ \
        free(fs); \
        res.individual[0] = initial->individual[0]; \
        uint64_t last_elem = 0; \
        _Pragma("omp parallel for schedule(dynamic,1) num_threads(MAX_OMP_THREAD)") \
        for(uint64_t i = 1; i < initial->size; i++){ \
            prob probability = PROBABILITY(0.5); \
            if(WILL_HAPPEN(probability)){ \
                _Pragma("omp atomic") \
                    ++last_elem; \
                res.individual[last_elem] = initial->individual[i]; \
            } \
        } \
        res.size = last_elem + 1; \
    }else{ \
        uint64_t last_elem = -1; \
        _Pragma("omp parallel for schedule(dynamic,1) num_threads(MAX_OMP_THREAD)") \
        for(uint64_t i = 0; i < initial->size; i++){ \
            double doubleprob = probability_roulette_##TYPE(fs[i], maxmin); \
            prob probability = PROBABILITY(doubleprob); \
            if(WILL_HAPPEN(probability)){ \
                _Pragma("omp atomic") \
                    ++last_elem; \
                res.individual[last_elem] = initial->individual[i]; \
            } \
        } \
        free(fs); \
        res.size = last_elem + 1; \
    } \
    memcpy(initial->individual, res.individual, res.size * sizeof(struct Individual)); \
    initial->size = res.size; \
    free(res.individual); \
} /* END MACRO */

static inline double * slot_sizes_roulette_MAXIMIZE(const struct Population *const pop){
    double *res = malloc(sizeof(double) * pop->size);
    if(res == NULL)
        MALLOC_FAIL;

    return res;
}

static inline double * slot_sizes_roulette_MINIMIZE(const struct Population *const pop){
    double *res = malloc(sizeof(double) * pop->size);
    if(res == NULL)
        MALLOC_FAIL;

    return res;
}

#define DECLARE_roulette_reseample(TYPE) \
void roulette_##TYPE(struct Population * initial, const union SelectionParams* params){ \
    struct Population res; \
	res.size = new_size->size; \
	res.individual = (struct Individual*) malloc(new_size->size * sizeof(struct Individual)); \
    if(res.individual == NULL) \
        MALLOC_FAIL; \
    shuffle_population(initial, *mse); \
	double *slots = slot_sizes_roulette_##TYPE(initial); \
    free(slots); \
} /* END MACRO */

#define SELECTION(NAME) \
    DECLARE_##NAME(MINIMIZE) \
    DECLARE_##NAME(MAXIMIZE) /* END MACRO */
SELECTION_MACRO
#undef SELECTION

#define SELECTION(NAME) \
    const struct Selection NAME = { .type = { \
        [MINIMIZE] = NAME##_MINIMIZE, \
        [MAXIMIZE] = NAME##_MAXIMIZE \
    }}; /* END MACRO */
SELECTION_MACRO
#undef SELECTION
