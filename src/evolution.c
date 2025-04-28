#include "evolution.h"

static inline uint64_t best_individ(const struct Population *const pop, const enum FitnessType ftype){
    ASSERT(pop->size > 0);
    double winner;
	uint64_t best_individ = 0;
    switch(ftype){
        case MAXIMIZE:
            winner = -DBL_MAX;
            for (uint64_t i = 0; i < pop->size; i++) {
                if (pop->individual[i].fitness > winner){
                    winner = pop->individual[i].fitness;
                    best_individ = i;
                }
            }
        break;
        case MINIMIZE:
            winner = DBL_MAX;
            for (uint64_t i = 0; i < pop->size; i++) {
                if (pop->individual[i].fitness < winner){
                    winner = pop->individual[i].fitness;
                    best_individ = i;
                }
            }
        break;
        default:
            ASSERT(0);
        break;
    }
	return best_individ;
}

static inline struct Program mutation(const struct LGPInput *const in, const struct Program *const parent, const uint64_t max_mut_len, const uint64_t max_individ_len) {
    ASSERT(parent->size > 0);
    ASSERT(parent->size <= max_individ_len);
    ASSERT(max_individ_len <= MAX_PROGRAM_SIZE);
	uint64_t start = RAND_UPTO(parent->size);
	uint64_t last_piece = parent->size - start;
	uint64_t substitution = RAND_UPTO(last_piece);
    uint64_t from_parent = parent->size - substitution;
    uint64_t max_mutation = max_individ_len - from_parent;
    if(max_mutation == 0){
        if(last_piece){
            substitution = RAND_BOUNDS(1, last_piece);
            from_parent = parent->size - substitution;
            max_mutation = max_individ_len - from_parent;
        }else{
            start = RAND_UPTO(parent->size - 1);
            last_piece = parent->size - start;
            substitution = RAND_BOUNDS(1, last_piece);
            from_parent = parent->size - substitution;
            max_mutation = max_individ_len - from_parent;
        }
    }
    const uint64_t real_limit = (max_mutation < max_mut_len) ? max_mutation : max_mut_len;
    uint64_t mutation_len;
    if(from_parent == 0){
        mutation_len = RAND_BOUNDS(1, real_limit);
    }else{
        mutation_len = RAND_UPTO(real_limit);
    }
    struct Program mutated = { .size = from_parent + mutation_len};
    ASSERT(mutated.size > 0);
    ASSERT(mutated.size <= max_individ_len);
    if(start)
        memcpy(mutated.content, parent->content, sizeof(struct Instruction) * start);
    const uint64_t end_mutation = start + mutation_len;
    for(uint64_t i = start; i < end_mutation; i++){
        mutated.content[i] = rand_instruction(in, mutated.size);
    }
    const uint64_t restart = start + substitution;
    last_piece -= substitution;
    if(last_piece)
        memcpy(mutated.content + end_mutation, parent->content + restart, sizeof(struct Instruction) * last_piece);
    mutated.content[mutated.size] = (struct Instruction) {.op = I_EXIT, .reg = {0, 0, 0}, .addr = 0};
    return mutated;
}

static inline struct ProgramCouple crossover(const struct Program *const father, const struct Program *const mother, const uint64_t max_individ_len) {
    ASSERT(father->size > 0);
    ASSERT(father->size <= MAX_PROGRAM_SIZE);
    ASSERT(mother->size > 0);
    ASSERT(mother->size <= MAX_PROGRAM_SIZE);
    ASSERT(max_individ_len <= MAX_PROGRAM_SIZE);
    const uint64_t start_f = RAND_UPTO(father->size - 1);
	uint64_t end_f = RAND_BOUNDS(start_f + 1, father->size);
	const uint64_t start_m = RAND_UPTO(mother->size - 1);
	uint64_t end_m = RAND_BOUNDS(start_m + 1, mother->size);
	uint64_t slice_m_size = end_m - start_m;
	uint64_t slice_f_size = end_f - start_f;
    const int64_t slice_diff = ((int64_t) slice_f_size) - ((int64_t) slice_m_size);
    if(slice_diff > 0){
        if((mother->size + ((uint64_t) slice_diff)) > max_individ_len){
            end_f -= ((uint64_t) slice_diff);
            slice_f_size = slice_m_size;
        }
    }else if(slice_diff < 0){
        if((((int64_t) father->size) - slice_diff) > ((int64_t) max_individ_len)){
            end_m += ((uint64_t) slice_diff);
            slice_m_size = slice_f_size;
        }
    }
    // FIRST CROSSOVER
    const uint64_t first_f_slice_m = start_f + slice_m_size;
	const uint64_t last_of_f = father->size - end_f;
    struct Program first = {.size = first_f_slice_m + last_of_f};
    ASSERT(first.size > 0);
    ASSERT(first.size <= max_individ_len);
    if(start_f)
        memcpy(first.content, father->content, sizeof(struct Instruction) * start_f);
    memcpy(first.content + start_f, mother->content + start_m, sizeof(struct Instruction) * slice_m_size);
    if(last_of_f)
        memcpy(first.content +first_f_slice_m, father->content + end_f, sizeof(struct Instruction) * last_of_f);
    // SECOND CROSSOVER
    const uint64_t first_m_slice_f = start_m + slice_f_size;
	const uint64_t last_of_m = mother->size - end_m;
    struct Program second = {.size = first_m_slice_f + last_of_m};
    ASSERT(second.size > 0);
    ASSERT(second.size <= max_individ_len);
    if(start_m)
        memcpy(second.content, mother->content, sizeof(struct Instruction) * start_m);
    memcpy(second.content + start_m, father->content + start_f, sizeof(struct Instruction) * slice_f_size);
    if(last_of_m)
        memcpy(second.content + first_m_slice_f, mother->content + end_m, sizeof(struct Instruction) * last_of_m); 
    struct ProgramCouple res ={.prog = {first, second}};
    return res;
}

struct LGPResult evolve(const struct LGPInput *const in, const struct LGPOptions *const args){
    ASSERT(args->max_individ_len <= MAX_PROGRAM_SIZE);
    ASSERT(args->init_params.minsize > 0);
    ASSERT(args->init_params.minsize <= args->init_params.maxsize);
    ASSERT(args->init_params.maxsize <= args->max_individ_len);
    ASSERT(in->rom_size > 0);
    ASSERT(in->input_num > 0);
    ASSERT(args->mutation_prob > 0.0);
    ASSERT(args->crossover_prob > 0.0);
    double mut_int;
    const double mut_frac = modf(args->mutation_prob, &mut_int);
    const uint64_t mut_times = (uint64_t) mut_int;
    const prob mut_prob = PROBABILITY(mut_frac);
    double cross_int;
    const double cross_frac = modf(args->crossover_prob, &cross_int);
    const uint64_t cross_times = (uint64_t) cross_int;
	const prob cross_prob = PROBABILITY(cross_frac);
    uint64_t evaluations = 0;
    // POPULATION INITIALIZATION
    struct Population pop;
    if(args->initialization_func != NULL){
        struct LGPResult res = args->initialization_func(in, &(args->init_params), &(args->fitness), args->max_clock);
        evaluations = res.evaluations;
        pop = res.pop;
    }else{
        pop = args->initial_pop;
    }
    uint64_t winner = best_individ(&pop, args->fitness.type);
    if(args->verbose)
		printf("Generation 0, best_mse %lf, population_size %ld\n", pop.individual[winner].fitness, pop.size);
    if(args->fitness.type == MINIMIZE){
        if (pop.individual[winner].fitness <= args->target){
            struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = 0, .best_individ = winner};
            return res;
        }
    }else if(args->fitness.type == MAXIMIZE){
        if(pop.individual[winner].fitness >= args->target){
            struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = 0, .best_individ = winner};
            return res;
        }
    }
    // GENERATIONS LOOP
    uint64_t buffer_size = pop.size;
    uint64_t gen;
    for(gen = 1; gen <= args->generations; gen++){
        // SELECTION
        args->selection.type[args->fitness.type](&pop, &(args->select_param));
        ASSERT(pop.size > 0);
        // EVOLUTION
        uint64_t oldsize = pop.size;
        uint64_t max_pop_size = oldsize * (mut_times + 1 + 2 * (cross_times + 1) + 1);
        if(buffer_size < max_pop_size){
            pop.individual = (struct Individual *) realloc(pop.individual, sizeof(struct Individual) * max_pop_size);
            if (pop.individual == NULL){
                MALLOC_FAIL;
            }
            buffer_size = max_pop_size;
        }
#pragma omp parallel for schedule(dynamic,1)
        for(uint64_t i = 0; i < oldsize; i++){
            ASSERT(pop.individual[i].prog.size > 0);
            // MUTATION
            for(uint64_t j = 0; j < mut_times + 1; j++){
                if (WILL_HAPPEN(mut_prob)){
                    const struct Program child = mutation(in, &(pop.individual[i].prog), args->max_mutation_len, args->max_individ_len);
                    ASSERT(child.size > 0);
                    const struct Individual mutated = {.prog = child, .fitness = args->fitness.fn(in, &child, args->max_clock)};
#pragma omp critical
                    {
                        pop.individual[pop.size] = mutated;
                        pop.size += 1;
                    }
                }
            }
            // MUTATION
            for(uint64_t j = 0; j < cross_times + 1; j++){
                if (WILL_HAPPEN(cross_prob)){
                    const uint64_t mate = RAND_UPTO(oldsize - 1);
                    ASSERT(pop.individual[mate].prog.size > 0);
                    const struct ProgramCouple children = crossover(&(pop.individual[i].prog), &(pop.individual[mate].prog), args->max_individ_len);
                    ASSERT(children.prog[0].size > 0);
                    ASSERT(children.prog[1].size > 0);
                    const struct Individual child1 = {.prog = children.prog[0], .fitness = args->fitness.fn(in, &children.prog[0], args->max_clock)};
                    const struct Individual child2 = {.prog = children.prog[1], .fitness = args->fitness.fn(in, &children.prog[1], args->max_clock)};
#pragma omp critical
                    {
                        pop.individual[pop.size] = child1;
                        pop.size += 1;
                        pop.individual[pop.size] = child2;
                        pop.size += 1;
                    }
                }
            }
        }
        evaluations += (pop.size - oldsize);
        winner = best_individ(&pop, args->fitness.type);
        if(args->verbose)
            printf("Generation %ld, best_mse %lf, population_size %ld, evaluations %ld\n", gen, pop.individual[winner].fitness, pop.size, evaluations);
        if(args->fitness.type == MINIMIZE){
            if(pop.individual[winner].fitness <= args->target){
                const struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = gen, .best_individ = winner};
                return res;
            }
        }else if(args->fitness.type == MAXIMIZE){
            if(pop.individual[winner].fitness >= args->target){
                const struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = gen, .best_individ = winner};
                return res;
            }
        }
    }
    gen -= 1; // the loop will stop at when res.generations = args->generations + 1; but only args->generations generations were applied
    const struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = gen, .best_individ = winner};
    return res;
}

void print_program(const struct Program *const prog){
	printf("\n");
	for (uint64_t i = 0; i < prog->size + 1; i++) {
		printf("%s ", INSTRSET[prog->content[i].op].name);
        for(uint64_t j = 0; j < INSTRSET[prog->content[i].op].regs; j++){
            printf("REG-%d ", prog->content[i].reg[j]);
        }
        if(INSTRSET[prog->content[i].op].addr){
            printf("ADDR-%d ", prog->content[i].addr);
        }
		printf("\n");
	}
}