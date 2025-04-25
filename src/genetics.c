#include "genetics.h"

void print_program(const struct Program *const prog){
	printf("\n");
	for (uint64_t i = 0; i < prog->size; i++) {
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

static inline uint64_t instr_to_u64(const struct Instruction inst){
    union InstrToU64 res = { .instr = inst};
    return res.u64;
}

static inline unsigned int equal_program(const struct Program *const prog1, const struct Program *const prog2){
	if(prog1->size != prog2->size){
		return 0;
	}
	for(uint64_t i = 0; i < prog1->size; i++){
		if(instr_to_u64(prog1->content[i]) != instr_to_u64(prog2->content[i])){
            return 0;
        }
	}
	return 1;
}

static inline uint64_t hash_program(const struct Program *const prog){
    uint64_t hash = prog->size;
    for(uint64_t i = 0; i < prog->size; i++){
		hash += instr_to_u64(prog->content[i]);
	}
    return hash;
}

static inline struct Instruction rand_instruction(const struct LGPInput *const in, const uint64_t prog_size){
    struct Operation op = in->instr_set.op[RAND_UPTO(in->instr_set.size - 1)];
    enum InstrCode opcode = op.code;
    uint32_t addr;
    switch(op.addr){
        case 1:
            addr = rand();
        break;
        case 2:
            addr = RAND_UPTO(RAM_SIZE - 1);
        break;
        case 3:
            addr = RAND_UPTO(prog_size + 1);
        break;
        case 4:
            addr = RAND_UPTO(in->rom_size - 1);
        break;
        case 5:
            addr = RAND_DOUBLE();
        break;
        case 0:
            addr = 0;
        break;
        default:
            unreachable();
        break;
    }
    uint8_t regs[3] = {0, 0, 0};
    for (uint64_t j = 0; j < op.regs; j++){
        regs[j] = RAND_UPTO(REG_NUM - 1);
    }
    struct Instruction res = { .op = opcode, .reg = {regs[0], regs[1], regs[2]}, .addr = addr};
    return res;
}


static inline struct Program rand_program(const struct LGPInput *const in, const uint64_t minsize, const uint64_t maxsize) {
	struct Program res = { .size = RAND_BOUNDS(minsize, maxsize) };
	for (uint64_t i = 0; i < res.size; i++) {
        res.content[i] = rand_instruction(in, res.size);
	}
    res.content[res.size + 1] = (struct Instruction) {.op = I_EXIT, .reg = {0, 0, 0}, .addr = 0};
	return res;
}

struct LGPResult rand_population(const struct LGPInput *const in, const struct InitializationParams *const params, const struct FitnessAssesment *const fitness, const uint64_t max_clock) {
	struct Population pop;
	pop.size = params->pop_size;
	pop.individual = (struct Individual *) malloc(sizeof(struct Individual) * pop.size);
	if (pop.individual == NULL) {
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic,1)
	for (uint64_t i = 0; i < pop.size; i++) {
        struct Program prog = rand_program(in, params->minsize, params->maxsize);
		pop.individual[i] = (struct Individual){ .prog = prog, .fitness = fitness->fn(in, &prog, max_clock)};
	}
    struct LGPResult res = {.generations = 0, .pop = pop, .evaluations = pop.size};
	return res;
}

double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock){
    if (prog->size == 0)
		return DBL_MAX;
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mse = 0;
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(&(vm.ram), 0, sizeof(union Memblock) * RAM_SIZE);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
			return DBL_MAX;
        }
		double actual_mse = in->memory[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
		mse += (actual_mse * actual_mse);
    }
    if(isfinite(mse))
		return mse / (double)in->input_num;
	else
		return DBL_MAX;
}

const struct FitnessAssesment MSE = {.fn = mse, .type = MINIMIZE};

static inline uint64_t best_individ(const struct Population *const pop, const enum FitnessType ftype){
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
    }
	return best_individ;
}

static inline void shuffle_population(struct Population* pop) {
	for (uint64_t i = pop->size - 1; i > 0; i--) {
		uint64_t j = RAND_UPTO(i - 1);
		struct Individual tmp = pop->individual[i];
		pop->individual[i] = pop->individual[j];
		pop->individual[j] = tmp;
	}
}

void tournament(struct Population* initial, const union SelectionParams* tourn_size, const enum FitnessType ftype){
    shuffle_population(initial);
    uint64_t tournaments = initial->size / tourn_size->size;
	uint64_t small_tourn = initial->size % tourn_size->size;
	struct Population res;
	res.size = tournaments + (small_tourn != 0);
	res.individual = (struct Individual*) malloc(sizeof(struct Individual) * res.size);
	if (res.individual == NULL){
		MALLOC_FAIL;
    }
    if(ftype == MINIMIZE){
#pragma omp parallel for schedule(static,1)
        for (uint64_t i = 0; i < tournaments; i++) {
            uint64_t winner = 0;
            for (uint64_t j = 1; j < tourn_size->size; j++) {
                if (initial->individual[i * tourn_size->size + winner].fitness > initial->individual[i * tourn_size->size + j].fitness) {
                    winner = j;
                }
            }
            res.individual[i] = initial->individual[i * tourn_size->size + winner];
        }
        if (small_tourn) {
            uint64_t winner = 1; // initial->size - 1
            for (uint64_t i = 2; i <= small_tourn; i++) {
                if (initial->individual[initial->size - winner].fitness > initial->individual[initial->size - i].fitness) {
                    winner = i;
                }
            }
            res.individual[res.size - 1] = initial->individual[initial->size - winner];
        }
    }else if(ftype == MAXIMIZE){
#pragma omp parallel for schedule(static,1)
        for (uint64_t i = 0; i < tournaments; i++) {
            uint64_t winner = 0;
            for (uint64_t j = 1; j < tourn_size->size; j++) {
                if (initial->individual[i * tourn_size->size + winner].fitness < initial->individual[i * tourn_size->size + j].fitness) {
                    winner = j;
                }
            }
            res.individual[i] = initial->individual[i * tourn_size->size + winner];
        }
        if (small_tourn) {
            uint64_t winner = 1; // initial->size - 1
            for (uint64_t i = 2; i <= small_tourn; i++) {
                if (initial->individual[initial->size - winner].fitness < initial->individual[initial->size - i].fitness) {
                    winner = i;
                }
            }
            res.individual[res.size - 1] = initial->individual[initial->size - winner];
        }
    }
    free(initial->individual);
    initial->individual = res.individual;
    initial->size = res.size;
}

static inline struct Program mutation(const struct LGPInput *const in, const struct Program *const parent, const uint64_t max_mut_len, const uint64_t max_individ_len) {
	uint64_t start = RAND_UPTO(parent->size);
	uint64_t last_piece = parent->size - start;
	uint64_t substitution = RAND_BOUNDS(0, last_piece);
    uint64_t from_parent = parent->size - substitution;
    uint64_t max_mutation = max_individ_len - from_parent;
    if(max_mutation == 0){
        if(last_piece){
            substitution = RAND_BOUNDS(1, last_piece);
            from_parent = parent->size - substitution;
            max_mutation = max_individ_len - from_parent;
        }else{
            uint64_t start = RAND_UPTO(parent->size - 1);
            uint64_t last_piece = parent->size - start;
            substitution = RAND_BOUNDS(1, last_piece);
            from_parent = parent->size - substitution;
            max_mutation = max_individ_len - from_parent;
        }
    }
    const uint64_t real_limit = (max_mutation < max_mut_len) ? max_mutation : max_mut_len;
    const uint64_t mutation_len = RAND_UPTO(real_limit);
    struct Program mutated;
    mutated.size = from_parent + mutation_len;
    if(mutated.size == 0){
        mutated.content[0] = rand_instruction(in, 1);
        mutated.size = 1;
        mutated.content[max_individ_len + 1] = (struct Instruction) {.op = I_EXIT, .reg = {0, 0, 0}, .addr = 0};
        return mutated;
    }
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
    mutated.content[max_individ_len + 1] = (struct Instruction) {.op = I_EXIT, .reg = {0, 0, 0}, .addr = 0};
    return mutated;
}

static inline struct ProgramCouple crossover(const struct Program *const father, const struct Program *const mother, const uint64_t max_individ_len) {
    const uint64_t start_f = RAND_UPTO(father->size - 1);
	uint64_t end_f = RAND_BOUNDS(start_f + 1, father->size);
	const uint64_t start_m = RAND_UPTO(mother->size - 1);
	uint64_t end_m = RAND_BOUNDS(start_m + 1, mother->size);
	uint64_t slice_m_size = end_m - start_m;
	uint64_t slice_f_size = end_f - start_f;
    const int64_t slice_diff = slice_f_size - slice_m_size;
    if(slice_diff > 0){
        if(mother->size + slice_diff > max_individ_len){
            end_f -= slice_diff;
            slice_f_size = slice_m_size;
        }
    }else if(slice_diff < 0){
        if(father->size - slice_diff > max_individ_len){
            end_m += slice_diff;
            slice_m_size = slice_f_size;
        }
    }
    // FIRST CROSSOVER
    const uint64_t first_f_slice_m = start_f + slice_m_size;
	const uint64_t last_of_f = father->size - end_f;
    struct Program first = {.size = first_f_slice_m + last_of_f};
    if(start_f)
        memcpy(first.content, father->content, sizeof(struct Instruction) * start_f);
    memcpy(first.content + start_f, mother->content + start_m, sizeof(struct Instruction) * slice_m_size);
    if(last_of_f)
        memcpy(first.content +first_f_slice_m, father->content + end_f, sizeof(struct Instruction) * last_of_f);
    // SECOND CROSSOVER
    const uint64_t first_m_slice_f = start_m + slice_f_size;
	const uint64_t last_of_m = mother->size - end_m;
    struct Program second = {.size = first_m_slice_f + last_of_m};
    if(start_m)
        memcpy(second.content, mother->content, sizeof(struct Instruction) * start_m);
    memcpy(second.content + start_m, father->content + start_f, sizeof(struct Instruction) * slice_f_size);
    if(last_of_m)
        memcpy(second.content + first_m_slice_f, mother->content + end_m, sizeof(struct Instruction) * last_of_m); 
    struct ProgramCouple res ={.prog = {first, second}};
    return res;
}

struct LGPResult evolve(const struct LGPInput *const in, const struct LGPOptions *const args){
    double mut_int;
    const double mut_frac = modf(args->mutation_prob, &mut_int);
    const uint64_t mut_times = (uint64_t) mut_int;
    const prob mut_prob = PROBABILITY(mut_frac);
    double cross_int;
    const double cross_frac = modf(args->mutation_prob, &cross_int);
    const uint64_t cross_times = (uint64_t) cross_int;
	const prob cross_prob = PROBABILITY(cross_frac);
	random_seed_init();
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
		printf("\nGeneration 0, best_mse %lf, population_size %ld", pop.individual[winner].fitness, pop.size);
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
    uint64_t gen;
    for(gen = 1; gen <= args->generations; gen++){
        // SELECTION
        args->selection_func(&pop, &(args->select_param), args->fitness.type);
        // EVOLUTION
        pop.individual = (struct Individual *) realloc(pop.individual, sizeof(struct Individual *) * (pop.size * (mut_times + 1 + 2 * (cross_times + 1))));
        if (pop.individual == NULL){
			MALLOC_FAIL;
        }
        uint64_t oldsize = pop.size;
#pragma omp parallel for schedule(dynamic,1)
        for(uint64_t i = 0; i < oldsize; i++){
            for(uint64_t j = 0; j < mut_times + 1; j++){
                if (WILL_HAPPEN(mut_prob)){
                    struct Program child = mutation(in, &(pop.individual[i].prog), args->max_mutation_len, args->max_individ_len);
                    struct Individual mutated = {.prog = child, .fitness = args->fitness.fn(in, &child, args->max_clock)};
#pragma omp critical
                    {
                        pop.individual[pop.size] = mutated;
                        pop.size += 1;
                    }
                }
            }
            for(uint64_t j = 0; j < cross_times + 1; j++){
                if (WILL_HAPPEN(cross_prob)){
                    uint64_t mate = RAND_UPTO(oldsize - 1);
                    struct ProgramCouple children = crossover(&(pop.individual[i].prog), &(pop.individual[mate].prog), args->max_individ_len);
                    struct Individual child1 = {.prog = children.prog[0], .fitness = args->fitness.fn(in, &children.prog[0], args->max_clock)};
                    struct Individual child2 = {.prog = children.prog[1], .fitness = args->fitness.fn(in, &children.prog[1], args->max_clock)};
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
            printf("\nGeneration %ld, best_mse %lf, population_size %ld, evaluations %ld", gen, pop.individual[winner].fitness, pop.size, evaluations);
        if(args->fitness.type == MINIMIZE){
            if(pop.individual[winner].fitness <= args->target){
                struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = gen, .best_individ = winner};
                return res;
            }
        }else if(args->fitness.type == MAXIMIZE){
            if(pop.individual[winner].fitness >= args->target){
                struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = gen, .best_individ = winner};
                return res;
            }
        }
    }
    gen -= 1; // the loop will stop at when res.generations = args->generations + 1; but only args->generations generations were applied
    struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = gen, .best_individ = winner};
    return res;
}