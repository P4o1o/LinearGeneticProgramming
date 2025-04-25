#include "genetics.h"

void print_program(const struct Program* prog){
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

static inline unsigned int equal_program(const struct Program* prog1, const struct Program* prog2){
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

static inline uint64_t hash_program(const struct Program* prog){
    uint64_t hash = prog->size;
    for(uint64_t i = 0; i < prog->size; i++){
		hash += instr_to_u64(prog->content[i]);
	}
    return hash;
}


static inline struct Program rand_program(const struct LGPInput* in, const uint64_t minsize, const uint64_t maxsize) {
	struct Program res = { .size = RAND_BOUNDS(minsize, maxsize) };
    memset(&(res.content), 0, sizeof(struct Instruction) * MAX_PROGRAM_SIZE);
	for (uint64_t i = 0; i < res.size; i++) {
		struct Operation op = in->op[RAND_UPTO(in->op_size - 1)];
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
                addr = RAND_UPTO(res.size);
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
        res.content[i] = (struct Instruction) { .op = opcode, .reg = {regs[0], regs[1], regs[2]}, .addr = addr};
	}
	return res;
}

struct LGPResult rand_population(const struct LGPInput* in, const struct InitializationParams* params, const struct FitnessAssesment fitness, const uint64_t max_clock) {
	struct Population pop;
	pop.size = params->pop_size;
	pop.individual = (struct Individual *) malloc(sizeof(struct Individual) * pop.size);
	if (pop.individual == NULL) {
		MALLOC_FAIL;
	}
#pragma omp parallel for schedule(dynamic,1)
	for (uint64_t i = 0; i < pop.size; i++) {
        struct Program prog = rand_program(in, params->minsize, params->maxsize);
		pop.individual[i] = (struct Individual){ .prog = prog, .fitness = fitness.fn(in, &prog, max_clock)};
	}
    struct LGPResult res = {.generations = 0, .pop = pop, .evaluations = pop.size};
	return res;
}

double mse(const struct LGPInput *in, const struct Program* prog, const uint64_t max_clock){
    if (prog->size == 0)
		return DBL_MAX;
    struct VirtualMachine vm;
    vm.program = prog->content;
    double mse = 0;
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(&(vm.ram), 0, sizeof(union Memblock) * RAM_SIZE);
        vm.rom = &(in->memories[(in->rom_size + in->res_size)* i]);
        run_vm(&vm, max_clock);
        double result = vm.ram[0].f64;
        if (!(isfinite(result))){
			return DBL_MAX;
        }
		double actual_mse = in->memories[(in->rom_size + in->res_size)* i + in->rom_size].f64 - result;
		mse += (actual_mse * actual_mse);
    }
    if(isfinite(mse))
		return mse / (double)in->input_num;
	else
		return DBL_MAX;
}

static inline uint64_t best_individ(struct Population *pop, enum FitnessType ftype){
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

static inline struct Program mutation(const struct Program* parent, const uint64_t max_mut_len, const uint64_t max_individ_len) {

}

static inline struct ProgramCouple crossover(const struct Program* father, const struct Program *mother, const uint64_t max_individ_len) {

}

struct LGPResult evolve(const struct LGPInput* in, const struct LGPOptions* args){
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
        struct LGPResult res = args->initialization_func(in, &(args->init_params), args->fitness, args->max_clock);
        evaluations = res.evaluations;
        pop = res.pop;
    }else{
        pop = args->initial_pop;
    }
    uint64_t winner = best_individ(&pop, args->fitness.type);
    if(args->verbose)
		printf("\nGeneration 0, best_mse %lf, population_size %ld", pop.individual[winner].fitness, pop.size);
	if (winner <= args->tollerance){
		struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = 0, .best_individ = winner};
        return res;
	}
    // GENERATIONS LOOP
    uint64_t gen;
    for(gen = 1; gen <= args->generations; gen++){
        // SELECTION
        args->selection_func(&pop, &(args->select_param));
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
                    struct Program child = mutation(&(pop.individual[i].prog), args->max_mutation_len, args->max_individ_len);
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
    }
    gen -= 1; // the loop will stop at when res.generations = args->generations + 1; but only args->generations generations were applied
    struct LGPResult res = {.evaluations = evaluations, .pop = pop, .generations = gen, .best_individ = winner};
    return res;
}