#ifndef GENETICS_H_INCLUDED
#define GENETICS_H_INCLUDED

#include "prob.h"
#include "vm.h"
#include "logger.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "memdebug.h"

#define MAX_PROGRAM_SIZE 254

struct Program{
    struct Instruction content[MAX_PROGRAM_SIZE + 1];
    uint64_t size;
};

struct Individual{
	struct Program prog;
    double fitness;
};

struct Population{
    struct Individual *individual;
    uint64_t size;
};

struct InstructionSet{
	const uint64_t size;
	const struct Operation *op;
};

struct LGPInput{
	const uint64_t input_num;
	const uint64_t rom_size;
	const uint64_t res_size;
	struct InstructionSet instr_set;
	union Memblock *memory; //problem1, solution1, problem2, solution2, problem3, ...
};

struct LGPResult{
	const struct Population pop; // resulting pupulation
	const uint64_t evaluations; // number of the total evaluations done in this evolution
	const uint64_t generations; // number of generations the evolution has done
	const uint64_t best_individ; // index of the best individual in the Population
};

inline struct Instruction rand_instruction(const struct LGPInput *const in, const uint64_t prog_size){
    ASSERT(prog_size > 0);
    ASSERT(prog_size <= MAX_PROGRAM_SIZE);
    ASSERT(in->rom_size > 0);
    const struct Operation op = in->instr_set.op[RAND_UPTO(in->instr_set.size - 1)];
    const enum InstrCode opcode = op.code;
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
            ASSERT(0);
        break;
    }
    uint8_t regs[3] = {0, 0, 0};
    for (uint64_t j = 0; j < op.regs; j++){
        regs[j] = RAND_UPTO(REG_NUM - 1);
    }
    const struct Instruction res = { .op = opcode, .reg = {regs[0], regs[1], regs[2]}, .addr = addr};
    return res;
}

#endif