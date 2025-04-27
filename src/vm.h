#ifndef LGP_VM_H_INCLUDED
#define LGP_VM_H_INCLUDED

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "memdebug.h"

#define ASSERT(x) \
	do \
		if(!(x)) unreachable(); \
	while(0)

#define INSTR_NUM 86

#define INSTR_MACRO \
	INSTRUCTION(EXIT,   0,	0,	1) \
	INSTRUCTION(LOAD_RAM,   1,	2,	0) \
	INSTRUCTION(STORE_RAM,  1,	2,	2) \
	INSTRUCTION(LOAD_ROM,   1,	4,	0) \
	INSTRUCTION(MOV,    2,	0,	0) \
	INSTRUCTION(CMOV_Z,   2,	0,	0) \
	INSTRUCTION(CMOV_NZ,  2,	0,	0) \
	INSTRUCTION(CMOV_L,   2,	0,	0) \
	INSTRUCTION(CMOV_G,   2,	0,	0) \
	INSTRUCTION(CMOV_LE,  2,	0,	0) \
	INSTRUCTION(CMOV_GE,  2,	0,	0) \
	INSTRUCTION(CMOV_EXIST, 2, 0,	0) \
	INSTRUCTION(CMOV_NEXIST, 2, 0,	0) \
	INSTRUCTION(CMOV_ODD, 2,	0,	0) \
	INSTRUCTION(CMOV_EVEN, 2,	0,	0) \
	INSTRUCTION(MOV_I,   1,	1,	0) \
	INSTRUCTION(JMP,    0,	3,	1) \
	INSTRUCTION(JMP_Z,     0,	3,	1) \
	INSTRUCTION(JMP_NZ,    0,	3,	1) \
	INSTRUCTION(JMP_L,     0,	3,	1) \
	INSTRUCTION(JMP_G,     0,	3,	1) \
	INSTRUCTION(JMP_LE,    0,	3,	1) \
	INSTRUCTION(JMP_GE,    0,	3,	1) \
	INSTRUCTION(JMP_EXIST, 0, 3,	1) \
	INSTRUCTION(JMP_NEXIST,0, 3,	1) \
	INSTRUCTION(JMP_EVEN,  0,	3,	1) \
	INSTRUCTION(JMP_ODD,   0,	3,	1) \
	INSTRUCTION(CLC,    0,	0,	1) \
	INSTRUCTION(CMP,    2,	0,	1) \
	INSTRUCTION(TEST,   1,	0,	1) \
	INSTRUCTION(ADD,    3,	0,	0) \
	INSTRUCTION(SUB,    3,	0,	0) \
	INSTRUCTION(MUL,    3,	0,	0) \
	INSTRUCTION(DIV,    3,	0,	0) \
	INSTRUCTION(MOD,    3,	0,	0) \
	INSTRUCTION(INC,    1,	0,	0) \
	INSTRUCTION(DEC,    1,	0,	0) \
	INSTRUCTION(AND,    3,	0,	0) \
	INSTRUCTION(OR,     3,	0,	0) \
	INSTRUCTION(XOR,    3,	0,	0) \
	INSTRUCTION(NOT,    2,	0,	0) \
	INSTRUCTION(SHL,    3,	0,	0) \
	INSTRUCTION(SHR,    3,	0,	0) \
	INSTRUCTION(CAST,   2,	0,	0) \
	INSTRUCTION(NOP,    0,	0,	0) \
	INSTRUCTION(LOAD_RAM_F,  1,	2,	0) \
	INSTRUCTION(LOAD_ROM_F,  1,	4,	0) \
	INSTRUCTION(STORE_RAM_F, 1,	2,	2) \
	INSTRUCTION(MOV_F,   2,	0,	0) \
	INSTRUCTION(CMOV_Z_F,  2,	0,	0) \
	INSTRUCTION(CMOV_NZ_F, 2,	0,	0) \
	INSTRUCTION(CMOV_L_F,  2,	0,	0) \
	INSTRUCTION(CMOV_G_F,  2,	0,	0) \
	INSTRUCTION(CMOV_LE_F, 2,	0,	0) \
	INSTRUCTION(CMOV_GE_F, 2,	0,	0) \
	INSTRUCTION(MOV_I_F,  1,	5,	0) \
	INSTRUCTION(CMOV_EXIST_F, 2, 0,	0) \
	INSTRUCTION(CMOV_NEXIST_F,2, 0,	0) \
	INSTRUCTION(CMOV_ODD_F, 2,	0,	0) \
	INSTRUCTION(CMOV_EVEN_F, 2,	0,	0) \
	INSTRUCTION(CMP_F,   2,	0,	1) \
	INSTRUCTION(TEST_F,  1,	0,	1) \
	INSTRUCTION(ADD_F,   3,	0,	0) \
	INSTRUCTION(SUB_F,   3,	0,	0) \
	INSTRUCTION(MUL_F,   3,	0,	0) \
	INSTRUCTION(DIV_F,   3,	0,	0) \
	INSTRUCTION(SQRT,   2,	0,	0) \
	INSTRUCTION(POW,    3,	0,	0) \
	INSTRUCTION(EXP,    2,	0,	0) \
	INSTRUCTION(LN,     2,	0,	0) \
	INSTRUCTION(LOG,    2,	0,	0) \
	INSTRUCTION(LOG10,  2,	0,	0) \
	INSTRUCTION(COS,    2,	0,	0) \
	INSTRUCTION(SIN,    2,	0,	0) \
	INSTRUCTION(TAN,    2,	0,	0) \
	INSTRUCTION(ACOS,   2,	0,	0) \
	INSTRUCTION(ASIN,   2,	0,	0) \
	INSTRUCTION(ATAN,   2,	0,	0) \
	INSTRUCTION(COSH,   2,	0,	0) \
	INSTRUCTION(SINH,   2,	0,	0) \
	INSTRUCTION(TANH,   2,	0,	0) \
	INSTRUCTION(ACOSH,  2,	0,	0) \
	INSTRUCTION(ASINH,  2,	0,	0) \
	INSTRUCTION(ATANH,  2,	0,	0) \
	INSTRUCTION(CAST_F,  2,	0,	0) \
	INSTRUCTION(RAND,   1,	0,	0)


#define INSTRUCTION(name, regs, addr, change) I_##name,
enum InstrCode : uint8_t {
	INSTR_MACRO
};
#undef INSTRUCTION

struct Operation {
	const char *name;
    const uint8_t regs;
    const int8_t addr; // 1 = immediate, 2 = ram memory address, 3 = program address, 4 = rom memory address, 5 = float immediate
	const uint8_t state_changer; // 0 = no, 1 = yes, 2 = memory change
	const enum InstrCode code;
};

extern const struct Operation INSTRSET[INSTR_NUM];

#define REG_NUM 4
#define FREG_NUM 4
#define RAM_SIZE 64

struct Instruction{
    enum InstrCode op;
    uint8_t reg[3];
    uint32_t addr;
};

union Memblock{
    uint64_t i64;
    double f64;
};

struct FlagReg{
    unsigned odd : 1;
    unsigned negative : 1;
    unsigned zero : 1;
	unsigned exist : 1;
};

struct Core {
    uint64_t reg[REG_NUM];
    double freg[FREG_NUM];
    struct FlagReg flag;
    uint64_t prcount;
};

struct VirtualMachine{
    struct Core core;
    union Memblock ram[RAM_SIZE];
    const union Memblock *rom;
    const struct Instruction *program;
};

void setup_vm(struct VirtualMachine *vm, struct Instruction *prog, uint64_t ram_size, uint64_t locked_addr);
uint64_t run_vm(struct VirtualMachine *env, uint64_t clock_limit);

#endif