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
	X(EXIT,   0,	0,	1) \
	X(LOAD_RAM,   1,	2,	0) \
	X(STORE_RAM,  1,	2,	2) \
	X(LOAD_ROM,   1,	4,	0) \
	X(MOV,    2,	0,	0) \
	X(CMOV_Z,   2,	0,	0) \
	X(CMOV_NZ,  2,	0,	0) \
	X(CMOV_L,   2,	0,	0) \
	X(CMOV_G,   2,	0,	0) \
	X(CMOV_LE,  2,	0,	0) \
	X(CMOV_GE,  2,	0,	0) \
	X(CMOV_EXIST, 2, 0,	0) \
	X(CMOV_NEXIST, 2, 0,	0) \
	X(CMOV_ODD, 2,	0,	0) \
	X(CMOV_EVEN, 2,	0,	0) \
	X(MOV_I,   1,	1,	0) \
	X(JMP,    0,	3,	1) \
	X(JMP_Z,     0,	3,	1) \
	X(JMP_NZ,    0,	3,	1) \
	X(JMP_L,     0,	3,	1) \
	X(JMP_G,     0,	3,	1) \
	X(JMP_LE,    0,	3,	1) \
	X(JMP_GE,    0,	3,	1) \
	X(JMP_EXIST, 0, 3,	1) \
	X(JMP_NEXIST,0, 3,	1) \
	X(JMP_EVEN,  0,	3,	1) \
	X(JMP_ODD,   0,	3,	1) \
	X(CLC,    0,	0,	1) \
	X(CMP,    2,	0,	1) \
	X(TEST,   1,	0,	1) \
	X(ADD,    3,	0,	0) \
	X(SUB,    3,	0,	0) \
	X(MUL,    3,	0,	0) \
	X(DIV,    3,	0,	0) \
	X(MOD,    3,	0,	0) \
	X(INC,    1,	0,	0) \
	X(DEC,    1,	0,	0) \
	X(AND,    3,	0,	0) \
	X(OR,     3,	0,	0) \
	X(XOR,    3,	0,	0) \
	X(NOT,    2,	0,	0) \
	X(SHL,    3,	0,	0) \
	X(SHR,    3,	0,	0) \
	X(CAST,   2,	0,	0) \
	X(NOP,    0,	0,	0) \
	X(LOAD_RAM_F,  1,	2,	0) \
	X(LOAD_ROM_F,  1,	4,	0) \
	X(STORE_RAM_F, 1,	2,	2) \
	X(MOV_F,   2,	0,	0) \
	X(CMOV_Z_F,  2,	0,	0) \
	X(CMOV_NZ_F, 2,	0,	0) \
	X(CMOV_L_F,  2,	0,	0) \
	X(CMOV_G_F,  2,	0,	0) \
	X(CMOV_LE_F, 2,	0,	0) \
	X(CMOV_GE_F, 2,	0,	0) \
	X(MOV_I_F,  1,	5,	0) \
	X(CMOV_EXIST_F, 2, 0,	0) \
	X(CMOV_NEXIST_F,2, 0,	0) \
	X(CMOV_ODD_F, 2,	0,	0) \
	X(CMOV_EVEN_F, 2,	0,	0) \
	X(CMP_F,   2,	0,	1) \
	X(TEST_F,  1,	0,	1) \
	X(ADD_F,   3,	0,	0) \
	X(SUB_F,   3,	0,	0) \
	X(MUL_F,   3,	0,	0) \
	X(DIV_F,   3,	0,	0) \
	X(SQRT,   2,	0,	0) \
	X(POW,    3,	0,	0) \
	X(EXP,    2,	0,	0) \
	X(LN,     2,	0,	0) \
	X(LOG,    2,	0,	0) \
	X(LOG10,  2,	0,	0) \
	X(COS,    2,	0,	0) \
	X(SIN,    2,	0,	0) \
	X(TAN,    2,	0,	0) \
	X(ACOS,   2,	0,	0) \
	X(ASIN,   2,	0,	0) \
	X(ATAN,   2,	0,	0) \
	X(COSH,   2,	0,	0) \
	X(SINH,   2,	0,	0) \
	X(TANH,   2,	0,	0) \
	X(ACOSH,  2,	0,	0) \
	X(ASINH,  2,	0,	0) \
	X(ATANH,  2,	0,	0) \
	X(CAST_F,  2,	0,	0) \
	X(RAND,   1,	0,	0)


#define X(name, regs, addr, change) I_##name,
enum InstrCode : uint8_t {
	INSTR_MACRO
};
#undef X

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