#ifndef LGP_VM_H_INCLUDED
#define LGP_VM_H_INCLUDED

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define INSTR_MACRO \
	X(EXIT,   0,	0,	1) \
	X(LOAD,   1,	2,	0) \
	X(STORE,  1,	2,	2) \
	X(MOV,    2,	0,	0) \
	X(MOVZ,   2,	0,	0) \
	X(MOVNZ,  2,	0,	0) \
	X(MOVL,   2,	0,	0) \
	X(MOVG,   2,	0,	0) \
	X(MOVLE,  2,	0,	0) \
	X(MOVGE,  2,	0,	0) \
	X(MOVODD, 2,	0,	0) \
	X(MOVEVEN, 2,	0,	0) \
	X(MOVI,   1,	1,	0) \
	X(JMP,    0,	3,	1) \
	X(JZ,     0,	3,	1) \
	X(JNZ,    0,	3,	1) \
	X(JL,     0,	3,	1) \
	X(JG,     0,	3,	1) \
	X(JLE,    0,	3,	1) \
	X(JGE,    0,	3,	1) \
	X(JEVEN,  0,	3,	1) \
	X(JODD,   0,	3,	1) \
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
	X(LOADF,  1,	2,	0) \
	X(STOREF, 1,	2,	2) \
	X(MOVF,   2,	0,	0) \
	X(MOVFZ,  2,	0,	0) \
	X(MOVFNZ, 2,	0,	0) \
	X(MOVFL,  2,	0,	0) \
	X(MOVFG,  2,	0,	0) \
	X(MOVFLE, 2,	0,	0) \
	X(MOVFGE, 2,	0,	0) \
	X(MOVFI,  1,	1,	0) \
	X(MOVFODD, 2,	0,	0) \
	X(MOVFEVEN, 2,	0,	0) \
	X(CMPF,   2,	0,	1) \
	X(TESTF,  1,	0,	1) \
	X(ADDF,   3,	0,	0) \
	X(SUBF,   3,	0,	0) \
	X(MULF,   3,	0,	0) \
	X(DIVF,   3,	0,	0) \
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
	X(CASTF,  2,	0,	0) \
	X(RAND,   1,	0,	0)


#define X(name, regs, addr, change) I_##name,
enum InstrCode : uint8_t {
	INSTR_MACRO
};
#undef X

struct Operation {
	char *name;
    uint8_t regs;
    uint8_t addr; // 1 = immediate, 2 = memory address, 3 = program address
	uint8_t state_changer; // 0 = no, 1 = yes, 2 = memory change
	enum InstrCode code;
};

extern const struct Operation INSTRSET[];

#define REG_NUM 4
#define FREG_NUM 4

struct Instruction{
    enum InstrCode op;
    uint8_t reg[3]
    uint32_t addr;
};

union memblock{
    uint64_t i64;
    double f64;
};

struct FlagReg{
    unsigned odd : 1;
    unsigned negative : 1;
    unsigned zero : 1;
};

struct Core {
    uint64_t reg[REG_NUM];
    double freg[FREG_NUM];
    struct FlagReg flag;
    uint64_t prcount;
};

struct VirtualMachine{
    struct Core core;
    union memblock *vmem;
    struct Instruction *program;
};

void setup_vm(struct VirtualMachine *vm, struct Instruction *prog, uint64_t ram_size, uint64_t locked_addr)
uint64_t run_vm(struct VirtualMachine *env, uint64_t clock_limit);

#endif