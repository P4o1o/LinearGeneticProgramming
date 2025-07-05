#ifndef LGP_VM_H_INCLUDED
#define LGP_VM_H_INCLUDED

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "macros.h"
#include "prob.h"

#define INSTR_NUM 87

#define INSTR_MACRO \
	INSTRUCTION(EXIT,  			0, 		0,	0,	1) \
	INSTRUCTION(LOAD_RAM,   	1,		1,	2,	0) \
	INSTRUCTION(STORE_RAM,  	2,		1,	2,	2) \
	INSTRUCTION(LOAD_ROM,   	3,		1,	4,	0) \
	INSTRUCTION(MOV,    		4, 		2,	0,	0) \
	INSTRUCTION(CMOV_Z,   		5,		2,	0,	0) \
	INSTRUCTION(CMOV_NZ,  		6,		2,	0,	0) \
	INSTRUCTION(CMOV_L,   		7,		2,	0,	0) \
	INSTRUCTION(CMOV_G,   		8,		2,	0,	0) \
	INSTRUCTION(CMOV_LE,  		9,		2,	0,	0) \
	INSTRUCTION(CMOV_GE,  		10,		2,	0,	0) \
	INSTRUCTION(CMOV_EXIST, 	11,		2, 	0,	0) \
	INSTRUCTION(CMOV_NEXIST, 	12,		2, 	0,	0) \
	INSTRUCTION(CMOV_ODD, 		13,		2,	0,	0) \
	INSTRUCTION(CMOV_EVEN, 		14,		2,	0,	0) \
	INSTRUCTION(MOV_I,   		15,		1,	1,	0) \
	INSTRUCTION(JMP,    		16,		0,	3,	1) \
	INSTRUCTION(JMP_Z,     		17,		0,	3,	1) \
	INSTRUCTION(JMP_NZ,   		18,		0,	3,	1) \
	INSTRUCTION(JMP_L,     		19,		0,	3,	1) \
	INSTRUCTION(JMP_G,     		20,		0,	3,	1) \
	INSTRUCTION(JMP_LE,    		21,		0,	3,	1) \
	INSTRUCTION(JMP_GE,    		22,		0,	3,	1) \
	INSTRUCTION(JMP_EXIST, 		23,		0, 	3,	1) \
	INSTRUCTION(JMP_NEXIST,		24,		0, 	3,	1) \
	INSTRUCTION(JMP_EVEN,  		25,		0,	3,	1) \
	INSTRUCTION(JMP_ODD,   		26,		0,	3,	1) \
	INSTRUCTION(CLC,    		27,		0,	0,	1) \
	INSTRUCTION(CMP,    		28,		2,	0,	1) \
	INSTRUCTION(TEST,   		29,		1,	0,	1) \
	INSTRUCTION(ADD,    		30,		3,	0,	0) \
	INSTRUCTION(SUB,    		31,		3,	0,	0) \
	INSTRUCTION(MUL,    		32,		3,	0,	0) \
	INSTRUCTION(DIV,    		33,		3,	0,	0) \
	INSTRUCTION(MOD,    		34,		3,	0,	0) \
	INSTRUCTION(INC,    		35,		1,	0,	0) \
	INSTRUCTION(DEC,    		36,		1,	0,	0) \
	INSTRUCTION(AND,   			37,		3,	0,	0) \
	INSTRUCTION(OR,     		38,		3,	0,	0) \
	INSTRUCTION(XOR,    		39,		3,	0,	0) \
	INSTRUCTION(NOT,    		40,		2,	0,	0) \
	INSTRUCTION(SHL,    		41,		3,	0,	0) \
	INSTRUCTION(SHR,    		42,		3,	0,	0) \
	INSTRUCTION(CAST,   		43,		2,	0,	0) \
	INSTRUCTION(NOP,    		44,		0,	0,	0) \
	INSTRUCTION(LOAD_RAM_F,  	45,		1,	2,	0) \
	INSTRUCTION(LOAD_ROM_F,  	46,		1,	4,	0) \
	INSTRUCTION(STORE_RAM_F, 	47,		1,	2,	2) \
	INSTRUCTION(MOV_F,   		48,		2,	0,	0) \
	INSTRUCTION(CMOV_Z_F,  		49,		2,	0,	0) \
	INSTRUCTION(CMOV_NZ_F, 		50,		2,	0,	0) \
	INSTRUCTION(CMOV_L_F,  		51,		2,	0,	0) \
	INSTRUCTION(CMOV_G_F,  		52,		2,	0,	0) \
	INSTRUCTION(CMOV_LE_F, 		53,		2,	0,	0) \
	INSTRUCTION(CMOV_GE_F, 		54,		2,	0,	0) \
	INSTRUCTION(MOV_I_F,  		55,		1,	5,	0) \
	INSTRUCTION(CMOV_EXIST_F, 	56,		2, 	0,	0) \
	INSTRUCTION(CMOV_NEXIST_F,	57,		2, 	0,	0) \
	INSTRUCTION(CMOV_ODD_F, 	58,		2,	0,	0) \
	INSTRUCTION(CMOV_EVEN_F, 	59,		2,	0,	0) \
	INSTRUCTION(CMP_F,   		60,		2,	0,	1) \
	INSTRUCTION(TEST_F,  		61,		1,	0,	1) \
	INSTRUCTION(ADD_F,   		62,		3,	0,	0) \
	INSTRUCTION(SUB_F,   		63,		3,	0,	0) \
	INSTRUCTION(MUL_F,   		64,		3,	0,	0) \
	INSTRUCTION(DIV_F,   		65,		3,	0,	0) \
	INSTRUCTION(SQRT,   		66,		2,	0,	0) \
	INSTRUCTION(POW,    		67,		3,	0,	0) \
	INSTRUCTION(EXP,    		68,		2,	0,	0) \
	INSTRUCTION(LN,     		69,		2,	0,	0) \
	INSTRUCTION(LOG,    		70,		2,	0,	0) \
	INSTRUCTION(LOG10,  		71,		2,	0,	0) \
	INSTRUCTION(COS,    		72,		2,	0,	0) \
	INSTRUCTION(SIN,    		73,		2,	0,	0) \
	INSTRUCTION(TAN,    		74,		2,	0,	0) \
	INSTRUCTION(ACOS,   		75,		2,	0,	0) \
	INSTRUCTION(ASIN,   		76,		2,	0,	0) \
	INSTRUCTION(ATAN,   		77,		2,	0,	0) \
	INSTRUCTION(COSH,   		78,		2,	0,	0) \
	INSTRUCTION(SINH,   		79,		2,	0,	0) \
	INSTRUCTION(TANH,   		80,		2,	0,	0) \
	INSTRUCTION(ACOSH,  		81,		2,	0,	0) \
	INSTRUCTION(ASINH,  		82,		2,	0,	0) \
	INSTRUCTION(ATANH,  		83,		2,	0,	0) \
	INSTRUCTION(CAST_F,  		84,		2,	0,	0) \
	INSTRUCTION(RAND,   		85,		1,	0,	0) \
	INSTRUCTION(ROUND,   		86,		2,	0,	0) 


#define INSTRUCTION(name, code, regs, addr, change) I_##name = code,
enum InstrCode
#if defined(C2X_SUPPORTED)
	: uint8_t
#endif
{
	INSTR_MACRO
};
#undef INSTRUCTION

struct Operation {
	const char *name;
    const uint8_t regs;
    const int8_t addr; // 1 = immediate, 2 = ram memory address, 3 = program address, 4 = rom memory address, 5 = float immediate
	const uint8_t state_changer; // 0 = no, 1 = yes, 2 = memory change
	const uint8_t code;
};

extern const struct Operation INSTRSET[INSTR_NUM];

#define INSTRUCTION(name, code, regs, addr, change) \
extern const struct Operation OP_##name;
INSTR_MACRO
#undef INSTRUCTION

#define REG_NUM 4
#define FREG_NUM 4

struct Instruction{
    uint8_t op;
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
    union Memblock *ram;
    const union Memblock *rom;
    const struct Instruction *program;
};

uint64_t run_vm(struct VirtualMachine *env, uint64_t clock_limit);

#endif
