#ifndef LGP_VM_H_INCLUDED
#define LGP_VM_H_INCLUDED

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "macros.h"
#include "prob.h"

#define INSTR_NUM 97

extern const uint64_t INSTR_NUM_WRAPPER;

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
	INSTRUCTION(CMOV_OVERFLOW,	15,		2,	0,	0) \
	INSTRUCTION(CMOV_ZERODIV,	16,		2,	0,	0) \
	INSTRUCTION(MOV_I,   		17,		1,	1,	0) \
	INSTRUCTION(JMP,    		18,		0,	3,	1) /*JMP instructions must be numbered from JMP to JMP_ODD*/ \
	INSTRUCTION(JMP_Z,     		19,		0,	3,	1) \
	INSTRUCTION(JMP_NZ,   		20,		0,	3,	1) \
	INSTRUCTION(JMP_L,     		21,		0,	3,	1) \
	INSTRUCTION(JMP_G,     		22,		0,	3,	1) \
	INSTRUCTION(JMP_LE,    		23,		0,	3,	1) \
	INSTRUCTION(JMP_GE,    		24,		0,	3,	1) \
	INSTRUCTION(JMP_EXIST, 		25,		0, 	3,	1) \
	INSTRUCTION(JMP_NEXIST,		26,		0, 	3,	1) \
	INSTRUCTION(JMP_EVEN,  		27,		0,	3,	1) \
	INSTRUCTION(JMP_OVERFLOW,	28,		0,	3,	1) \
	INSTRUCTION(JMP_ZERODIV,	29,		0,	3,	1) \
	INSTRUCTION(JMP_ODD,   		30,		0,	3,	1) /*JMP instructions must be numbered from JMP to JMP_ODD*/ \
	INSTRUCTION(CLC,    		31,		0,	0,	1) \
	INSTRUCTION(CMP,    		32,		2,	0,	1) \
	INSTRUCTION(TEST,   		33,		1,	0,	1) \
	INSTRUCTION(ADD,    		34,		3,	0,	0) \
	INSTRUCTION(SUB,    		35,		3,	0,	0) \
	INSTRUCTION(MUL,    		36,		3,	0,	0) \
	INSTRUCTION(DIV,    		37,		3,	0,	0) \
	INSTRUCTION(MOD,    		38,		3,	0,	0) \
	INSTRUCTION(INC,    		39,		1,	0,	0) \
	INSTRUCTION(DEC,    		40,		1,	0,	0) \
	INSTRUCTION(AND,   			41,		3,	0,	0) \
	INSTRUCTION(OR,     		42,		3,	0,	0) \
	INSTRUCTION(XOR,    		43,		3,	0,	0) \
	INSTRUCTION(NOT,    		44,		2,	0,	0) \
	INSTRUCTION(SHL,    		45,		3,	0,	0) \
	INSTRUCTION(SHR,    		46,		3,	0,	0) \
	INSTRUCTION(CAST,   		47,		2,	0,	0) \
	INSTRUCTION(NOP,    		48,		0,	0,	0) \
	INSTRUCTION(LOAD_RAM_F,  	49,		1,	2,	0) \
	INSTRUCTION(LOAD_ROM_F,  	50,		1,	4,	0) \
	INSTRUCTION(STORE_RAM_F, 	51,		1,	2,	2) \
	INSTRUCTION(MOV_F,   		52,		2,	0,	0) \
	INSTRUCTION(CMOV_Z_F,  		53,		2,	0,	0) \
	INSTRUCTION(CMOV_NZ_F, 		54,		2,	0,	0) \
	INSTRUCTION(CMOV_L_F,  		55,		2,	0,	0) \
	INSTRUCTION(CMOV_G_F,  		56,		2,	0,	0) \
	INSTRUCTION(CMOV_LE_F, 		57,		2,	0,	0) \
	INSTRUCTION(CMOV_GE_F, 		58,		2,	0,	0) \
	INSTRUCTION(MOV_I_F,  		59,		1,	5,	0) \
	INSTRUCTION(CMOV_EXIST_F, 	60,		2, 	0,	0) \
	INSTRUCTION(CMOV_NEXIST_F,	61,		2, 	0,	0) \
	INSTRUCTION(CMOV_ODD_F, 	62,		2,	0,	0) \
	INSTRUCTION(CMOV_EVEN_F, 	63,		2,	0,	0) \
	INSTRUCTION(CMOV_OVERFLOW_F,64,		2,	0,	0) \
	INSTRUCTION(CMOV_ZERODIV_F,	65,		2,	0,	0) \
	INSTRUCTION(CMP_F,   		66,		2,	0,	1) \
	INSTRUCTION(TEST_F,  		67,		1,	0,	1) \
	INSTRUCTION(ADD_F,   		68,		3,	0,	0) \
	INSTRUCTION(SUB_F,   		69,		3,	0,	0) \
	INSTRUCTION(MUL_F,   		70,		3,	0,	0) \
	INSTRUCTION(DIV_F,   		71,		3,	0,	0) \
	INSTRUCTION(SQRT,   		72,		2,	0,	0) \
	INSTRUCTION(POW,    		73,		3,	0,	0) \
	INSTRUCTION(EXP,    		74,		2,	0,	0) \
	INSTRUCTION(LN,     		75,		2,	0,	0) \
	INSTRUCTION(LOG,    		76,		2,	0,	0) \
	INSTRUCTION(LOG10,  		77,		2,	0,	0) \
	INSTRUCTION(COS,    		78,		2,	0,	0) \
	INSTRUCTION(SIN,    		79,		2,	0,	0) \
	INSTRUCTION(TAN,    		80,		2,	0,	0) \
	INSTRUCTION(ACOS,   		81,		2,	0,	0) \
	INSTRUCTION(ASIN,   		82,		2,	0,	0) \
	INSTRUCTION(ATAN,   		83,		2,	0,	0) \
	INSTRUCTION(COSH,   		84,		2,	0,	0) \
	INSTRUCTION(SINH,   		85,		2,	0,	0) \
	INSTRUCTION(TANH,   		86,		2,	0,	0) \
	INSTRUCTION(ACOSH,  		87,		2,	0,	0) \
	INSTRUCTION(ASINH,  		88,		2,	0,	0) \
	INSTRUCTION(ATANH,  		89,		2,	0,	0) \
	INSTRUCTION(CAST_F,  		90,		2,	0,	0) \
	INSTRUCTION(RAND,   		91,		1,	0,	0) \
	INSTRUCTION(ROUND,   		92,		2,	0,	0) \
	INSTRUCTION(MUL_S,   		93,		3,	0,	0) \
	INSTRUCTION(DIV_S,   		94,		3,	0,	0) \
	INSTRUCTION(ABS,   			95,		2,	0,	0) \
	INSTRUCTION(ABS_F,   		96,		2,	0,	0) 


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

#define INSTRUCTION(name, code, regs, addr, change) extern const struct Operation OP_##name;
INSTR_MACRO
#undef INSTRUCTION

#define REG_NUM 4
#define FREG_NUM 4
#define VIREG_NUM 8
#define VFREG_NUM 8

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
	unsigned int_overflow : 1;
	unsigned zero_div : 1;
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
