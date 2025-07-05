#include "vm.h"

struct MT19937 random_engines[NUMBER_OF_OMP_THREADS];

#define INSTRUCTION(name, code, regs, addr, change) \
const struct Operation OP_##name = {#name, regs, addr, change, code};
INSTR_MACRO
#undef INSTRUCTION

#define INSTRUCTION(name, code, regs, addr, change) [code] = {#name, regs, addr, change, code},
const struct Operation INSTRSET[] = {
    INSTR_MACRO
};
#undef INSTRUCTION

uint64_t run_vm(struct VirtualMachine *env, const uint64_t clock_limit){
    ASSERT(clock_limit > 0);
    uint64_t i;
    for(i = 0; i < clock_limit; i++){

        struct Instruction bytecode = env->program[env->core.prcount];
        env->core.prcount += 1;

        uint64_t imm;
        double immf;
        const uint8_t reg1 = bytecode.reg[0];
        const uint8_t reg2 = bytecode.reg[1];
        const uint8_t reg3 = bytecode.reg[2];
        const uint32_t addr = bytecode.addr;

        switch (bytecode.op){
            case I_EXIT: // EXIT
                return i;
            case I_CLC: // CLC
                memset(&env->core.flag, 0, sizeof(struct FlagReg));
            break;
            case I_CMP: // CMP
                imm = env->core.reg[reg1] - env->core.reg[reg2];
                env->core.flag.zero = (imm == ((uint64_t) 0));
                env->core.flag.negative = (imm >> ((uint64_t) 63));
                env->core.flag.odd |= (imm & ((uint64_t) 1));
                env->core.flag.exist = 1;
            break;
            case I_TEST: // TEST
                imm = env->core.reg[reg1];
                env->core.flag.zero = (imm == ((uint64_t) 0));
                env->core.flag.negative = (imm >> ((uint64_t) 63));
                env->core.flag.odd = (imm & ((uint64_t) 1));
                env->core.flag.exist = 1;
            break;
            case I_CMP_F: // CMP_F
                immf = env->core.freg[reg1] - env->core.freg[reg2];
                env->core.flag.zero = (immf == 0.0);  // 01 => 0; 10 => -; 00 => +
                env->core.flag.negative = (immf < 0.0);
                env->core.flag.exist = isfinite(immf);
            break;
            case I_TEST_F: // TEST_F
                immf = env->core.freg[reg1];
                env->core.flag.zero = (immf == 0.0);  // 01 => 0; 10 => -; 00 => +
                env->core.flag.negative = (immf < 0.0);
                env->core.flag.exist = isfinite(immf);
            break;
            case I_JMP: // JUMP
                env->core.prcount=addr;
            break;
            case I_JMP_Z: // JZ
                if (env->core.flag.zero) env->core.prcount = addr;
            break;
            case I_JMP_NZ: // JNZ
                if (! env->core.flag.zero) env->core.prcount = addr;
            break;
            case I_JMP_EXIST: // JNAN
                if (env->core.flag.exist) env->core.prcount = addr;
            break;
            case I_JMP_NEXIST: // JNNAN
                if (! env->core.flag.exist) env->core.prcount = addr;
            break;
            case I_JMP_L: // JL
                if (env->core.flag.negative) env->core.prcount = addr;
            break;
            case I_JMP_G: // JG
                if ((! env->core.flag.zero) && (! env->core.flag.negative)) env->core.prcount = addr;
            break;
            case I_JMP_LE: // JLE
                if (env->core.flag.zero || env->core.flag.negative) env->core.prcount = addr;
            break;
            case I_JMP_GE: // JGE
                if (! env->core.flag.negative) env->core.prcount = addr;
            break;
            case I_JMP_ODD:
                if (env->core.flag.odd) env->core.prcount = addr;
            break;
            case I_JMP_EVEN:
                if (! env->core.flag.odd) env->core.prcount = addr;
            break;
            case I_MOV: // MOVE
                env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOV_I: // MOVI
                env->core.reg[reg1] = ((uint64_t) addr);
            break;
            case I_CMOV_Z: // MOVZ
                if (env->core.flag.zero) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_NZ: // MOVNZ
                if (! env->core.flag.zero) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_L: // MOVL
                if (env->core.flag.negative) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_G: // MOVG
                if ((! env->core.flag.zero) && (! env->core.flag.negative)) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_LE: // MOVLE
                if (env->core.flag.zero || env->core.flag.negative) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_GE: // MOVGE
                if (! env->core.flag.negative) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_EXIST: // MOVGE
                if (env->core.flag.exist) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_NEXIST: // MOVGE
                if (! env->core.flag.exist) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_ODD:
                if (env->core.flag.odd) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_CMOV_EVEN:
                if (! env->core.flag.odd) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_LOAD_RAM: // LOAD
                env->core.reg[reg1] = env->ram[addr].i64;
            break;
            case I_LOAD_ROM: // LOAD
                env->core.reg[reg1] = env->rom[addr].i64;
            break;
            case I_STORE_RAM: // STORE
                env->ram[addr].i64 = env->core.reg[reg1];                
            break;
            case I_MOV_F: // MOVF
                env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOV_I_F: // MOVFI
                env->core.freg[reg1] = addr;// WEWE
            break;
            case I_CMOV_Z_F: // MOVFZ
                if (env->core.flag.zero) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_NZ_F: // MOVFNZ
                if (! env->core.flag.zero) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_L_F: // MOVFL
                if ((env->core.flag.negative)) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_G_F: // MOVFG
                if ((! env->core.flag.zero) && (! env->core.flag.negative)) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_LE_F: // MOVFLE
                if (env->core.flag.zero || env->core.flag.negative) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_GE_F: // MOVFGE
                if (! env->core.flag.negative) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_EXIST_F: // MOVGE
                if (env->core.flag.exist) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_NEXIST_F: // MOVGE
                if (! env->core.flag.exist) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_ODD_F:
                if (env->core.flag.odd) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_CMOV_EVEN_F:
                if (! env->core.flag.odd) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_LOAD_RAM_F: // LOAD
                env->core.freg[reg1] = env->ram[addr].f64;
            break;
            case I_LOAD_ROM_F: // LOAD
                env->core.freg[reg1] = env->rom[addr].f64;
            break;
            case I_STORE_RAM_F: // STORE
                env->ram[addr].f64 = env->core.freg[reg1];                
            break;

            case I_NOP:  // NOP
            break;

            case I_ADD:  // ADD
                env->core.reg[reg1] = env->core.reg[reg2] + env->core.reg[reg3];
            break;
            case I_SUB:  // SUB
                env->core.reg[reg1] = env->core.reg[reg2] - env->core.reg[reg3];
            break;
            case I_MUL:  // MUL
                env->core.reg[reg1] = env->core.reg[reg2] * env->core.reg[reg3];
            break;
            case I_DIV: // DIV
                if(env->core.reg[reg3] != 0)
                    env->core.reg[reg1] = env->core.reg[reg2] / env->core.reg[reg3];
            break;
            case I_ADD_F:  // ADDF
                env->core.freg[reg1] = env->core.freg[reg2] + env->core.freg[reg3];
            break;
            case I_SUB_F:  // SUBF
                env->core.freg[reg1] = env->core.freg[reg2] - env->core.freg[reg3];
            break;
            case I_MUL_F:  // MUL
                env->core.freg[reg1] = env->core.freg[reg2] * env->core.freg[reg3];
            break;
            case I_DIV_F: // DIVF
                env->core.freg[reg1] = env->core.freg[reg2] / env->core.freg[reg3];
            break;
            case I_MOD: // MOD
                if(env->core.reg[reg3] != 0)
                    env->core.reg[reg1] = env->core.reg[reg2] % env->core.reg[reg3];
            break;
            case I_INC: // INC
                env->core.reg[reg1] += 1;
            break;
            case I_DEC: // DEC
                env->core.reg[reg1] -= 1;
            break;

            case I_AND: // AND
                env->core.reg[reg1] = env->core.reg[reg2] & env->core.reg[reg3];
            break;
            case I_OR: // OR
                env->core.reg[reg1] = env->core.reg[reg2] | env->core.reg[reg3];
            break;
            case I_XOR: // XOR
                env->core.reg[reg1] = env->core.reg[reg2] ^ env->core.reg[reg3];
            break;
            case I_NOT: // NOT
                env->core.reg[reg1] = ~ env->core.reg[reg1];
            break;
            case I_SHL: // SHL
                if(env->core.reg[reg3] < 64)
                    env->core.reg[reg1] = env->core.reg[reg2] << env->core.reg[reg3];
            break;
            case I_SHR: // SHR
                if(env->core.reg[reg3] < 64)
                    env->core.reg[reg1] = env->core.reg[reg2] >> env->core.reg[reg3];
            break;

            case I_CAST: // CAST
                immf = fabs(env->core.freg[reg2]);
                if(immf < ((double) 0xFFFFFFFFFFFFFFFFULL / 2ULL)){
                    env->core.reg[reg1] = (uint64_t) immf;
                    if(env->core.freg[reg2] < ((uint64_t) 0)){
                        env->core.reg[reg1] |= (((uint64_t) 1) << ((uint64_t) 63));
                    }
                }
            break;
            case I_ROUND: // CAST
                immf = fabs(env->core.freg[reg2]);
                if(immf < ((double) 0xFFFFFFFFFFFFFFFFULL / 2ULL)){
                    env->core.reg[reg1] = (uint64_t) (immf + 0.5);
                    if(env->core.reg[reg1] && (env->core.freg[reg2] < ((uint64_t) 0))){
                        env->core.reg[reg1] |= (((uint64_t) 1) << ((uint64_t) 63));
                    }
                }
            break;
            case I_CAST_F: // CASTF
                env->core.freg[reg1] = (double) env->core.reg[reg2];
            break;

            case I_COS: // COS
                env->core.freg[reg1] = cos(env->core.freg[reg2]);
            break;
            case I_SIN: // SEN
                env->core.freg[reg1] = sin(env->core.freg[reg2]);
            break;
            case I_TAN: // TAN
                env->core.freg[reg1] = tan(env->core.freg[reg2]);
            break;
            case I_ACOS: // ACOS
                env->core.freg[reg1] = acos(env->core.freg[reg2]);
            break;
            case I_ASIN: // ASEN
                env->core.freg[reg1] = asin(env->core.freg[reg2]);
            break;
            case I_ATAN: // ATAN
                env->core.freg[reg1] = atan(env->core.freg[reg2]);
            break;
            case I_COSH: // COSH
                env->core.freg[reg1] = cosh(env->core.freg[reg2]);
            break;
            case I_SINH: // SENH
                env->core.freg[reg1] = sinh(env->core.freg[reg2]);
            break;
            case I_TANH: // TANH
                env->core.freg[reg1] = tanh(env->core.freg[reg2]);
            break;
            case I_ACOSH: // ACOSH
                env->core.freg[reg1] = acosh(env->core.freg[reg2]);
            break;
            case I_ASINH: // ASENH
                env->core.freg[reg1] = asinh(env->core.freg[reg2]);
            break;
            case I_ATANH: // ATANH
                env->core.freg[reg1] = atanh(env->core.freg[reg2]);
            break;

            case I_SQRT: // SQRT
                env->core.freg[reg1] = sqrt(env->core.freg[reg2]);
            break;
            case I_POW: // POW
                env->core.freg[reg1] = pow(env->core.freg[reg2], env->core.freg[reg3]);
            break;
            case I_EXP: // EXP
                env->core.freg[reg1] = exp(env->core.freg[reg2]);
            break;
            case I_LOG: // LOG
                env->core.freg[reg1] = log(env->core.freg[reg2]);
            break;
            case I_LN: // LN
                env->core.freg[reg1] = log2(env->core.freg[reg2]);
            break;
            case I_LOG10: // LOG10
                env->core.freg[reg1] = log10(env->core.freg[reg2]);
            break;
            case I_RAND:
                env->core.reg[reg1] = random();
            break;
            default:
                ASSERT(0);
            break;
        }
    }
    return i;
}
