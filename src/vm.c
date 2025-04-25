#include "vm.h"

#define X(name, regs, addr, change) [I_##name] = {#name, regs, addr, change, I_##name},
const struct Operation INSTRSET[] = {
    INSTR_MACRO
};
#undef X

void setup_vm(struct VirtualMachine *vm, struct Instruction *prog, uint64_t ram_size, uint64_t locked_addr){
    memset(vm->core, 0, sizeof(struct Core));
    memset(vm->vmem + locked_addr, 0, (ram_size - locked_addr) * sizeof(union memblock));
    vm->program = prog;
}

uint64_t run_vm(struct VirtualMachine *env, const uint64_t clock_limit){
    for(uint64_t i = 0; i < clock_limit; i++){

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
                env->core.flag.zero = (imm == 0);  // 01 => 0; 10 => -; 00 => +
                env->core.flag.negative = (imm >> ((uint64_t) 63));
                env->core.flag.odd |= ((uint32_t) (imm & 1)); // 1-- odd, 0-- even
            break;
            case I_TEST: // TEST
                imm = env->core.reg[reg1];
                env->core.flag.zero = (imm == 0);  // 01 => 0; 10 => -; 00 => +
                env->core.flag.negative = (imm >> ((uint64_t) 63));
                env->core.flag.odd |= (imm & 1); // 1-- odd, 0-- even
            break;
            case I_CMPF: // CMPF
                immf = env->core.freg[reg1] - env->core.freg[reg2];
                env->core.flag.zero = (immf == 0.0);  // 01 => 0; 10 => -; 00 => +
                env->core.flag.negative = (immf < 0.0);
            break;
            case I_TESTF: // TESTF
                immf = env->core.freg[reg1];
                env->core.flag.zero = (immf == 0.0);  // 01 => 0; 10 => -; 00 => +
                env->core.flag.negative = (immf < 0.0);
            break;
            case I_JMP: // JUMP
                env->core.prcount=addr;
            break;
            case I_JZ: // JZ
                if (env->core.flag.zero) env->core.prcount = addr;
            break;
            case I_JNZ: // JNZ
                if (! env->core.flag.zero) env->core.prcount = addr;
            break;
            case I_JL: // JL
                if (env->core.flag.negative) env->core.prcount = addr;
            break;
            case I_JG: // JG
                if ((! env->core.flag.zero) && (! env->core.flag.negative)) env->core.prcount = addr;
            break;
            case I_JLE: // JLE
                if (env->core.flag.zero || env->core.flag.negative) env->core.prcount = addr;
            break;
            case I_JGE: // JGE
                if (! env->core.flag.negative) env->core.prcount = addr;
            break;
            case I_JODD:
                if (env->core.flag.odd) env->core.prcount = addr;
            break;
            case I_JEVEN:
                if (! env->core.flag.odd) env->core.prcount = addr;
            break;
            case I_MOV: // MOVE
                env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVI: // MOVI
                env->core.reg[reg1] = ((uint64_t) addr);
            break;
            case I_MOVZ: // MOVZ
                if (env->core.flag.zero) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVNZ: // MOVNZ
                if (! env->core.flag.zero) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVL: // MOVL
                if (env->core.flag.negative) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVG: // MOVG
                if ((! env->core.flag.zero) && (! env->core.flag.negative)) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVLE: // MOVLE
                if (env->core.flag.zero || env->core.flag.negative) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVGE: // MOVGE
                if (! env->core.flag.negative) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVODD:
                if (env->core.flag.odd) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_MOVEVEN:
                if (! env->core.flag.odd) env->core.reg[reg1] = env->core.reg[reg2];
            break;
            case I_LOAD: // LOAD
                env->core.reg[reg1] = env->vmem[addr].i64;
            break;
            case I_STORE: // STORE
                env->vmem[addr].i64 = env->core.reg[reg1];                
            break;
            case I_MOVF: // MOVF
                env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFI: // MOVFI
                env->core.freg[reg1] = addr;// WEWE
            break;
            case I_MOVFZ: // MOVFZ
                if (env->core.flag.zero) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFNZ: // MOVFNZ
                if (! env->core.flag.zero) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFL: // MOVFL
                if ((env->core.flag.negative)) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFG: // MOVFG
                if ((! env->core.flag.zero) && (! env->core.flag.negative)) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFLE: // MOVFLE
                if (env->core.flag.zero || env->core.flag.negative) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFGE: // MOVFGE
                if (! env->core.flag.negative) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFODD:
                if (env->core.flag.odd) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_MOVFEVEN:
                if (! env->core.flag.odd) env->core.freg[reg1] = env->core.freg[reg2];
            break;
            case I_LOADF: // LOAD
                env->core.freg[reg1] = env->vmem[addr].f64;
            break;
            case I_STOREF: // STORE
                env->vmem[addr].f64 = env->core.freg[reg1];                
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
                env->core.reg[reg1] = env->core.reg[reg2] / env->core.reg[reg3];
            break;
            case I_ADDF:  // ADDF
                env->core.freg[reg1] = env->core.freg[reg2] + env->core.freg[reg3];
            break;
            case I_SUBF:  // SUBF
                env->core.freg[reg1] = env->core.freg[reg2] - env->core.freg[reg3];
            break;
            case I_MULF:  // MUL
                env->core.freg[reg1] = env->core.freg[reg2] * env->core.freg[reg3];
            break;
            case I_DIVF: // DIVF
                env->core.freg[reg1] = env->core.freg[reg2] / env->core.freg[reg3];
            break;
            case I_MOD: // MOD
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
                env->core.reg[reg1] = env->core.reg[reg2] << env->core.reg[reg3];
            break;
            case I_SHR: // SHR
                env->core.reg[reg1] = env->core.reg[reg2] >> env->core.reg[reg3];
            break;

            case I_CAST: // CAST
                env->core.reg[reg1] = (uint64_t) env->core.freg[reg2];
            break;
            case I_CASTF: // CASTF
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
                env->core.reg[reg1] = rand();
            break;
            default:
                unreachable();
            break;
        }
    }
    return clock_limit;
}
