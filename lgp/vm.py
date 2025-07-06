"""
VM structures and Operation enumeration - corresponds to vm.h
"""

from .base import Structure, Union, POINTER, c_uint8, c_uint32, c_uint64, c_double, c_char_p, c_int8, IntEnum, ctypes, liblgp
from enum import Enum

class Memblock(Union):
    """Corrisponde a union Memblock in vm.h"""
    _fields_ = [
        ("i64", c_uint64),
        ("f64", c_double)
    ]


class Instruction(Structure):
    """Corrisponde a struct Instruction in vm.h"""
    _fields_ = [
        ("op", c_uint8),
        ("reg", c_uint8 * 3),
        ("addr", c_uint32)
    ]


class FlagReg(Structure):
    """Corrisponde a struct FlagReg in vm.h"""
    _fields_ = [
        ("odd", c_uint8, 1),
        ("negative", c_uint8, 1),
        ("zero", c_uint8, 1),
        ("exist", c_uint8, 1)
    ]


class Core(Structure):
    """Corrisponde a struct Core in vm.h"""
    _fields_ = [
        ("reg", c_uint64 * 4),
        ("freg", c_double * 4),
        ("flag", FlagReg),
        ("prcount", c_uint64)
    ]


class VirtualMachine(Structure):
    """Corrisponde a struct VirtualMachine in vm.h"""
    _fields_ = [
        ("core", Core),
        ("ram", POINTER(Memblock)),
        ("rom", POINTER(Memblock)),
        ("program", POINTER(Instruction))
    ]


class Program(Structure):
    """Corrisponde a struct Program in genetics.h"""
    _fields_ = [
        ("content", POINTER(Instruction)),
        ("size", c_uint64)
    ]


class OperationStruct(Structure):
    """Corrisponde a struct Operation in vm.h"""
    _fields_ = [
        ("name", c_char_p),
        ("regs", c_uint8),
        ("addr", c_int8),
        ("state_changer", c_uint8),
        ("code", c_uint8)
    ]


class Operation(Enum):
    """Enumerazione delle 87 operazioni della VM con OperationWrapper integrato"""
    
    # Define all 87 operations with their OperationWrapper instances
    EXIT = OperationStruct(name=b"EXIT", regs=0, addr=0, state_changer=0, code=0)
    LOAD_RAM = OperationStruct(name=b"LOAD_RAM", regs=2, addr=1, state_changer=0, code=1)
    STORE_RAM = OperationStruct(name=b"STORE_RAM", regs=2, addr=1, state_changer=1, code=2)
    LOAD_ROM = OperationStruct(name=b"LOAD_ROM", regs=2, addr=1, state_changer=0, code=3)
    MOV = OperationStruct(name=b"MOV", regs=2, addr=0, state_changer=0, code=4)
    CMOV_Z = OperationStruct(name=b"CMOV_Z", regs=2, addr=0, state_changer=0, code=5)
    CMOV_NZ = OperationStruct(name=b"CMOV_NZ", regs=2, addr=0, state_changer=0, code=6)
    CMOV_L = OperationStruct(name=b"CMOV_L", regs=2, addr=0, state_changer=0, code=7)
    CMOV_G = OperationStruct(name=b"CMOV_G", regs=2, addr=0, state_changer=0, code=8)
    CMOV_LE = OperationStruct(name=b"CMOV_LE", regs=2, addr=0, state_changer=0, code=9)
    CMOV_GE = OperationStruct(name=b"CMOV_GE", regs=2, addr=0, state_changer=0, code=10)
    CMOV_EXIST = OperationStruct(name=b"CMOV_EXIST", regs=2, addr=0, state_changer=0, code=11)
    CMOV_NEXIST = OperationStruct(name=b"CMOV_NEXIST", regs=2, addr=0, state_changer=0, code=12)
    CMOV_ODD = OperationStruct(name=b"CMOV_ODD", regs=2, addr=0, state_changer=0, code=13)
    CMOV_EVEN = OperationStruct(name=b"CMOV_EVEN", regs=2, addr=0, state_changer=0, code=14)
    MOV_I = OperationStruct(name=b"MOV_I", regs=1, addr=1, state_changer=0, code=15)
    JMP = OperationStruct(name=b"JMP", regs=0, addr=1, state_changer=1, code=16)
    JMP_Z = OperationStruct(name=b"JMP_Z", regs=0, addr=1, state_changer=1, code=17)
    JMP_NZ = OperationStruct(name=b"JMP_NZ", regs=0, addr=1, state_changer=1, code=18)
    JMP_L = OperationStruct(name=b"JMP_L", regs=0, addr=1, state_changer=1, code=19)
    JMP_G = OperationStruct(name=b"JMP_G", regs=0, addr=1, state_changer=1, code=20)
    JMP_LE = OperationStruct(name=b"JMP_LE", regs=0, addr=1, state_changer=1, code=21)
    JMP_GE = OperationStruct(name=b"JMP_GE", regs=0, addr=1, state_changer=1, code=22)
    JMP_EXIST = OperationStruct(name=b"JMP_EXIST", regs=0, addr=1, state_changer=1, code=23)
    JMP_NEXIST = OperationStruct(name=b"JMP_NEXIST", regs=0, addr=1, state_changer=1, code=24)
    JMP_EVEN = OperationStruct(name=b"JMP_EVEN", regs=0, addr=1, state_changer=1, code=25)
    JMP_ODD = OperationStruct(name=b"JMP_ODD", regs=0, addr=1, state_changer=1, code=26)
    CLC = OperationStruct(name=b"CLC", regs=0, addr=0, state_changer=1, code=27)
    CMP = OperationStruct(name=b"CMP", regs=2, addr=0, state_changer=1, code=28)
    TEST = OperationStruct(name=b"TEST", regs=1, addr=0, state_changer=1, code=29)
    ADD = OperationStruct(name=b"ADD", regs=3, addr=0, state_changer=0, code=30)
    SUB = OperationStruct(name=b"SUB", regs=3, addr=0, state_changer=0, code=31)
    MUL = OperationStruct(name=b"MUL", regs=3, addr=0, state_changer=0, code=32)
    DIV = OperationStruct(name=b"DIV", regs=3, addr=0, state_changer=0, code=33)
    MOD = OperationStruct(name=b"MOD", regs=3, addr=0, state_changer=0, code=34)
    INC = OperationStruct(name=b"INC", regs=1, addr=0, state_changer=0, code=35)
    DEC = OperationStruct(name=b"DEC", regs=1, addr=0, state_changer=0, code=36)
    AND = OperationStruct(name=b"AND", regs=3, addr=0, state_changer=0, code=37)
    OR = OperationStruct(name=b"OR", regs=3, addr=0, state_changer=0, code=38)
    XOR = OperationStruct(name=b"XOR", regs=3, addr=0, state_changer=0, code=39)
    NOT = OperationStruct(name=b"NOT", regs=2, addr=0, state_changer=0, code=40)
    SHL = OperationStruct(name=b"SHL", regs=3, addr=0, state_changer=0, code=41)
    SHR = OperationStruct(name=b"SHR", regs=3, addr=0, state_changer=0, code=42)
    CAST = OperationStruct(name=b"CAST", regs=2, addr=0, state_changer=0, code=43)
    NOP = OperationStruct(name=b"NOP", regs=0, addr=0, state_changer=0, code=44)
    LOAD_RAM_F = OperationStruct(name=b"LOAD_RAM_F", regs=2, addr=1, state_changer=0, code=45)
    LOAD_ROM_F = OperationStruct(name=b"LOAD_ROM_F", regs=1, addr=4, state_changer=0, code=46)
    STORE_RAM_F = OperationStruct(name=b"STORE_RAM_F", regs=1, addr=2, state_changer=2, code=47)
    MOV_F = OperationStruct(name=b"MOV_F", regs=2, addr=0, state_changer=0, code=48)
    CMOV_Z_F = OperationStruct(name=b"CMOV_Z_F", regs=2, addr=0, state_changer=0, code=49)
    CMOV_NZ_F = OperationStruct(name=b"CMOV_NZ_F", regs=2, addr=0, state_changer=0, code=50)
    CMOV_L_F = OperationStruct(name=b"CMOV_L_F", regs=2, addr=0, state_changer=0, code=51)
    CMOV_G_F = OperationStruct(name=b"CMOV_G_F", regs=2, addr=0, state_changer=0, code=52)
    CMOV_LE_F = OperationStruct(name=b"CMOV_LE_F", regs=2, addr=0, state_changer=0, code=53)
    CMOV_GE_F = OperationStruct(name=b"CMOV_GE_F", regs=2, addr=0, state_changer=0, code=54)
    MOV_I_F = OperationStruct(name=b"MOV_I_F", regs=1, addr=1, state_changer=0, code=55)
    CMOV_EXIST_F = OperationStruct(name=b"CMOV_EXIST_F", regs=2, addr=0, state_changer=0, code=56)
    CMOV_NEXIST_F = OperationStruct(name=b"CMOV_NEXIST_F", regs=2, addr=0, state_changer=0, code=57)
    CMOV_ODD_F = OperationStruct(name=b"CMOV_ODD_F", regs=2, addr=0, state_changer=0, code=58)
    CMOV_EVEN_F = OperationStruct(name=b"CMOV_EVEN_F", regs=2, addr=0, state_changer=0, code=59)
    CMP_F = OperationStruct(name=b"CMP_F", regs=2, addr=0, state_changer=1, code=60)
    TEST_F = OperationStruct(name=b"TEST_F", regs=1, addr=0, state_changer=1, code=61)
    ADD_F = OperationStruct(name=b"ADD_F", regs=3, addr=0, state_changer=0, code=62)
    SUB_F = OperationStruct(name=b"SUB_F", regs=3, addr=0, state_changer=0, code=63)
    MUL_F = OperationStruct(name=b"MUL_F", regs=3, addr=0, state_changer=0, code=64)
    DIV_F = OperationStruct(name=b"DIV_F", regs=3, addr=0, state_changer=0, code=65)
    SQRT = OperationStruct(name=b"SQRT", regs=2, addr=0, state_changer=0, code=66)
    POW = OperationStruct(name=b"POW", regs=3, addr=0, state_changer=0, code=67)
    EXP = OperationStruct(name=b"EXP", regs=2, addr=0, state_changer=0, code=68)
    LN = OperationStruct(name=b"LN", regs=2, addr=0, state_changer=0, code=69)
    LOG = OperationStruct(name=b"LOG", regs=3, addr=0, state_changer=0, code=70)
    LOG10 = OperationStruct(name=b"LOG10", regs=2, addr=0, state_changer=0, code=71)
    COS = OperationStruct(name=b"COS", regs=2, addr=0, state_changer=0, code=72)
    SIN = OperationStruct(name=b"SIN", regs=2, addr=0, state_changer=0, code=73)
    TAN = OperationStruct(name=b"TAN", regs=2, addr=0, state_changer=0, code=74)
    ACOS = OperationStruct(name=b"ACOS", regs=2, addr=0, state_changer=0, code=75)
    ASIN = OperationStruct(name=b"ASIN", regs=2, addr=0, state_changer=0, code=76)
    ATAN = OperationStruct(name=b"ATAN", regs=2, addr=0, state_changer=0, code=77)
    COSH = OperationStruct(name=b"COSH", regs=2, addr=0, state_changer=0, code=78)
    SINH = OperationStruct(name=b"SINH", regs=2, addr=0, state_changer=0, code=79)
    TANH = OperationStruct(name=b"TANH", regs=2, addr=0, state_changer=0, code=80)
    ACOSH = OperationStruct(name=b"ACOSH", regs=2, addr=0, state_changer=0, code=81)
    ASINH = OperationStruct(name=b"ASINH", regs=2, addr=0, state_changer=0, code=82)
    ATANH = OperationStruct(name=b"ATANH", regs=2, addr=0, state_changer=0, code=83)
    CAST_F = OperationStruct(name=b"CAST_F", regs=2, addr=0, state_changer=0, code=84)
    RAND = OperationStruct(name=b"RAND", regs=1, addr=0, state_changer=0, code=85)
    ROUND = OperationStruct(name=b"ROUND", regs=2, addr=0, state_changer=0, code=86)
    
    def name(self) -> str:
        """Ritorna il nome dell'operazione"""
        return self.value.name.decode('utf-8') if self.value.name else f"OP_{self.value.code}"
        
    def code(self) -> int:
        """Ritorna il codice dell'operazione"""
        return self.value.code
    
    @property
    def c_wrapper(self) -> OperationStruct:
        """Ritorna l'OperationWrapper integrato"""
        return self.value

__all__ = ['MemblockWrapper', 'InstructionWrapper', 'FlagRegWrapper', 'CoreWrapper', 
           'VirtualMachineWrapper', 'ProgramWrapper', 'OperationWrapper', 'Operation']