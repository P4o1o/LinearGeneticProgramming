from .base import Structure, Union, POINTER, c_uint8, c_uint32, c_uint64, c_double, c_char_p, c_int8, IntEnum, ctypes, liblgp
from enum import Enum

class Memblock(Union):
    _fields_ = [
        ("i64", c_uint64),
        ("f64", c_double)
    ]

class Instruction(Structure):
    _fields_ = [
        ("op", c_uint8),
        ("reg", c_uint8 * 3),
        ("addr", c_uint32)
    ]

class FlagReg(Structure):
    _fields_ = [
        ("odd", c_uint8, 1),
        ("negative", c_uint8, 1),
        ("zero", c_uint8, 1),
        ("exist", c_uint8, 1)
    ]

class Core(Structure):
    _fields_ = [
        ("reg", c_uint64 * 4),
        ("freg", c_double * 4),
        ("flag", FlagReg),
        ("prcount", c_uint64)
    ]

class VirtualMachine(Structure):
    _fields_ = [
        ("core", Core),
        ("ram", POINTER(Memblock)),
        ("rom", POINTER(Memblock)),
        ("program", POINTER(Instruction))
    ]

class Program(Structure):
    _fields_ = [
        ("content", POINTER(Instruction)),
        ("size", c_uint64)
    ]


class OperationStruct(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("regs", c_uint8),
        ("addr", c_int8),
        ("state_changer", c_uint8),
        ("code", c_uint8)
    ]


class Operation(Enum):
    EXIT = OperationStruct.in_dll(liblgp, "OP_EXIT")
    LOAD_RAM = OperationStruct.in_dll(liblgp, "OP_LOAD_RAM")
    STORE_RAM = OperationStruct.in_dll(liblgp, "OP_STORE_RAM")
    LOAD_ROM = OperationStruct.in_dll(liblgp, "OP_LOAD_ROM")
    MOV = OperationStruct.in_dll(liblgp, "OP_MOV")
    CMOV_Z = OperationStruct.in_dll(liblgp, "OP_CMOV_Z")
    CMOV_NZ = OperationStruct.in_dll(liblgp, "OP_CMOV_NZ")
    CMOV_L = OperationStruct.in_dll(liblgp, "OP_CMOV_L")
    CMOV_G = OperationStruct.in_dll(liblgp, "OP_CMOV_G")
    CMOV_LE = OperationStruct.in_dll(liblgp, "OP_CMOV_LE")
    CMOV_GE = OperationStruct.in_dll(liblgp, "OP_CMOV_GE")
    CMOV_EXIST = OperationStruct.in_dll(liblgp, "OP_CMOV_EXIST")
    CMOV_NEXIST = OperationStruct.in_dll(liblgp, "OP_CMOV_NEXIST")
    CMOV_ODD = OperationStruct.in_dll(liblgp, "OP_CMOV_ODD")
    CMOV_EVEN = OperationStruct.in_dll(liblgp, "OP_CMOV_EVEN")
    CMOV_OVERFLOW = OperationStruct.in_dll(liblgp, "OP_CMOV_OVERFLOW")
    CMOV_ZERODIV = OperationStruct.in_dll(liblgp, "OP_CMOV_ZERODIV")
    MOV_I = OperationStruct.in_dll(liblgp, "OP_MOV_I")
    JMP = OperationStruct.in_dll(liblgp, "OP_JMP")
    JMP_Z = OperationStruct.in_dll(liblgp, "OP_JMP_Z")
    JMP_NZ = OperationStruct.in_dll(liblgp, "OP_JMP_NZ")
    JMP_L = OperationStruct.in_dll(liblgp, "OP_JMP_L")
    JMP_G = OperationStruct.in_dll(liblgp, "OP_JMP_G")
    JMP_LE = OperationStruct.in_dll(liblgp, "OP_JMP_LE")
    JMP_GE = OperationStruct.in_dll(liblgp, "OP_JMP_GE")
    JMP_EXIST = OperationStruct.in_dll(liblgp, "OP_JMP_EXIST")
    JMP_NEXIST = OperationStruct.in_dll(liblgp, "OP_JMP_NEXIST")
    JMP_EVEN = OperationStruct.in_dll(liblgp, "OP_JMP_EVEN")
    JMP_OVERFLOW = OperationStruct.in_dll(liblgp, "OP_JMP_OVERFLOW")
    JMP_ZERODIV = OperationStruct.in_dll(liblgp, "OP_JMP_ZERODIV")
    JMP_ODD = OperationStruct.in_dll(liblgp, "OP_JMP_ODD")
    CLC = OperationStruct.in_dll(liblgp, "OP_CLC")
    CMP = OperationStruct.in_dll(liblgp, "OP_CMP")
    TEST = OperationStruct.in_dll(liblgp, "OP_TEST")
    ADD = OperationStruct.in_dll(liblgp, "OP_ADD")
    SUB = OperationStruct.in_dll(liblgp, "OP_SUB")
    MUL = OperationStruct.in_dll(liblgp, "OP_MUL")
    DIV = OperationStruct.in_dll(liblgp, "OP_DIV")
    MOD = OperationStruct.in_dll(liblgp, "OP_MOD")
    INC = OperationStruct.in_dll(liblgp, "OP_INC")
    DEC = OperationStruct.in_dll(liblgp, "OP_DEC")
    AND = OperationStruct.in_dll(liblgp, "OP_AND")
    OR = OperationStruct.in_dll(liblgp, "OP_OR")
    XOR = OperationStruct.in_dll(liblgp, "OP_XOR")
    NOT = OperationStruct.in_dll(liblgp, "OP_NOT")
    SHL = OperationStruct.in_dll(liblgp, "OP_SHL")
    SHR = OperationStruct.in_dll(liblgp, "OP_SHR")
    CAST = OperationStruct.in_dll(liblgp, "OP_CAST")
    NOP = OperationStruct.in_dll(liblgp, "OP_NOP")
    LOAD_RAM_F = OperationStruct.in_dll(liblgp, "OP_LOAD_RAM_F")
    LOAD_ROM_F = OperationStruct.in_dll(liblgp, "OP_LOAD_ROM_F")
    STORE_RAM_F = OperationStruct.in_dll(liblgp, "OP_STORE_RAM_F")
    MOV_F = OperationStruct.in_dll(liblgp, "OP_MOV_F")
    CMOV_Z_F = OperationStruct.in_dll(liblgp, "OP_CMOV_Z_F")
    CMOV_NZ_F = OperationStruct.in_dll(liblgp, "OP_CMOV_NZ_F")
    CMOV_L_F = OperationStruct.in_dll(liblgp, "OP_CMOV_L_F")
    CMOV_G_F = OperationStruct.in_dll(liblgp, "OP_CMOV_G_F")
    CMOV_LE_F = OperationStruct.in_dll(liblgp, "OP_CMOV_LE_F")
    CMOV_GE_F = OperationStruct.in_dll(liblgp, "OP_CMOV_GE_F")
    MOV_I_F = OperationStruct.in_dll(liblgp, "OP_MOV_I_F")
    CMOV_EXIST_F = OperationStruct.in_dll(liblgp, "OP_CMOV_EXIST_F")
    CMOV_NEXIST_F = OperationStruct.in_dll(liblgp, "OP_CMOV_NEXIST_F")
    CMOV_ODD_F = OperationStruct.in_dll(liblgp, "OP_CMOV_ODD_F")
    CMOV_EVEN_F = OperationStruct.in_dll(liblgp, "OP_CMOV_EVEN_F")
    CMOV_OVERFLOW_F = OperationStruct.in_dll(liblgp, "OP_CMOV_OVERFLOW_F")
    CMOV_ZERODIV_F = OperationStruct.in_dll(liblgp, "OP_CMOV_ZERODIV_F")
    CMP_F = OperationStruct.in_dll(liblgp, "OP_CMP_F")
    TEST_F = OperationStruct.in_dll(liblgp, "OP_TEST_F")
    ADD_F = OperationStruct.in_dll(liblgp, "OP_ADD_F")
    SUB_F = OperationStruct.in_dll(liblgp, "OP_SUB_F")
    MUL_F = OperationStruct.in_dll(liblgp, "OP_MUL_F")
    DIV_F = OperationStruct.in_dll(liblgp, "OP_DIV_F")
    SQRT = OperationStruct.in_dll(liblgp, "OP_SQRT")
    POW = OperationStruct.in_dll(liblgp, "OP_POW")
    EXP = OperationStruct.in_dll(liblgp, "OP_EXP")
    LN = OperationStruct.in_dll(liblgp, "OP_LN")
    LOG = OperationStruct.in_dll(liblgp, "OP_LOG")
    LOG10 = OperationStruct.in_dll(liblgp, "OP_LOG10")
    COS = OperationStruct.in_dll(liblgp, "OP_COS")
    SIN = OperationStruct.in_dll(liblgp, "OP_SIN")
    TAN = OperationStruct.in_dll(liblgp, "OP_TAN")
    ACOS = OperationStruct.in_dll(liblgp, "OP_ACOS")
    ASIN = OperationStruct.in_dll(liblgp, "OP_ASIN")
    ATAN = OperationStruct.in_dll(liblgp, "OP_ATAN")
    COSH = OperationStruct.in_dll(liblgp, "OP_COSH")
    SINH = OperationStruct.in_dll(liblgp, "OP_SINH")
    TANH = OperationStruct.in_dll(liblgp, "OP_TANH")
    ACOSH = OperationStruct.in_dll(liblgp, "OP_ACOSH")
    ASINH = OperationStruct.in_dll(liblgp, "OP_ASINH")
    ATANH = OperationStruct.in_dll(liblgp, "OP_ATANH")
    CAST_F = OperationStruct.in_dll(liblgp, "OP_CAST_F")
    RAND = OperationStruct.in_dll(liblgp, "OP_RAND")
    ROUND = OperationStruct.in_dll(liblgp, "OP_ROUND")
    MUL_S = OperationStruct.in_dll(liblgp, "OP_MUL_S")
    DIV_S = OperationStruct.in_dll(liblgp, "OP_DIV_S")
    ABS = OperationStruct.in_dll(liblgp, "OP_ABS")
    ABS_F = OperationStruct.in_dll(liblgp, "OP_ABS_F")
    
    @property
    def name(self) -> str:
        return self.value.name.decode('utf-8') if self.value.name else f"OP_{self.value.code}"
        
    @property
    def code(self) -> int:
        return self.value.code
    
    @property
    def c_wrapper(self) -> OperationStruct:
        return self.value

__all__ = ['Memblock', 'Instruction', 'FlagReg', 'Core', 
           'VirtualMachine', 'Program', 'OperationStruct', 'Operation']