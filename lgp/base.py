import ctypes
from ctypes import Structure, Union, POINTER, c_uint8, c_uint32, c_uint64, c_double, c_char_p, c_void_p, c_uint, c_int8
from typing import Tuple, Optional, List
from enum import IntEnum

try:
    liblgp = ctypes.CDLL('./liblgp.so')
    VECT_ALIGNMENT = ctypes.c_uint64.in_dll(liblgp, "VECT_ALIGNMENT_WRAPPER").value
    INSTR_NUM = ctypes.c_uint64.in_dll(liblgp, "INSTR_NUM_WRAPPER").value
    NUMBER_OF_OMP_THREADS = number_of_threads = c_uint64.in_dll(liblgp, "NUMBER_OF_THREADS").value
except OSError as e:
    print(f"Error: Failed to load liblgp.so - {e}")
    print("Please ensure the library is compiled with: make python")
    import sys
    sys.exit(1)

__all__ = ['ctypes', 'Structure', 'Union', 'POINTER', 'c_uint8', 'c_uint32', 'c_uint64', 
           'c_double', 'c_char_p', 'c_void_p', 'c_uint', 'c_int8', 'Tuple', 'Optional', 
           'List', 'IntEnum', 'liblgp', 'VECT_ALIGNMENT', 'INSTR_NUM', 'NUMBER_OF_OMP_THREADS']
