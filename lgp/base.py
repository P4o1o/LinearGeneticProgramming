"""
Base module for LGP library loading and common imports
"""

import ctypes
from ctypes import Structure, Union, POINTER, c_uint8, c_uint32, c_uint64, c_double, c_char_p, c_void_p, c_uint, c_int8
from typing import Tuple, Optional, List
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from enum import IntEnum

# Carica la libreria dinamica
try:
    liblgp = ctypes.CDLL('./liblgp.so')
    
    # Accesso corretto alle variabili globali C
    VECT_ALIGNMENT = ctypes.c_uint64.in_dll(liblgp, "VECT_ALIGNMENT_WRAPPER").value
    INSTR_NUM = ctypes.c_uint64.in_dll(liblgp, "INSTR_NUM_WRAPPER").value
    
except OSError as e:
    print(f"Error: Failed to load liblgp.so - {e}")
    print("Please ensure the library is compiled with: make python")
    import sys
    sys.exit(1)
except ValueError as e:
    print(f"Error: Failed to access library constants - {e}")
    print("Using fallback values...")
    VECT_ALIGNMENT = 32  # Fallback value
    INSTR_NUM = 87      # Fallback value

__all__ = ['ctypes', 'Structure', 'Union', 'POINTER', 'c_uint8', 'c_uint32', 'c_uint64', 
           'c_double', 'c_char_p', 'c_void_p', 'c_uint', 'c_int8', 'Tuple', 'Optional', 
           'List', 'np', 'pd', 'IntEnum', 'liblgp']
