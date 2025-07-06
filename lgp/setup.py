"""
Setup module for library function signatures
"""

from .base import POINTER, c_uint32, c_uint64, c_void_p, liblgp
from .vm import Program
from .genetics import LGPInput, InstructionSet, LGPResult
from .evolution import LGPOptions
from .fitness import FitnessAssessment, FitnessParams
from .creation import InitializationParams

def setup_library():
    """Setup dei tipi per le funzioni della libreria"""
    
    # Setup evolve function
    liblgp.evolve.argtypes = [POINTER(LGPInput), POINTER(LGPOptions)]
    liblgp.evolve.restype = LGPResult

    # Setup print_program function
    liblgp.print_program.argtypes = [POINTER(Program)]
    liblgp.print_program.restype = None

    # Setup per vector_distance
    liblgp.vector_distance.argtypes = [POINTER(InstructionSet), c_uint64, c_uint64]
    liblgp.vector_distance.restype = LGPInput

    # Setup per le funzioni di initialization
    liblgp.unique_population.argtypes = [POINTER(LGPInput), POINTER(InitializationParams), 
                                        POINTER(FitnessAssessment), c_uint64, POINTER(FitnessParams)]
    liblgp.unique_population.restype = LGPResult

    liblgp.rand_population.argtypes = [POINTER(LGPInput), POINTER(InitializationParams), 
                                      POINTER(FitnessAssessment), c_uint64, POINTER(FitnessParams)]
    liblgp.rand_population.restype = LGPResult

    # Setup per init_MT19937 se disponibile
    try:
        liblgp.init_MT19937.argtypes = [c_uint32, c_void_p]
        liblgp.init_MT19937.restype = None
    except AttributeError:
        pass

# Call setup when module is imported
setup_library()

__all__ = ['setup_library']
