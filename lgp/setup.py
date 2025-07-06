"""
Setup module for library function signatures
"""

from .base import POINTER, c_uint32, c_uint64, c_void_p, liblgp
from .vm import ProgramWrapper
from .genetics import LGPInputWrapper, InstructionSetWrapper
from .evolution import LGPOptionsWrapper, LGPResultWrapper
from .fitness import FitnessAssessmentWrapper, FitnessParamsWrapper
from .creation import InitializationParamsWrapper

def setup_library():
    """Setup dei tipi per le funzioni della libreria"""
    
    # Setup evolve function
    liblgp.evolve.argtypes = [POINTER(LGPInputWrapper), POINTER(LGPOptionsWrapper)]
    liblgp.evolve.restype = LGPResultWrapper

    # Setup print_program function
    liblgp.print_program.argtypes = [POINTER(ProgramWrapper)]
    liblgp.print_program.restype = None

    # Setup per vector_distance
    liblgp.vector_distance.argtypes = [POINTER(InstructionSetWrapper), c_uint64, c_uint64]
    liblgp.vector_distance.restype = LGPInputWrapper

    # Setup per le funzioni di initialization
    liblgp.unique_population.argtypes = [POINTER(LGPInputWrapper), POINTER(InitializationParamsWrapper), 
                                        POINTER(FitnessAssessmentWrapper), c_uint64, POINTER(FitnessParamsWrapper)]
    liblgp.unique_population.restype = LGPResultWrapper

    liblgp.rand_population.argtypes = [POINTER(LGPInputWrapper), POINTER(InitializationParamsWrapper), 
                                      POINTER(FitnessAssessmentWrapper), c_uint64, POINTER(FitnessParamsWrapper)]
    liblgp.rand_population.restype = LGPResultWrapper

    # Setup per init_MT19937 se disponibile
    try:
        liblgp.init_MT19937.argtypes = [c_uint32, c_void_p]
        liblgp.init_MT19937.restype = None
    except AttributeError:
        pass

# Call setup when module is imported
setup_library()

__all__ = ['setup_library']
