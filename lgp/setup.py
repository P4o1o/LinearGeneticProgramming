from .base import POINTER, c_uint32, c_uint64, c_double, liblgp
from .vm import Program
from .genetics import LGPInput, InstructionSet, LGPResult, Population
from .evolution import LGPOptions
from .fitness.interface import FitnessFunction, FitnessParams
from .creation import InitializationParams

def setup_library():
    
    # ========== EVOLUTION FUNCTIONS ==========
    liblgp.evolve.argtypes = [POINTER(LGPInput), POINTER(LGPOptions)]
    liblgp.evolve.restype = LGPResult

    # ========== UTILITY FUNCTIONS ==========
    liblgp.print_program.argtypes = [POINTER(Program)]
    liblgp.print_program.restype = None
    
    liblgp.random_init_wrapper.argtypes = [c_uint32, c_uint32]
    liblgp.random_init_wrapper.restype = None
    
    liblgp.random_init_all.argtypes = [c_uint32]
    liblgp.random_init_all.restype = None

    # ========== PROBLEM GENERATION ==========
    liblgp.vector_distance.argtypes = [POINTER(InstructionSet), c_uint64, c_uint64]
    liblgp.vector_distance.restype = LGPInput

    liblgp.bouncing_balls.argtypes = [POINTER(InstructionSet), c_uint64]
    liblgp.bouncing_balls.restype = LGPInput

    liblgp.dice_game.argtypes = [POINTER(InstructionSet), c_uint64]
    liblgp.dice_game.restype = LGPInput

    liblgp.shopping_list.argtypes = [POINTER(InstructionSet), c_uint64, c_uint64]
    liblgp.shopping_list.restype = LGPInput

    liblgp.snow_day.argtypes = [POINTER(InstructionSet), c_uint64]
    liblgp.snow_day.restype = LGPInput

    # ========== INITIALIZATION FUNCTIONS ==========
    liblgp.unique_population.argtypes = [POINTER(LGPInput), POINTER(InitializationParams), 
                                        POINTER(FitnessFunction), c_uint64, POINTER(FitnessParams)]
    liblgp.unique_population.restype = LGPResult

    liblgp.rand_population.argtypes = [POINTER(LGPInput), POINTER(InitializationParams), 
                                      POINTER(FitnessFunction), c_uint64, POINTER(FitnessParams)]
    liblgp.rand_population.restype = LGPResult

    # ========== MEMORY MANAGEMENT ==========
    liblgp.free_population.argtypes = [POINTER(Population)]
    liblgp.free_population.restype = None

    liblgp.free_lgp_input.argtypes = [POINTER(LGPInput)]
    liblgp.free_lgp_input.restype = None

    # ========== FITNESS FUNCTIONS ==========
    
    liblgp.mse.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.mse.restype = c_double
    
    liblgp.rmse.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.rmse.restype = c_double
    
    liblgp.mae.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.mae.restype = c_double
    
    liblgp.accuracy.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.accuracy.restype = c_double
    
    liblgp.f1_score.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.f1_score.restype = c_double
    
    liblgp.length_penalized_mse.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.length_penalized_mse.restype = c_double
    
    liblgp.clock_penalized_mse.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.clock_penalized_mse.restype = c_double
    
    liblgp.r_squared.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.r_squared.restype = c_double
    
    liblgp.balanced_accuracy.argtypes = [POINTER(LGPInput), POINTER(Program), c_uint64, POINTER(FitnessParams)]
    liblgp.balanced_accuracy.restype = c_double

__all__ = ['setup_library']