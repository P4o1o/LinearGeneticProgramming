"""
Evolution structures and functions - corresponds to evolution.h
"""

from .base import Structure, POINTER, c_uint64, c_uint, c_double, c_void_p, Tuple, Optional, ctypes, liblgp
from .genetics import LGPInput, LGPResult, Population
from .fitness import MSE, Fitness, FitnessFunction, FitnessParams
from .selection import SelectionFunction, SelectionParams, Selection, Tournament
from .creation import InitializationParams, Initialization, UniquePopulation

# Definizione del tipo initialization_fn come nel C
initialization_fn = c_void_p

class LGPOptions(Structure):
    """Corresponds to struct LGPOptions in evolution.h"""
    _fields_ = [
        ("fitness", FitnessFunction),
        ("fitness_param", FitnessParams),
        ("selection", SelectionFunction),
        ("select_param", SelectionParams),
        ("initialization_func", initialization_fn),
        ("init_params", InitializationParams),
        ("initial_pop", Population),  # Struct not pointer come nel C
        ("target", c_double),
        ("mutation_prob", c_double),
        ("crossover_prob", c_double),
        ("max_clock", c_uint64),
        ("max_individ_len", c_uint64),
        ("max_mutation_len", c_uint64),
        ("generations", c_uint64),
        ("verbose", c_uint)
    ]


def evolve(lgp_input: LGPInput, 
          fitness: Fitness = MSE(),
          selection: Selection = Tournament(3),
          initialization: Initialization = None,
          initial_pop: Optional[Population] = None,
          target: float = 1e-27,
          mutation_prob: float = 0.76,
          crossover_prob: float = 0.95,
          max_clock: int = 5000,
          max_individ_len: int = 50,
          max_mutation_len: int = 5,
          generations: int = 40,
          verbose: int = 1) -> Tuple[Population, int, int, int]:
    """
    Execute LGP evolution
    
    Args:
        lgp_input: Input for LGP
        fitness: Fitness function (default: MSE)
        selection: Selection method (default: Tournament)
        initialization: Initialization method (default: UniquePopulation)
        init_params: Initialization parameters (pop_size, minsize, maxsize)
        target: Fitness target to stop evolution
        mutation_prob: Mutation probability
        crossover_prob: Crossover probability
        max_clock: Maximum number of clock cycles per program
        max_individ_len: Maximum individual length
        max_mutation_len: Maximum mutation length
        generations: Maximum number of generations
        verbose: Verbosity level
        
    Returns:
        Tuple containing (final population, evaluations, generations, best individual index)
    """

    if initialization is not None:
        init_func = initialization.function
        init_params = initialization.parameters
        initial_pop = Population()
    else:
        if initial_pop is not None:
            init_func = c_void_p(0)
            init_params = InitializationParams()
        else:
            raise ValueError("Either initialization or initial_pop must be provided")
    
    options = LGPOptions(
        fitness=fitness.function,
        fitness_param=fitness.parameters,
        selection=selection.function,
        select_param=selection.parameters,
        initialization_func=init_func,
        init_params=init_params,
        initial_pop=initial_pop,
        target=target,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        max_clock=max_clock,
        max_individ_len=max_individ_len,
        max_mutation_len=max_mutation_len,
        generations=generations,
        verbose=verbose
    )
    result = liblgp.evolve(ctypes.byref(lgp_input), ctypes.byref(options))
    
    return (
        result.pop,
        result.evaluations,
        result.generations,
        result.best_individ
    )

__all__ = ['LGPOptions', 'evolve']
