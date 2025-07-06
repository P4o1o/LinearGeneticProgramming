"""
Evolution structures and functions - corresponds to evolution.h
"""

from .base import Structure, POINTER, c_uint64, c_uint, c_double, c_void_p, Tuple, Optional, ctypes, liblgp
from .genetics import LGPInput, LGPResult, Population
from .fitness import FitnessParams, FitnessAssessment
from .selection import SelectionParams, Selection
from .creation import InitializationParams, Initialization

# Definizione del tipo initialization_fn come nel C
initialization_fn = c_void_p

class LGPOptions(Structure):
    """Corresponds to struct LGPOptions in evolution.h"""
    _fields_ = [
        ("fitness", FitnessAssessment),
        ("fitness_param", FitnessParams),
        ("selection", Selection),
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
          fitness: FitnessAssessment = None,
          selection: Selection = None,
          initialization: Initialization = None,
          init_params: Optional[Tuple[int, int, int]] = None,
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
    from .fitness import MSE
    from .selection import Tournament
    from .creation import UniquePopulation as UP
    from .genetics import Population
    
    # Set default values
    if fitness is None:
        fitness = MSE()
    if selection is None:
        selection = Tournament(3)
    if initialization is None:
        initialization = UP()
    if init_params is None:
        init_params = (1000, 2, 5)
    
    # Prepare fitness parameters usando get_params() solo se la fitness li richiede
    try:
        fitness_param_union = fitness.get_params()
    except:
        # Se get_params() fallisce o non è implementato, usa parametri vuoti
        fitness_param_union = FitnessParams()
    
    # Prepare selection parameters using get_params()
    select_param_union = selection.get_params()
    
    init_params_struct = InitializationParams(
        pop_size=init_params[0],
        minsize=init_params[1],
        maxsize=init_params[2]
    )
    
    # Create LGPOptions structure seguendo l'esempio del main.c
    # Se abbiamo initial_pop, non usiamo initialization_func e viceversa
    if initial_pop is not None:
        # Se abbiamo una popolazione iniziale, initialization_func deve essere NULL
        init_func = c_void_p(0)  # NULL nel C
        initial_pop_struct = initial_pop
    else:
        # Altrimenti usa la funzione di inizializzazione
        init_func = initialization.c_wrapper
        # initial_pop deve essere una struttura vuota (azzerata come nel C)
        # Creiamo una struttura completamente vuota usando ctypes
        initial_pop_struct = Population()
        # ctypes inizializza automaticamente i campi a zero/NULL, ma forziamo l'azzeramento
        ctypes.memset(ctypes.byref(initial_pop_struct), 0, ctypes.sizeof(Population))
    
    options = LGPOptions(
        fitness=fitness.c_wrapper,
        fitness_param=fitness_param_union,
        selection=selection.c_wrapper,
        select_param=select_param_union,
        initialization_func=init_func,
        init_params=init_params_struct,
        initial_pop=initial_pop_struct,
        target=target,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        max_clock=max_clock,
        max_individ_len=max_individ_len,
        max_mutation_len=max_mutation_len,
        generations=generations,
        verbose=verbose
    )
    
    # Call evolve function - IMPORTANTE: impostare signature prima della chiamata
    liblgp.evolve.argtypes = [POINTER(LGPInput), POINTER(LGPOptions)]
    liblgp.evolve.restype = LGPResult
    
    result = liblgp.evolve(ctypes.byref(lgp_input), ctypes.byref(options))
    
    return (
        result.pop,
        result.evaluations,
        result.generations,
        result.best_individ
    )

__all__ = ['LGPOptions', 'evolve']
