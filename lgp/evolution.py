"""
Evolution structures and functions - corresponds to evolution.h
"""

from .base import Structure, POINTER, c_uint64, c_uint, c_double, c_void_p, Tuple, Optional, ctypes, liblgp
from .genetics import LGPInput, LGPResultWrapper, Population, PopulationWrapper
from .fitness import FitnessParamsWrapper, FitnessAssessmentWrapper, FitnessAssessment
from .selection import SelectionParamsWrapper, Selection, SelectionWrapper
from .creation import InitializationParamsWrapper, Initialization

class LGPOptionsWrapper(Structure):
    """Corrisponde a struct LGPOptions in evolution.h"""
    _fields_ = [
        ("fitness", FitnessAssessmentWrapper),
        ("fitness_param", FitnessParamsWrapper),
        ("selection", SelectionWrapper),
        ("select_param", SelectionParamsWrapper),
        ("initialization_func", c_void_p),
        ("init_params", InitializationParamsWrapper),
        ("initial_pop", PopulationWrapper),
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
          fitness_param: Optional[float] = None,
          selection: Selection = None,
          select_param: Optional[float] = None,
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
    Esegue l'evoluzione del LGP
    
    Args:
        lgp_input: Input per l'LGP
        fitness: Funzione di fitness (default: MSE)
        fitness_param: Parametro per la funzione di fitness
        selection: Metodo di selezione (default: Tournament)
        select_param: Parametro per la selezione
        initialization: Metodo di inizializzazione (default: UniquePopulation)
        init_params: Parametri di inizializzazione (pop_size, minsize, maxsize)
        target: Target di fitness per fermare l'evoluzione
        mutation_prob: Probabilità di mutazione
        crossover_prob: Probabilità di crossover
        max_clock: Massimo numero di cicli di clock per programma
        max_individ_len: Lunghezza massima degli individui
        max_mutation_len: Lunghezza massima delle mutazioni
        generations: Numero massimo di generazioni
        verbose: Livello di verbosità
        
    Returns:
        Tupla contenente (popolazione finale, valutazioni, generazioni, indice miglior individuo)
    """
    from .fitness import MSE
    from .selection import Tournament
    from .creation import UniquePopulation as UP
    from .genetics import Population
    
    # Imposta valori di default
    if fitness is None:
        fitness = MSE()
    if selection is None:
        selection = Tournament(3)
    if initialization is None:
        initialization = UP()
    if init_params is None:
        init_params = (1000, 2, 5)
    
    # Prepara i parametri
    fitness_param_union = FitnessParamsWrapper()
    if fitness_param is not None:
        fitness_param_union.alpha = fitness_param
    
    select_param_union = SelectionParamsWrapper()
    if select_param is not None:
        if isinstance(select_param, (int, float)) and select_param >= 1:
            select_param_union.size = int(select_param)
        else:
            select_param_union.val = float(select_param)
    
    init_params_struct = InitializationParamsWrapper(
        pop_size=init_params[0],
        minsize=init_params[1],
        maxsize=init_params[2]
    )
    
    # Crea la struttura LGPOptionsWrapper
    options = LGPOptionsWrapper(
        fitness=fitness.c_wrapper,
        fitness_param=fitness_param_union,
        selection=selection.c_wrapper,
        select_param=select_param_union,
        initialization_func=initialization.c_wrapper,
        init_params=init_params_struct,
        initial_pop=initial_pop.c_wrapper if initial_pop else None,
        target=target,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        max_clock=max_clock,
        max_individ_len=max_individ_len,
        max_mutation_len=max_mutation_len,
        generations=generations,
        verbose=verbose
    )
    
    # Chiama la funzione evolve
    result = liblgp.evolve(ctypes.byref(lgp_input.c_wrapper), ctypes.byref(options))
    
    return (
        Population(result.pop),
        result.evaluations,
        result.generations,
        result.best_individ
    )

__all__ = ['LGPOptionsWrapper', 'evolve']
