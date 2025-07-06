"""
Utility functions for LGP
"""

from .base import ctypes, liblgp, c_uint64, c_uint32, Tuple
from .vm import ProgramWrapper
from .genetics import Individual
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .genetics import Individual

def print_program(individual: Individual) -> None:
    """
    Stampa il programma di un individuo
    
    Args:
        individual: L'individuo di cui stampare il programma
    """
    # Configura la funzione C
    liblgp.print_program.argtypes = [ctypes.POINTER(ProgramWrapper)]
    liblgp.print_program.restype = None
    
    # Chiama la funzione C
    liblgp.print_program(ctypes.byref(individual.prog))


def random_init(seed: int, threadnum: int = 0) -> None:
    """
    Inizializza il generatore di numeri casuali
    
    Args:
        seed: Seme per il generatore
        threadnum: Numero di thread (default: 0)
    """
    # Configura la funzione C
    liblgp.random_init_wrapper.argtypes = [c_uint64, c_uint32]
    liblgp.random_init_wrapper.restype = None
    
    # Chiama la funzione C
    liblgp.random_init_wrapper(c_uint64(seed), c_uint32(threadnum))

__all__ = ['evolve', 'print_program', 'random_init']