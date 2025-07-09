"""
Utility functions for LGP
"""

from .base import ctypes, liblgp, c_uint32, POINTER
from .vm import Program
from .genetics import Individual

def print_program(individual: Individual) -> None:
    """
    Prints the program of an individual
    
    Args:
        individual: The individual whose program to print
    """
    # Safety checks before calling C function
    if individual.size == 0:
        print("   [Empty program - no instructions]")
        return
    
    if individual.size > 100000:  # Sanity check for reasonable program size
        print(f"   [Program too large to display safely: {individual.size} instructions]")
        return
    liblgp.print_program(ctypes.cast(ctypes.byref(individual.prog), POINTER(Program)))


def random_init(seed: int, thread_num: int = 0) -> None:
    """
    Initializes the random number generator for a specific thread
    
    Args:
        seed: Seed for the generator
        thread_num: Thread number (default: 0)
    """
    liblgp.random_init_wrapper(c_uint32(seed), c_uint32(thread_num))


def random_init_all(seed: int) -> None:
    """
    Initializes the random number generator for all available threads
    
    Args:
        seed: Seed for all generators
    """
    liblgp.random_init_all(c_uint32(seed))

__all__ = ['print_program', 'random_init', 'random_init_all']