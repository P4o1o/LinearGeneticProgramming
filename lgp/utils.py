"""
Utility functions for LGP
"""

from .base import ctypes, liblgp, c_uint64, c_uint32, Tuple, POINTER
from .vm import Program
from .genetics import Individual
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    
    try:
        # Configure the C function
        liblgp.print_program.argtypes = [POINTER(Program)]
        liblgp.print_program.restype = None
        
        # Call the C function with the individual's program
        liblgp.print_program(ctypes.byref(individual.prog))
    except Exception as e:
        print(f"   [Error printing program: {e}]")
        print(f"   [Program size: {individual.size}, fitness: {individual.fitness}]")


def random_init(seed: int, thread_num: int = 0) -> None:
    """
    Initializes the random number generator for a specific thread
    
    Args:
        seed: Seed for the generator
        thread_num: Thread number (default: 0)
    """
    # Configure the C function
    liblgp.random_init_wrapper.argtypes = [c_uint32, c_uint32]
    liblgp.random_init_wrapper.restype = None
    
    # Call the C function
    liblgp.random_init_wrapper(c_uint32(seed), c_uint32(thread_num))


def random_init_all(seed: int) -> None:
    """
    Initializes the random number generator for all available threads
    
    Args:
        seed: Seed for all generators
    """
    # Configure the C function
    liblgp.random_init_all.argtypes = [c_uint32]
    liblgp.random_init_all.restype = None
    
    # Call the C function
    liblgp.random_init_all(c_uint32(seed))


def get_number_of_threads() -> int:
    """
    Returns the number of available OpenMP threads
    
    Returns:
        Number of threads configured at compile time
    """
    # Access the global constant NUMBER_OF_THREADS from the C library
    # It's defined as extern const uint64_t NUMBER_OF_THREADS in prob.h
    number_of_threads = c_uint64.in_dll(liblgp, "NUMBER_OF_THREADS")
    return int(number_of_threads.value)


# Export the constant as a module-level variable for convenience
try:
    NUMBER_OF_OMP_THREADS = get_number_of_threads()
except:
    # Fallback if the constant is not available
    NUMBER_OF_OMP_THREADS = 1

__all__ = ['print_program', 'random_init', 'random_init_all', 'get_number_of_threads', 'NUMBER_OF_OMP_THREADS']