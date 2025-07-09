"""
Creation/Initialization structures and classes - corresponds to creation.h
"""

from lgp.fitness import Fitness
from lgp.genetics import LGPInput, LGPResult
from .base import Structure, POINTER, c_uint64, c_void_p, ctypes, liblgp

class InitializationParams(Structure):
    """Corresponds to struct InitializationParams in creation.h"""
    _fields_ = [
        ("pop_size", c_uint64),
        ("minsize", c_uint64),
        ("maxsize", c_uint64)
    ]

    def __init__(self, pop_size: int = 100, minsize: int = 1, maxsize: int = 10):
        super().__init__()
        if pop_size <= 0 or minsize <= 0 or maxsize < minsize:
            raise ValueError("Invalid parameters for InitializationParams")
        self.pop_size = pop_size
        self.minsize = minsize
        self.maxsize = maxsize


class Initialization:
    """Base class for initialization methods"""
    
    def __init__(self, func, params: InitializationParams):
        self._func = func
        self._params = params

    @property
    def parameters(self) -> InitializationParams:
        return self._params

    @property
    def function(self) -> c_void_p:
        return ctypes.cast(self._func, c_void_p)

    def __call__(self, lgp_input: LGPInput, fitness: Fitness, max_clock: int) -> LGPResult:
        return self._func(lgp_input, self._params, fitness.function, max_clock, fitness.parameters)


class UniquePopulation(Initialization):
    """Unique Population initialization - avoids duplicates"""
    
    def __init__(self, pop_size: int = 100, minsize: int = 1, maxsize: int = 10):
        super().__init__(liblgp.unique_population, InitializationParams(pop_size, minsize, maxsize))


class RandPopulation(Initialization):
    """Random Population initialization - completely random"""
    
    def __init__(self, pop_size: int = 100, minsize: int = 1, maxsize: int = 10):
        super().__init__(liblgp.rand_population, InitializationParams(pop_size, minsize, maxsize))


__all__ = ['InitializationParams', 'Initialization', 'UniquePopulation', 'RandPopulation']