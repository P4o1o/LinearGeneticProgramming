"""
Creation/Initialization structures and classes - corresponds to creation.h
"""

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
    
    def __init__(self, params: InitializationParams):
        self.params = params

    @property
    def parameters(self) -> InitializationParams:
        return self.params

    @property
    def function(self) -> c_void_p:
        """Override in subclasses to return the corresponding C function"""
        raise NotImplementedError


class UniquePopulation(Initialization):
    """Unique Population initialization - avoids duplicates"""
    
    def __init__(self, pop_size: int = 100, minsize: int = 1, maxsize: int = 10):
        super().__init__(InitializationParams(pop_size, minsize, maxsize))

    @property
    def function(self) -> c_void_p:
        return ctypes.cast(liblgp.unique_population, c_void_p)


class RandPopulation(Initialization):
    """Random Population initialization - completely random"""
    
    def __init__(self, pop_size: int = 100, minsize: int = 1, maxsize: int = 10):
        super().__init__(InitializationParams(pop_size, minsize, maxsize))
    
    @property
    def function(self) -> c_void_p:
        return ctypes.cast(liblgp.rand_population, c_void_p)

__all__ = ['InitializationParams', 'Initialization', 'UniquePopulation', 'RandPopulation']