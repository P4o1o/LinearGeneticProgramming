"""
Creation/Initialization structures and classes - corresponds to creation.h
"""

from .base import Structure, POINTER, c_uint64, c_void_p, ctypes, liblgp
from .fitness import FitnessAssessment, FitnessParams
from .genetics import LGPInput, LGPResult

class InitializationParams(Structure):
    """Corresponds to struct InitializationParams in creation.h"""
    _fields_ = [
        ("pop_size", c_uint64),
        ("minsize", c_uint64),
        ("maxsize", c_uint64)
    ]


class Initialization:
    """Base class for initialization methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    def c_wrapper(self) -> c_void_p:
        """Override in subclasses to return the corresponding C function"""
        raise NotImplementedError


class UniquePopulation(Initialization):
    """Unique Population initialization - avoids duplicates"""
    
    def __init__(self):
        super().__init__("Unique Population")
    
    @property
    def c_wrapper(self) -> c_void_p:
        return ctypes.cast(liblgp.unique_population, c_void_p)


class RandPopulation(Initialization):
    """Random Population initialization - completely random"""
    
    def __init__(self):
        super().__init__("Random Population")
    
    @property
    def c_wrapper(self) -> c_void_p:
        return ctypes.cast(liblgp.rand_population, c_void_p)

__all__ = ['InitializationParams', 'Initialization', 'UniquePopulation', 'RandPopulation']