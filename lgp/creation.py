"""
Creation/Initialization structures and classes - corresponds to creation.h
"""

from .base import Structure, POINTER, c_uint64, c_void_p, ctypes, liblgp
from .fitness import FitnessAssessmentWrapper, FitnessParamsWrapper
from .genetics import LGPInputWrapper, LGPResultWrapper

class InitializationParamsWrapper(Structure):
    """Corrisponde a struct InitializationParams in creation.h"""
    _fields_ = [
        ("pop_size", c_uint64),
        ("minsize", c_uint64),
        ("maxsize", c_uint64)
    ]


class Initialization:
    """Classe base per i metodi di inizializzazione"""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    def c_wrapper(self) -> c_void_p:
        """Override in sottoclassi per restituire la funzione C corrispondente"""
        raise NotImplementedError


class UniquePopulation(Initialization):
    """Unique Population initialization - evita duplicati"""
    
    def __init__(self):
        super().__init__("Unique Population")
    
    @property
    def c_wrapper(self) -> c_void_p:
        return liblgp.unique_population


class RandPopulation(Initialization):
    """Random Population initialization - completamente casuale"""
    
    def __init__(self):
        super().__init__("Random Population")
    
    @property
    def c_wrapper(self) -> c_void_p:
        return liblgp.rand_population

__all__ = ['InitializationParamsWrapper', 'Initialization', 'UniquePopulation', 'RandPopulation']