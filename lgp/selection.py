"""
Selection structures and classes - corresponds to selection.h
"""

from .base import Structure, Union, POINTER, c_uint64, c_double, c_void_p, ctypes, liblgp

class SelectionParamsWrapper(Union):
    """Corrisponde a union SelectionParams in selection.h"""
    _fields_ = [
        ("size", c_uint64),
        ("val", c_double),
        # fitness sharing parameters omessi per semplicità
    ]


class SelectionWrapper(Structure):
    """Corrisponde a struct Selection in selection.h"""
    _fields_ = [
        ("type", c_void_p * 2)  # FITNESS_TYPE_NUM = 2
    ]


class Selection:
    """Classe base per i metodi di selezione"""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        """Override in sottoclassi per restituire la struttura C corrispondente"""
        raise NotImplementedError


class Tournament(Selection):
    """Tournament selection"""
    
    def __init__(self, tournament_size: int = 3):
        super().__init__("Tournament")
        self.tournament_size = tournament_size
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        tournament_ptr = ctypes.cast(liblgp.tournament, POINTER(SelectionWrapper))
        return tournament_ptr.contents


class Elitism(Selection):
    """Elitism selection"""
    
    def __init__(self, elite_size: int = 10):
        super().__init__("Elitism")
        self.elite_size = elite_size
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        elitism_ptr = ctypes.cast(liblgp.elitism, POINTER(SelectionWrapper))
        return elitism_ptr.contents


class PercentualElitism(Selection):
    """Percentual Elitism selection"""
    
    def __init__(self, elite_percentage: float = 0.1):
        super().__init__("Percentual Elitism")
        self.elite_percentage = elite_percentage
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        perc_elit_ptr = ctypes.cast(liblgp.percentual_elitism, POINTER(SelectionWrapper))
        return perc_elit_ptr.contents


class Roulette(Selection):
    """Roulette wheel selection"""
    
    def __init__(self, sampling_size: int = 100):
        super().__init__("Roulette")
        self.sampling_size = sampling_size
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        roulette_ptr = ctypes.cast(liblgp.roulette, POINTER(SelectionWrapper))
        return roulette_ptr.contents


class FitnessSharingTournament(Selection):
    """Fitness Sharing Tournament selection"""
    
    def __init__(self, tournament_size: int = 3, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Tournament")
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        fs_tournament_ptr = ctypes.cast(liblgp.fitness_sharing_tournament, POINTER(SelectionWrapper))
        return fs_tournament_ptr.contents


class FitnessSharingElitism(Selection):
    """Fitness Sharing Elitism selection"""
    
    def __init__(self, elite_size: int = 10, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Elitism")
        self.elite_size = elite_size
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        fs_elitism_ptr = ctypes.cast(liblgp.fitness_sharing_elitism, POINTER(SelectionWrapper))
        return fs_elitism_ptr.contents


class FitnessSharingPercentualElitism(Selection):
    """Fitness Sharing Percentual Elitism selection"""
    
    def __init__(self, elite_percentage: float = 0.1, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Percentual Elitism")
        self.elite_percentage = elite_percentage
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        fs_perc_elit_ptr = ctypes.cast(liblgp.fitness_sharing_percentual_elitism, POINTER(SelectionWrapper))
        return fs_perc_elit_ptr.contents


class FitnessSharingRoulette(Selection):
    """Fitness Sharing Roulette selection"""
    
    def __init__(self, sampling_size: int = 100, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Roulette")
        self.sampling_size = sampling_size
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> SelectionWrapper:
        fs_roulette_ptr = ctypes.cast(liblgp.fitness_sharing_roulette, POINTER(SelectionWrapper))
        return fs_roulette_ptr.contents

__all__ = ['SelectionParamsWrapper', 'SelectionWrapper', 'Selection', 'Tournament', 'Elitism', 
           'PercentualElitism', 'Roulette', 'FitnessSharingTournament', 'FitnessSharingElitism', 
           'FitnessSharingPercentualElitism', 'FitnessSharingRoulette']
