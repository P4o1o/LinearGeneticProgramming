"""
Selection structures and classes - corresponds to selection.h
"""

from .base import Structure, Union, POINTER, c_uint64, c_double, c_void_p, ctypes, liblgp

class SelectFactor(Union):
    """Corrisponde all'union select_factor __all__ = ['SelectFactor', 'FitnessSharingParams', 'SelectionParams', 'Selection', 'Tournament', 'Elitism', 
           'PercentualElitism', 'Roulette', 'FitnessSharingTournament', 'FitnessSharingElitism', 
           'FitnessSharingPercentualElitism', 'FitnessSharingRoulette']struct FitnessSharingParams"""
    _fields_ = [
        ("size", c_uint64),
        ("val", c_double)
    ]


class FitnessSharingParams(Structure):
    """Corrisponde a struct FitnessSharingParams in selection.h"""
    _fields_ = [
        ("alpha", c_double),
        ("beta", c_double),
        ("sigma", c_double),
        ("select_factor", SelectFactor)
    ]


class SelectionParams(Union):
    """Corrisponde a union SelectionParams in selection.h"""
    _fields_ = [
        ("size", c_uint64),
        ("val", c_double),
        ("fs_params", FitnessSharingParams)
    ]


class Selection(Structure):
    """Corrisponde a struct Selection in selection.h - unified wrapper + interface"""
    _fields_ = [
        ("type", c_void_p * 2)  # FITNESS_TYPE_NUM = 2
    ]
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    @property
    def c_wrapper(self):
        """Restituisce se stesso come wrapper C"""
        return self
    
    def get_params(self) -> SelectionParams:
        """Override in sottoclassi per restituire i parametri appropriati"""
        raise NotImplementedError


class Tournament(Selection):
    """Tournament selection"""
    
    def __init__(self, tournament_size: int = 3):
        super().__init__("Tournament")
        self.tournament_size = tournament_size
    
    @property
    def c_wrapper(self):
        return Selection.in_dll(liblgp, "tournament")
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per tournament selection"""
        params = SelectionParams()
        params.size = c_uint64(self.tournament_size)
        return params


class Elitism(Selection):
    """Elitism selection"""
    
    def __init__(self, elite_size: int = 10):
        super().__init__("Elitism")
        self.elite_size = elite_size
    
    @property
    def c_wrapper(self) -> Selection:
        return Selection.in_dll(liblgp, "elitism")
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per elitism selection"""
        params = SelectionParams()
        params.size = c_uint64(self.elite_size)
        return params


class PercentualElitism(Selection):
    """Percentual Elitism selection"""
    
    def __init__(self, elite_percentage: float = 0.1):
        super().__init__("Percentual Elitism")
        self.elite_percentage = elite_percentage
    
    @property
    def c_wrapper(self) -> Selection:
        perc_elit_ptr = ctypes.cast(liblgp.percentual_elitism, POINTER(Selection))
        return perc_elit_ptr.contents
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per percentual elitism selection"""
        params = SelectionParams()
        params.val = c_double(self.elite_percentage)
        return params


class Roulette(Selection):
    """Roulette wheel selection"""
    
    def __init__(self, sampling_size: int = 100):
        super().__init__("Roulette")
        self.sampling_size = sampling_size
    
    @property
    def c_wrapper(self) -> Selection:
        roulette_ptr = ctypes.cast(liblgp.roulette, POINTER(Selection))
        return roulette_ptr.contents
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per roulette selection"""
        params = SelectionParams()
        params.size = c_uint64(self.sampling_size)
        return params


class FitnessSharingTournament(Selection):
    """Fitness Sharing Tournament selection"""
    
    def __init__(self, tournament_size: int = 3, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Tournament")
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> Selection:
        fs_tournament_ptr = ctypes.cast(liblgp.fitness_sharing_tournament, POINTER(Selection))
        return fs_tournament_ptr.contents
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per fitness sharing tournament selection"""
        params = SelectionParams()
        params.fs_params = FitnessSharingParams()
        params.fs_params.alpha = c_double(self.alpha)
        params.fs_params.beta = c_double(self.beta)
        params.fs_params.sigma = c_double(self.sigma)
        params.fs_params.select_factor.size = c_uint64(self.tournament_size)
        return params


class FitnessSharingElitism(Selection):
    """Fitness Sharing Elitism selection"""
    
    def __init__(self, elite_size: int = 10, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Elitism")
        self.elite_size = elite_size
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> Selection:
        fs_elitism_ptr = ctypes.cast(liblgp.fitness_sharing_elitism, POINTER(Selection))
        return fs_elitism_ptr.contents
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per fitness sharing elitism selection"""
        params = SelectionParams()
        params.fs_params = FitnessSharingParams()
        params.fs_params.alpha = c_double(self.alpha)
        params.fs_params.beta = c_double(self.beta)
        params.fs_params.sigma = c_double(self.sigma)
        params.fs_params.select_factor.size = c_uint64(self.elite_size)
        return params


class FitnessSharingPercentualElitism(Selection):
    """Fitness Sharing Percentual Elitism selection"""
    
    def __init__(self, elite_percentage: float = 0.1, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Percentual Elitism")
        self.elite_percentage = elite_percentage
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> Selection:
        fs_perc_elit_ptr = ctypes.cast(liblgp.fitness_sharing_percentual_elitism, POINTER(Selection))
        return fs_perc_elit_ptr.contents
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per fitness sharing percentual elitism selection"""
        params = SelectionParams()
        params.fs_params = FitnessSharingParams()
        params.fs_params.alpha = c_double(self.alpha)
        params.fs_params.beta = c_double(self.beta)
        params.fs_params.sigma = c_double(self.sigma)
        params.fs_params.select_factor.val = c_double(self.elite_percentage)
        return params


class FitnessSharingRoulette(Selection):
    """Fitness Sharing Roulette selection"""
    
    def __init__(self, sampling_size: int = 100, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__("Fitness Sharing Roulette")
        self.sampling_size = sampling_size
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> Selection:
        fs_roulette_ptr = ctypes.cast(liblgp.fitness_sharing_roulette, POINTER(Selection))
        return fs_roulette_ptr.contents
    
    def get_params(self) -> SelectionParams:
        """Restituisce i parametri per fitness sharing roulette selection"""
        params = SelectionParams()
        params.fs_params = FitnessSharingParams()
        params.fs_params.alpha = c_double(self.alpha)
        params.fs_params.beta = c_double(self.beta)
        params.fs_params.sigma = c_double(self.sigma)
        params.fs_params.select_factor.size = c_uint64(self.sampling_size)
        return params

__all__ = ['SelectFactor', 'FitnessSharingParams', 'SelectionParams', 'Selection', 'Selection', 'Tournament', 'Elitism', 
           'PercentualElitism', 'Roulette', 'FitnessSharingTournament', 'FitnessSharingElitism', 
           'FitnessSharingPercentualElitism', 'FitnessSharingRoulette']
