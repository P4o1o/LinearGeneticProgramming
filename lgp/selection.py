"""
Selection structures and classes - corresponds to selection.h
"""

from .base import Structure, Union, c_uint64, c_double, c_void_p, liblgp

class SelectFactor(Union):
    """Corrisponde all'union select_factor in selection.h"""
    _fields_ = [
        ("size", c_uint64),
        ("val", c_double)
    ]

    def __init__(self):
        super().__init__()

    @staticmethod
    def new_size(size: int = 0) -> "SelectFactor":
        """Imposta il valore di size"""
        if size < 0:
            raise ValueError("Invalid size for SelectFactor in FitnessSharingParams")
        res = SelectFactor()
        res.size = c_uint64(size)
        return res

    @staticmethod
    def new_value(val: float = 0.0) -> "SelectFactor":
        """Imposta il valore di val"""
        if val < 0.0:
            raise ValueError("Invalid value for SelectFactor in FitnessSharingParams")
        res = SelectFactor()
        res.val = c_double(val)
        return res


class FitnessSharingParams(Structure):
    """Corrisponde a struct FitnessSharingParams in selection.h"""
    _fields_ = [
        ("alpha", c_double),
        ("beta", c_double),
        ("sigma", c_double),
        ("select_factor", SelectFactor)
    ]

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0, select_factor: SelectFactor = None):
        super().__init__()
        self.select_factor = select_factor
        if alpha < 0.0 or beta < 0.0 or sigma < 0.0:
            raise ValueError("Invalid parameters for FitnessSharingParams")
        self.alpha = c_double(alpha)
        self.beta = c_double(beta)
        self.sigma = c_double(sigma)


class SelectionParams(Union):
    """Corrisponde a union SelectionParams in selection.h"""
    _fields_ = [
        ("size", c_uint64),
        ("val", c_double),
        ("fs_params", FitnessSharingParams)
    ]

    def __init__(self):
        super().__init__()

    @staticmethod
    def new_fitness_sharing(fs_params: FitnessSharingParams) -> "SelectionParams":
        """Crea un'istanza di SelectionParams da FitnessSharingParams"""
        params = SelectionParams()
        params.fs_params = fs_params
        return params
    
    @staticmethod
    def new_size(size: int = 0) -> "SelectionParams":
        """Crea un'istanza di SelectionParams con un valore di size"""
        if size < 0:
            raise ValueError("Invalid size for SelectionParams")
        params = SelectionParams()
        params.size = c_uint64(size)
        return params
    
    @staticmethod
    def new_value(val: float = 0.0) -> "SelectionParams":
        """Crea un'istanza di SelectionParams con un valore di val"""
        if val < 0.0:
            raise ValueError("Invalid value for SelectionParams")
        params = SelectionParams()
        params.val = c_double(val)
        return params

class SelectionFunction(Structure):
    """Corrisponde a struct Selection in selection.h - unified wrapper + interface"""
    _fields_ = [
        ("type", c_void_p * 2)  # FITNESS_TYPE_NUM = 2
    ]

class Selection():

    def __init__(self, params: SelectionParams):
        self.params = params

    @property
    def function(self) -> SelectionFunction:
        """Override in subclasses to return the corresponding C struct"""
        raise NotImplementedError
    
    @property
    def parameters(self) -> SelectionParams:
        return self.params


class Tournament(Selection):
    """Tournament selection"""
    
    def __init__(self, tournament_size: int = 3):
        super().__init__(SelectionParams.new_size(tournament_size))
    
    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "tournament")
    


class Elitism(Selection):
    """Elitism selection"""
    
    def __init__(self, elite_size: int = 10):
        super().__init__(SelectionParams.new_size(elite_size))
    
    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "elitism")


class PercentualElitism(Selection):
    """Percentual Elitism selection"""
    
    def __init__(self, elite_percentage: float = 0.1):
        super().__init__(SelectionParams.new_value(elite_percentage))
    
    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "percentual_elitism")


class Roulette(Selection):
    """Roulette wheel selection"""
    
    def __init__(self, sampling_size: int = 100):
        super().__init__(SelectionParams.new_size(sampling_size))

    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "roulette")


class FitnessSharingTournament(Selection):
    """Fitness Sharing Tournament selection"""
    
    def __init__(self, tournament_size: int = 3, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__(SelectionParams.new_fitness_sharing(FitnessSharingParams(alpha, beta, sigma, SelectFactor.new_size(tournament_size))))
    
    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "fitness_sharing_tournament")


class FitnessSharingElitism(Selection):
    """Fitness Sharing Elitism selection"""
    
    def __init__(self, elite_size: int = 10, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__(SelectionParams.new_fitness_sharing(FitnessSharingParams(alpha, beta, sigma, SelectFactor.new_size(elite_size))))
    
    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "fitness_sharing_elitism")


class FitnessSharingPercentualElitism(Selection):
    """Fitness Sharing Percentual Elitism selection"""
    
    def __init__(self, elite_percentage: float = 0.1, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__(SelectionParams.new_fitness_sharing(FitnessSharingParams(alpha, beta, sigma, SelectFactor.new_value(elite_percentage))))
    
    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "fitness_sharing_percentual_elitism")


class FitnessSharingRoulette(Selection):
    """Fitness Sharing Roulette selection"""
    
    def __init__(self, sampling_size: int = 100, alpha: float = 1.0, beta: float = 1.0, sigma: float = 1.0):
        super().__init__(SelectionParams.new_fitness_sharing(FitnessSharingParams(alpha, beta, sigma, SelectFactor.new_size(sampling_size))))
    
    @property
    def function(self) -> SelectionFunction:
        return SelectionFunction.in_dll(liblgp, "fitness_sharing_roulette")

__all__ = ['SelectFactor', 'FitnessSharingParams', 'SelectionParams', 'Selection', 'Tournament', 'Elitism', 
           'PercentualElitism', 'Roulette', 'FitnessSharingTournament', 'FitnessSharingElitism', 
           'FitnessSharingPercentualElitism', 'FitnessSharingRoulette']
