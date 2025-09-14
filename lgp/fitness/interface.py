"""
Linear Genetic Programming - Fitness Interface Module

Core fitness structures, enums, and base classes for the LGP fitness system.
Corresponds to src/fitness/interface.h/c in the C implementation.
"""

from typing import override
from ..vm import Program
from ..base import Structure, Union, c_uint64, POINTER, c_uint, c_double, c_char_p, c_void_p, IntEnum, ctypes, liblgp
from ..genetics import LGPInput, Individual


class ClusteringFactor(Union):
    """Union for clustering-specific parameters."""
    _fields_ = [
        ("num_clusters", c_uint64),
        ("fuzziness", c_double)
    ]


class FitnessFactor(Union):
    """Union containing different fitness function parameters."""
    _fields_ = [
        ("threshold", c_double),
        ("alpha", c_double),
        ("beta", c_double),
        ("delta", c_double),
        ("quantile", c_double),
        ("tolerance", c_double),
        ("sigma", c_double),
        ("perturbation_vector", POINTER(c_double)),
        ("clustering", ClusteringFactor)
    ]


class FitnessParams(Structure):
    """Parameters for fitness function evaluation."""
    _fields_ = [
        ("start", c_uint64),
        ("end", c_uint64),
        ("fact", FitnessFactor)
    ]

    def __init__(self, start: int = 0, end: int = 0):
        super().__init__()
        if start < 0 or end < 0:
            raise ValueError("Invalid start or end for FitnessParams")
        self.start = c_uint64(start)
        self.end = c_uint64(end)

    @staticmethod
    def new_threshold(threshold: float = 0.5, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with threshold parameter."""
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Invalid threshold for FitnessParams")
        res = FitnessParams(start, end)
        res.fact.threshold = c_double(threshold)
        return res
    
    @staticmethod
    def new_alpha(alpha: float = 0.01, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with alpha parameter."""
        if alpha < 0.0:
            raise ValueError("Invalid alpha for FitnessParams")
        res = FitnessParams(start, end)
        res.fact.alpha = c_double(alpha)
        return res
    
    @staticmethod
    def new_beta(beta: float = 1.0, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with beta parameter."""
        if beta < 0.0:
            raise ValueError("Invalid beta for FitnessParams")
        res = FitnessParams(start, end)
        res.fact.beta = c_double(beta)
        return res
    
    @staticmethod
    def new_delta(delta: float = 1.0, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with delta parameter."""
        if delta < 0.0:
            raise ValueError("Invalid delta for FitnessParams")
        res = FitnessParams(start, end)
        res.fact.delta = c_double(delta)
        return res
    
    @staticmethod
    def new_quantile(quantile: float = 0.5, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with quantile parameter."""
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("Invalid quantile for FitnessParams")
        res = FitnessParams(start, end)
        res.fact.quantile = c_double(quantile)
        return res
    
    @staticmethod
    def new_tolerance(tolerance: float = 1e-15, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with tolerance parameter."""
        if tolerance < 0.0:
            raise ValueError("Invalid tolerance for FitnessParams")
        res = FitnessParams(start, end)
        res.fact.tolerance = c_double(tolerance)
        return res
    
    @staticmethod
    def new_sigma(sigma: float = 1.0, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with sigma parameter."""
        if sigma <= 0.0:
            raise ValueError("Invalid sigma for FitnessParams")
        res = FitnessParams(start, end)
        res.fact.sigma = c_double(sigma)
        return res
    
    @staticmethod
    def new_perturbation_vector(vector, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams with perturbation vector parameter."""
        import numpy as np
        if not isinstance(vector, np.ndarray):
            raise TypeError("perturbation_vector must be a numpy array")
        if vector.size < 1:
            raise ValueError("perturbation_vector must have at least one element")
        res = FitnessParams(start, end)
        arr_type = c_double * vector.size
        arr = arr_type(*vector)
        res.fact.perturbation_vector = ctypes.cast(arr, POINTER(c_double))
        return res
    
    @staticmethod
    def new_clustering(num_clusters: int, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams for clustering functions."""
        if num_clusters < 1:
            raise ValueError("num_clusters must be positive")
        res = FitnessParams(start, end)
        res.fact.clustering.num_clusters = c_uint64(num_clusters)
        return res
    
    @staticmethod
    def new_fuzzy_clustering(num_clusters: int, fuzziness: float = 2.0, start: int = 0, end: int = 0) -> "FitnessParams":
        """Create FitnessParams for fuzzy clustering functions."""
        if num_clusters < 1:
            raise ValueError("num_clusters must be positive")
        if fuzziness <= 1.0:
            raise ValueError("fuzziness must be > 1.0")
        res = FitnessParams(start, end)
        res.fact.clustering.fuzziness = c_double(fuzziness)
        return res


class FitnessType(IntEnum):
    """Enum for fitness optimization direction."""
    MINIMIZE = 0
    MAXIMIZE = 1


class FitnessFunction(Structure):
    """Structure representing a fitness function in C."""
    _fields_ = [
        ("fn", c_void_p),
        ("step", c_void_p),
        ("combine", c_void_p),
        ("finalize", c_void_p),
        ("init_acc", c_void_p),
        ("type", c_uint),
        ("data_type", c_uint),
        ("name", c_char_p)
    ]


class Fitness:
    """Base class for all fitness functions."""
    
    def __init__(self, func: FitnessFunction, params: FitnessParams):
        self._func = func
        self._params = params
        self._func_wrapper = ctypes.cast(self._func.fn, ctypes.CFUNCTYPE(
            c_double,                    # return type
            POINTER(LGPInput),           # LGPInput*
            POINTER(Program),            # Program*
            c_uint64,                    # max_clock
            POINTER(FitnessParams)       # FitnessParams*
        ))
  
    @property
    def function(self) -> FitnessFunction:
        """Get the underlying C fitness function."""
        return self._func

    @property
    def parameters(self) -> FitnessParams:
        """Get the fitness parameters."""
        return self._params
    
    def check_input(self, lgp_input: LGPInput):
        """Validate input for this fitness function. Override in subclasses if needed."""
        pass

    def __call__(self, lgp_input: LGPInput, individual: Individual, max_clock: int = 5000) -> float:
        """Evaluate fitness for the given individual and input."""
        if self._params.end == 0:
            self._params.end = lgp_input.res_size  # Default to the full range if end is not set
        if self._params.start >= lgp_input.res_size or self._params.end > lgp_input.res_size:
            raise ValueError("Invalid start or end for FitnessParams")
        self.check_input(lgp_input)
        result = self._func_wrapper(
            ctypes.byref(lgp_input),
            ctypes.cast(ctypes.byref(individual.prog), POINTER(Program)),  # Cast like print_program does
            c_uint64(max_clock),
            ctypes.byref(self._params)
        )
        
        return float(result)


__all__ = ['FitnessFactor', 'FitnessParams', 'FitnessType', 'FitnessFunction', 'Fitness']
