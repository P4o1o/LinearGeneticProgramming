from typing import override
from .vm import Program
from .base import Structure, Union, c_uint64, POINTER, c_uint, c_double, c_char_p, c_void_p, IntEnum, ctypes, liblgp

from .genetics import LGPInput, Individual

class FitnessFactor(Union):
    _fields_ = [
        ("threshold", c_double),
        ("alpha", c_double),
        ("beta", c_double),
        ("delta", c_double),
        ("quantile", c_double),
        ("tolerance", c_double),
        ("sigma", c_double),
        ("perturbation_vector", POINTER(c_double))
    ]

class FitnessParams(Structure):
    _fields_ = [
        ("start", c_uint64),
        ("end", c_uint64),
        ("factor", FitnessFactor)
    ]

    def __init__(self, start: int = 0, end: int = 0):
        super().__init__()
        if(start < 0 or end < 0):
            raise ValueError("Invalid start or end for FitnessParams")
        self.start = c_uint64(start)
        self.end = c_uint64(end)

    @staticmethod
    def new_threshold(threshold: float = 0.5, start: int = 0, end: int = 0) -> "FitnessParams":
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Invalid threshold for FitnessParams")
        res = FitnessParams(start, end)
        res.factor.threshold = c_double(threshold)
        return res
    
    @staticmethod
    def new_alpha(alpha: float = 0.01, start: int = 0, end: int = 0) -> "FitnessParams":
        if alpha < 0.0:
            raise ValueError("Invalid alpha for FitnessParams")
        res = FitnessParams(start, end)
        res.factor.alpha = c_double(alpha)
        return res
    
    @staticmethod
    def new_beta(beta: float = 1.0, start: int = 0, end: int = 0) -> "FitnessParams":
        if beta < 0.0:
            raise ValueError("Invalid beta for FitnessParams")
        res = FitnessParams(start, end)
        res.factor.beta = c_double(beta)
        return res
    
    @staticmethod
    def new_delta(delta: float = 1.0, start: int = 0, end: int = 0) -> "FitnessParams":
        if delta < 0.0:
            raise ValueError("Invalid delta for FitnessParams")
        res = FitnessParams(start, end)
        res.factor.delta = c_double(delta)
        return res
    
    @staticmethod
    def new_quantile(quantile: float = 0.5, start: int = 0, end: int = 0) -> "FitnessParams":
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("Invalid quantile for FitnessParams")
        res = FitnessParams(start, end)
        res.factor.quantile = c_double(quantile)
        return res
    
    @staticmethod
    def new_tolerance(tolerance: float = 1e-15, start: int = 0, end: int = 0) -> "FitnessParams":
        if tolerance < 0.0:
            raise ValueError("Invalid tolerance for FitnessParams")
        res = FitnessParams(start, end)
        res.factor.tolerance = c_double(tolerance)
        return res
    
    @staticmethod
    def new_sigma(sigma: float = 1.0, start: int = 0, end: int = 0) -> "FitnessParams":
        if sigma <= 0.0:
            raise ValueError("Invalid sigma for FitnessParams")
        res = FitnessParams(start, end)
        res.factor.sigma = c_double(sigma)
        return res
    
    @staticmethod
    def new_perturbation_vector(vector, start: int = 0, end: int = 0) -> "FitnessParams":
        import numpy as np
        if not isinstance(vector, np.ndarray):
            raise TypeError("perturbation_vector must be a numpy array")
        if vector.size < 1:
            raise ValueError("perturbation_vector must have at least one element")
        res = FitnessParams(start, end)
        arr_type = c_double * vector.size
        arr = arr_type(*vector)
        res.factor.perturbation_vector = ctypes.cast(arr, POINTER(c_double))
        return res

class FitnessType(IntEnum):
    MINIMIZE = 0
    MAXIMIZE = 1

class FitnessFunction(Structure):
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

class Fitness():
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
        return self._func

    @property
    def parameters(self) -> FitnessParams:
        return self._params
    
    def check_input(self, lgp_input: LGPInput):
        # Throw a ValueError if the input is not valid
        pass

    def __call__(self, lgp_input: LGPInput, individual: Individual, max_clock: int = 5000) -> float:
        if(self._params.end == 0):
            self._params.end = lgp_input.res_size  # Default to the full range if end is not set
        if(self._params.start >= lgp_input.res_size or self._params.end > lgp_input.res_size):
            raise ValueError("Invalid start or end for FitnessParams")
        self.check_input(lgp_input)
        result = self._func_wrapper(
            ctypes.byref(lgp_input),
            ctypes.cast(ctypes.byref(individual.prog), POINTER(Program)),  # Cast like print_program does
            c_uint64(max_clock),
            ctypes.byref(self._params)
        )
        
        return float(result)


class MSE(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MSE"), FitnessParams(start, end))


class RMSE(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "RMSE"), FitnessParams(start, end))


class LengthPenalizedMSE(Fitness):
    def __init__(self, alpha: float = 0.01, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "LENGTH_PENALIZED_MSE"), FitnessParams.new_alpha(alpha, start, end))


class ClockPenalizedMSE(Fitness):
    def __init__(self, alpha: float = 0.01, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "CLOCK_PENALIZED_MSE"), FitnessParams.new_alpha(alpha, start, end))


class MAE(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MAE"), FitnessParams(start, end))


class Accuracy(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "ACCURACY"), FitnessParams(start, end))


class F1Score(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "F1_SCORE"), FitnessParams(start, end))


class MAPE(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MAPE"), FitnessParams(start, end))


class SymmetricMAPE(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "SYMMETRIC_MAPE"), FitnessParams(start, end))


class LogCosh(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "LOGCOSH"), FitnessParams(start, end))


class WorstCaseError(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "WORST_CASE_ERROR"), FitnessParams(start, end))


class HuberLoss(Fitness):
    def __init__(self, delta: float = 1.0, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "HUBER_LOSS"), FitnessParams.new_delta(delta, start, end))


class RSquared(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "R_SQUARED"), FitnessParams(start, end))


class PinballLoss(Fitness):
    def __init__(self, quantile: float = 0.5, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "PINBALL_LOSS"), FitnessParams.new_quantile(quantile, start, end))


class PearsonCorrelation(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "PEARSON_CORRELATION"), FitnessParams(start, end))


class ThresholdAccuracy(Fitness):
    def __init__(self, threshold: float = 0.5, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "THRESHOLD_ACCURACY"), FitnessParams.new_threshold(threshold, start, end))


class BalancedAccuracy(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BALANCED_ACCURACY"), FitnessParams(start, end))


class GMean(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "G_MEAN"), FitnessParams(start, end))


class FBetaScore(Fitness):
    def __init__(self, beta: float = 1.0, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "F_BETA_SCORE"), FitnessParams.new_beta(beta, start, end))

class BinaryCrossEntropy(Fitness):
    def __init__(self, tolerance: float = 1e-15, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BINARY_CROSS_ENTROPY"), FitnessParams.new_tolerance(tolerance, start, end))


class GaussianLogLikelihood(Fitness):
    def __init__(self, sigma: float = 1.0, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "GAUSSIAN_LOG_LIKELIHOOD"), FitnessParams.new_sigma(sigma, start, end))


class MatthewsCorrelation(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MATTHEWS_CORRELATION"), FitnessParams(start, end))


class HingeLoss(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "HINGE_LOSS"), FitnessParams(start, end))


class CohensKappa(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "COHENS_KAPPA"), FitnessParams(start, end))


class AdversarialPerturbationSensitivity(Fitness):
    def __init__(self, perturbation_vector, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "ADVERSARIAL_PERTURBATION_SENSITIVITY"), FitnessParams.new_perturbation_vector(perturbation_vector, start, end))

    @override
    def check_input(self, lgp_input: LGPInput):
        if(lgp_input.input_num != len(self._params.factor.perturbation_vector.contents)):
            raise ValueError("Invalid perturbation vector for AdversarialPerturbationSensitivity")

class ConditionalValueAtRisk(Fitness):
    def __init__(self, alpha: float = 0.05, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "CONDITIONAL_VALUE_AT_RISK"), FitnessParams.new_alpha(alpha, start, end))
        if(alpha >= 1.0):
            raise ValueError("Invalid alpha for ConditionalValueAtRisk, must be in (0.0, 1.0)")

    @override
    def check_input(self, lgp_input: LGPInput):
        if(lgp_input.input_num * self._params.factor.alpha + 0.5 <= 0):
            raise ValueError("Invalid alpha for ConditionalValueAtRisk")


class StrictAccuracy(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "STRICT_ACCURACY"), FitnessParams(start, end))


class BinaryAccuracy(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BINARY_ACCURACY"), FitnessParams(start, end))


class StrictBinaryAccuracy(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "STRICT_BINARY_ACCURACY"), FitnessParams(start, end))


class StrictThresholdAccuracy(Fitness):
    def __init__(self, threshold: float = 0.5, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "STRICT_THRESHOLD_ACCURACY"), FitnessParams.new_threshold(threshold, start, end))


class BrierScore(Fitness):
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BRIER_SCORE"), FitnessParams(start, end))


__all__ = ['FitnessParams', 'FitnessType', 'FitnessFunction', 'Fitness', 
           'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'Accuracy', 'F1Score',
           'MAPE', 'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared', 
           'PinballLoss', 'PearsonCorrelation', 'ThresholdAccuracy', 'BalancedAccuracy', 
           'GMean', 'FBetaScore', 'BinaryCrossEntropy', 'GaussianLogLikelihood', 
           'MatthewsCorrelation', 'HingeLoss', 'CohensKappa', 'AdversarialPerturbationSensitivity', 
           'ConditionalValueAtRisk', 'StrictAccuracy', 'BinaryAccuracy', 'StrictBinaryAccuracy', 
           'StrictThresholdAccuracy', 'BrierScore']
