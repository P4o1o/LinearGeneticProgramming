"""
Fitness structures and classes - corresponds to fitness.h
"""

from .base import Structure, Union, POINTER, c_uint, c_double, c_char_p, c_void_p, IntEnum, ctypes, liblgp, np

class FitnessParams(Union):
    """Corrisponde a union FitnessParams in fitness.h"""
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

    def __init__(self):
        super().__init__()

    @staticmethod
    def new_threshold(threshold: float = 0.5) -> "FitnessParams":
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Invalid threshold for FitnessParams")
        res = FitnessParams()
        res.threshold = c_double(threshold)
        return res
    
    @staticmethod
    def new_alpha(alpha: float = 0.01) -> "FitnessParams":
        if alpha < 0.0:
            raise ValueError("Invalid alpha for FitnessParams")
        res = FitnessParams()
        res.alpha = c_double(alpha)
        return res
    
    @staticmethod
    def new_beta(beta: float = 1.0) -> "FitnessParams":
        if beta < 0.0:
            raise ValueError("Invalid beta for FitnessParams")
        res = FitnessParams()
        res.beta = c_double(beta)
        return res
    
    @staticmethod
    def new_delta(delta: float = 1.0) -> "FitnessParams":
        if delta < 0.0:
            raise ValueError("Invalid delta for FitnessParams")
        res = FitnessParams()
        res.delta = c_double(delta)
        return res
    
    @staticmethod
    def new_quantile(quantile: float = 0.5) -> "FitnessParams":
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("Invalid quantile for FitnessParams")
        res = FitnessParams()
        res.quantile = c_double(quantile)
        return res
    
    @staticmethod
    def new_tolerance(tolerance: float = 1e-15) -> "FitnessParams":
        if tolerance < 0.0:
            raise ValueError("Invalid tolerance for FitnessParams")
        res = FitnessParams()
        res.tolerance = c_double(tolerance)
        return res
    
    @staticmethod
    def new_sigma(sigma: float = 1.0) -> "FitnessParams":
        if sigma <= 0.0:
            raise ValueError("Invalid sigma for FitnessParams")
        res = FitnessParams()
        res.sigma = c_double(sigma)
        return res
    
    @staticmethod
    def new_perturbation_vector(vector: np.array) -> "FitnessParams":
        if vector.size == 0:
            raise ValueError("perturbation_vector cannot be empty")
        res = FitnessParams()
        arr_type = c_double * vector.size
        arr = arr_type(*vector)
        res.perturbation_vector = ctypes.cast(arr, POINTER(c_double))
        return res

class FitnessType(IntEnum):
    """Corrisponde a enum FitnessType in fitness.h"""
    MINIMIZE = 0
    MAXIMIZE = 1

class FitnessFunction(Structure):
    """Corrisponde a struct Fitness in fitness.h"""
    _fields_ = [
        ("fn", c_void_p),
        ("type", c_uint),
        ("name", c_char_p)
    ]

class Fitness():
    def __init__(self, params: FitnessParams):
        self.params = params
  
    @property
    def function(self) -> FitnessFunction:
        """Restituisce la struttura Fitness dal codice C"""
        raise NotImplementedError

    @property
    def parameters(self) -> FitnessParams:
        return self.params


class MSE(Fitness):
    """Mean Squared Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "MSE")


class RMSE(Fitness):
    """Root Mean Squared Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "RMSE")


class LengthPenalizedMSE(Fitness):
    """Length Penalized MSE fitness"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__(FitnessParams.new_alpha(alpha))
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "LENGTH_PENALIZED_MSE")


class ClockPenalizedMSE(Fitness):
    """Clock Penalized MSE fitness"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__(FitnessParams.new_alpha(alpha))
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "CLOCK_PENALIZED_MSE")


class MAE(Fitness):
    """Mean Absolute Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "MAE")


class Accuracy(Fitness):
    """Accuracy fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "ACCURACY")


class F1Score(Fitness):
    """F1 Score fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "F1_SCORE")


class MAPE(Fitness):
    """Mean Absolute Percentage Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "MAPE")


class SymmetricMAPE(Fitness):
    """Symmetric Mean Absolute Percentage Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "SYMMETRIC_MAPE")


class LogCosh(Fitness):
    """LogCosh fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "LOGCOSH")


class WorstCaseError(Fitness):
    """Worst Case Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "WORST_CASE_ERROR")


class HuberLoss(Fitness):
    """Huber Loss fitness"""
    
    def __init__(self, delta: float = 1.0):
        super().__init__(FitnessParams.new_delta(delta))

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "HUBER_LOSS")


class RSquared(Fitness):
    """R-Squared fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "R_SQUARED")


class PinballLoss(Fitness):
    """Pinball Loss fitness"""
    
    def __init__(self, quantile: float = 0.5):
        super().__init__(FitnessParams.new_quantile(quantile))
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "PINBALL_LOSS")


class PearsonCorrelation(Fitness):
    """Pearson Correlation fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "PEARSON_CORRELATION")


class ThresholdAccuracy(Fitness):
    """Threshold Accuracy fitness"""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(FitnessParams.new_threshold(threshold))

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "THRESHOLD_ACCURACY")


class BalancedAccuracy(Fitness):
    """Balanced Accuracy fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "BALANCED_ACCURACY")


class GMean(Fitness):
    """Geometric Mean fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "G_MEAN")


class FBetaScore(Fitness):
    """F-Beta Score fitness"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__(FitnessParams.new_beta(beta))

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "F_BETA_SCORE")

class BinaryCrossEntropy(Fitness):
    """Binary Cross Entropy fitness"""
    
    def __init__(self, tolerance: float = 1e-15):
        super().__init__(FitnessParams.new_tolerance(tolerance))

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "BINARY_CROSS_ENTROPY")


class GaussianLogLikelihood(Fitness):
    """Gaussian Log Likelihood fitness"""
    
    def __init__(self, sigma: float = 1.0):
        super().__init__(FitnessParams.new_sigma(sigma))

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "GAUSSIAN_LOG_LIKELIHOOD")


class MatthewsCorrelation(Fitness):
    """Matthews Correlation Coefficient fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "MATTHEWS_CORRELATION")


class HingeLoss(Fitness):
    """Hinge Loss fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "HINGE_LOSS")


class CohensKappa(Fitness):
    """Cohen's Kappa fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())

    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "COHENS_KAPPA")


class AdversarialPerturbationSensitivity(Fitness):
    """Adversarial Perturbation Sensitivity fitness"""
    
    def __init__(self, perturbation_vector: np.array):
        super().__init__(FitnessParams.new_perturbation_vector(perturbation_vector))
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "ADVERSARIAL_PERTURBATION_SENSITIVITY")

class ConditionalValueAtRisk(Fitness):
    """Conditional Value at Risk fitness"""
    
    def __init__(self):
        super().__init__(FitnessParams())
    
    @property
    def function(self) -> FitnessFunction:
        return FitnessFunction.in_dll(liblgp, "CONDITIONAL_VALUE_AT_RISK")

__all__ = ['FitnessParams', 'FitnessType', 'FitnessFunction', 'Fitness', 
           'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'Accuracy', 'F1Score',
           'MAPE', 'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared', 
           'PinballLoss', 'PearsonCorrelation', 'ThresholdAccuracy', 'BalancedAccuracy', 
           'GMean', 'FBetaScore', 'BinaryCrossEntropy', 'GaussianLogLikelihood', 
           'MatthewsCorrelation', 'HingeLoss', 'CohensKappa', 'AdversarialPerturbationSensitivity', 
           'ConditionalValueAtRisk']
