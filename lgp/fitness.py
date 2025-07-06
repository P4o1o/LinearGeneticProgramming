"""
Fitness structures and classes - corresponds to fitness.h
"""

from .base import Structure, Union, POINTER, c_uint, c_double, c_char_p, c_void_p, IntEnum, ctypes, liblgp

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


class FitnessType(IntEnum):
    """Corrisponde a enum FitnessType in fitness.h"""
    MINIMIZE = 0
    MAXIMIZE = 1


class FitnessAssessment(Structure):
    """Corrisponde a struct FitnessAssessment in fitness.h - ora corretto il typo"""
    _fields_ = [
        ("fn", c_void_p),
        ("type", c_uint),
        ("name", c_char_p)
    ]
    
    def __init__(self, fitness_type: FitnessType, name: str):
        super().__init__()
        self.fitness_type = fitness_type
        self.name = name.encode('utf-8')  # Converti in bytes per c_char_p
        
    @property
    def c_wrapper(self):
        """Restituisce se stesso come wrapper C"""
        return self
    
    def get_params(self) -> FitnessParams:
        """Override in sottoclassi che necessitano parametri. Default: parametri vuoti"""
        return FitnessParams()  # Parametri vuoti di default


class MSE(FitnessAssessment):
    """Mean Squared Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "MSE")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        # MSE è una struttura globale, la accediamo direttamente
        # liblgp.MSE è l'indirizzo della struttura globale
        return FitnessAssessment.in_dll(liblgp, "MSE")


class RMSE(FitnessAssessment):
    """Root Mean Squared Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "RMSE")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        return FitnessAssessment.in_dll(liblgp, "RMSE")


class LengthPenalizedMSE(FitnessAssessment):
    """Length Penalized MSE fitness"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__(FitnessType.MINIMIZE, "Length Penalized MSE")
        self.alpha = alpha
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        return FitnessAssessment.in_dll(liblgp, "LENGHT_PENALIZED_MSE")
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Length Penalized MSE"""
        params = FitnessParams()
        params.alpha = self.alpha
        return params


class ClockPenalizedMSE(FitnessAssessment):
    """Clock Penalized MSE fitness"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__(FitnessType.MINIMIZE, "Clock Penalized MSE")
        self.alpha = alpha
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        return FitnessAssessment.in_dll(liblgp, "CLOCK_PENALIZED_MSE")
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Clock Penalized MSE"""
        params = FitnessParams()
        params.alpha = self.alpha
        return params


class MAE(FitnessAssessment):
    """Mean Absolute Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "MAE")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        return FitnessAssessment.in_dll(liblgp, "MAE")


class Accuracy(FitnessAssessment):
    """Accuracy fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Accuracy")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        return FitnessAssessment.in_dll(liblgp, "ACCURACY")


class F1Score(FitnessAssessment):
    """F1 Score fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "F1 Score")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        f1_ptr = ctypes.cast(liblgp.F1_SCORE, POINTER(FitnessAssessment))
        return f1_ptr.contents


class MAPE(FitnessAssessment):
    """Mean Absolute Percentage Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "MAPE")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        mape_ptr = ctypes.cast(liblgp.MAPE, POINTER(FitnessAssessment))
        return mape_ptr.contents


class SymmetricMAPE(FitnessAssessment):
    """Symmetric Mean Absolute Percentage Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Symmetric MAPE")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        smape_ptr = ctypes.cast(liblgp.SYMMETRIC_MAPE, POINTER(FitnessAssessment))
        return smape_ptr.contents


class LogCosh(FitnessAssessment):
    """LogCosh fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "LogCosh")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        logcosh_ptr = ctypes.cast(liblgp.LOGCOSH, POINTER(FitnessAssessment))
        return logcosh_ptr.contents


class WorstCaseError(FitnessAssessment):
    """Worst Case Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Worst Case Error")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        wce_ptr = ctypes.cast(liblgp.WORST_CASE_ERROR, POINTER(FitnessAssessment))
        return wce_ptr.contents


class HuberLoss(FitnessAssessment):
    """Huber Loss fitness"""
    
    def __init__(self, delta: float = 1.0):
        super().__init__(FitnessType.MINIMIZE, "Huber Loss")
        self.delta = delta
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        huber_ptr = ctypes.cast(liblgp.HUBER_LOSS, POINTER(FitnessAssessment))
        return huber_ptr.contents
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Huber Loss"""
        params = FitnessParams()
        params.delta = self.delta
        return params


class RSquared(FitnessAssessment):
    """R-Squared fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "R-Squared")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        rsq_ptr = ctypes.cast(liblgp.RSQUARED, POINTER(FitnessAssessment))
        return rsq_ptr.contents


class PinballLoss(FitnessAssessment):
    """Pinball Loss fitness"""
    
    def __init__(self, quantile: float = 0.5):
        super().__init__(FitnessType.MINIMIZE, "Pinball Loss")
        self.quantile = quantile
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        pinball_ptr = ctypes.cast(liblgp.PINBALL_LOSS, POINTER(FitnessAssessment))
        return pinball_ptr.contents
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Pinball Loss"""
        params = FitnessParams()
        params.quantile = self.quantile
        return params


class PearsonCorrelation(FitnessAssessment):
    """Pearson Correlation fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Pearson Correlation")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        pearson_ptr = ctypes.cast(liblgp.PEARSON_CORRELATION, POINTER(FitnessAssessment))
        return pearson_ptr.contents


class ThresholdAccuracy(FitnessAssessment):
    """Threshold Accuracy fitness"""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(FitnessType.MAXIMIZE, "Threshold Accuracy")
        self.threshold = threshold
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        thresh_acc_ptr = ctypes.cast(liblgp.THRESHOLD_ACCURACY, POINTER(FitnessAssessment))
        return thresh_acc_ptr.contents
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Threshold Accuracy"""
        params = FitnessParams()
        params.threshold = self.threshold
        return params


class BalancedAccuracy(FitnessAssessment):
    """Balanced Accuracy fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Balanced Accuracy")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        bal_acc_ptr = ctypes.cast(liblgp.BALANCED_ACCURACY, POINTER(FitnessAssessment))
        return bal_acc_ptr.contents


class GMean(FitnessAssessment):
    """Geometric Mean fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "G-Mean")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        gmean_ptr = ctypes.cast(liblgp.G_MEAN, POINTER(FitnessAssessment))
        return gmean_ptr.contents


class FBetaScore(FitnessAssessment):
    """F-Beta Score fitness"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__(FitnessType.MAXIMIZE, "F-Beta Score")
        self.beta = beta
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        fbeta_ptr = ctypes.cast(liblgp.F_BETA_SCORE, POINTER(FitnessAssessment))
        return fbeta_ptr.contents
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for F-Beta Score"""
        params = FitnessParams()
        params.beta = self.beta
        return params


class BinaryCrossEntropy(FitnessAssessment):
    """Binary Cross Entropy fitness"""
    
    def __init__(self, tolerance: float = 1e-15):
        super().__init__(FitnessType.MINIMIZE, "Binary Cross Entropy")
        self.tolerance = tolerance
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        bce_ptr = ctypes.cast(liblgp.BINARY_CROSS_ENTROPY, POINTER(FitnessAssessment))
        return bce_ptr.contents
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Binary Cross Entropy"""
        params = FitnessParams()
        params.tolerance = self.tolerance
        return params


class GaussianLogLikelihood(FitnessAssessment):
    """Gaussian Log Likelihood fitness"""
    
    def __init__(self, sigma: float = 1.0):
        super().__init__(FitnessType.MAXIMIZE, "Gaussian Log Likelihood")
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        gll_ptr = ctypes.cast(liblgp.GAUSSIAN_LOG_LIKELIHOOD, POINTER(FitnessAssessment))
        return gll_ptr.contents
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Gaussian Log Likelihood"""
        params = FitnessParams()
        params.sigma = self.sigma
        return params


class MatthewsCorrelation(FitnessAssessment):
    """Matthews Correlation Coefficient fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Matthews Correlation")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        mcc_ptr = ctypes.cast(liblgp.MATTHEWS_CORRELATION, POINTER(FitnessAssessment))
        return mcc_ptr.contents


class HingeLoss(FitnessAssessment):
    """Hinge Loss fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Hinge Loss")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        hinge_ptr = ctypes.cast(liblgp.HINGE_LOSS, POINTER(FitnessAssessment))
        return hinge_ptr.contents


class CohensKappa(FitnessAssessment):
    """Cohen's Kappa fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Cohen's Kappa")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        kappa_ptr = ctypes.cast(liblgp.COHENS_KAPPA, POINTER(FitnessAssessment))
        return kappa_ptr.contents


class AdversarialPerturbationSensitivity(FitnessAssessment):
    """Adversarial Perturbation Sensitivity fitness"""
    
    def __init__(self, perturbation_vector: list = None):
        super().__init__(FitnessType.MINIMIZE, "Adversarial Perturbation Sensitivity")
        self.perturbation_vector = perturbation_vector
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        aps_ptr = ctypes.cast(liblgp.ADVERSARIAL_PERTURBATION_SENSITIVITY, POINTER(FitnessAssessment))
        return aps_ptr.contents
    
    def get_params(self) -> FitnessParams:
        """Returns parameters for Adversarial Perturbation Sensitivity"""
        params = FitnessParams()
        if self.perturbation_vector:
            # Convert list to ctypes array pointer
            arr_type = c_double * len(self.perturbation_vector)
            arr = arr_type(*self.perturbation_vector)
            params.perturbation_vector = ctypes.cast(arr, POINTER(c_double))
        return params


class ConditionalValueAtRisk(FitnessAssessment):
    """Conditional Value at Risk fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Conditional Value at Risk")
    
    @property
    def c_wrapper(self) -> FitnessAssessment:
        cvar_ptr = ctypes.cast(liblgp.CONDITIONAL_VALUE_AT_RISK, POINTER(FitnessAssessment))
        return cvar_ptr.contents

__all__ = ['FitnessParams', 'FitnessAssessment', 'FitnessType', 'FitnessAssessment', 
           'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'Accuracy', 'F1Score',
           'MAPE', 'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared', 
           'PinballLoss', 'PearsonCorrelation', 'ThresholdAccuracy', 'BalancedAccuracy', 
           'GMean', 'FBetaScore', 'BinaryCrossEntropy', 'GaussianLogLikelihood', 
           'MatthewsCorrelation', 'HingeLoss', 'CohensKappa', 'AdversarialPerturbationSensitivity', 
           'ConditionalValueAtRisk']
