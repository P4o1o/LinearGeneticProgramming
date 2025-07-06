"""
Fitness structures and classes - corresponds to fitness.h
"""

from .base import Structure, Union, POINTER, c_uint, c_double, c_char_p, c_void_p, IntEnum, ctypes, liblgp

class FitnessParamsWrapper(Union):
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


class FitnessAssessmentWrapper(Structure):
    """Corrisponde a struct FitnessAssesment in fitness.h"""
    _fields_ = [
        ("fn", c_void_p),
        ("type", c_uint),
        ("name", c_char_p)
    ]


class FitnessType(IntEnum):
    """Corrisponde a enum FitnessType in fitness.h"""
    MINIMIZE = 0
    MAXIMIZE = 1


class FitnessAssessment:
    """Classe base per le funzioni di fitness"""
    
    def __init__(self, fitness_type: FitnessType, name: str):
        self.fitness_type = fitness_type
        self.name = name
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        """Override in sottoclassi per restituire la struttura C corrispondente"""
        raise NotImplementedError


class MSE(FitnessAssessment):
    """Mean Squared Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "MSE")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        # MSE è un simbolo globale, non una funzione
        mse_ptr = ctypes.cast(liblgp.MSE, POINTER(FitnessAssessmentWrapper))
        return mse_ptr.contents


class RMSE(FitnessAssessment):
    """Root Mean Squared Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "RMSE")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        rmse_ptr = ctypes.cast(liblgp.RMSE, POINTER(FitnessAssessmentWrapper))
        return rmse_ptr.contents


class LengthPenalizedMSE(FitnessAssessment):
    """Length Penalized MSE fitness"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__(FitnessType.MINIMIZE, "Length Penalized MSE")
        self.alpha = alpha
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        lpmse_ptr = ctypes.cast(liblgp.LENGHT_PENALIZED_MSE, POINTER(FitnessAssessmentWrapper))
        return lpmse_ptr.contents


class ClockPenalizedMSE(FitnessAssessment):
    """Clock Penalized MSE fitness"""
    
    def __init__(self, alpha: float = 0.01):
        super().__init__(FitnessType.MINIMIZE, "Clock Penalized MSE")
        self.alpha = alpha
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        cpmse_ptr = ctypes.cast(liblgp.CLOCK_PENALIZED_MSE, POINTER(FitnessAssessmentWrapper))
        return cpmse_ptr.contents


class MAE(FitnessAssessment):
    """Mean Absolute Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "MAE")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        mae_ptr = ctypes.cast(liblgp.MAE, POINTER(FitnessAssessmentWrapper))
        return mae_ptr.contents


class Accuracy(FitnessAssessment):
    """Accuracy fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Accuracy")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        acc_ptr = ctypes.cast(liblgp.ACCURACY, POINTER(FitnessAssessmentWrapper))
        return acc_ptr.contents


class F1Score(FitnessAssessment):
    """F1 Score fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "F1 Score")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        f1_ptr = ctypes.cast(liblgp.F1_SCORE, POINTER(FitnessAssessmentWrapper))
        return f1_ptr.contents


class MAPE(FitnessAssessment):
    """Mean Absolute Percentage Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "MAPE")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        mape_ptr = ctypes.cast(liblgp.MAPE, POINTER(FitnessAssessmentWrapper))
        return mape_ptr.contents


class SymmetricMAPE(FitnessAssessment):
    """Symmetric Mean Absolute Percentage Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Symmetric MAPE")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        smape_ptr = ctypes.cast(liblgp.SYMMETRIC_MAPE, POINTER(FitnessAssessmentWrapper))
        return smape_ptr.contents


class LogCosh(FitnessAssessment):
    """LogCosh fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "LogCosh")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        logcosh_ptr = ctypes.cast(liblgp.LOGCOSH, POINTER(FitnessAssessmentWrapper))
        return logcosh_ptr.contents


class WorstCaseError(FitnessAssessment):
    """Worst Case Error fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Worst Case Error")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        wce_ptr = ctypes.cast(liblgp.WORST_CASE_ERROR, POINTER(FitnessAssessmentWrapper))
        return wce_ptr.contents


class HuberLoss(FitnessAssessment):
    """Huber Loss fitness"""
    
    def __init__(self, delta: float = 1.0):
        super().__init__(FitnessType.MINIMIZE, "Huber Loss")
        self.delta = delta
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        huber_ptr = ctypes.cast(liblgp.HUBER_LOSS, POINTER(FitnessAssessmentWrapper))
        return huber_ptr.contents


class RSquared(FitnessAssessment):
    """R-Squared fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "R-Squared")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        rsq_ptr = ctypes.cast(liblgp.RSQUARED, POINTER(FitnessAssessmentWrapper))
        return rsq_ptr.contents


class PinballLoss(FitnessAssessment):
    """Pinball Loss fitness"""
    
    def __init__(self, quantile: float = 0.5):
        super().__init__(FitnessType.MINIMIZE, "Pinball Loss")
        self.quantile = quantile
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        pinball_ptr = ctypes.cast(liblgp.PINBALL_LOSS, POINTER(FitnessAssessmentWrapper))
        return pinball_ptr.contents


class PearsonCorrelation(FitnessAssessment):
    """Pearson Correlation fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Pearson Correlation")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        pearson_ptr = ctypes.cast(liblgp.PEARSON_CORRELATION, POINTER(FitnessAssessmentWrapper))
        return pearson_ptr.contents


class ThresholdAccuracy(FitnessAssessment):
    """Threshold Accuracy fitness"""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(FitnessType.MAXIMIZE, "Threshold Accuracy")
        self.threshold = threshold
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        thresh_acc_ptr = ctypes.cast(liblgp.THRESHOLD_ACCURACY, POINTER(FitnessAssessmentWrapper))
        return thresh_acc_ptr.contents


class BalancedAccuracy(FitnessAssessment):
    """Balanced Accuracy fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Balanced Accuracy")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        bal_acc_ptr = ctypes.cast(liblgp.BALANCED_ACCURACY, POINTER(FitnessAssessmentWrapper))
        return bal_acc_ptr.contents


class GMean(FitnessAssessment):
    """Geometric Mean fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "G-Mean")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        gmean_ptr = ctypes.cast(liblgp.G_MEAN, POINTER(FitnessAssessmentWrapper))
        return gmean_ptr.contents


class FBetaScore(FitnessAssessment):
    """F-Beta Score fitness"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__(FitnessType.MAXIMIZE, "F-Beta Score")
        self.beta = beta
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        fbeta_ptr = ctypes.cast(liblgp.F_BETA_SCORE, POINTER(FitnessAssessmentWrapper))
        return fbeta_ptr.contents


class BinaryCrossEntropy(FitnessAssessment):
    """Binary Cross Entropy fitness"""
    
    def __init__(self, tolerance: float = 1e-15):
        super().__init__(FitnessType.MINIMIZE, "Binary Cross Entropy")
        self.tolerance = tolerance
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        bce_ptr = ctypes.cast(liblgp.BINARY_CROSS_ENTROPY, POINTER(FitnessAssessmentWrapper))
        return bce_ptr.contents


class GaussianLogLikelihood(FitnessAssessment):
    """Gaussian Log Likelihood fitness"""
    
    def __init__(self, sigma: float = 1.0):
        super().__init__(FitnessType.MAXIMIZE, "Gaussian Log Likelihood")
        self.sigma = sigma
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        gll_ptr = ctypes.cast(liblgp.GAUSSIAN_LOG_LIKELIHOOD, POINTER(FitnessAssessmentWrapper))
        return gll_ptr.contents


class MatthewsCorrelation(FitnessAssessment):
    """Matthews Correlation Coefficient fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Matthews Correlation")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        mcc_ptr = ctypes.cast(liblgp.MATTHEWS_CORRELATION, POINTER(FitnessAssessmentWrapper))
        return mcc_ptr.contents


class HingeLoss(FitnessAssessment):
    """Hinge Loss fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Hinge Loss")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        hinge_ptr = ctypes.cast(liblgp.HINGE_LOSS, POINTER(FitnessAssessmentWrapper))
        return hinge_ptr.contents


class CohensKappa(FitnessAssessment):
    """Cohen's Kappa fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MAXIMIZE, "Cohen's Kappa")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        kappa_ptr = ctypes.cast(liblgp.COHENS_KAPPA, POINTER(FitnessAssessmentWrapper))
        return kappa_ptr.contents


class AdversarialPerturbationSensitivity(FitnessAssessment):
    """Adversarial Perturbation Sensitivity fitness"""
    
    def __init__(self, perturbation_vector: list = None):
        super().__init__(FitnessType.MINIMIZE, "Adversarial Perturbation Sensitivity")
        self.perturbation_vector = perturbation_vector
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        aps_ptr = ctypes.cast(liblgp.ADVERSARIAL_PERTURBATION_SENSITIVITY, POINTER(FitnessAssessmentWrapper))
        return aps_ptr.contents


class ConditionalValueAtRisk(FitnessAssessment):
    """Conditional Value at Risk fitness"""
    
    def __init__(self):
        super().__init__(FitnessType.MINIMIZE, "Conditional Value at Risk")
    
    @property
    def c_wrapper(self) -> FitnessAssessmentWrapper:
        cvar_ptr = ctypes.cast(liblgp.CONDITIONAL_VALUE_AT_RISK, POINTER(FitnessAssessmentWrapper))
        return cvar_ptr.contents

__all__ = ['FitnessParamsWrapper', 'FitnessAssessmentWrapper', 'FitnessType', 'FitnessAssessment', 
           'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'Accuracy', 'F1Score',
           'MAPE', 'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared', 
           'PinballLoss', 'PearsonCorrelation', 'ThresholdAccuracy', 'BalancedAccuracy', 
           'GMean', 'FBetaScore', 'BinaryCrossEntropy', 'GaussianLogLikelihood', 
           'MatthewsCorrelation', 'HingeLoss', 'CohensKappa', 'AdversarialPerturbationSensitivity', 
           'ConditionalValueAtRisk']
