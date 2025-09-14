"""
Linear Genetic Programming - Classification Fitness Module

Classification metrics for the LGP fitness system.
Corresponds to src/fitness/classification.h/c in the C implementation.

Contains metrics like accuracy, F1-score, Cohen's kappa, Matthews correlation, etc.
"""

from ..base import liblgp
from .interface import Fitness, FitnessFunction, FitnessParams


class Accuracy(Fitness):
    """Basic accuracy fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "ACCURACY"), FitnessParams(start, end))


class StrictAccuracy(Fitness):
    """Strict accuracy fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "STRICT_ACCURACY"), FitnessParams(start, end))


class BinaryAccuracy(Fitness):
    """Binary accuracy fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BINARY_ACCURACY"), FitnessParams(start, end))


class StrictBinaryAccuracy(Fitness):
    """Strict binary accuracy fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "STRICT_BINARY_ACCURACY"), FitnessParams(start, end))


class ThresholdAccuracy(Fitness):
    """Threshold-based accuracy fitness function."""
    
    def __init__(self, threshold: float = 0.5, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "THRESHOLD_ACCURACY"), FitnessParams.new_threshold(threshold, start, end))


class StrictThresholdAccuracy(Fitness):
    """Strict threshold-based accuracy fitness function."""
    
    def __init__(self, threshold: float = 0.5, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "STRICT_THRESHOLD_ACCURACY"), FitnessParams.new_threshold(threshold, start, end))


class BalancedAccuracy(Fitness):
    """Balanced accuracy fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BALANCED_ACCURACY"), FitnessParams(start, end))


class GMean(Fitness):
    """Geometric mean of sensitivity and specificity."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "G_MEAN"), FitnessParams(start, end))


class F1Score(Fitness):
    """F1-score fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "F1_SCORE"), FitnessParams(start, end))


class FBetaScore(Fitness):
    """F-beta score fitness function with configurable beta parameter."""
    
    def __init__(self, beta: float = 1.0, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "F_BETA_SCORE"), FitnessParams.new_beta(beta, start, end))


class MatthewsCorrelation(Fitness):
    """Matthews correlation coefficient fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MATTHEWS_CORRELATION"), FitnessParams(start, end))


class CohensKappa(Fitness):
    """Cohen's kappa coefficient fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "COHENS_KAPPA"), FitnessParams(start, end))


__all__ = [
    'Accuracy', 'StrictAccuracy', 'BinaryAccuracy', 'StrictBinaryAccuracy',
    'ThresholdAccuracy', 'StrictThresholdAccuracy', 'BalancedAccuracy', 'GMean', 
    'F1Score', 'FBetaScore', 'MatthewsCorrelation', 'CohensKappa'
]
