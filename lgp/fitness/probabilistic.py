"""
Linear Genetic Programming - Probabilistic Fitness Module

Probabilistic metrics for the LGP fitness system.
Corresponds to src/fitness/probabilistic.h/c in the C implementation.

Contains metrics like cross-entropy, Gaussian likelihood, Brier score, hinge loss, etc.
"""

from ..base import liblgp
from .interface import Fitness, FitnessFunction, FitnessParams


class BinaryCrossEntropy(Fitness):
    """Binary cross-entropy fitness function with tolerance parameter."""
    
    def __init__(self, tolerance: float = 1e-15, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BINARY_CROSS_ENTROPY"), FitnessParams.new_tolerance(tolerance, start, end))


class GaussianLogLikelihood(Fitness):
    """Gaussian log-likelihood fitness function with sigma parameter."""
    
    def __init__(self, sigma: float = 1.0, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "GAUSSIAN_LOG_LIKELIHOOD"), FitnessParams.new_sigma(sigma, start, end))


class HingeLoss(Fitness):
    """Hinge loss fitness function (used in SVM)."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "HINGE_LOSS"), FitnessParams(start, end))


class BrierScore(Fitness):
    """Brier score fitness function for probabilistic predictions."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "BRIER_SCORE"), FitnessParams(start, end))


__all__ = [
    'BinaryCrossEntropy', 'GaussianLogLikelihood', 'HingeLoss', 'BrierScore'
]
