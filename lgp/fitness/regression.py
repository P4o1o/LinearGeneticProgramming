"""
Linear Genetic Programming - Regression Fitness Module

Regression metrics for the LGP fitness system.
Corresponds to src/fitness/regression.h/c in the C implementation.

Contains metrics like MSE, RMSE, MAE, R², Pearson correlation, etc.
"""

from ..base import liblgp
from .interface import Fitness, FitnessFunction, FitnessParams


class MSE(Fitness):
    """Mean Squared Error fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MSE"), FitnessParams(start, end))


class RMSE(Fitness):
    """Root Mean Squared Error fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "RMSE"), FitnessParams(start, end))


class LengthPenalizedMSE(Fitness):
    """MSE with program length penalty."""
    
    def __init__(self, alpha: float = 0.01, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "LENGTH_PENALIZED_MSE"), FitnessParams.new_alpha(alpha, start, end))


class ClockPenalizedMSE(Fitness):
    """MSE with execution time penalty."""
    
    def __init__(self, alpha: float = 0.01, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "CLOCK_PENALIZED_MSE"), FitnessParams.new_alpha(alpha, start, end))


class MAE(Fitness):
    """Mean Absolute Error fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MAE"), FitnessParams(start, end))


class MAPE(Fitness):
    """Mean Absolute Percentage Error fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "MAPE"), FitnessParams(start, end))


class SymmetricMAPE(Fitness):
    """Symmetric Mean Absolute Percentage Error fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "SYMMETRIC_MAPE"), FitnessParams(start, end))


class LogCosh(Fitness):
    """Log-cosh loss fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "LOGCOSH"), FitnessParams(start, end))


class WorstCaseError(Fitness):
    """Worst case (maximum) error fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "WORST_CASE_ERROR"), FitnessParams(start, end))


class HuberLoss(Fitness):
    """Huber loss fitness function with delta parameter."""
    
    def __init__(self, delta: float = 1.0, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "HUBER_LOSS"), FitnessParams.new_delta(delta, start, end))


class RSquared(Fitness):
    """R-squared coefficient of determination fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "R_SQUARED"), FitnessParams(start, end))


class PinballLoss(Fitness):
    """Pinball loss (quantile regression) fitness function."""
    
    def __init__(self, quantile: float = 0.5, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "PINBALL_LOSS"), FitnessParams.new_quantile(quantile, start, end))


class PearsonCorrelation(Fitness):
    """Pearson correlation coefficient fitness function."""
    
    def __init__(self, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "PEARSON_CORRELATION"), FitnessParams(start, end))


__all__ = [
    'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'MAPE', 
    'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared', 
    'PinballLoss', 'PearsonCorrelation'
]
