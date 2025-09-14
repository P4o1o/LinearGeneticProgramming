"""
Linear Genetic Programming - Advanced Fitness Module

Advanced metrics for the LGP fitness system.
Corresponds to src/fitness/advanced.h/c in the C implementation.

Contains specialized metrics like adversarial perturbation sensitivity and conditional value at risk.
"""

from typing import override
from ..base import liblgp
from ..genetics import LGPInput
from .interface import Fitness, FitnessFunction, FitnessParams


class AdversarialPerturbationSensitivity(Fitness):
    """Adversarial perturbation sensitivity fitness function with perturbation vector."""
    
    def __init__(self, perturbation_vector, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "ADVERSARIAL_PERTURBATION_SENSITIVITY"), FitnessParams.new_perturbation_vector(perturbation_vector, start, end))

    @override
    def check_input(self, lgp_input: LGPInput):
        """Validate that the perturbation vector matches the input dimensions."""
        if lgp_input.input_num != len(self._params.factor.perturbation_vector.contents):
            raise ValueError("Invalid perturbation vector for AdversarialPerturbationSensitivity")


class ConditionalValueAtRisk(Fitness):
    """Conditional Value at Risk (CVaR) fitness function with alpha parameter."""
    
    def __init__(self, alpha: float = 0.05, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "CONDITIONAL_VALUE_AT_RISK"), FitnessParams.new_alpha(alpha, start, end))
        if alpha >= 1.0:
            raise ValueError("Invalid alpha for ConditionalValueAtRisk, must be in (0.0, 1.0)")

    @override
    def check_input(self, lgp_input: LGPInput):
        """Validate that alpha is compatible with the input size."""
        if lgp_input.input_num * self._params.factor.alpha + 0.5 <= 0:
            raise ValueError("Invalid alpha for ConditionalValueAtRisk")


__all__ = [
    'AdversarialPerturbationSensitivity', 'ConditionalValueAtRisk'
]
