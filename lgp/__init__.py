"""
Linear Genetic Programming (LGP) Python Interface
Unified wrapper classes combining C structures with user-friendly interfaces
"""

# Main functions
from .utils import print_program, random_init, random_init_all, get_number_of_threads, NUMBER_OF_OMP_THREADS
from .evolution import evolve

# Core classes (unified wrapper + interface)
from .genetics import LGPInput, Individual, Population, VectorDistance, InstructionSet, LGPResult

# Fitness assessment
from .fitness import (
    FitnessAssessment, FitnessType,
    MSE, RMSE, LengthPenalizedMSE, ClockPenalizedMSE, MAE, Accuracy, F1Score,
    MAPE, SymmetricMAPE, LogCosh, WorstCaseError, HuberLoss, RSquared,
    PinballLoss, PearsonCorrelation, ThresholdAccuracy, BalancedAccuracy,
    GMean, FBetaScore, BinaryCrossEntropy, GaussianLogLikelihood,
    MatthewsCorrelation, HingeLoss, CohensKappa, AdversarialPerturbationSensitivity,
    ConditionalValueAtRisk
)

# Selection methods
from .selection import (
    Selection, Tournament, Elitism, PercentualElitism, Roulette,
    FitnessSharingTournament, FitnessSharingElitism, FitnessSharingPercentualElitism,
    FitnessSharingRoulette, SelectionParams, FitnessSharingParams
)

# Initialization methods
from .creation import Initialization, UniquePopulation, RandPopulation

# VM and Operations
from .vm import Operation

# Initialize random seed for all threads automatically on import
# This ensures thread-safe random number generation for LGP operations
random_init_all(0)

__version__ = "1.0.0"

__all__ = [
    # Main functions
    'evolve', 'print_program', 'random_init', 'random_init_all', 'get_number_of_threads', 'NUMBER_OF_OMP_THREADS',
    
    # Core classes (unified)
    'LGPInput', 'Individual', 'Population', 'VectorDistance', 'InstructionSet', 'LGPResult',
    
    # Fitness
    'FitnessAssessment', 'FitnessType',
    'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'Accuracy', 'F1Score',
    'MAPE', 'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared',
    'PinballLoss', 'PearsonCorrelation', 'ThresholdAccuracy', 'BalancedAccuracy',
    'GMean', 'FBetaScore', 'BinaryCrossEntropy', 'GaussianLogLikelihood',
    'MatthewsCorrelation', 'HingeLoss', 'CohensKappa', 'AdversarialPerturbationSensitivity',
    'ConditionalValueAtRisk',
    
    # Selection
    'Selection', 'Tournament', 'Elitism', 'PercentualElitism', 'Roulette',
    'FitnessSharingTournament', 'FitnessSharingElitism', 'FitnessSharingPercentualElitism',
    'FitnessSharingRoulette', 'SelectionParams', 'FitnessSharingParams',
    
    # Initialization
    'Initialization', 'UniquePopulation', 'RandPopulation',
    
    # VM
    'Operation'
]