"""
Linear Genetic Programming (LGP) Python Interface
Unified wrapper classes combining C structures with user-friendly interfaces
"""

# Main functions
from .base import VECT_ALIGNMENT, INSTR_NUM, NUMBER_OF_OMP_THREADS
from .setup import setup_library
setup_library()
from .utils import print_program, random_init, random_init_all
from .evolution import evolve

# Core classes (unified wrapper + interface)
from .genetics import LGPInput, Program, Individual, Population, VectorDistance, InstructionSet, LGPResult

# Fitness assessment
from .fitness import (
    Fitness, FitnessType,
    MSE, RMSE, LengthPenalizedMSE, ClockPenalizedMSE, MAE, Accuracy, F1Score,
    MAPE, SymmetricMAPE, LogCosh, WorstCaseError, HuberLoss, RSquared,
    PinballLoss, PearsonCorrelation, ThresholdAccuracy, BalancedAccuracy,
    GMean, FBetaScore, BinaryCrossEntropy, GaussianLogLikelihood,
    MatthewsCorrelation, HingeLoss, CohensKappa, AdversarialPerturbationSensitivity,
    ConditionalValueAtRisk, SilhouetteScore,
    DistanceFunction, EuclideanDistance, ManhattanDistance, ChebyshevDistance, CosineDistance
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
    # Costants
    'NUMBER_OF_OMP_THREADS', 'VECT_ALIGNMENT', 'INSTR_NUM',

    # Main functions
    'evolve', 'print_program', 'random_init', 'random_init_all',
    
    # Core classes (unified)
    'LGPInput', 'Program', 'Individual', 'Population', 'VectorDistance', 'InstructionSet', 'LGPResult',
    
    # Fitness
    'Fitness', 'FitnessType',
    'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'Accuracy', 'F1Score',
    'MAPE', 'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared',
    'PinballLoss', 'PearsonCorrelation', 'ThresholdAccuracy', 'BalancedAccuracy',
    'GMean', 'FBetaScore', 'BinaryCrossEntropy', 'GaussianLogLikelihood',
    'MatthewsCorrelation', 'HingeLoss', 'CohensKappa', 'AdversarialPerturbationSensitivity',
    'ConditionalValueAtRisk', 'SilhouetteScore',
    'DistanceFunction', 'EuclideanDistance', 'ManhattanDistance', 'ChebyshevDistance', 'CosineDistance',
    
    # Selection
    'Selection', 'Tournament', 'Elitism', 'PercentualElitism', 'Roulette',
    'FitnessSharingTournament', 'FitnessSharingElitism', 'FitnessSharingPercentualElitism',
    'FitnessSharingRoulette', 'SelectionParams', 'FitnessSharingParams',
    
    # Initialization
    'Initialization', 'UniquePopulation', 'RandPopulation',
    
    # VM
    'Operation'
]