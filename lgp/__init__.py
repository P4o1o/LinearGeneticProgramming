"""
Linear Genetic Programming (LGP) Python Interface
"""

# Funzioni principali
from .utils import print_program, random_init

from .evolution import evolve

# Classi di base
from .genetics import LGPInput, Individual, Population, VectorDistance, InstructionSet

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
    FitnessSharingRoulette
)

# Initialization methods
from .creation import Initialization, UniquePopulation, RandPopulation

# VM and Operations
from .vm import Operation

__version__ = "1.0.0"

__all__ = [
    # Funzioni principali
    'evolve', 'print_program', 'random_init',
    
    # Classi di base
    'LGPInput', 'Individual', 'Population', 'VectorDistance', 'InstructionSet',
    
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
    'FitnessSharingRoulette',
    
    # Initialization
    'Initialization', 'UniquePopulation', 'RandPopulation',
    
    # VM
    'Operation', 'create_instruction_set'
]