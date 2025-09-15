"""
Fitness evaluation module for Linear Genetic Programming.

This module provides various fitness functions organized by category:
- regression: Mean squared error, mean absolute error, etc.
- classification: Accuracy, precision, recall, F1-score, etc.
- probabilistic: Log-likelihood, cross-entropy, KL-divergence, etc.
- advanced: Custom and specialized fitness functions
- clustering: Silhouette score, inertia, adjusted rand index, etc.
"""

from . import interface
from . import regression
from . import classification
from . import probabilistic
from . import advanced
from . import clustering

__all__ = ['interface', 'regression', 'classification', 'probabilistic', 'advanced', 'clustering']

# Core fitness structures and base classes
from .interface import (
    FitnessFactor,
    FitnessParams, 
    FitnessType,
    FitnessFunction,
    Fitness
)

# Regression metrics
from .regression import (
    MSE,
    RMSE,
    LengthPenalizedMSE,
    ClockPenalizedMSE,
    MAE,
    MAPE,
    SymmetricMAPE,
    LogCosh,
    WorstCaseError,
    HuberLoss,
    RSquared,
    PinballLoss,
    PearsonCorrelation
)

# Classification metrics
from .classification import (
    Accuracy,
    StrictAccuracy,
    BinaryAccuracy,
    StrictBinaryAccuracy,
    ThresholdAccuracy,
    StrictThresholdAccuracy,
    BalancedAccuracy,
    GMean,
    F1Score,
    FBetaScore,
    MatthewsCorrelation,
    CohensKappa
)

# Probabilistic metrics
from .probabilistic import (
    BinaryCrossEntropy,
    GaussianLogLikelihood,
    HingeLoss,
    BrierScore
)

# Advanced metrics
from .advanced import (
    AdversarialPerturbationSensitivity,
    ConditionalValueAtRisk
)

# Clustering metrics
from .clustering import (
    SilhouetteScore
)

# Distance functions
from .distances import (
    DistanceFunction, EuclideanDistance, ManhattanDistance, 
    ChebyshevDistance, CosineDistance
)

__all__ = [
    # Core interface
    'FitnessFactor', 'FitnessParams', 'FitnessType', 'FitnessFunction', 'Fitness',
    
    # Regression metrics
    'MSE', 'RMSE', 'LengthPenalizedMSE', 'ClockPenalizedMSE', 'MAE', 'MAPE', 
    'SymmetricMAPE', 'LogCosh', 'WorstCaseError', 'HuberLoss', 'RSquared', 
    'PinballLoss', 'PearsonCorrelation',
    
    # Classification metrics
    'Accuracy', 'StrictAccuracy', 'BinaryAccuracy', 'StrictBinaryAccuracy',
    'ThresholdAccuracy', 'StrictThresholdAccuracy', 'BalancedAccuracy', 'GMean', 
    'F1Score', 'FBetaScore', 'MatthewsCorrelation', 'CohensKappa',
    
    # Probabilistic metrics
    'BinaryCrossEntropy', 'GaussianLogLikelihood', 'HingeLoss', 'BrierScore',
    
    # Advanced metrics
    'AdversarialPerturbationSensitivity', 'ConditionalValueAtRisk',
    
    # Clustering metrics
    'SilhouetteScore',
    
    # Distance functions
    'DistanceFunction', 'EuclideanDistance', 'ManhattanDistance', 
    'ChebyshevDistance', 'CosineDistance'
]
