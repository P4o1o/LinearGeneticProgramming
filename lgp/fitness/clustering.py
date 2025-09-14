"""
Clustering fitness functions for unsupervised learning evaluation.

This module provides fitness functions specifically designed for clustering problems.
These functions expect the program to output integer cluster assignments for each input.
For fuzzy clustering, outputs should be float values in [0, 1] range.
"""

import ctypes
from typing import Optional, List, Any

from .interface import Fitness, FitnessType, FitnessParams, FitnessFunction
from ..base import liblgp


class SilhouetteScore(Fitness):
    """
    Silhouette Score fitness function.
    
    Evaluates clustering quality by measuring how similar an object is to its own cluster
    compared to other clusters. Values range from -1 to 1, with higher values indicating
    better clustering.
    
    Expected program output: Integer cluster ID for each input
    """
    
    def __init__(self, num_clusters: int, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "SILHOUETTE_SCORE"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


class Inertia(Fitness):
    """
    Inertia (Within-cluster Sum of Squares) fitness function.
    
    Measures the sum of squared distances from each point to its cluster centroid.
    Lower values indicate tighter clusters.
    
    Expected program output: Integer cluster ID for each input
    """
    
    def __init__(self, num_clusters: int, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "INERTIA"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


class AdjustedRandIndex(Fitness):
    """
    Adjusted Rand Index fitness function.
    
    Measures the similarity between two clustering assignments.
    Requires ground truth cluster labels for comparison.
    Values range from -1 to 1, with 1 indicating perfect clustering.
    
    Expected program output: Integer cluster ID for each input
    """
    
    def __init__(self, num_clusters: int, true_labels: List[int], start: int = 0, end: int = 0):
        # For now, just use num_clusters - true_labels would need special handling in C
        super().__init__(FitnessFunction.in_dll(liblgp, "ADJUSTED_RAND_INDEX"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


class CalinskiHarabaszIndex(Fitness):
    """
    Calinski-Harabasz Index fitness function.
    
    Evaluates clustering quality as the ratio of between-cluster dispersion
    to within-cluster dispersion. Higher values indicate better clustering.
    
    Expected program output: Integer cluster ID for each input
    """
    
    def __init__(self, num_clusters: int, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "CALINSKI_HARABASZ_INDEX"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


class DaviesBouldinIndex(Fitness):
    """
    Davies-Bouldin Index fitness function.
    
    Measures the average similarity between clusters, where similarity is the ratio
    of within-cluster distances to between-cluster distances.
    Lower values indicate better clustering.
    
    Expected program output: Integer cluster ID for each input
    """
    
    def __init__(self, num_clusters: int, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "DAVIES_BOULDIN_INDEX"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


class DunnIndex(Fitness):
    """
    Dunn Index fitness function.
    
    Measures clustering quality as the ratio of minimum inter-cluster distance
    to maximum intra-cluster distance. Higher values indicate better clustering.
    
    Expected program output: Integer cluster ID for each input
    """
    
    def __init__(self, num_clusters: int, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "DUNN_INDEX"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


class FuzzyPartitionCoefficient(Fitness):
    """
    Fuzzy Partition Coefficient fitness function.
    
    Measures the amount of "fuzziness" in a fuzzy clustering.
    Values range from 0 to 1, with values closer to 1 indicating crisp clustering.
    
    Expected program output: Float membership values in [0, 1] for each cluster
    """
    
    def __init__(self, num_clusters: int, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "FUZZY_PARTITION_COEFFICIENT"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


class FuzzyPartitionEntropy(Fitness):
    """
    Fuzzy Partition Entropy fitness function.
    
    Measures the fuzziness of a clustering by computing the entropy of
    membership values. Lower values indicate better clustering.
    
    Expected program output: Float membership values in [0, 1] for each cluster
    """
    
    def __init__(self, num_clusters: int, start: int = 0, end: int = 0):
        super().__init__(FitnessFunction.in_dll(liblgp, "FUZZY_PARTITION_ENTROPY"), 
                        FitnessParams.new_clustering(num_clusters, start, end))


# Convenience factory functions (maintain backward compatibility)
def silhouette_score(num_clusters: int, start: int = 0, end: int = 0) -> SilhouetteScore:
    """Create a SilhouetteScore fitness function."""
    return SilhouetteScore(num_clusters, start, end)

def inertia(num_clusters: int, start: int = 0, end: int = 0) -> Inertia:
    """Create an Inertia fitness function."""
    return Inertia(num_clusters, start, end)

def adjusted_rand_index(num_clusters: int, true_labels: List[int], start: int = 0, end: int = 0) -> AdjustedRandIndex:
    """Create an AdjustedRandIndex fitness function."""
    return AdjustedRandIndex(num_clusters, true_labels, start, end)

def calinski_harabasz_index(num_clusters: int, start: int = 0, end: int = 0) -> CalinskiHarabaszIndex:
    """Create a CalinskiHarabaszIndex fitness function."""
    return CalinskiHarabaszIndex(num_clusters, start, end)

def davies_bouldin_index(num_clusters: int, start: int = 0, end: int = 0) -> DaviesBouldinIndex:
    """Create a DaviesBouldinIndex fitness function."""
    return DaviesBouldinIndex(num_clusters, start, end)

def dunn_index(num_clusters: int, start: int = 0, end: int = 0) -> DunnIndex:
    """Create a DunnIndex fitness function."""
    return DunnIndex(num_clusters, start, end)

def fuzzy_partition_coefficient(num_clusters: int, start: int = 0, end: int = 0) -> FuzzyPartitionCoefficient:
    """Create a FuzzyPartitionCoefficient fitness function."""
    return FuzzyPartitionCoefficient(num_clusters, start, end)

def fuzzy_partition_entropy(num_clusters: int, start: int = 0, end: int = 0) -> FuzzyPartitionEntropy:
    """Create a FuzzyPartitionEntropy fitness function."""
    return FuzzyPartitionEntropy(num_clusters, start, end)


# List of all available clustering fitness functions
__all__ = [
    # Classes
    'SilhouetteScore', 'Inertia', 'AdjustedRandIndex', 'CalinskiHarabaszIndex',
    'DaviesBouldinIndex', 'DunnIndex', 'FuzzyPartitionCoefficient', 'FuzzyPartitionEntropy',
    # Factory functions
    'silhouette_score', 'inertia', 'adjusted_rand_index', 'calinski_harabasz_index',
    'davies_bouldin_index', 'dunn_index', 'fuzzy_partition_coefficient', 'fuzzy_partition_entropy'
]
