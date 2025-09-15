"""
Distance functions for clustering and similarity calculations.

This module provides various distance metrics that can be used in clustering
and other similarity-based algorithms.
"""

import ctypes
from typing import Any

from ..base import liblgp, POINTER
from ..genetics import LGPInput


class DistanceFunction:
    """Base class for distance function wrappers."""
    
    def __init__(self, c_function_name: str):
        try:
            self.c_function = getattr(liblgp, c_function_name)
            self.c_function.argtypes = [POINTER(LGPInput)]  # LGPInput pointer
            self.c_function.restype = ctypes.POINTER(ctypes.c_double)
        except AttributeError:
            raise ValueError(f"C function '{c_function_name}' not found in liblgp library")


class EuclideanDistance(DistanceFunction):
    """
    Euclidean distance function.
    
    Computes the L2 norm (straight-line distance) between points.
    Most common distance metric for clustering.
    
    Formula: sqrt(sum((x_i - y_i)²))
    """
    
    def __init__(self):
        super().__init__("euclidean_distances")


class ManhattanDistance(DistanceFunction):
    """
    Manhattan distance function.
    
    Computes the L1 norm (city block distance) between points.
    Sum of absolute differences of coordinates.
    
    Formula: sum(|x_i - y_i|)
    """
    
    def __init__(self):
        super().__init__("manhattan_distances")


class ChebyshevDistance(DistanceFunction):
    """
    Chebyshev distance function.
    
    Computes the L∞ norm (maximum coordinate difference) between points.
    Also known as chessboard distance.
    
    Formula: max(|x_i - y_i|)
    """
    
    def __init__(self):
        super().__init__("chebyshev_distances")


class CosineDistance(DistanceFunction):
    """
    Cosine distance function.
    
    Computes 1 - cosine_similarity between points.
    Measures the angle between vectors, independent of magnitude.
    
    Formula: 1 - (dot(x,y) / (||x|| * ||y||))
    """
    
    def __init__(self):
        super().__init__("cosine_distances")


# Convenience factory functions
def euclidean() -> EuclideanDistance:
    """Create an Euclidean distance function."""
    return EuclideanDistance()

def manhattan() -> ManhattanDistance:
    """Create a Manhattan distance function."""
    return ManhattanDistance()

def chebyshev() -> ChebyshevDistance:
    """Create a Chebyshev distance function."""
    return ChebyshevDistance()

def cosine() -> CosineDistance:
    """Create a Cosine distance function."""
    return CosineDistance()


# List of all available distance functions
__all__ = [
    # Classes
    'DistanceFunction', 'EuclideanDistance', 'ManhattanDistance', 
    'ChebyshevDistance', 'CosineDistance',
    # Factory functions
    'euclidean', 'manhattan', 'chebyshev', 'cosine'
]
