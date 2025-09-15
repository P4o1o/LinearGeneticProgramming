"""
Clustering fitness functions for unsupervised learning evaluation.

This module provides fitness functions specifically designed for clustering problems.
These functions expect the program to output integer cluster assignments for each input.
"""

import ctypes

from .interface import Fitness, FitnessType, FitnessParams, FitnessFunction
from .distances import DistanceFunction, EuclideanDistance
from ..base import liblgp
from ..genetics import LGPInput

# Configure the C function for freeing distance tables
_free_distance_table = liblgp.free_distance_table
_free_distance_table.argtypes = [ctypes.POINTER(FitnessParams)]
_free_distance_table.restype = None


class SilhouetteScore(Fitness):
    """
    Silhouette Score fitness function.
    
    Evaluates clustering quality by measuring how similar an object is to its own cluster
    compared to other clusters. Values range from -1 to 1, with higher values indicating
    better clustering.
    
    Expected program output: Integer cluster ID for each input
    """
    
    def __init__(self, num_clusters: int, lgp_input: LGPInput, distance_fn: DistanceFunction):
        """
        Initialize Silhouette Score clustering fitness.
        
        Args:
            num_clusters: Number of clusters expected
            lgp_input: The input data structure containing the points to cluster
            distance_fn: Distance function to use (default: Euclidean)
        """

        # Store references for cleanup
        self.distance_fn = distance_fn
        self.lgp_input = lgp_input
        self._params_ptr = None
        
        # Create FitnessParams with clustering configuration
        params = FitnessParams.new_clustering(num_clusters, distance_fn, lgp_input)
        
        super().__init__(FitnessFunction.in_dll(liblgp, "SILHOUETTE_SCORE"), params)
    
    def __del__(self):
        """
        Destructor to free the distance table memory.
        
        This ensures that the pre-computed distance table allocated by the C code
        is properly freed when the Python object is garbage collected.
        """
        _params_ptr = ctypes.pointer(self._params)
        _free_distance_table(_params_ptr)


# List of all available clustering fitness functions
__all__ = [
    # Classes
    'SilhouetteScore',
]
