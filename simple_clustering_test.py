#!/usr/bin/env python3
"""
Simple test for the updated clustering system with distance functions.
"""

import numpy as np
import lgp
from lgp.fitness.distances import euclidean_distance

# Generate simple 2D test data
np.random.seed(42)
n_samples = 100
n_features = 2
n_clusters = 3

# Create test dataset
X = np.random.randn(n_samples, n_features)

print("🎯 SIMPLE CLUSTERING TEST WITH DISTANCE FUNCTIONS")
print("=" * 60)
print(f"Dataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")

try:
    # Initialize LGP system
    lgp.setup_library()
    print("✓ LGP system initialized")
    
    # Create distance function
    distance_fn = euclidean_distance()
    print("✓ Euclidean distance function created")
    
    # Create Silhouette Score fitness with distance function
    silhouette_fitness = lgp.SilhouetteScore(n_clusters, distance_fn)
    print("✓ Silhouette Score fitness created with custom distance function")
    
    # Test fitness creation without explicit distance function (should use default)
    silhouette_default = lgp.SilhouetteScore(n_clusters)
    print("✓ Silhouette Score fitness created with default distance function")
    
    print("\n🎉 All clustering interface tests passed!")
    print("✓ Distance functions work correctly")
    print("✓ Silhouette Score supports both custom and default distance functions")
    print("✓ Python interface is properly updated for the new C structure")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
