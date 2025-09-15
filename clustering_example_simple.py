#!/usr/bin/env python3
"""
Simple Clustering Example with LGP - Simplified Interface

This demonstrates a straightforward evolution using SilhouetteScore
with different distance functions.
"""

import lgp
from lgp.fitness.distances import EuclideanDistance, ManhattanDistance, ChebyshevDistance, CosineDistance
from lgp.fitness.clustering import SilhouetteScore
import numpy as np
import time


def create_test_data():
    """Create simple 2D test data with 3 clusters"""
    np.random.seed(42)
    
    # Create 3 well-separated clusters
    cluster1 = np.random.normal([2.0, 2.0], 0.5, (30, 2))
    cluster2 = np.random.normal([6.0, 6.0], 0.5, (30, 2))
    cluster3 = np.random.normal([2.0, 6.0], 0.5, (30, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    return X


def run_evolution_example():
    """Run a simple evolution example with SilhouetteScore"""
    print("🚀 LGP Clustering Evolution Example")
    print("=" * 50)
    
    # Initialize LGP
    lgp.random_init_all(42)
    print("✓ LGP initialized")
    
    # Create test data
    X = create_test_data()
    dummy_y = np.zeros(X.shape[0])
    print(f"✓ Test data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create instruction set
    instruction_set = lgp.InstructionSet.complete()
    
    # Create LGP input
    lgp_input = lgp.LGPInput.from_numpy(X, dummy_y, instruction_set, ram_size=15)
    print("✓ LGP input created")
    
    # Test different distance functions
    distances = [
        ("Euclidean", EuclideanDistance()),
        ("Manhattan", ManhattanDistance()),
        ("Chebyshev", ChebyshevDistance()),
        ("Cosine", CosineDistance())
    ]
    
    for dist_name, distance_fn in distances:
        print(f"\n🔬 Testing {dist_name} Distance:")
        
        try:
            # Create SilhouetteScore fitness function
            silhouette = SilhouetteScore(3, lgp_input, distance_fn)
            print(f"   ✓ SilhouetteScore created")
            
            # Run evolution
            start_time = time.time()
            population, evaluations, generations, best_idx = lgp.evolve(
                lgp_input,
                fitness=silhouette,
                selection=lgp.Tournament(3),
                initialization=lgp.UniquePopulation(50, 4, 15),  # Reduced population size
                mutation_prob=0.7,
                crossover_prob=0.8,
                max_clock=2000,
                max_individ_len=30,
                generations=50,  # Reduced generations
                verbose=1  # Show progress
            )
            
            elapsed = time.time() - start_time
            best_individual = population.get(best_idx)
            
            print(f"   ✓ Evolution completed!")
            print(f"   ✓ Best fitness: {best_individual.fitness:.6f}")
            print(f"   ✓ Generations: {generations}")
            print(f"   ✓ Evaluations: {evaluations:,}")
            print(f"   ✓ Time: {elapsed:.2f}s")
            print(f"   ✓ Performance: {evaluations/elapsed:.0f} evals/sec")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Clustering evolution examples completed!")


if __name__ == "__main__":
    run_evolution_example()
