#!/usr/bin/env python3
"""
Clustering Examples with Linear Genetic Programming (LGP)

This file demonstrates the use of LGP for clustering problems using the
newly implemented clustering fitness functions. Examples include:
1. Simple 2D Clustering with Silhouette Score
2. Multi-dimensional Clustering with Inertia
3. Advanced Clustering with multiple metrics
"""

import lgp
import numpy as np
import pandas as pd
import time

def setup_clustering_lgp():
    """Initial LGP system setup for clustering"""
    print("🚀 Initializing LGP system for clustering...")
    
    # Initialize with custom seed for reproducible examples
    lgp.random_init_all(42)
    print(f"✓ Available OpenMP threads: {lgp.NUMBER_OF_OMP_THREADS}")
    print(f"✓ LGP system ready for clustering evolution")
    print(f"✓ Available VM operations: {len([op for op in lgp.Operation])}")
    print(f"✓ PRNG initialized with seed 42 for reproducible results")
    print()


def example_simple_2d_clustering():
    """
    Example 1: Simple 2D Clustering with Silhouette Score
    Objective: Evolve a program that assigns cluster labels based on 2D coordinates
    """
    print("=" * 60)
    print("🎯 EXAMPLE 1: SIMPLE 2D CLUSTERING")
    print("=" * 60)
    print("Objective: Evolve clustering based on 2D coordinates using Silhouette Score")
    print()
    
    # 1. Generate synthetic 2D clustering dataset (without sklearn)
    print("📊 Generating 2D clustering dataset...")
    np.random.seed(123)
    n_samples_per_cluster = 50
    n_clusters = 3
    
    # Create 3 well-separated blob clusters manually
    cluster1 = np.random.normal([2.0, 2.0], 0.8, (n_samples_per_cluster, 2))
    cluster2 = np.random.normal([6.0, 6.0], 0.8, (n_samples_per_cluster, 2))
    cluster3 = np.random.normal([2.0, 6.0], 0.8, (n_samples_per_cluster, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    y_true = np.array([0]*n_samples_per_cluster + [1]*n_samples_per_cluster + [2]*n_samples_per_cluster)
    
    n_samples = X.shape[0]
    
    print(f"✓ Dataset: {n_samples} samples, {n_clusters} true clusters")
    print(f"✓ Feature dimensions: {X.shape[1]} (2D coordinates)")
    print(f"✓ X range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"✓ Y range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    print()
    
    # 2. Creating instruction set optimized for clustering
    print("🔧 Configuring instruction set for clustering...")
    clustering_operations = [
        # Basic arithmetic for distance calculations
        lgp.Operation.ADD_F, lgp.Operation.SUB_F, 
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        # Advanced functions for distance/similarity
        lgp.Operation.POW, lgp.Operation.SQRT,
        # Memory operations
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F,
        # Comparison and conditional operations
        lgp.Operation.CMP_F, lgp.Operation.TEST_F,
        lgp.Operation.CMOV_G_F, lgp.Operation.CMOV_L_F,
        # Mathematical functions
        lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.EXP
    ]
    instruction_set = lgp.InstructionSet(clustering_operations)
    print(f"✓ Instruction set: {instruction_set.size} operations")
    print()
    
    # 3. Create LGP input (no target labels needed for clustering)
    print("🎯 Creating LGP input for clustering...")
    # For clustering, we don't provide target labels
    dummy_y = np.zeros(n_samples)  # Placeholder target
    lgp_input = lgp.LGPInput.from_numpy(X, dummy_y, instruction_set, ram_size=15)
    print(f"✓ Input: {lgp_input.input_num} samples")
    print(f"✓ ROM size: {lgp_input.rom_size} (features)")
    print(f"✓ RAM size: {lgp_input.ram_size}")
    print()
    
    # 4. Evolution with Silhouette Score fitness
    print("🧬 Starting evolution with Silhouette Score...")
    print("Parameters: pop_size=100, generations=50, tournament_size=3")
    print()
    
    start_time = time.time()
    
    try:
        population, evaluations, generations, best_idx = lgp.evolve(
            lgp_input,
            fitness=lgp.fitness.clustering.SilhouetteScore(num_clusters=n_clusters),
            selection=lgp.Tournament(3),
            initialization=lgp.UniquePopulation(100, 6, 25),  # pop_size, min_len, max_len
            target=0.8,  # Terminate if Silhouette Score > 0.8
            mutation_prob=0.7,
            crossover_prob=0.9,
            max_clock=5000,
            max_individ_len=30,
            generations=50,
            verbose=1
        )
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        return None, None
    
    elapsed_time = time.time() - start_time
    
    # 5. Results analysis
    print()
    print("📈 CLUSTERING EVOLUTION RESULTS:")
    print("-" * 40)
    
    best_individual = population.get(best_idx)
    
    print(f"✓ Evolution completed in {elapsed_time:.2f} seconds")
    print(f"✓ Generations executed: {generations}")
    print(f"✓ Total evaluations: {evaluations:,}")
    print(f"✓ Evaluations/second: {evaluations/elapsed_time:.0f}")
    print()
    
    print(f"🏆 BEST CLUSTERING SOLUTION:")
    print(f"   Silhouette Score: {best_individual.fitness:.6f}")
    print(f"   (Higher is better, max = 1.0)")
    
    print()
    print("📝 EVOLVED CLUSTERING PROGRAM:")
    print("-" * 40)
    lgp.print_program(best_individual)
    print()
    
    return best_individual, lgp_input, X, y_true


def example_multidimensional_clustering():
    """
    Example 2: Multi-dimensional Clustering with Inertia
    Objective: Cluster high-dimensional data using Within-Cluster Sum of Squares
    """
    print("=" * 60)
    print("🌐 EXAMPLE 2: MULTI-DIMENSIONAL CLUSTERING")
    print("=" * 60)
    print("Objective: Cluster 4D data using Inertia (WCSS) minimization")
    print()
    
    # 1. Generate high-dimensional clustering dataset manually
    print("📊 Generating 4D clustering dataset...")
    np.random.seed(456)
    n_samples_per_cluster = 40
    n_features = 4
    n_clusters = 3
    
    # Create 3 clusters in 4D space
    cluster1 = np.random.normal([1.0, 1.0, 1.0, 1.0], 1.0, (n_samples_per_cluster, n_features))
    cluster2 = np.random.normal([5.0, 5.0, 5.0, 5.0], 1.2, (n_samples_per_cluster, n_features))
    cluster3 = np.random.normal([1.0, 5.0, 1.0, 5.0], 1.0, (n_samples_per_cluster, n_features))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    y_true = np.array([0]*n_samples_per_cluster + [1]*n_samples_per_cluster + [2]*n_samples_per_cluster)
    n_samples = X.shape[0]
    
    print(f"✓ Dataset: {n_samples} samples, {n_clusters} true clusters")
    print(f"✓ Feature dimensions: {n_features}D")
    print(f"✓ Feature ranges:")
    for i in range(n_features):
        print(f"    Feature {i+1}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    print()
    
    # 2. Create instruction set for high-dimensional clustering
    print("🔧 Configuring instruction set for multi-dimensional clustering...")
    advanced_ops = [
        # Core arithmetic
        lgp.Operation.ADD_F, lgp.Operation.SUB_F, 
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        # Advanced mathematical functions
        lgp.Operation.POW, lgp.Operation.SQRT,
        lgp.Operation.EXP, lgp.Operation.LOG,
        # Memory and control
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F,
        # Conditional operations for decision trees
        lgp.Operation.CMP_F, lgp.Operation.TEST_F,
        lgp.Operation.CMOV_G_F, lgp.Operation.CMOV_L_F,
        lgp.Operation.CMOV_GE_F, lgp.Operation.CMOV_LE_F,
        # Trigonometric functions for non-linear transformations
        lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.TAN
    ]
    instruction_set = lgp.InstructionSet(advanced_ops)
    print(f"✓ Instruction set: {instruction_set.size} operations")
    print()
    
    # 3. Create LGP input
    print("🎯 Creating LGP input for multi-dimensional clustering...")
    dummy_y = np.zeros(n_samples)
    lgp_input = lgp.LGPInput.from_numpy(X, dummy_y, instruction_set, ram_size=20)
    print(f"✓ Input: {lgp_input.input_num} samples")
    print(f"✓ ROM size: {lgp_input.rom_size}")
    print(f"✓ RAM size: {lgp_input.ram_size}")
    print()
    
    # 4. Evolution with Inertia fitness (minimize WCSS)
    print("🧬 Starting evolution with Inertia minimization...")
    print("Parameters: pop_size=120, generations=60, tournament_size=4")
    print()
    
    start_time = time.time()
    
    try:
        population, evaluations, generations, best_idx = lgp.evolve(
            lgp_input,
            fitness=lgp.fitness.clustering.Inertia(num_clusters=n_clusters),
            selection=lgp.Tournament(4),
            initialization=lgp.UniquePopulation(120, 8, 35),
            target=0.1,  # Terminate if Inertia < 0.1 (lower is better)
            mutation_prob=0.75,
            crossover_prob=0.95,
            max_clock=6000,
            max_individ_len=40,
            generations=60,
            verbose=1
        )
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        return None, None
    
    elapsed_time = time.time() - start_time
    
    # 5. Results analysis
    print()
    print("📈 MULTI-DIMENSIONAL CLUSTERING RESULTS:")
    print("-" * 45)
    
    best_individual = population.get(best_idx)
    
    print(f"✓ Evolution completed in {elapsed_time:.2f} seconds")
    print(f"✓ Generations executed: {generations}")
    print(f"✓ Total evaluations: {evaluations:,}")
    print()
    
    print(f"🏆 BEST CLUSTERING SOLUTION:")
    print(f"   Inertia (WCSS): {best_individual.fitness:.6f}")
    print(f"   (Lower is better - tighter clusters)")
    
    print()
    print("📝 EVOLVED PROGRAM:")
    print("-" * 40)
    lgp.print_program(best_individual)
    print()
    
    return best_individual, lgp_input, X, y_true


def example_advanced_clustering_comparison():
    """
    Example 3: Advanced Clustering with Multiple Metrics
    Compare different clustering fitness functions on the same dataset
    """
    print("=" * 60)
    print("⚖️  EXAMPLE 3: CLUSTERING METRICS COMPARISON")
    print("=" * 60)
    print("Objective: Compare different clustering metrics on complex dataset")
    print()
    
    # 1. Generate challenging clustering dataset (manually create circles-like data)
    print("📊 Generating challenging overlapping clusters dataset...")
    np.random.seed(789)
    n_samples_per_cluster = 60
    
    # Inner circle (cluster 0)
    angles = np.linspace(0, 2*np.pi, n_samples_per_cluster)
    radius1 = 2.0 + 0.5 * np.random.normal(0, 0.2, n_samples_per_cluster)
    inner_x = radius1 * np.cos(angles) + np.random.normal(0, 0.2, n_samples_per_cluster)
    inner_y = radius1 * np.sin(angles) + np.random.normal(0, 0.2, n_samples_per_cluster)
    
    # Outer circle (cluster 1)  
    radius2 = 5.0 + 0.7 * np.random.normal(0, 0.3, n_samples_per_cluster)
    outer_x = radius2 * np.cos(angles) + np.random.normal(0, 0.3, n_samples_per_cluster)
    outer_y = radius2 * np.sin(angles) + np.random.normal(0, 0.3, n_samples_per_cluster)
    
    X = np.vstack([
        np.column_stack([inner_x, inner_y]),
        np.column_stack([outer_x, outer_y])
    ])
    y_true = np.array([0]*n_samples_per_cluster + [1]*n_samples_per_cluster)
    n_samples = X.shape[0]
    
    print(f"✓ Dataset: {n_samples} samples (concentric circles)")
    print(f"✓ True clusters: 2 (inner and outer circle)")
    print(f"✓ Challenge: Non-linearly separable")
    print()
    
    # 2. Configure instruction set for non-linear clustering
    print("🔧 Configuring advanced instruction set...")
    nonlinear_ops = [
        # Standard arithmetic
        lgp.Operation.ADD_F, lgp.Operation.SUB_F, 
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        # Non-linear functions crucial for circles
        lgp.Operation.POW, lgp.Operation.SQRT,
        lgp.Operation.EXP, lgp.Operation.LOG,
        # Trigonometric for circular patterns
        lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.TAN,
        lgp.Operation.ASIN, lgp.Operation.ACOS, lgp.Operation.ATAN,
        # Memory operations
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F,
        # Advanced conditionals
        lgp.Operation.CMP_F, lgp.Operation.TEST_F,
        lgp.Operation.CMOV_G_F, lgp.Operation.CMOV_L_F,
        lgp.Operation.CMOV_GE_F, lgp.Operation.CMOV_LE_F
    ]
    instruction_set = lgp.InstructionSet(nonlinear_ops)
    print(f"✓ Instruction set: {instruction_set.size} operations")
    print()
    
    # 3. Create LGP input
    print("🎯 Creating LGP input...")
    dummy_y = np.zeros(n_samples)
    lgp_input = lgp.LGPInput.from_numpy(X, dummy_y, instruction_set, ram_size=18)
    print(f"✓ Input prepared for {lgp_input.input_num} samples")
    print()
    
    # 4. Test multiple clustering metrics
    clustering_metrics = [
        ("Silhouette Score", lgp.fitness.clustering.SilhouetteScore(num_clusters=2)),
        ("Calinski-Harabasz Index", lgp.fitness.clustering.CalinskiHarabaszIndex(num_clusters=2)),
        ("Davies-Bouldin Index", lgp.fitness.clustering.DaviesBouldinIndex(num_clusters=2))
    ]
    
    results = {}
    
    for metric_name, fitness_func in clustering_metrics:
        print(f"🧬 Evolving with {metric_name}...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            population, evaluations, generations, best_idx = lgp.evolve(
                lgp_input,
                fitness=fitness_func,
                selection=lgp.Tournament(3),
                initialization=lgp.UniquePopulation(80, 6, 28),
                target=0.7 if "Silhouette" in metric_name else None,
                mutation_prob=0.8,
                crossover_prob=0.9,
                max_clock=4000,
                max_individ_len=35,
                generations=40,
                verbose=0  # Reduced verbosity for comparison
            )
            
            elapsed_time = time.time() - start_time
            best_individual = population.get(best_idx)
            
            results[metric_name] = {
                'fitness': best_individual.fitness,
                'generations': generations,
                'evaluations': evaluations,
                'time': elapsed_time,
                'program': best_individual
            }
            
            print(f"✓ {metric_name}: {best_individual.fitness:.6f}")
            print(f"  Time: {elapsed_time:.2f}s, Generations: {generations}")
            print()
            
        except Exception as e:
            print(f"❌ Error with {metric_name}: {e}")
            print()
    
    # 5. Compare results
    print("📊 CLUSTERING METRICS COMPARISON:")
    print("=" * 50)
    
    for metric_name, result in results.items():
        print(f"{metric_name}:")
        print(f"  📈 Fitness: {result['fitness']:.6f}")
        print(f"  ⏱️  Time: {result['time']:.2f} seconds")
        print(f"  🔄 Generations: {result['generations']}")
        print(f"  🧮 Evaluations: {result['evaluations']:,}")
        print(f"  ⚡ Evals/sec: {result['evaluations']/result['time']:.0f}")
        print()
    
    # Find best performer
    if results:
        best_metric = max(results.keys(), key=lambda k: results[k]['fitness'])
        print(f"🏆 Best performing metric: {best_metric}")
        print(f"   Final fitness: {results[best_metric]['fitness']:.6f}")
        print()
        
        print("📝 BEST EVOLVED PROGRAM:")
        print("-" * 40)
        lgp.print_program(results[best_metric]['program'])
    
    return results, lgp_input, X, y_true


def main():
    """Run all clustering examples"""
    print("🎯 LINEAR GENETIC PROGRAMMING - CLUSTERING EXAMPLES")
    print("=" * 60)
    print("Demonstrating LGP clustering capabilities with different metrics")
    print()
    
    # Setup
    setup_clustering_lgp()
    
    # Run examples
    try:
        # Example 1: Simple 2D clustering
        result1 = example_simple_2d_clustering()
        
        # Example 2: Multi-dimensional clustering  
        result2 = example_multidimensional_clustering()
        
        # Example 3: Multiple metrics comparison
        result3 = example_advanced_clustering_comparison()
        
        print()
        print("🎉 All clustering examples completed successfully!")
        print("✓ 2D clustering with Silhouette Score")
        print("✓ Multi-dimensional clustering with Inertia")
        print("✓ Advanced clustering metrics comparison")
        print()
        print("💡 These examples demonstrate the versatility of LGP for clustering tasks")
        print("💡 Different metrics suit different clustering challenges")
        print("💡 Non-linear instruction sets help with complex cluster shapes")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
    print('\nEvoluzione completata! Il sistema di clustering LGP è funzionante.')

if __name__ == '__main__':
    main()
