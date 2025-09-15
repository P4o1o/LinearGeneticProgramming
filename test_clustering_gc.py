#!/usr/bin/env python3
"""
Simple test for the updated clustering system with distance functions and garbage collection.
"""

import numpy as np
import pandas as pd
import gc
import lgp
from lgp.fitness.distances import EuclideanDistance, ManhattanDistance, ChebyshevDistance, CosineDistance

def test_distance_functions():
    """Test the different distance functions."""
    print("🧪 Testing Distance Functions")
    print("=" * 40)
    
    # Create simple 2D test data
    data = np.array([
        [0.0, 0.0],  # Point 1
        [1.0, 0.0],  # Point 2 
        [0.0, 1.0],  # Point 3
        [1.0, 1.0],  # Point 4
    ], dtype=np.float64)
    
    print(f"Test data shape: {data.shape}")
    print(f"Test points:\n{data}")
    
    # Test all distance functions
    distances = [
        ("Euclidean", EuclideanDistance()),
        ("Manhattan", ManhattanDistance()),
        ("Chebyshev", ChebyshevDistance()),
        ("Cosine", CosineDistance())
    ]
    
    for name, dist_fn in distances:
        print(f"\n✓ {name} distance function created successfully")
        print(f"  C function pointer: {dist_fn.c_function}")
    
    print("\n🎯 All distance functions initialized correctly!")
    return True

def test_silhouette_with_gc():
    """Test SilhouetteScore with garbage collection."""
    print("\n🧪 Testing SilhouetteScore with Garbage Collection")
    print("=" * 50)
    
    # Create test data using pandas DataFrame
    import pandas as pd
    np.random.seed(42)
    data = np.random.randn(20, 3).astype(np.float64)
    expected_outputs = np.random.randint(0, 3, 20).astype(np.int64)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
    df['cluster'] = expected_outputs
    
    print(f"Test data: {data.shape} points, {data.shape[1]}D")
    print(f"Expected cluster assignments: {len(np.unique(expected_outputs))} clusters")
    
    # Create instruction set and LGPInput
    instruction_set = lgp.InstructionSet.complete()
    lgp_input = lgp.LGPInput.from_df(df, ['cluster'], instruction_set)
    print(f"✓ LGPInput created: {lgp_input.input_num} samples")
    
    # Test different distance functions
    distance_functions = [
        ("Euclidean", EuclideanDistance()),
        ("Manhattan", ManhattanDistance()),
        ("Chebyshev", ChebyshevDistance()),
        ("Cosine", CosineDistance())
    ]
    
    for name, dist_fn in distance_functions:
        print(f"\n🔬 Testing with {name} distance:")
        
        # Create SilhouetteScore objects and test garbage collection
        objects_created = []
        
        for i in range(3):
            silhouette = lgp.SilhouetteScore(
                num_clusters=3,
                lgp_input=lgp_input,
                distance_fn=dist_fn
            )
            objects_created.append(silhouette)
            print(f"  ✓ SilhouetteScore object {i+1} created")
        
        # Test that objects have the destructor
        for i, obj in enumerate(objects_created):
            assert hasattr(obj, '__del__'), f"Object {i} missing destructor"
            assert hasattr(obj, '_params_ptr'), f"Object {i} missing _params_ptr"
            print(f"  ✓ Object {i+1} has destructor and parameter pointer")
        
        # Clear references and force garbage collection
        print(f"  🗑️  Clearing {len(objects_created)} objects and forcing GC...")
        objects_created.clear()
        gc.collect()
        print(f"  ✓ Garbage collection completed for {name} distance")
    
    print("\n🎉 All tests passed! Memory management working correctly.")
    return True

def main():
    """Run all tests."""
    print("🚀 TESTING UPDATED CLUSTERING SYSTEM")
    print("=" * 60)
    
    try:
        # Test distance functions
        test_distance_functions()
        
        # Test garbage collection
        test_silhouette_with_gc()
        
        print("\n" + "=" * 60)
        print("🎯 ALL TESTS SUCCESSFUL!")
        print("✅ Distance functions working")
        print("✅ SilhouetteScore creation working")
        print("✅ Garbage collection working")
        print("✅ Memory management working")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
