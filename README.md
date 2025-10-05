# Linear Genetic Programming (LGP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue.svg)](https://github.com/P4o1o/LinearGeneticProgramming)
[![Performance](https://img.shields.io/badge/performance-30k%2B%20evals%2Fsec-green.svg)](https://github.com/P4o1o/LinearGeneticProgramming)

High-performance Linear Genetic Programming framework for symbolic regression, classification, and optimization. Combines an optimized C core with a Python interface for maximum performance and usability.

## Features

- **Dual Architecture**: C core for performance + Python wrapper for ease of use
- **Cross-Platform**: Linux, macOS, Windows, FreeBSD
- **Optimized**: SIMD vectorization (SSE/AVX/AVX-512/NEON) and OpenMP parallelization
- **Comprehensive**: 40+ fitness functions including regression, classification, clustering, and advanced metrics
- **Advanced VM**: 101 specialized instructions with dual-type registers and vector operations
- **Scalable**: Thread-safe design with configurable parallelization

## Installation

### Native Setup

**Linux (Ubuntu/Debian)**:
```bash
sudo apt install build-essential cmake python3-dev libomp-dev
```

**macOS**:
```bash
brew install cmake python libomp
```

**Windows**:
```bash
# Install Visual Studio 2019+ with C++ build tools
# Install Python from python.org
```

**Clone and Build**:
```bash
git clone https://github.com/P4o1o/LinearGeneticProgramming.git
cd LinearGeneticProgramming

# Linux/macOS/Windows (MSYS2)
make clean && make THREADS=8    # Build C core
make python                     # Build Python library
./test.sh                       # Run tests

# Windows (Visual Studio)
mkdir build && cd build
cmake .. -DTHREADS=8
cmake --build . --config Release
```

### Docker/Podman
```bash
docker build -t lgp .
docker run -it --rm lgp

# Or with Podman
podman build -t lgp .
podman run -it --rm lgp
```

### Docker Compose
```bash
# Development
docker-compose --profile dev up lgp-dev

# Testing
docker-compose --profile test up lgp-test
```

## �️ Architecture

**C Core (`src/`)**: Ultra-optimized implementation with SIMD acceleration, OpenMP parallelization, and zero-overhead abstractions.

**Python Interface (`lgp/`)**: High-level wrapper with NumPy integration, input validation, and automatic memory management.

**Virtual Machine**: 101 specialized instructions, dual-type registers (4 int + 4 float + 8 vector), flexible memory model (ROM/RAM).

## 🎯 Key Features

- **Multiple Selection Methods**: Tournament, elitism, roulette wheel, fitness sharing
- **Vector Operations**: Native support for dynamic vector allocation and manipulation
- **Advanced VM Instructions**: 101 instructions including vector ops (NEWVEC_I, LOAD_VEC_RAM, LOAD_VEC_ROM, STORE_VEC_RAM)
- **Benchmark Problems (PSB2)**: Vector distance, bouncing balls, dice game, shopping list, snow day
- **SIMD Vectorization**: Automatic detection of SSE, AVX, AVX-512, ARM NEON
- **Memory Optimization**: SIMD-aligned allocation, efficient hash tables, branch prediction

## 🛠️ Build Options

```bash
# Quick build
make                    # Optimized build
make python            # Python library
make DEBUG=1           # Debug build
make THREADS=8         # Set thread count
make CC=clang          # Specify compiler

# CMake alternative
mkdir build && cd build
cmake .. -DTHREADS=8 -DDEBUG=ON
make
```

## 🎯 Quick Start

### Python Interface (Recommended)
```python
import lgp
import numpy as np

# Generate sample data
X = np.random.uniform(-2, 2, (100, 1))
y = X[:, 0]**2 + 3*X[:, 0] + 1  # Target: x² + 3x + 1

# Create instruction set with vector operations
instruction_set = lgp.InstructionSet([
    lgp.Operation.ADD_F, lgp.Operation.MUL_F, lgp.Operation.POW,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F,
    # New vector operations
    lgp.Operation.NEWVEC_I,      # Create new vector
    lgp.Operation.LOAD_VEC_RAM,  # Load vector from RAM
    lgp.Operation.LOAD_VEC_ROM,  # Load vector from ROM
    lgp.Operation.STORE_VEC_RAM  # Store vector to RAM
])

# Create LGP input and run evolution
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=5)
population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input,
    fitness=lgp.fitness.MSE(),  # New modular fitness organization
    selection=lgp.Tournament(4),
    initialization=lgp.UniquePopulation(150, 8, 30),  # pop_size, min_len, max_len
    target=0.05,  # Terminate if MSE < 0.05
    mutation_prob=0.8,
    crossover_prob=0.95,
    max_clock=8000,
    max_individ_len=20,
    generations=80,
    verbose=1
)

# Display results
best = population.get(best_idx)
print(f"Best fitness: {best.fitness:.6e}")
best.print_program()
```

### Vector Operations Example
```python
# Example using vector operations for batch processing
instruction_set = lgp.InstructionSet([
    lgp.Operation.NEWVEC_I,      # Create vectors with specified capacity
    lgp.Operation.LOAD_VEC_ROM,  # Load data vectors from ROM  
    lgp.Operation.ADD_F,         # Element-wise operations
    lgp.Operation.STORE_VEC_RAM  # Store results
])

# The VM now supports:
# - 8 vector registers (vreg[0] to vreg[7])
# - Dynamic vector allocation with SIMD-aligned memory
# - Efficient bulk data operations
# - Automatic memory management
```

### C Interface (Maximum Performance)
```c
#include "src/evolution.h"
#include "src/psb2.h"
#include "src/fitness/regression.h"

int main() {
    random_init_all(42);
    
    // Create instruction set and problem
    struct Operation ops[] = {OP_ADD_F, OP_MUL_F, OP_LOAD_ROM_F, OP_STORE_RAM_F};
    struct InstructionSet instr_set = {.size = 4, .op = ops};
    struct LGPInput input = vector_distance(&instr_set, 2, 100);
    
    // Configure and run evolution
    const struct LGPOptions options = {
        .fitness = MSE,
		.fitness_param = (struct FitnessParams) {
			.start = 0,
			.end = 1,
		},
		.selection = tournament,
		.select_param = (union SelectionParams) {.size = 3},
		.initialization_func = unique_population,
		.init_params = (struct InitializationParams) {
			.pop_size = 1000,
			.minsize = 2,
			.maxsize = 5
		},
		.target = 1e-27,
		.mutation_prob = 0.76,
		.max_mutation_len = 5,
		.crossover_prob = 0.95,
		.max_clock = 5000,
		.max_individ_len = 50,
		.generations = 10,
		.verbose = 1
    };
    
    struct LGPResult result = evolve(&input, &options);
    printf("Best fitness: %e\n", result.pop.individual[result.best_individ].fitness);
    return 0;
}
```

## 🎯 Fitness Functions (Modular Organization)

The framework provides **40+ fitness functions** organized in specialized modules:

### Regression
```python
# Basic metrics
lgp.fitness.MSE()           # Mean Squared Error
lgp.fitness.RMSE()          # Root Mean Squared Error  
lgp.fitness.MAE()           # Mean Absolute Error
lgp.fitness.RSquared()      # Coefficient of Determination

# Advanced metrics
lgp.fitness.MAPE()          # Mean Absolute Percentage Error
lgp.fitness.HuberLoss()     # Robust regression loss
lgp.fitness.PinballLoss()   # Quantile regression
```

### Classification
```python
lgp.fitness.Accuracy()          # Classification accuracy
lgp.fitness.F1Score()           # F1 score
lgp.fitness.BalancedAccuracy()  # Balanced accuracy
lgp.fitness.ThresholdAccuracy() # Accuracy with custom threshold
```

### Clustering (NEW)
```python
# Clustering quality metrics
lgp.fitness.SilhouetteScore()        # Silhouette analysis
lgp.fitness.CalinskiHarabaszIndex()  # Variance ratio criterion
lgp.fitness.DaviesBouldinIndex()     # Cluster separation measure
lgp.fitness.DunnIndex()              # Compactness/separation ratio

# Clustering partition metrics  
lgp.fitness.AdjustedRandIndex()      # Adjusted Rand Index
lgp.fitness.Inertia()                # Within-cluster sum of squares

# Fuzzy clustering metrics
lgp.fitness.FuzzyPartitionCoefficient()  # Partition coefficient
lgp.fitness.FuzzyPartitionEntropy()      # Partition entropy
```

### Advanced Metrics
```python
lgp.fitness.ConditionalValueAtRisk()  # Risk management
lgp.fitness.BinaryCrossEntropy()     # Probabilistic loss
lgp.fitness.GaussianLogLikelihood()  # Maximum likelihood
```

## 🚀 Use Cases

- **Symbolic Regression**: Discover mathematical formulas from data
- **Classification**: Evolve decision rules and logic circuits
- **Clustering**: Unsupervised pattern discovery and data partitioning  
- **Vector Processing**: Batch operations on time series and feature vectors
- **Time Series**: Model temporal patterns and forecasting
- **Feature Engineering**: Automatic feature construction
- **Control Systems**: Evolve controllers and decision logic

## 🆕 New Vector Operations

The latest version introduces powerful vector operations for enhanced data processing:

### Vector Instructions
- **NEWVEC_I**: Dynamically allocate vectors with SIMD-aligned memory
- **LOAD_VEC_RAM**: Efficiently load vector data from RAM
- **LOAD_VEC_ROM**: Load vector data from ROM for batch processing
- **STORE_VEC_RAM**: Store computed vectors back to memory

### Vector Registers
- **8 Vector Registers** (`vreg[0]` to `vreg[7]`) for complex operations
- **Dynamic Capacity**: Each vector can grow as needed
- **Memory Efficient**: SIMD-aligned allocation for performance

### Applications
```python
# Example: Time series processing
instruction_set = lgp.InstructionSet([
    lgp.Operation.NEWVEC_I,      # Create result vector
    lgp.Operation.LOAD_VEC_ROM,  # Load time series data
    lgp.Operation.ADD_F,         # Element-wise operations
    lgp.Operation.STORE_VEC_RAM  # Store processed results
])
```

Run the vector operations example:
```bash
python3 vector_operations_example.py
```

## ⚡ Performance

- **30k+ evaluations/sec** on modern hardware  
- **Enhanced vector processing** with SIMD-aligned memory operations
- **Linear scaling** across CPU cores with OpenMP
- **SIMD acceleration** with automatic instruction detection
- **Memory efficient** with SIMD-aligned allocation

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

