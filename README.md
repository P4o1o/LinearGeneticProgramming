# Linear Genetic Programming (LGP) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue.svg)](https://github.com/P4o1o/LinearGeneticProgramming)
[![Performance](https://img.shields.io/badge/performance-30k%2B%20evals%2Fsec-green.svg)](https://github.com/P4o1o/LinearGeneticProgramming)

A high-performance, cross-platform Linear Genetic Programming framework designed for symbolic regression, classification, and complex optimization problems. This framework combines an ultra-optimized C core with a comprehensive Python interface, delivering maximum performance while maintaining ease of use.

## 🚀 Overview

Linear Genetic Programming is an evolutionary computation technique that evolves imperative programs (sequences of instructions) rather than tree-based structures. This framework provides:

- **Dual Interface Architecture**: Ultra-fast C core with comprehensive Python wrapper
- **Cross-Platform Compatibility**: Windows, macOS, Linux, and FreeBSD support  
- **Automatic Optimization**: Vector instructions (SSE2/3/4, AVX, AVX2, AVX-512, ARM NEON) and OpenMP parallelization
- **Comprehensive Fitness Functions**: 30+ fitness functions for regression and classification with vectorial output support
- **Advanced VM Architecture**: Custom instruction set with 87 operations, supporting both integer and floating-point computation
- **Thread-Safe Design**: Full OpenMP parallelization with configurable thread count
- **Memory Optimization**: SIMD-aligned memory allocation and efficient garbage collection

## 🏗️ Architecture

### C Core (`src/`)
The C implementation provides maximum performance with minimal overhead:
- **Zero-overhead abstractions** - direct memory access optimized for performance-critical applications
- **SIMD acceleration** - automatically detects and uses SSE2/3/4, AVX, AVX2, AVX-512, or ARM NEON when available
- **OpenMP parallelization** - scales across multiple CPU cores with configurable thread count (default: 16)
- **Memory efficiency** - SIMD-aligned memory allocation with automatic padding
- **Thread-safe design** - concurrent fitness evaluation and thread-local random number generation

### Python Interface (`lgp/`)
The Python wrapper adds safety and convenience while maintaining performance:
- **Comprehensive input validation** - type checking, bounds validation, and error handling
- **NumPy/Pandas integration** - seamless data handling with `from_numpy()` and `from_df()` methods
- **Object-oriented design** - high-level abstractions that map directly to C structures
- **Automatic memory management** - proper cleanup and garbage collection
- **Scientific computing integration** - native support for vectorial operations and multi-output problems

### Virtual Machine Architecture
The LGP VM executes evolved programs with:
- **87 specialized instructions** covering arithmetic, logic, control flow, and mathematical functions
- **Dual-type register system** - 4 integer registers (64-bit) and 4 floating-point registers (double precision)
- **Flexible memory model** - ROM (read-only problem data) and RAM (read-write working memory)
- **Advanced control flow** - conditional jumps, conditional moves, and flag-based operations
- **Numerical safety** - robust handling of NaN, infinity, and division by zero

## 📊 Fitness Functions

The framework includes 30+ specialized fitness functions designed for different problem types with vectorial output support:

### Regression Problems (Floating-Point Output)
These fitness functions analyze **vectorial floating-point results** from `vm.ram[params->start]` to `vm.ram[params->end-1]`:

**Error-Based Metrics (MINIMIZE):**
- **MSE/RMSE**: Mean Squared/Root Mean Squared Error
- **MAE**: Mean Absolute Error  
- **MAPE/Symmetric MAPE**: Mean Absolute Percentage Error variants
- **Huber Loss**: Robust loss function with configurable delta parameter
- **Logcosh**: Logarithm of hyperbolic cosine for smooth gradients
- **Worst Case Error**: Maximum error across all samples
- **Pinball Loss**: Quantile regression loss with configurable quantile
- **Binary Cross Entropy**: Cross-entropy loss for probabilistic outputs
- **Gaussian Log Likelihood**: Maximum likelihood estimation with configurable sigma
- **Brier Score**: Probabilistic forecasting accuracy
- **Hinge Loss**: Support Vector Machine loss function

**Correlation-Based Metrics (MAXIMIZE):**
- **R-Squared**: Coefficient of determination (vectorial version)
- **Pearson Correlation**: Statistical correlation measure (vectorial average)

**Regularized Variants:**
- **Length Penalized MSE**: MSE + α × program_length (encourages smaller programs)
- **Clock Penalized MSE**: MSE + α × execution_time (encourages faster programs)

### Classification Problems (Integer/Boolean Output)
These fitness functions interpret the **sign bit** of vectorial integer results from `vm.ram[params->start]` to `vm.ram[params->end-1]` (negative = false, positive = true):

**Accuracy Metrics (MAXIMIZE):**
- **Accuracy**: Per-label classification accuracy (supports multi-label)
- **Strict Accuracy**: Exact match for entire output vector per sample
- **Binary Accuracy**: Optimized for binary classification problems
- **Threshold Accuracy**: Tolerance-based accuracy with configurable threshold
- **Balanced Accuracy**: Average of sensitivity and specificity (handles class imbalance)

**Advanced Classification Metrics (MAXIMIZE):**
- **F1-Score**: Harmonic mean of precision and recall (multi-label support)
- **F-Beta Score**: Generalized F-score with configurable beta parameter
- **Matthews Correlation**: Balanced metric considering all confusion matrix elements
- **Cohen's Kappa**: Inter-rater agreement statistic accounting for chance agreement
- **G-Mean**: Geometric mean of sensitivity and specificity

### Specialized Functions
- **Adversarial Perturbation Sensitivity**: Robustness measure using perturbation vectors
- **Conditional Value at Risk**: Risk management metric with configurable alpha parameter

### Fitness Function Data Types
The framework provides **four distinct data type categories** for different problem types:

**[FLOAT]** - **Arbitrary floating-point values**: Regression problems expecting any real numbers (MSE, RMSE, MAE, R², Pearson correlation, etc.)

**[INT]** - **Exact integer matching**: Discrete classification with integer class labels (ACCURACY, BINARY_ACCURACY, etc.)

**[SIGN_BIT]** - **Binary classification via sign bit**: Uses sign of integer outputs where negative = class 0/false, positive = class 1/true (F1_SCORE, BALANCED_ACCURACY, MATTHEWS_CORRELATION, etc.)

**[PROB]** - **Probability values [0,1]**: Probabilistic classification expecting outputs between 0.0 and 1.0 (BINARY_CROSS_ENTROPY, BRIER_SCORE)

All fitness functions support:
- **Vectorial output evaluation** - analyze multiple outputs simultaneously
- **Configurable output ranges** - specify which RAM positions to evaluate
- **Robust error handling** - proper handling of NaN, infinity, and numerical edge cases
- **OpenMP parallelization** - concurrent evaluation across multiple samples

## 🎯 Key Features

### Evolutionary Algorithms
- **Multiple Selection Methods**: Tournament, elitism, roulette wheel, and fitness sharing variants
- **Advanced Genetic Operators**: Adaptive mutation and crossover with configurable probabilities
- **Population Initialization**: Random and unique population generation strategies
- **Diversity Preservation**: Fitness sharing mechanisms to maintain genetic diversity
- **Early Termination**: Automatic stopping when target fitness is reached

### Benchmark Problems (PSB2)
Five challenging benchmark problems from the Program Synthesis Benchmark Suite 2:
- **Vector Distance**: Euclidean distance calculation between n-dimensional vectors
- **Bouncing Balls**: Physics simulation predicting ball trajectory with gravity and collisions
- **Dice Game**: Probability calculation for optimal dice game strategies
- **Shopping List**: Budget optimization with item costs and discount calculations
- **Snow Day**: Weather prediction modeling snow accumulation and melting

### Performance Optimizations
- **SIMD Vectorization**: Automatic detection and use of available vector instruction sets
- **Memory Alignment**: SIMD-aligned memory allocation for optimal performance
- **Parallel Evaluation**: OpenMP-based concurrent fitness evaluation
- **Efficient Hash Tables**: xxHash-based program deduplication in unique population generation
- **Branch Prediction**: Optimized control flow for modern CPU architectures

### Specialized Functions
- **Threshold Accuracy**: Regression with tolerance threshold (vectorial)
- **Binary Cross Entropy**: Probabilistic classification loss (vectorial)
- **Gaussian Log Likelihood**: Maximum likelihood estimation (vectorial)
- **Hinge Loss**: Support Vector Machine loss function (vectorial)
- **Brier Score**: Probabilistic forecasting accuracy (vectorial)
- **Adversarial Perturbation Sensitivity**: Robustness measure with perturbation vectors
- **Conditional Value at Risk**: Risk management metric using worst α% of samples

## 🛠️ Building

### Prerequisites
- **C Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **Build System**: CMake 3.15+ or Make
- **Python** (optional): Python 3.8+ with NumPy for Python interface
- **OpenMP** (optional): For parallel fitness evaluation

### Quick Build
```bash
# Build optimized C executable
make

# Build Python shared library  
make python

# Clean build artifacts
make clean

# Show detailed build configuration
make info
```

### Build Configuration
```bash
# Configure OpenMP threads (default: 16)
make THREADS=8

# Enable debug mode with assertions (default: 0)
make DEBUG=1

# Specify C standard (default: auto-detect)
make C_STD=c11

# Specify compiler (default: auto-detect)
make CC=clang

# Disable vector optimizations
make VECTOR=0
```

### CMake Alternative
```bash
mkdir build && cd build
cmake .. -DTHREADS=8 -DDEBUG=ON -DC_STD=c11
make
```

### Platform-Specific Notes
- **Windows**: Use MinGW-w64 or Visual Studio with MSVC
- **macOS**: Requires Xcode Command Line Tools
- **Linux**: GCC or Clang with development headers
- **FreeBSD**: Uses system compiler, may require gmake

## 📚 Documentation

- **[C Core Documentation](src/README.md)** - Comprehensive guide to the C implementation, VM architecture, and performance optimization
- **[Python Interface Documentation](lgp/README.md)** - Complete Python API reference with examples and best practices

## � Contributing

We welcome contributions! Please see our contribution guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Performance benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚡ Performance Notes

- **Single-threaded performance**: Optimized for modern x86-64 and ARM architectures
- **Multi-threaded scaling**: Linear scaling up to available CPU cores
- **Memory usage**: Approximately 1-10 MB for typical problems (100-1000 individuals)
- **SIMD acceleration**: 2-4x speedup with AVX2/AVX-512 on supported hardware

### Python Interface (Recommended)
```python
import lgp
import numpy as np

# Generate sample regression data
X = np.random.uniform(-2, 2, (100, 1))
y = X[:, 0]**2 + 3*X[:, 0] + 1  # Target: x² + 3x + 1

# Create instruction set optimized for symbolic regression
instruction_set = lgp.InstructionSet([
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, 
    lgp.Operation.DIV_F, lgp.Operation.POW, lgp.Operation.SQRT,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.MOV_F
])

# Create LGP input from NumPy arrays
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=5)

# Execute evolution with optimized parameters
population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input,
    fitness=lgp.MSE(),                              # Mean Squared Error
    selection=lgp.Tournament(tournament_size=4),    # Tournament selection
    initialization=lgp.UniquePopulation(            # Unique population
        pop_size=200, minsize=5, maxsize=25
    ),
    target=1e-6,                                    # Stop when MSE < 1e-6
    mutation_prob=0.8,                              # 80% mutation probability
    crossover_prob=0.95,                            # 95% crossover probability
    max_clock=5000,                                 # VM execution limit
    generations=50,                                 # Maximum generations
    verbose=1                                       # Print progress
)

# Analyze results
best_individual = population.get(best_idx)
print(f"Best fitness: {best_individual.fitness:.6e}")
print(f"Program size: {best_individual.size} instructions")
print(f"Evaluations performed: {evaluations}")
print(f"Generations executed: {generations}")

# Display evolved program
print("\nEvolved program:")
best_individual.print_program()
```

### C Interface (Maximum Performance)
```c
#include "evolution.h"
#include "psb2.h"

int main() {
    // Initialize random number generation
    random_init_all(42);
    
    // Create instruction set
    struct Operation ops[] = {
        OP_ADD_F, OP_SUB_F, OP_MUL_F, OP_DIV_F, OP_POW,
        OP_LOAD_ROM_F, OP_STORE_RAM_F, OP_MOV_F
    };
    struct InstructionSet instr_set = {.size = 8, .op = ops};
    
    // Create problem instance (vector distance benchmark)
    struct LGPInput input = vector_distance(&instr_set, 2, 100);
    
    // Configure evolution parameters
    const struct LGPOptions options = {
        .fitness = MSE,
        .fitness_param = {.start = 0, .end = 1},
        .selection = tournament,
        .select_param = {.size = 3},
        .initialization_func = unique_population,
        .init_params = {
            .pop_size = 200,
            .minsize = 5,
            .maxsize = 25
        },
        .target = 1e-6,
        .mutation_prob = 0.8,
        .crossover_prob = 0.95,
        .max_clock = 5000,
        .max_individ_len = 50,
        .max_mutation_len = 5,
        .generations = 50,
        .verbose = 1
    };
    
    // Execute evolution
    struct LGPResult result = evolve(&input, &options);
    
    // Display best solution
    printf("Best fitness: %e\n", 
           result.pop.individual[result.best_individ].fitness);
    print_program(&result.pop.individual[result.best_individ].prog);
    
    // Cleanup
    free_population(&result.pop);
    free(input.memory);
    
    return 0;
}
```

## 🎯 Use Cases

- **Symbolic Regression**: Discover mathematical formulas from data
- **Boolean Function Learning**: Evolve logic circuits and decision rules  
- **Time Series Prediction**: Model temporal patterns and forecasting
- **Feature Engineering**: Automatic feature construction and selection
- **Control Systems**: Evolve controllers and decision-making logic
- **Data Mining**: Pattern discovery and rule extraction

## 🔧 Advanced Configuration

### Custom Instruction Sets
```python
# Create custom operation set
from lgp.vm import Operation
custom_ops = [
    Operation.ADD_F, Operation.SUB_F, Operation.MUL_F,
    Operation.JMP_Z, Operation.CMP, Operation.STORE_RAM_F
]
instruction_set = lgp.InstructionSet(custom_ops)
```

### Memory Configuration
```python
# Customize RAM size for complex problems
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=64)
```

### Parallel Execution
```python
# The framework automatically uses all available CPU cores
# Number of threads can be controlled via environment:
import os
os.environ['OMP_NUM_THREADS'] = '8'
```

## 📈 Performance

- **Vectorized Operations**: Automatic SIMD optimization (up to 8x speedup)
- **Memory Aligned**: Cache-friendly data structures  
- **OpenMP Scaling**: Near-linear speedup across CPU cores
- **JIT-like VM**: Efficient instruction execution engine
- **Zero-Copy Interface**: Minimal overhead between Python and C

## 🤝 Contributing

We welcome contributions! Please see:
- `src/README.md` for C core documentation
- `lgp/README.md` for Python interface documentation
- Examples in `examples.py` for usage patterns

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **Documentation**: See individual module README files
- **Examples**: Check `examples.py` for comprehensive usage examples
- **Issues**: Report bugs and request features via project issues
- **Build System**: Cross-platform Makefile and CMake support

---

