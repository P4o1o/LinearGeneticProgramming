# Linear Genetic Programming (LGP) Framework

A high-performance, cross-platform Linear Genetic Programming framework designed for symbolic regression, boolean function learning, and optimization problems. This framework combines a super-optimized C core with a user-friendly Python interface, delivering maximum performance while maintaining ease of use.

## 🚀 Overview

Linear Genetic Programming is an evolutionary computation technique that evolves imperative programs (sequences of instructions) rather than tree-based structures. This framework provides:

- **Dual Interface Architecture**: Ultra-fast C core with comprehensive Python wrapper
- **Cross-Platform Compatibility**: Windows, macOS, Linux, and FreeBSD support
- **Automatic Optimization**: Vector instructions (SSE, AVX, AVX-512) and OpenMP when available
- **Comprehensive Fitness Functions**: 20+ fitness functions for regression and classification
- **Flexible VM Architecture**: Custom instruction set with 87+ operations
- **Thread-Safe Design**: Full OpenMP parallelization support

## 🏗️ Architecture

### C Core (`src/`)
The C implementation provides maximum performance with minimal overhead:
- **No input validation** - assumes correct usage for speed
- **Direct memory access** - optimized for performance-critical applications
- **Vector instruction support** - automatically uses SSE2, AVX, AVX2, AVX-512 when available
- **OpenMP parallelization** - scales across multiple CPU cores

### Python Interface (`lgp/`)
The Python wrapper adds safety and convenience:
- **Input validation** - comprehensive error checking and type validation
- **NumPy/Pandas integration** - seamless data handling with `from_numpy()` and `from_df()`
- **User-friendly classes** - high-level abstractions for easy use
- **Memory management** - automatic cleanup and garbage collection

## 📊 Fitness Functions

The framework includes specialized fitness functions for different problem types:

### Regression Problems (Floating-Point Output)
These fitness functions analyze the **floating-point result** from `vm.ram[0].f64`:
- **MSE/RMSE**: Mean Squared/Root Mean Squared Error
- **MAE**: Mean Absolute Error  
- **MAPE/Symmetric MAPE**: Mean Absolute Percentage Error
- **Huber Loss**: Robust loss function
- **R-Squared**: Coefficient of determination
- **Logcosh**: Logarithm of hyperbolic cosine
- **Pearson Correlation**: Statistical correlation measure

### Classification Problems (Integer/Boolean Output)
These fitness functions interpret the **sign bit** of `vm.ram[0].i64` (negative = false, positive = true):
- **Accuracy**: Classification accuracy
- **F1-Score/F-Beta**: Harmonic mean of precision and recall
- **Matthews Correlation**: Balanced classification metric
- **Balanced Accuracy**: Accuracy corrected for class imbalance
- **Cohen's Kappa**: Inter-rater agreement statistic
- **G-Mean**: Geometric mean of sensitivity and specificity

### Specialized Functions
- **Threshold Accuracy**: Regression with tolerance threshold
- **Binary Cross Entropy**: Probabilistic classification loss
- **Gaussian Log Likelihood**: Maximum likelihood estimation
- **Adversarial Perturbation Sensitivity**: Robustness measure (requires NumPy)

## 🛠️ Building

### Prerequisites
- C compiler (GCC, Clang, or MSVC)
- CMake 3.15+ or Make
- Python 3.8+ (for Python interface)

### Quick Build
```bash
# Build optimized executable
make

# Build Python shared library  
make python

# Clean build artifacts
make clean

# Show build configuration
make info
```

### Build Variables
```bash
# Number of OpenMP threads (default: 16)
make THREADS=8

# Enable debug mode (default: 0)
make DEBUG=1

# Specify C standard (default: auto-detect)
make C_STD=c11

# Specify compiler (default: auto-detect)
make CC=clang
```

### CMake Alternative
```bash
mkdir build && cd build
cmake .. -DTHREADS=8 -DDEBUG=ON -DC_STD=c11
make
```

## 🧬 Quick Start

### Python Interface
```python
import lgp
import numpy as np

# Generate sample data
X = np.random.uniform(-2, 2, (100, 1))
y = X[:, 0]**2 + 3*X[:, 0] + 1

# Create LGP input
instruction_set = lgp.InstructionSet.complete()
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set)

# Configure evolution parameters
params = lgp.LGPOptions(
    fitness=lgp.MSE(),
    population_size=500,
    generations=50,
    mutation_rate=0.1,
    target_fitness=1e-6
)

# Evolve solution
result = lgp.evolve(lgp_input, params)
best_individual = result.population.get(result.best_individual)

# Display results
print(f"Best fitness: {best_individual.fitness}")
best_individual.print_program()
```

### C Interface
```c
#include "evolution.h"
#include "psb2.h"

int main() {
    // Initialize random number generation
    random_init_all(42);
    
    // Create instruction set
    struct Operation ops[] = {OP_ADD_F, OP_SUB_F, OP_MUL_F, OP_DIV_F};
    struct InstructionSet instr_set = {.size = 4, .op = ops};
    
    // Create problem instance
    struct LGPInput input = vector_distance(&instr_set, 2, 100);
    
    // Configure evolution parameters
    struct LGPOptions params = {
        .fitness = MSE,
        .selection = tournament,
        .init_params = {.pop_size = 500, .minsize = 5, .maxsize = 20},
        .target = 1e-6,
        .generations = 50
    };
    
    // Evolve solution
    struct LGPResult result = evolve(&input, &params);
    
    // Display best solution
    print_program(&result.pop.individual[result.best_individ].prog);
    
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

