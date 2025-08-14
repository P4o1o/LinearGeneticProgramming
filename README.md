# Linear Genetic Programming (LGP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue.svg)](https://github.com/P4o1o/LinearGeneticProgramming)
[![Performance](https://img.shields.io/badge/performance-30k%2B%20evals%2Fsec-green.svg)](https://github.com/P4o1o/LinearGeneticProgramming)

High-performance Linear Genetic Programming framework for symbolic regression, classification, and optimization. Combines an optimized C core with a Python interface for maximum performance and usability.

## Features

- **Dual Architecture**: C core for performance + Python wrapper for ease of use
- **Cross-Platform**: Linux, macOS, Windows, FreeBSD
- **Optimized**: SIMD vectorization (SSE/AVX/AVX-512/NEON) and OpenMP parallelization
- **Comprehensive**: 30+ fitness functions with vectorial output support
- **Advanced VM**: 87 specialized instructions with dual-type registers
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

**Virtual Machine**: 87 specialized instructions, dual-type registers (4 int + 4 float), flexible memory model (ROM/RAM).

## Quick Start

**Python** (recommended) - See [Python Documentation](lgp/README.md):
```python
import lgp
import numpy as np

X = np.random.uniform(-2, 2, (100, 1))
y = X[:, 0]**2 + 3*X[:, 0] + 1

instruction_set = lgp.InstructionSet([
    lgp.Operation.ADD_F, lgp.Operation.MUL_F, lgp.Operation.POW,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F
])

lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=5)
pop, evals, gens, best_idx = lgp.evolve(
    lgp_input, fitness=lgp.MSE(), 
    selection=lgp.Tournament(tournament_size=4),
    initialization=lgp.UniquePopulation(pop_size=200, minsize=5, maxsize=25),
    target=1e-6, generations=50, verbose=1
)

best = pop.get(best_idx)
print(f"Best fitness: {best.fitness:.6e}")
best.print_program()
```

**C** (maximum performance) - See [C Documentation](src/README.md):
```c
#include "src/evolution.h"
#include "src/psb2.h"

int main() {
    random_init_all(42);
    struct Operation ops[] = {OP_ADD_F, OP_MUL_F, OP_LOAD_ROM_F, OP_STORE_RAM_F};
    struct InstructionSet instr_set = {.size = 4, .op = ops};
    struct LGPInput input = vector_distance(&instr_set, 2, 100);
    
    const struct LGPOptions options = {
        .fitness = MSE, .selection = tournament,
        .initialization_func = unique_population,
        .init_params = {.pop_size = 200, .minsize = 5, .maxsize = 25},
        .target = 1e-6, .generations = 50, .verbose = 1
    };
    
    struct LGPResult result = evolve(&input, &options);
    printf("Best fitness: %e
", result.pop.individual[result.best_individ].fitness);
    return 0;
}
```

## 🎯 Key Features

- **Multiple Selection Methods**: Tournament, elitism, roulette wheel, fitness sharing
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

# Create instruction set
instruction_set = lgp.InstructionSet([
    lgp.Operation.ADD_F, lgp.Operation.MUL_F, lgp.Operation.POW,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F
])

# Create LGP input and run evolution
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=5)
population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input,
    fitness=lgp.MSE(),
    selection=lgp.Tournament(tournament_size=4),
    initialization=lgp.UniquePopulation(pop_size=200, minsize=5, maxsize=25),
    target=1e-6,
    generations=50,
    verbose=1
)

# Display results
best = population.get(best_idx)
print(f"Best fitness: {best.fitness:.6e}")
best.print_program()
```

### C Interface (Maximum Performance)
```c
#include "src/evolution.h"
#include "src/psb2.h"

int main() {
    random_init_all(42);
    
    // Create instruction set and problem
    struct Operation ops[] = {OP_ADD_F, OP_MUL_F, OP_LOAD_ROM_F, OP_STORE_RAM_F};
    struct InstructionSet instr_set = {.size = 4, .op = ops};
    struct LGPInput input = vector_distance(&instr_set, 2, 100);
    
    // Configure and run evolution
    const struct LGPOptions options = {
        .fitness = MSE, .selection = tournament,
        .initialization_func = unique_population,
        .init_params = {.pop_size = 200, .minsize = 5, .maxsize = 25},
        .target = 1e-6, .generations = 50, .verbose = 1
    };
    
    struct LGPResult result = evolve(&input, &options);
    printf("Best fitness: %e\n", result.pop.individual[result.best_individ].fitness);
    return 0;
}
```

## 🚀 Use Cases

- **Symbolic Regression**: Discover mathematical formulas from data
- **Classification**: Evolve decision rules and logic circuits  
- **Time Series**: Model temporal patterns and forecasting
- **Feature Engineering**: Automatic feature construction
- **Control Systems**: Evolve controllers and decision logic

## ⚡ Performance

- **30k+ evaluations/sec** on modern hardware
- **Linear scaling** across CPU cores with OpenMP
- **SIMD acceleration** with automatic instruction detection
- **Memory efficient** with SIMD-aligned allocation

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

