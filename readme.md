# Linear Genetic Programming (LGP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A cross-platform, parallel implementation of Linear Genetic Programming in C. This framework enables automatic program synthesis through evolutionary computation, representing programs as linear sequences of instructions operating on virtual registers.

## Features

- **High Performance**: Optimized C implementation with OpenMP parallelization
- **Cross-Platform**: Works on Linux, Windows, and macOS
- **Modular Design**: Easily extendable with new operations and selection methods
- **Code Bloat Control**: Automatic removal of ineffective instructions
- **Multiple Selection Methods**:
  - Elitism
  - Tournament selection
  - Roulette wheel selection
  - Percentual elitism
  - Fitness sharing variants

## Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/lgp.git
cd lgp

# Build the project
make

# Run the example
./lgp
```

### CLANG
before compiling with clang you must have libomp-dev installed
```bash
sudo apt update
sudo apt install libomp-dev
```

## Usage Example

```c
#include "genetics.h"
#include "evolution.h"

int main() {
    // Create a genetic environment with basic operations
    struct genetic_env env = simple_genv(5);
    
    // Create input data for the problem
    struct genetic_input input = vector_distance(&env, 3, 100);
    
    // Configure evolution parameters
    struct genetic_options params;
    params.tollerance = 1e-12;
    params.generations = 100;
    params.initial_pop_size = 4000;
    params.init_type = unique_population;
    params.dna_minsize = 2;
    params.dna_maxsize = 5;
    params.evolution_cycles = 1;
    params.crossover_prob = 0.77;
    params.mutation_prob = 0.96;
    params.mut_max_len = 5;
    params.select_type = elitism;
    params.select_param.size = 1000;
    params.verbose = 1;
    
    // Execute evolution
    print_evolution(&input, &params);
    
    // Clean up
    free_genetic_input(&input);
    
    return 0;
}
```

## Project Structure

```
├── src/             # Source code
│   ├── benchmarks.c # Benchmark problem implementations
│   ├── creations.c  # Population initialization functions
│   ├── evolution.c  # Main evolutionary algorithm
│   ├── float_psb2.c # PSB2 problem implementations
│   ├── genetics.c   # Core genetic operations
│   ├── logger.c     # Logging utilities
│   ├── main.c       # Main program entry
│   ├── operations.c # Primitive operations
│   ├── prob.c       # Probability utilities
│   └── selections.c # Selection methods
├── bin/             # Compiled binaries
├── DEAP/            # DEAP comparison code
├── doc/             # Documentation
├── graphs/          # Performance visualizations
├── tests/           # Test files
└── Makefile         # Build system
```

## Dependencies

- C compiler with C99 support
- OpenMP for parallelization

## Contributing

Contributions are welcome! Some areas for potential improvement:

- Additional primitive operations
- More advanced code bloat control
- Support for additional data types
- Implementation of control structures
- Integration with machine learning techniques
- GUI and visualization tools
