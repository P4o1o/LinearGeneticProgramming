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
#include <stdio.h>

#define NUMBER_OF_OMP_THREADS 16

#include "evolution.h"
#include "psb2.h"
 

static inline double get_time_sec() {
	#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L // >= C11
		struct timespec ts;
		if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
			return ts.tv_sec + ts.tv_nsec * 1e-9;
		}
	#endif
	return (double)clock() / (double)CLOCKS_PER_SEC;    
}

int main(int argc, char *argv[]){
	random_init(7, 0);
	for(uint64_t i = 0; i < MAX_OMP_THREAD; i++){
		uint32_t seed = random();
		printf("seed %ld: %0x\n", i, seed);
		random_init(seed, i);
	}
	const struct LGPOptions par = {
		.fitness = MSE,
		.selection = elitism,
		.select_param = (union SelectionParams) {.size = 5000},
		.initialization_func = unique_population,
		.init_params = (struct InitializationParams) {
			.pop_size = 10000,
			.minsize = 5,
			.maxsize = 20
		},
		.target = 1e-27,
		.mutation_prob = 1.0,
		.max_mutation_len = 10,
		.crossover_prob = 1.0,
		.max_clock = 2200,
		.max_individ_len = MAX_PROGRAM_SIZE,
		.generations = 300,
		.verbose = 1
	};
	struct Operation opset[9] = {OP_ADD_F, OP_SUB_F, OP_MUL_F, OP_DIV_F, OP_SQRT, OP_LOAD_RAM_F, OP_LOAD_ROM_F, OP_STORE_RAM_F, OP_MOV_F};
	struct InstructionSet instr_set = (struct InstructionSet) {
		.size = 9, .op = opset,
	};
	struct LGPInput in = vector_distance(&instr_set, 2, 100);
	double start = get_time_sec();
	const struct LGPResult res = evolve(&in, &par);
	double end = get_time_sec();
	free(in.memory);
	printf("Solution:\n");
	print_program(&(res.pop.individual[res.best_individ].prog));
	printf("Time: %lf, evaluations: %lu, eval/sec: %lf\n", end - start, res.evaluations, ((double) res.evaluations) / (end - start));
	free(res.pop.individual);
	return 0;
}
```

You should get this output:
```
seed 0: 1388f0af
seed 1: e4598df
seed 2: aa034494
seed 3: 8080eea0
seed 4: 853e7fd5
seed 5: 97f02c44
seed 6: 8e669fb2
seed 7: 593a42f0
seed 8: 10fc29eb
seed 9: c5908fb1
seed 10: 5e3b8c1a
seed 11: f6a5574e
seed 12: 69f3cb7a
seed 13: 82a7766f
seed 14: 2d72df96
seed 15: 70fa473
Generation 0, best_mse 4958.227465, population_size 10000
Generation 1, best_mse 0.000000, population_size 20000, evaluations 25000
Solution:

EXIT 
Time: 0.064297, evaluations: 25000, eval/sec: 388820.866134

```

## Project Structure

```
├── src/             # Source code
│   ├── benchmarks.c # Benchmark problem implementations
│   ├── creation.c  # Population initialization functions
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
