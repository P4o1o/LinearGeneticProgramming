# Linear Genetic Programming - Core C Library

This directory contains the core C implementation of the Linear Genetic Programming (LGP) system. The library provides a high-performance, thread-safe implementation with SIMD optimizations for evolutionary computation.

## Architecture Overview

The LGP system consists of several interconnected components:

### Virtual Machine (`vm.h`, `vm.c`)
The VM executes LGP programs with:
- **87 instructions** defined in `INSTR_MACRO` supporting integer/floating-point operations
- **4 integer registers** (`uint64_t reg[REG_NUM]` where `REG_NUM = 4`)
- **4 floating-point registers** (`double freg[FREG_NUM]` where `FREG_NUM = 4`)
- **Flag register** with `odd`, `negative`, `zero`, `exist` bits
- **Memory access** through `union Memblock` supporting both `i64` and `f64` data
- **Program counter** (`prcount`) with jump instruction support
- **Clock limit** enforcement to prevent infinite loops

The VM instruction set includes arithmetic, logic, control flow, memory access, and mathematical functions (trigonometric, logarithmic, etc.).

### Genetic System (`genetics.h`, `genetics.c`)
Core data structures for evolutionary computation:

```c
struct Program {
    struct Instruction *content;  // Array of VM instructions
    uint64_t size;               // Number of instructions
};

struct Individual {
    struct Program prog;         // The genetic program
    double fitness;             // Fitness evaluation result
};

struct Population {
    struct Individual *individual;  // Array of individuals
    uint64_t size;                  // Population size
};
```

### Fitness Functions (`fitness.h`, `fitness.c`)
The library provides **30+ fitness evaluation functions** for different types of machine learning problems:

#### Regression Functions (MINIMIZE)
- **`MSE`**: Mean Squared Error
- **`RMSE`**: Root Mean Squared Error
- **`LENGTH_PENALIZED_MSE`**: MSE + α × program_length (requires `params->fact.alpha`)
- **`CLOCK_PENALIZED_MSE`**: MSE + α × execution_time (requires `params->fact.alpha`)
- **`MAE`**: Mean Absolute Error
- **`MAPE`**: Mean Absolute Percentage Error
- **`SYMMETRIC_MAPE`**: Symmetric Mean Absolute Percentage Error
- **`LOGCOSH`**: Log-Cosh loss function
- **`WORST_CASE_ERROR`**: Maximum error across samples
- **`HUBER_LOSS`**: Robust Huber loss (requires `params->fact.delta`)
- **`PINBALL_LOSS`**: Quantile regression loss (requires `params->fact.quantile`)
- **`GAUSSIAN_LOG_LIKELIHOOD`**: Maximum likelihood estimation (requires `params->fact.sigma`)
- **`BRIER_SCORE`**: Probabilistic forecasting accuracy
- **`HINGE_LOSS`**: Support Vector Machine loss
- **`BINARY_CROSS_ENTROPY`**: Cross-entropy for probabilities (requires `params->fact.tolerance`)

#### Regression Functions (MAXIMIZE)
- **`R_SQUARED`**: Coefficient of determination
- **`PEARSON_CORRELATION`**: Statistical correlation

#### Classification Functions (MAXIMIZE)
Programs output classification decisions via sign bit of integer results (`vm.ram[j].i64 & (1ULL << 63)`):

- **`ACCURACY`**: Per-label classification accuracy
- **`STRICT_ACCURACY`**: Exact match for entire output vector per sample
- **`BINARY_ACCURACY`**: Binary classification accuracy
- **`STRICT_BINARY_ACCURACY`**: Strict binary classification accuracy
- **`THRESHOLD_ACCURACY`**: Tolerance-based accuracy (requires `params->fact.threshold`)
- **`STRICT_THRESHOLD_ACCURACY`**: Strict threshold accuracy
- **`BALANCED_ACCURACY`**: Average of sensitivity and specificity
- **`G_MEAN`**: Geometric mean of sensitivity and specificity
- **`F1_SCORE`**: F1 score (harmonic mean of precision/recall)
- **`F_BETA_SCORE`**: F-Beta score (requires `params->fact.beta`)
- **`MATTHEWS_CORRELATION`**: Matthews correlation coefficient
- **`COHENS_KAPPA`**: Cohen's kappa statistic

#### Specialized Functions
- **`ADVERSARIAL_PERTURBATION_SENSITIVITY`**: Robustness measure (requires `params->fact.perturbation_vector`)
- **`CONDITIONAL_VALUE_AT_RISK`**: Risk management metric (requires `params->fact.alpha`)

Each fitness function:
- Takes parameters: `(LGPInput*, Program*, uint64_t max_clock, FitnessParams*)`
- Executes the program on test instances using `run_vm`
- For regression: compares program output from `vm.ram[params->start]` to `vm.ram[params->end-1]` (as `f64`)
- For classification: uses sign bit of integer output (`vm.ram[j].i64 & (1ULL << 63)`)
- Handles numerical errors (NaN, infinity) by returning `DBL_MAX` (minimization) or `0.0` (maximization)
- Returns `double` fitness value

#### Fitness Parameters
```c
union FitnessFactor {
    const double threshold;         // used in threshold_accuracy
    const double alpha;             // used in length_penalized_mse, clock_penalized_mse, conditional_value_at_risk
    const double beta;              // used in f_beta_score
    const double delta;             // used in huber_loss
    const double quantile;          // used in pinball_loss
    const double tolerance;         // used in binary_cross_entropy
    const double sigma;             // used in gaussian_log_likelihood
    const double *perturbation_vector; // used in adversarial_perturbation_sensitivity
};

struct FitnessParams {
    const uint64_t start;      // First output index to evaluate (0-based)
    const uint64_t end;        // Last output index to evaluate + 1 (exclusive upper bound)
    union FitnessFactor fact;  // Additional parameters specific to fitness function
};
```

**Important**: Programs output to RAM indices `[start, end)`. To evaluate outputs at RAM indices 0-2, use `start=0, end=3`.

### Selection Algorithms (`selection.h`, `selection.c`)
Eight selection algorithms implemented with both MINIMIZE/MAXIMIZE variants:

- **`tournament`**: Tournament selection with configurable tournament size
- **`fitness_sharing_tournament`**: Tournament with diversity preservation
- **`elitism`**: Keep best N individuals
- **`fitness_sharing_elitism`**: Elitism with diversity preservation
- **`percentual_elitism`**: Keep top percentage of population
- **`fitness_sharing_percentual_elitism`**: Percentage elitism with diversity
- **`roulette`**: Roulette wheel selection
- **`fitness_sharing_roulette`**: Roulette with diversity preservation

Selection parameters use `union SelectionParams` with different fields depending on algorithm type.

### Population Initialization (`creation.h`, `creation.c`)
Two initialization strategies:

- **`rand_population`**: Generates random programs within size bounds
- **`unique_population`**: Ensures all programs are genetically unique using xxHash-based deduplication

Both functions return `LGPResult` containing the initial population and evaluation count.

### Evolution Engine (`evolution.h`, `evolution.c`)
The main evolutionary loop implementing:
- **Mutation**: Replaces a random program segment with new random instructions
- **Crossover**: Exchanges segments between two parent programs
- **Jump address correction**: Automatically fixes jump targets that exceed program length
- **Parallel evaluation**: OpenMP-based concurrent fitness evaluation

## Input/Output Interface

### Problem Input (`LGPInput`)
```c
struct LGPInput {
    const uint64_t input_num;    // Number of test instances
    const uint64_t rom_size;     // Problem data size per instance
    const uint64_t res_size;     // Expected solution size per instance
    const uint64_t ram_size;     // RAM size (must be >= res_size)
    const struct InstructionSet instr_set;  // Allowed instruction set
    union Memblock *memory;      // Memory layout: [instance0_data][instance0_solution][instance1_data][instance1_solution]...
};
```

The memory layout has `(rom_size + res_size)` blocks per instance, totaling `input_num * (rom_size + res_size)` blocks.

### Evolution Configuration (`LGPOptions`)
All fields are **required**:

```c
struct LGPOptions {
    // Fitness configuration
    const struct Fitness fitness;              // One of: MSE, RMSE, MAE, ACCURACY, R_SQUARED, etc. (30+ available)
    const struct FitnessParams fitness_param;  // start, end, and fact (union with specific parameters)
    
    // Selection configuration  
    const struct Selection selection;           // Selection algorithm
    const union SelectionParams select_param;  // Algorithm-specific parameters
    
    // Population initialization (must provide ONE of these)
    const initialization_fn initialization_func;     // Function pointer (rand_population or unique_population) OR
    const struct Population initial_pop;             // Pre-built population (if initialization_func == NULL)
    const struct InitializationParams init_params;   // pop_size, minsize, maxsize (used if initialization_func != NULL)
    
    // Evolution parameters
    const double target;                // Target fitness for early termination
    const double mutation_prob;         // Mutation probability (>= 0.0, can be > 1.0 for multiple mutations per individual)
    const double crossover_prob;        // Crossover probability (>= 0.0, can be > 1.0 for multiple crossovers per individual)
    const uint64_t max_clock;          // VM execution limit per program run
    const uint64_t max_individ_len;    // Maximum program length
    const uint64_t max_mutation_len;   // Maximum length of mutation segments
    const uint64_t generations;        // Maximum generations to run
    const unsigned verbose;            // 0 = silent, 1 = print per-generation statistics
};
```

### Evolution Result (`LGPResult`)
```c
struct LGPResult {
    const struct Population pop;    // Final population
    const uint64_t evaluations;    // Total fitness evaluations performed
    const uint64_t generations;    // Actual generations executed (may be < max if target reached)
    const uint64_t best_individ;   // Index of best individual in pop.individual array
};
```

## Main Evolution Function

```c
struct LGPResult evolve(const struct LGPInput *const in, const struct LGPOptions *const args);
```

The `evolve` function executes the complete evolutionary algorithm. Early termination occurs if the best individual's fitness reaches the target value.

## Memory Management

- **Aligned allocation**: Uses `aligned_alloc()` with `VECT_ALIGNMENT` for SIMD compatibility
- **VECT_ALIGNMENT**: 16 bytes (SSE2), 32 bytes (AVX2), or 64 bytes (AVX512)
- **Program padding**: Extra instructions beyond program size are filled with `I_EXIT` 
- **Memory cleanup**: Use `free_individual()`, `free_population()`, `free_lgp_input()` for cleanup
- **Instruction alignment**: Programs allocated with extra space for SIMD alignment

## Concurrency and Performance

- **OpenMP support**: Controlled by `NUMBER_OF_OMP_THREADS` macro
- **Thread-safe random generation**: Each thread has its own MT19937 state
- **SIMD optimizations**: AVX/AVX512 implementations for MT19937 and xxHash
- **Parallel sections**: Fitness evaluation, population initialization, genetic operations

Random number access via `random()` macro automatically uses thread-local generator.

## Random Number Generation (`prob.h`, `mt19937.h`)

High-performance MT19937 implementation with:
- **Vectorized generation**: AVX512/AVX2/SSE2 batch generation
- **Thread-local state**: `random_engines[NUMBER_OF_OMP_THREADS]` array
- **Utility macros**: 
  - `WILL_HAPPEN(prob)` - probabilistic events
  - `RAND_BOUNDS(min, max)` - integers in range
  - `RAND_DBL_BOUNDS(min, max)` - doubles in range

Initialize with `random_init_all(seed)` before use.

## Benchmark Problems (PSB2)

Five PSB2 benchmark problems in `psb2.h`:

1. **`vector_distance(instr_set, vector_len, instances)`**: Euclidean distance between vectors
2. **`bouncing_balls(instr_set, instances)`**: Ball trajectory physics simulation  
3. **`dice_game(instr_set, instances)`**: Dice game probability calculation
4. **`shopping_list(instr_set, num_items, instances)`**: Shopping total with discounts
5. **`snow_day(instr_set, instances)`**: Snow accumulation/melting simulation

Each returns a configured `LGPInput` ready for evolution.

## Error Handling and Logging

- **Assertion system**: `ASSERT(condition)` macro calls `unreachable()` on failure
- **Logging**: `LOG_EXIT(message)` and `LOG_EXIT_THREADSAFE(message)` log to `genetic.log` and exit
- **Memory allocation**: `MALLOC_FAIL` and `MALLOC_FAIL_THREADSAFE` macros for allocation failures
- **Fitness error handling**: NaN/infinity results properly handled with penalty values

## Build Configuration

The library auto-detects SIMD capabilities:
- **SIMD support**: Automatic AVX512/AVX2/SSE2 detection via compiler flags
- **OpenMP**: Detected via `_OPENMP` preprocessor define
- **C standard**: Supports C89/C99/C11/C17/C23 with appropriate fallbacks
- **Alignment**: Automatic aligned allocation based on available SIMD instructions

## Thread Safety

The library is thread-safe when:
- `random_init_all()` called before parallel sections
- Each thread uses his own MT19937
- Memory allocation/deallocation not shared between threads

## Basic Usage Example

```c
#include "evolution.h"
#include "psb2.h"

int main() {
    // Initialize random number generation
    random_init_all(42);
    
    // Create instruction set (all vm's operations)
    struct InstructionSet instr_set = {.size = INSTR_NUM, .op = INSTRSET};
    
    // Create problem using PSB2 benchmark
    struct LGPInput input = vector_distance(&instr_set, 3, 100);
    
    // Configure evolution
    struct LGPOptions options = {
        .fitness = MSE,
        .fitness_param = {.start = 0, .end = 1},
        .selection = tournament,
        .select_param = {.size = 3},
        .initialization_func = unique_population,
        .init_params = {.pop_size = 500, .minsize = 5, .maxsize = 30},
        .target = 1e-6,
        .mutation_prob = 0.1,
        .crossover_prob = 0.9,
        .max_clock = 1000,
        .max_individ_len = 100,
        .max_mutation_len = 10,
        .generations = 100,
        .verbose = 1
    };
    
    // Run evolution
    struct LGPResult result = evolve(&input, &options);
    
    // Access best individual
    struct Individual *best = &result.pop.individual[result.best_individ];
    printf("Best fitness: %f\n", best->fitness);
    printf("Generations: %lu\n", result.generations);
    printf("Evaluations: %lu\n", result.evaluations);
    
    // Print the evolved program
    print_program(&best->prog);
    
    // Cleanup
    free_population(&result.pop);
    free_lgp_input(&input);
    
    return 0;
}
```
