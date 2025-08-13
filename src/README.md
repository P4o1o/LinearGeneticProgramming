# Linear Genetic Programming - Core C Library

This directory contains the high-performance C implementation of the Linear Genetic Programming (LGP) system. The library provides a thread-safe, SIMD-optimized implementation designed for maximum performance in evolutionary computation applications.

## ⚠️ CRITICAL: NO INPUT VALIDATION

**This library is designed for maximum performance and does NOT perform input validation.** All functions expect properly formatted inputs and will crash, corrupt memory, or produce undefined behavior with invalid data.

### Requirements for ALL functions:
- **Memory alignment**: Program content arrays must use `VECT_ALIGNMENT` for SIMD optimization
- **Null pointer safety**: No NULL checks - passing NULL will crash immediately
- **Array bounds**: No bounds checking - invalid indices cause memory corruption  
- **Data type consistency**: Fitness functions expect specific data types (see detailed sections)
- **Structure completeness**: All required fields must be properly initialized before use
- **Threading**: Must call `random_init_all()` before any OpenMP parallel sections

## Architecture Overview

The LGP system consists of multiple interconnected components designed for maximum performance:

### Virtual Machine (`vm.h`, `vm.c`)

The LGP Virtual Machine executes linear genetic programs with a sophisticated instruction set architecture.

#### Core VM Structure
```c
struct VirtualMachine {
    struct Core core;           // CPU state (registers, flags, program counter)
    struct Instruction *program; // Program instructions array
    union Memblock *ram;        // Working memory (read/write)
    union Memblock *rom;        // Problem data (read-only)
};

struct Core {
    uint64_t reg[4];           // 4 integer registers
    double freg[4];            // 4 floating-point registers
    uint64_t prcount;          // Program counter
    uint8_t odd;               // Odd flag
    uint8_t negative;          // Negative flag  
    uint8_t zero;              // Zero flag
    uint8_t exist;             // Exist flag
};
```

#### Instruction Set Architecture

**87 Total Instructions** defined in `INSTR_MACRO` covering:

**Control Flow Instructions (17 instructions)**:
- `EXIT`: Terminates program execution
- `JMP`, `JMP_Z`, `JMP_NZ`: Unconditional and conditional jumps
- `JMP_L`, `JMP_G`, `JMP_LE`, `JMP_GE`: Comparison-based jumps
- `JMP_EXIST`, `JMP_NEXIST`: Existence flag jumps
- `JMP_ODD`, `JMP_EVEN`: Parity-based jumps
- `CLC`: Clear all flags
- `CMP`, `CMP_F`: Compare integers/floats and set flags
- `TEST`, `TEST_F`: Test values and set existence flags

**Memory Operations (6 instructions)**:
- `LOAD_RAM`, `LOAD_RAM_F`: Load from RAM to registers (int/float)
- `STORE_RAM`, `STORE_RAM_F`: Store from registers to RAM (int/float)  
- `LOAD_ROM`, `LOAD_ROM_F`: Load from ROM to registers (int/float)

**Data Movement (24 instructions)**:
- `MOV`, `MOV_F`: Unconditional register moves
- `MOV_I`, `MOV_I_F`: Move immediate values to registers
- `CMOV_Z`, `CMOV_Z_F`: Conditional moves based on zero flag
- `CMOV_NZ`, `CMOV_NZ_F`: Conditional moves based on non-zero flag
- `CMOV_L`, `CMOV_L_F`: Conditional moves based on less-than comparison
- `CMOV_G`, `CMOV_G_F`: Conditional moves based on greater-than comparison
- `CMOV_LE`, `CMOV_LE_F`: Conditional moves based on less-equal comparison
- `CMOV_GE`, `CMOV_GE_F`: Conditional moves based on greater-equal comparison
- `CMOV_EXIST`, `CMOV_EXIST_F`: Conditional moves based on existence flag
- `CMOV_NEXIST`, `CMOV_NEXIST_F`: Conditional moves based on non-existence flag
- `CMOV_ODD`, `CMOV_ODD_F`: Conditional moves based on odd flag
- `CMOV_EVEN`, `CMOV_EVEN_F`: Conditional moves based on even flag

**Integer Arithmetic (14 instructions)**:
- `ADD`, `SUB`, `MUL`: Basic arithmetic operations
- `DIV`, `MOD`: Division and modulo with zero-check
- `INC`, `DEC`: Increment and decrement
- `AND`, `OR`, `XOR`, `NOT`: Bitwise logical operations
- `SHL`, `SHR`: Bit shift operations
- `CAST`: Integer to float conversion

**Floating-Point Arithmetic (4 instructions)**:
- `ADD_F`, `SUB_F`, `MUL_F`, `DIV_F`: Basic float operations with NaN/infinity handling

**Mathematical Functions (20 instructions)**:
- `SQRT`, `POW`, `EXP`: Power and exponential functions
- `LN`, `LOG`, `LOG10`: Logarithmic functions
- `COS`, `SIN`, `TAN`: Trigonometric functions
- `ACOS`, `ASIN`, `ATAN`: Inverse trigonometric functions  
- `COSH`, `SINH`, `TANH`: Hyperbolic functions
- `ACOSH`, `ASINH`, `ATANH`: Inverse hyperbolic functions
- `CAST_F`: Float to integer conversion
- `ROUND`: Round floating-point number

**Utility Instructions (2 instructions)**:
- `NOP`: No operation (useful for padding)
- `RAND`: Generate random numbers using thread-local MT19937
- `RAND`: Generate random numbers using thread-local MT19937

#### Memory Model

```c
union Memblock {
    uint64_t i64;    // 64-bit integer access
    double f64;      // 64-bit floating-point access
};
```

**Key Features**:
- **Unified memory**: Same memory block accessible as integer or float
- **SIMD alignment**: All memory allocations use `VECT_ALIGNMENT` for vector operations
- **Cache optimization**: Contiguous memory layout for better spatial locality
- **Bounds safety**: ROM operations include bounds checking, RAM operations don't

#### VM Execution Model

```c
uint64_t run_vm(struct VirtualMachine *vm, uint64_t max_clock);
```

**Execution Features**:
- **Clock limiting**: Prevents infinite loops with maximum instruction count
- **Flag-based control**: Complex conditional execution based on computed flags
- **Error handling**: Division by zero, NaN, infinity handled gracefully
- **Thread safety**: Independent execution contexts per thread
- **Performance**: Optimized instruction dispatch with minimal overhead

### Genetic System (`genetics.h`, `genetics.c`)

Core data structures for genetic programming with comprehensive type definitions.

#### Program Representation

```c
struct Program {
    struct Instruction *content;  // MUST be VECT_ALIGNMENT aligned
    uint64_t size;               // Number of instructions (> 0)
};

struct Instruction {
    uint64_t opcode;        // Instruction type (0-86)
    uint64_t operand[3];    // Up to 3 operands depending on instruction
};
```

**Program Requirements**:
- **Alignment**: Content must be allocated with `aligned_alloc(VECT_ALIGNMENT, ...)`
- **Termination**: Programs automatically padded with `I_EXIT` instructions
- **Size bounds**: Must be between initialization `minsize` and `maxsize`
- **Memory safety**: Extra space allocated beyond program size for safe execution

#### Individual and Population Management

```c
struct Individual {
    struct Program prog;         // Genetic program
    double fitness;             // Cached fitness value (DBL_MAX if not evaluated)
};

struct Population {
    struct Individual *individual;  // Array of individuals
    uint64_t size;                  // Population size (> 0)
};
```

**Multi-Objective Support**:
```c
struct MultiIndividual {
    struct Program prog;         // Genetic program  
    double *fitness;            // Array of fitness values for multiple objectives
};

struct MultiPopulation {
    struct MultiIndividual *individual;  // Multi-objective individuals
    uint64_t size;                       // Population size
};
```

#### Problem Input Structure

```c
struct LGPInput {
    const uint64_t input_num;             // Number of training instances (> 0)
    const uint64_t rom_size;              // Read-only memory size per instance (> 0)
    const uint64_t res_size;              // Expected result size per instance (> 0)  
    const uint64_t ram_size;              // Working memory size (MUST be >= res_size)
    const struct InstructionSet instr_set; // Available VM instructions
    union Memblock *restrict memory;      // Training data storage
};

struct InstructionSet {
    const uint64_t size;           // Number of available operations (> 0)
    const struct Operation *op;    // Array of operation definitions
};
```

**Memory Layout Design**:
```
Total memory: input_num × (rom_size + res_size) blocks
Layout: [instance0_ROM][instance0_RAM][instance1_ROM][instance1_RAM]...

For each instance:
- ROM section: Input features (read-only during program execution)  
- RAM section: Expected outputs in first res_size positions, working memory after
```

**Critical Requirements**:
- **Memory allocation**: Must use `aligned_alloc()` for SIMD compatibility
- **Data consistency**: ROM data immutable, RAM data contains expected results
- **Size constraints**: `ram_size >= res_size` always required
- **Result extraction**: Programs write results to `ram[0]` through `ram[res_size-1]`

### Fitness Functions (`fitness.h`, `fitness.c`)

The library provides **33 specialized fitness evaluation functions** for different machine learning problems. Each function has specific data type requirements and optimization objectives.

#### Function Categories and Data Type Requirements

**⚠️ CRITICAL**: Each fitness function expects specific data types. Using wrong types will produce incorrect results or crashes.

#### Regression Functions (MINIMIZE - lower values are better)

**Basic Error Metrics (expect floating-point outputs)**:
```c
double mse(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double rmse(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double mae(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Data expectation**: Program outputs `vm.ram[i].f64` compared with targets `actual[i].f64`
- **Range**: Any real numbers (no constraints)
- **Usage**: Standard regression problems, continuous prediction

**Advanced Error Metrics**:
```c
double mape(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double symmetric_mape(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double logcosh(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double worst_case_error(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **MAPE requirement**: Target values MUST NOT be zero (division by zero)
- **Robust alternatives**: `logcosh` and `symmetric_mape` handle edge cases better

**Parameterized Loss Functions**:
```c
double huber_loss(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double pinball_loss(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double gaussian_log_likelihood(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Huber loss**: Requires `params->fact.delta > 0.0` (robustness parameter)
- **Pinball loss**: Requires `0.0 < params->fact.quantile < 1.0` (quantile regression)
- **Gaussian likelihood**: Requires `params->fact.sigma > 0.0` (standard deviation)

**Penalized Variants**:
```c
double length_penalized_mse(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double clock_penalized_mse(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Length penalized**: `MSE + alpha × program_length`
- **Clock penalized**: `MSE + alpha × execution_time`
- **Both require**: `params->fact.alpha >= 0.0` (penalization strength)

#### Regression Functions (MAXIMIZE - higher values are better)

```c
double r_squared(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double pearson_correlation(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Range**: `r_squared` in [0, 1], `pearson_correlation` in [-1, 1]
- **Usage**: Statistical quality measures for regression

#### Classification Functions 

**Integer-Based Classification (MAXIMIZE)**:
```c
double accuracy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double strict_accuracy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double binary_accuracy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double strict_binary_accuracy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Data expectation**: Program outputs `vm.ram[i].i64` compared with targets `actual[i].i64`
- **Matching**: Exact integer equality required
- **Strict variants**: Require entire output vector to match exactly

**Threshold-Based Classification**:
```c
double threshold_accuracy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double strict_threshold_accuracy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Data expectation**: Floating-point outputs compared with tolerance
- **Requirement**: `params->fact.threshold >= 0.0` (tolerance value)
- **Usage**: Classification with numerical tolerance

**Sign-Bit Binary Classification (MAXIMIZE)**:
```c
double f1_score(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double balanced_accuracy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double g_mean(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double matthews_correlation(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double cohens_kappa(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Data interpretation**: Uses sign bit of `vm.ram[i].i64 & (1ULL << 63)`
- **Encoding**: Negative integers = class 0/false, positive integers = class 1/true
- **Metrics**: Advanced binary classification measures handling class imbalance

#### Probabilistic Functions

```c
double binary_cross_entropy(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double brier_score(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **Data expectation**: Program outputs `vm.ram[i].f64` MUST be in [0.0, 1.0] range
- **Targets**: Usually binary (0.0 or 1.0) but can be probability distributions
- **Binary cross-entropy**: Requires `params->fact.tolerance > 0.0` for numerical stability

#### Specialized Functions

```c
double conditional_value_at_risk(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double adversarial_perturbation_sensitivity(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
double hinge_loss(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, const struct FitnessParams *params);
```
- **CVaR**: Financial risk metric, requires `params->fact.alpha` in (0.0, 1.0)
- **Adversarial**: Robustness measure, requires `params->fact.perturbation_vector` array of size `input_num`
- **Hinge loss**: SVM-style loss, targets should be -1 or +1

#### Fitness Parameters Structure

```c
struct FitnessParams {
    const uint64_t start;      // Start RAM index (inclusive, >= 0)
    const uint64_t end;        // End RAM index (exclusive, MUST be > start)
    union FitnessFactor fact;  // Function-specific parameters
};

union FitnessFactor {
    const double threshold;     // Threshold/tolerance (>= 0.0)
    const double alpha;         // Penalization/risk parameter (>= 0.0)  
    const double beta;          // F-beta score beta (> 0.0)
    const double delta;         // Huber loss robustness (> 0.0)
    const double quantile;      // Pinball loss quantile (0.0 < q < 1.0)
    const double tolerance;     // Cross-entropy tolerance (> 0.0, typically 1e-10)
    const double sigma;         // Gaussian likelihood std dev (> 0.0)
    const double *perturbation_vector; // Adversarial perturbations (size = input_num)
};
```

**Critical Usage Requirements**:
1. **Output Range**: `[start, end)` defines which RAM positions to evaluate
2. **Range validation**: `end > start` and `end <= ram_size` required
3. **Parameter matching**: Each function expects specific `FitnessFactor` fields
4. **Memory safety**: Invalid ranges cause crashes (no bounds checking)

#### Multi-Objective Fitness

```c
double *eval_multifitness(const struct LGPInput *in, const struct Program *prog, uint64_t max_clock, struct MultiFitness *fitness);

struct MultiFitness {
    const uint64_t size;                    // Number of objectives (> 0)
    const struct FitnessFunction *functions; // Array of fitness function descriptors
    const struct FitnessParams *params;     // Parameters for each function
};
```

**Usage**: Simultaneous evaluation of multiple objectives for Pareto optimization.

### Selection Algorithms (`selection.h`, `selection.c`)

Eight sophisticated selection algorithms with both single and multi-objective variants.

#### Selection Function Interface

```c
typedef struct Population (*selection_fn)(
    const struct Population *const pop,      // Input population (MUST be valid)
    const union SelectionParams *const par, // Algorithm parameters (MUST match type)
    const uint64_t max_size                 // Desired output population size (> 0)
);
```

#### Selection Parameter Types

```c
union SelectionParams {
    uint64_t size;                    // Tournament size, elite count, sample size
    double percentage;                // Elite percentage (0.0 <= p <= 1.0)
    struct FitnessSharingParams sharing; // For fitness sharing variants
};

struct FitnessSharingParams {
    double alpha;    // Sharing function exponent (typically 1.0)
    double beta;     // Fitness scaling power (typically 1.0)  
    double sigma;    // Sharing radius (problem-specific, usually 0.01-0.1)
};
```

#### Available Selection Methods

**Basic Selection Algorithms**:

```c
struct Population tournament(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
```
- **Parameter**: `par->size` = tournament size (recommended: 2-7)
- **Mechanism**: Randomly select `size` individuals, pick best fitness
- **Performance**: O(max_size × tournament_size), excellent diversity preservation
- **Usage**: General-purpose selection, good balance of selection pressure and diversity

```c  
struct Population elitism(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
```
- **Parameter**: `par->size` = number of elites to preserve (≤ pop->size)
- **Mechanism**: Keep absolute best `size` individuals
- **Performance**: O(pop_size × log(pop_size)) for sorting
- **Usage**: Ensure best solutions survive, high selection pressure

```c
struct Population percentual_elitism(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
```
- **Parameter**: `par->percentage` = elite proportion (0.0 ≤ p ≤ 1.0)
- **Mechanism**: Keep top percentage of population  
- **Advantage**: Scales automatically with population size changes
- **Usage**: Adaptive elitism, good for variable population sizes

```c
struct Population roulette(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
```
- **Parameter**: `par->size` = sampling size
- **Mechanism**: Fitness-proportional probability selection
- **Requirements**: All fitness values MUST be non-negative for minimization problems
- **Usage**: Classical genetic algorithm selection, preserves fitness proportions

**Fitness Sharing Variants** (Promote Diversity):

```c
struct Population fitness_sharing_tournament(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
struct Population fitness_sharing_elitism(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
struct Population fitness_sharing_percentual_elitism(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
struct Population fitness_sharing_roulette(const struct Population *pop, const union SelectionParams *par, uint64_t max_size);
```

**Fitness Sharing Mechanism**:
- **Distance calculation**: Programs compared using Hamming distance on instruction sequences
- **Sharing formula**: `shared_fitness = original_fitness / (1 + sharing_factor)`
- **Effect**: Similar programs receive reduced fitness, promoting diversity
- **Parameters**: `alpha` controls sharing intensity, `sigma` controls sharing radius

**Parameter Requirements**:
- **`sigma`**: Critical parameter determining neighborhood size (typically 0.01-0.1)
- **`alpha`**: Sharing function shape (1.0 = linear sharing)  
- **`beta`**: Fitness scaling exponent (1.0 = no scaling)

#### Multi-Objective Selection

```c
typedef struct MultiPopulation (*multi_selection_fn)(
    const struct MultiPopulation *const pop,
    const union MultiSelectionParams *const par,
    const uint64_t max_size
);
```

**Approaches**:
- **Pareto dominance**: Non-dominated sorting for multi-objective optimization
- **Lexicographic**: Priority-based objective ordering
- **Weighted sum**: Linear combination of objectives

### Population Initialization (`creation.h`, `creation.c`)

Two initialization strategies optimized for different use cases.

#### Initialization Function Interface

```c
typedef struct LGPResult (*initialization_fn)(
    const struct LGPInput *const in,        // Problem definition (MUST be valid)
    const struct InitializationParams *const par // Initialization parameters
);

struct InitializationParams {
    uint64_t pop_size;    // Desired population size (> 0)
    uint64_t minsize;     // Minimum program length (> 0)
    uint64_t maxsize;     // Maximum program length (>= minsize)
};
```

#### Available Initialization Functions

**Random Population Generation**:
```c
struct LGPResult rand_population(const struct LGPInput *in, const struct InitializationParams *par);
```
- **Speed**: Very fast generation, minimal overhead
- **Quality**: May contain duplicate individuals
- **Memory**: Simple allocation without deduplication
- **Usage**: Quick experiments, large populations, when duplicates acceptable

**Unique Population Generation**:
```c
struct LGPResult unique_population(const struct LGPInput *in, const struct InitializationParams *par);
```
- **Quality**: Guarantees genetically unique individuals
- **Algorithm**: xxHash-based deduplication with linear probing hash table
- **Overhead**: Higher computational cost for uniqueness checking
- **Memory**: Additional hash table memory required
- **Usage**: High-quality initialization, preventing premature convergence

**Generation Process**:
1. **Instruction selection**: Random sampling from provided `InstructionSet`
2. **Length variation**: Uniform distribution between `minsize` and `maxsize`  
3. **Operand generation**: Context-appropriate operand values based on instruction type
4. **Memory alignment**: All programs allocated with `VECT_ALIGNMENT` for SIMD
5. **Termination padding**: Extra instructions filled with `I_EXIT` for safety

### Evolution Engine (`evolution.h`, `evolution.c`)

The main evolutionary loop implementing sophisticated genetic operators and evolution strategies.

#### Main Evolution Function

```c
struct LGPResult evolve(
    const struct LGPInput *const in,    // Problem definition (MUST be properly initialized)
    const struct LGPOptions *const args // Evolution configuration (ALL fields required)
);
```

#### Evolution Configuration Structure

```c
struct LGPOptions {
    // Fitness evaluation configuration
    const struct Fitness fitness;              // Fitness function selection
    const struct FitnessParams fitness_param;  // Function parameters and output range
    
    // Selection strategy configuration  
    const struct Selection selection;           // Selection algorithm choice
    const union SelectionParams select_param;  // Algorithm-specific parameters
    
    // Population initialization (choose ONE approach)
    const initialization_fn initialization_func;     // Function pointer (rand_population/unique_population)
    const struct InitializationParams init_params;   // Population size and length bounds
    const struct Population initial_pop;             // Alternative: pre-built population
    
    // Termination criteria
    const double target;                // Target fitness for early termination
    const uint64_t generations;        // Maximum generations to execute
    
    // Genetic operator configuration
    const double mutation_prob;         // Mutation probability (≥ 0.0, can be > 1.0)
    const double crossover_prob;        // Crossover probability (≥ 0.0, can be > 1.0)
    const uint64_t max_mutation_len;   // Maximum mutation segment length
    
    // Execution constraints
    const uint64_t max_clock;          // VM execution limit per program run
    const uint64_t max_individ_len;    // Maximum program length (prevents bloat)
    
    // Progress monitoring
    const unsigned verbose;            // Verbosity (0=silent, 1=per-generation, N=every N generations)
};
```

**Critical Configuration Notes**:
- **Probabilities > 1.0**: Enable multiple genetic operations per individual per generation
- **Initialization choice**: Use `initialization_func != NULL` OR pre-built `initial_pop`
- **Clock limits**: Prevent infinite loops in evolved programs
- **Length limits**: Control code bloat and memory usage

#### Genetic Operators

**Mutation Operator**:
- **Type**: Segment replacement mutation
- **Mechanism**: Replace random program segment with new random instructions
- **Length variation**: Can increase or decrease program size within bounds
- **Instruction diversity**: Uses full available `InstructionSet`
- **Multiple mutations**: Probability > 1.0 enables multiple mutations per individual

**Crossover Operator**:
- **Type**: Segment exchange crossover
- **Mechanism**: Exchange random segments between two parent programs
- **Size preservation**: Maintains reasonable program lengths after operation
- **Genetic combination**: Creates novel instruction combinations
- **Multiple crossovers**: Probability > 1.0 enables multiple exchanges per generation

**Jump Address Correction**:
- **Automatic repair**: Fixes jump targets exceeding program length after genetic operations
- **Boundary enforcement**: Ensures all jumps remain within valid program bounds  
- **Integrity maintenance**: Preserves program executability after modifications

#### Evolution Process

1. **Population initialization**: Create initial population using specified method
2. **Fitness evaluation**: Parallel evaluation using OpenMP when available  
3. **Selection**: Apply configured selection algorithm
4. **Genetic operations**: Apply mutation and crossover with specified probabilities
5. **Address correction**: Fix any invalid jump addresses
6. **Termination check**: Stop if target fitness reached or max generations exceeded
7. **Progress reporting**: Print statistics based on verbosity level

#### Evolution Result Structure

```c
struct LGPResult {
    const struct Population pop;      // Final evolved population
    const uint64_t evaluations;      // Total fitness evaluations performed
    const uint64_t generations;      // Actual generations executed
    const uint64_t best_individ;     // Index of best individual in population
};
```

**Result Analysis**:
- **Best individual**: Access via `result.pop.individual[result.best_individ]`
- **Performance metrics**: Use `evaluations` and `generations` for algorithm analysis
- **Early termination**: `generations` < configured max indicates target reached

#### Multi-Objective Evolution

```c
struct LGPMultiResult multi_evolve(
    const struct LGPInput *const in, 
    const struct LGPMultiOptions *const args
);
```

**Advanced Features**:
- **Pareto optimization**: Non-dominated solution discovery
- **Lexicographic ordering**: Priority-based multi-objective handling
- **Objective trade-offs**: Balanced optimization across multiple criteria

## Build System and Configuration

The Makefile provides comprehensive build configuration with automatic feature detection.

### User-Configurable Build Options

```bash
# Debug configuration
make DEBUG=1          # Enable assertions, debug symbols, verbose logging
make DEBUG=0          # Release build with optimizations (default)

# Thread configuration  
make THREADS=32       # Set OpenMP thread count (default: 16)

# C standard selection
make C_STD=c11        # Specify C standard (auto, c89, c99, c11, c17, c2x)
make C_STD=auto       # Auto-detect best supported standard (default)

# Compiler selection
make CC=gcc           # GNU Compiler Collection
make CC=clang         # Clang/LLVM compiler  
make CC=icc           # Intel C Compiler
```

### Automatic Feature Detection

The build system automatically detects and optimizes for:

**SIMD Instruction Sets**:
- **x86-64**: SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA, AVX-512 variants
- **ARM**: NEON vector extensions for AArch64
- **Fallback**: Scalar implementations for compatibility

**Compiler Types**:
- **GCC**: GNU Compiler Collection with GCC-specific optimizations
- **Clang**: LLVM-based compiler with Clang-specific features
- **MSVC**: Microsoft Visual C++ for Windows builds
- **ICC**: Intel C Compiler with Intel-specific optimizations

**Operating Systems**:
- **Linux**: Standard POSIX environment with Linux-specific optimizations
- **macOS**: Darwin environment with Apple-specific configurations  
- **Windows**: Windows API integration and MSVC compatibility
- **FreeBSD**: BSD-specific configurations and library handling

**OpenMP Support**:
- **Automatic detection**: Based on compiler support and `_OPENMP` preprocessor define
- **Flag selection**: Compiler-appropriate OpenMP flags (`-fopenmp`, `/openmp`)
- **Thread configuration**: Configurable thread count with `THREADS` parameter

### Available Build Targets

```bash
make                  # Build C executable (LGP)
make python          # Build Python shared library (liblgp.so/.dylib/.dll)
make clean           # Clean all
make help            # Display all available targets
make info            # Show detected build configuration
```

### Optimization Features

**SIMD Vectorization**:
- **Automatic**: Compiler automatically vectorizes suitable operations
- **Alignment**: Memory allocations use appropriate alignment for target SIMD
- **Fallback**: Graceful degradation when SIMD unavailable

**Memory Optimization**:
- **Alignment**: `VECT_ALIGNMENT` set based on available SIMD (16/32/64 bytes)
- **Cache locality**: Contiguous memory layouts for better performance
- **Memory pooling**: Efficient allocation patterns for genetic operations

## Memory Management and Safety

### Memory Allocation Requirements

**Alignment Requirements**:
```c
// Only program content arrays MUST use VECT_ALIGNMENT
struct Instruction *content = aligned_alloc(VECT_ALIGNMENT, size * sizeof(struct Instruction));
struct Individual *pop = malloc(pop_size * sizeof(struct Individual));
```

**Memory Layout Standards**:
- **Programs**: Extra space beyond `size` filled with `I_EXIT` instructions
- **Populations**: Contiguous allocation for SIMD operations
- **Problem data**: `LGPInput.memory` must be properly allocated and aligned

### Memory Cleanup Functions

```c
void free_individual(struct Individual *ind);      // Free single individual and its program
void free_population(struct Population *pop);      // Free entire population recursively  
void free_lgp_input(struct LGPInput *input);      // Free problem input and associated memory
```

**Critical Notes**:
- **Manual management**: ALL allocations must be manually freed
- **Order dependency**: Free populations before freeing individuals
- **Double-free safety**: Functions handle NULL pointers safely

### Error Handling Macros

```c
// Defined in logger.h
#define MALLOC_FAIL LOG_EXIT("malloc failed")
#define MALLOC_FAIL_THREADSAFE LOG_EXIT_THREADSAFE("malloc failed")

// Usage pattern
struct Program *prog = aligned_alloc(VECT_ALIGNMENT, size);
if (!prog) MALLOC_FAIL;
```

**Error Handling Features**:
- **Automatic logging**: Failures logged to `genetic.log` file
- **Clean exit**: Program terminates gracefully after logging error
- **Thread safety**: `*_THREADSAFE` variants for OpenMP contexts

## Thread Safety and Parallelization

### OpenMP Integration

**Thread Configuration**:
```c
// Compile-time configuration
#define NUMBER_OF_OMP_THREADS 16  // Set by THREADS=N parameter

// Runtime initialization required
random_init_all(seed);  // MUST be called before any code
```

**Parallel Sections**:
- **Fitness evaluation**: Population evaluated in parallel across threads
- **Population initialization**: Concurrent individual generation
- **Genetic operations**: Parallel mutation and crossover operations

### Thread-Local Storage

**Random Number Generation**:
```c
// Thread-local MT19937 generators
extern struct RandomEngine random_engines[NUMBER_OF_OMP_THREADS];

// Thread-safe access
uint64_t value = random();  // Automatically uses thread-local generator
double prob = random_double();
```

**Memory Isolation**:
- **VM execution**: Each thread uses independent `VirtualMachine` instances
- **Fitness evaluation**: No shared state during parallel evaluation
- **Memory allocation**: Thread-local allocation patterns where possible

### Thread Safety Guidelines

1. **Initialization**: Always call `random_init_all()` before parallel sections
2. **Function selection**: Use `*_THREADSAFE` variants in OpenMP contexts
3. **Memory sharing**: Minimize shared mutable state between threads
4. **Synchronization**: Use OpenMP pragmas for critical sections when needed

## Random Number Generation

High-quality pseudorandom number generation using Mersenne Twister algorithm.

### MT19937 Implementation

**Core Features**:
- **Long period**: 2^19937-1 ensures no repetition in practical applications
- **High quality**: Passes comprehensive statistical randomness tests
- **Thread safety**: Independent generator state per OpenMP thread
- **SIMD optimization**: Vectorized generation using AVX-512/AVX2/SSE2 when available

### Random Generation Functions

```c
// Initialization (REQUIRED before use)
void random_init_all(uint64_t seed);           // Initialize all thread-local generators
void random_init(uint64_t seed);               // Initialize current thread generator

// Random number generation  
uint64_t random();                             // Thread-local macro for random_uint64()
uint64_t random_uint64();                      // Full-range 64-bit integer
double random_double();                        // Uniform [0,1) floating-point
uint64_t random_range(uint64_t max);          // Uniform [0, max) integer

// Utility macros (defined in prob.h)
WILL_HAPPEN(prob)                             // Probabilistic events (prob ∈ [0,1])
RAND_BOUNDS(min, max)                         // Integer in [min, max] range
RAND_DBL_BOUNDS(min, max)                     // Double in [min, max] range
```

### Thread-Local Architecture

```c
// Thread-local generator array
struct RandomEngine random_engines[NUMBER_OF_OMP_THREADS];

// Automatic thread detection
#define random() random_uint64_threadsafe(omp_get_thread_num())
```

**Deterministic Behavior**:
- **Same seed**: Identical results regardless of thread count
- **Thread independence**: Each thread maintains separate PRNG state  
- **No synchronization**: Lock-free operation during generation

## Benchmark Problems (PSB2)

The library includes five challenging benchmark problems from the Program Synthesis Benchmark Suite 2.

### Available Benchmark Functions

#### 1. Vector Distance Problem
```c
struct LGPInput vector_distance(
    const struct InstructionSet *instr_set, 
    uint64_t vector_len,     // Dimension of vectors (recommended: 2-10)
    uint64_t instances       // Number of training instances
);
```
- **Objective**: Calculate Euclidean distance between n-dimensional vectors
- **Input format**: Two vectors of length `vector_len` each
- **Output**: Single floating-point distance value
- **Mathematical challenge**: Requires square, square root, and summation operations
- **Recommended fitness**: `mse` or `rmse` for distance prediction accuracy

#### 2. Bouncing Balls Physics Simulation  
```c
struct LGPInput bouncing_balls(
    const struct InstructionSet *instr_set, 
    uint64_t instances       // Number of physics scenarios
);
```
- **Objective**: Predict ball trajectory with gravity and collisions
- **Input format**: Initial position, velocity, gravity coefficient, time
- **Output**: Final ball position after bouncing
- **Physics challenge**: Complex dynamics with collision detection and energy conservation
- **Recommended fitness**: `mae` for position prediction (robust to outliers)

#### 3. Dice Game Strategy Optimization
```c
struct LGPInput dice_game(
    const struct InstructionSet *instr_set, 
    uint64_t instances       // Number of game scenarios
);
```
- **Objective**: Compute optimal strategy for dice-based decision game
- **Input format**: Current game state, available moves, probabilities
- **Output**: Optimal move selection (integer choice)
- **Game theory challenge**: Requires probability reasoning and strategy optimization
- **Recommended fitness**: `accuracy` for exact move matching

#### 4. Shopping List Budget Optimization
```c
struct LGPInput shopping_list(
    const struct InstructionSet *instr_set, 
    uint64_t num_of_items,   // Number of available items (complexity parameter)
    uint64_t instances       // Number of shopping scenarios
);
```
- **Objective**: Optimize item selection within budget constraints
- **Input format**: Item prices, available budget, discount rules
- **Output**: Binary selection for each item (0/1 choices)
- **Optimization challenge**: Combinatorial optimization with complex constraints
- **Recommended fitness**: Custom fitness combining budget adherence and value maximization

#### 5. Snow Day Weather Prediction
```c
struct LGPInput snow_day(
    const struct InstructionSet *instr_set, 
    uint64_t instances       // Number of weather scenarios
);
```
- **Objective**: Predict snow accumulation based on weather conditions
- **Input format**: Temperature, precipitation, wind speed, humidity
- **Output**: Predicted snow accumulation (continuous value)
- **Meteorological challenge**: Multi-variable environmental modeling
- **Recommended fitness**: `mse` or `huber_loss` for robust weather prediction

### Benchmark Usage Patterns

**Standard Evaluation Protocol**:
```c
// Create benchmark problem
struct LGPInput input = vector_distance(&instr_set, 3, 200);

// Configure appropriate fitness
struct LGPOptions options = {
    .fitness = MSE,
    .fitness_param = {.start = 0, .end = 1},  // Single output
    // ... other evolution parameters
};

// Run evolution
struct LGPResult result = evolve(&input, &options);
```

**Performance Benchmarking**:
- **Instance scaling**: Increase `instances` for harder problems
- **Dimension scaling**: Increase vector length or item count for complexity
- **Cross-validation**: Use separate test instances for validation
- **Statistical analysis**: Multiple runs for statistical significance

### Research Applications

**Algorithm Comparison**:
- **Standardized problems**: Compare different GP approaches on same benchmarks
- **Parameter tuning**: Optimize evolution parameters for problem classes
- **Performance profiling**: Measure execution speed and convergence rates

**Publication Support**:
- **Reproducible results**: Deterministic benchmarks enable result reproduction
- **Standard metrics**: Consistent evaluation across research papers
- **Baseline performance**: Established difficulty levels for comparison

## Error Handling and Debugging

### Assertion System

```c
// Debug mode assertions (enabled with DEBUG=1)
#define ASSERT(condition) do { \
    if (!(condition)) { \
        unreachable(); \
    } \
} while(0)
```

**Assertion Categories**:
- **Size validation**: Program sizes, population sizes, parameter ranges
- **Memory validation**: Non-NULL pointers, proper alignment
- **Logic validation**: Loop invariants, pre/post-conditions
- **Data consistency**: Structure field relationships

### Logging System

```c
// Single-threaded logging
void LOG_EXIT(const char *message);

// Thread-safe logging (for OpenMP contexts)  
void LOG_EXIT_THREADSAFE(const char *message);
```

**Logging Features**:
- **File output**: All logs written to `genetic.log` in working directory
- **Automatic exit**: Program terminates after logging critical errors
- **Thread identification**: Thread-safe version includes thread information
- **Timestamp**: Log entries include execution timestamps

### Numerical Error Handling

**Fitness Function Robustness**:
- **NaN detection**: Invalid floating-point results replaced with penalty values
- **Infinity handling**: Infinite values clamped to maximum representable values
- **Division by zero**: Safe handling in both VM execution and fitness evaluation
- **Overflow protection**: Integer operations include overflow detection

**VM Execution Safety**:
- **Clock limits**: Prevents infinite loops with maximum instruction counts
- **Memory bounds**: ROM operations include bounds checking (RAM operations don't)
- **Flag validation**: Proper flag state management for conditional operations

## Performance Optimization and Profiling

### Performance Measurement

**Built-in Timing**:
```c
// High-resolution timing for performance measurement
static inline double get_time_sec(void) {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
        return ts.tv_sec + ts.tv_nsec * 1e-9;
    }
    return (double)clock() / (double)CLOCKS_PER_SEC;
}
```

**Performance Metrics**:
- **Evaluations per second**: Total fitness evaluations divided by execution time
- **Generation speed**: Time per generation for algorithm comparison
- **Memory efficiency**: Peak memory usage and allocation patterns
- **Scalability**: Performance scaling with population size and thread count

### Optimization Guidelines

**Algorithm Selection**:
- **Tournament selection**: Best general-purpose selection (balanced speed/diversity)
- **Unique initialization**: Use for high-quality runs (slower but better diversity)
- **Appropriate fitness**: Match fitness function to problem type for efficiency

**Parameter Tuning**:
- **Population size**: Balance exploration and computation time (50-500 typical)
- **Generations**: Early termination reduces unnecessary computation
- **Mutation/crossover rates**: Problem-dependent optimization (0.1-0.9 typical ranges)

**System-Level Optimization**:
- **Thread count**: Match `THREADS` to available CPU cores
- **Memory allocation**: Pre-allocate large structures when possible
- **SIMD utilization**: Ensure proper memory alignment for vectorization

## Critical Safety Notes

### Input Validation Absence

1. **NO BOUNDS CHECKING**: Array access is never validated - invalid indices cause crashes
2. **NO NULL CHECKS**: Passing NULL pointers will immediately crash the program  
3. **NO TYPE VALIDATION**: Wrong data types cause incorrect results or undefined behavior
4. **NO MEMORY ALIGNMENT CHECKS**: Improper alignment causes crashes on some architectures
5. **NO PARAMETER VALIDATION**: Invalid parameters cause undefined behavior or crashes

### Memory Safety Requirements

- **Alignment**: Program content arrays must use `VECT_ALIGNMENT` for SIMD compatibility
- **Initialization**: All structure fields must be properly initialized before use
- **Cleanup**: Manual memory management required - all allocations must be freed
- **Thread safety**: Must call `random_init_all()` before any parallel operations

### Performance vs Safety Trade-offs

**This library prioritizes performance over safety:**
- **Zero-overhead principle**: No runtime checks that impact performance
- **Unsafe by design**: Assumes correct usage by experienced developers
- **Performance critical**: Designed for high-throughput evolutionary computation
- **Expert use**: Requires thorough understanding of genetic programming principles

**Use this library only if you:**
- Understand the performance vs safety trade-offs
- Can ensure correct input data and parameter validation
- Need maximum performance for evolutionary computation
- Are experienced with genetic programming concepts and C programming
